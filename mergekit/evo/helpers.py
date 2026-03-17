# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.evo.config import PHASE1_TASK_MIX_PROFILES, TaskConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.monkeypatch import monkeypatch_lmeval_vllm

# Try to import multi-method genome exception
try:
    from mergekit.evo.multi_method_genome import (
        InvalidGenotypeError as MultiMethodInvalidGenotypeError,
    )
except ImportError:
    MultiMethodInvalidGenotypeError = InvalidGenotypeError

from mergekit.merge import run_merge
from mergekit.options import MergeOptions

LOG = logging.getLogger(__name__)

_CHAT_TEMPLATE_WARNING_EMITTED = False
_LOWER_IS_BETTER_HINTS = (
    "loss",
    "error",
    "perplexity",
    "ppl",
    "wer",
    "cer",
    "rmse",
    "mae",
    "mse",
)
_HIGHER_IS_BETTER_HINTS = (
    "acc",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "mcc",
    "pearson",
    "spearman",
    "pass@",
)
_DATASETS_TRUST_REMOTE_CODE_FRAGMENT = "`trust_remote_code` is not supported anymore."
_LOCAL_MODEL_SHA_WARNING_PREFIX = "Failed to get model SHA for "
_LOCAL_MODEL_SHA_WARNING_FRAGMENT = "Repo id must be in the form"


def _emit_chat_template_retry_warning() -> None:
    """Log the chat template fallback warning only once per process."""
    global _CHAT_TEMPLATE_WARNING_EMITTED
    if not _CHAT_TEMPLATE_WARNING_EMITTED:
        LOG.warning(
            "Chat template requested but tokenizer lacks template; retrying without chat formatting."
        )
        _CHAT_TEMPLATE_WARNING_EMITTED = True


class _ExpectedEvalNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _DATASETS_TRUST_REMOTE_CODE_FRAGMENT in message:
            return False
        if (
            message.startswith(_LOCAL_MODEL_SHA_WARNING_PREFIX)
            and _LOCAL_MODEL_SHA_WARNING_FRAGMENT in message
        ):
            return False
        return True


@contextmanager
def _suppress_expected_eval_noise():
    """Temporarily filter known low-signal lm-eval/HF warnings.

    These messages are expected for local merged checkpoints and for legacy
    task YAMLs that still set `trust_remote_code` in dataset kwargs. They add
    log spam but do not signal actionable failures for GA runs.
    """

    noise_filter = _ExpectedEvalNoiseFilter()
    root_logger = logging.getLogger()
    target_loggers = [
        logging.getLogger("datasets.load"),
        logging.getLogger("lm-eval"),
    ]

    seen_handlers = set()
    handlers: List[logging.Handler] = []
    for logger in [root_logger, *target_loggers]:
        for handler in logger.handlers:
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            handlers.append(handler)

    for logger in target_loggers:
        logger.addFilter(noise_filter)
    for handler in handlers:
        handler.addFilter(noise_filter)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(noise_filter)
        for logger in target_loggers:
            logger.removeFilter(noise_filter)


def _infer_higher_is_better(metric_name: str) -> bool:
    metric_lower = metric_name.lower()
    if any(token in metric_lower for token in _LOWER_IS_BETTER_HINTS):
        return False
    if any(token in metric_lower for token in _HIGHER_IS_BETTER_HINTS):
        return True
    return True


def _metric_score_sign(
    results: Dict[str, Any], task_name: str, metric_name: str
) -> float:
    higher_is_better = (
        results.get("higher_is_better", {}).get(task_name, {}).get(metric_name)
    )
    if higher_is_better is None:
        higher_is_better = _infer_higher_is_better(metric_name)
    return 1.0 if higher_is_better else -1.0


def _normalize_higher_is_better_metric(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    if 0.0 <= value <= 100.0:
        return value / 100.0
    return value / (1.0 + abs(value))


def _normalize_lower_is_better_metric(value: float) -> float:
    safe_value = max(0.0, float(value))
    return 1.0 / (1.0 + safe_value)


def _weighted_average(pairs: List[tuple[float, float]]) -> float:
    total_weight = float(sum(weight for weight, _ in pairs))
    if total_weight <= 0.0:
        return 0.0
    return float(sum(weight * value for weight, value in pairs) / total_weight)


def _resolve_metric_value(
    task: TaskConfiguration, task_results: Dict[str, Any]
) -> tuple[Optional[str], Optional[float]]:
    metric_value = None
    selected_metric = task.metric

    if task.metric in task_results:
        metric_value = task_results[task.metric]
    else:
        metric_alternatives = {
            "ppl,none": [
                "word_perplexity,none",
                "perplexity,none",
                "byte_perplexity,none",
            ],
            "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
            "acc_norm,none": ["acc_norm,none", "acc,none", "accuracy,none"],
        }
        for alt_metric in metric_alternatives.get(task.metric, []):
            if alt_metric in task_results:
                metric_value = task_results[alt_metric]
                selected_metric = alt_metric
                break

        if metric_value is None:
            available_metrics = list(task_results.keys())
            if "ppl" in task.metric or "perplexity" in task.metric:
                for metric in available_metrics:
                    if "perplexity" in metric.lower() and "stderr" not in metric:
                        metric_value = task_results[metric]
                        selected_metric = metric
                        break
            elif "acc" in task.metric:
                for metric in available_metrics:
                    if "acc" in metric.lower() and "stderr" not in metric:
                        metric_value = task_results[metric]
                        selected_metric = metric
                        break

    if metric_value is None:
        return None, None
    if isinstance(metric_value, float) and math.isnan(metric_value):
        return selected_metric, None
    try:
        return selected_metric, float(metric_value)
    except (TypeError, ValueError):
        return selected_metric, None


def score_evaluation_payload(
    eval_payload: Dict[str, Any],
    tasks: List[TaskConfiguration],
    *,
    fitness_mode: str = "weighted_sum",
    task_mix_profile: Optional[str] = None,
    behavior_probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_results_map = eval_payload.get("results", {})
    weighted_sum_score = 0.0
    task_pairs: List[tuple[float, float]] = []
    language_pairs: List[tuple[float, float]] = []
    valid_metrics = 0

    for task in tasks:
        if task.name not in task_results_map:
            logging.warning("Task %s not found in results", task.name)
            continue

        selected_metric, metric_value = _resolve_metric_value(
            task, task_results_map[task.name]
        )
        if metric_value is None:
            logging.error(
                "Could not resolve metric %s for task %s. Available: %s",
                task.metric,
                task.name,
                list(task_results_map[task.name].keys()),
            )
            continue

        sign = _metric_score_sign(
            eval_payload, task.name, selected_metric or task.metric
        )
        weighted_sum_score += sign * metric_value * task.weight
        valid_metrics += 1

        if sign > 0:
            task_pairs.append(
                (float(task.weight), _normalize_higher_is_better_metric(metric_value))
            )
        else:
            language_pairs.append(
                (float(task.weight), _normalize_lower_is_better_metric(metric_value))
            )

    if fitness_mode != "structured_phase1_tiny":
        return {
            "score": weighted_sum_score,
            "results": task_results_map,
            "fitness_components": {
                "raw_weighted_score": weighted_sum_score,
                "behavior_diversity_score": (
                    float(behavior_probe.get("behavior_diversity_score", 0.0))
                    if behavior_probe
                    else 0.0
                ),
            },
        }

    profile = PHASE1_TASK_MIX_PROFILES.get(task_mix_profile or "tiny_local_default")
    if not profile:
        return {
            "score": weighted_sum_score,
            "results": task_results_map,
            "fitness_components": {"raw_weighted_score": weighted_sum_score},
        }

    task_score = _weighted_average(task_pairs)
    language_quality = _weighted_average(language_pairs)
    metric_coverage = float(valid_metrics / len(tasks)) if tasks else 0.0
    behavior_stability = (
        float(behavior_probe.get("stability_score", 1.0)) if behavior_probe else 1.0
    )
    stability_score = (metric_coverage + behavior_stability) / 2.0
    structured_score = (
        float(profile["task_score_weight"]) * task_score
        + float(profile["language_quality_weight"]) * language_quality
        + float(profile["stability_weight"]) * stability_score
    )
    return {
        "score": structured_score,
        "results": task_results_map,
        "fitness_components": {
            "raw_weighted_score": weighted_sum_score,
            "task_score": task_score,
            "language_quality": language_quality,
            "stability_score": stability_score,
            "behavior_diversity_score": (
                float(behavior_probe.get("behavior_diversity_score", 0.0))
                if behavior_probe
                else 0.0
            ),
            "diversity_bonus": 0.0,
        },
    }


def _failure_result(
    stage: str,
    error_type: str,
    error_message: str,
    *,
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "score": None,
        "results": results or {},
        "error_stage": stage,
        "error_type": error_type,
        "error_message": error_message,
    }


def _distinct_token_ratio(text: str) -> float:
    tokens = [token for token in text.split() if token]
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


def _has_repetition_loop(text: str, ngram_size: int) -> bool:
    tokens = [token for token in text.split() if token]
    if len(tokens) < ngram_size * 2:
        return False
    seen = {}
    for idx in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[idx : idx + ngram_size])
        if ngram in seen and idx - seen[ngram] <= ngram_size:
            return True
        seen[ngram] = idx
    return False


def _run_behavior_probe(
    merged_path: str,
    prompts: List[str],
    *,
    max_new_tokens: int,
    repetition_ngram_size: int,
    min_distinct_ratio: float,
    device: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {"local_files_only": True}
    extra_kwargs = dict(model_kwargs or {})
    dtype_value = extra_kwargs.get("dtype")
    if isinstance(dtype_value, str) and hasattr(torch, dtype_value):
        load_kwargs["torch_dtype"] = getattr(torch, dtype_value)
    if extra_kwargs.get("quantization_config") is not None:
        load_kwargs["quantization_config"] = extra_kwargs["quantization_config"]
    if extra_kwargs.get("low_cpu_mem_usage") is not None:
        load_kwargs["low_cpu_mem_usage"] = extra_kwargs["low_cpu_mem_usage"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        merged_path, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        merged_path, **load_kwargs
    )
    model.to(device)
    model.eval()

    outputs: List[str] = []
    degenerate_outputs = 0
    distinct_ratios: List[float] = []
    reject_reasons: List[str] = []
    try:
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion_tokens = generated[0][encoded["input_ids"].shape[-1] :]
            completion = tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            ).strip()
            outputs.append(completion)
            distinct_ratio = _distinct_token_ratio(completion)
            distinct_ratios.append(distinct_ratio)

            reasons = []
            if not completion:
                reasons.append("empty_output")
            if _has_repetition_loop(completion, repetition_ngram_size):
                reasons.append("repetition_loop")
            if distinct_ratio < min_distinct_ratio:
                reasons.append("low_distinct_ratio")
            if reasons:
                degenerate_outputs += 1
                reject_reasons.append(f"{prompt[:24]}...:{','.join(reasons)}")
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    unique_output_fraction = float(len(set(outputs)) / len(outputs)) if outputs else 0.0
    mean_distinct_ratio = (
        float(sum(distinct_ratios) / len(distinct_ratios)) if distinct_ratios else 0.0
    )
    behavior_diversity_score = (unique_output_fraction + mean_distinct_ratio) / 2.0
    stability_score = (
        float(1.0 - (degenerate_outputs / len(outputs))) if outputs else 0.0
    )
    rejected = degenerate_outputs > 0
    return {
        "outputs": outputs,
        "unique_output_fraction": unique_output_fraction,
        "mean_distinct_ratio": mean_distinct_ratio,
        "behavior_diversity_score": behavior_diversity_score,
        "degenerate_output_count": float(degenerate_outputs),
        "stability_score": stability_score,
        "rejected": rejected,
        "reject_reason": ";".join(reject_reasons) if reject_reasons else None,
    }


_SIGNATURE_FIELDS = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "max_position_embeddings",
    "sliding_window",
    "rope_scaling",
    "rope_theta",
    "vocab_size",
)

_BASE_ARCHITECTURE_FIELDS = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "max_position_embeddings",
    "sliding_window",
    "rope_scaling",
    "rope_theta",
)


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, _normalize_signature_value(val)) for key, val in sorted(value.items())
        )
    if isinstance(value, list):
        return tuple(_normalize_signature_value(v) for v in value)
    return value


def _extract_model_signature(config: "transformers.PretrainedConfig") -> Dict[str, Any]:
    cfg_dict = config.to_dict()

    def _get(key: str) -> Any:
        if hasattr(config, key):
            return getattr(config, key)
        return cfg_dict.get(key)

    signature: Dict[str, Any] = {}
    signature["architectures"] = tuple(_get("architectures") or []) or None
    # Prefer hidden_size but fall back to d_model for encoder-decoder configs
    hidden_size = _get("hidden_size") or _get("d_model")
    signature["hidden_size"] = hidden_size

    for field in _SIGNATURE_FIELDS:
        if field in signature:
            continue
        signature[field] = _get(field)

    # Normalize complex structures so they can be compared/hashable
    for key, value in list(signature.items()):
        if value is not None:
            signature[key] = _normalize_signature_value(value)

    return signature


def _format_signature_value(value: Any) -> str:
    if isinstance(value, tuple):
        # Reconstruct dict-style display for normalized tuples
        try:
            if all(isinstance(item, tuple) and len(item) == 2 for item in value):
                inner = ", ".join(
                    f"{k}: {_format_signature_value(v)}" for k, v in value
                )
                return "{" + inner + "}"
        except Exception:  # pragma: no cover - defensive formatting
            pass
        return "(" + ", ".join(_format_signature_value(v) for v in value) + ")"
    return repr(value)


def _find_signature_incompatibilities(
    signatures: Dict[str, Dict[str, Any]],
    fields: tuple[str, ...] = _SIGNATURE_FIELDS,
) -> List[str]:
    mismatches: List[str] = []
    for field in fields:
        values: Dict[Any, List[str]] = {}
        missing: List[str] = []
        for model_name, sig in signatures.items():
            value = sig.get(field)
            if value is None:
                missing.append(model_name)
                continue
            values.setdefault(value, []).append(model_name)

        if len(values) > 1:
            parts = []
            for value, models in values.items():
                parts.append(
                    f"{', '.join(sorted(models))} -> {_format_signature_value(value)}"
                )
            mismatches.append(f"{field}: {'; '.join(parts)}")
        elif values and missing:
            present_value, present_models = next(iter(values.items()))
            parts = [
                f"{', '.join(sorted(present_models))} -> {_format_signature_value(present_value)}",
                f"{', '.join(sorted(missing))} -> <missing>",
            ]
            mismatches.append(f"{field}: {'; '.join(parts)}")
    return mismatches


def _load_model_signatures(
    model_refs: List[ModelReference],
    merge_options: MergeOptions,
    *,
    require_all: bool = False,
) -> Dict[str, Dict[str, Any]]:
    signatures: Dict[str, Dict[str, Any]] = {}
    load_errors: List[str] = []

    for model_ref in model_refs:
        model_name = str(model_ref)
        if model_name in signatures:
            continue

        try:
            cfg = model_ref.config(trust_remote_code=merge_options.trust_remote_code)
        except Exception as exc:  # pragma: no cover - network or HF errors
            if require_all:
                load_errors.append(f"{model_name}: {exc}")
            else:
                LOG.warning("Failed to load config for %s", model_ref, exc_info=exc)
            continue

        signatures[model_name] = _extract_model_signature(cfg)

    if load_errors:
        raise RuntimeError(
            "Failed to load model configs for compatibility check:\n  - "
            + "\n  - ".join(load_errors)
        )

    return signatures


def validate_input_model_architecture(
    model_refs: List[ModelReference],
    merge_options: MergeOptions,
) -> None:
    if len(model_refs) <= 1:
        return

    signatures = _load_model_signatures(
        model_refs,
        merge_options,
        require_all=True,
    )
    if len(signatures) <= 1:
        return

    mismatches = _find_signature_incompatibilities(
        signatures,
        fields=_BASE_ARCHITECTURE_FIELDS,
    )
    if mismatches:
        message = (
            "Input models do not share the same base architecture. "
            "Use models from the same architecture family, or pass "
            "--allow-crimes to override:\n  - " + "\n  - ".join(mismatches)
        )
        if merge_options.allow_crimes:
            LOG.warning("%s", message)
        else:
            raise RuntimeError(message)


def _validate_merge_compatibility(
    merge_config: MergeConfiguration, merge_options: MergeOptions
) -> None:
    referenced_models = merge_config.referenced_models()
    if len(referenced_models) <= 1:
        return

    signatures = _load_model_signatures(referenced_models, merge_options)
    if len(signatures) <= 1:
        return

    mismatches = _find_signature_incompatibilities(signatures)
    if mismatches:
        message = "Incompatible model configurations detected:\n  - " + "\n  - ".join(
            mismatches
        )
        if merge_options.allow_crimes:
            LOG.warning("%s", message)
        else:
            raise InvalidGenotypeError(message)


def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    fitness_mode: str = "weighted_sum",
    task_mix_profile: Optional[str] = None,
    behavior_probe: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    with _suppress_expected_eval_noise():
        results = lm_eval.simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=list(set([task.name for task in tasks])),
            log_samples=False,
            verbosity="WARNING",
            task_manager=task_manager,
            **kwargs,
        )
    logging.info(results["results"])
    return score_evaluation_payload(
        results,
        tasks,
        fitness_mode=fitness_mode,
        task_mix_profile=task_mix_profile,
        behavior_probe=behavior_probe,
    )


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    fitness_mode: str = "weighted_sum",
    task_mix_profile: Optional[str] = None,
    behavior_prompts: Optional[List[str]] = None,
    behavior_probe_max_new_tokens: int = 24,
    behavior_repetition_ngram_size: int = 4,
    behavior_min_distinct_ratio: float = 0.2,
    behavior_reject_on_degenerate: bool = False,
    **kwargs,
) -> dict:
    # monkeypatch_tqdm()
    monkeypatch_lmeval_vllm()
    try:
        if not merged_path:
            logging.error("No merged model path provided; skipping evaluation.")
            return _failure_result(
                "eval",
                "missing_merged_path",
                "No merged model path provided; skipping evaluation.",
            )
        extra_model_kwargs = dict(model_kwargs or {})
        requested_device = extra_model_kwargs.pop("device", None)
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
            "local_files_only": True,
            **extra_model_kwargs,
        }

        eval_kwargs = dict(kwargs)
        device_arg = eval_kwargs.pop("device", None)

        if vllm:
            model_args.setdefault("gpu_memory_utilization", 0.8)
            model_args["tensor_parallel_size"] = max(1, int(tensor_parallel_size))
            model_args.setdefault("batch_size", "auto")
            model_args.setdefault("max_model_len", 4096)
        else:
            model_args["use_cache"] = True
            if device_arg is None:
                device_arg = requested_device
            if device_arg is None and torch.cuda.is_available():
                device_arg = "cuda"

        if device_arg is not None:
            eval_kwargs["device"] = device_arg

        behavior_probe = None
        try:
            if behavior_prompts:
                behavior_probe = _run_behavior_probe(
                    merged_path,
                    list(behavior_prompts),
                    max_new_tokens=behavior_probe_max_new_tokens,
                    repetition_ngram_size=behavior_repetition_ngram_size,
                    min_distinct_ratio=behavior_min_distinct_ratio,
                    device=str(
                        eval_kwargs.get("device", "cuda" if not vllm else "cuda")
                    ),
                    model_kwargs=model_args,
                )
                if behavior_reject_on_degenerate and behavior_probe.get("rejected"):
                    return _failure_result(
                        "smoke_test",
                        "degenerate_output",
                        str(
                            behavior_probe.get("reject_reason")
                            or "Behavior probe failed"
                        ),
                        results={"behavior_probe": behavior_probe},
                    )
            try:
                res = _eval_model(
                    "vllm" if vllm else "huggingface",
                    tasks,
                    model_args,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    batch_size=batch_size,
                    task_manager=task_manager,
                    fitness_mode=fitness_mode,
                    task_mix_profile=task_mix_profile,
                    behavior_probe=behavior_probe,
                    bootstrap_iters=0,
                    **eval_kwargs,
                )
            except ValueError as exc:
                message = str(exc).lower()
                if "chat template" in message and eval_kwargs.get(
                    "apply_chat_template"
                ):
                    _emit_chat_template_retry_warning()
                    fallback_kwargs = dict(eval_kwargs)
                    fallback_kwargs["apply_chat_template"] = False
                    fallback_kwargs["fewshot_as_multiturn"] = False
                    res = _eval_model(
                        "vllm" if vllm else "huggingface",
                        tasks,
                        model_args,
                        num_fewshot=num_fewshot,
                        limit=limit,
                        batch_size=batch_size,
                        task_manager=task_manager,
                        fitness_mode=fitness_mode,
                        task_mix_profile=task_mix_profile,
                        behavior_probe=behavior_probe,
                        bootstrap_iters=0,
                        **fallback_kwargs,
                    )
                else:
                    raise
        except Exception as exc:
            logging.error("Model evaluation failed", exc_info=exc)
            return _failure_result(
                "eval",
                type(exc).__name__,
                str(exc),
                results={"behavior_probe": behavior_probe} if behavior_probe else None,
            )
        else:
            _apply_metric_guards(res)
            if behavior_probe is not None:
                res["behavior_probe"] = behavior_probe
        return res
    finally:
        if merged_path:
            shutil.rmtree(merged_path, ignore_errors=True)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)


def evaluate_model_cpu(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    fitness_mode: str = "weighted_sum",
    task_mix_profile: Optional[str] = None,
    behavior_prompts: Optional[List[str]] = None,
    behavior_probe_max_new_tokens: int = 24,
    behavior_repetition_ngram_size: int = 4,
    behavior_min_distinct_ratio: float = 0.2,
    behavior_reject_on_degenerate: bool = False,
    **kwargs,
) -> dict:
    """CPU-only evaluation using HuggingFace backend and float32."""
    monkeypatch_lmeval_vllm()
    try:
        if not merged_path:
            logging.error("No merged model path provided; skipping CPU evaluation.")
            return _failure_result(
                "eval",
                "missing_merged_path",
                "No merged model path provided; skipping CPU evaluation.",
            )
        extra_kwargs = dict(model_kwargs or {})
        # Force CPU execution regardless of caller-specified overrides
        device_override = extra_kwargs.pop("device", "cpu")
        extra_kwargs.setdefault("dtype", "float32")
        extra_kwargs.setdefault("device_map", None)
        extra_kwargs.setdefault("low_cpu_mem_usage", False)
        model_args = {
            "pretrained": merged_path,
            "use_cache": True,
            "local_files_only": True,
            **extra_kwargs,
        }
        eval_kwargs: Dict[str, Any] = {"device": device_override}
        eval_kwargs.update(kwargs)
        behavior_probe = None
        try:
            if behavior_prompts:
                behavior_probe = _run_behavior_probe(
                    merged_path,
                    list(behavior_prompts),
                    max_new_tokens=behavior_probe_max_new_tokens,
                    repetition_ngram_size=behavior_repetition_ngram_size,
                    min_distinct_ratio=behavior_min_distinct_ratio,
                    device="cpu",
                    model_kwargs=model_args,
                )
                if behavior_reject_on_degenerate and behavior_probe.get("rejected"):
                    return _failure_result(
                        "smoke_test",
                        "degenerate_output",
                        str(
                            behavior_probe.get("reject_reason")
                            or "Behavior probe failed"
                        ),
                        results={"behavior_probe": behavior_probe},
                    )
            try:
                res = _eval_model(
                    "huggingface",
                    tasks,
                    model_args,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    batch_size=batch_size,
                    task_manager=task_manager,
                    fitness_mode=fitness_mode,
                    task_mix_profile=task_mix_profile,
                    behavior_probe=behavior_probe,
                    bootstrap_iters=0,
                    **eval_kwargs,
                )
            except ValueError as exc:
                message = str(exc).lower()
                if "chat template" in message and eval_kwargs.get(
                    "apply_chat_template"
                ):
                    _emit_chat_template_retry_warning()
                    fallback_kwargs = dict(eval_kwargs)
                    fallback_kwargs["apply_chat_template"] = False
                    fallback_kwargs["fewshot_as_multiturn"] = False
                    res = _eval_model(
                        "huggingface",
                        tasks,
                        model_args,
                        num_fewshot=num_fewshot,
                        limit=limit,
                        batch_size=batch_size,
                        task_manager=task_manager,
                        fitness_mode=fitness_mode,
                        task_mix_profile=task_mix_profile,
                        behavior_probe=behavior_probe,
                        bootstrap_iters=0,
                        **fallback_kwargs,
                    )
                else:
                    raise
        except Exception as exc:
            logging.error("CPU model evaluation failed", exc_info=exc)
            return _failure_result(
                "eval",
                type(exc).__name__,
                str(exc),
                results={"behavior_probe": behavior_probe} if behavior_probe else None,
            )
        else:
            _apply_metric_guards(res)
            if behavior_probe is not None:
                res["behavior_probe"] = behavior_probe
        return res
    finally:
        if merged_path:
            shutil.rmtree(merged_path, ignore_errors=True)


evaluate_model_ray_cpu = ray.remote(num_cpus=1)(evaluate_model_cpu)


def merge_model_with_details(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> Dict[str, Any]:
    # monkeypatch_tqdm()
    try:
        plan = None
        if hasattr(genome, "genotype_to_merge_plan"):
            plan = genome.genotype_to_merge_plan(genotype)
            if plan["kind"] == "config":
                cfg = plan["config"]
            else:
                cfg = None
        elif hasattr(genome, "genotype_to_merge_config"):
            cfg = genome.genotype_to_merge_config(genotype)
        else:
            cfg = genome.genotype_merge_config(genotype)

        if cfg is not None:
            _validate_merge_compatibility(cfg, merge_options)
    except (InvalidGenotypeError, MultiMethodInvalidGenotypeError) as e:
        logging.error("Invalid genotype: %s", e)
        return {
            "merged_path": None,
            "error_stage": "merge",
            "error_type": "invalid_genotype",
            "error_message": str(e),
        }

    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    try:
        if cfg is not None:
            run_merge(cfg, out_path=res, options=merge_options)
            return {
                "merged_path": res,
                "error_stage": None,
                "error_type": None,
                "error_message": None,
                "resolved_merge_config": cfg.model_dump(
                    exclude_defaults=True, mode="json"
                ),
            }

        component_root = tempfile.mkdtemp(
            prefix="layered-components", dir=model_storage_path
        )
        component_paths: Dict[str, str] = {}
        try:
            for component in plan["components"]:
                component_cfg: MergeConfiguration = component["config"]
                _validate_merge_compatibility(component_cfg, merge_options)
                component_path = os.path.join(component_root, component["name"])
                run_merge(component_cfg, out_path=component_path, options=merge_options)
                component_paths[component["name"]] = component_path

            final_slices = []
            for slice_def in plan["final_slices"]:
                component_name = slice_def["component"]
                component_path = component_paths[component_name]
                final_slices.append(
                    {
                        "sources": [
                            {
                                "model": component_path,
                                "layer_range": slice_def["layer_range"],
                            }
                        ]
                    }
                )

            tokenizer_source = None
            if plan["final_slices"]:
                tokenizer_source = component_paths[plan["final_slices"][0]["component"]]
            final_config = MergeConfiguration.model_validate(
                {
                    "merge_method": "passthrough",
                    "slices": final_slices,
                    "dtype": "bfloat16",
                    "tokenizer_source": tokenizer_source,
                }
            )
            run_merge(final_config, out_path=res, options=merge_options)
        finally:
            shutil.rmtree(component_root, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - run_merge handles many cases
        logging.error("Merge execution failed", exc_info=exc)
        shutil.rmtree(res, ignore_errors=True)
        return {
            "merged_path": None,
            "error_stage": "merge",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "merged_path": res,
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "resolved_merge_plan": (
            plan.get("methods") if isinstance(plan, dict) else None
        ),
    }


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    result = merge_model_with_details(
        genotype,
        genome,
        model_storage_path,
        merge_options,
    )
    return result["merged_path"]


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)

merge_model_with_details_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model_with_details)

merge_model_ray_cpu = ray.remote(
    num_cpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)

merge_model_with_details_ray_cpu = ray.remote(
    num_cpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model_with_details)


def _apply_metric_guards(result: dict) -> None:
    """Clamp obviously invalid evaluation metrics to safe defaults.

    Marks a result as failed (score=None) when perplexity explodes, accuracy vanishes,
    or metric payloads are malformed. Keeps downstream consumers from learning from
    catastrophic merges while still logging raw outputs for debugging.
    """

    if not result:
        return

    score = result.get("score")
    metrics = result.get("results") or {}

    if score is None:
        return

    for task_name, task_metrics in metrics.items():
        if not isinstance(task_metrics, dict):
            continue

        perplexity = task_metrics.get("perplexity,none")
        accuracy = task_metrics.get("acc,none")

        if perplexity is not None:
            try:
                if float(perplexity) > 1e5:
                    message = f"Perplexity {float(perplexity):.3g} for task {task_name} exceeds guard threshold"
                    logging.warning(
                        "%s; marking evaluation as failed",
                        message,
                    )
                    result["score"] = None
                    result.setdefault("error_stage", "eval")
                    result.setdefault("error_type", "metric_guard")
                    result.setdefault("error_message", message)
                    return
            except (TypeError, ValueError):
                message = f"Non-numeric perplexity {perplexity!r} for task {task_name}"
                logging.warning(
                    "%s; marking evaluation as failed",
                    message,
                )
                result["score"] = None
                result.setdefault("error_stage", "eval")
                result.setdefault("error_type", "metric_guard")
                result.setdefault("error_message", message)
                return

        if accuracy is not None:
            try:
                if float(accuracy) < 1e-3:
                    message = f"Accuracy {float(accuracy):.3g} for task {task_name} below guard threshold"
                    logging.warning(
                        "%s; marking evaluation as failed",
                        message,
                    )
                    result["score"] = None
                    result.setdefault("error_stage", "eval")
                    result.setdefault("error_type", "metric_guard")
                    result.setdefault("error_message", message)
                    return
            except (TypeError, ValueError):
                message = f"Non-numeric accuracy {accuracy!r} for task {task_name}"
                logging.warning(
                    "%s; marking evaluation as failed",
                    message,
                )
                result["score"] = None
                result.setdefault("error_stage", "eval")
                result.setdefault("error_type", "metric_guard")
                result.setdefault("error_message", message)
                return
