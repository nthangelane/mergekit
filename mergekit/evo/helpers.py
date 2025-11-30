# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
import shutil
import tempfile
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

from mergekit.evo.config import TaskConfiguration
from mergekit.config import MergeConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.monkeypatch import monkeypatch_lmeval_vllm

# Try to import multi-method genome exception
try:
    from mergekit.evo.multi_method_genome import InvalidGenotypeError as MultiMethodInvalidGenotypeError
except ImportError:
    MultiMethodInvalidGenotypeError = InvalidGenotypeError

from mergekit.merge import run_merge
from mergekit.options import MergeOptions


LOG = logging.getLogger(__name__)

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


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, _normalize_signature_value(val))
            for key, val in sorted(value.items())
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
    signatures: Dict[str, Dict[str, Any]]
) -> List[str]:
    mismatches: List[str] = []
    for field in _SIGNATURE_FIELDS:
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


def _validate_merge_compatibility(
    merge_config: MergeConfiguration, merge_options: MergeOptions
) -> None:
    referenced_models = merge_config.referenced_models()
    if len(referenced_models) <= 1:
        return

    signatures: Dict[str, Dict[str, Any]] = {}
    for model_ref in referenced_models:
        try:
            cfg = model_ref.config(trust_remote_code=merge_options.trust_remote_code)
        except Exception as exc:  # pragma: no cover - network or HF errors
            LOG.warning("Failed to load config for %s", model_ref, exc_info=exc)
            continue
        signatures[str(model_ref)] = _extract_model_signature(cfg)

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
    **kwargs,
) -> Dict[str, Any]:
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
    res = 0
    for task in tasks:
        if task.name not in results["results"]:
            logging.warning(f"Task {task.name} not found in results")
            continue

        task_results = results["results"][task.name]
        metric_value = None

        # Try the exact metric first
        if task.metric in task_results:
            metric_value = task_results[task.metric]
        else:
            # Auto-detect common metric alternatives
            metric_alternatives = {
                "ppl,none": [
                    "word_perplexity,none",
                    "perplexity,none",
                    "byte_perplexity,none",
                ],
                "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
                "acc_norm,none": ["acc_norm,none", "acc,none", "accuracy,none"],
            }

            alternatives = metric_alternatives.get(task.metric, [])
            for alt_metric in alternatives:
                if alt_metric in task_results:
                    metric_value = task_results[alt_metric]
                    logging.info(
                        f"Auto-detected metric for {task.name}: {alt_metric} instead of {task.metric}"
                    )
                    break

            # If still not found, try pattern matching
            if metric_value is None:
                available_metrics = list(task_results.keys())
                if "ppl" in task.metric or "perplexity" in task.metric:
                    # Look for any perplexity metric
                    for metric in available_metrics:
                        if "perplexity" in metric.lower() and "stderr" not in metric:
                            metric_value = task_results[metric]
                            logging.info(
                                f"Pattern-matched perplexity metric for {task.name}: {metric}"
                            )
                            break
                elif "acc" in task.metric:
                    # Look for any accuracy metric
                    for metric in available_metrics:
                        if "acc" in metric.lower() and "stderr" not in metric:
                            metric_value = task_results[metric]
                            logging.info(
                                f"Pattern-matched accuracy metric for {task.name}: {metric}"
                            )
                            break

        # Handle invalid values
        if metric_value is None:
            logging.error(
                f"Could not find metric {task.metric} for task {task.name}. Available: {list(task_results.keys())}"
            )
            continue

        # Handle NaN values
        import math

        if isinstance(metric_value, float) and math.isnan(metric_value):
            logging.warning(f"NaN result for {task.name}:{task.metric}, skipping")
            continue

        res += metric_value * task.weight
    return {"score": res, "results": results["results"]}


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> dict:
    # monkeypatch_tqdm()
    monkeypatch_lmeval_vllm()
    try:
        if not merged_path:
            logging.error("No merged model path provided; skipping evaluation.")
            return {"score": None, "results": {}}
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
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
            model_args["max_model_len"] = 4096
        else:
            model_args["use_cache"] = True
            if device_arg is None:
                device_arg = requested_device
            if device_arg is None and torch.cuda.is_available():
                device_arg = "cuda"

        if device_arg is not None:
            eval_kwargs["device"] = device_arg

            try:
                res = _eval_model(
                    "vllm" if vllm else "huggingface",
                    tasks,
                    model_args,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    batch_size=batch_size,
                    task_manager=task_manager,
                    bootstrap_iters=0,
                    **eval_kwargs,
                )
        except ValueError as exc:
            message = str(exc).lower()
            if "chat template" in message and eval_kwargs.get("apply_chat_template"):
                logging.warning(
                    "Chat template requested but tokenizer lacks template; retrying without chat formatting."
                )
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
                    bootstrap_iters=0,
                    **fallback_kwargs,
                )
            else:
                raise
            _apply_metric_guards(res)
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
    **kwargs,
) -> dict:
    """CPU-only evaluation using HuggingFace backend and float32."""
    monkeypatch_lmeval_vllm()
    try:
        if not merged_path:
            logging.error("No merged model path provided; skipping CPU evaluation.")
            return {"score": None, "results": {}}
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
        try:
            res = _eval_model(
                "huggingface",
                tasks,
                model_args,
                num_fewshot=num_fewshot,
                limit=limit,
                batch_size=batch_size,
                task_manager=task_manager,
                bootstrap_iters=0,
                **eval_kwargs,
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "chat template" in message and eval_kwargs.get("apply_chat_template"):
                logging.warning(
                    "Chat template requested but tokenizer lacks template; retrying without chat formatting."
                )
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
                    bootstrap_iters=0,
                    **fallback_kwargs,
                )
            else:
                raise
            _apply_metric_guards(res)
        return res
    finally:
        if merged_path:
            shutil.rmtree(merged_path, ignore_errors=True)


evaluate_model_ray_cpu = ray.remote(num_cpus=1)(evaluate_model_cpu)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    # monkeypatch_tqdm()
    try:
        # Handle both traditional and multi-method genomes
        if hasattr(genome, "genotype_to_merge_config"):
            # MultiMethodGenome
            cfg = genome.genotype_to_merge_config(genotype)
        else:
            # Traditional ModelGenome
            cfg = genome.genotype_merge_config(genotype)
        _validate_merge_compatibility(cfg, merge_options)
    except (InvalidGenotypeError, MultiMethodInvalidGenotypeError) as e:
        logging.error("Invalid genotype: %s", e)
        return None

    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    try:
        run_merge(cfg, out_path=res, options=merge_options)
    except Exception as exc:  # pragma: no cover - run_merge handles many cases
        logging.error("Merge execution failed", exc_info=exc)
        shutil.rmtree(res, ignore_errors=True)
        return None
    return res


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)

merge_model_ray_cpu = ray.remote(
    num_cpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)


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
                    logging.warning(
                        "Perplexity %.3g for task %s exceeds guard threshold; marking evaluation as failed",
                        float(perplexity),
                        task_name,
                    )
                    result["score"] = None
                    return
            except (TypeError, ValueError):
                logging.warning(
                    "Non-numeric perplexity %r for task %s; marking evaluation as failed",
                    perplexity,
                    task_name,
                )
                result["score"] = None
                return

        if accuracy is not None:
            try:
                if float(accuracy) < 1e-3:
                    logging.warning(
                        "Accuracy %.3g for task %s below guard threshold; marking evaluation as failed",
                        float(accuracy),
                        task_name,
                    )
                    result["score"] = None
                    return
            except (TypeError, ValueError):
                logging.warning(
                    "Non-numeric accuracy %r for task %s; marking evaluation as failed",
                    accuracy,
                    task_name,
                )
                result["score"] = None
                return
