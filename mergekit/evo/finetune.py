from __future__ import annotations

import csv
import gc
import os
import random
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from mergekit.evo.baselines import collect_task_metrics
from mergekit.evo.checkpoint import atomic_write_json
from mergekit.evo.config import FitnessConfiguration, TaskConfiguration
from mergekit.evo.contamination import (
    assert_disjoint_from_evaluation,
    dataset_split_identity,
)
from mergekit.evo.orchestrator import resolve_device
from mergekit.evo.progress import ProgressLogger, ga_compute_seconds

FINETUNE_RESULTS_FILENAME = "finetune_results.csv"
BUDGET_FILENAME = "budget.json"
TRAINED_MODEL_DIRNAME = "trained_model"


def _default_target_tasks() -> List[TaskConfiguration]:
    return [
        TaskConfiguration(name="wikitext", weight=0.40, metric="byte_perplexity,none"),
        TaskConfiguration(name="boolq", weight=0.35, metric="acc,none"),
        TaskConfiguration(name="sciq", weight=0.25, metric="acc,none"),
    ]


def _default_retention_tasks() -> List[TaskConfiguration]:
    return [
        TaskConfiguration(name="lambada_openai", weight=0.50, metric="acc,none"),
        TaskConfiguration(name="piqa", weight=0.50, metric="acc,none"),
    ]


class FineTuneBaselineConfiguration(BaseModel, frozen=True):
    arm: Literal["lora", "full"]
    model: str
    dataset: str
    dataset_config: Optional[str] = None
    split: str = "train"
    text_column: Optional[str] = None
    budget_seconds: Optional[float] = None
    budget_steps: Optional[int] = None
    match_ga_run: Optional[str] = None
    seed: int = 11
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 4
    seq_len: int = 256
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    max_train_examples: Optional[int] = None
    lora_r: int = 8
    lora_alpha: int = 16
    target_tasks: List[TaskConfiguration] = Field(default_factory=_default_target_tasks)
    retention_tasks: List[TaskConfiguration] = Field(
        default_factory=_default_retention_tasks
    )
    search_limit: int = 6
    export_limit: int = 12
    num_fewshot: int = 0
    eval_batch_size: Optional[int] = 1
    task_search_path: List[str] = Field(default_factory=list)
    fitness: FitnessConfiguration = FitnessConfiguration(version="v2")
    provenance: Literal["warn", "fail", "off"] = "warn"
    trust_remote_code: bool = False
    campaign: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_settings(self):
        budget_fields = [
            self.budget_seconds is not None,
            self.budget_steps is not None,
            self.match_ga_run is not None,
        ]
        if sum(budget_fields) != 1:
            raise ValueError(
                "Specify exactly one of budget_seconds, budget_steps, or match_ga_run"
            )
        if self.budget_seconds is not None and self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be > 0")
        if self.budget_steps is not None and self.budget_steps <= 0:
            raise ValueError("budget_steps must be > 0")
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError("batch_size and seq_len must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be > 0")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be > 0")
        if self.search_limit <= 0 or self.export_limit <= 0:
            raise ValueError("search_limit and export_limit must be > 0")

        training_source = dataset_split_identity(
            self.dataset,
            self.split,
            subset=self.dataset_config,
        )
        assert_disjoint_from_evaluation(
            training_source,
            [*self.target_tasks, *self.retention_tasks],
            source_label="Training dataset",
        )
        return self


@dataclass
class TrainingResult:
    model_path: str
    budget: Dict[str, Any]


def resolve_training_budget(
    config: FineTuneBaselineConfiguration,
) -> tuple[Optional[int], Optional[float], str]:
    if config.budget_steps is not None:
        return int(config.budget_steps), None, "steps"
    if config.budget_seconds is not None:
        return None, float(config.budget_seconds), "seconds"

    matched_seconds = ga_compute_seconds(str(config.match_ga_run))
    if matched_seconds <= 0:
        raise ValueError(
            f"Matched GA run has no positive measured compute time: {config.match_ga_run}"
        )
    return None, matched_seconds, "matched_ga_seconds"


def _extract_record_text(record: Mapping[str, Any], text_column: Optional[str]) -> str:
    if text_column:
        if text_column not in record:
            raise ValueError(f"Training dataset has no text column {text_column!r}")
        return str(record.get(text_column) or "").strip()

    direct_columns = ("text", "content", "sentence")
    for column in direct_columns:
        value = record.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()

    composite_columns = (
        "instruction",
        "context",
        "prompt",
        "question",
        "passage",
        "response",
        "completion",
        "answer",
    )
    parts = []
    for column in composite_columns:
        value = record.get(column)
        if value is not None and str(value).strip():
            parts.append(f"{column.capitalize()}: {str(value).strip()}")
    if parts:
        return "\n".join(parts)

    for value in record.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_training_texts(config: FineTuneBaselineConfiguration) -> List[str]:
    from datasets import load_dataset

    dataset_args = [config.dataset]
    if config.dataset_config:
        dataset_args.append(config.dataset_config)
    dataset = load_dataset(*dataset_args, split=config.split)
    texts: List[str] = []
    for record in dataset:
        text = _extract_record_text(record, config.text_column)
        if text:
            texts.append(text)
        if config.max_train_examples is not None and len(texts) >= int(
            config.max_train_examples
        ):
            break
    if not texts:
        raise ValueError("Training dataset yielded no non-empty text examples")
    return texts


def infer_attention_projection_targets(model: torch.nn.Module) -> List[str]:
    targets: List[str] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        module_path = set(lowered.split("."))
        if not name or not module_path.intersection({"attention", "attn", "self_attn"}):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            targets.append(name)
    if not targets:
        raise ValueError(
            "Unable to identify attention projection modules for LoRA; "
            "set up a supported causal-LM architecture."
        )
    return sorted(set(targets))


def _peak_memory_bytes(device: str) -> int:
    if device == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _training_collator(tokenizer, seq_len: int):
    def collate(texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        labels = encoded["input_ids"].clone()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        encoded["labels"] = labels
        return encoded

    return collate


def train_model(
    config: FineTuneBaselineConfiguration,
    output_dir: str,
    progress: ProgressLogger,
) -> TrainingResult:
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device, None)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()

    progress.write("training_data_loading", arm=config.arm, dataset=config.dataset)
    texts = load_training_texts(config)
    generator = torch.Generator().manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        trust_remote_code=config.trust_remote_code,
    )
    model.to(device=device, dtype=torch.float32)
    lora_targets: List[str] = []
    if config.arm == "lora":
        lora_targets = infer_attention_projection_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=lora_targets,
            ),
        )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Fine-tuning arm has no trainable parameters")
    trainable_parameters = int(sum(parameter.numel() for parameter in trainable))
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    data_loader = DataLoader(
        texts,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_training_collator(tokenizer, config.seq_len),
        drop_last=False,
    )
    data_iterator = iter(data_loader)
    budget_steps, budget_seconds, budget_type = resolve_training_budget(config)
    progress.write(
        "training_started",
        arm=config.arm,
        device=device,
        requested_steps=budget_steps,
        requested_seconds=budget_seconds,
        trainable_parameters=trainable_parameters,
    )

    model.train()
    step = 0
    losses: List[float] = []
    training_start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - training_start
        if budget_steps is not None and step >= budget_steps:
            break
        if budget_seconds is not None and step > 0 and elapsed >= budget_seconds:
            break
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            batch = next(data_iterator)

        step_start = time.perf_counter()
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step + 1}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        step += 1
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        progress.write(
            "training_step_completed",
            idx=step,
            scores={"loss": loss_value},
            secs=time.perf_counter() - step_start,
            compute=True,
            elapsed_seconds=time.perf_counter() - training_start,
        )

    training_seconds = time.perf_counter() - training_start
    model.eval()
    if config.arm == "lora":
        model = model.merge_and_unload()
    trained_model_path = os.path.join(output_dir, TRAINED_MODEL_DIRNAME)
    shutil.rmtree(trained_model_path, ignore_errors=True)
    model.save_pretrained(trained_model_path, safe_serialization=True)
    tokenizer.save_pretrained(trained_model_path)

    budget = {
        "schema_version": 1,
        "arm": config.arm,
        "model": config.model,
        "dataset": config.dataset,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "budget_type": budget_type,
        "requested_steps": budget_steps,
        "requested_seconds": budget_seconds,
        "matched_ga_run": config.match_ga_run,
        "actual_steps": step,
        "actual_training_seconds": training_seconds,
        "peak_memory_bytes": _peak_memory_bytes(device),
        "device": device,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "learning_rate": config.learning_rate,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "lora_r": config.lora_r if config.arm == "lora" else None,
        "lora_alpha": config.lora_alpha if config.arm == "lora" else None,
        "lora_target_modules": lora_targets,
        "mean_training_loss": float(np.mean(losses)) if losses else None,
        "final_training_loss": losses[-1] if losses else None,
    }
    atomic_write_json(os.path.join(output_dir, BUDGET_FILENAME), budget)
    progress.write(
        "training_completed",
        idx=step,
        scores={"loss": losses[-1] if losses else None},
        secs=training_seconds,
        actual_steps=step,
        model_path=trained_model_path,
    )
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return TrainingResult(model_path=trained_model_path, budget=budget)


def evaluate_checkpoint(
    model_name: str,
    tasks: List[TaskConfiguration],
    *,
    limit: int,
    config: FineTuneBaselineConfiguration,
    device: str,
) -> Dict[str, Any]:
    from mergekit.evo.helpers import _eval_model
    from mergekit.evo.task_utils import create_task_manager

    task_manager = create_task_manager(
        config.task_search_path,
        required_tasks=[task.name for task in tasks],
    )
    model_args: Dict[str, Any] = {
        "pretrained": model_name,
        "dtype": "float32",
        "use_cache": True,
        "trust_remote_code": config.trust_remote_code,
    }
    if os.path.isdir(model_name):
        model_args["local_files_only"] = True
    return _eval_model(
        "huggingface",
        tasks,
        model_args,
        num_fewshot=config.num_fewshot,
        limit=limit,
        batch_size=config.eval_batch_size,
        task_manager=task_manager,
        fitness_mode="weighted_sum",
        fitness_version=config.fitness.version,
        lower_is_better_transform=config.fitness.lower_is_better_transform,
        bootstrap_iters=0,
        device=device,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
    )


def _write_results_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    preferred = [
        "arm",
        "requested_arm",
        "model_role",
        "model",
        "suite",
        "protocol",
        "limit",
        "weighted_score",
        "fitness_version",
        "lower_is_better_transform",
        "evaluation_seconds",
        "error",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [*preferred, *extra]
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_finetune_experiment(
    config: FineTuneBaselineConfiguration,
    output_dir: str,
    *,
    trainer: Callable[
        [FineTuneBaselineConfiguration, str, ProgressLogger], TrainingResult
    ] = train_model,
    evaluator: Callable[..., Dict[str, Any]] = evaluate_checkpoint,
) -> Dict[str, Any]:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    progress = ProgressLogger(output_dir, seed=config.seed)
    total_start = time.perf_counter()
    progress.write(
        "finetune_run_started",
        arm=config.arm,
        model=config.model,
        dataset=config.dataset,
        split=config.split,
    )

    try:
        training_result = trainer(config, output_dir, progress)
    except KeyboardInterrupt:
        progress.write("finetune_run_interrupted", status="interrupted")
        raise
    except Exception as exc:
        progress.write(
            "finetune_run_failed",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    device = resolve_device(config.device, None)
    protocols = [("search", config.search_limit), ("export", config.export_limit)]
    suites = [
        ("target", config.target_tasks),
        ("retention", config.retention_tasks),
    ]
    model_roles = [
        ("parent", "parent", config.model),
        (config.arm, "trained", training_result.model_path),
    ]
    rows: List[Dict[str, Any]] = []
    evaluation_seconds = 0.0
    for arm_label, model_role, model_name in model_roles:
        for suite_name, tasks in suites:
            for protocol_name, protocol_limit in protocols:
                eval_start = time.perf_counter()
                error = None
                try:
                    result = evaluator(
                        model_name,
                        tasks,
                        limit=protocol_limit,
                        config=config,
                        device=device,
                    )
                except Exception as exc:
                    result = {"score": None, "results": {}}
                    error = f"{type(exc).__name__}: {exc}"
                if error is None and result.get("score") is None:
                    error = "Evaluation returned no weighted score"
                elapsed = time.perf_counter() - eval_start
                evaluation_seconds += elapsed
                row: Dict[str, Any] = {
                    "arm": arm_label,
                    "requested_arm": config.arm,
                    "model_role": model_role,
                    "model": model_name,
                    "suite": suite_name,
                    "protocol": protocol_name,
                    "limit": protocol_limit,
                    "weighted_score": result.get("score"),
                    "fitness_version": config.fitness.version,
                    "lower_is_better_transform": (
                        config.fitness.lower_is_better_transform
                    ),
                    "evaluation_seconds": elapsed,
                    "error": error,
                }
                row.update(collect_task_metrics(result, tasks))
                rows.append(row)
                progress.write(
                    "evaluation_completed",
                    idx=len(rows),
                    scores={"weighted_score": result.get("score")},
                    secs=elapsed,
                    compute=True,
                    arm=arm_label,
                    model_role=model_role,
                    suite=suite_name,
                    protocol=protocol_name,
                    limit=protocol_limit,
                    error=error,
                )

    results_path = _write_results_csv(
        os.path.join(output_dir, FINETUNE_RESULTS_FILENAME), rows
    )
    budget = dict(training_result.budget)
    budget["evaluation_seconds"] = evaluation_seconds
    budget["total_wall_seconds"] = time.perf_counter() - total_start
    atomic_write_json(os.path.join(output_dir, BUDGET_FILENAME), budget)
    failed_evaluations = sum(row.get("error") is not None for row in rows)
    progress.write(
        "finetune_run_finished",
        idx=len(rows),
        secs=budget["total_wall_seconds"],
        status="success" if failed_evaluations == 0 else "failed",
        failed_evaluations=failed_evaluations,
        results_path=results_path,
    )
    if failed_evaluations:
        raise RuntimeError(
            f"{failed_evaluations} fine-tuning evaluation(s) failed; "
            f"see {results_path}"
        )
    return {
        "budget_path": os.path.join(output_dir, BUDGET_FILENAME),
        "results_path": results_path,
        "model_path": training_result.model_path,
        "rows": rows,
    }
