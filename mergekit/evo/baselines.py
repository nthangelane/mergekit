# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas

from mergekit.common import ModelReference
from mergekit.evo.config import EvolMergeConfiguration, TaskConfiguration

LOGGER = logging.getLogger("mergekit.evolve_ga.baselines")
StageLogger = Callable[..., None]


def _stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    LOGGER.log(level, "[%s] %s", stage, message)


def _require_ray():
    import ray

    return ray


def _eval_model(*args, **kwargs):
    from mergekit.evo.helpers import _eval_model as implementation

    return implementation(*args, **kwargs)


def _create_task_manager(*args, **kwargs):
    from mergekit.evo.task_utils import create_task_manager

    return create_task_manager(*args, **kwargs)


def best_weighted_score_from_frame(
    frame: "pandas.DataFrame",
) -> Optional[float]:
    if "weighted_score" not in frame.columns:
        return None
    numeric_scores = pandas.to_numeric(
        frame["weighted_score"], errors="coerce"
    ).dropna()
    if numeric_scores.empty:
        return None
    return float(numeric_scores.max())


def reusable_baseline_csv_path(
    baseline_csv_path: str,
    expected_models: List[str],
    *,
    expected_fitness_version: Optional[str] = None,
    expected_lower_is_better_transform: Optional[str] = None,
    stage_logger: StageLogger = _stage_log,
) -> Optional[str]:
    if not os.path.exists(baseline_csv_path):
        return None
    try:
        frame = pandas.read_csv(baseline_csv_path)
    except Exception as exc:
        stage_logger(
            "Stage-Baseline",
            f"Existing baseline_results.csv could not be read; rerunning baselines: {exc}",
            level=logging.WARNING,
        )
        return None
    if "model" not in frame.columns or "weighted_score" not in frame.columns:
        return None

    expected_metadata = {
        "fitness_version": expected_fitness_version,
        "lower_is_better_transform": expected_lower_is_better_transform,
    }
    for column, expected_value in expected_metadata.items():
        if expected_value is None:
            continue
        if column not in frame.columns:
            return None
        if set(frame[column].dropna().astype(str)) != {str(expected_value)}:
            return None

    reusable_models = set(
        frame.loc[
            pandas.to_numeric(frame["weighted_score"], errors="coerce").notna(),
            "model",
        ]
        .dropna()
        .astype(str)
    )
    return baseline_csv_path if set(expected_models).issubset(reusable_models) else None


def unique_model_refs(model_refs: List[ModelReference]) -> List[ModelReference]:
    unique: List[ModelReference] = []
    seen: set[str] = set()
    for model_ref in model_refs:
        model_name = str(model_ref)
        if model_name not in seen:
            seen.add(model_name)
            unique.append(model_ref)
    return unique


def configured_task_names(config: EvolMergeConfiguration) -> List[str]:
    task_names: List[str] = []
    for task in [*config.tasks, *(config.stage1_tasks or [])]:
        if task.name not in task_names:
            task_names.append(task.name)
    return task_names


def collect_task_metrics(
    result: Dict[str, Any],
    tasks: List[TaskConfiguration],
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for task_cfg in tasks:
        task_results = result.get("results", {}).get(task_cfg.name, {})
        metric_value = task_results.get(task_cfg.metric)
        if metric_value is None:
            metric_alternatives = {
                "ppl,none": [
                    "word_perplexity,none",
                    "perplexity,none",
                    "byte_perplexity,none",
                ],
                "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
                "acc_norm,none": ["acc_norm,none", "acc,none", "accuracy,none"],
            }
            for alternative in metric_alternatives.get(task_cfg.metric, []):
                if alternative in task_results:
                    metric_value = task_results[alternative]
                    break

        if metric_value is None:
            lowered_metric = task_cfg.metric.lower()
            for metric_name, value in task_results.items():
                lowered_name = metric_name.lower()
                if "stderr" in lowered_name:
                    continue
                if (
                    "ppl" in lowered_metric or "perplexity" in lowered_metric
                ) and "perplexity" in lowered_name:
                    metric_value = value
                    break
                if "acc" in lowered_metric and "acc" in lowered_name:
                    metric_value = value
                    break

        if isinstance(metric_value, float) and math.isnan(metric_value):
            metric_value = None
        metrics[f"{task_cfg.name}:{task_cfg.metric}"] = metric_value
    return metrics


def _evaluate_baseline_model(
    model_name: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    batch_size: Optional[int],
    task_search_path: List[str],
    required_tasks: List[str],
    trust_remote_code: bool,
    fitness_mode: str,
    fitness_version: str,
    lower_is_better_transform: str,
    task_mix_profile: Optional[str],
    device: str,
) -> Dict[str, Any]:
    task_manager = _create_task_manager(
        task_search_path,
        required_tasks=required_tasks,
    )
    model_args: Dict[str, Any] = {
        "pretrained": model_name,
        "dtype": "bfloat16" if device == "cuda" else "float32",
        "use_cache": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        result = _eval_model(
            "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            task_manager=task_manager,
            fitness_mode=fitness_mode,
            fitness_version=fitness_version,
            lower_is_better_transform=lower_is_better_transform,
            task_mix_profile=task_mix_profile,
            bootstrap_iters=0,
            device=device,
        )
        return {
            "score": result.get("score"),
            "results": result.get("results"),
            "error": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "results": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_baseline_evaluations(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
    ray_observer=None,
    use_ray: bool = True,
    *,
    stage_logger: StageLogger = _stage_log,
    ray_initializer: Optional[Callable[[], None]] = None,
    ray_loader: Callable[[], Any] = _require_ray,
) -> Optional[str]:
    stage_logger("Stage-Baseline", "Starting baseline evaluation phase...")
    storage_dir = os.path.abspath(storage_path)
    os.makedirs(storage_dir, exist_ok=True)

    models: List[ModelReference] = list(config.genome.models)
    if config.genome.base_model is not None:
        models.append(config.genome.base_model)
    models = unique_model_refs(models)
    if not models:
        stage_logger(
            "Stage-Baseline", "No models found in the genome; skipping baselines."
        )
        if ray_observer is not None:
            ray_observer.set_phase(
                "baseline", baseline_model_index=0, baseline_model_total=0
            )
        return None

    if ray_observer is not None:
        ray_observer.set_phase(
            "baseline", baseline_model_index=0, baseline_model_total=len(models)
        )

    baseline_csv_path = reusable_baseline_csv_path(
        os.path.join(storage_dir, "baseline_results.csv"),
        [str(model_ref) for model_ref in models],
        expected_fitness_version=config.fitness.version,
        expected_lower_is_better_transform=(config.fitness.lower_is_better_transform),
        stage_logger=stage_logger,
    )
    if baseline_csv_path is not None:
        stage_logger(
            "Stage-Baseline",
            f"Reusing existing baseline metrics from {baseline_csv_path}",
        )
        if ray_observer is not None:
            ray_observer.set_phase(
                "baseline",
                baseline_model_index=len(models),
                baseline_model_total=len(models),
            )
        return baseline_csv_path

    required_tasks = configured_task_names(config)
    _create_task_manager(task_search_path, required_tasks=required_tasks)
    use_cuda = (merge_cuda or (num_gpus or 0) > 0) and (num_gpus or 0) > 0
    device = "cuda" if use_cuda else "cpu"
    stage_logger(
        "Stage-Baseline",
        (
            f"Using Ray {'GPU' if use_cuda else 'CPU'} workers for baseline evaluations."
            if use_ray
            else "Using local CPU execution for serial baseline evaluations."
        ),
    )

    def baseline_args(model_ref: ModelReference) -> Tuple[Any, ...]:
        return (
            str(model_ref),
            config.tasks,
            config.num_fewshot,
            config.limit,
            batch_size,
            task_search_path,
            required_tasks,
            trust_remote_code,
            config.fitness_mode,
            config.fitness.version,
            config.fitness.lower_is_better_transform,
            config.task_mix_profile,
            device,
        )

    ray = None
    baseline_refs: Dict[str, Any] = {}
    if use_ray:
        (ray_initializer or (lambda: None))()
        ray = ray_loader()
        baseline_remote = ray.remote(num_cpus=1, num_gpus=1.0 if use_cuda else 0)(
            _evaluate_baseline_model
        )
        baseline_refs = {
            str(model_ref): baseline_remote.remote(*baseline_args(model_ref))
            for model_ref in models
        }

    metric_columns = [f"{task.name}:{task.metric}" for task in config.tasks]
    baseline_rows: List[Dict[str, Union[str, float, None]]] = []
    successes = 0
    failures = 0
    for model_index, model_ref in enumerate(models, start=1):
        model_name = str(model_ref)
        row: Dict[str, Union[str, float, None]] = {
            "model": model_name,
            "weighted_score": None,
            "fitness_version": config.fitness.version,
            "lower_is_better_transform": config.fitness.lower_is_better_transform,
            "error": None,
        }
        for column in metric_columns:
            row[column] = None

        stage_logger("Stage-Baseline", f"Evaluating {model_name}...")
        try:
            result = (
                ray.get(baseline_refs[model_name])
                if ray is not None
                else _evaluate_baseline_model(*baseline_args(model_ref))
            )
        except Exception as exc:  # pragma: no cover - cluster-specific failure
            result = {"score": None, "results": None, "error": str(exc)}
            LOGGER.debug("Baseline evaluation error", exc_info=exc)

        if not result or result.get("error"):
            failures += 1
            row["error"] = (
                result.get("error")
                if result
                else "Baseline evaluation returned no result"
            )
            stage_logger(
                "Stage-Baseline",
                f"Evaluation failed for {model_name}: {row['error']}",
                level=logging.ERROR,
            )
            if ray_observer is not None:
                ray_observer.record_baseline_progress(
                    model_index=model_index,
                    model_total=len(models),
                    model_name=model_name,
                    score=None,
                    failed=True,
                )
            baseline_rows.append(row)
            continue

        successes += 1
        weighted_score = result.get("score")
        row["weighted_score"] = weighted_score
        row.update(collect_task_metrics(result, config.tasks))
        if ray_observer is not None:
            ray_observer.record_baseline_progress(
                model_index=model_index,
                model_total=len(models),
                model_name=model_name,
                score=(
                    float(weighted_score)
                    if weighted_score is not None and math.isfinite(weighted_score)
                    else None
                ),
                failed=False,
            )
        baseline_rows.append(row)

    if not baseline_rows:
        stage_logger(
            "Stage-Baseline", "No baseline results recorded; skipping CSV output."
        )
        return None

    ordered_columns = [
        "model",
        "weighted_score",
        "fitness_version",
        "lower_is_better_transform",
        *metric_columns,
        "error",
    ]
    baseline_df = pandas.DataFrame(baseline_rows)
    for column in ordered_columns:
        if column not in baseline_df.columns:
            baseline_df[column] = None
    baseline_df = baseline_df[ordered_columns]
    baseline_df.sort_values(
        "weighted_score", ascending=False, inplace=True, na_position="last"
    )
    baseline_csv_path = os.path.join(storage_dir, "baseline_results.csv")
    baseline_df.to_csv(baseline_csv_path, index=False)

    stage_logger("Stage-Baseline", f"Baseline metrics saved to {baseline_csv_path}")
    stage_logger(
        "Stage-Baseline",
        f"Completed evaluations: {successes}; failures: {failures}",
    )
    if ray_observer is not None:
        ray_observer.set_phase(
            "baseline",
            baseline_model_index=len(models),
            baseline_model_total=len(models),
            failed_evals=failures,
        )
    return baseline_csv_path
