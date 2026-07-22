# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import csv
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas
import torch

from mergekit.evo.baselines import collect_task_metrics, configured_task_names
from mergekit.evo.checkpoint import GA_STATE_FILENAME
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.fitness import FITNESS_DEFINITION_FILENAME
from mergekit.evo.provenance import PARENT_LINEAGE_FILENAME

LOGGER = logging.getLogger("mergekit.evolve_ga.reporting")
StageLogger = Callable[..., None]


def _stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    LOGGER.log(level, "[%s] %s", stage, message)


def _eval_model(*args, **kwargs):
    from mergekit.evo.helpers import _eval_model as implementation

    return implementation(*args, **kwargs)


def _create_task_manager(*args, **kwargs):
    from mergekit.evo.task_utils import create_task_manager

    return create_task_manager(*args, **kwargs)


def sanitize_metric_key_fragment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return cleaned or "unknown"


def configured_merge_methods(config: EvolMergeConfiguration) -> List[str]:
    genome_config = config.genome
    if hasattr(genome_config, "allowed_methods"):
        return [str(method) for method in genome_config.allowed_methods]
    merge_method = getattr(genome_config, "merge_method", None)
    return [str(merge_method)] if merge_method else []


def collect_merge_method_outcomes(
    genotype_iterable: List[np.ndarray],
    results: List[dict],
    genome: Any,
    configured_methods: List[str],
) -> Dict[str, Any]:
    method_counter: Counter[str] = Counter()
    success_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    score_values: Dict[str, List[float]] = defaultdict(list)
    all_methods = {str(method) for method in configured_methods}

    for genotype_candidate, result in zip(genotype_iterable, results):
        try:
            if hasattr(genome, "method_label_for_genotype"):
                method_name = str(genome.method_label_for_genotype(genotype_candidate))
            else:
                config = (
                    genome.genotype_to_merge_config(genotype_candidate)
                    if hasattr(genome, "genotype_to_merge_config")
                    else genome.genotype_merge_config(genotype_candidate)
                )
                method_name = str(getattr(config, "merge_method", None) or "unknown")
        except Exception:  # pragma: no cover - diagnostic only
            LOGGER.debug("Unable to decode merge method", exc_info=True)
            method_name = "decode_error"

        all_methods.add(method_name)
        method_counter[method_name] += 1
        score = result.get("score")
        if score is None:
            failure_counter[method_name] += 1
        else:
            success_counter[method_name] += 1
            score_values[method_name].append(float(score))

    metrics: Dict[str, float] = {}
    history_rows: List[Dict[str, Union[str, float, int, None]]] = []
    for method_name in sorted(all_methods):
        total = int(method_counter.get(method_name, 0))
        successes = int(success_counter.get(method_name, 0))
        failures = int(failure_counter.get(method_name, 0))
        success_rate = float(successes / total) if total else 0.0
        metric_key = sanitize_metric_key_fragment(method_name)
        metrics[f"merge_method/{metric_key}/count"] = float(total)
        metrics[f"merge_method/{metric_key}/success_count"] = float(successes)
        metrics[f"merge_method/{metric_key}/failure_count"] = float(failures)
        metrics[f"merge_method/{metric_key}/success_rate"] = success_rate

        values = score_values.get(method_name, [])
        mean_score = float(sum(values) / len(values)) if values else None
        best_score = float(max(values)) if values else None
        if mean_score is not None:
            metrics[f"merge_method/{metric_key}/mean_score"] = mean_score
            metrics[f"merge_method/{metric_key}/best_score"] = best_score
        history_rows.append(
            {
                "merge_method": method_name,
                "count": total,
                "success_count": successes,
                "failure_count": failures,
                "success_rate": success_rate,
                "mean_score": mean_score,
                "best_score": best_score,
            }
        )

    return {
        "method_counts": method_counter,
        "method_success_counts": success_counter,
        "method_failure_counts": failure_counter,
        "metrics": metrics,
        "history_rows": history_rows,
    }


def classify_solution_novelty(
    method_name: str,
    genotype_candidate: Optional[np.ndarray] = None,
    genome: Optional[Any] = None,
) -> Dict[str, Any]:
    normalized_method = str(method_name or "unknown")
    if normalized_method == "passthrough":
        return {
            "is_novel_solution": False,
            "novelty_class": "baseline_control",
            "novelty_reason": "passthrough preserves one parent and is not a new merge",
        }
    if normalized_method in {"decode_error", "unknown"}:
        return {
            "is_novel_solution": False,
            "novelty_class": "unknown",
            "novelty_reason": "candidate method could not be decoded",
        }
    if (
        genotype_candidate is None
        or genome is None
        or not hasattr(genome, "genotype_to_param_arrays")
    ):
        return {
            "is_novel_solution": True,
            "novelty_class": "candidate_merge",
            "novelty_reason": "non-passthrough merge method",
        }

    try:
        params = genome.genotype_to_param_arrays(genotype_candidate)
    except Exception:  # pragma: no cover - diagnostic only
        LOGGER.debug("Unable to decode genotype novelty details", exc_info=True)
        return {
            "is_novel_solution": True,
            "novelty_class": "candidate_merge",
            "novelty_reason": "non-passthrough merge method",
        }

    methods = [str(method) for method in params.get("merge_method", [])]
    if methods and all(method == "passthrough" for method in methods):
        return {
            "is_novel_solution": False,
            "novelty_class": "baseline_control",
            "novelty_reason": "all layer groups are passthrough",
        }

    selection_columns = [
        key
        for key in params.keys()
        if key.startswith("model_") and key.endswith("_selection")
    ]
    minimum_sources = None
    for row_index, layer_method in enumerate(methods):
        if layer_method == "passthrough":
            continue
        selected = 0
        for column in selection_columns:
            values = params.get(column) or []
            if row_index >= len(values):
                continue
            try:
                selected += int(float(values[row_index] or 0.0) > 1e-6)
            except (TypeError, ValueError):
                continue
        minimum_sources = (
            selected if minimum_sources is None else min(minimum_sources, selected)
        )

    if minimum_sources is not None and minimum_sources < 2:
        return {
            "is_novel_solution": False,
            "novelty_class": "degenerate_merge",
            "novelty_reason": "non-passthrough layer selects fewer than two sources",
        }
    return {
        "is_novel_solution": True,
        "novelty_class": "candidate_merge",
        "novelty_reason": "uses a non-passthrough merge over multiple sources",
    }


def _append_history_rows(
    output_path: str,
    generation: int,
    fevals: int,
    rows: List[Dict[str, Any]],
) -> None:
    file_exists = os.path.exists(output_path)
    dynamic_fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in dynamic_fields:
                dynamic_fields.append(key)
    with open(output_path, "a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["generation", "fevals", *dynamic_fields],
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"generation": generation, "fevals": fevals, **row})


def write_merge_method_history(
    storage_path: str,
    generation: int,
    fevals: int,
    rows: List[Dict[str, Any]],
) -> None:
    _append_history_rows(
        os.path.join(storage_path, "ga_method_history.csv"),
        generation,
        fevals,
        rows,
    )


def write_candidate_history(
    storage_path: str,
    generation: int,
    fevals: int,
    rows: List[Dict[str, Any]],
) -> None:
    _append_history_rows(
        os.path.join(storage_path, "ga_candidate_history.csv"),
        generation,
        fevals,
        rows,
    )


def log_run_artifacts(tracker, storage_path: str) -> None:
    artifacts = {
        "ga_history": "ga_history.csv",
        "ga_candidate_history": "ga_candidate_history.csv",
        "ga_audit_history": "ga_audit_history.csv",
        "ga_reentry_history": "ga_reentry_history.csv",
        "ga_method_history": "ga_method_history.csv",
        "ga_summary": "ga_summary.txt",
        "ga_stop_details": "ga_stop_details.json",
        "ga_history_plot": "ga_history_plot.png",
        "baseline_results": "baseline_results.csv",
        "failed_genotypes": "failed_genotypes.csv",
        "failed_genotype_blacklist": "failed_genotype_blacklist.csv",
        "final_comparison": "ga_final_comparison.csv",
        "final_comparison_plot": "ga_final_comparison.png",
        "mlflow_run_info": "mlflow_run_info.md",
        "mlflow_ui_log": "mlflow_ui.log",
        "ray_observability": "ray_observability.json",
        "parent_lineage": PARENT_LINEAGE_FILENAME,
        "fitness_definition": FITNESS_DEFINITION_FILENAME,
        "ga_state": GA_STATE_FILENAME,
        "run_abort": "run_abort.json",
        "final_repair": "final_repair.json",
    }
    for artifact_name, filename in artifacts.items():
        path = os.path.join(storage_path, filename)
        if os.path.exists(path):
            tracker.log_artifact(path, artifact_name)


def score_improvement(
    current_score: float,
    baseline_score: float,
) -> Tuple[float, Optional[float]]:
    delta = current_score - baseline_score
    baseline_magnitude = abs(float(baseline_score))
    if baseline_magnitude == 0.0:
        return delta, None
    return delta, (delta / baseline_magnitude) * 100.0


def meets_improvement_thresholds(
    delta: float,
    percentage: Optional[float],
    min_abs: float,
    min_pct: float,
) -> bool:
    if delta < min_abs:
        return False
    if min_pct <= 0.0:
        return True
    return percentage is not None and percentage >= min_pct


def write_stop_details(
    storage_path: str,
    *,
    resolved_stop: Dict[str, Any],
    stop_details: Optional[Dict[str, Any]],
) -> str:
    output_path = os.path.join(storage_path, "ga_stop_details.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            {"resolved_stop": resolved_stop, "final_stop": stop_details or {}},
            output_file,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    return output_path


def evaluate_and_write_final_comparison(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
    *,
    stage_logger: StageLogger = _stage_log,
) -> None:
    baseline_csv = os.path.join(storage_path, "baseline_results.csv")
    final_model_path = os.path.join(storage_path, "final_model")
    if not os.path.exists(final_model_path):
        stage_logger(
            "Stage-GA", "final_model not found; skipping final comparison table."
        )
        return

    baseline_rows: List[Dict[str, Union[str, float, None]]] = []
    if os.path.exists(baseline_csv):
        baseline_rows = pandas.read_csv(baseline_csv).to_dict(orient="records")
    else:
        stage_logger(
            "Stage-GA",
            "baseline_results.csv not found; comparison will include only final model.",
            level=logging.WARNING,
        )

    task_manager = _create_task_manager(
        task_search_path,
        required_tasks=configured_task_names(config),
    )
    use_cuda = torch.cuda.is_available() and (merge_cuda or (num_gpus or 0) > 0)
    device = "cuda" if use_cuda else "cpu"
    audit_config = getattr(config, "audit", None)
    audited = bool(audit_config is not None and audit_config.enabled)
    comparison_limit = audit_config.limit if audited else config.limit
    stage_logger("Stage-GA", "Evaluating final merged model for comparison table...")
    try:
        result = _eval_model(
            "huggingface",
            config.tasks,
            {
                "pretrained": final_model_path,
                "dtype": "float32",
                "use_cache": True,
                "trust_remote_code": trust_remote_code,
            },
            num_fewshot=config.num_fewshot,
            limit=comparison_limit,
            batch_size=batch_size,
            task_manager=task_manager,
            fitness_mode=config.fitness_mode,
            fitness_version=config.fitness.version,
            lower_is_better_transform=config.fitness.lower_is_better_transform,
            task_mix_profile=config.task_mix_profile,
            bootstrap_iters=0,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - runtime-dependent
        stage_logger(
            "Stage-GA",
            f"Final model evaluation failed; skipping comparison table: {exc}",
            level=logging.ERROR,
        )
        return

    metric_columns = [f"{task.name}:{task.metric}" for task in config.tasks]
    final_row: Dict[str, Union[str, float, None]] = {
        "model": "final_merged",
        "weighted_score": result.get("score"),
        "fitness_version": config.fitness.version,
        "lower_is_better_transform": config.fitness.lower_is_better_transform,
        "audited": audited,
        "error": None,
    }
    final_row.update(collect_task_metrics(result, config.tasks))
    comparison = pandas.DataFrame([*baseline_rows, final_row])
    ordered_columns = [
        "model",
        "weighted_score",
        "fitness_version",
        "lower_is_better_transform",
        "audited",
        *metric_columns,
        "error",
    ]
    for column in ordered_columns:
        if column not in comparison.columns:
            comparison[column] = None
    comparison = comparison[ordered_columns]
    comparison.to_csv(
        os.path.join(storage_path, "ga_final_comparison.csv"), index=False
    )
    write_comparison_plot(
        comparison,
        os.path.join(storage_path, "ga_final_comparison.png"),
        stage_logger=stage_logger,
    )


def write_comparison_plot(
    table: "pandas.DataFrame",
    output_path: str,
    *,
    stage_logger: StageLogger = _stage_log,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(10, max(2.5, 0.35 * (len(table) + 1))))
        axes.axis("off")
        display = table.copy()
        if "weighted_score" in display.columns:
            display["weighted_score"] = display["weighted_score"].map(
                lambda value: (
                    f"{value:.4f}" if isinstance(value, (int, float)) else value
                )
            )
        rendered_table = axes.table(
            cellText=display.values,
            colLabels=display.columns,
            cellLoc="center",
            loc="center",
        )
        rendered_table.auto_set_font_size(False)
        rendered_table.set_fontsize(8)
        rendered_table.scale(1.0, 1.2)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
    except Exception as exc:  # pragma: no cover - optional plotting
        stage_logger(
            "Stage-GA",
            f"Skipping comparison plot generation: {exc}",
            level=logging.WARNING,
        )


def write_ga_outputs(
    storage_path: str,
    *,
    stop_details: Optional[Dict[str, Any]] = None,
    stage_logger: StageLogger = _stage_log,
) -> None:
    history_path = os.path.join(storage_path, "ga_history.csv")
    if not os.path.exists(history_path):
        stage_logger("Stage-GA", "ga_history.csv not found; skipping summary outputs.")
        return
    with open(history_path, "r", encoding="utf-8", newline="") as history_file:
        rows = list(csv.DictReader(history_file))
    if not rows:
        stage_logger("Stage-GA", "ga_history.csv is empty; skipping summary outputs.")
        return

    with open(
        os.path.join(storage_path, "ga_summary.txt"), "w", encoding="utf-8"
    ) as summary_file:
        summary_file.write(
            "gen  fevals  gen_best   best_so_far  eval_s  cache_hits  crossover\n"
        )
        for row in rows:
            try:
                generation = int(float(row.get("generation", "0") or 0))
            except (TypeError, ValueError):
                generation = 0
            try:
                fevals = int(float(row.get("fevals", "0") or 0))
            except (TypeError, ValueError):
                fevals = 0
            gen_best_raw = row.get("gen_best", "")
            if gen_best_raw in ("", "None", "none", "-inf"):
                generation_best = float("nan")
            else:
                try:
                    generation_best = float(gen_best_raw)
                except (TypeError, ValueError):
                    generation_best = float("nan")
            best_raw = row.get("best_so_far", "")
            best = "NaN" if best_raw in ("", "None", "-inf") else best_raw
            eval_seconds_raw = row.get("eval_seconds", "0")
            try:
                evaluation_seconds = float(eval_seconds_raw)
            except (TypeError, ValueError):
                evaluation_seconds = 0.0
            try:
                cache_hits = int(float(row.get("cache_hits", "0") or 0))
            except (TypeError, ValueError):
                cache_hits = 0
            crossover = row.get("crossover_type", "")
            summary_file.write(
                f"{generation:>2}  {fevals:>6}  {generation_best:>8.5f}  "
                f"{best:>10}  {evaluation_seconds:>6.1f}     {cache_hits:>3}       "
                f"{crossover}\n"
            )

        def _safe_float(value, default=float("nan")):
            if value in (None, "", "None", "none", "-inf", "inf"):
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        generation_best_values = [_safe_float(row.get("gen_best")) for row in rows]
        finite_values = [
            value for value in generation_best_values if math.isfinite(value)
        ]
        blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        summary_file.write("\nGen-best sparkline:\n")
        if not finite_values:
            summary_file.write("(insufficient finite values)\nmin=N/A max=N/A\n")
        else:
            minimum = min(finite_values)
            maximum = max(finite_values)
            if maximum == minimum:
                sparkline = "".join(blocks[0] for _ in generation_best_values)
            else:
                sparkline = "".join(
                    blocks[
                        min(
                            len(blocks) - 1,
                            max(
                                0,
                                int(
                                    (
                                        (value if math.isfinite(value) else minimum)
                                        - minimum
                                    )
                                    / (maximum - minimum)
                                    * (len(blocks) - 1)
                                ),
                            ),
                        )
                    ]
                    for value in generation_best_values
                )
            summary_file.write(f"{sparkline}\nmin={minimum:.5f} max={maximum:.5f}\n")
        if stop_details:
            summary_file.write("\nStop details:\n")
            for key in sorted(stop_details):
                summary_file.write(f"{key}={stop_details[key]}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _safe_int(value):
            try:
                return int(float(value or 0))
            except (TypeError, ValueError):
                return 0

        generations = [_safe_int(row.get("generation")) for row in rows]

        def _plot_safe_float(value):
            if value in (None, "", "None", "none", "-inf", "inf"):
                return float("nan")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        generation_best = [_plot_safe_float(row.get("gen_best")) for row in rows]
        generation_mean = [_plot_safe_float(row.get("gen_mean")) for row in rows]
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(generations, generation_best, marker="o", label="gen_best")
        plt.plot(generations, generation_mean, marker="x", label="gen_mean")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("GA Progress")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(storage_path, "ga_history_plot.png"), dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover - optional plotting
        stage_logger(
            "Stage-GA",
            f"Skipping ga_history_plot.png generation: {exc}",
            level=logging.WARNING,
        )
