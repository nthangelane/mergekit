# Copyright (C) 2026 Nkululeko Thangelane
#
# Deterministic two-parent linear sweep for local validation experiments.

import csv
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import click
import yaml

from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration, TaskConfiguration
from mergekit.evo.helpers import evaluate_model_cpu, validate_input_model_architecture
from mergekit.evo.task_utils import create_task_manager
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

LOGGER = logging.getLogger("mergekit.linear_sweep")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    LOGGER.log(level, "[%s] %s", stage, message)


@dataclass(frozen=True)
class SweepCandidate:
    alpha: float
    index: int


def _round_alpha(value: float) -> float:
    return round(float(value), 6)


def generate_alphas(
    alpha_step: Optional[float] = None,
    explicit_alphas: Optional[Sequence[float]] = None,
) -> List[float]:
    if explicit_alphas:
        cleaned = sorted({_round_alpha(a) for a in explicit_alphas})
        if not cleaned:
            raise ValueError("No valid alpha values provided.")
        for alpha in cleaned:
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"Alpha {alpha} is outside the valid [0, 1] range.")
        return cleaned

    step = 0.05 if alpha_step is None else float(alpha_step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("--alpha-step must be in the (0, 1] interval.")

    alphas: List[float] = []
    current = 0.0
    while current < 1.0:
        alphas.append(_round_alpha(current))
        current += step
    alphas.append(1.0)
    return sorted(set(alphas))


def select_top_candidates(
    rows: Sequence[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    successful = [
        row
        for row in rows
        if row.get("weighted_score") is not None
        and math.isfinite(float(row["weighted_score"]))
        and not row.get("error")
    ]
    ranked = sorted(
        successful,
        key=lambda row: (-float(row["weighted_score"]), int(row["candidate_index"])),
    )
    return ranked[: max(0, int(top_k))]


def _flatten_metrics(
    result: Dict[str, Any], tasks: Sequence[TaskConfiguration]
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    task_results = result.get("results") or {}
    for task in tasks:
        metrics = task_results.get(task.name) or {}
        if not isinstance(metrics, dict):
            continue
        metric_name = str(task.metric)
        value = metrics.get(metric_name)
        if value is None and "," in metric_name:
            base_name = metric_name.split(",", 1)[0]
            for key, metric_value in metrics.items():
                if key.startswith(base_name):
                    value = metric_value
                    break
        flattened[f"{task.name}:{metric_name}"] = value
    return flattened


def build_linear_merge_config(
    *,
    model_a: str,
    model_b: str,
    alpha: float,
    tokenizer_source: Optional[Any],
) -> MergeConfiguration:
    alpha = _round_alpha(alpha)
    beta = _round_alpha(1.0 - alpha)
    return MergeConfiguration.model_validate(
        {
            "merge_method": "linear",
            "models": [
                {"model": model_a, "parameters": {"weight": float(alpha)}},
                {"model": model_b, "parameters": {"weight": float(beta)}},
            ],
            "parameters": {"normalize": True, "int8_mask": True},
            "dtype": "bfloat16",
            "tokenizer_source": tokenizer_source,
        }
    )


def _write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_summary(
    *,
    model_a: str,
    model_b: str,
    coarse_limit: int,
    refine_limit: Optional[int],
    top_k: int,
    coarse_rows: Sequence[Dict[str, Any]],
    refine_rows: Sequence[Dict[str, Any]],
    final_best: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "Deterministic linear sweep summary",
        "",
        f"- Model A: {model_a}",
        f"- Model B: {model_b}",
        f"- Coarse limit: {coarse_limit}",
        f"- Refine limit: {refine_limit if refine_limit is not None else 'disabled'}",
        f"- Requested top-k: {top_k}",
        f"- Coarse candidates: {len(coarse_rows)}",
        f"- Refined candidates: {len(refine_rows)}",
    ]
    if final_best is not None:
        lines.extend(
            [
                f"- Best alpha: {final_best['alpha']}",
                f"- Best stage: {final_best['stage']}",
                f"- Best score: {final_best['weighted_score']}",
            ]
        )
    else:
        lines.append("- Best result: none")
    return "\n".join(lines) + "\n"


def _run_candidate(
    *,
    config: EvolMergeConfiguration,
    merge_options: MergeOptions,
    storage_path: Path,
    task_manager: Any,
    candidate: SweepCandidate,
    model_a: str,
    model_b: str,
    stage_name: str,
    eval_limit: int,
    batch_size: int,
    dry_run: bool,
) -> Dict[str, Any]:
    merge_config = build_linear_merge_config(
        model_a=model_a,
        model_b=model_b,
        alpha=candidate.alpha,
        tokenizer_source=config.genome.tokenizer_source,
    )

    candidate_dir = storage_path / "candidate_configs"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_config_path = (
        candidate_dir / f"{stage_name}_alpha_{candidate.alpha:.6f}.yaml"
    )
    candidate_config_path.write_text(merge_config.to_yaml() + "\n", encoding="utf-8")

    row: Dict[str, Any] = {
        "stage": stage_name,
        "candidate_index": candidate.index,
        "alpha": candidate.alpha,
        "beta": round(1.0 - candidate.alpha, 6),
        "model_a": model_a,
        "model_b": model_b,
        "limit": eval_limit,
        "weighted_score": None,
        "error_stage": None,
        "error_type": None,
        "error": None,
        "config_path": str(candidate_config_path),
    }

    if dry_run:
        row["error"] = "dry_run"
        return row

    merged_root = storage_path / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    merged_path = Path(tempfile.mkdtemp(prefix="merged-", dir=str(merged_root)))

    try:
        run_merge(
            merge_config,
            out_path=str(merged_path),
            options=merge_options,
            config_source=merge_config.to_yaml(),
        )
    except Exception as exc:
        shutil.rmtree(merged_path, ignore_errors=True)
        row.update(
            {
                "error_stage": "merge",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return row

    result = evaluate_model_cpu(
        str(merged_path),
        config.tasks,
        config.num_fewshot,
        eval_limit,
        batch_size=batch_size,
        task_manager=task_manager,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
    )
    row["weighted_score"] = result.get("score")
    row["error_stage"] = result.get("error_stage")
    row["error_type"] = result.get("error_type")
    row["error"] = result.get("error_message")
    row.update(_flatten_metrics(result, config.tasks))
    return row


def _save_final_model(
    *,
    best_row: Dict[str, Any],
    config: EvolMergeConfiguration,
    merge_options: MergeOptions,
    storage_path: Path,
    model_a: str,
    model_b: str,
) -> None:
    final_dir = storage_path / "final_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)

    merge_config = build_linear_merge_config(
        model_a=model_a,
        model_b=model_b,
        alpha=float(best_row["alpha"]),
        tokenizer_source=config.genome.tokenizer_source,
    )
    run_merge(
        merge_config,
        out_path=str(final_dir),
        options=merge_options,
        config_source=merge_config.to_yaml(),
    )

    (storage_path / "best_config.yaml").write_text(
        merge_config.to_yaml() + "\n", encoding="utf-8"
    )


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--storage-path",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for sweep artifacts.",
)
@click.option(
    "--alpha-step",
    type=float,
    default=0.05,
    show_default=True,
    help="Step size for deterministic alpha generation in [0, 1].",
)
@click.option(
    "--alphas",
    type=str,
    default=None,
    help="Comma-separated explicit alpha list. Overrides --alpha-step.",
)
@click.option(
    "--coarse-limit",
    type=int,
    default=4,
    show_default=True,
    help="Cheap first-pass evaluation limit.",
)
@click.option(
    "--refine-limit",
    type=int,
    default=32,
    show_default=True,
    help="Higher-fidelity reevaluation limit for promoted candidates.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Number of coarse candidates to promote for reevaluation.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="lm-eval batch size.",
)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=False,
    show_default=True,
)
@click.option(
    "--allow-benchmark-tasks/--disallow-benchmark-tasks",
    default=False,
    show_default=True,
)
@click.option(
    "--save-final-model/--no-save-final-model",
    default=True,
    show_default=True,
    help="Rebuild and keep the best candidate after the sweep.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="Write planned candidates and summaries without merging or evaluating.",
)
def main(
    config_path: Path,
    storage_path: Path,
    alpha_step: float,
    alphas: Optional[str],
    coarse_limit: int,
    refine_limit: int,
    top_k: int,
    batch_size: int,
    trust_remote_code: bool,
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    dry_run: bool,
) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = EvolMergeConfiguration.model_validate(raw_config)

    from mergekit.evo.config import check_for_naughty_config

    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    source_models = list(config.genome.models)
    if len(source_models) != 2:
        raise click.ClickException(
            "Deterministic linear sweep requires exactly two source models."
        )

    explicit_alphas = None
    if alphas:
        explicit_alphas = [
            float(part.strip()) for part in alphas.split(",") if part.strip()
        ]

    alpha_values = generate_alphas(alpha_step, explicit_alphas)
    candidates = [
        SweepCandidate(alpha=alpha, index=index)
        for index, alpha in enumerate(alpha_values)
    ]

    storage_path = storage_path.resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    stage_log("Stage-Init", f"Storage path: {storage_path}")
    stage_log("Stage-Init", f"Evaluating {len(candidates)} deterministic alpha values")

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=False,
        low_cpu_memory=False,
        out_shard_size=1_000_000_000_000,
        trust_remote_code=trust_remote_code,
        random_seed=42,
        quiet=True,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    validate_input_model_architecture(source_models, merge_options)
    task_manager = create_task_manager(
        [],
        required_tasks=[task.name for task in config.tasks],
    )

    model_a = str(source_models[0])
    model_b = str(source_models[1])

    coarse_rows = [
        _run_candidate(
            config=config,
            merge_options=merge_options,
            storage_path=storage_path,
            task_manager=task_manager,
            candidate=candidate,
            model_a=model_a,
            model_b=model_b,
            stage_name="coarse",
            eval_limit=coarse_limit,
            batch_size=batch_size,
            dry_run=dry_run,
        )
        for candidate in candidates
    ]
    _write_rows(storage_path / "coarse_results.csv", coarse_rows)

    refine_rows: List[Dict[str, Any]] = []
    promoted = select_top_candidates(coarse_rows, top_k)
    if refine_limit > coarse_limit and promoted and not dry_run:
        stage_log(
            "Stage-Refine",
            f"Promoting {len(promoted)} coarse candidates to limit {refine_limit}",
        )
        refine_rows = [
            _run_candidate(
                config=config,
                merge_options=merge_options,
                storage_path=storage_path,
                task_manager=task_manager,
                candidate=SweepCandidate(
                    alpha=float(row["alpha"]),
                    index=int(row["candidate_index"]),
                ),
                model_a=model_a,
                model_b=model_b,
                stage_name="refine",
                eval_limit=refine_limit,
                batch_size=batch_size,
                dry_run=False,
            )
            for row in promoted
        ]
    _write_rows(storage_path / "refine_results.csv", refine_rows)

    final_pool = refine_rows or coarse_rows
    ranked_final = select_top_candidates(final_pool, 1)
    best_row = ranked_final[0] if ranked_final else None
    if best_row and save_final_model and not dry_run:
        stage_log("Stage-Final", f"Saving final model for alpha={best_row['alpha']}")
        _save_final_model(
            best_row=best_row,
            config=config,
            merge_options=merge_options,
            storage_path=storage_path,
            model_a=model_a,
            model_b=model_b,
        )

    summary = _build_summary(
        model_a=model_a,
        model_b=model_b,
        coarse_limit=coarse_limit,
        refine_limit=refine_limit if refine_limit > coarse_limit else None,
        top_k=top_k,
        coarse_rows=coarse_rows,
        refine_rows=refine_rows,
        final_best=best_row,
    )
    (storage_path / "summary.txt").write_text(summary, encoding="utf-8")
    if best_row is not None:
        stage_log(
            "Stage-Done",
            (
                f"Best candidate alpha={best_row['alpha']} "
                f"stage={best_row['stage']} score={best_row['weighted_score']}"
            ),
        )
    else:
        stage_log("Stage-Done", "No successful sweep candidates were produced.")


if __name__ == "__main__":
    main()
