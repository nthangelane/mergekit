"""Finalize an evolutionary run whose search completed before export."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import click
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.progress import ProgressLogger, iter_progress_events
from mergekit.evo.reporting import evaluate_and_write_final_comparison, write_ga_outputs
from mergekit.evo.resources import ensure_free_disk
from mergekit.evo.run_validation import (
    EvoRunValidation,
    EvoRunValidationError,
    validate_evo_run,
)
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


class EvoRunFinalizationError(ValueError):
    """Raised when an interrupted run is not safe to finalize in place."""


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvoRunFinalizationError(f"Unable to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvoRunFinalizationError(f"{path} must contain a JSON object")
    return payload


def _read_candidate_scores(path: Path) -> tuple[int, list[float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
    except (OSError, csv.Error) as exc:
        raise EvoRunFinalizationError(f"Unable to read {path}: {exc}") from exc
    scores = []
    for row in rows:
        try:
            score = float(row.get("score", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append(score)
    return len(rows), scores


def _completed_search_context(run_dir: Path) -> Dict[str, Any]:
    try:
        events = list(iter_progress_events(str(run_dir)))
    except (FileNotFoundError, ValueError) as exc:
        raise EvoRunFinalizationError(str(exc)) from exc
    start_indexes = [
        index for index, event in enumerate(events) if event.get("msg") == "run_started"
    ]
    if not start_indexes:
        raise EvoRunFinalizationError("progress.log has no run_started event")
    attempt = events[start_indexes[-1] :]
    start = attempt[0]
    if start.get("mode") != "random_search":
        raise EvoRunFinalizationError(
            "Only completed random-search runs can be finalized from saved artifacts"
        )
    completed = [event for event in attempt if event.get("msg") == "search_completed"]
    if not completed:
        raise EvoRunFinalizationError("Latest attempt has no search_completed event")

    stop_payload = _read_json_object(run_dir / "ga_stop_details.json")
    final_stop = stop_payload.get("final_stop")
    if not isinstance(final_stop, dict):
        raise EvoRunFinalizationError("ga_stop_details.json has no final_stop object")
    if final_stop.get("reason") != "random_search_complete":
        raise EvoRunFinalizationError(
            "Run did not stop with reason 'random_search_complete'"
        )
    try:
        expected = int(start.get("max_fevals") or 0)
        fevals = int(final_stop.get("fevals") or 0)
        generation = int(final_stop.get("generation") or 0)
        best_score = float(final_stop.get("best_score"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvoRunFinalizationError(
            "Run completion metadata contains invalid numeric values"
        ) from exc
    row_count, scores = _read_candidate_scores(run_dir / "ga_candidate_history.csv")
    if expected <= 0 or fevals != expected or row_count != expected:
        raise EvoRunFinalizationError(
            "Random-search budget is incomplete: "
            f"expected={expected}, stop_fevals={fevals}, candidates={row_count}"
        )
    if not scores:
        raise EvoRunFinalizationError("Candidate history has no finite score")
    observed_best = max(scores)
    if not math.isclose(observed_best, best_score, rel_tol=1e-9, abs_tol=1e-9):
        raise EvoRunFinalizationError(
            f"Best-score mismatch: stop={best_score}, candidates={observed_best}"
        )
    return {
        "seed": int(start.get("seed") or 0),
        "fevals": fevals,
        "generation": generation,
        "best_score": best_score,
        "stop_details": final_stop,
    }


def finalize_evo_run(
    config_path: Path | str,
    run_dir: Path | str,
    *,
    batch_size: Optional[int] = None,
    device: str = "auto",
    trust_remote_code: bool = False,
    min_free_disk_gb: float = 5.0,
) -> EvoRunValidation:
    """Materialize and validate a winner without repeating a completed search."""
    config_path = Path(config_path).resolve()
    run_dir = Path(run_dir).resolve()
    try:
        return validate_evo_run(run_dir, config_path=config_path)
    except EvoRunValidationError:
        pass

    context = _completed_search_context(run_dir)
    try:
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = EvolMergeConfiguration.model_validate(config_payload)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise EvoRunFinalizationError(f"Unable to load run config: {exc}") from exc
    if config.repair is not None and config.repair.enabled:
        raise EvoRunFinalizationError(
            "Repair-enabled runs require the live strategy state and cannot be finalized "
            "from best_config.yaml alone"
        )

    best_config_path = run_dir / "best_config.yaml"
    try:
        config_source = best_config_path.read_text(encoding="utf-8")
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(config_source))
    except (OSError, yaml.YAMLError, ValueError, RuntimeError) as exc:
        raise EvoRunFinalizationError(
            f"Unable to load saved winner {best_config_path}: {exc}"
        ) from exc

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise EvoRunFinalizationError("CUDA was requested but is not available")
    merge_cuda = resolved_device == "cuda"

    write_ga_outputs(
        str(run_dir),
        stop_details=context["stop_details"],
    )
    ensure_free_disk(str(run_dir), min_free_disk_gb)
    final_model_path = run_dir / "final_model"
    shutil.rmtree(final_model_path, ignore_errors=True)
    merge_options = MergeOptions(
        transformers_cache=str(run_dir / "transformers_cache"),
        lora_merge_cache=str(run_dir / "lora_merge_cache"),
        cuda=merge_cuda,
        device=resolved_device,
        low_cpu_memory=merge_cuda,
        out_shard_size=1_000_000_000_000,
        trust_remote_code=trust_remote_code,
        quiet=True,
        read_to_gpu=merge_cuda,
        copy_tokenizer=True,
        safe_serialization=True,
        min_free_disk_gb=min_free_disk_gb,
        reuse_scratch_dir=True,
    )
    merge_options.apply_global_options()
    run_merge(
        merge_config,
        str(final_model_path),
        options=merge_options,
        config_source=config_source,
    )
    evaluate_and_write_final_comparison(
        config,
        str(run_dir),
        batch_size,
        merge_cuda,
        1 if merge_cuda else 0,
        [],
        trust_remote_code,
    )
    comparison_path = run_dir / "ga_final_comparison.csv"
    if not comparison_path.is_file():
        raise EvoRunFinalizationError(
            "Final model evaluation did not produce ga_final_comparison.csv"
        )

    ProgressLogger(str(run_dir), seed=context["seed"]).write(
        "run_finished",
        idx=context["fevals"],
        generation=context["generation"],
        scores={"best": context["best_score"]},
        status="success",
        recovered_after_search=True,
    )
    return validate_evo_run(run_dir, config_path=config_path)


@click.command("mergekit-finalize-evo-run")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--batch-size", type=int, default=None)
@click.option("--device", type=click.Choice(("auto", "cpu", "cuda")), default="auto")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False)
@click.option("--min-free-disk-gb", type=float, default=5.0, show_default=True)
def main(
    config_path: Path,
    run_dir: Path,
    batch_size: Optional[int],
    device: str,
    trust_remote_code: bool,
    min_free_disk_gb: float,
) -> None:
    try:
        result = finalize_evo_run(
            config_path,
            run_dir,
            batch_size=batch_size,
            device=device,
            trust_remote_code=trust_remote_code,
            min_free_disk_gb=min_free_disk_gb,
        )
    except (EvoRunFinalizationError, EvoRunValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"FINALIZED {result.run_dir} mode={result.mode} fevals={result.fevals} "
        f"candidates={result.candidate_rows} successful={result.successful_candidates}"
    )


if __name__ == "__main__":
    main()
