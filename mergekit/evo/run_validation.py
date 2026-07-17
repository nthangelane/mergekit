"""Validation for completed evolutionary experiment runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mergekit.evo.progress import iter_progress_events


class EvoRunValidationError(ValueError):
    """Raised when a run was reported as complete without usable artifacts."""


@dataclass(frozen=True)
class EvoRunValidation:
    run_dir: Path
    mode: str
    candidate_rows: int
    successful_candidates: int
    method_rows: int
    fevals: int
    stop_reason: str
    repair_recorded: bool


def _read_csv_rows(path: Path, errors: list[str]) -> list[Dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        errors.append(f"invalid {path.name}: {exc}")
        return []
    if not reader.fieldnames:
        errors.append(f"{path.name} has no header")
    if not rows:
        errors.append(f"{path.name} has no data rows")
    return rows


def _read_json_object(path: Path, errors: list[str]) -> Dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return payload


def _read_int(value: Any, field: str, errors: list[str]) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        errors.append(f"{field} must be an integer")
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{field} must be an integer")
        return 0


def _repair_enabled(config_path: Optional[Path], errors: list[str]) -> bool:
    if config_path is None:
        return False
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"unable to read run config {config_path}: {exc}")
        return False
    repair = payload.get("repair") if isinstance(payload, dict) else None
    return isinstance(repair, dict) and repair.get("enabled") is True


def validate_evo_run(
    run_dir: Path | str,
    *,
    config_path: Optional[Path | str] = None,
    require_baseline: bool = True,
) -> EvoRunValidation:
    """Require a successful latest attempt and its thesis-critical artifacts."""
    run_dir = Path(run_dir).resolve()
    resolved_config = Path(config_path).resolve() if config_path is not None else None
    errors: list[str] = []

    try:
        events = list(iter_progress_events(str(run_dir)))
    except (FileNotFoundError, ValueError) as exc:
        events = []
        errors.append(str(exc))

    start_indexes = [
        index for index, event in enumerate(events) if event.get("msg") == "run_started"
    ]
    latest_start: Dict[str, Any] = {}
    attempt_events: list[Dict[str, Any]] = []
    if not start_indexes:
        errors.append("progress.log has no run_started event")
    else:
        latest_start_index = start_indexes[-1]
        latest_start = events[latest_start_index]
        attempt_events = events[latest_start_index:]
        terminal = attempt_events[-1]
        if terminal.get("msg") != "run_finished":
            errors.append("latest attempt does not end with run_finished")
        elif terminal.get("status") != "success":
            errors.append(
                "latest attempt finished with status "
                f"{terminal.get('status', 'missing')!r}"
            )
        if not any(
            event.get("msg") == "generation_completed" for event in attempt_events
        ):
            errors.append("latest attempt has no completed evaluation batch")

    mode = str(latest_start.get("mode") or "unknown")
    expected_fevals = _read_int(
        latest_start.get("max_fevals"), "progress max_fevals", errors
    )
    candidate_rows = _read_csv_rows(run_dir / "ga_candidate_history.csv", errors)
    method_rows = _read_csv_rows(run_dir / "ga_method_history.csv", errors)
    successful_candidates = 0
    for row in candidate_rows:
        try:
            if math.isfinite(float(row.get("score", ""))):
                successful_candidates += 1
        except (TypeError, ValueError):
            continue
    if candidate_rows and successful_candidates == 0:
        errors.append("ga_candidate_history.csv has no finite candidate score")
    if require_baseline:
        _read_csv_rows(run_dir / "baseline_results.csv", errors)

    stop_payload = _read_json_object(run_dir / "ga_stop_details.json", errors)
    final_stop = stop_payload.get("final_stop", {})
    if not isinstance(final_stop, dict):
        errors.append("ga_stop_details.json final_stop must be an object")
        final_stop = {}
    stop_reason = str(final_stop.get("reason") or "")
    fevals = _read_int(final_stop.get("fevals"), "final stop fevals", errors)
    if not stop_reason:
        errors.append("ga_stop_details.json has no final stop reason")
    if fevals <= 0:
        errors.append("ga_stop_details.json reports no completed evaluations")

    if mode == "random_search":
        if stop_reason != "random_search_complete":
            errors.append(
                "random search did not stop with reason 'random_search_complete'"
            )
        if expected_fevals <= 0:
            errors.append("random search progress has no positive max_fevals")
        elif len(candidate_rows) != expected_fevals:
            errors.append(
                "random search candidate count mismatch: "
                f"expected {expected_fevals}, found {len(candidate_rows)}"
            )
        if expected_fevals > 0 and fevals != expected_fevals:
            errors.append(
                f"random search stop fevals mismatch: expected {expected_fevals}, "
                f"found {fevals}"
            )

    repair_recorded = False
    if _repair_enabled(resolved_config, errors):
        repair_payload = _read_json_object(run_dir / "final_repair.json", errors)
        repair_recorded = "repaired" in repair_payload
        if repair_payload and not repair_recorded:
            errors.append("final_repair.json has no repaired outcome")

    if errors:
        raise EvoRunValidationError(
            f"Invalid evolutionary run at {run_dir}: " + "; ".join(errors)
        )

    return EvoRunValidation(
        run_dir=run_dir,
        mode=mode,
        candidate_rows=len(candidate_rows),
        successful_candidates=successful_candidates,
        method_rows=len(method_rows),
        fevals=fevals,
        stop_reason=stop_reason,
        repair_recorded=repair_recorded,
    )
