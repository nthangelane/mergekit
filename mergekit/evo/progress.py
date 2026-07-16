from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mergekit.evo.checkpoint import json_safe
from mergekit.evo.resources import free_disk_gb

PROGRESS_FILENAME = "progress.log"


class ProgressLogger:
    """Append-only JSON-lines progress log shared by local and cloud runs."""

    def __init__(self, run_dir: str, *, seed: int):
        self.run_dir = os.path.abspath(run_dir)
        self.path = os.path.join(self.run_dir, PROGRESS_FILENAME)
        self.seed = int(seed)
        os.makedirs(self.run_dir, exist_ok=True)

    def write(
        self,
        msg: str,
        *,
        idx: Optional[int] = None,
        generation: Optional[int] = None,
        scores: Optional[Dict[str, Any]] = None,
        secs: Optional[float] = None,
        compute: bool = False,
        **fields: Any,
    ) -> Dict[str, Any]:
        try:
            disk_free = free_disk_gb(self.run_dir)
        except (OSError, ValueError):
            disk_free = None
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": str(msg),
            "seed": self.seed,
            "disk_free_gb": disk_free,
        }
        if idx is not None:
            record["idx"] = int(idx)
        if generation is not None:
            record["generation"] = int(generation)
        if scores is not None:
            record["scores"] = scores
        if secs is not None:
            record["secs"] = float(secs)
        if compute:
            record["compute"] = True
        record.update(fields)
        record = json_safe(record)

        with open(self.path, "a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(record, sort_keys=True) + "\n")
            progress_file.flush()
            os.fsync(progress_file.fileno())
        return record


def iter_progress_events(path_or_run_dir: str) -> Iterable[Dict[str, Any]]:
    path = Path(path_or_run_dir)
    if path.is_dir():
        path = path / PROGRESS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Progress log not found: {path}")
    with path.open("r", encoding="utf-8") as progress_file:
        for line_number, line in enumerate(progress_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"Progress event at {path}:{line_number} is not an object"
                )
            yield event


def ga_compute_seconds(path_or_run_dir: str) -> float:
    """Return measured GA evaluation seconds from a progress log."""

    return float(
        sum(
            float(event.get("secs") or 0.0)
            for event in iter_progress_events(path_or_run_dir)
            if event.get("compute") is True
        )
    )
