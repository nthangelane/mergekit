"""Helpers for exposing GA runtime state through Ray metrics and workspace files."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ray.util import metrics as ray_metrics
except Exception:  # pragma: no cover - optional dependency
    ray_metrics = None

LOG = logging.getLogger(__name__)

_PHASE_CODES = {
    "failed": -1.0,
    "init": 0.0,
    "baseline": 1.0,
    "tracking": 2.0,
    "reshard": 3.0,
    "ga": 4.0,
    "final_compare": 5.0,
    "finished": 6.0,
}

_ACTOR_STAGE_CODES = {
    "failed": -1.0,
    "idle": 0.0,
    "merge": 1.0,
    "evaluate": 2.0,
    "stage1": 3.0,
    "stage2": 4.0,
    "baseline": 5.0,
}


def sanitize_observability_label(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_")
    return cleaned[:120] or "unknown"


def default_run_label(storage_path: Optional[str], strategy: str) -> str:
    base = Path(storage_path).name if storage_path else "mergekit"
    return sanitize_observability_label(f"{base}-{strategy}")


def worker_actor_name(run_label: str, strategy: str, index: int) -> str:
    return sanitize_observability_label(
        f"mergekit-{run_label}-{strategy}-worker-{index:02d}"
    )


class RayRunObserver:
    """Emit lightweight custom metrics for Ray UI and persist a live snapshot."""

    def __init__(
        self,
        *,
        run_label: str,
        strategy: str,
        role: str,
        storage_path: Optional[str] = None,
        actor_name: Optional[str] = None,
        snapshot_enabled: bool = False,
    ):
        self.run_label = sanitize_observability_label(run_label)
        self.strategy = sanitize_observability_label(strategy)
        self.role = sanitize_observability_label(role)
        self.actor_name = (
            sanitize_observability_label(actor_name) if actor_name else None
        )
        self.snapshot_enabled = bool(snapshot_enabled and storage_path)
        self.snapshot_path = (
            Path(storage_path) / "ray_observability.json"
            if self.snapshot_enabled and storage_path
            else None
        )
        self._state: Dict[str, Any] = {
            "run_label": self.run_label,
            "strategy": self.strategy,
            "role": self.role,
            "actor_name": self.actor_name,
            "phase": "init",
            "updated_at": time.time(),
        }

        self._metrics_available = ray_metrics is not None
        if self._metrics_available:
            common_tags = ("run", "strategy", "role")
            actor_tags = ("run", "strategy", "role", "actor", "stage")
            candidate_tags = ("run", "strategy", "role", "phase", "method")

            self._phase_gauge = ray_metrics.Gauge(
                "mergekit_phase_code",
                description="Current high-level phase of the mergekit GA run.",
                tag_keys=common_tags,
            )
            self._generation_gauge = ray_metrics.Gauge(
                "mergekit_generation",
                description="Current generation number for the mergekit GA run.",
                tag_keys=common_tags,
            )
            self._fevals_gauge = ray_metrics.Gauge(
                "mergekit_fevals_completed",
                description="Completed function evaluations for the mergekit GA run.",
                tag_keys=common_tags,
            )
            self._best_score_gauge = ray_metrics.Gauge(
                "mergekit_best_score",
                description="Best score seen so far in the mergekit GA run.",
                tag_keys=common_tags,
            )
            self._generation_best_gauge = ray_metrics.Gauge(
                "mergekit_generation_best_score",
                description="Best score in the most recent GA generation.",
                tag_keys=common_tags,
            )
            self._generation_mean_gauge = ray_metrics.Gauge(
                "mergekit_generation_mean_score",
                description="Mean score in the most recent GA generation.",
                tag_keys=common_tags,
            )
            self._cache_hits_gauge = ray_metrics.Gauge(
                "mergekit_cache_hits",
                description="Cache hits accumulated in the latest GA generation.",
                tag_keys=common_tags,
            )
            self._failures_gauge = ray_metrics.Gauge(
                "mergekit_failed_evals",
                description="Failed evaluations in the latest GA generation.",
                tag_keys=common_tags,
            )
            self._candidate_index_gauge = ray_metrics.Gauge(
                "mergekit_candidate_index",
                description="Currently active candidate index for this run.",
                tag_keys=common_tags,
            )
            self._baseline_index_gauge = ray_metrics.Gauge(
                "mergekit_baseline_model_index",
                description="Baseline model currently being evaluated.",
                tag_keys=common_tags,
            )
            self._baseline_total_gauge = ray_metrics.Gauge(
                "mergekit_baseline_model_total",
                description="Total number of baseline models for this run.",
                tag_keys=common_tags,
            )
            self._mlflow_ui_gauge = ray_metrics.Gauge(
                "mergekit_mlflow_ui_up",
                description="Whether an MLflow UI endpoint is available for this run.",
                tag_keys=common_tags,
            )
            self._actor_active_gauge = ray_metrics.Gauge(
                "mergekit_actor_active",
                description="Whether a named mergekit worker actor is actively processing work.",
                tag_keys=actor_tags,
            )
            self._actor_generation_gauge = ray_metrics.Gauge(
                "mergekit_actor_generation",
                description="Generation currently assigned to a named mergekit worker actor.",
                tag_keys=actor_tags,
            )
            self._actor_candidate_gauge = ray_metrics.Gauge(
                "mergekit_actor_candidate_index",
                description="Candidate index currently assigned to a named mergekit worker actor.",
                tag_keys=actor_tags,
            )
            self._actor_score_gauge = ray_metrics.Gauge(
                "mergekit_actor_last_score",
                description="Most recent score produced by a named mergekit worker actor.",
                tag_keys=actor_tags,
            )
            self._actor_error_gauge = ray_metrics.Gauge(
                "mergekit_actor_error",
                description="Whether the most recent actor action ended in an error.",
                tag_keys=actor_tags,
            )
            self._candidate_started_counter = ray_metrics.Counter(
                "mergekit_candidate_started_total",
                description="Count of candidate evaluations started by the mergekit GA run.",
                tag_keys=candidate_tags,
            )
            self._candidate_completed_counter = ray_metrics.Counter(
                "mergekit_candidate_completed_total",
                description="Count of candidate evaluations completed by the mergekit GA run.",
                tag_keys=candidate_tags,
            )
            self._candidate_failed_counter = ray_metrics.Counter(
                "mergekit_candidate_failed_total",
                description="Count of candidate evaluations that failed in the mergekit GA run.",
                tag_keys=candidate_tags,
            )
            self._generation_completed_counter = ray_metrics.Counter(
                "mergekit_generation_completed_total",
                description="Count of completed generations in the mergekit GA run.",
                tag_keys=("run", "strategy", "role", "phase"),
            )
            self._baseline_completed_counter = ray_metrics.Counter(
                "mergekit_baseline_completed_total",
                description="Count of completed baseline model evaluations.",
                tag_keys=("run", "strategy", "role", "phase"),
            )

        self._write_snapshot()

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]],
        *,
        actor_name: Optional[str] = None,
        role: Optional[str] = None,
        snapshot_enabled: bool = False,
    ) -> Optional["RayRunObserver"]:
        if not config:
            return None
        return cls(
            run_label=str(config.get("run_label") or "mergekit"),
            strategy=str(config.get("strategy") or "unknown"),
            role=str(role or config.get("role") or "worker"),
            storage_path=config.get("storage_path"),
            actor_name=actor_name or config.get("actor_name"),
            snapshot_enabled=snapshot_enabled or bool(config.get("snapshot_enabled")),
        )

    def export_config(self, *, role: str = "worker") -> Dict[str, Any]:
        return {
            "run_label": self.run_label,
            "strategy": self.strategy,
            "role": role,
            "storage_path": None,
            "snapshot_enabled": False,
        }

    def _common_tags(self) -> Dict[str, str]:
        return {
            "run": self.run_label,
            "strategy": self.strategy,
            "role": self.role,
        }

    def _record_gauge(
        self, gauge, value: Optional[float], tags: Dict[str, str]
    ) -> None:
        if (
            not self._metrics_available
            or gauge is None
            or value is None
            or not math.isfinite(float(value))
        ):
            return
        try:
            gauge.record(float(value), tags=tags)
        except Exception as exc:  # pragma: no cover - runtime telemetry only
            LOG.debug("Failed to record Ray gauge %s", gauge, exc_info=exc)

    def _increment_counter(
        self, counter, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        if not self._metrics_available or counter is None or value <= 0:
            return
        try:
            counter.inc(float(value), tags=tags or self._common_tags())
        except Exception as exc:  # pragma: no cover - runtime telemetry only
            LOG.debug("Failed to increment Ray counter %s", counter, exc_info=exc)

    def _write_snapshot(self) -> None:
        if not self.snapshot_path:
            return
        try:
            payload = dict(self._state)
            payload["updated_at"] = time.time()
            self.snapshot_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - diagnostic helper only
            LOG.debug("Failed to write Ray observability snapshot", exc_info=exc)

    def set_mlflow_ui(self, available: bool) -> None:
        self._state["mlflow_ui_available"] = bool(available)
        self._record_gauge(
            self._mlflow_ui_gauge,
            1.0 if available else 0.0,
            self._common_tags(),
        )
        self._write_snapshot()

    def set_phase(self, phase: str, **fields: Any) -> None:
        phase_name = sanitize_observability_label(phase)
        self._state["phase"] = phase_name
        self._state.update(fields)
        tags = self._common_tags()
        self._record_gauge(
            self._phase_gauge,
            _PHASE_CODES.get(phase_name, _PHASE_CODES["init"]),
            tags,
        )
        self._record_gauge(self._generation_gauge, fields.get("generation"), tags)
        self._record_gauge(self._fevals_gauge, fields.get("fevals_completed"), tags)
        self._record_gauge(self._best_score_gauge, fields.get("best_score"), tags)
        self._record_gauge(
            self._generation_best_gauge, fields.get("generation_best"), tags
        )
        self._record_gauge(
            self._generation_mean_gauge, fields.get("generation_mean"), tags
        )
        self._record_gauge(self._cache_hits_gauge, fields.get("cache_hits"), tags)
        self._record_gauge(self._failures_gauge, fields.get("failed_evals"), tags)
        self._record_gauge(
            self._candidate_index_gauge, fields.get("candidate_index"), tags
        )
        self._record_gauge(
            self._baseline_index_gauge, fields.get("baseline_model_index"), tags
        )
        self._record_gauge(
            self._baseline_total_gauge, fields.get("baseline_model_total"), tags
        )
        self._write_snapshot()

    def record_generation_start(
        self,
        *,
        generation: int,
        fevals_completed: int,
        fevals_limit: int,
        population_size: int,
        best_score: float,
    ) -> None:
        self.set_phase(
            "ga",
            generation=generation,
            fevals_completed=fevals_completed,
            fevals_limit=fevals_limit,
            population_size=population_size,
            best_score=best_score if math.isfinite(best_score) else None,
            candidate_index=0,
        )

    def record_generation_end(
        self,
        *,
        generation: int,
        fevals_completed: int,
        generation_best: Optional[float],
        generation_mean: Optional[float],
        best_score: Optional[float],
        cache_hits: int,
        failed_evals: int,
    ) -> None:
        self.set_phase(
            "ga",
            generation=generation,
            fevals_completed=fevals_completed,
            generation_best=generation_best,
            generation_mean=generation_mean,
            best_score=best_score,
            cache_hits=cache_hits,
            failed_evals=failed_evals,
        )
        self._increment_counter(
            self._generation_completed_counter,
            tags={**self._common_tags(), "phase": "ga"},
        )

    def record_baseline_progress(
        self,
        *,
        model_index: int,
        model_total: int,
        model_name: str,
        score: Optional[float] = None,
        failed: bool = False,
    ) -> None:
        self.set_phase(
            "baseline",
            baseline_model_index=model_index,
            baseline_model_total=model_total,
            baseline_model_name=model_name,
            best_score=score,
            failed_evals=1 if failed else 0,
        )
        self._increment_counter(
            self._baseline_completed_counter,
            tags={**self._common_tags(), "phase": "baseline"},
        )

    def record_candidate_start(
        self,
        *,
        phase: str,
        candidate_index: int,
        generation: Optional[int],
        method: Optional[str],
    ) -> None:
        method_label = sanitize_observability_label(method)
        phase_label = sanitize_observability_label(phase)
        self.set_phase(
            phase_label,
            generation=generation,
            candidate_index=candidate_index,
        )
        self._increment_counter(
            self._candidate_started_counter,
            tags={
                **self._common_tags(),
                "phase": phase_label,
                "method": method_label,
            },
        )

    def record_candidate_end(
        self,
        *,
        phase: str,
        candidate_index: int,
        generation: Optional[int],
        method: Optional[str],
        score: Optional[float],
        failed: bool,
    ) -> None:
        method_label = sanitize_observability_label(method)
        phase_label = sanitize_observability_label(phase)
        self.set_phase(
            phase_label,
            generation=generation,
            candidate_index=candidate_index,
            best_score=score,
            failed_evals=1 if failed else 0,
        )
        tags = {
            **self._common_tags(),
            "phase": phase_label,
            "method": method_label,
        }
        self._increment_counter(self._candidate_completed_counter, tags=tags)
        if failed:
            self._increment_counter(self._candidate_failed_counter, tags=tags)

    def record_actor_status(
        self,
        *,
        stage: str,
        active: bool,
        generation: Optional[int] = None,
        candidate_index: Optional[int] = None,
        score: Optional[float] = None,
        error_type: Optional[str] = None,
    ) -> None:
        if not self.actor_name:
            return
        stage_label = sanitize_observability_label(stage)
        tags = {
            **self._common_tags(),
            "actor": self.actor_name,
            "stage": stage_label,
        }
        self._state.update(
            {
                "phase": stage_label,
                "generation": generation,
                "candidate_index": candidate_index,
                "last_score": score,
                "last_error_type": error_type,
                "active": bool(active),
            }
        )
        self._record_gauge(self._actor_active_gauge, 1.0 if active else 0.0, tags)
        self._record_gauge(self._actor_generation_gauge, generation, tags)
        self._record_gauge(self._actor_candidate_gauge, candidate_index, tags)
        self._record_gauge(self._actor_score_gauge, score, tags)
        self._record_gauge(
            self._actor_error_gauge,
            1.0 if error_type else 0.0,
            tags,
        )

    def current_state(self) -> Dict[str, Any]:
        return dict(self._state)
