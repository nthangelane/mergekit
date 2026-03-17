import math
import os
from types import SimpleNamespace

import pandas
import pytest

from mergekit.evo.tracking import MLflowTracker
from mergekit.scripts.evolve_ga import (
    _best_weighted_score_from_frame,
    _collect_merge_method_outcomes,
    _init_ray_for_baselines,
    _meets_improvement_thresholds,
    _reusable_baseline_csv_path,
    _score_improvement,
    _unique_model_refs,
    _write_mlflow_run_info,
)


def test_best_weighted_score_from_frame_uses_max_normalized_score():
    frame = pandas.DataFrame(
        {"weighted_score": ["-64.27", -63.73, None, "not-a-number", float("nan")]}
    )

    assert _best_weighted_score_from_frame(frame) == -63.73


def test_score_improvement_uses_baseline_magnitude_for_negative_scores():
    delta, pct = _score_improvement(-63.73, -64.27)

    assert math.isclose(delta, 0.54)
    assert math.isclose(pct, 0.54 / 64.27 * 100.0)


def test_score_improvement_handles_positive_scores():
    delta, pct = _score_improvement(0.84, 0.80)

    assert math.isclose(delta, 0.04)
    assert math.isclose(pct, 5.0)


def test_score_improvement_returns_undefined_pct_for_zero_baseline():
    delta, pct = _score_improvement(0.25, 0.0)

    assert math.isclose(delta, 0.25)
    assert pct is None


def test_improvement_thresholds_accept_negative_baseline_improvement():
    delta, pct = _score_improvement(-63.73, -64.27)

    assert _meets_improvement_thresholds(delta, pct, min_abs=0.1, min_pct=0.5)


def test_improvement_thresholds_reject_undefined_pct_when_pct_required():
    delta, pct = _score_improvement(0.25, 0.0)

    assert not _meets_improvement_thresholds(delta, pct, min_abs=0.1, min_pct=1.0)


def test_reusable_baseline_csv_path_accepts_complete_existing_scores(tmp_path):
    csv_path = tmp_path / "baseline_results.csv"
    pandas.DataFrame(
        [
            {"model": "model-a", "weighted_score": 0.5},
            {"model": "model-b", "weighted_score": 0.4},
        ]
    ).to_csv(csv_path, index=False)

    assert _reusable_baseline_csv_path(str(csv_path), ["model-a", "model-b"]) == str(
        csv_path
    )


def test_reusable_baseline_csv_path_rejects_missing_scores(tmp_path):
    csv_path = tmp_path / "baseline_results.csv"
    pandas.DataFrame(
        [
            {"model": "model-a", "weighted_score": 0.5},
            {"model": "model-b", "weighted_score": None},
        ]
    ).to_csv(csv_path, index=False)

    assert _reusable_baseline_csv_path(str(csv_path), ["model-a", "model-b"]) is None


def test_init_ray_for_baselines_falls_back_to_local_runtime(monkeypatch):
    init_calls = []

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr("mergekit.scripts.evolve_ga.ray.is_initialized", lambda: False)

    def fake_init(*args, **kwargs):
        init_calls.append(kwargs)
        if kwargs.get("address") == "auto":
            raise ConnectionError("no ray instance")
        return None

    monkeypatch.setattr("mergekit.scripts.evolve_ga.ray.init", fake_init)
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga.stage_log", lambda *args, **kwargs: None
    )

    _init_ray_for_baselines()

    assert len(init_calls) == 2
    assert init_calls[0]["address"] == "auto"
    assert "address" not in init_calls[1]


def test_unique_model_refs_preserves_order_while_deduping():
    unique = _unique_model_refs(["model-a", "model-b", "model-a", "model-c"])

    assert unique == ["model-a", "model-b", "model-c"]


class _DummyGenome:
    def genotype_to_merge_config(self, genotype):
        return SimpleNamespace(merge_method=str(genotype))


def test_collect_merge_method_outcomes_computes_success_rates():
    outcomes = _collect_merge_method_outcomes(
        ["linear", "linear", "passthrough"],
        [
            {"score": 0.2},
            {"score": None},
            {"score": 0.5},
        ],
        _DummyGenome(),
        ["linear", "passthrough"],
    )

    assert outcomes["method_counts"]["linear"] == 2
    assert outcomes["method_success_counts"]["linear"] == 1
    assert outcomes["method_failure_counts"]["linear"] == 1
    assert math.isclose(outcomes["metrics"]["merge_method/linear/success_rate"], 0.5)
    assert math.isclose(
        outcomes["metrics"]["merge_method/passthrough/success_rate"], 1.0
    )


def test_write_mlflow_run_info_writes_review_url(tmp_path):
    tracker = SimpleNamespace(
        get_run_metadata=lambda: {
            "tracker_type": "mlflow",
            "experiment_id": "123",
            "run_id": "abc",
            "tracking_uri": "file://./mlruns",
            "local_store_path": str(tmp_path / "mlruns"),
            "run_url": "http://127.0.0.1:5000/#/experiments/123/runs/abc",
        }
    )

    info_path = _write_mlflow_run_info(
        str(tmp_path), tracker, project_name="mergekit-evolve-ga"
    )

    assert info_path is not None
    content = (tmp_path / "mlflow_run_info.md").read_text(encoding="utf-8")
    assert "MLflow Review URL" in content
    assert "http://127.0.0.1:5000/#/experiments/123/runs/abc" in content


def test_mlflow_tracker_prefers_ui_url_env(monkeypatch):
    tracker = MLflowTracker()
    tracker.tracking_uri = "file:///tmp/mlruns"
    tracker.experiment_name = "demo"
    tracker.experiment_id = "42"
    tracker.run_id = "run-123"

    monkeypatch.setenv("MLFLOW_UI_URL", "http://127.0.0.1:5001")
    metadata = tracker.get_run_metadata()

    assert metadata["run_url"] == "http://127.0.0.1:5001/#/experiments/42/runs/run-123"
