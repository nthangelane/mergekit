import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas
import pytest

from mergekit.evo.tracking import MLflowTracker
from mergekit.scripts.evolve_ga import (
    _best_weighted_score_from_frame,
    _classify_solution_novelty,
    _collect_merge_method_outcomes,
    _has_valid_solution,
    _init_ray_for_baselines,
    _meets_improvement_thresholds,
    _mlflow_ui_responds,
    _reusable_baseline_csv_path,
    _score_improvement,
    _start_mlflow_ui_if_needed,
    _unique_model_refs,
    _write_mlflow_run_info,
)


def test_valid_solution_requires_finite_score():
    assert _has_valid_solution(object(), 0.5)
    assert not _has_valid_solution(None, 0.5)
    assert not _has_valid_solution(object(), None)
    assert not _has_valid_solution(object(), float("nan"))
    assert not _has_valid_solution(object(), float("-inf"))


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


def test_reusable_baseline_csv_path_rejects_different_fitness_definition(tmp_path):
    csv_path = tmp_path / "baseline_results.csv"
    pandas.DataFrame(
        [
            {
                "model": "model-a",
                "weighted_score": 0.5,
                "fitness_version": "v1",
                "lower_is_better_transform": "legacy_reciprocal",
            }
        ]
    ).to_csv(csv_path, index=False)

    assert (
        _reusable_baseline_csv_path(
            str(csv_path),
            ["model-a"],
            expected_fitness_version="v2",
            expected_lower_is_better_transform="log_reciprocal",
        )
        is None
    )


def test_init_ray_for_baselines_falls_back_to_local_runtime(monkeypatch):
    init_calls = []

    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    def fake_init(*args, **kwargs):
        init_calls.append(kwargs)
        if kwargs.get("address") == "auto":
            raise ConnectionError("no ray instance")
        return None

    fake_ray = SimpleNamespace(is_initialized=lambda: False, init=fake_init)
    monkeypatch.setattr("mergekit.scripts.evolve_ga._require_ray", lambda: fake_ray)
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga.stage_log", lambda *args, **kwargs: None
    )

    _init_ray_for_baselines()

    assert len(init_calls) == 2
    assert init_calls[0]["address"] == "auto"
    assert "address" not in init_calls[1]


def test_evolve_ga_import_does_not_require_ray():
    code = """
import builtins
original_import = builtins.__import__
def import_without_ray(name, *args, **kwargs):
    if name == 'ray' or name.startswith('ray.'):
        raise ImportError('ray intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_ray
import mergekit.scripts.evolve_ga
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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


def test_classify_solution_novelty_treats_passthrough_as_control():
    novelty = _classify_solution_novelty("passthrough")

    assert novelty["is_novel_solution"] is False
    assert novelty["novelty_class"] == "baseline_control"


def test_classify_solution_novelty_accepts_non_passthrough_merge():
    novelty = _classify_solution_novelty("linear")

    assert novelty["is_novel_solution"] is True
    assert novelty["novelty_class"] == "candidate_merge"


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


def test_write_mlflow_run_info_includes_ui_details(tmp_path):
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

    _write_mlflow_run_info(
        str(tmp_path),
        tracker,
        project_name="mergekit-evolve-ga",
        mlflow_ui_info={
            "ui_url": "http://127.0.0.1:5001",
            "pid": 4321,
            "log_path": str(tmp_path / "mlflow_ui.log"),
        },
    )

    content = (tmp_path / "mlflow_run_info.md").read_text(encoding="utf-8")
    assert "MLflow UI URL" in content
    assert "http://127.0.0.1:5001" in content
    assert "4321" in content
    assert "mlflow_ui.log" in content


def test_start_mlflow_ui_if_needed_reuses_existing_port(monkeypatch, tmp_path):
    tracker = SimpleNamespace(
        get_run_metadata=lambda: {
            "tracker_type": "mlflow",
            "local_store_path": str(tmp_path / "mlruns"),
        }
    )

    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._is_local_port_open", lambda host, port: True
    )
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._mlflow_ui_responds", lambda url: True
    )

    info = _start_mlflow_ui_if_needed(
        str(tmp_path),
        tracker,
        enabled=True,
        port=5007,
    )

    assert info == {"ui_url": "http://127.0.0.1:5007", "started": False}
    assert os.environ["MLFLOW_UI_URL"] == "http://127.0.0.1:5007"
    os.environ.pop("MLFLOW_UI_URL", None)


def test_start_mlflow_ui_if_needed_starts_subprocess(monkeypatch, tmp_path):
    tracker = SimpleNamespace(
        get_run_metadata=lambda: {
            "tracker_type": "mlflow",
            "local_store_path": str(tmp_path / "mlruns"),
        }
    )
    popen_calls = []

    class DummyProcess:
        pid = 24680

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return DummyProcess()

    monkeypatch.delenv("MLFLOW_UI_URL", raising=False)
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._is_local_port_open", lambda host, port: False
    )
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._mlflow_ui_responds", lambda url: False
    )
    monkeypatch.setattr("mergekit.scripts.evolve_ga.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga.stage_log", lambda *args, **kwargs: None
    )

    info = _start_mlflow_ui_if_needed(
        str(tmp_path),
        tracker,
        enabled=True,
        port=5011,
    )

    assert info is not None
    assert info["ui_url"] == "http://127.0.0.1:5011"
    assert info["pid"] == 24680
    assert info["started"] is True
    assert popen_calls
    assert "--port" in popen_calls[0][0]
    assert popen_calls[0][1]["env"]["GUNICORN_CMD_ARGS"] == "--workers=1 --timeout 120"
    assert (tmp_path / "mlflow_ui.pid").read_text(encoding="utf-8").strip() == "24680"
    os.environ.pop("MLFLOW_UI_URL", None)


def test_start_mlflow_ui_if_needed_does_not_reuse_unhealthy_port(monkeypatch, tmp_path):
    tracker = SimpleNamespace(
        get_run_metadata=lambda: {
            "tracker_type": "mlflow",
            "local_store_path": str(tmp_path / "mlruns"),
        }
    )
    messages = []

    monkeypatch.delenv("MLFLOW_UI_URL", raising=False)
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._is_local_port_open", lambda host, port: True
    )
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga._mlflow_ui_responds", lambda url: False
    )
    monkeypatch.setattr(
        "mergekit.scripts.evolve_ga.stage_log",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    info = _start_mlflow_ui_if_needed(
        str(tmp_path),
        tracker,
        enabled=True,
        port=5008,
    )

    assert info is None
    assert messages


def test_mlflow_ui_responds_returns_false_on_error(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr("mergekit.scripts.evolve_ga.urlopen", fake_urlopen)

    assert _mlflow_ui_responds("http://127.0.0.1:1") is False


def test_mlflow_tracker_prefers_ui_url_env(monkeypatch):
    tracker = MLflowTracker()
    tracker.tracking_uri = "file:///tmp/mlruns"
    tracker.experiment_name = "demo"
    tracker.experiment_id = "42"
    tracker.run_id = "run-123"

    monkeypatch.setenv("MLFLOW_UI_URL", "http://127.0.0.1:5001")
    metadata = tracker.get_run_metadata()

    assert metadata["run_url"] == "http://127.0.0.1:5001/#/experiments/42/runs/run-123"
