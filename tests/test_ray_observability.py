import json

from mergekit.evo.ray_observability import (
    RayRunObserver,
    default_run_label,
    sanitize_observability_label,
    worker_actor_name,
)


def test_run_label_and_worker_names_are_sanitized():
    run_label = default_run_label("/tmp/my run/with spaces", "pool")

    assert run_label == "with_spaces-pool"
    assert sanitize_observability_label("hello world!") == "hello_world"
    assert (
        worker_actor_name("demo run", "pool", 3) == "mergekit-demo_run-pool-worker-03"
    )


def test_ray_run_observer_writes_live_snapshot(tmp_path):
    observer = RayRunObserver(
        run_label="demo-run",
        strategy="pool",
        role="driver",
        storage_path=str(tmp_path),
        snapshot_enabled=True,
    )

    observer.set_mlflow_ui(True)
    observer.record_generation_start(
        generation=2,
        fevals_completed=8,
        fevals_limit=32,
        population_size=4,
        best_score=0.61,
    )
    observer.record_candidate_start(
        phase="stage1",
        candidate_index=1,
        generation=2,
        method="linear",
    )
    observer.record_candidate_end(
        phase="stage1",
        candidate_index=1,
        generation=2,
        method="linear",
        score=0.57,
        failed=False,
    )
    observer.record_generation_end(
        generation=2,
        fevals_completed=12,
        generation_best=0.62,
        generation_mean=0.55,
        best_score=0.62,
        cache_hits=1,
        failed_evals=0,
    )

    snapshot = json.loads(
        (tmp_path / "ray_observability.json").read_text(encoding="utf-8")
    )
    assert snapshot["run_label"] == "demo-run"
    assert snapshot["strategy"] == "pool"
    assert snapshot["phase"] == "ga"
    assert snapshot["generation"] == 2
    assert snapshot["fevals_completed"] == 12
    assert snapshot["best_score"] == 0.62
    assert snapshot["mlflow_ui_available"] is True


def test_observer_export_config_round_trips_worker_instance(tmp_path):
    observer = RayRunObserver(
        run_label="demo-run",
        strategy="pool",
        role="driver",
        storage_path=str(tmp_path),
        snapshot_enabled=True,
    )

    worker = RayRunObserver.from_config(
        observer.export_config(role="worker"),
        actor_name="worker-01",
        role="worker",
    )

    assert worker is not None
    assert worker.run_label == "demo-run"
    assert worker.strategy == "pool"
    assert worker.role == "worker"
    assert worker.actor_name == "worker-01"
