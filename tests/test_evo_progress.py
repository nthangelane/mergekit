import json

from mergekit.evo.progress import ProgressLogger, ga_compute_seconds


def test_progress_logger_emits_required_jsonl_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("MERGEKIT_FREE_DISK_GB_OVERRIDE", "12.5")
    progress = ProgressLogger(str(tmp_path), seed=11)

    progress.write(
        "generation_completed",
        idx=8,
        generation=1,
        scores={"best": 0.4},
        secs=3.25,
        compute=True,
    )

    event = json.loads((tmp_path / "progress.log").read_text(encoding="utf-8"))
    assert event["msg"] == "generation_completed"
    assert event["seed"] == 11
    assert event["idx"] == 8
    assert event["generation"] == 1
    assert event["scores"] == {"best": 0.4}
    assert event["secs"] == 3.25
    assert event["disk_free_gb"] == 12.5
    assert event["ts"]


def test_ga_compute_seconds_sums_only_compute_events(tmp_path):
    progress = ProgressLogger(str(tmp_path), seed=11)
    progress.write("run_started", secs=100.0)
    progress.write("generation_completed", secs=2.0, compute=True)
    progress.write("generation_completed", secs=3.5, compute=True)

    assert ga_compute_seconds(str(tmp_path)) == 5.5
