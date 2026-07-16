import os
from pathlib import Path

import yaml

from mergekit.evo.progress import ProgressLogger
from mergekit.scripts.run_campaign import build_job, repository_root, validate_preset

PRESETS = [
    "exp26_aos_on",
    "exp26_aos_off",
    "exp27_native",
    "exp28_adaptive",
    "exp28_random",
    "exp29_probe",
    "exp29_full",
    "exp210_lora",
    "exp210_full_ft",
]


def _config_path(name: str) -> Path:
    return repository_root() / "experiments" / "configs" / f"{name}.yaml"


def test_all_campaign_presets_validate_and_pin_protocol():
    for preset in PRESETS:
        payload, metadata = validate_preset(_config_path(preset))
        assert metadata["seeds"]
        assert metadata["budget_fevals"] == 192
        assert metadata["two_stage_limits"] == [2, 3, 6]
        assert payload["provenance"] == "warn"


def test_exp26_aos_arms_only_change_adaptive_sampling():
    with _config_path("exp26_aos_on").open("r", encoding="utf-8") as config_file:
        enabled = yaml.safe_load(config_file)
    with _config_path("exp26_aos_off").open("r", encoding="utf-8") as config_file:
        disabled = yaml.safe_load(config_file)

    assert enabled["ga"]["adaptive_method_sampling"] is True
    assert disabled["ga"]["adaptive_method_sampling"] is False
    enabled["ga"]["adaptive_method_sampling"] = False
    assert enabled == disabled


def test_campaign_rerun_resumes_existing_ga_checkpoint(tmp_path):
    run_dir = tmp_path / "exp26_aos_off" / "seed-11"
    run_dir.mkdir(parents=True)
    (run_dir / "ga_state.json").write_text("{}\n", encoding="utf-8")

    job = build_job(
        "exp26_aos_off",
        _config_path("exp26_aos_off"),
        tmp_path,
        seed=11,
        python_executable="python",
    )

    assert job.resume is True
    assert "--resume" in job.command
    assert "--storage-path" not in job.command


def test_campaign_skips_successfully_finished_run(tmp_path):
    run_dir = tmp_path / "exp26_aos_off" / "seed-11"
    progress = ProgressLogger(str(run_dir), seed=11)
    progress.write("run_finished", status="success")

    job = build_job(
        "exp26_aos_off",
        _config_path("exp26_aos_off"),
        tmp_path,
        seed=11,
        python_executable="python",
    )

    assert job.skip is True
    assert job.command == []


def test_campaign_shell_is_executable():
    shell_path = repository_root() / "experiments" / "run_campaign.sh"

    assert os.access(shell_path, os.X_OK)
