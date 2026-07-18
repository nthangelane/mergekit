import json
import os
from pathlib import Path

import yaml

from mergekit.evo.config import EvolMergeConfiguration
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


def _local_m1_config_path(name: str) -> Path:
    return (
        repository_root()
        / "experiments"
        / "thesis"
        / "local_mac"
        / "pending_experiments"
        / name
    )


def _write_valid_ga_run(run_dir: Path) -> None:
    progress = ProgressLogger(str(run_dir), seed=11)
    progress.write("run_started", mode="ga", max_fevals=192)
    progress.write("generation_completed", generation=1, idx=2)
    csv_payload = "generation,fevals,value,score\n1,2,1,0.5\n"
    for filename in (
        "baseline_results.csv",
        "ga_candidate_history.csv",
        "ga_method_history.csv",
    ):
        (run_dir / filename).write_text(csv_payload, encoding="utf-8")
    (run_dir / "ga_stop_details.json").write_text(
        json.dumps({"final_stop": {"reason": "max_fevals", "fevals": 2}}) + "\n",
        encoding="utf-8",
    )
    progress.write("run_finished", status="success", idx=2)


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


def test_local_m1_exp26_arms_only_change_adaptive_sampling():
    with _local_m1_config_path("thesis_aos_current.yml").open(
        "r", encoding="utf-8"
    ) as config_file:
        enabled = EvolMergeConfiguration.model_validate(yaml.safe_load(config_file))
    with _local_m1_config_path("thesis_aos_off.yml").open(
        "r", encoding="utf-8"
    ) as config_file:
        disabled = EvolMergeConfiguration.model_validate(yaml.safe_load(config_file))

    assert enabled.ga.adaptive_method_sampling is True
    assert disabled.ga.adaptive_method_sampling is False
    enabled_payload = enabled.model_dump(mode="json")
    enabled_payload["ga"]["adaptive_method_sampling"] = False
    assert enabled_payload == disabled.model_dump(mode="json")


def test_local_m1_configs_use_enhanced_v2_protocol():
    for name in (
        "thesis_aos_current.yml",
        "thesis_aos_off.yml",
        "exp28_connected_adaptive.yml",
        "exp29_repair_probe.yml",
    ):
        with _local_m1_config_path(name).open("r", encoding="utf-8") as config_file:
            config = EvolMergeConfiguration.model_validate(yaml.safe_load(config_file))

        assert config.optimizer == "enhanced"
        assert config.fitness.version == "v2"
        assert config.fitness.lower_is_better_transform == "log_reciprocal"


def test_local_m1_launcher_requires_valid_artifacts_before_done_marker():
    launcher_path = _local_m1_config_path("run_thesis_campaign.sh")
    launcher = launcher_path.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in launcher
    assert "mergekit.scripts.validate_evo_run" in launcher
    assert "mergekit.scripts.finalize_evo_run" in launcher
    launcher_lines = launcher.splitlines()
    validation_lines = [
        index for index, line in enumerate(launcher_lines) if "validate_run " in line
    ]
    done_lines = [
        index
        for index, line in enumerate(launcher_lines)
        if 'date -Iseconds > "$DIR/DONE"' in line
    ]
    assert done_lines
    for done_line in done_lines:
        preceding_validation = max(
            index for index in validation_lines if index < done_line
        )
        assert done_line - preceding_validation <= 20
    assert launcher.index("finalize completed search") < launcher.index(
        "archive incomplete"
    )
    assert "restore validated $NAME seed $SEED" in launcher
    assert launcher.index("restore validated $NAME seed $SEED") < launcher.index(
        "archive incomplete"
    )
    assert os.access(launcher_path, os.X_OK)


def test_local_m1_launchers_run_from_repo_and_reuse_hf_cache():
    for name in ("run_thesis_campaign.sh", "run_exp26.sh"):
        launcher = _local_m1_config_path(name).read_text(encoding="utf-8")

        assert 'cd "$REPO"' in launcher
        assert 'SHARED_HF_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"' in (
            launcher
        )
        assert 'ln -s "$SHARED_HF_CACHE" "$DIR/transformers_cache"' in launcher


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
    _write_valid_ga_run(run_dir)

    job = build_job(
        "exp26_aos_off",
        _config_path("exp26_aos_off"),
        tmp_path,
        seed=11,
        python_executable="python",
    )

    assert job.skip is True
    assert job.command == []


def test_campaign_does_not_skip_progress_only_run(tmp_path):
    run_dir = tmp_path / "exp26_aos_off" / "seed-11"
    progress = ProgressLogger(str(run_dir), seed=11)
    progress.write("run_started", mode="ga", max_fevals=192)
    progress.write("run_finished", status="success")

    job = build_job(
        "exp26_aos_off",
        _config_path("exp26_aos_off"),
        tmp_path,
        seed=11,
        python_executable="python",
    )

    assert job.skip is False
    assert job.command


def test_campaign_shell_is_executable():
    shell_path = repository_root() / "experiments" / "run_campaign.sh"

    assert os.access(shell_path, os.X_OK)
