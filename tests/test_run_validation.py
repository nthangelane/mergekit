import json
from pathlib import Path

import pytest

from mergekit.evo.progress import ProgressLogger
from mergekit.evo.run_validation import EvoRunValidationError, validate_evo_run


def _write_csv(path: Path, rows: int) -> None:
    lines = ["generation,fevals,value,score"]
    lines.extend(f"1,{index + 1},{index},{0.5 + index}" for index in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_valid_run(
    run_dir: Path,
    *,
    mode: str = "ga",
    fevals: int = 2,
    candidate_rows: int = 2,
) -> None:
    progress = ProgressLogger(str(run_dir), seed=11)
    progress.write("run_started", mode=mode, max_fevals=fevals)
    progress.write("generation_completed", generation=1, idx=fevals)
    _write_csv(run_dir / "baseline_results.csv", 2)
    _write_csv(run_dir / "ga_method_history.csv", 2)
    _write_csv(run_dir / "ga_candidate_history.csv", candidate_rows)
    reason = "random_search_complete" if mode == "random_search" else "max_fevals"
    (run_dir / "ga_stop_details.json").write_text(
        json.dumps({"final_stop": {"reason": reason, "fevals": fevals}}) + "\n",
        encoding="utf-8",
    )
    progress.write("run_finished", status="success", idx=fevals)


def test_validate_evo_run_accepts_complete_ga_run(tmp_path):
    _write_valid_run(tmp_path)

    result = validate_evo_run(tmp_path)

    assert result.mode == "ga"
    assert result.fevals == 2
    assert result.candidate_rows == 2
    assert result.successful_candidates == 2


def test_validate_evo_run_rejects_false_success_marker(tmp_path):
    progress = ProgressLogger(str(tmp_path), seed=11)
    progress.write("run_started", mode="ga", max_fevals=192)

    with pytest.raises(EvoRunValidationError, match="does not end with run_finished"):
        validate_evo_run(tmp_path)


def test_validate_evo_run_requires_random_search_budget(tmp_path):
    _write_valid_run(
        tmp_path,
        mode="random_search",
        fevals=4,
        candidate_rows=3,
    )

    with pytest.raises(EvoRunValidationError, match="candidate count mismatch"):
        validate_evo_run(tmp_path)


def test_validate_evo_run_requires_repair_artifact(tmp_path):
    _write_valid_run(tmp_path)
    config_path = tmp_path / "repair.yml"
    config_path.write_text("repair:\n  enabled: true\n", encoding="utf-8")

    with pytest.raises(EvoRunValidationError, match="missing final_repair.json"):
        validate_evo_run(tmp_path, config_path=config_path)

    (tmp_path / "final_repair.json").write_text(
        '{"repaired": false, "probe_slope": 0.0}\n',
        encoding="utf-8",
    )
    result = validate_evo_run(tmp_path, config_path=config_path)
    assert result.repair_recorded is True


def test_validate_evo_run_rejects_all_failed_candidates(tmp_path):
    _write_valid_run(tmp_path)
    (tmp_path / "ga_candidate_history.csv").write_text(
        "generation,fevals,score,error_type\n"
        "1,1,,PydanticUserError\n"
        "1,2,,PydanticUserError\n",
        encoding="utf-8",
    )

    with pytest.raises(EvoRunValidationError, match="no finite candidate score"):
        validate_evo_run(tmp_path)


def test_validate_evo_run_reports_malformed_numeric_metadata(tmp_path):
    _write_valid_run(tmp_path)
    stop_path = tmp_path / "ga_stop_details.json"
    stop_path.write_text(
        json.dumps({"final_stop": {"reason": "max_fevals", "fevals": "many"}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvoRunValidationError, match="final stop fevals must be"):
        validate_evo_run(tmp_path)
