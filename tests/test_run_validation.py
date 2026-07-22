import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mergekit.evo.progress import ProgressLogger
from mergekit.evo.run_validation import EvoRunValidationError, validate_evo_run
from mergekit.scripts.validate_evo_run import main as validate_evo_run_cli


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


def test_validate_evo_run_requires_audit_history_and_final_row(tmp_path):
    _write_valid_run(tmp_path)
    config_path = tmp_path / "audit.yml"
    config_path.write_text(
        "audit:\n  enabled: true\n  final_audit: true\n", encoding="utf-8"
    )

    with pytest.raises(EvoRunValidationError, match="missing ga_audit_history.csv"):
        validate_evo_run(tmp_path, config_path=config_path)

    (tmp_path / "ga_audit_history.csv").write_text(
        "generation,genotype_hash,search_score,audit_score,delta,seconds,final\n"
        "1,abc,0.5,0.4,-0.1,1.0,false\n"
        "1,abc,0.4,0.4,0.0,1.0,true\n",
        encoding="utf-8",
    )
    result = validate_evo_run(tmp_path, config_path=config_path)
    assert result.audit_rows == 2


def test_validate_evo_run_matches_reentry_log_to_checkpoints(tmp_path):
    _write_valid_run(tmp_path)
    checkpoint = tmp_path / "reentrant" / "abc"
    checkpoint.mkdir(parents=True)
    (tmp_path / "ga_reentry_history.csv").write_text(
        "generation,genotype_hash,checkpoint_path,pre_score,post_score,gain\n"
        f"1,abc,{checkpoint},0.4,0.45,0.05\n",
        encoding="utf-8",
    )

    result = validate_evo_run(tmp_path)
    assert result.reentry_rows == 1

    (tmp_path / "reentrant" / "unlogged").mkdir()
    with pytest.raises(EvoRunValidationError, match="do not match"):
        validate_evo_run(tmp_path)


def test_validate_evo_run_warns_with_invalid_genotype_count(tmp_path):
    _write_valid_run(tmp_path)
    (tmp_path / "ga_candidate_history.csv").write_text(
        "generation,fevals,score,error_type\n"
        "1,1,0.5,\n"
        "1,2,,invalid_genotype\n"
        "1,3,,invalid_genotype\n",
        encoding="utf-8",
    )

    result = validate_evo_run(tmp_path)

    assert result.invalid_genotype_count == 2
    assert result.warnings == (
        "ga_candidate_history.csv contains 2 invalid_genotype rejection(s)",
    )
    cli_result = CliRunner().invoke(validate_evo_run_cli, [str(tmp_path)])
    assert cli_result.exit_code == 0
    assert "invalid_genotypes=2" in cli_result.output
    assert "WARNING:" in cli_result.output
