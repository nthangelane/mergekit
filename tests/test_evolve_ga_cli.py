import yaml
from click.testing import CliRunner

import mergekit.scripts.evolve_ga as evolve_ga
from mergekit.evo.config import EvolMergeConfiguration


class _StopAfterConfigCheck(RuntimeError):
    pass


class _StopAfterBaselineHook(RuntimeError):
    pass


def test_limit_cli_override_updates_resolved_config(monkeypatch, tmp_path):
    config_path = tmp_path / "evolve.yml"
    storage_path = tmp_path / "storage"
    config_path.write_text(
        yaml.safe_dump(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext"],
                "limit": 16,
            }
        ),
        encoding="utf-8",
    )

    def fake_check_for_naughty_config(config, allow):
        assert config.limit == 7
        raise _StopAfterConfigCheck("stop after config validation")

    monkeypatch.setattr(
        evolve_ga, "check_for_naughty_config", fake_check_for_naughty_config
    )

    runner = CliRunner()
    result = runner.invoke(
        evolve_ga.main,
        [
            str(config_path),
            "--storage-path",
            str(storage_path),
            "--no-baseline",
            "--no-reshard",
            "--limit",
            "7",
        ],
    )

    assert isinstance(result.exception, _StopAfterConfigCheck), result.output


def test_failed_blacklist_loader_filters_by_genome_scope(tmp_path):
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "models": ["author/model-a", "author/model-b"],
                "merge_method": "linear",
            },
            "tasks": ["wikitext"],
        }
    )
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    blacklist_path = storage_path / evolve_ga.FAILED_BLACKLIST_FILENAME
    blacklist_path.write_text(
        "genome_scope,genotype_hash,error_stage,error_type,error_message\n"
        "matchscope,deadbeef,merge,invalid_genotype,bad weights\n"
        "otherscope,feedface,merge,invalid_genotype,other config\n",
        encoding="utf-8",
    )

    current_scope = evolve_ga._failed_blacklist_scope(config)
    rewritten = blacklist_path.read_text(encoding="utf-8").replace(
        "matchscope", current_scope, 1
    )
    blacklist_path.write_text(rewritten, encoding="utf-8")

    blacklist = evolve_ga._load_failed_genotype_blacklist(
        str(storage_path), current_scope
    )

    assert blacklist == {
        "deadbeef": {
            "score": None,
            "results": None,
            "error_stage": "merge",
            "error_type": "invalid_genotype",
            "error_message": "bad weights",
        }
    }


def test_num_gpus_zero_disables_merge_cuda_before_baseline(monkeypatch, tmp_path):
    config_path = tmp_path / "evolve.yml"
    storage_path = tmp_path / "storage"
    config_path.write_text(
        yaml.safe_dump(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evolve_ga, "check_for_naughty_config", lambda config, allow: None
    )

    def fake_run_baseline_evaluations(
        config,
        storage_path,
        batch_size,
        merge_cuda,
        num_gpus,
        task_search_path,
        trust_remote_code,
    ):
        assert merge_cuda is False
        assert num_gpus == 0
        raise _StopAfterBaselineHook("stop after merge-cuda resolution")

    monkeypatch.setattr(
        evolve_ga, "run_baseline_evaluations", fake_run_baseline_evaluations
    )

    runner = CliRunner()
    result = runner.invoke(
        evolve_ga.main,
        [
            str(config_path),
            "--storage-path",
            str(storage_path),
            "--num-gpus",
            "0",
            "--no-reshard",
        ],
    )

    assert isinstance(result.exception, _StopAfterBaselineHook), result.output
