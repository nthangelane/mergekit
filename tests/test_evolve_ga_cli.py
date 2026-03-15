import yaml
from click.testing import CliRunner

import mergekit.scripts.evolve_ga as evolve_ga


class _StopAfterConfigCheck(RuntimeError):
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

    monkeypatch.setattr(evolve_ga, "check_for_naughty_config", fake_check_for_naughty_config)

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
