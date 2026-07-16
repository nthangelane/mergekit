import json

import pytest

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.fitness import ensure_fitness_definition, fitness_definition_payload


def _config(version="v2"):
    return EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "type": "multi_method",
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
            },
            "tasks": ["sciq"],
            "fitness_mode": "structured_phase1_tiny",
            "fitness": {"version": version},
            "task_mix_profile": "pythia70m_phase1",
        }
    )


def test_v2_fitness_definition_records_formula_and_profile_weights():
    payload = fitness_definition_payload(_config())

    assert payload["fitness_version"] == "v2"
    assert payload["lower_is_better_transform"] == "log_reciprocal"
    assert payload["lower_is_better_formula"] == ("1 / (1 + log1p(max(0, x)))")
    assert payload["task_mix_weights"]["language_quality_weight"] == 0.35


def test_resume_rejects_different_existing_fitness_definition(tmp_path):
    definition_path = ensure_fitness_definition(
        str(tmp_path),
        _config("v1"),
        resume=False,
    )
    original = json.loads(tmp_path.joinpath("fitness_definition.json").read_text())

    with pytest.raises(ValueError, match="does not match the existing resumed run"):
        ensure_fitness_definition(
            str(tmp_path),
            _config("v2"),
            resume=True,
        )

    assert definition_path == str(tmp_path / "fitness_definition.json")
    assert (
        json.loads(tmp_path.joinpath("fitness_definition.json").read_text()) == original
    )
