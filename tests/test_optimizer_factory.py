import pytest

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.ga import GAParams
from mergekit.evo.optimizer_factory import (
    build_enhanced_ga_params,
    resolve_optimizer_kind,
)


def _standard_config(*, optimizer="auto", ga=None):
    payload = {
        "genome": {
            "merge_method": "linear",
            "models": ["author/model-a", "author/model-b"],
            "layer_granularity": 0,
        },
        "tasks": ["sciq"],
        "optimizer": optimizer,
    }
    if ga is not None:
        payload["ga"] = ga
    return EvolMergeConfiguration.model_validate(payload)


def _multi_method_config(*, optimizer="auto", ga=None):
    payload = {
        "genome": {
            "type": "multi_method",
            "models": ["author/model-a", "author/model-b"],
            "allowed_methods": ["linear"],
            "max_models_per_layer": 2,
        },
        "tasks": ["sciq"],
        "optimizer": optimizer,
    }
    if ga is not None:
        payload["ga"] = ga
    return EvolMergeConfiguration.model_validate(payload)


def test_optimizer_auto_preserves_legacy_basic_ga_selection():
    config = _standard_config(ga={"population_size": 8})

    assert resolve_optimizer_kind(config, "standard") == "enhanced"


def test_optimizer_can_explicitly_select_standard_for_basic_ga():
    config = _standard_config(
        optimizer="standard",
        ga={"population_size": 8},
    )

    assert resolve_optimizer_kind(config, "standard") == "standard"


def test_standard_optimizer_rejects_multi_method_genome():
    config = _multi_method_config(optimizer="standard")

    with pytest.raises(ValueError, match="multi_method genome"):
        resolve_optimizer_kind(config, "multi_method")


def test_standard_optimizer_rejects_enhanced_ga_feature():
    config = _standard_config(
        optimizer="standard",
        ga={"adaptive_method_sampling": True},
    )

    with pytest.raises(ValueError, match="ga.adaptive_method_sampling"):
        resolve_optimizer_kind(config, "standard")


def test_build_enhanced_params_combines_resolved_and_yaml_values():
    config = _multi_method_config(
        optimizer="enhanced",
        ga={
            "adaptive_method_sampling": True,
            "initial_method_probs": {"linear": 1.0},
            "operator_temperature": 0.5,
        },
    )
    base = GAParams(population_size=7, mutation_sigma=0.012)

    enhanced = build_enhanced_ga_params(base, config)

    assert enhanced.population_size == 7
    assert enhanced.mutation_sigma == pytest.approx(0.012)
    assert enhanced.adaptive_method_sampling is True
    assert enhanced.initial_method_probs == {"linear": 1.0}
    assert enhanced.operator_temperature == pytest.approx(0.5)
    assert enhanced.fitness_mode == config.fitness_mode
