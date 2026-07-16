import pytest

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.orchestrator import (
    build_run_signature,
    resolve_device,
    resolve_ga_params,
    resolve_stop_configuration,
)


def _config(**updates):
    payload = {
        "genome": {
            "merge_method": "linear",
            "models": ["author/model-a", "author/model-b"],
            "layer_granularity": 0,
        },
        "tasks": ["sciq"],
        "ga": {"population_size": 8, "mutation_sigma": 0.02},
        "stop": {"max_fevals": 40, "max_time_seconds": 600},
    }
    payload.update(updates)
    return EvolMergeConfiguration.model_validate(payload)


def test_stop_resolution_prefers_cli_and_keeps_yaml_fallbacks():
    resolved = resolve_stop_configuration(
        _config(),
        max_fevals_cli=12,
        timeout_cli=None,
    )

    assert resolved["max_fevals"] == 12
    assert resolved["timeout_seconds"] == 600.0


def test_ga_resolution_applies_cli_yaml_and_stop_precedence():
    config = _config()
    resolved_stop = resolve_stop_configuration(
        config,
        max_fevals_cli=None,
        timeout_cli=None,
    )
    params = resolve_ga_params(
        config,
        population_size=None,
        elite_fraction=None,
        mutation_rate=None,
        mutation_sigma=0.03,
        crossover=None,
        tournament_size=None,
        resolved_stop=resolved_stop,
        baseline_best_score=0.4,
    )

    assert params.population_size == 8
    assert params.mutation_sigma == 0.03
    assert params.target_reference_score == 0.4


def test_run_signature_changes_with_fitness_definition():
    legacy = _config()
    version_two = _config(fitness={"version": "v2"})
    params = resolve_ga_params(
        legacy,
        population_size=None,
        elite_fraction=None,
        mutation_rate=None,
        mutation_sigma=None,
        crossover=None,
        tournament_size=None,
        resolved_stop=resolve_stop_configuration(
            legacy, max_fevals_cli=None, timeout_cli=None
        ),
        baseline_best_score=None,
    )
    kwargs = {
        "random_seed": 11,
        "strategy": "serial",
        "device": "cpu",
        "batch_size": 1,
        "merge_cuda": False,
        "trust_remote_code": False,
        "vllm": False,
        "tensor_parallel_size": 1,
        "num_gpus": 0,
    }

    assert build_run_signature(legacy, params, **kwargs) != build_run_signature(
        version_two, params, **kwargs
    )


def test_auto_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(
        "mergekit.evo.orchestrator.torch.cuda.is_available", lambda: False
    )

    assert resolve_device("auto", None) == "cpu"


def test_cpu_device_rejects_positive_gpu_allocation():
    with pytest.raises(ValueError, match="--device cpu conflicts"):
        resolve_device("cpu", 1)


def test_cuda_device_rejects_zero_gpu_allocation(monkeypatch):
    monkeypatch.setattr(
        "mergekit.evo.orchestrator.torch.cuda.is_available", lambda: True
    )

    with pytest.raises(ValueError, match="--device cuda conflicts"):
        resolve_device("cuda", 0)
