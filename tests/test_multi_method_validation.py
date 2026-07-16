from pathlib import Path

import pytest
import yaml

from mergekit.common import ModelReference
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.helpers import merge_model_with_details
from mergekit.evo.multi_method_genome import (
    MergeMethod,
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)
from mergekit.options import MergeOptions


def test_multi_method_allows_slerp_with_limited_selection():
    config = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
                "author/model-c",
            ],
            "base_model": "author/model-a",
            "allowed_methods": ["slerp"],
            "max_models_per_layer": 2,
        }
    )

    assert config.max_models_per_layer == 2


def test_multi_method_rejects_insufficient_models_per_layer_for_slerp():
    with pytest.raises(ValueError):
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": [
                    "author/model-a",
                    "author/model-b",
                    "author/model-c",
                ],
                "base_model": "author/model-a",
                "allowed_methods": ["slerp"],
                "max_models_per_layer": 1,
            }
        )


def test_multi_method_allows_linear_with_passthrough():
    config = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
                "author/model-c",
            ],
            "allowed_methods": ["linear", "passthrough"],
            "max_models_per_layer": 2,
        }
    )

    assert config.allowed_methods == ["linear", "passthrough"]


def test_multi_method_rejects_invalid_linear_constraints():
    with pytest.raises(ValueError):
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
                "linear_min_source_weight": 1.0,
            }
        )

    with pytest.raises(ValueError):
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
                "linear_max_scale": -0.1,
            }
        )


def test_linear_constraints_prevent_near_parent_extrapolation(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
                "linear_min_source_weight": 0.15,
                "linear_max_scale": 1.0,
            }
        )
    )
    genotype = genome.initial_genotype(random=False)
    model_start = genome.method_dim
    genotype[model_start : model_start + 2] = 0.0
    genotype[model_start] = 100.0
    genotype[model_start + 1] = 0.001
    genotype[model_start + genome.model_selection_dim] = 1.4

    decoded = genome.decode_genotype(genotype)[0]
    assert decoded.model_selection[0] == pytest.approx(0.85)
    assert decoded.model_selection[1] == pytest.approx(0.15)
    assert decoded.parameters[0] == pytest.approx(1.0)

    config = genome.genotype_to_merge_config(genotype)
    weights = [source.parameters["weight"] for source in config.slices[0].sources]
    assert weights == pytest.approx([0.85, 0.15])


def test_linear_constraints_activate_zero_weight_fallback_source(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
                "linear_min_source_weight": 0.15,
                "linear_max_scale": 1.0,
            }
        )
    )
    genotype = genome.initial_genotype(random=False)
    model_start = genome.method_dim
    genotype[model_start : model_start + 2] = 0.0
    genotype[model_start] = 1.0
    genotype[model_start + genome.model_selection_dim] = 1.0

    decoded = genome.decode_genotype(genotype)[0]
    assert decoded.model_selection[0] == pytest.approx(0.85)
    assert decoded.model_selection[1] == pytest.approx(0.15)

    config = genome.genotype_to_merge_config(genotype)
    weights = [source.parameters["weight"] for source in config.slices[0].sources]
    assert weights == pytest.approx([0.85, 0.15])


def test_multi_method_genotype_to_param_arrays_is_tabular(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear", "passthrough"],
                "layer_granularity": 2,
                "max_models_per_layer": 2,
            }
        )
    )
    genotype = genome.initial_genotype(random=False)
    second_group_offset = genome.layer_group_dim
    genotype[second_group_offset] = genome.method_gene_value("passthrough")

    params = genome.genotype_to_param_arrays(genotype)

    assert params["layer_group"] == [0, 1]
    assert params["merge_method"] == ["linear", "passthrough"]
    assert len(params["model_0_selection"]) == 2
    assert len(params["param_0"]) == 2


def test_slerp_config_includes_layer_ranges(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    definition = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
                "author/model-c",
            ],
            "base_model": "author/model-a",
            "allowed_methods": ["slerp"],
            "layer_granularity": 0,
            "enable_method_evolution": True,
            "enable_model_selection": True,
            "max_models_per_layer": 2,
        }
    )

    genome = MultiMethodGenome(definition)
    genotype = genome.initial_genotype(random=False)
    config = genome.genotype_to_merge_config(genotype)

    assert config.slices is not None
    assert len(config.slices) == 1
    source0 = config.slices[0].sources[0]
    source1 = config.slices[0].sources[1]
    assert source0.layer_range == (0, 4)
    assert source1.layer_range == (0, 4)
    assert config.slices[0].parameters["t"] == pytest.approx(0.5, rel=1e-6)


def test_layered_slerp_config_keeps_layer_group_ranges(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    definition = MultiMethodGenomeDefinition.model_validate(
        {
            "models": ["author/model-a", "author/model-b"],
            "base_model": "author/model-a",
            "allowed_methods": ["slerp"],
            "layer_granularity": 2,
            "enable_method_evolution": True,
            "enable_model_selection": True,
            "max_models_per_layer": 2,
        }
    )

    genome = MultiMethodGenome(definition)
    genotype = genome.initial_genotype(random=False)
    config = genome.genotype_to_merge_config(genotype)

    assert config.merge_method == "slerp"
    assert config.slices is not None
    assert [slice_def.sources[0].layer_range for slice_def in config.slices] == [
        (0, 2),
        (2, 4),
    ]
    assert [slice_def.sources[1].layer_range for slice_def in config.slices] == [
        (0, 2),
        (2, 4),
    ]


def test_passthrough_config_selects_one_model(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    definition = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
                "author/model-c",
            ],
            "allowed_methods": ["linear", "passthrough"],
            "layer_granularity": 0,
            "enable_method_evolution": True,
            "enable_model_selection": True,
            "max_models_per_layer": 2,
        }
    )

    genome = MultiMethodGenome(definition)
    genotype = genome.initial_genotype(random=False)
    genotype[0] = 1.0  # passthrough in allowed_methods order
    genotype[1:3] = 0.0
    genotype[1] = 1.0

    config = genome.genotype_to_merge_config(genotype)

    assert config.merge_method == "passthrough"
    assert config.slices is not None
    assert len(config.slices) == 1
    assert len(config.slices[0].sources) == 1
    assert str(config.slices[0].sources[0].model) == "author/model-a"


def test_default_genotype_prefers_linear_even_if_passthrough_is_listed_first(
    monkeypatch,
):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    definition = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
            ],
            "base_model": "author/model-a",
            "allowed_methods": ["passthrough", "linear", "slerp"],
            "layer_granularity": 0,
            "enable_method_evolution": True,
            "enable_model_selection": True,
            "max_models_per_layer": 2,
        }
    )

    genome = MultiMethodGenome(definition)
    genotype = genome.initial_genotype(random=False)
    config = genome.genotype_to_merge_config(genotype)

    assert config.merge_method == "linear"


def test_decode_genotype_without_method_evolution_uses_allowed_method(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    definition = MultiMethodGenomeDefinition.model_validate(
        {
            "models": [
                "author/model-a",
                "author/model-b",
            ],
            "allowed_methods": ["passthrough"],
            "layer_granularity": 0,
            "enable_method_evolution": False,
            "enable_model_selection": True,
            "max_models_per_layer": 2,
        }
    )

    genome = MultiMethodGenome(definition)
    genotype = genome.initial_genotype(random=False)
    decoded = genome.decode_genotype(genotype)

    assert decoded[0].method == MergeMethod.PASSTHROUGH


def test_evol_merge_configuration_accepts_stop_policy():
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "type": "multi_method",
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["passthrough", "linear", "slerp"],
                "base_model": "author/model-a",
                "max_models_per_layer": 2,
            },
            "tasks": [{"name": "boolq", "weight": 1.0, "metric": "acc,none"}],
            "stop": {
                "max_fevals": 200,
                "max_time_seconds": 14400,
                "target_improvement_pct": 5.0,
                "target_reference": "best_baseline",
                "min_generations_before_target_stop": 5,
                "require_stage2_for_target": True,
                "stagnation_patience_generations": 8,
                "stagnation_min_delta": 0.005,
            },
        }
    )

    assert config.stop is not None
    assert config.stop.max_fevals == 200
    assert config.stop.target_improvement_pct == pytest.approx(5.0)
    assert config.stop.require_stage2_for_target is True
    assert config.stop.stagnation_patience_generations == 8


def test_evol_config_defaults_existing_runs_to_v1_fitness():
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "type": "multi_method",
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
            },
            "tasks": ["sciq"],
        }
    )

    assert config.fitness.version == "v1"
    assert config.fitness.lower_is_better_transform == "legacy_reciprocal"


def test_evol_config_resolves_v2_fitness_transform():
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "type": "multi_method",
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
            },
            "tasks": ["sciq"],
            "fitness": {"version": "v2"},
        }
    )

    assert config.fitness.version == "v2"
    assert config.fitness.lower_is_better_transform == "log_reciprocal"


def test_evol_config_rejects_mislabeled_fitness_transform():
    with pytest.raises(ValueError, match="fitness.version 'v2' requires"):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "type": "multi_method",
                    "models": ["author/model-a", "author/model-b"],
                    "allowed_methods": ["linear"],
                    "max_models_per_layer": 2,
                },
                "tasks": ["sciq"],
                "fitness": {
                    "version": "v2",
                    "lower_is_better_transform": "legacy_reciprocal",
                },
            }
        )


def test_m1_micro_example_uses_layer_blocks(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 24
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    config_path = (
        Path(__file__).resolve().parents[1] / "examples" / "evolve_ga_m1_micro.yml"
    )
    config_data = yaml.safe_load(config_path.read_text())

    evol_config = EvolMergeConfiguration.model_validate(config_data)
    assert isinstance(evol_config.genome, MultiMethodGenomeDefinition)
    assert evol_config.genome.layer_granularity == 4
    assert "nuslerp" in evol_config.genome.allowed_methods

    genome = MultiMethodGenome(evol_config.genome)
    assert genome.num_layer_groups == 6  # 24 layers / 4-block granularity
    assert genome.max_models == 2


def test_eks_pythia70m_bridge_matches_constrained_local_preset():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "thesis"
        / "eks_gpu"
        / "exp08_pythia70m_adaptive_bridge"
        / "config.yml"
    )
    config_data = yaml.safe_load(config_path.read_text())

    evol_config = EvolMergeConfiguration.model_validate(config_data)

    assert isinstance(evol_config.genome, MultiMethodGenomeDefinition)
    assert evol_config.genome.allowed_methods == ["passthrough", "linear"]
    assert evol_config.genome.linear_min_source_weight == pytest.approx(0.15)
    assert evol_config.genome.linear_max_scale == pytest.approx(1.0)
    assert evol_config.stage1_limit == 4
    assert evol_config.stage2_limit == 24
    assert evol_config.stage2_top_k == 2
    assert evol_config.limit == 24
    assert evol_config.ga.initial_method_probs == {
        "passthrough": pytest.approx(0.25),
        "linear": pytest.approx(0.75),
    }


def test_evol_config_rejects_negative_initial_method_probabilities():
    with pytest.raises(ValueError):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "type": "multi_method",
                    "models": ["author/model-a", "author/model-b"],
                    "allowed_methods": ["linear", "passthrough"],
                    "max_models_per_layer": 2,
                },
                "tasks": ["sciq"],
                "ga": {
                    "adaptive_method_sampling": True,
                    "initial_method_probs": {"linear": 1.0, "passthrough": -0.1},
                },
            }
        )


def test_evol_config_rejects_invalid_passthrough_max_fraction():
    with pytest.raises(ValueError):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "type": "multi_method",
                    "models": ["author/model-a", "author/model-b"],
                    "allowed_methods": ["linear", "passthrough"],
                    "max_models_per_layer": 2,
                },
                "tasks": ["sciq"],
                "ga": {
                    "passthrough_max_fraction": 1.2,
                },
            }
        )


def test_phase1_profile_rejects_ties_method():
    with pytest.raises(ValueError):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "type": "multi_method",
                    "models": ["author/model-a", "author/model-b"],
                    "base_model": "author/model-a",
                    "allowed_methods": ["linear", "ties"],
                    "max_models_per_layer": 2,
                },
                "tasks": ["sciq"],
                "fitness_mode": "structured_phase1_tiny",
                "task_mix_profile": "pythia70m_phase1",
                "ga": {
                    "mutation_rate": 0.1,
                    "mutation_sigma": 0.01,
                },
            }
        )


def test_phase1_profile_rejects_large_mutation():
    with pytest.raises(ValueError):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "type": "multi_method",
                    "models": ["author/model-a", "author/model-b"],
                    "base_model": "author/model-a",
                    "allowed_methods": ["linear", "slerp", "passthrough"],
                    "max_models_per_layer": 2,
                },
                "tasks": ["sciq"],
                "fitness_mode": "structured_phase1_tiny",
                "task_mix_profile": "pythia70m_phase1",
                "ga": {
                    "mutation_rate": 0.2,
                    "mutation_sigma": 0.03,
                },
            }
        )


def test_phase3_profile_allows_expanded_method_family():
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "type": "multi_method",
                "models": ["author/model-a", "author/model-b"],
                "base_model": "author/model-a",
                "allowed_methods": [
                    "linear",
                    "slerp",
                    "ties",
                    "dare_linear",
                    "dare_ties",
                    "passthrough",
                ],
                "max_models_per_layer": 2,
            },
            "tasks": ["sciq"],
            "fitness_mode": "weighted_rank",
            "task_mix_profile": "pythia70m_phase3",
            "ga": {
                "mutation_rate": 0.1,
                "mutation_sigma": 0.01,
                "rank_objective_weights": {
                    "task_score": 0.4,
                    "language_quality": 0.3,
                    "stability_score": 0.2,
                    "archive_novelty_score": 0.1,
                },
            },
        }
    )

    assert config.fitness_mode == "weighted_rank"
    assert config.task_mix_profile == "pythia70m_phase3"


def test_layered_single_method_config_emits_real_slices(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "layer_granularity": 2,
                "max_models_per_layer": 2,
            }
        )
    )

    config = genome.genotype_to_merge_config(genome.initial_genotype(random=False))

    assert config.merge_method == "linear"
    assert config.slices is not None
    assert len(config.slices) == 2
    assert config.slices[0].sources[0].layer_range == (0, 2)
    assert config.slices[1].sources[0].layer_range == (2, 4)


def test_layered_mixed_method_plan_builds_real_slice_config(monkeypatch):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear", "passthrough"],
                "layer_granularity": 2,
                "max_models_per_layer": 2,
            }
        )
    )

    genotype = genome.initial_genotype(random=False)
    second_group_offset = genome.layer_group_dim
    genotype[second_group_offset] = genome.method_gene_value("passthrough")
    genotype[
        second_group_offset
        + genome.method_dim : second_group_offset
        + genome.method_dim
        + genome.model_selection_dim
    ] = 0.0
    genotype[second_group_offset + genome.method_dim] = 1.0

    plan = genome.genotype_to_merge_plan(genotype)

    assert plan["kind"] == "config"
    config = plan["config"]
    assert config.slices is not None
    assert config.slices[0].merge_method == "linear"
    assert config.slices[1].merge_method == "passthrough"
    assert config.slices[0].sources[0].layer_range == (0, 2)
    assert config.slices[1].sources[0].layer_range == (2, 4)


def test_merge_model_with_details_executes_native_layered_config(monkeypatch, tmp_path):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    genome = MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear", "passthrough"],
                "layer_granularity": 2,
                "max_models_per_layer": 2,
            }
        )
    )
    genotype = genome.initial_genotype(random=False)
    second_group_offset = genome.layer_group_dim
    genotype[second_group_offset] = genome.method_gene_value("passthrough")
    genotype[
        second_group_offset
        + genome.method_dim : second_group_offset
        + genome.method_dim
        + genome.model_selection_dim
    ] = 0.0
    genotype[second_group_offset + genome.method_dim] = 1.0

    seen_configs = []

    def fake_run_merge(cfg, out_path, options, **kwargs):
        seen_configs.append(cfg)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        (Path(out_path) / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("mergekit.evo.helpers.run_merge", fake_run_merge)
    monkeypatch.setattr(
        "mergekit.evo.helpers._validate_merge_compatibility",
        lambda *args, **kwargs: None,
    )

    result = merge_model_with_details(
        genotype,
        genome,
        str(tmp_path),
        MergeOptions(reuse_scratch_dir=True, min_free_disk_gb=0),
    )

    assert result["merged_path"] is not None
    assert result["merged_path"].startswith(str(tmp_path / "candidate-scratch-"))
    assert len(seen_configs) == 1
    assert seen_configs[0].slices is not None
    assert [slice_def.merge_method for slice_def in seen_configs[0].slices] == [
        "linear",
        "passthrough",
    ]
    assert "resolved_merge_config" in result
