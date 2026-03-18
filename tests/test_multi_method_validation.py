from pathlib import Path

import pytest
import yaml

from mergekit.common import ModelReference
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.helpers import merge_model_with_details
from mergekit.evo.multi_method_genome import (
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
        MergeOptions(),
    )

    assert result["merged_path"] is not None
    assert len(seen_configs) == 1
    assert seen_configs[0].slices is not None
    assert [slice_def.merge_method for slice_def in seen_configs[0].slices] == [
        "linear",
        "passthrough",
    ]
    assert "resolved_merge_config" in result
