from pathlib import Path

import pytest
import yaml

from mergekit.common import ModelReference
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)


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
    assert config.models is not None
    assert len(config.models) == 1
    assert str(config.models[0].model) == "author/model-a"


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
