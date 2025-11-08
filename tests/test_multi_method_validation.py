import pytest

from mergekit.common import ModelReference
from mergekit.evo.multi_method_genome import MultiMethodGenome, MultiMethodGenomeDefinition


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


def test_multi_method_rejects_excess_models_for_slerp():
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
                "max_models_per_layer": 3,
            }
        )


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
