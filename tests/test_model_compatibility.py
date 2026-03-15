import pytest

from mergekit.common import ModelReference
from mergekit.evo.helpers import (
    _find_signature_incompatibilities,
    validate_input_model_architecture,
)
from mergekit.options import MergeOptions


def test_find_signature_incompatibility_detects_hidden_size():
    signatures = {
        "model_a": {"hidden_size": 512, "model_type": "llama"},
        "model_b": {"hidden_size": 768, "model_type": "llama"},
    }

    mismatches = _find_signature_incompatibilities(signatures)

    assert any("hidden_size" in entry for entry in mismatches)


def test_find_signature_incompatibility_handles_missing_values():
    signatures = {
        "model_a": {"hidden_size": None, "model_type": "llama"},
        "model_b": {"hidden_size": 512, "model_type": "llama"},
    }

    mismatches = _find_signature_incompatibilities(signatures)

    assert any("hidden_size" in entry for entry in mismatches)
    assert any("<missing>" in entry for entry in mismatches)


def test_find_signature_incompatibility_no_mismatch_when_equal():
    signatures = {
        "model_a": {"hidden_size": 512, "model_type": "llama"},
        "model_b": {"hidden_size": 512, "model_type": "llama"},
    }

    mismatches = _find_signature_incompatibilities(signatures)

    assert mismatches == []


class DummyConfig:
    def __init__(self, **overrides):
        values = {
            "architectures": ["DummyForCausalLM"],
            "model_type": "dummy",
            "hidden_size": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
            "max_position_embeddings": 4096,
            "sliding_window": None,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "vocab_size": 32000,
        }
        values.update(overrides)
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__.copy()


def test_validate_input_model_architecture_rejects_mixed_architectures(monkeypatch):
    configs = {
        "author/model-a": DummyConfig(
            architectures=["Qwen2ForCausalLM"],
            model_type="qwen2",
        ),
        "author/model-b": DummyConfig(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
        ),
    }

    def fake_config(self, trust_remote_code: bool = False):
        return configs[str(self)]

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    with pytest.raises(RuntimeError, match="same base architecture"):
        validate_input_model_architecture(
            [
                ModelReference.parse("author/model-a"),
                ModelReference.parse("author/model-b"),
            ],
            MergeOptions(),
        )


def test_validate_input_model_architecture_ignores_vocab_size_changes(monkeypatch):
    configs = {
        "author/model-a": DummyConfig(vocab_size=32000),
        "author/model-b": DummyConfig(vocab_size=32032),
    }

    def fake_config(self, trust_remote_code: bool = False):
        return configs[str(self)]

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)

    validate_input_model_architecture(
        [
            ModelReference.parse("author/model-a"),
            ModelReference.parse("author/model-b"),
        ],
        MergeOptions(),
    )
