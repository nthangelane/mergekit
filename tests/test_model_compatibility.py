from mergekit.evo.helpers import _find_signature_incompatibilities


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
