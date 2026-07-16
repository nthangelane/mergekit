import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.repair import ParentDistiller, repair


class _TinyLM(torch.nn.Module):
    def __init__(self, seed, vocab_size=11):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Embedding(vocab_size, 6)
        self.output = torch.nn.Linear(6, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _corpus():
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }
    ]


def test_repair_gate_restores_child_when_probe_does_not_improve():
    child = _TinyLM(3)
    teacher = copy.deepcopy(child)
    before = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    distiller = ParentDistiller([teacher], tau_distill=2.0)

    result = repair(
        child,
        distiller,
        _corpus(),
        probe_steps=3,
        max_steps=5,
        gate_min_slope=1e-5,
        lr=1e-3,
        batch_size=2,
        seq_len=4,
        seed=17,
    )

    assert result.repaired is False
    assert result.steps_used == 3
    for name, value in child.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_repair_probe_slope_is_deterministic():
    teacher_one = _TinyLM(5)
    teacher_two = copy.deepcopy(teacher_one)
    child_one = _TinyLM(7)
    child_two = copy.deepcopy(child_one)

    result_one = repair(
        child_one,
        ParentDistiller([teacher_one]),
        _corpus(),
        probe_steps=4,
        max_steps=4,
        gate_min_slope=1.0,
        lr=1e-3,
        batch_size=2,
        seq_len=4,
        seed=29,
    )
    result_two = repair(
        child_two,
        ParentDistiller([teacher_two]),
        _corpus(),
        probe_steps=4,
        max_steps=4,
        gate_min_slope=1.0,
        lr=1e-3,
        batch_size=2,
        seq_len=4,
        seed=29,
    )

    assert result_one.probe_slope == pytest.approx(result_two.probe_slope, abs=1e-6)


def test_repair_aligns_parent_and_child_padded_vocabularies():
    child = _TinyLM(7, vocab_size=11)
    parent = _TinyLM(5, vocab_size=13)

    result = repair(
        child,
        ParentDistiller([parent]),
        _corpus(),
        probe_steps=1,
        max_steps=1,
        gate_min_slope=1.0,
        lr=1e-3,
        batch_size=2,
        seq_len=4,
        seed=29,
    )

    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.probe_end_loss)


def test_repair_does_not_perturb_caller_torch_rng():
    child = _TinyLM(7)
    parent = _TinyLM(5)
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()

    repair(
        child,
        ParentDistiller([parent]),
        _corpus(),
        probe_steps=1,
        max_steps=1,
        gate_min_slope=1.0,
        lr=1e-3,
        batch_size=2,
        seq_len=4,
        seed=29,
    )

    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0.0, atol=0.0)


def test_repair_config_rejects_overlapping_corpus_split():
    with pytest.raises(ValueError, match="Repair corpus contamination"):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext_train"],
                "two_stage": True,
                "stage1_tasks": ["wikitext"],
                "stage2_top_k": 2,
                "repair": {"enabled": True},
            }
        )


def test_repair_requires_two_stage_protocol():
    with pytest.raises(ValueError, match="requires two_stage"):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext"],
                "repair": {"enabled": True},
            }
        )


def test_repair_rejects_corpus_without_verifiable_split():
    with pytest.raises(ValueError, match="known corpus with a verifiable split"):
        EvolMergeConfiguration.model_validate(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext"],
                "two_stage": True,
                "stage1_tasks": ["wikitext"],
                "stage2_top_k": 2,
                "repair": {"enabled": True, "corpus": "untracked-corpus"},
            }
        )
