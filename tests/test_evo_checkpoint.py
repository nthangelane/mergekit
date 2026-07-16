import random

import numpy as np
import pytest
import torch

from mergekit.evo.checkpoint import (
    atomic_write_json,
    capture_rng_state,
    load_ga_state,
    restore_rng_state,
)
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer, EnhancedGAParams
from mergekit.evo.resources import ensure_free_disk


def test_checkpoint_atomic_round_trip(tmp_path):
    payload = {
        "schema_version": 1,
        "generation": 2,
        "population": np.array([[1.0, 2.0]], dtype=np.float32),
    }

    atomic_write_json(str(tmp_path / "ga_state.json"), payload)

    state = load_ga_state(str(tmp_path))
    assert state["generation"] == 2
    assert state["population"] == [[1.0, 2.0]]
    assert not (tmp_path / ".ga_state.json.tmp").exists()


def test_rng_state_round_trip_includes_optimizer_stream():
    optimizer_rng = np.random.RandomState(19)
    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    state = capture_rng_state(optimizer_rng)

    expected = (
        random.random(),
        np.random.random(),
        float(torch.rand(1).item()),
        optimizer_rng.random_sample(),
    )

    restore_rng_state(state, optimizer_rng)
    actual = (
        random.random(),
        np.random.random(),
        float(torch.rand(1).item()),
        optimizer_rng.random_sample(),
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


class _Genome:
    def initial_genotype(self, random=False):
        return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)


class _Strategy:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def evaluate_genotypes(self, genotypes):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return [
            {"score": float(np.asarray(genotype).sum()), "results": {}}
            for genotype in genotypes
        ]


def _enhanced_optimizer(state_path, strategy, resume_state=None):
    return EnhancedGAOptimizer(
        genome=_Genome(),
        strategy=strategy,
        params=EnhancedGAParams(
            population_size=4,
            elite_fraction=0.25,
            crossover="arithmetic",
            mutation_rate=0.5,
            mutation_sigma=0.1,
        ),
        seed=23,
        checkpoint_path=str(state_path),
        resume_state=resume_state,
        config_signature="determinism-test",
    )


def test_interrupted_resume_matches_uninterrupted_run(tmp_path):
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted_path = uninterrupted_dir / "ga_state.json"
    resumed_path = resumed_dir / "ga_state.json"

    uninterrupted = _enhanced_optimizer(uninterrupted_path, _Strategy())
    expected_best, expected_score = uninterrupted.run(max_fevals=12)

    interrupted = _enhanced_optimizer(resumed_path, _Strategy(fail_on_call=2))
    try:
        interrupted.run(max_fevals=12)
    except RuntimeError as exc:
        assert "simulated interruption" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected the simulated interruption")

    saved_state = load_ga_state(str(resumed_dir))
    assert saved_state["generation"] == 1
    assert saved_state["fevals"] == 4

    resumed = _enhanced_optimizer(
        resumed_path,
        _Strategy(),
        resume_state=saved_state,
    )
    actual_best, actual_score = resumed.run(max_fevals=12)

    np.testing.assert_array_equal(actual_best, expected_best)
    assert actual_score == expected_score
    assert (
        load_ga_state(str(resumed_dir))["fitness_history"]
        == load_ga_state(str(uninterrupted_dir))["fitness_history"]
    )


def test_low_disk_abort_leaves_resumable_state(monkeypatch, tmp_path):
    class _LowDiskStrategy:
        def evaluate_genotypes(self, genotypes):
            ensure_free_disk(str(tmp_path), 5.0)
            raise AssertionError("disk guard should have raised")

    monkeypatch.setenv("MERGEKIT_FREE_DISK_GB_OVERRIDE", "0.1")
    state_path = tmp_path / "ga_state.json"
    optimizer = _enhanced_optimizer(state_path, _LowDiskStrategy())

    with pytest.raises(RuntimeError, match="Insufficient free disk space"):
        optimizer.run(max_fevals=4)

    state = load_ga_state(str(tmp_path))
    assert state["status"] == "ready"
    assert state["generation"] == 0
    assert state["fevals"] == 0
