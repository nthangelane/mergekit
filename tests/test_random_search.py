import numpy as np
import torch

from mergekit.evo.random_search import RandomSearchOptimizer


class _Genome:
    def initial_genotype(self, random=False):
        assert random
        return torch.rand(4)


class _Strategy:
    def __init__(self):
        self.seen = []

    def evaluate_genotypes(self, genotypes):
        self.seen.extend(np.asarray(genotype).copy() for genotype in genotypes)
        return [
            {"score": float(np.asarray(genotype).sum()), "results": {}}
            for genotype in genotypes
        ]


def _optimizer(seed, strategy=None, **kwargs):
    return RandomSearchOptimizer(
        genome=_Genome(),
        strategy=strategy or _Strategy(),
        num_samples=8,
        population_size=4,
        seed=seed,
        **kwargs,
    )


def test_random_search_seed_reproduces_identical_genotypes():
    first = _optimizer(11).sample_population()
    second = _optimizer(11).sample_population()
    different = _optimizer(12).sample_population()

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_random_search_evaluates_and_logs_every_candidate():
    strategy = _Strategy()
    population_events = []
    best_events = []
    optimizer = _optimizer(
        11,
        strategy=strategy,
        on_population_evaluated=lambda results, population, step, info: (
            population_events.append((results, population, step, info))
        ),
        on_new_best=lambda genotype, score, step: best_events.append(
            (genotype, score, step)
        ),
    )

    best, score = optimizer.run()

    assert len(strategy.seen) == 8
    assert len(population_events) == 2
    assert [len(event[0]) for event in population_events] == [4, 4]
    assert [event[2] for event in population_events] == [4, 8]
    assert [event[3]["role_counts"] for event in population_events] == [
        {"random_search": 4},
        {"random_search": 4},
    ]
    assert all(len(event[1]) == 4 for event in population_events)
    assert best_events[-1][2] in {4, 8}
    assert np.isclose(score, best.sum())
    assert optimizer.last_stop_details["reason"] == "random_search_complete"
