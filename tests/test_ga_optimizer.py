import numpy as np
import torch

from mergekit.evo.ga import GAOptimizer, GAParams


class FakeGenome:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def initial_genotype(self, random: bool = False):
        # Shape: [1, n_models=dim, 1, 1] to match expectations
        if random:
            x = torch.rand(1, self.dim, 1, 1)
        else:
            x = torch.zeros(1, self.dim, 1, 1)
        return x


class FakeStrategy:
    def __init__(self, target: np.ndarray):
        self.target = target.astype(np.float32).ravel()

    def evaluate_genotypes(self, genotypes):
        # Higher is better: negative L2 distance to target
        res = []
        for g in genotypes:
            g = np.array(g).ravel().astype(np.float32)
            score = -float(np.linalg.norm(g - self.target))
            res.append({"score": score, "results": None})
        return res


class CountingFailureStrategy:
    def __init__(self):
        self.calls = 0

    def evaluate_genotypes(self, genotypes):
        self.calls += len(genotypes)
        res = []
        for genotype in genotypes:
            flat = np.array(genotype).ravel().astype(np.float32)
            if flat[0] < 0:
                res.append(
                    {
                        "score": None,
                        "results": None,
                        "error_stage": "merge",
                        "error_type": "invalid_genotype",
                        "error_message": "negative leading weight",
                    }
                )
            else:
                res.append({"score": float(flat.sum()), "results": {}})
        return res


def test_ga_optimizer_improves_score():
    dim = 6
    genome = FakeGenome(dim=dim)
    target = np.linspace(0.0, 1.0, dim)
    strat = FakeStrategy(target=target)

    params = GAParams(
        population_size=20,
        elite_fraction=0.2,
        mutation_rate=0.3,
        mutation_sigma=0.1,
        crossover="arithmetic",
        tournament_size=3,
    )

    opt = GAOptimizer(
        genome=genome,
        strategy=strat,  # type: ignore[arg-type]
        params=params,
        random_init=False,
        seed=42,
    )

    best_x, best_score = opt.run(max_fevals=200, timeout=None)

    # Baseline (x0 is all zeros) score against target
    baseline = -float(np.linalg.norm(np.zeros(dim) - target))

    assert best_score > baseline, "GA should improve over baseline"
    assert best_x.shape[0] == dim, "Returned best_x should be flattened genotype"


def test_ga_optimizer_caches_failed_genotypes_without_retry():
    genome = FakeGenome(dim=4)
    strategy = CountingFailureStrategy()
    params = GAParams(population_size=4)
    opt = GAOptimizer(
        genome=genome,
        strategy=strategy,  # type: ignore[arg-type]
        params=params,
        seed=0,
    )
    opt._fitness_cache = {}

    failed = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    valid = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pop = np.stack([failed, valid], axis=0)

    fitness_first, results_first = opt._evaluate_population(pop)
    fitness_second, results_second = opt._evaluate_population(pop)

    assert strategy.calls == 2
    assert np.isneginf(fitness_first[0])
    assert fitness_first[1] == 1.0
    assert np.array_equal(fitness_first, fitness_second)
    assert results_first[0]["error_type"] == "invalid_genotype"
    assert results_second[0]["error_message"] == "negative leading weight"
    assert opt._last_eval_stats["evaluations"] == 0.0
    assert opt._last_eval_stats["cache_hits"] == 2.0
    assert opt._last_eval_stats["failed_evals"] == 1.0
    assert opt._last_eval_stats["failure_reasons"] == "merge:invalid_genotype:1"
