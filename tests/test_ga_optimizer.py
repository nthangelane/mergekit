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

