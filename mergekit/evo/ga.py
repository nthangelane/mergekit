# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from mergekit.evo.genome import ModelGenome
from mergekit.evo.strategy import EvaluationStrategyBase


OnPopulationEvaluated = Callable[[List[dict], int], None]
OnNewBest = Callable[[np.ndarray, float, int], None]


@dataclass
class GAParams:
    population_size: int = 32
    elite_fraction: float = 0.125
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.05
    crossover: str = "arithmetic"  # or "uniform"
    tournament_size: int = 4


class GAOptimizer:
    """Simple genetic algorithm optimizer for model genome parameters.

    Uses the provided EvaluationStrategy to score genotypes and evolves a
    population to maximize the score.
    """

    def __init__(
        self,
        genome: ModelGenome,
        strategy: EvaluationStrategyBase,
        params: GAParams,
        random_init: bool = False,
        seed: Optional[int] = None,
        on_population_evaluated: Optional[OnPopulationEvaluated] = None,
        on_new_best: Optional[OnNewBest] = None,
    ):
        self.genome = genome
        self.strategy = strategy
        self.params = params
        self.random_init = random_init
        self.rs = np.random.RandomState(seed) if seed is not None else np.random
        self.on_population_evaluated = on_population_evaluated
        self.on_new_best = on_new_best

        x0 = self.genome.initial_genotype(random=self.random_init).view(-1).numpy()
        self.dim = x0.shape[0]

        self.pop_size = max(2, int(self.params.population_size))
        self.n_elite = max(1, int(self.params.elite_fraction * self.pop_size))

    def run(self, max_fevals: int, timeout: Optional[float] = None) -> Tuple[np.ndarray, float]:
        pop = self._init_population()
        fevals = 0
        start_time = time.time()
        best_x = pop[0].copy()
        best_score = -np.inf

        while fevals < max_fevals and (timeout is None or (time.time() - start_time) < timeout):
            fitness, res_list = self._evaluate_population(pop)
            fevals += self.pop_size

            if self.on_population_evaluated:
                self.on_population_evaluated(res_list, fevals)

            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = float(fitness[gen_best_idx])
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_x = pop[gen_best_idx].copy()
                if self.on_new_best:
                    self.on_new_best(best_x, best_score, fevals)

            # Elitism
            order = np.argsort(-fitness)
            elites = pop[order[: self.n_elite]].copy()

            # Create next generation
            next_pop = [*elites]
            while len(next_pop) < self.pop_size:
                p1 = pop[self._select_parent(fitness)]
                p2 = pop[self._select_parent(fitness)]
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_pop.append(child.astype(np.float32))

            pop = np.stack(next_pop, axis=0)

        return best_x, best_score

    # --- internals ---
    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim), dtype=np.float32)
        x0 = self.genome.initial_genotype(random=self.random_init).view(-1).numpy()
        for i in range(self.pop_size):
            if self.random_init:
                pop[i] = self.genome.initial_genotype(random=True).view(-1).numpy()
            else:
                pop[i] = x0 + self.rs.randn(self.dim).astype(np.float32) * 0.05
        return pop

    def _evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        results = self.strategy.evaluate_genotypes([ind for ind in pop])
        fitness = np.array([r["score"] if r["score"] is not None else -np.inf for r in results])
        return fitness, list(results)

    def _select_parent(self, fitness: np.ndarray) -> int:
        k = max(2, int(self.params.tournament_size))
        idxs = self.rs.randint(0, self.pop_size, size=k)
        best = idxs[0]
        best_fit = fitness[best]
        for idx in idxs[1:]:
            if fitness[idx] > best_fit:
                best = idx
                best_fit = fitness[idx]
        return best

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.params.crossover == "uniform":
            mask = self.rs.rand(a.size) < 0.5
            child = a.copy()
            child[mask] = b[mask]
            return child
        # arithmetic (default)
        alpha = self.rs.rand()
        return alpha * a + (1 - alpha) * b

    def _mutate(self, x: np.ndarray):
        rate = max(0.0, min(1.0, float(self.params.mutation_rate)))
        mask = self.rs.rand(x.size) < rate
        noise = self.rs.randn(x.size) * float(self.params.mutation_sigma)
        x[mask] += noise[mask]
