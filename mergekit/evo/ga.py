# Copyright (C) 2025 Nkululeko Thangelane
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mergekit.evo.cache_utils import genotype_cache_key, persisted_failure_result
from mergekit.evo.genome import ModelGenome
from mergekit.evo.strategy import EvaluationStrategyBase

OnPopulationEvaluated = Callable[[List[dict], np.ndarray, int, Dict[str, Any]], None]
OnNewBest = Callable[[np.ndarray, float, int], None]
OnGenerationStart = Callable[[int, int, int, int, float], None]


@dataclass
class GAParams:
    population_size: int = 32
    elite_fraction: float = 0.125
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.05
    crossover: str = "arithmetic"  # or "uniform" or "sbx"
    tournament_size: int = 4
    # caching and diversity
    cache_round: float = 1e-4
    immigrant_fraction: float = 0.0
    # adaptive mutation
    patience: int = (
        0  # generations without improvement before decaying sigma; 0 disables
    )
    sigma_decay: float = 0.5
    min_mutation_sigma: float = 0.005


def _uniform_crossover(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    mask = np.random.randint(0, 2, size=x1.shape, dtype=np.bool_)
    return np.where(mask, x1, x2)


def _summarize_failure_reasons(results: List[dict]) -> str:
    counts: Dict[str, int] = {}
    for result in results:
        if result.get("score") is not None:
            continue
        stage = result.get("error_stage") or "unknown"
        error_type = result.get("error_type") or "unknown"
        key = f"{stage}:{error_type}"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return ";".join(f"{key}:{count}" for key, count in sorted(counts.items()))


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
        persisted_failed_genotypes: Optional[Dict[str, dict]] = None,
        on_population_evaluated: Optional[OnPopulationEvaluated] = None,
        on_new_best: Optional[OnNewBest] = None,
        on_generation_start: Optional[OnGenerationStart] = None,
    ):
        self.genome = genome
        self.strategy = strategy
        self.params = params
        self.random_init = random_init
        self.rs = np.random.RandomState(seed) if seed is not None else np.random
        self._persisted_failed_genotypes = dict(persisted_failed_genotypes or {})
        self.on_population_evaluated = on_population_evaluated
        self.on_new_best = on_new_best
        self.on_generation_start = on_generation_start
        self._last_eval_stats: Dict[str, Any] = {
            "evaluations": 0,
            "cache_hits": 0,
            "failed_evals": 0,
            "failure_reasons": "",
        }

        x0 = self.genome.initial_genotype(random=self.random_init).view(-1).numpy()
        self._baseline_genotype = x0.astype(np.float32)
        self.dim = x0.shape[0]

        self.pop_size = max(2, int(self.params.population_size))
        self.n_elite = max(1, int(self.params.elite_fraction * self.pop_size))

    def run(
        self, max_fevals: int, timeout: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        pop = self._init_population()
        fevals = 0
        start_time = time.time()
        best_x = pop[0].copy()
        best_score = -np.inf
        no_improve = 0
        self._fitness_cache: Dict[Tuple[int, ...], Tuple[float, dict]] = {}

        while fevals < max_fevals and (
            timeout is None or (time.time() - start_time) < timeout
        ):
            generation_idx = fevals // self.pop_size + 1
            # Ensure every generation evaluates a known-good linear baseline so we never regress
            pop[0] = self._baseline_genotype.copy()
            if self.on_generation_start:
                self.on_generation_start(
                    generation_idx,
                    fevals,
                    max_fevals,
                    self.pop_size,
                    float(best_score),
                )
            t0 = time.time()
            fitness, res_list = self._evaluate_population(pop)
            fevals += self.pop_size
            eval_seconds = time.time() - t0

            # Calculate generation statistics
            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = float(fitness[gen_best_idx])
            gen_mean = float(np.mean(fitness))
            gen_std = float(np.std(fitness))

            if self.on_population_evaluated:
                import datetime
                info = {
                    "eval_seconds": float(eval_seconds),
                    "mutation_sigma": float(self.params.mutation_sigma),
                    "gen_best": gen_best_score,
                    "gen_mean": gen_mean,
                    "gen_std": gen_std,
                    "best_so_far": float(best_score),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "generation": int(generation_idx),
                }
                info.update(self._last_eval_stats)
                self.on_population_evaluated(res_list, pop, fevals, info)
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_x = pop[gen_best_idx].copy()
                if self.on_new_best:
                    self.on_new_best(best_x, best_score, fevals)
                no_improve = 0
            else:
                no_improve += 1
                if self.params.patience and no_improve >= self.params.patience:
                    self.params.mutation_sigma = max(
                        self.params.min_mutation_sigma,
                        self.params.mutation_sigma * float(self.params.sigma_decay),
                    )
                    no_improve = 0

            # Elitism
            order = np.argsort(-fitness)
            elites = pop[order[: self.n_elite]].copy()

            # Create next generation
            next_pop = [*elites]
            # Inject random immigrants if requested
            n_imm = int(self.params.immigrant_fraction * self.pop_size)
            # Breed children
            while len(next_pop) < self.pop_size:
                p1 = pop[self._select_parent(fitness)]
                p2 = pop[self._select_parent(fitness)]
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_pop.append(child.astype(np.float32))
            # Replace tail with immigrants
            for i in range(n_imm):
                if len(next_pop) - 1 - i < self.n_elite:
                    break
                next_pop[-1 - i] = (
                    self.genome.initial_genotype(random=True)
                    .view(-1)
                    .numpy()
                    .astype(np.float32)
                )

            pop = np.stack(next_pop, axis=0)
            # Re-seed the baseline individual for the next generation's evaluation pass
            pop[0] = self._baseline_genotype.copy()

        return best_x, best_score

    # --- internals ---
    def _init_population(self) -> np.ndarray:
        pop = []
        x0_t = self.genome.initial_genotype(random=False)
        x0 = x0_t.view(-1).numpy()
        if not self.random_init:
            # seed with baseline equal-weights
            baseline = self._baseline_genotype.copy()
            pop.append(baseline)
            # seed with one-hot per model (weight channel index 0)
            n_layer_groups, n_models, n_param_sets, n_params = x0_t.shape
            for m in range(n_models):
                seed = x0_t.clone()
                # zero weights and set model m to 1
                seed[:, :, :, 0] = 0
                seed[:, m, :, 0] = 1
                pop.append(seed.view(-1).numpy().astype(np.float32))

        # fill remaining with random or noisy around x0
        while len(pop) < self.pop_size:
            if self.random_init:
                pop.append(
                    self.genome.initial_genotype(random=True)
                    .view(-1)
                    .numpy()
                    .astype(np.float32)
                )
            else:
                pop.append(
                    (x0 + self.rs.randn(self.dim).astype(np.float32) * 0.05).astype(
                        np.float32
                    )
                )
        return np.stack(pop[: self.pop_size], axis=0)

    def _evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        keys = [self._hash(ind) for ind in pop]
        to_eval = []
        to_eval_idx = []
        results: List[Optional[dict]] = [None] * len(pop)
        for i, k in enumerate(keys):
            cached = self._fitness_cache.get(k)
            if cached is not None:
                results[i] = cached[1]
                continue

            blacklisted = persisted_failure_result(
                self._persisted_failed_genotypes, pop[i]
            )
            if blacklisted is not None:
                self._fitness_cache[k] = (-np.inf, blacklisted)
                results[i] = blacklisted
                continue

            to_eval.append(pop[i])
            to_eval_idx.append(i)

        eval_results: List[dict] = []
        if to_eval:
            eval_results = list(self.strategy.evaluate_genotypes(to_eval))
            for i, res in zip(to_eval_idx, eval_results):
                score = res.get("score")
                cache_score = float(score) if score is not None else -np.inf
                self._fitness_cache[keys[i]] = (cache_score, res)
                results[i] = res

        # type: ignore
        results_final: List[dict] = [r for r in results]  # all filled
        fitness = np.array(
            [r["score"] if r["score"] is not None else -np.inf for r in results_final]
        )
        failures = sum(1 for r in results_final if r.get("score") is None)
        self._last_eval_stats = {
            "evaluations": float(len(to_eval)),
            "cache_hits": float(len(pop) - len(to_eval)),
            "failed_evals": float(failures),
            "failure_reasons": _summarize_failure_reasons(results_final),
        }
        return fitness, results_final

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
        if self.params.crossover == "sbx":
            # Simulated Binary Crossover for continuous params
            eta = 2.0
            u = self.rs.rand(a.size)
            beta = np.where(
                u <= 0.5,
                (2 * u) ** (1 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1 / (eta + 1)),
            )
            child = 0.5 * ((1 + beta) * a + (1 - beta) * b)
            return child.astype(np.float32)
        # arithmetic (default)
        alpha = self.rs.rand()
        return alpha * a + (1 - alpha) * b

    def _mutate(self, x: np.ndarray):
        rate = max(0.0, min(1.0, float(self.params.mutation_rate)))
        mask = self.rs.rand(x.size) < rate
        noise = self.rs.randn(x.size) * float(self.params.mutation_sigma)
        x[mask] += noise[mask]

    def _hash(self, x: np.ndarray) -> Tuple[int, ...]:
        return genotype_cache_key(x, self.params.cache_round)
