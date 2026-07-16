import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from mergekit.evo.cache_utils import genotype_cache_key, persisted_failure_result

OnPopulationEvaluated = Callable[[List[dict], np.ndarray, int, Dict[str, Any]], None]
OnNewBest = Callable[[np.ndarray, float, int], None]
OnGenerationStart = Callable[[int, int, int, int, float], None]


def _failure_summary(results: List[dict]) -> str:
    counts: Dict[str, int] = {}
    for result in results:
        if result.get("score") is not None:
            continue
        key = (
            f"{result.get('error_stage') or 'unknown'}:"
            f"{result.get('error_type') or 'unknown'}"
        )
        counts[key] = counts.get(key, 0) + 1
    return ";".join(f"{key}:{count}" for key, count in sorted(counts.items()))


class RandomSearchOptimizer:
    """Uniform random baseline using the production evaluation strategy."""

    def __init__(
        self,
        *,
        genome,
        strategy,
        num_samples: int,
        seed: int,
        population_size: Optional[int] = None,
        persisted_failed_genotypes: Optional[Dict[str, dict]] = None,
        on_population_evaluated: Optional[OnPopulationEvaluated] = None,
        on_new_best: Optional[OnNewBest] = None,
        on_generation_start: Optional[OnGenerationStart] = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.genome = genome
        self.strategy = strategy
        self.num_samples = int(num_samples)
        self.population_size = max(
            1, min(int(population_size or num_samples), self.num_samples)
        )
        self.seed = int(seed)
        self._persisted_failed_genotypes = dict(persisted_failed_genotypes or {})
        self.on_population_evaluated = on_population_evaluated
        self.on_new_best = on_new_best
        self.on_generation_start = on_generation_start
        self.last_stop_details: Optional[Dict[str, Any]] = None
        self._fitness_cache: Dict[Tuple[int, ...], dict] = {}

    def sample_population(self) -> np.ndarray:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                samples = []
                for _ in range(self.num_samples):
                    genotype = self.genome.initial_genotype(random=True)
                    if isinstance(genotype, torch.Tensor):
                        genotype = genotype.detach().cpu().reshape(-1).numpy()
                    samples.append(np.asarray(genotype, dtype=np.float32).reshape(-1))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return np.stack(samples, axis=0)

    def _evaluate_population(self, population: np.ndarray) -> Tuple[List[dict], int]:
        results: List[Optional[dict]] = [None] * len(population)
        pending: List[np.ndarray] = []
        pending_indices: List[int] = []

        for index, genotype in enumerate(population):
            key = genotype_cache_key(genotype, 1e-4)
            if key in self._fitness_cache:
                results[index] = self._fitness_cache[key]
                continue
            failed = persisted_failure_result(
                self._persisted_failed_genotypes, genotype
            )
            if failed is not None:
                self._fitness_cache[key] = failed
                results[index] = failed
                continue
            pending.append(genotype)
            pending_indices.append(index)

        if pending:
            evaluated = self.strategy.evaluate_genotypes(pending)
            for index, result in zip(pending_indices, evaluated):
                key = genotype_cache_key(population[index], 1e-4)
                self._fitness_cache[key] = result
                results[index] = result

        complete = [result for result in results if result is not None]
        if len(complete) != len(population):  # pragma: no cover - strategy contract
            raise RuntimeError("Evaluation strategy returned an incomplete result set")
        return complete, len(pending)

    def run(
        self, max_fevals: Optional[int] = None, timeout: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        del max_fevals
        start_time = time.time()
        population = self.sample_population()
        best_x = population[0].copy()
        best_score = float("-inf")
        best_generation = None
        best_score_source = None
        completed = 0
        generation = 0
        stop_reason = "random_search_complete"

        while completed < self.num_samples:
            elapsed = time.time() - start_time
            if timeout is not None and elapsed >= timeout:
                stop_reason = "timeout"
                break
            generation += 1
            batch = population[
                completed : min(completed + self.population_size, self.num_samples)
            ]
            if self.on_generation_start:
                self.on_generation_start(
                    generation,
                    completed,
                    self.num_samples,
                    len(batch),
                    best_score,
                )
            eval_start = time.time()
            results, evaluations = self._evaluate_population(batch)
            eval_seconds = time.time() - eval_start
            completed += len(batch)
            scores = np.array(
                [
                    (
                        float(result["score"])
                        if result.get("score") is not None
                        else float("-inf")
                    )
                    for result in results
                ],
                dtype=np.float64,
            )
            finite_scores = scores[np.isfinite(scores)]
            batch_best_index = int(np.argmax(scores))
            batch_best_score = float(scores[batch_best_index])

            if self.on_population_evaluated:
                self.on_population_evaluated(
                    results,
                    batch,
                    completed,
                    {
                        "generation": generation,
                        "eval_seconds": float(eval_seconds),
                        "mutation_sigma": 0.0,
                        "gen_best": (
                            batch_best_score if np.isfinite(batch_best_score) else None
                        ),
                        "gen_mean": (
                            float(np.mean(finite_scores))
                            if finite_scores.size
                            else None
                        ),
                        "gen_std": (
                            float(np.std(finite_scores)) if finite_scores.size else None
                        ),
                        "best_so_far": (
                            best_score if np.isfinite(best_score) else None
                        ),
                        "evaluations": float(evaluations),
                        "cache_hits": float(len(batch) - evaluations),
                        "failed_evals": float(
                            sum(result.get("score") is None for result in results)
                        ),
                        "failure_reasons": _failure_summary(results),
                        "crossover_children": 0.0,
                        "crossover_type": "none",
                        "immigrants": float(len(batch)),
                        "role_counts": {"random_search": len(batch)},
                        "population_metadata": [
                            {"origin": "random_search", "role": "random_search"}
                            for _ in range(len(batch))
                        ],
                    },
                )

            if batch_best_score > best_score:
                best_x = batch[batch_best_index].copy()
                best_score = batch_best_score
                best_generation = generation
                best_score_source = results[batch_best_index].get("score_source")
                if np.isfinite(best_score) and self.on_new_best:
                    self.on_new_best(best_x, best_score, completed)

        elapsed = time.time() - start_time

        self.last_stop_details = {
            "reason": stop_reason,
            "generation": generation,
            "fevals": completed,
            "elapsed_seconds": float(elapsed),
            "best_score": best_score if np.isfinite(best_score) else None,
            "best_generation": best_generation,
            "best_score_source": best_score_source,
            "max_fevals": self.num_samples,
            "timeout_seconds": float(timeout) if timeout is not None else None,
        }
        return best_x, best_score
