# Copyright (C) 2025 Nkululeko Thangelane
# Enhanced GA optimizer with semantic operations for multi-method genomes

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mergekit.evo.genome import ModelGenome
from mergekit.evo.multi_method_genome import MultiMethodGenome
from mergekit.evo.strategy import EvaluationStrategyBase

OnPopulationEvaluated = Callable[[List[dict], np.ndarray, int, Dict[str, float]], None]
OnNewBest = Callable[[np.ndarray, float, int], None]


@dataclass
class EnhancedGAParams:
    """Enhanced GA parameters supporting semantic operations."""
    population_size: int = 32
    elite_fraction: float = 0.125
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.05
    crossover: str = "semantic"  # "semantic", "arithmetic", "uniform", "sbx"
    tournament_size: int = 4
    
    # Semantic operation parameters
    method_mutation_rate: float = 0.05  # Chance to change merge method
    model_mutation_rate: float = 0.1    # Chance to modify model selection
    parameter_mutation_rate: float = 0.2 # Chance to modify parameters
    
    # Multi-method specific
    semantic_crossover_prob: float = 0.8  # Use semantic vs generic crossover
    
    # Original parameters
    cache_round: float = 1e-4
    immigrant_fraction: float = 0.0
    patience: int = 0
    sigma_decay: float = 0.5
    min_mutation_sigma: float = 0.005


class EnhancedGAOptimizer:
    """GA optimizer with semantic operations for evolving merge strategies."""

    def __init__(
        self,
        genome: Union[ModelGenome, MultiMethodGenome],
        strategy: EvaluationStrategyBase,
        params: EnhancedGAParams,
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

        # Determine genome type and capabilities
        self.is_multi_method = isinstance(genome, MultiMethodGenome)
        
        x0 = self.genome.initial_genotype(random=self.random_init)
        if isinstance(x0, torch.Tensor):
            x0 = x0.view(-1).numpy()
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
            t0 = time.time()
            fitness, res_list = self._evaluate_population(pop)
            fevals += self.pop_size
            eval_seconds = time.time() - t0

            if self.on_population_evaluated:
                info = {
                    "eval_seconds": float(eval_seconds),
                    "mutation_sigma": float(self.params.mutation_sigma),
                    "is_multi_method": self.is_multi_method,
                }
                self.on_population_evaluated(res_list, pop, fevals, info)

            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = float(fitness[gen_best_idx])
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
            
            # Breed children with enhanced operations
            while len(next_pop) < self.pop_size:
                p1 = pop[self._select_parent(fitness)]
                p2 = pop[self._select_parent(fitness)]
                child = self._enhanced_crossover(p1, p2)
                child = self._enhanced_mutate(child)
                next_pop.append(child.astype(np.float32))
                
            # Replace tail with immigrants
            for i in range(n_imm):
                if len(next_pop) - 1 - i < self.n_elite:
                    break
                x0 = self.genome.initial_genotype(random=True)
                if isinstance(x0, torch.Tensor):
                    x0 = x0.view(-1).numpy()
                next_pop[-1 - i] = x0.astype(np.float32)

            pop = np.stack(next_pop, axis=0)

        return best_x, best_score

    # --- Enhanced genetic operations ---
    
    def _enhanced_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Enhanced crossover with semantic awareness."""
        if (self.is_multi_method and 
            self.params.crossover == "semantic" and 
            self.rs.random() < self.params.semantic_crossover_prob):
            # Use semantic crossover for multi-method genomes
            a_torch = torch.from_numpy(a).float()
            b_torch = torch.from_numpy(b).float()
            child_torch = self.genome.crossover_semantic(a_torch, b_torch)
            return child_torch.numpy()
        else:
            # Fall back to traditional crossover
            return self._traditional_crossover(a, b)
            
    def _traditional_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Traditional crossover operations."""
        if self.params.crossover == "uniform":
            mask = self.rs.rand(a.size) < 0.5
            child = a.copy()
            child[mask] = b[mask]
            return child
        elif self.params.crossover == "sbx":
            # Simulated Binary Crossover
            eta = 2.0
            u = self.rs.rand(a.size)
            beta = np.where(
                u <= 0.5,
                (2 * u) ** (1 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1 / (eta + 1)),
            )
            child = 0.5 * ((1 + beta) * a + (1 - beta) * b)
            return child.astype(np.float32)
        else:  # arithmetic (default)
            alpha = self.rs.rand()
            return alpha * a + (1 - alpha) * b

    def _enhanced_mutate(self, x: np.ndarray) -> np.ndarray:
        """Enhanced mutation with semantic awareness."""
        if self.is_multi_method and self.params.crossover == "semantic":
            # Use semantic mutation for multi-method genomes
            x_torch = torch.from_numpy(x).float()
            mutated_torch = self.genome.mutate_semantic(
                x_torch, 
                mutation_rate=self.params.mutation_rate,
                mutation_sigma=self.params.mutation_sigma
            )
            return mutated_torch.numpy()
        else:
            # Traditional mutation
            return self._traditional_mutate(x)
            
    def _traditional_mutate(self, x: np.ndarray) -> np.ndarray:
        """Traditional Gaussian mutation."""
        x = x.copy()
        rate = max(0.0, min(1.0, float(self.params.mutation_rate)))
        mask = self.rs.rand(x.size) < rate
        noise = self.rs.randn(x.size) * float(self.params.mutation_sigma)
        x[mask] += noise[mask]
        return x

    # --- Population initialization ---
    
    def _init_population(self) -> np.ndarray:
        pop = []
        
        # Always start with a baseline individual
        x0 = self.genome.initial_genotype(random=False)
        if isinstance(x0, torch.Tensor):
            x0 = x0.view(-1).numpy()
        pop.append(x0.copy().astype(np.float32))
        
        if not self.random_init and hasattr(self.genome, 'models'):
            # Add model-specific seeds for traditional genomes
            if hasattr(self.genome, 'definition') and hasattr(self.genome.definition, 'models'):
                models = self.genome.definition.models
                x0_t = self.genome.initial_genotype(random=False)
                if len(x0_t.shape) == 4:  # Traditional genome format
                    n_layer_groups, n_models, n_param_sets, n_params = x0_t.shape
                    for m in range(min(n_models, self.pop_size - 1)):
                        seed = x0_t.clone()
                        seed[:, :, :, 0] = 0  # Zero all weights
                        seed[:, m, :, 0] = 1  # Set model m to 1
                        pop.append(seed.view(-1).numpy().astype(np.float32))

        # Fill remaining with random or noisy individuals
        while len(pop) < self.pop_size:
            if self.random_init:
                x_rand = self.genome.initial_genotype(random=True)
                if isinstance(x_rand, torch.Tensor):
                    x_rand = x_rand.view(-1).numpy()
                pop.append(x_rand.astype(np.float32))
            else:
                # Add noise to baseline
                noisy = x0 + self.rs.randn(self.dim).astype(np.float32) * 0.05
                pop.append(noisy.astype(np.float32))
                
        return np.stack(pop[: self.pop_size], axis=0)

    # --- Evaluation and selection (unchanged) ---
    
    def _evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        keys = [self._hash(ind) for ind in pop]
        to_eval = []
        to_eval_idx = []
        results: List[Optional[dict]] = [None] * len(pop)
        for i, k in enumerate(keys):
            cached = self._fitness_cache.get(k)
            if cached is not None:
                results[i] = cached[1]
            else:
                to_eval.append(pop[i])
                to_eval_idx.append(i)

        eval_results: List[dict] = []
        if to_eval:
            eval_results = list(self.strategy.evaluate_genotypes(to_eval))
            for i, res in zip(to_eval_idx, eval_results):
                self._fitness_cache[keys[i]] = (float(res.get("score") or -np.inf), res)
                results[i] = res

        results_final: List[dict] = [r for r in results]
        fitness = np.array(
            [r["score"] if r["score"] is not None else -np.inf for r in results_final]
        )
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

    def _hash(self, x: np.ndarray) -> Tuple[int, ...]:
        r = max(self.params.cache_round, 1e-9)
        q = np.round(x / r).astype(np.int64)
        return tuple(q.tolist())


# Backwards compatibility - use enhanced optimizer with traditional parameters
class GAOptimizer(EnhancedGAOptimizer):
    """Backwards compatible GA optimizer."""
    
    def __init__(self, genome, strategy, params, **kwargs):
        # Convert old GAParams to EnhancedGAParams
        if hasattr(params, '__dict__'):
            enhanced_params = EnhancedGAParams()
            for key, value in params.__dict__.items():
                if hasattr(enhanced_params, key):
                    setattr(enhanced_params, key, value)
        else:
            enhanced_params = params
            
        super().__init__(genome, strategy, enhanced_params, **kwargs)