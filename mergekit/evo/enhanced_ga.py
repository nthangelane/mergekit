# Copyright (C) 2025 Nkululeko Thangelane
# Enhanced GA optimizer with semantic operations for multi-method genomes

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mergekit.evo.cache_utils import genotype_cache_key, persisted_failure_result
from mergekit.evo.genome import ModelGenome
from mergekit.evo.multi_method_genome import MultiMethodGenome
from mergekit.evo.ranking import weighted_rank_scores
from mergekit.evo.stop_policy import (
    StopDetails,
    evaluate_stagnation_stop,
    evaluate_target_stop,
)
from mergekit.evo.strategy import EvaluationStrategyBase

OnPopulationEvaluated = Callable[[List[dict], np.ndarray, int, Dict[str, Any]], None]
OnNewBest = Callable[[np.ndarray, float, int], None]
OnGenerationStart = Callable[[int, int, int, int, float], None]


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
    model_mutation_rate: float = 0.1  # Chance to modify model selection
    parameter_mutation_rate: float = 0.2  # Chance to modify parameters

    # Multi-method specific
    semantic_crossover_prob: float = 0.8  # Use semantic vs generic crossover
    adaptive_method_sampling: bool = False
    initial_method_probs: Optional[Dict[str, float]] = None
    operator_temperature: float = 1.0
    operator_update_smoothing: float = 0.2
    operator_avg_child_weight: float = 0.5
    operator_parent_improvement_weight: float = 0.3
    operator_survival_weight: float = 0.2
    passthrough_penalty: float = 0.0
    passthrough_max_fraction: float = 1.0
    explorer_fraction: float = 0.0
    diversity_parent_selection: bool = False
    diversity_parent_weight: float = 0.0
    gene_diversity_bonus_weight: float = 0.0
    behavior_diversity_bonus_weight: float = 0.0
    archive_novelty_bonus_weight: float = 0.0
    novelty_archive_size: int = 64
    fitness_mode: str = "weighted_sum"
    rank_objective_weights: Optional[Dict[str, float]] = None
    duplicate_retry_limit: int = 3

    # Original parameters
    cache_round: float = 1e-4
    immigrant_fraction: float = 0.0
    patience: int = 0
    sigma_decay: float = 0.5
    min_mutation_sigma: float = 0.005
    target_improvement_abs: Optional[float] = None
    target_improvement_pct: Optional[float] = None
    target_reference: str = "best_baseline"
    target_reference_score: Optional[float] = None
    min_generations_before_target_stop: int = 0
    require_stage2_for_target: bool = False
    stagnation_patience_generations: int = 0
    stagnation_min_delta: float = 0.0


class EnhancedGAOptimizer:
    """GA optimizer with semantic operations for evolving merge strategies."""

    def __init__(
        self,
        genome: Union[ModelGenome, MultiMethodGenome],
        strategy: EvaluationStrategyBase,
        params: EnhancedGAParams,
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

        # Determine genome type and capabilities
        self.is_multi_method = isinstance(genome, MultiMethodGenome)
        self._last_eval_stats: Dict[str, Any] = {
            "evaluations": 0,
            "cache_hits": 0,
            "failed_evals": 0,
            "failure_reasons": "",
        }
        self._prev_generation_breeding: Dict[str, Any] = {
            "crossover_children": 0,
            "crossover_type": getattr(self.params, "crossover", "arithmetic"),
            "immigrants": 0,
        }
        self._novelty_archive: List[np.ndarray] = []
        self._fitness_cache: Dict[Tuple[int, ...], Tuple[float, dict]] = {}

        x0 = self.genome.initial_genotype(random=self.random_init)
        if isinstance(x0, torch.Tensor):
            x0 = x0.view(-1).numpy()
        self.dim = x0.shape[0]

        self.pop_size = max(2, int(self.params.population_size))
        self.n_elite = max(1, int(self.params.elite_fraction * self.pop_size))
        self._configured_methods: List[str] = (
            list(self.genome.definition.allowed_methods)
            if self.is_multi_method and hasattr(self.genome, "definition")
            else []
        )
        self._use_method_probabilities = bool(
            self.is_multi_method
            and self._configured_methods
            and (
                bool(getattr(self.params, "adaptive_method_sampling", False))
                or getattr(self.params, "initial_method_probs", None) is not None
                or float(getattr(self.params, "passthrough_max_fraction", 1.0)) < 1.0
                or float(getattr(self.params, "explorer_fraction", 0.0)) > 0.0
            )
        )
        self._method_probs: Dict[str, float] = self._initialize_method_probs()
        self._population_metadata: List[Dict[str, Any]] = []
        self._last_operator_summary: Dict[str, Any] = {}
        self.last_stop_details: Optional[Dict[str, Any]] = None

    def _initialize_method_probs(self) -> Dict[str, float]:
        if not self._configured_methods:
            return {}

        configured = list(self._configured_methods)
        initial = getattr(self.params, "initial_method_probs", None) or {}
        if initial:
            probs = {
                method: max(0.0, float(initial.get(method, 0.0)))
                for method in configured
            }
            if sum(probs.values()) <= 0:
                probs = {method: 1.0 for method in configured}
        else:
            probs = {method: 1.0 for method in configured}
        return self._normalize_probabilities(probs)

    def _normalize_probabilities(
        self, probabilities: Dict[str, float]
    ) -> Dict[str, float]:
        if not probabilities:
            return {}

        cleaned = {
            str(method): max(0.0, float(prob)) for method, prob in probabilities.items()
        }
        total = float(sum(cleaned.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(cleaned))
            return {method: uniform for method in cleaned}
        return {method: value / total for method, value in cleaned.items()}

    def _softmax_method_scores(
        self, method_scores: Dict[str, float]
    ) -> Dict[str, float]:
        if not method_scores:
            return {}

        temperature = max(1e-6, float(self.params.operator_temperature))
        methods = list(method_scores.keys())
        values = np.array(
            [float(method_scores[method]) for method in methods], dtype=np.float64
        )
        scaled = values / temperature
        scaled -= np.max(scaled)
        exp_values = np.exp(scaled)
        denom = float(exp_values.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            return self._normalize_probabilities({method: 1.0 for method in methods})
        return {
            method: float(exp_values[idx] / denom) for idx, method in enumerate(methods)
        }

    def _method_name_for_genotype(self, genotype: np.ndarray) -> str:
        if (
            self.is_multi_method
            and hasattr(self.genome, "method_dim")
            and getattr(self.genome, "method_dim", 0) > 0
        ):
            if hasattr(self.genome, "method_label_for_genotype"):
                return str(self.genome.method_label_for_genotype(genotype))
            flat = np.asarray(genotype, dtype=np.float32).reshape(-1)
            return self.genome.method_name_from_gene_value(float(flat[0]))

        try:
            cfg = (
                self.genome.genotype_to_merge_config(genotype)
                if hasattr(self.genome, "genotype_to_merge_config")
                else self.genome.genotype_merge_config(genotype)
            )
            return str(getattr(cfg, "merge_method", None) or "unknown")
        except Exception:
            return "unknown"

    def _seed_population_metadata(self, pop: np.ndarray) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for individual in pop:
            actual_method = (
                self._method_name_for_genotype(individual)
                if self.is_multi_method
                else None
            )
            metadata.append(
                {
                    "origin": "seed",
                    "generation_born": 0,
                    "parent_indices": [],
                    "parent_scores": [],
                    "sampled_method": (
                        actual_method
                        if actual_method in self._configured_methods
                        else None
                    ),
                    "actual_method": actual_method,
                }
            )
        return metadata

    def _sample_target_method(
        self,
        sampled_method_counts: Counter[str],
        target_children: int,
        exploration: bool = False,
    ) -> Optional[str]:
        if not self._use_method_probabilities or not self._method_probs:
            return None

        probabilities = dict(self._method_probs)
        passthrough_fraction = float(self.params.passthrough_max_fraction)
        if "passthrough" in probabilities and passthrough_fraction < 1.0:
            max_passthrough = int(
                np.floor(passthrough_fraction * float(target_children))
            )
            if passthrough_fraction > 0.0 and max_passthrough == 0:
                max_passthrough = 1
            if sampled_method_counts.get("passthrough", 0) >= max_passthrough:
                probabilities["passthrough"] = 0.0

        if exploration:
            max_prob = max(probabilities.values()) if probabilities else 0.0
            probabilities = {
                method: max(0.0, (max_prob - prob) + 1e-6)
                for method, prob in probabilities.items()
            }
            non_passthrough = [
                method for method in probabilities if method != "passthrough"
            ]
            if non_passthrough:
                probabilities["passthrough"] = 0.0

        if sum(probabilities.values()) <= 0.0:
            probabilities = {
                method: 1.0
                for method in self._configured_methods
                if method != "passthrough"
            }
            if not probabilities:
                probabilities = dict(self._method_probs)

        normalized = self._normalize_probabilities(probabilities)
        methods = list(normalized.keys())
        weights = np.array([normalized[method] for method in methods], dtype=np.float64)
        return str(self.rs.choice(methods, p=weights))

    def _apply_sampled_method(
        self, child: np.ndarray, method_name: Optional[str]
    ) -> np.ndarray:
        if (
            not method_name
            or not self.is_multi_method
            or getattr(self.genome, "method_dim", 0) <= 0
        ):
            return child.astype(np.float32)

        updated = np.asarray(child, dtype=np.float32).copy()
        method_gene_value = self.genome.method_gene_value(method_name)
        sampled_params = self.genome.sample_parameters_for_method(
            method_name, rs=self.rs
        )

        for layer_idx in range(getattr(self.genome, "num_layer_groups", 1)):
            offset = layer_idx * self.genome.layer_group_dim
            updated[offset] = method_gene_value

            model_start = offset + self.genome.method_dim
            model_end = model_start + self.genome.model_selection_dim
            if self.genome.model_selection_dim > 0:
                projected = self.genome.project_model_selection_for_method(
                    updated[model_start:model_end], method_name
                )
                updated[model_start:model_end] = projected

            param_start = model_end
            updated[param_start : param_start + self.genome.param_dim] = 0.0
            if sampled_params.size > 0:
                updated[param_start : param_start + sampled_params.size] = (
                    sampled_params
                )

        return updated.astype(np.float32)

    def _select_parent_pair(
        self,
        fitness: np.ndarray,
        pop: np.ndarray,
        method_name: Optional[str] = None,
    ) -> Tuple[int, int]:
        first = self._select_parent(fitness)
        if self.params.diversity_parent_selection:
            second = self._select_diverse_parent(fitness, pop, first)
        else:
            second = self._select_parent(fitness)
        retries = 0
        while (
            self.pop_size > 1
            and second == first
            and method_name != "passthrough"
            and retries < 4
        ):
            second = (
                self._select_diverse_parent(fitness, pop, first)
                if self.params.diversity_parent_selection
                else self._select_parent(fitness)
            )
            retries += 1
        return first, second

    def _select_diverse_parent(
        self, fitness: np.ndarray, pop: np.ndarray, anchor_idx: int
    ) -> int:
        k = max(2, int(self.params.tournament_size))
        idxs = self.rs.randint(0, self.pop_size, size=k)
        best_idx = idxs[0]
        best_score = float("-inf")
        anchor = np.asarray(pop[anchor_idx], dtype=np.float32).reshape(-1)
        denom = max(1.0, float(np.sqrt(anchor.size)))

        for idx in idxs:
            candidate = np.asarray(pop[idx], dtype=np.float32).reshape(-1)
            diversity = float(np.linalg.norm(candidate - anchor) / denom)
            combined = float(fitness[idx]) + (
                float(self.params.diversity_parent_weight) * diversity
            )
            if combined > best_score:
                best_idx = int(idx)
                best_score = combined
        return best_idx

    def _adjust_result_score(self, result: dict, genotype: np.ndarray) -> dict:
        score = result.get("score")
        if score is None:
            return result

        adjusted_score = float(score)
        fitness_components = dict(result.get("fitness_components") or {})
        gene_diversity_score = self._intrinsic_gene_diversity_score(genotype)
        fitness_components["gene_diversity_score"] = gene_diversity_score
        if float(self.params.gene_diversity_bonus_weight) > 0.0:
            gene_bonus = (
                float(self.params.gene_diversity_bonus_weight) * gene_diversity_score
            )
            adjusted_score += gene_bonus
            fitness_components["gene_diversity_bonus"] = gene_bonus

        behavior_probe = result.get("behavior_probe") or {}
        behavior_diversity_score = float(
            behavior_probe.get(
                "behavior_diversity_score",
                fitness_components.get("behavior_diversity_score", 0.0),
            )
        )
        fitness_components["behavior_diversity_score"] = behavior_diversity_score
        if float(self.params.behavior_diversity_bonus_weight) > 0.0:
            behavior_bonus = (
                float(self.params.behavior_diversity_bonus_weight)
                * behavior_diversity_score
            )
            adjusted_score += behavior_bonus
            fitness_components["behavior_diversity_bonus"] = behavior_bonus

        archive_novelty_score = self._archive_novelty_score(genotype)
        fitness_components["archive_novelty_score"] = archive_novelty_score
        if float(getattr(self.params, "archive_novelty_bonus_weight", 0.0)) > 0.0:
            novelty_bonus = (
                float(self.params.archive_novelty_bonus_weight) * archive_novelty_score
            )
            adjusted_score += novelty_bonus
            fitness_components["archive_novelty_bonus"] = novelty_bonus

        method_name = self._method_name_for_genotype(genotype)
        if (
            method_name == "passthrough"
            and float(self.params.passthrough_penalty) > 0.0
        ):
            penalty = float(self.params.passthrough_penalty)
            adjusted_score -= penalty
            fitness_components["passthrough_penalty"] = penalty

        if adjusted_score == float(score) and not fitness_components:
            return result

        updated = dict(result)
        updated["raw_score"] = float(score)
        updated["score"] = adjusted_score
        if fitness_components:
            updated["fitness_components"] = fitness_components
            objectives = dict(result.get("fitness_objectives") or {})
            objectives.update(
                {
                    key: value
                    for key, value in fitness_components.items()
                    if key.endswith("_score")
                }
            )
            updated["fitness_objectives"] = objectives
        return updated

    def _intrinsic_gene_diversity_score(self, genotype: np.ndarray) -> float:
        flat = np.asarray(genotype, dtype=np.float32).reshape(-1)
        if not self.is_multi_method:
            denom = max(1.0, float(np.linalg.norm(flat)))
            return float(min(1.0, np.std(flat) / denom))

        layer_group_dim = int(getattr(self.genome, "layer_group_dim", flat.size))
        model_selection_dim = int(getattr(self.genome, "model_selection_dim", 0))
        method_dim = int(getattr(self.genome, "method_dim", 0))
        num_layer_groups = int(getattr(self.genome, "num_layer_groups", 1))
        entropy_scores: List[float] = []
        method_labels: List[str] = []

        for layer_idx in range(num_layer_groups):
            offset = layer_idx * layer_group_dim
            if method_dim > 0:
                method_labels.append(
                    str(self.genome.method_name_from_gene_value(float(flat[offset])))
                )
            if model_selection_dim <= 0:
                continue
            model_start = offset + method_dim
            model_end = model_start + model_selection_dim
            weights = np.clip(flat[model_start:model_end], 0.0, None)
            total = float(weights.sum())
            if total <= 1e-8:
                entropy_scores.append(0.0)
                continue
            probs = weights / total
            entropy = -float(
                np.sum([p * np.log(max(float(p), 1e-8)) for p in probs if p > 0.0])
            )
            denom = max(np.log(max(len(probs), 2)), 1e-8)
            entropy_scores.append(float(entropy / denom))

        mean_entropy = float(np.mean(entropy_scores)) if entropy_scores else 0.0
        method_diversity = 0.0
        if method_labels:
            unique_methods = len(set(method_labels))
            max_unique = max(len(set(self._configured_methods)), 1)
            method_diversity = float(unique_methods - 1) / float(max(max_unique - 1, 1))
        return float(min(1.0, (0.7 * mean_entropy) + (0.3 * method_diversity)))

    def _archive_novelty_score(self, genotype: np.ndarray) -> float:
        if not self._novelty_archive:
            return 0.0
        flat = np.asarray(genotype, dtype=np.float32).reshape(-1)
        flat_norm = float(np.linalg.norm(flat))
        distances = []
        for archived in self._novelty_archive:
            denom = max(flat_norm, float(np.linalg.norm(archived)), 1e-8)
            distance = float(np.linalg.norm(flat - archived) / denom)
            distances.append(distance)
        if not distances:
            return 0.0
        return float(min(1.0, max(0.0, min(distances))))

    def _update_novelty_archive(
        self,
        pop: np.ndarray,
        results: List[dict],
        elite_indices: Optional[np.ndarray] = None,
    ) -> None:
        max_archive = max(1, int(getattr(self.params, "novelty_archive_size", 64)))
        survivor_set = (
            {int(idx) for idx in elite_indices.tolist()}
            if elite_indices is not None
            else None
        )
        additions: List[np.ndarray] = []
        for idx, (genotype, result) in enumerate(zip(pop, results)):
            if result.get("score") is None:
                continue
            if survivor_set is not None and idx not in survivor_set:
                continue
            additions.append(np.asarray(genotype, dtype=np.float32).reshape(-1).copy())
        if not additions:
            return
        self._novelty_archive.extend(additions)
        if len(self._novelty_archive) > max_archive:
            self._novelty_archive = self._novelty_archive[-max_archive:]

    def _rank_population_fitness(
        self, results: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        objective_weights = getattr(self.params, "rank_objective_weights", None)
        ranked_fitness, rank_details = weighted_rank_scores(
            results,
            objective_weights=objective_weights,
        )
        enriched_results: List[dict] = []
        for result, detail in zip(results, rank_details):
            if not detail:
                enriched_results.append(result)
                continue
            updated = dict(result)
            updated["raw_score"] = float(result.get("score"))
            updated["score"] = float(detail["weighted_rank_score"])
            updated["fitness_proxy"] = detail["fitness_proxy"]
            updated["rank_details"] = {
                "objective_values": dict(detail.get("objective_values") or {}),
                "objective_rank_scores": dict(
                    detail.get("objective_rank_scores") or {}
                ),
            }
            enriched_results.append(updated)
        return ranked_fitness, enriched_results

    def _update_operator_state(
        self,
        pop: np.ndarray,
        results: List[dict],
        fitness: np.ndarray,
        elite_indices: np.ndarray,
    ) -> Dict[str, Any]:
        if not self._configured_methods:
            return {}

        survivor_set = {int(idx) for idx in elite_indices.tolist()}
        old_probs = dict(self._method_probs)
        methods = list(self._configured_methods)
        per_method: Dict[str, Dict[str, Any]] = {
            method: {
                "count": 0,
                "success_count": 0,
                "failure_count": 0,
                "survivor_count": 0,
                "score_values": [],
                "improvement_count": 0,
                "improvement_denominator": 0,
            }
            for method in methods
        }

        for idx, (individual, result) in enumerate(zip(pop, results)):
            meta = (
                self._population_metadata[idx]
                if idx < len(self._population_metadata)
                else {}
            )
            sampled_method = meta.get("sampled_method")
            actual_method = meta.get("actual_method") or self._method_name_for_genotype(
                individual
            )
            method_name: Optional[str] = None
            if sampled_method in self._configured_methods:
                method_name = str(sampled_method)
            elif actual_method in self._configured_methods:
                method_name = str(actual_method)

            if method_name is None:
                continue

            stats = per_method[method_name]
            stats["count"] += 1
            if idx in survivor_set:
                stats["survivor_count"] += 1

            score = result.get("score")
            if score is None:
                stats["failure_count"] += 1
                continue

            score_value = float(score)
            stats["success_count"] += 1
            stats["score_values"].append(score_value)

            parent_scores = [
                float(parent_score)
                for parent_score in meta.get("parent_scores", [])
                if parent_score is not None and np.isfinite(parent_score)
            ]
            if parent_scores:
                stats["improvement_denominator"] += 1
                if score_value > max(parent_scores):
                    stats["improvement_count"] += 1

        mean_scores = {
            method: (
                float(np.mean(values["score_values"]))
                if values["score_values"]
                else None
            )
            for method, values in per_method.items()
        }
        finite_means = [
            float(mean_score)
            for mean_score in mean_scores.values()
            if mean_score is not None and np.isfinite(mean_score)
        ]
        normalized_means: Dict[str, float] = {}
        if finite_means:
            min_mean = min(finite_means)
            max_mean = max(finite_means)
            for method, mean_score in mean_scores.items():
                if mean_score is None or not np.isfinite(float(mean_score)):
                    normalized_means[method] = 0.0
                elif max_mean == min_mean:
                    normalized_means[method] = 1.0
                else:
                    normalized_means[method] = float(
                        (float(mean_score) - min_mean) / (max_mean - min_mean)
                    )
        else:
            normalized_means = {method: 0.0 for method in per_method}

        method_scores: Dict[str, float] = {}
        history_rows: List[Dict[str, Union[str, float, int]]] = []
        metrics: Dict[str, float] = {}
        for method in methods:
            stats = per_method[method]
            count = int(stats["count"])
            successes = int(stats["success_count"])
            failures = int(stats["failure_count"])
            success_rate = float(successes / count) if count else 0.0
            survival_rate = float(stats["survivor_count"] / count) if count else 0.0
            improvement_denominator = int(stats["improvement_denominator"])
            parent_improvement_rate = (
                float(stats["improvement_count"] / improvement_denominator)
                if improvement_denominator
                else 0.0
            )
            mean_score = mean_scores.get(method)
            best_score = (
                float(max(stats["score_values"])) if stats["score_values"] else None
            )
            operator_score = (
                float(self.params.operator_avg_child_weight)
                * normalized_means.get(method, 0.0)
                + float(self.params.operator_parent_improvement_weight)
                * parent_improvement_rate
                + float(self.params.operator_survival_weight) * survival_rate
            )
            method_scores[method] = operator_score

            metrics[f"adaptive_method/{method}/parent_improvement_rate"] = (
                parent_improvement_rate
            )
            metrics[f"adaptive_method/{method}/survival_rate"] = survival_rate
            metrics[f"adaptive_method/{method}/operator_score"] = operator_score

            history_rows.append(
                {
                    "merge_method": method,
                    "count": count,
                    "success_count": successes,
                    "failure_count": failures,
                    "success_rate": success_rate,
                    "mean_score": mean_score,
                    "best_score": best_score,
                    "parent_improvement_rate": parent_improvement_rate,
                    "survival_rate": survival_rate,
                    "operator_score": operator_score,
                    "probability_before": float(old_probs.get(method, 0.0)),
                }
            )

        adapted_probs = self._softmax_method_scores(method_scores)
        if self.params.adaptive_method_sampling and adapted_probs:
            smoothing = float(self.params.operator_update_smoothing)
            blended = {}
            for method in methods:
                old_prob = float(old_probs.get(method, 0.0))
                adapted_prob = float(adapted_probs.get(method, 0.0))
                blended[method] = ((1.0 - smoothing) * old_prob) + (
                    smoothing * adapted_prob
                )
            new_probs = self._normalize_probabilities(blended)
        else:
            new_probs = old_probs or adapted_probs

        if new_probs:
            self._method_probs = new_probs

        for row in history_rows:
            method = str(row["merge_method"])
            row["probability_after"] = float(self._method_probs.get(method, 0.0))
            metrics[f"adaptive_method/{method}/probability_before"] = float(
                old_probs.get(method, 0.0)
            )
            metrics[f"adaptive_method/{method}/probability_after"] = float(
                self._method_probs.get(method, 0.0)
            )

        return {
            "metrics": metrics,
            "history_rows": history_rows,
            "method_scores": method_scores,
            "probabilities_before": old_probs,
            "probabilities_after": dict(self._method_probs),
        }

    def run(
        self, max_fevals: int, timeout: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        pop = self._init_population()
        self._population_metadata = self._seed_population_metadata(pop)
        fevals = 0
        start_time = time.time()
        best_x = pop[0].copy()
        best_score = -np.inf
        best_generation: Optional[int] = None
        best_score_source: Optional[str] = None
        no_improve = 0
        stagnation_generations = 0
        self._fitness_cache: Dict[Tuple[int, ...], Tuple[float, dict]] = {}
        effective_max_fevals = max(1, int(max_fevals))
        self.last_stop_details = None

        while True:
            elapsed_seconds = time.time() - start_time
            completed_generations = max(0, fevals // self.pop_size)
            if fevals >= effective_max_fevals:
                self.last_stop_details = StopDetails(
                    reason="max_fevals",
                    generation=completed_generations,
                    fevals=int(fevals),
                    elapsed_seconds=float(elapsed_seconds),
                    best_score=(float(best_score) if np.isfinite(best_score) else None),
                    best_generation=best_generation,
                    best_score_source=best_score_source,
                    max_fevals=effective_max_fevals,
                ).to_dict()
                break
            if timeout is not None and elapsed_seconds >= timeout:
                self.last_stop_details = StopDetails(
                    reason="timeout",
                    generation=completed_generations,
                    fevals=int(fevals),
                    elapsed_seconds=float(elapsed_seconds),
                    best_score=(float(best_score) if np.isfinite(best_score) else None),
                    best_generation=best_generation,
                    best_score_source=best_score_source,
                    timeout_seconds=float(timeout),
                ).to_dict()
                break

            generation_idx = fevals // self.pop_size + 1
            if self.on_generation_start:
                self.on_generation_start(
                    generation_idx,
                    fevals,
                    effective_max_fevals,
                    self.pop_size,
                    float(best_score),
                )
            t0 = time.time()
            fitness, res_list = self._evaluate_population(pop)
            fevals += self.pop_size
            eval_seconds = time.time() - t0
            order = np.argsort(-fitness)
            elite_indices = order[: self.n_elite]
            self._last_operator_summary = self._update_operator_state(
                pop, res_list, fitness, elite_indices
            )

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
                    "is_multi_method": self.is_multi_method,
                    "gen_best": gen_best_score,
                    "gen_mean": gen_mean,
                    "gen_std": gen_std,
                    "best_so_far": float(best_score),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "generation": int(generation_idx),
                    "adaptive_method_sampling": float(
                        1.0 if self.params.adaptive_method_sampling else 0.0
                    ),
                    "method_probabilities": dict(self._method_probs),
                    "operator_summary": self._last_operator_summary,
                    "population_metadata": [
                        dict(meta) for meta in self._population_metadata
                    ],
                }
                info.update(self._last_eval_stats)
                info.update(self._prev_generation_breeding)
                self.on_population_evaluated(res_list, pop, fevals, info)

            previous_best = float(best_score)
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_x = pop[gen_best_idx].copy()
                best_generation = generation_idx
                best_score_source = res_list[gen_best_idx].get("score_source")
                if self.on_new_best:
                    self.on_new_best(best_x, best_score, fevals)
                no_improve = 0
                if np.isfinite(previous_best):
                    improvement_delta = gen_best_score - previous_best
                    if improvement_delta <= float(self.params.stagnation_min_delta):
                        stagnation_generations += 1
                    else:
                        stagnation_generations = 0
                else:
                    stagnation_generations = 0
            else:
                no_improve += 1
                stagnation_generations += 1
                if self.params.patience and no_improve >= self.params.patience:
                    self.params.mutation_sigma = max(
                        self.params.min_mutation_sigma,
                        self.params.mutation_sigma * float(self.params.sigma_decay),
                    )
                    no_improve = 0

            target_stop = evaluate_target_stop(
                best_score=float(best_score),
                best_score_source=best_score_source,
                generation=generation_idx,
                fevals=fevals,
                elapsed_seconds=time.time() - start_time,
                best_generation=best_generation,
                policy=self.params,
            )
            if target_stop is not None:
                self.last_stop_details = target_stop.to_dict()
                break

            stagnation_stop = evaluate_stagnation_stop(
                best_score=float(best_score),
                best_score_source=best_score_source,
                generation=generation_idx,
                fevals=fevals,
                elapsed_seconds=time.time() - start_time,
                best_generation=best_generation,
                stagnation_generations=stagnation_generations,
                policy=self.params,
            )
            if stagnation_stop is not None:
                self.last_stop_details = stagnation_stop.to_dict()
                break

            # Elitism
            elites = pop[order[: self.n_elite]].copy()

            # Create next generation
            next_pop = [*elites]
            next_meta = [
                {
                    **dict(self._population_metadata[int(idx)]),
                    "role": "elite",
                }
                for idx in order[: self.n_elite]
            ]
            # Inject random immigrants if requested
            n_imm = int(self.params.immigrant_fraction * self.pop_size)
            n_explorer = int(self.params.explorer_fraction * self.pop_size)
            immigrants_added = 0

            # Breed children with enhanced operations
            crossover_children = 0
            sampled_method_counts: Counter[str] = Counter()
            role_counts: Counter[str] = Counter()
            role_counts["elite"] = len(next_pop)
            duplicate_resamples = 0
            generation_hashes = {self._hash(ind) for ind in next_pop}
            target_children = max(self.pop_size - len(next_pop), 0)
            exploiter_target = max(target_children - n_imm - n_explorer, 0)
            while len(next_pop) < self.pop_size:
                child_index = len(next_pop) - self.n_elite
                child_role = (
                    "explorer"
                    if child_index >= exploiter_target
                    and child_index < (exploiter_target + n_explorer)
                    else "exploiter"
                )
                sampled_method = self._sample_target_method(
                    sampled_method_counts,
                    target_children,
                    exploration=(child_role == "explorer"),
                )
                p1_idx, p2_idx = self._select_parent_pair(fitness, pop, sampled_method)
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]
                child = self._enhanced_crossover(p1, p2)
                child = self._enhanced_mutate(child)
                child = self._apply_sampled_method(child, sampled_method)
                child, retry_count = self._make_child_unique(
                    child,
                    existing_hashes=generation_hashes,
                    sampled_method=sampled_method,
                )
                duplicate_resamples += retry_count
                next_pop.append(child.astype(np.float32))
                generation_hashes.add(self._hash(child))
                actual_method = (
                    self._method_name_for_genotype(child)
                    if self.is_multi_method
                    else None
                )
                if sampled_method is not None:
                    sampled_method_counts[sampled_method] += 1
                role_counts[child_role] += 1
                next_meta.append(
                    {
                        "origin": "child",
                        "role": child_role,
                        "generation_born": generation_idx + 1,
                        "parent_indices": [int(p1_idx), int(p2_idx)],
                        "parent_scores": [
                            float(fitness[p1_idx]),
                            float(fitness[p2_idx]),
                        ],
                        "sampled_method": sampled_method or actual_method,
                        "actual_method": actual_method,
                    }
                )
                crossover_children += 1

            # Replace tail with immigrants
            for i in range(n_imm):
                if len(next_pop) - 1 - i < self.n_elite:
                    break
                x0 = self.genome.initial_genotype(random=True)
                if isinstance(x0, torch.Tensor):
                    x0 = x0.view(-1).numpy()
                next_pop[-1 - i] = x0.astype(np.float32)
                next_meta[-1 - i] = {
                    "origin": "immigrant",
                    "role": "immigrant",
                    "generation_born": generation_idx + 1,
                    "parent_indices": [],
                    "parent_scores": [],
                    "sampled_method": (
                        self._method_name_for_genotype(x0)
                        if self.is_multi_method
                        else None
                    ),
                    "actual_method": (
                        self._method_name_for_genotype(x0)
                        if self.is_multi_method
                        else None
                    ),
                }
                generation_hashes.add(self._hash(next_pop[-1 - i]))
                role_counts["immigrant"] += 1
                immigrants_added += 1

            pop = np.stack(next_pop, axis=0)
            self._population_metadata = next_meta
            self._prev_generation_breeding = {
                "crossover_children": float(crossover_children),
                "crossover_type": getattr(self.params, "crossover", "arithmetic"),
                "immigrants": float(immigrants_added),
                "sampled_method_counts": dict(sampled_method_counts),
                "role_counts": dict(role_counts),
                "duplicate_resamples": float(duplicate_resamples),
            }

        if self.last_stop_details is None:
            self.last_stop_details = StopDetails(
                reason="completed",
                generation=max(0, fevals // self.pop_size),
                fevals=int(fevals),
                elapsed_seconds=float(time.time() - start_time),
                best_score=float(best_score) if np.isfinite(best_score) else None,
                best_generation=best_generation,
                best_score_source=best_score_source,
                max_fevals=effective_max_fevals,
                timeout_seconds=float(timeout) if timeout is not None else None,
            ).to_dict()

        return best_x, best_score

    # --- Enhanced genetic operations ---

    def _enhanced_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Enhanced crossover with semantic awareness."""
        if (
            self.is_multi_method
            and self.params.crossover == "semantic"
            and self.rs.random() < self.params.semantic_crossover_prob
        ):
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
                mutation_sigma=self.params.mutation_sigma,
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

    def _make_child_unique(
        self,
        child: np.ndarray,
        *,
        existing_hashes: set[Tuple[int, ...]],
        sampled_method: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        candidate = np.asarray(child, dtype=np.float32).copy()
        retries = 0
        max_retries = max(0, int(getattr(self.params, "duplicate_retry_limit", 0)))

        while retries < max_retries and self._hash(candidate) in existing_hashes:
            candidate = self._enhanced_mutate(candidate)
            if sampled_method:
                candidate = self._apply_sampled_method(candidate, sampled_method)
            retries += 1

        return candidate.astype(np.float32), retries

    # --- Population initialization ---

    def _init_population(self) -> np.ndarray:
        pop = []

        # Always start with a baseline individual
        x0 = self.genome.initial_genotype(random=False)
        if isinstance(x0, torch.Tensor):
            x0 = x0.view(-1).numpy()
        pop.append(x0.copy().astype(np.float32))

        if (
            not self.random_init
            and self.is_multi_method
            and hasattr(self.genome, "definition")
            and "passthrough" in self.genome.definition.allowed_methods
            and getattr(self.genome, "method_dim", 0) > 0
            and getattr(self.genome, "model_selection_dim", 0) > 0
        ):
            passthrough_idx = self.genome.definition.allowed_methods.index(
                "passthrough"
            )
            model_start = self.genome.method_dim
            model_end = model_start + self.genome.model_selection_dim
            max_passthrough_seeds = min(
                self.genome.model_selection_dim,
                len(self.genome.definition.models),
                self.pop_size - len(pop),
            )
            passthrough_fraction = float(
                getattr(self.params, "passthrough_max_fraction", 1.0)
            )
            if passthrough_fraction < 1.0:
                allowed_passthrough = int(
                    np.floor(passthrough_fraction * float(self.pop_size))
                )
                if passthrough_fraction > 0.0 and allowed_passthrough == 0:
                    allowed_passthrough = 1
                existing_passthrough = (
                    1 if self._method_name_for_genotype(x0) == "passthrough" else 0
                )
                max_passthrough_seeds = min(
                    max_passthrough_seeds,
                    max(0, allowed_passthrough - existing_passthrough),
                )

            for model_idx in range(max_passthrough_seeds):
                seed = x0.copy()
                seed[0] = float(passthrough_idx)
                seed[model_start:model_end] = 0.0
                seed[model_start + model_idx] = 1.0
                pop.append(seed.astype(np.float32))

        if not self.random_init and hasattr(self.genome, "models"):
            # Add model-specific seeds for traditional genomes
            if hasattr(self.genome, "definition") and hasattr(
                self.genome.definition, "models"
            ):
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
                if self.is_multi_method and getattr(self.genome, "method_dim", 0) > 0:
                    for layer_idx in range(getattr(self.genome, "num_layer_groups", 1)):
                        offset = layer_idx * self.genome.layer_group_dim
                        noisy[offset] = x0[offset]
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
                adjusted_res = self._adjust_result_score(res, pop[i])
                score = adjusted_res.get("score")
                cache_score = float(score) if score is not None else -np.inf
                self._fitness_cache[keys[i]] = (cache_score, adjusted_res)
                results[i] = adjusted_res

        results_final: List[dict] = [r for r in results]
        fitness = np.array(
            [r["score"] if r["score"] is not None else -np.inf for r in results_final],
            dtype=np.float32,
        )
        if getattr(self.params, "fitness_mode", "weighted_sum") == "weighted_rank":
            fitness, results_final = self._rank_population_fitness(results_final)
        failures = sum(1 for r in results_final if r.get("score") is None)
        successful_indices = np.where(np.isfinite(fitness))[0]
        self._update_novelty_archive(pop, results_final, successful_indices)
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

    def _hash(self, x: np.ndarray) -> Tuple[int, ...]:
        return genotype_cache_key(x, self.params.cache_round)


# Backwards compatibility - use enhanced optimizer with traditional parameters
class GAOptimizer(EnhancedGAOptimizer):
    """Backwards compatible GA optimizer."""

    def __init__(self, genome, strategy, params, **kwargs):
        # Convert old GAParams to EnhancedGAParams
        if hasattr(params, "__dict__"):
            enhanced_params = EnhancedGAParams()
            for key, value in params.__dict__.items():
                if hasattr(enhanced_params, key):
                    setattr(enhanced_params, key, value)
        else:
            enhanced_params = params

        super().__init__(genome, strategy, enhanced_params, **kwargs)
