from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mergekit.common import ModelReference
from mergekit.evo.cache_utils import genotype_exact_hash
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer, EnhancedGAParams
from mergekit.evo.ga import GAOptimizer, GAParams
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)


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


class ConstantScoreStrategy:
    def __init__(self, score: float = 1.0):
        self.score = float(score)

    def evaluate_genotypes(self, genotypes):
        return [{"score": self.score, "results": {}} for _ in genotypes]


class Stage1ConstantScoreStrategy(ConstantScoreStrategy):
    def evaluate_genotypes(self, genotypes):
        return [
            {"score": self.score, "results": {}, "score_source": "stage1"}
            for _ in genotypes
        ]


class ObjectiveStrategy:
    def evaluate_genotypes(self, genotypes):
        results = []
        for genotype in genotypes:
            flat = np.array(genotype).ravel().astype(np.float32)
            raw_score = float(flat[0])
            task_score = float(flat[1])
            results.append(
                {
                    "score": raw_score,
                    "results": {},
                    "fitness_components": {
                        "raw_weighted_score": raw_score,
                        "task_score": task_score,
                        "language_quality": task_score,
                        "stability_score": 1.0,
                    },
                }
            )
        return results


def _build_multi_method_genome(monkeypatch, allowed_methods, layer_granularity=0):
    class DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.architectures = ["DummyForCausalLM"]
            self.model_type = "dummy"

        def to_dict(self):
            return {
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 16,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def fake_config(self, trust_remote_code: bool = False):
        return DummyConfig()

    monkeypatch.setattr(ModelReference, "config", fake_config, raising=False)
    return MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": list(allowed_methods),
                "base_model": "author/model-a" if "slerp" in allowed_methods else None,
                "max_models_per_layer": 2,
                "layer_granularity": int(layer_granularity),
            }
        )
    )


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


def test_ga_optimizer_skips_persisted_failed_blacklist_entries():
    genome = FakeGenome(dim=4)
    strategy = CountingFailureStrategy()
    params = GAParams(population_size=4)
    failed = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    valid = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    opt = GAOptimizer(
        genome=genome,
        strategy=strategy,  # type: ignore[arg-type]
        params=params,
        seed=0,
        persisted_failed_genotypes={
            genotype_exact_hash(failed): {
                "error_stage": "merge",
                "error_type": "invalid_genotype",
                "error_message": "negative leading weight",
            }
        },
    )
    opt._fitness_cache = {}

    fitness, results = opt._evaluate_population(np.stack([failed, valid], axis=0))

    assert strategy.calls == 1
    assert np.isneginf(fitness[0])
    assert fitness[1] == 1.0
    assert results[0]["blacklisted"] is True
    assert results[0]["error_message"] == "negative leading weight"
    assert opt._last_eval_stats["evaluations"] == 1.0
    assert opt._last_eval_stats["cache_hits"] == 1.0
    assert opt._last_eval_stats["failed_evals"] == 1.0


def test_enhanced_ga_seeds_passthrough_individuals(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])

    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(population_size=3),
        random_init=False,
        seed=0,
    )

    pop = opt._init_population()

    assert pop.shape[0] == 3
    assert any(individual[0] == 1.0 for individual in pop[1:])
    assert any(
        np.allclose(individual[1:3], np.array([1.0, 0.0], dtype=np.float32))
        or np.allclose(individual[1:3], np.array([0.0, 1.0], dtype=np.float32))
        for individual in pop[1:]
    )


def test_enhanced_ga_respects_passthrough_seed_cap(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["passthrough", "linear", "slerp"])

    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            passthrough_max_fraction=0.25,
        ),
        random_init=False,
        seed=0,
    )

    pop = opt._init_population()
    methods = [opt._method_name_for_genotype(individual) for individual in pop]

    assert methods.count("passthrough") <= 1
    assert methods.count("linear") >= 2


def test_enhanced_ga_applies_passthrough_penalty(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    strategy = ConstantScoreStrategy(score=1.0)
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=strategy,  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=2,
            passthrough_penalty=0.02,
        ),
        seed=0,
    )
    opt._fitness_cache = {}

    linear = genome.initial_genotype(random=False).view(-1).numpy().astype(np.float32)
    passthrough = linear.copy()
    passthrough[0] = genome.method_gene_value("passthrough")
    passthrough[1:3] = np.array([1.0, 0.0], dtype=np.float32)
    pop = np.stack([linear, passthrough], axis=0)

    fitness, results = opt._evaluate_population(pop)

    assert fitness[0] == pytest.approx(1.0)
    assert fitness[1] == pytest.approx(0.98)
    assert results[1]["raw_score"] == pytest.approx(1.0)
    assert results[1]["fitness_components"]["passthrough_penalty"] == pytest.approx(
        0.02
    )


def test_enhanced_ga_updates_method_probabilities_from_operator_outcomes(
    monkeypatch,
):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=2,
            adaptive_method_sampling=True,
            initial_method_probs={"linear": 0.5, "passthrough": 0.5},
            operator_update_smoothing=1.0,
        ),
        seed=0,
    )

    linear = genome.initial_genotype(random=False).view(-1).numpy().astype(np.float32)
    passthrough = linear.copy()
    passthrough[0] = genome.method_gene_value("passthrough")
    passthrough[1:3] = np.array([1.0, 0.0], dtype=np.float32)
    pop = np.stack([linear, passthrough], axis=0)
    results = [{"score": 1.0}, {"score": 0.2}]
    fitness = np.array([1.0, 0.2], dtype=np.float32)
    opt._population_metadata = [
        {"parent_scores": [0.4, 0.5]},
        {"parent_scores": [0.4, 0.5]},
    ]

    summary = opt._update_operator_state(pop, results, fitness, np.array([0]))

    assert (
        summary["probabilities_after"]["linear"]
        > summary["probabilities_after"]["passthrough"]
    )
    history_by_method = {row["merge_method"]: row for row in summary["history_rows"]}
    assert history_by_method["linear"]["parent_improvement_rate"] == pytest.approx(1.0)
    assert history_by_method["passthrough"]["survival_rate"] == pytest.approx(0.0)


def test_enhanced_ga_does_not_promote_layered_mixed_to_sampleable_method(monkeypatch):
    genome = _build_multi_method_genome(
        monkeypatch,
        ["linear", "passthrough"],
        layer_granularity=2,
    )
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=2,
            adaptive_method_sampling=True,
            initial_method_probs={"linear": 0.5, "passthrough": 0.5},
            operator_update_smoothing=1.0,
        ),
        seed=0,
    )

    layered_mixed = (
        genome.initial_genotype(random=False).view(-1).numpy().astype(np.float32)
    )
    layered_mixed[0] = genome.method_gene_value("linear")
    layered_mixed[genome.layer_group_dim] = genome.method_gene_value("passthrough")
    layered_mixed[1:3] = np.array([1.0, 0.0], dtype=np.float32)
    layered_mixed[genome.layer_group_dim + 1 : genome.layer_group_dim + 3] = np.array(
        [1.0, 0.0], dtype=np.float32
    )

    linear = genome.initial_genotype(random=False).view(-1).numpy().astype(np.float32)
    pop = np.stack([layered_mixed, linear], axis=0)
    opt._population_metadata = opt._seed_population_metadata(pop)

    summary = opt._update_operator_state(
        pop,
        [{"score": 0.4}, {"score": 0.8}],
        np.array([0.4, 0.8], dtype=np.float32),
        np.array([1]),
    )

    assert "layered_mixed" not in summary["probabilities_after"]
    assert set(summary["probabilities_after"]) == {"linear", "passthrough"}


def test_enhanced_ga_passthrough_cap_forces_non_passthrough_sampling(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            initial_method_probs={"passthrough": 1.0, "linear": 0.0},
            passthrough_max_fraction=0.25,
        ),
        seed=0,
    )

    sampled = opt._sample_target_method(Counter({"passthrough": 1}), target_children=4)

    assert sampled == "linear"


def test_enhanced_ga_diverse_parent_selection_prefers_distant_candidate(monkeypatch):
    genome = FakeGenome(dim=3)
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=3,
            diversity_parent_selection=True,
            diversity_parent_weight=5.0,
            tournament_size=2,
        ),
        seed=0,
    )
    pop = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    fitness = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    opt.rs = SimpleNamespace(randint=lambda low, high=None, size=None: np.array([1, 2]))
    selected = opt._select_diverse_parent(fitness, pop, anchor_idx=0)

    assert selected == 2


def test_enhanced_ga_role_separation_records_explorer_children(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=ConstantScoreStrategy(score=1.0),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=6,
            explorer_fraction=0.34,
            initial_method_probs={"linear": 0.8, "passthrough": 0.2},
            adaptive_method_sampling=True,
        ),
        seed=0,
    )

    opt.run(max_fevals=6)

    assert opt._prev_generation_breeding["role_counts"]["explorer"] >= 1
    assert opt._prev_generation_breeding["role_counts"]["elite"] >= 1


def test_explorer_sampling_prefers_less_common_non_passthrough_method(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["passthrough", "linear", "slerp"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            adaptive_method_sampling=True,
            initial_method_probs={
                "passthrough": 0.30,
                "linear": 0.60,
                "slerp": 0.10,
            },
        ),
        seed=0,
    )

    sampled = opt._sample_target_method(Counter(), target_children=3, exploration=True)

    assert sampled == "slerp"


def test_enhanced_ga_resamples_duplicate_children(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(population_size=4, duplicate_retry_limit=2),
        seed=0,
    )

    child = genome.initial_genotype(random=False).view(-1).numpy()
    existing = {opt._hash(child)}

    calls = {"count": 0}

    def fake_mutate(x):
        calls["count"] += 1
        updated = np.array(x, dtype=np.float32).copy()
        updated[1] += 0.25
        return updated

    monkeypatch.setattr(opt, "_enhanced_mutate", fake_mutate)

    unique_child, retries = opt._make_child_unique(child, existing_hashes=existing)

    assert retries == 1
    assert calls["count"] == 1
    assert opt._hash(unique_child) not in existing


def test_enhanced_ga_weighted_rank_can_override_raw_score():
    genome = FakeGenome(dim=2)
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=ObjectiveStrategy(),
        params=EnhancedGAParams(
            population_size=2,
            fitness_mode="weighted_rank",
            rank_objective_weights={
                "raw_weighted_score": 0.1,
                "task_score": 0.7,
                "language_quality": 0.1,
                "stability_score": 0.1,
            },
        ),
        seed=0,
    )

    pop = np.array([[0.9, 0.1], [0.8, 0.9]], dtype=np.float32)
    fitness, results = opt._evaluate_population(pop)

    assert fitness[1] > fitness[0]
    assert results[0]["raw_score"] == pytest.approx(0.9)
    assert results[1]["fitness_proxy"] == "weighted_rank"


def test_enhanced_ga_archive_novelty_bonus_rewards_new_regions():
    genome = FakeGenome(dim=2)
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=ConstantScoreStrategy(score=1.0),
        params=EnhancedGAParams(
            population_size=2,
            archive_novelty_bonus_weight=0.5,
            novelty_archive_size=8,
        ),
        seed=0,
    )
    opt._novelty_archive = [np.array([0.0, 0.0], dtype=np.float32)]

    baseline = opt._adjust_result_score(
        {"score": 1.0, "results": {}}, np.array([0.0, 0.0], dtype=np.float32)
    )
    novel = opt._adjust_result_score(
        {"score": 1.0, "results": {}}, np.array([1.0, 1.0], dtype=np.float32)
    )

    assert baseline["fitness_components"]["archive_novelty_score"] == pytest.approx(0.0)
    assert novel["fitness_components"]["archive_novelty_score"] > 0.0
    assert novel["score"] > baseline["score"]


def test_enhanced_ga_adds_gene_and_behavior_diversity_bonus(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            gene_diversity_bonus_weight=0.1,
            behavior_diversity_bonus_weight=0.2,
        ),
        seed=0,
    )

    genotype = genome.initial_genotype(random=False)
    adjusted = opt._adjust_result_score(
        {
            "score": 1.0,
            "results": {},
            "behavior_probe": {"behavior_diversity_score": 0.5},
        },
        genotype,
    )

    assert adjusted["score"] > 1.0
    assert adjusted["fitness_components"]["gene_diversity_score"] >= 0.0
    assert adjusted["fitness_components"]["gene_diversity_bonus"] >= 0.0
    assert adjusted["fitness_components"]["behavior_diversity_bonus"] == pytest.approx(
        0.1
    )


def test_enhanced_ga_semantic_crossover_uses_genome_operation(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            crossover="semantic",
            semantic_crossover_prob=1.0,
        ),
        seed=0,
    )

    parent_a = genome.initial_genotype(random=False).view(-1).numpy()
    parent_b = genome.initial_genotype(random=True).view(-1).numpy()
    seen = {}

    def fake_crossover_semantic(a_torch, b_torch):
        seen["shapes"] = (tuple(a_torch.shape), tuple(b_torch.shape))
        return torch.ones_like(a_torch)

    monkeypatch.setattr(genome, "crossover_semantic", fake_crossover_semantic)

    child = opt._enhanced_crossover(parent_a, parent_b)

    assert seen["shapes"] == (parent_a.shape, parent_b.shape)
    assert np.allclose(child, 1.0)


def test_enhanced_ga_semantic_mutation_uses_genome_operation(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=object(),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=4,
            crossover="semantic",
            mutation_rate=0.23,
            mutation_sigma=0.07,
        ),
        seed=0,
    )

    genotype = genome.initial_genotype(random=True).view(-1).numpy()
    seen = {}

    def fake_mutate_semantic(x_torch, mutation_rate, mutation_sigma):
        seen["params"] = (
            tuple(x_torch.shape),
            float(mutation_rate),
            float(mutation_sigma),
        )
        return torch.zeros_like(x_torch) + 0.25

    monkeypatch.setattr(genome, "mutate_semantic", fake_mutate_semantic)

    mutated = opt._enhanced_mutate(genotype)

    assert seen["params"] == (genotype.shape, 0.23, 0.07)
    assert np.allclose(mutated, 0.25)


def test_ga_optimizer_stops_on_target_improvement():
    genome = FakeGenome(dim=4)
    opt = GAOptimizer(
        genome=genome,
        strategy=ConstantScoreStrategy(score=1.0),  # type: ignore[arg-type]
        params=GAParams(
            population_size=2,
            target_improvement_abs=0.4,
            target_reference_score=0.5,
            min_generations_before_target_stop=1,
        ),
        seed=0,
    )

    _best_x, best_score = opt.run(max_fevals=10)

    assert best_score == pytest.approx(1.0)
    assert opt.last_stop_details is not None
    assert opt.last_stop_details["reason"] == "target_improvement"
    assert opt.last_stop_details["generation"] == 1
    assert opt.last_stop_details["fevals"] == 2
    assert opt.last_stop_details["best_improvement_abs"] == pytest.approx(0.5)


def test_ga_optimizer_stops_on_stagnation():
    genome = FakeGenome(dim=4)
    opt = GAOptimizer(
        genome=genome,
        strategy=ConstantScoreStrategy(score=1.0),  # type: ignore[arg-type]
        params=GAParams(
            population_size=2,
            stagnation_patience_generations=2,
            stagnation_min_delta=0.05,
        ),
        seed=0,
    )

    _best_x, best_score = opt.run(max_fevals=20)

    assert best_score == pytest.approx(1.0)
    assert opt.last_stop_details is not None
    assert opt.last_stop_details["reason"] == "stagnation"
    assert opt.last_stop_details["generation"] == 3
    assert opt.last_stop_details["fevals"] == 6
    assert opt.last_stop_details["stagnation_generations"] == 2


def test_enhanced_ga_target_stop_can_require_stage2(monkeypatch):
    genome = _build_multi_method_genome(monkeypatch, ["linear", "passthrough"])
    opt = EnhancedGAOptimizer(
        genome=genome,
        strategy=Stage1ConstantScoreStrategy(score=1.0),  # type: ignore[arg-type]
        params=EnhancedGAParams(
            population_size=2,
            target_improvement_abs=0.1,
            target_reference_score=0.5,
            require_stage2_for_target=True,
        ),
        seed=0,
    )

    _best_x, best_score = opt.run(max_fevals=4)

    assert best_score == pytest.approx(1.0)
    assert opt.last_stop_details is not None
    assert opt.last_stop_details["reason"] == "max_fevals"
