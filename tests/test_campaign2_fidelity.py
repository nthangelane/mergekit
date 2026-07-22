import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mergekit.common import ModelReference
from mergekit.evo.config import AuditConfiguration
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer, EnhancedGAParams
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)
from mergekit.evo.reentry import (
    align_reentrant_parent_vocab,
    capture_genome_geometry,
    pad_genotype_for_reentry,
    register_reentrant_parent,
)
from mergekit.evo.strategy import (
    EvaluationStrategyBase,
    _evaluate_genotype_serial_cpu_impl,
)
from mergekit.options import MergeOptions


class DummyConfig(SimpleNamespace):
    def model_copy(self, update):
        payload = dict(self.__dict__)
        payload.update(update)
        return DummyConfig(**payload)


class QuarantineStrategy(EvaluationStrategyBase):
    def __init__(self, mode):
        self.config = DummyConfig(
            two_stage=True,
            tasks=[SimpleNamespace(name="sciq")],
            stage1_tasks=[SimpleNamespace(name="sciq")],
            num_fewshot=0,
            limit=10,
            stage1_limit=2,
            stage2_limit=10,
            stage2_top_k=2,
            metric_guard_mode=mode,
            quarantine_audit_limit=100,
            fitness_mode="weighted_sum",
            ga=None,
        )
        self.genome = SimpleNamespace(
            method_label_for_genotype=lambda genotype: "passthrough"
        )
        self.current_generation = 1
        self.current_phase = "ga"

    def _evaluate_genotypes_once(self, genotypes, eval_config):
        rows = []
        for genotype in genotypes:
            candidate = int(genotype[0])
            if eval_config.limit == 2 and candidate == 1:
                rows.append(
                    {
                        "score": None,
                        "results": {"sciq": {"acc,none": 0.0}},
                        "error_stage": "eval",
                        "error_type": "metric_guard",
                        "error_message": "accuracy below threshold",
                        "guarded_task": "sciq",
                    }
                )
            elif eval_config.limit == 100 and candidate == 1:
                rows.append({"score": 0.42, "results": {"sciq": {"acc,none": 0.42}}})
            else:
                score = 0.5 if candidate == 1 else 0.2
                rows.append({"score": score, "results": {"sciq": {"acc,none": score}}})
        return rows


def test_quarantine_recovers_stage1_guarded_candidate():
    strategy = QuarantineStrategy("quarantine")
    results = strategy.evaluate_genotypes([np.array([1]), np.array([2])])

    assert results[0]["score"] == 0.5
    assert results[0]["quarantined"] is True
    assert results[0]["quarantine_outcome"] == "passed"
    assert results[0]["score_source"] == "stage2"


def test_reject_mode_preserves_stage1_metric_guard_rejection():
    strategy = QuarantineStrategy("reject")
    results = strategy.evaluate_genotypes([np.array([1]), np.array([2])])

    assert results[0]["score"] is None
    assert results[0]["error_type"] == "metric_guard"
    assert results[0]["quarantined"] is False


class AuditGenome:
    def initial_genotype(self, random=False):
        if random:
            return torch.rand(1, 2, 1, 1)
        return torch.tensor([[[[0.5]], [[0.5]]]], dtype=torch.float32)


class AuditStrategy:
    def __init__(self):
        self.config = SimpleNamespace(
            audit=AuditConfiguration(
                enabled=True,
                every_generations=1,
                top_n=1,
                limit=50,
                final_audit=True,
            ),
            audit_max_total_seconds=10_000.0,
            repair=None,
        )

    def evaluate_genotypes(self, genotypes):
        return [
            {"score": float(np.asarray(genotype).sum()), "results": {}}
            for genotype in genotypes
        ]

    def audit_genotypes(self, genotypes, *, limit):
        assert limit == 50
        return [
            {"score": float(np.asarray(genotype).sum()) - 0.1, "results": {}}
            for genotype in genotypes
        ]


def _run_audited_optimizer(seed):
    rows = []
    optimizer = EnhancedGAOptimizer(
        genome=AuditGenome(),
        strategy=AuditStrategy(),
        params=EnhancedGAParams(
            population_size=2,
            elite_fraction=0.5,
            mutation_rate=0.0,
            mutation_sigma=0.0,
        ),
        seed=seed,
        on_audit=lambda row: rows.append(dict(row)),
    )
    winner, score = optimizer.run(max_fevals=6)
    return optimizer, winner, score, rows


def test_three_generation_audit_loop_is_deterministic_and_corrects_cache():
    first, first_winner, first_score, first_rows = _run_audited_optimizer(17)
    second, second_winner, second_score, second_rows = _run_audited_optimizer(17)

    assert len(first_rows) == 4
    assert [row["final"] for row in first_rows] == [False, False, False, True]
    stable_fields = (
        "generation",
        "genotype_hash",
        "search_score",
        "audit_score",
        "delta",
        "final",
    )
    assert [tuple(row[key] for key in stable_fields) for row in first_rows] == [
        tuple(row[key] for key in stable_fields) for row in second_rows
    ]
    assert np.array_equal(first_winner, second_winner)
    assert first_score == second_score
    corrected = [
        cached_result
        for _, cached_result in first._fitness_cache.values()
        if cached_result.get("audited")
    ]
    assert corrected
    assert all(row["score"] == row["audit_score"] for row in corrected)


def _multi_method_genome(monkeypatch):
    class DummyModelConfig:
        num_hidden_layers = 2
        architectures = ["DummyForCausalLM"]
        model_type = "dummy"

        def to_dict(self):
            return {
                "num_hidden_layers": 2,
                "architectures": self.architectures,
                "model_type": self.model_type,
                "hidden_size": 8,
            }

    monkeypatch.setattr(
        ModelReference,
        "config",
        lambda self, trust_remote_code=False: DummyModelConfig(),
    )
    return MultiMethodGenome(
        MultiMethodGenomeDefinition.model_validate(
            {
                "models": ["author/model-a", "author/model-b"],
                "allowed_methods": ["linear"],
                "max_models_per_layer": 2,
            }
        )
    )


def test_reentry_padding_preserves_existing_decode(monkeypatch, tmp_path):
    genome = _multi_method_genome(monkeypatch)
    old_genotype = genome.initial_genotype(random=False).numpy()
    old_decoded = genome.decode_genotype(old_genotype)[0]
    geometry = capture_genome_geometry(genome)

    register_reentrant_parent(genome, str(tmp_path / "repaired"))
    padded = pad_genotype_for_reentry(old_genotype, geometry)
    new_decoded = genome.decode_genotype(padded)[0]

    assert new_decoded.method == old_decoded.method
    assert np.allclose(new_decoded.parameters, old_decoded.parameters)
    assert np.allclose(new_decoded.model_selection[:2], old_decoded.model_selection)
    assert new_decoded.model_selection[2] == 0.0


def test_mocked_audited_gain_triggers_exactly_one_reentry(monkeypatch, tmp_path):
    genome = _multi_method_genome(monkeypatch)
    merged_dir = tmp_path / "merged"
    checkpoint = merged_dir / "candidate"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    strategy = SimpleNamespace(
        config=SimpleNamespace(
            audit=None,
            repair=SimpleNamespace(
                reentry=True, reentry_min_gain=0.02, max_reentries=1
            ),
        ),
        model_storage_path=str(merged_dir),
    )
    rows = []
    optimizer = EnhancedGAOptimizer(
        genome=genome,
        strategy=strategy,
        params=EnhancedGAParams(population_size=2),
        seed=9,
        on_reentry=lambda row: rows.append(dict(row)),
    )
    genotype = genome.initial_genotype(random=False).numpy()
    population = np.stack([genotype, genotype], axis=0)
    results = [
        {
            "repair_pre_score": 0.40,
            "repair_post_score": 0.45,
            "repair_comparison_audited": True,
            "reentry_checkpoint_path": str(checkpoint),
        },
        {},
    ]

    expanded, best = optimizer._process_reentries(
        population, results, genotype, generation=1
    )
    optimizer._process_reentries(expanded, results, best, generation=2)

    assert len(rows) == 1
    assert np.isclose(rows[0]["gain"], 0.05)
    assert len(genome.definition.models) == 3
    assert expanded.shape[1] == genotype.size + 1
    assert (tmp_path / "reentrant" / rows[0]["genotype_hash"]).is_dir()


def test_reentrant_checkpoint_vocab_is_expanded_to_source_width(monkeypatch, tmp_path):
    genome = _multi_method_genome(monkeypatch)
    checkpoint = tmp_path / "repaired"
    checkpoint.mkdir()
    resized = []

    class ChildConfig:
        vocab_size = 3

    class ChildModel:
        def resize_token_embeddings(self, vocab_size):
            resized.append(vocab_size)

        def save_pretrained(self, path, safe_serialization):
            assert path == str(checkpoint)
            assert safe_serialization is True

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: ChildConfig(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: ChildModel(),
    )
    monkeypatch.setattr(
        ModelReference,
        "config",
        lambda self, trust_remote_code=False: SimpleNamespace(vocab_size=4),
    )

    changed = align_reentrant_parent_vocab(genome, str(checkpoint))

    assert changed is True
    assert resized == [4]


def test_reentrant_passthrough_and_original_parent_blend_evaluate(
    monkeypatch, tmp_path
):
    genome = _multi_method_genome(monkeypatch)
    definition = genome.definition.model_copy(
        update={"allowed_methods": ["passthrough", "linear"]}
    )
    genome = MultiMethodGenome(definition)
    checkpoint = tmp_path / "repaired"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_score.json").write_text(
        json.dumps({"score": 0.73}), encoding="utf-8"
    )
    register_reentrant_parent(genome, str(checkpoint))

    # Build candidates explicitly: method | three model weights | parameters.
    passthrough = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    blend = np.array([1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0], dtype=np.float32)
    seen_sources = []

    def fake_run_merge(config, out_path, options):
        del options
        sources = [str(model) for model in config.referenced_models()]
        seen_sources.append(sources)
        output = tmp_path / out_path
        output.mkdir(parents=True, exist_ok=True)
        score = 0.73 if sources == [str(checkpoint)] else 0.61
        (output / "score.json").write_text(
            json.dumps({"score": score}), encoding="utf-8"
        )

    def fake_evaluate(path, *args, **kwargs):
        del args, kwargs
        score = json.loads((tmp_path / path / "score.json").read_text())["score"]
        return {"score": score, "results": {}}

    monkeypatch.setattr("mergekit.evo.helpers.ensure_free_disk", lambda *args: 1.0)
    monkeypatch.setattr("mergekit.evo.helpers.run_merge", fake_run_merge)
    monkeypatch.setattr("mergekit.evo.strategy.evaluate_model_cpu", fake_evaluate)
    eval_config = DummyConfig(
        tasks=[],
        num_fewshot=0,
        limit=1,
        repair=None,
        fitness_mode="weighted_sum",
        fitness=None,
        task_mix_profile=None,
        behavior_prompts=None,
    )
    options = MergeOptions(device="cpu", cuda=False, quiet=True)

    passthrough_result = _evaluate_genotype_serial_cpu_impl(
        passthrough,
        eval_config,
        genome,
        options,
        model_storage_path=str(tmp_path),
    )
    blend_result = _evaluate_genotype_serial_cpu_impl(
        blend,
        eval_config,
        genome,
        options,
        model_storage_path=str(tmp_path),
    )

    assert passthrough_result["score"] == pytest.approx(0.73)
    assert blend_result["score"] == pytest.approx(0.61)
    assert seen_sources[0] == [str(checkpoint)]
    assert str(checkpoint) in seen_sources[1]
    assert str(genome.definition.models[0]) in seen_sources[1]


def test_resume_registers_reentrant_parent_with_evaluator(monkeypatch, tmp_path):
    from mergekit.evo.checkpoint import capture_rng_state

    checkpoint = tmp_path / "reentrant" / "abc"
    checkpoint.mkdir(parents=True)
    genome = _multi_method_genome(monkeypatch)
    geometry = capture_genome_geometry(genome)
    register_reentrant_parent(genome, str(checkpoint))
    population = np.stack(
        [
            pad_genotype_for_reentry(
                np.array([1.0, 0.5, 0.5, 1.0, 0.0, 0.0], dtype=np.float32),
                geometry,
            )
            for _ in range(2)
        ]
    )
    state = {
        "optimizer": "enhanced_ga",
        "population": population.tolist(),
        "fevals": 2,
        "elapsed_seconds": 1.0,
        "best_x": population[0].tolist(),
        "best_score": 0.5,
        "best_generation": 1,
        "best_score_source": "search",
        "no_improve": 0,
        "stagnation_generations": 0,
        "mutation_sigma": 0.1,
        "adaptive_operator_state": {},
        "fitness_cache": [],
        "audited_candidates": [],
        "fitness_history": [],
        "reentrant_parents": [str(checkpoint)],
        "rng_state": capture_rng_state(np.random.RandomState(13)),
    }

    resumed_genome = _multi_method_genome(monkeypatch)

    class SyncingStrategy:
        def __init__(self):
            self.genome = resumed_genome
            self.config = SimpleNamespace(audit=None, repair=None)
            self.registered = []

        def register_reentrant_parent(self, path):
            self.registered.append(path)
            return register_reentrant_parent(self.genome, path)

    strategy = SyncingStrategy()
    optimizer = EnhancedGAOptimizer(
        genome=resumed_genome,
        strategy=strategy,
        params=EnhancedGAParams(population_size=2),
        seed=13,
        resume_state=state,
    )

    restored = optimizer._restore_checkpoint()

    assert strategy.registered == [str(checkpoint)]
    assert len(strategy.genome.definition.models) == 3
    assert restored["population"].shape == (2, 7)


def test_archive_novelty_handles_mixed_widths_after_reentry():
    """Regression: after a memetic re-entry grows the genome, archive entries
    recorded pre-re-entry are shorter than new genotypes; the distance
    computation must zero-pad rather than raise (Campaign 2 seed-22 crash,
    20 Jul 2026)."""
    import numpy as np

    from mergekit.evo.enhanced_ga import EnhancedGAOptimizer

    opt = EnhancedGAOptimizer.__new__(EnhancedGAOptimizer)
    opt._novelty_archive = [np.ones(6, dtype=np.float32)]
    # 7-dim candidate vs 6-dim archive must not raise
    score = opt._archive_novelty_score(np.ones(7, dtype=np.float32))
    assert 0.0 <= score <= 1.0
    # A pre-existing genotype padded with the weight-0 slot is identical
    padded = np.zeros(7, dtype=np.float32)
    padded[:6] = 1.0
    assert opt._archive_novelty_score(padded) == 0.0
