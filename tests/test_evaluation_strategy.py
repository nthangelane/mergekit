from types import SimpleNamespace

import numpy as np

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.strategy import (
    EvaluationStrategyBase,
    SerialEvaluationStrategy,
    _evaluate_genotype_serial_cpu_impl,
    _gpu_worker_capacity,
    _gpus_per_evaluation,
)


class DummyConfig(SimpleNamespace):
    def model_copy(self, update):
        data = dict(self.__dict__)
        data.update(update)
        return DummyConfig(**data)


class DummyTwoStageStrategy(EvaluationStrategyBase):
    def _evaluate_genotypes_once(self, genotypes, eval_config):
        results = []
        for genotype in genotypes:
            limit = float(eval_config.limit or 0)
            results.append(
                {
                    "score": float(genotype[0]) + (limit / 100.0),
                    "results": {
                        eval_config.tasks[0].name: {"acc,none": float(genotype[0])}
                    },
                }
            )
        return results


def test_serial_strategy_cpu_evaluates_genotypes_sequentially(monkeypatch):
    strategy = SerialEvaluationStrategy.__new__(SerialEvaluationStrategy)
    strategy.num_gpus = 0
    strategy.config = SimpleNamespace(tasks=["dummy"], num_fewshot=0, limit=4)
    strategy.genome = object()
    strategy.merge_options = object()
    strategy.model_storage_path = "/tmp/unused"
    strategy.batch_size = 2
    strategy.task_manager = object()

    seen = []

    def fake_eval(genotype, config, genome, merge_options, **kwargs):
        seen.append(int(genotype[0]))
        return {"score": float(genotype[0]), "results": {}}

    monkeypatch.setattr(
        "mergekit.evo.strategy._evaluate_genotype_serial_cpu_impl",
        fake_eval,
    )

    genotypes = [np.array([3]), np.array([1]), np.array([2])]
    results = strategy.evaluate_genotypes(genotypes)

    assert seen == [3, 1, 2]
    assert [result["score"] for result in results] == [3.0, 1.0, 2.0]


def test_evaluation_strategy_two_stage_promotes_top_k():
    strategy = DummyTwoStageStrategy.__new__(DummyTwoStageStrategy)
    strategy.config = DummyConfig(
        two_stage=True,
        tasks=[SimpleNamespace(name="stage2-task")],
        stage1_tasks=[SimpleNamespace(name="stage1-task")],
        num_fewshot=0,
        limit=10,
        stage1_limit=2,
        stage2_limit=10,
        stage2_top_k=2,
    )

    results = strategy.evaluate_genotypes([np.array([1]), np.array([3]), np.array([2])])

    assert results[0]["score"] == 1.02
    assert results[0]["stage2_skipped"] is True
    assert results[0]["score_source"] == "stage1"
    assert results[1]["score"] == 3.10
    assert results[1]["stage2_skipped"] is False
    assert results[1]["score_source"] == "stage2"
    assert results[1]["stage1_limit"] == 2
    assert results[1]["stage2_limit"] == 10
    assert results[1]["stage1_tasks"] == ["stage1-task"]
    assert results[1]["stage2_tasks"] == ["stage2-task"]


def test_serial_two_stage_repairs_only_top_k(monkeypatch):
    config = EvolMergeConfiguration.model_validate(
        {
            "genome": {
                "models": ["author/model-a", "author/model-b"],
                "merge_method": "linear",
            },
            "tasks": ["stage2-task"],
            "stage1_tasks": ["stage1-task"],
            "two_stage": True,
            "stage2_top_k": 2,
            "repair": {
                "enabled": True,
                "probe_steps": 1,
                "max_steps": 1,
            },
        }
    )
    strategy = SerialEvaluationStrategy.__new__(SerialEvaluationStrategy)
    strategy.config = config
    strategy.num_gpus = 0
    strategy.genome = SimpleNamespace(
        method_label_for_genotype=lambda genotype: "linear"
    )
    strategy.merge_options = object()
    strategy.model_storage_path = "/tmp/unused"
    strategy.batch_size = 1
    strategy.task_manager = object()
    strategy.run_observer = None
    strategy.current_generation = 1
    strategy.current_phase = "ga"

    repaired = []

    def fake_repair(path, genotype, eval_config):
        del path, eval_config
        repaired.append(int(genotype[0]))
        return {"repaired": True, "probe_slope": 0.1, "steps_used": 1}

    strategy.repair_checkpoint = fake_repair

    def fake_eval(genotype, eval_config, genome, merge_options, **kwargs):
        del genome, merge_options
        repair_callback = kwargs.get("repair_callback")
        result = {"score": float(genotype[0]), "results": {}}
        if repair_callback is not None:
            pre_score = result["score"]
            result["score"] += 10.0
            result["repair"] = repair_callback("/tmp/merged", genotype, eval_config)
            result["repair_pre_score"] = pre_score
            result["repair_post_score"] = result["score"]
            result["repair_comparison_limit"] = eval_config.limit
            result["repair_comparison_audited"] = False
        return result

    monkeypatch.setattr(
        "mergekit.evo.strategy._evaluate_genotype_serial_cpu_impl", fake_eval
    )

    results = strategy.evaluate_genotypes(
        [np.array([1]), np.array([4]), np.array([2]), np.array([3])]
    )

    assert repaired == [4, 3]
    assert sum(not result.get("stage2_skipped", False) for result in results) == 2
    assert results[1]["repair_pre_score"] == 4.0
    assert results[1]["repair_post_score"] == 14.0
    assert results[1]["repair"]["steps_used"] == 1


def test_candidate_contexts_capture_generation_stage_and_method():
    strategy = DummyTwoStageStrategy.__new__(DummyTwoStageStrategy)
    strategy.config = DummyConfig(
        two_stage=True,
        tasks=[SimpleNamespace(name="stage2-task")],
        stage1_tasks=[SimpleNamespace(name="stage1-task")],
        num_fewshot=0,
        limit=10,
        stage1_limit=2,
        stage2_limit=10,
        stage2_top_k=2,
    )
    strategy.current_generation = 7
    strategy.current_phase = "ga"

    class DummyGenome:
        def method_label_for_genotype(self, genotype):
            return "linear" if int(genotype[0]) == 1 else "slerp"

    strategy.genome = DummyGenome()

    contexts = strategy._candidate_contexts(
        [np.array([1]), np.array([2])],
        strategy._stage_config(stage=1),
    )

    assert contexts == [
        {
            "generation": 7,
            "candidate_index": 0,
            "evaluation_stage": "stage1",
            "phase": "ga",
            "merge_method": "linear",
        },
        {
            "generation": 7,
            "candidate_index": 1,
            "evaluation_stage": "stage1",
            "phase": "ga",
            "merge_method": "slerp",
        },
    ]


def test_two_stage_weighted_rank_uses_objective_ranking():
    class DummyWeightedRankStrategy(EvaluationStrategyBase):
        def _evaluate_genotypes_once(self, genotypes, eval_config):
            results = []
            for genotype in genotypes:
                raw_score = float(genotype[0])
                task_score = float(genotype[1])
                results.append(
                    {
                        "score": raw_score,
                        "results": {eval_config.tasks[0].name: {"acc,none": raw_score}},
                        "fitness_components": {
                            "raw_weighted_score": raw_score,
                            "task_score": task_score,
                            "language_quality": task_score,
                            "stability_score": 1.0,
                        },
                    }
                )
            return results

    strategy = DummyWeightedRankStrategy.__new__(DummyWeightedRankStrategy)
    strategy.config = DummyConfig(
        two_stage=True,
        tasks=[SimpleNamespace(name="stage2-task")],
        stage1_tasks=[SimpleNamespace(name="stage1-task")],
        num_fewshot=0,
        limit=10,
        stage1_limit=2,
        stage2_limit=10,
        stage2_top_k=1,
        fitness_mode="weighted_rank",
        ga=SimpleNamespace(
            rank_objective_weights={
                "raw_weighted_score": 0.1,
                "task_score": 0.6,
                "language_quality": 0.2,
                "stability_score": 0.1,
            }
        ),
    )

    results = strategy.evaluate_genotypes(
        [np.array([0.9, 0.1]), np.array([0.8, 0.9]), np.array([0.7, 0.3])]
    )

    assert results[0]["stage2_skipped"] is True
    assert results[1]["stage2_skipped"] is False
    assert results[1]["score_source"] == "stage2"


def test_gpu_worker_capacity_scales_with_tensor_parallel():
    assert _gpus_per_evaluation(total_gpus=10, vllm=True, tensor_parallel_size=2) == 2
    assert _gpu_worker_capacity(total_gpus=10, vllm=True, tensor_parallel_size=2) == 5


def test_tensor_parallel_requires_vllm_backend():
    try:
        _gpus_per_evaluation(total_gpus=4, vllm=False, tensor_parallel_size=2)
    except ValueError as exc:
        assert "requires the vLLM backend" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected tensor parallel validation to fail")


def test_serial_cpu_helper_reports_merge_failure(monkeypatch, capsys):
    def fake_merge_model_with_details(*args, **kwargs):
        return {
            "merged_path": None,
            "error_stage": "merge",
            "error_type": "invalid_genotype",
            "error_message": "bad genotype",
        }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("evaluate_model_cpu should not run after merge failure")

    monkeypatch.setattr(
        "mergekit.evo.strategy.merge_model_with_details",
        fake_merge_model_with_details,
    )
    monkeypatch.setattr("mergekit.evo.strategy.evaluate_model_cpu", fail_if_called)

    config = SimpleNamespace(tasks=["dummy"], num_fewshot=0, limit=4)
    result = _evaluate_genotype_serial_cpu_impl(
        np.array([1]),
        config,
        object(),
        object(),
        model_storage_path="/tmp/unused",
        batch_size=2,
        task_manager=object(),
    )

    output = capsys.readouterr().out
    assert result["score"] is None
    assert result["error"] == "merge_failed"
    assert result["error_type"] == "invalid_genotype"
    assert result["error_message"] == "bad genotype"
    assert "Merge failed after" in output
    assert "reason=invalid_genotype: bad genotype" in output
    assert "skipping evaluation" in output
    assert "Total time:" in output


def test_serial_cpu_helper_reports_timing_on_success(monkeypatch, capsys):
    def fake_merge_model_with_details(*args, **kwargs):
        return {
            "merged_path": "/tmp/fake-merged",
            "error_stage": None,
            "error_type": None,
            "error_message": None,
        }

    def fake_evaluate_model_cpu(*args, **kwargs):
        return {"score": 1.25, "results": {}}

    monkeypatch.setattr(
        "mergekit.evo.strategy.merge_model_with_details",
        fake_merge_model_with_details,
    )
    monkeypatch.setattr(
        "mergekit.evo.strategy.evaluate_model_cpu",
        fake_evaluate_model_cpu,
    )

    config = SimpleNamespace(tasks=["dummy"], num_fewshot=0, limit=4)
    result = _evaluate_genotype_serial_cpu_impl(
        np.array([1]),
        config,
        object(),
        object(),
        model_storage_path="/tmp/unused",
        batch_size=2,
        task_manager=object(),
    )

    output = capsys.readouterr().out
    assert result["score"] == 1.25
    assert "Merge completed in" in output
    assert "Evaluation completed in" in output
    assert "Total time:" in output


def test_repair_pre_and_post_scores_use_same_audit_fidelity(monkeypatch, tmp_path):
    checkpoint = tmp_path / "merged"
    checkpoint.mkdir()
    state = {"repaired": False}
    limits = []

    monkeypatch.setattr(
        "mergekit.evo.strategy.merge_model_with_details",
        lambda *args, **kwargs: {
            "merged_path": str(checkpoint),
            "error_stage": None,
            "error_type": None,
            "error_message": None,
        },
    )

    def fake_evaluate_model_cpu(*args, **kwargs):
        limits.append(kwargs["limit"])
        return {"score": 0.45 if state["repaired"] else 0.40, "results": {}}

    monkeypatch.setattr(
        "mergekit.evo.strategy.evaluate_model_cpu", fake_evaluate_model_cpu
    )

    def fake_repair(path, genotype, eval_config):
        del path, genotype
        assert eval_config.limit == 50
        state["repaired"] = True
        return {"repaired": True, "probe_slope": 0.1}

    search_config = DummyConfig(
        tasks=[SimpleNamespace(name="sciq")],
        num_fewshot=0,
        limit=10,
        fitness_mode="weighted_sum",
        fitness=SimpleNamespace(
            version="v2", lower_is_better_transform="log_reciprocal"
        ),
    )
    audit_config = search_config.model_copy(update={"limit": 50})
    result = _evaluate_genotype_serial_cpu_impl(
        np.array([1]),
        search_config,
        object(),
        object(),
        model_storage_path=str(tmp_path),
        repair_callback=fake_repair,
        repair_eval_config=audit_config,
    )

    assert limits == [50, 50]
    assert result["repair_pre_score"] == 0.40
    assert result["repair_post_score"] == 0.45
    assert result["repair_comparison_limit"] == 50
    assert result["repair_comparison_audited"] is True
