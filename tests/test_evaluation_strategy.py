from types import SimpleNamespace

import numpy as np

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
