from types import SimpleNamespace

import lm_eval

from mergekit.evo import actors
from mergekit.evo import monkeypatch as evo_monkeypatch


def test_accelerated_eval_model_kwargs_for_hf_gpu():
    model_kwargs = actors._accelerated_eval_model_kwargs(vllm=False)

    assert model_kwargs["device"] == "cuda"
    assert model_kwargs["dtype"] == "bfloat16"
    assert model_kwargs["device_map"] is None
    assert model_kwargs["low_cpu_mem_usage"] is True


def test_accelerated_eval_model_kwargs_for_vllm_omits_hf_overrides():
    assert actors._accelerated_eval_model_kwargs(vllm=True) is None


def test_get_lm_eval_vllm_class_returns_none_when_backend_is_missing(monkeypatch):
    monkeypatch.delattr(actors.lm_eval.models, "vllm_causallms", raising=False)

    assert actors._get_lm_eval_vllm_class() is None


def test_monkeypatch_lmeval_vllm_noops_when_backend_is_missing(monkeypatch):
    monkeypatch.delattr(lm_eval.models, "vllm_causallms", raising=False)

    evo_monkeypatch.monkeypatch_lmeval_vllm()


def test_evaluate_merged_path_accelerated_uses_gpu_eval(monkeypatch):
    captured = {}

    def fake_evaluate_model(
        merged_path,
        tasks,
        num_fewshot,
        limit,
        vllm,
        tensor_parallel_size,
        batch_size,
        task_manager,
        apply_chat_template,
        fewshot_as_multiturn,
        model_kwargs,
        **kwargs,
    ):
        captured["merged_path"] = merged_path
        captured["tasks"] = tasks
        captured["num_fewshot"] = num_fewshot
        captured["limit"] = limit
        captured["vllm"] = vllm
        captured["tensor_parallel_size"] = tensor_parallel_size
        captured["batch_size"] = batch_size
        captured["task_manager"] = task_manager
        captured["apply_chat_template"] = apply_chat_template
        captured["fewshot_as_multiturn"] = fewshot_as_multiturn
        captured["model_kwargs"] = model_kwargs
        captured["extra_kwargs"] = kwargs
        return {"score": 0.5, "results": {}}

    monkeypatch.setattr("mergekit.evo.actors.evaluate_model", fake_evaluate_model)

    config = SimpleNamespace(
        tasks=["task-a"],
        num_fewshot=0,
        limit=64,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
    )

    result = actors._evaluate_merged_path_accelerated(
        "/tmp/merged-model",
        config,
        vllm=False,
        tensor_parallel_size=1,
        batch_size=8,
        task_manager="task-manager",
        quantization_config=None,
    )

    assert result["score"] == 0.5
    assert captured["merged_path"] == "/tmp/merged-model"
    assert captured["tasks"] == ["task-a"]
    assert captured["vllm"] is False
    assert captured["tensor_parallel_size"] == 1
    assert captured["model_kwargs"]["device"] == "cuda"
    assert "fitness_mode" in captured["extra_kwargs"]
    assert captured["apply_chat_template"] is True
    assert captured["fewshot_as_multiturn"] is False
