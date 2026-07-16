import csv
import json

import pytest
import torch
from click.testing import CliRunner

import mergekit.scripts.finetune_baseline as finetune_cli
from mergekit.evo.finetune import (
    FineTuneBaselineConfiguration,
    TrainingResult,
    infer_attention_projection_targets,
    resolve_training_budget,
    run_finetune_experiment,
)
from mergekit.evo.progress import ProgressLogger


def _config(**updates):
    payload = {
        "arm": "lora",
        "model": "author/model",
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "budget_steps": 50,
        "device": "cpu",
    }
    payload.update(updates)
    return FineTuneBaselineConfiguration.model_validate(payload)


def test_finetune_config_rejects_evaluation_split_contamination():
    with pytest.raises(ValueError, match="Training dataset contamination"):
        _config(
            dataset="super_glue",
            dataset_config="boolq",
            split="validation",
        )


def test_finetune_budget_can_match_measured_ga_compute(tmp_path):
    progress = ProgressLogger(str(tmp_path), seed=11)
    progress.write("generation_completed", secs=2.5, compute=True)
    progress.write("generation_completed", secs=3.0, compute=True)
    config = _config(budget_steps=None, match_ga_run=str(tmp_path))

    steps, seconds, budget_type = resolve_training_budget(config)

    assert steps is None
    assert seconds == 5.5
    assert budget_type == "matched_ga_seconds"


def test_attention_target_detection_excludes_non_attention_linears():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.mlp = torch.nn.Linear(4, 4)

    assert infer_attention_projection_targets(Model()) == ["attention.q_proj"]


def test_fifty_step_lora_cpu_training_completes(monkeypatch, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

    source_model = tmp_path / "source_model"
    source_model.mkdir()
    tokenizer_backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "hello": 3,
                "world": 4,
                "science": 5,
                "question": 6,
                "answer": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(source_model)
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(source_model, safe_serialization=True)

    monkeypatch.setattr(
        "mergekit.evo.finetune.load_training_texts",
        lambda _config: ["hello world science question answer"] * 8,
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    config = _config(
        model=str(source_model),
        budget_steps=50,
        batch_size=2,
        seq_len=16,
        learning_rate=1e-3,
    )

    def fake_evaluator(model_name, tasks, *, limit, config, device):
        task_results = {
            task.name: {task.metric: 10.0 if "perplexity" in task.metric else 0.5}
            for task in tasks
        }
        return {"score": 0.25, "results": task_results}

    result = run_finetune_experiment(
        config,
        str(output_dir),
        evaluator=fake_evaluator,
    )
    budget = json.loads((output_dir / "budget.json").read_text())

    assert budget["actual_steps"] == 50
    assert budget["trainable_parameters"] < budget["total_parameters"]
    assert budget["lora_target_modules"]
    with open(result["results_path"], "r", encoding="utf-8", newline="") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 8
    assert (output_dir / "trained_model" / "config.json").is_file()
    assert (output_dir / "budget.json").is_file()


def test_finetune_experiment_emits_eight_comparison_rows(tmp_path):
    def fake_trainer(config, output_dir, progress):
        trained_model = tmp_path / "trained_model"
        trained_model.mkdir()
        return TrainingResult(
            model_path=str(trained_model),
            budget={
                "schema_version": 1,
                "arm": config.arm,
                "actual_steps": 50,
                "actual_training_seconds": 1.0,
                "peak_memory_bytes": 1024,
            },
        )

    def fake_evaluator(model_name, tasks, *, limit, config, device):
        task_results = {}
        for task in tasks:
            value = 10.0 if "perplexity" in task.metric else 0.5
            task_results[task.name] = {task.metric: value}
        return {"score": 0.25, "results": task_results}

    result = run_finetune_experiment(
        _config(),
        str(tmp_path),
        trainer=fake_trainer,
        evaluator=fake_evaluator,
    )

    with open(result["results_path"], "r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 8
    assert len({(row["arm"], row["suite"], row["protocol"]) for row in rows}) == 8
    assert {row["model_role"] for row in rows} == {"parent", "trained"}
    assert json.loads((tmp_path / "budget.json").read_text())["evaluation_seconds"] >= 0
    progress_events = [
        json.loads(line)
        for line in (tmp_path / "progress.log").read_text().splitlines()
    ]
    assert progress_events[-1]["msg"] == "finetune_run_finished"


def test_finetune_evaluation_failure_is_terminal(tmp_path):
    def fake_trainer(config, output_dir, progress):
        trained_model = tmp_path / "trained_model"
        trained_model.mkdir()
        return TrainingResult(
            model_path=str(trained_model),
            budget={"schema_version": 1, "actual_steps": 50},
        )

    def failed_evaluator(*args, **kwargs):
        raise RuntimeError("simulated evaluation failure")

    with pytest.raises(RuntimeError, match="8 fine-tuning evaluation"):
        run_finetune_experiment(
            _config(),
            str(tmp_path),
            trainer=fake_trainer,
            evaluator=failed_evaluator,
        )

    progress_events = [
        json.loads(line)
        for line in (tmp_path / "progress.log").read_text().splitlines()
    ]
    assert progress_events[-1]["msg"] == "finetune_run_finished"
    assert progress_events[-1]["status"] == "failed"


def test_cli_budget_override_replaces_preset_budget(monkeypatch, tmp_path):
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text(
        "\n".join(
            [
                "arm: lora",
                "model: author/model",
                "dataset: databricks/databricks-dolly-15k",
                "split: train",
                "budget_steps: 192",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run(config, output_dir):
        assert config.budget_steps is None
        assert config.budget_seconds == 7.5
        return {
            "budget_path": "budget.json",
            "results_path": "results.csv",
            "model_path": "trained_model",
        }

    monkeypatch.setattr(finetune_cli, "run_finetune_experiment", fake_run)
    result = CliRunner().invoke(
        finetune_cli.main,
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--budget-seconds",
            "7.5",
        ],
    )

    assert result.exit_code == 0, result.output
