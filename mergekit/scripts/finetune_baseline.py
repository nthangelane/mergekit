import logging
import os
from typing import Optional, Tuple

import click
import yaml

from mergekit.evo.finetune import FineTuneBaselineConfiguration, run_finetune_experiment


@click.command("mergekit-finetune-baseline")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional YAML preset; explicit CLI options override it.",
)
@click.option("--arm", type=click.Choice(["lora", "full"]), default=None)
@click.option("--model", type=str, default=None)
@click.option("--dataset", type=str, default=None)
@click.option("--dataset-config", type=str, default=None)
@click.option("--split", type=str, default=None)
@click.option("--text-column", type=str, default=None)
@click.option("--budget-seconds", type=click.FloatRange(min=0.0), default=None)
@click.option("--budget-steps", type=click.IntRange(min=1), default=None)
@click.option(
    "--match-ga-run",
    type=click.Path(exists=True, path_type=str),
    default=None,
    help="Set the training wall-time budget from RUN_DIR/progress.log.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=str),
    required=True,
)
@click.option("--seed", type=int, default=None)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default=None,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=None)
@click.option("--seq-len", type=click.IntRange(min=1), default=None)
@click.option("--learning-rate", type=click.FloatRange(min=0.0), default=None)
@click.option("--max-train-examples", type=click.IntRange(min=1), default=None)
@click.option("--lora-r", type=click.IntRange(min=1), default=None)
@click.option("--lora-alpha", type=click.IntRange(min=1), default=None)
@click.option("--search-limit", type=click.IntRange(min=1), default=None)
@click.option("--export-limit", type=click.IntRange(min=1), default=None)
@click.option("--eval-batch-size", type=click.IntRange(min=1), default=None)
@click.option("--task-search-path", multiple=True, type=str)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=None,
)
def main(
    config_path: Optional[str],
    arm: Optional[str],
    model: Optional[str],
    dataset: Optional[str],
    dataset_config: Optional[str],
    split: Optional[str],
    text_column: Optional[str],
    budget_seconds: Optional[float],
    budget_steps: Optional[int],
    match_ga_run: Optional[str],
    output_dir: str,
    seed: Optional[int],
    device: Optional[str],
    batch_size: Optional[int],
    seq_len: Optional[int],
    learning_rate: Optional[float],
    max_train_examples: Optional[int],
    lora_r: Optional[int],
    lora_alpha: Optional[int],
    search_limit: Optional[int],
    export_limit: Optional[int],
    eval_batch_size: Optional[int],
    task_search_path: Tuple[str, ...],
    trust_remote_code: Optional[bool],
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}
        if not isinstance(payload, dict):
            raise click.ClickException("Fine-tuning config must be a YAML mapping")

    overrides = {
        "arm": arm,
        "model": model,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "text_column": text_column,
        "budget_seconds": budget_seconds,
        "budget_steps": budget_steps,
        "match_ga_run": match_ga_run,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "max_train_examples": max_train_examples,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "search_limit": search_limit,
        "export_limit": export_limit,
        "eval_batch_size": eval_batch_size,
        "trust_remote_code": trust_remote_code,
    }
    if any(value is not None for value in (budget_seconds, budget_steps, match_ga_run)):
        for budget_key in ("budget_seconds", "budget_steps", "match_ga_run"):
            payload.pop(budget_key, None)
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    if task_search_path:
        payload["task_search_path"] = list(task_search_path)

    try:
        config = FineTuneBaselineConfiguration.model_validate(payload)
        result = run_finetune_experiment(config, os.path.abspath(output_dir))
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Budget metadata: {result['budget_path']}")
    click.echo(f"Evaluation results: {result['results_path']}")
    click.echo(f"Trained model: {result['model_path']}")


if __name__ == "__main__":
    main()
