# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas
import ray
import torch
import tqdm
import transformers
import yaml
import lm_eval.tasks


try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference, call_with_dtype
from mergekit.evo.helpers import _eval_model
from mergekit.evo.config import (
    EvolMergeConfiguration,
    ModelGenomeDefinition,
    check_for_naughty_config,
)
from mergekit.evo.ga import GAOptimizer, GAParams
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer, EnhancedGAParams
from mergekit.evo.genome import ModelGenome
from mergekit.evo.multi_method_genome import MultiMethodGenome, MultiMethodGenomeDefinition
from mergekit.evo.strategy import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
)
from mergekit.evo.tracking import create_tracker
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


LOGGER = logging.getLogger("mergekit.evolve_ga.cli")


def stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    """Emit a structured log message for high-level run stages."""
    LOGGER.log(level, "[%s] %s", stage, message)


@click.command("mergekit-evolve-ga")
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=100)
@click.option(
    "--population-size",
    type=int,
    default=None,
    help="Population size (overrides YAML if set)",
)
@click.option(
    "--elite-fraction",
    type=float,
    default=None,
    help="Elitism fraction [0,1] (overrides YAML if set)",
)
@click.option(
    "--mutation-rate",
    type=float,
    default=None,
    help="Per-gene mutation probability (overrides YAML if set)",
)
@click.option(
    "--mutation-sigma",
    type=float,
    default=None,
    help="Stddev for Gaussian mutation noise (overrides YAML if set)",
)
@click.option(
    "--crossover",
    type=str,
    default=None,
    help="Crossover operator: arithmetic | uniform | sbx (overrides YAML if set)",
)
@click.option(
    "--tournament-size",
    type=int,
    default=None,
    help="Tournament size for selection (overrides YAML if set)",
)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    help="Evaluation scheduling strategy",
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    default=False,
    help="Use in-memory merge & evaluation",
)
@click.option(
    "--storage-path",
    type=str,
    help="Path to storage accessible to all nodes for model storage",
    required=True,
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of CPU workers when GPUs=0 (pool/buffered)",
)
@click.option("--merge-cuda/--no-merge-cuda", is_flag=True, default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option("--allow-crimes/--no-allow-crimes", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=0)
@click.option("--batch-size", type=int, default=None, help="Batch size for evaluation")
@click.option("use_wandb", "--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--wandb-project", type=str, help="Wandb project name")
@click.option("--wandb-entity", type=str, help="Wandb entity name")
@click.option("use_mlflow", "--mlflow/--no-mlflow", is_flag=True, default=False)
@click.option("--mlflow-experiment", type=str, help="MLflow experiment name")
@click.option(
    "--mlflow-tracking-uri", type=str, help="MLflow tracking URI (default: ./mlruns)"
)
@click.option(
    "--task-search-path",
    type=str,
    multiple=True,
    help="Path to search for lmeval tasks",
)
@click.option(
    "--i-understand-the-depths-of-the-evils-i-am-unleashing",
    "allow_benchmark_tasks",
    is_flag=True,
    default=False,
    help="Allow benchmark tasks as objectives",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    is_flag=True,
    default=True,
    help="Save the final merged model",
)
@click.option(
    "--reshard/--no-reshard",
    is_flag=True,
    default=True,
    help="Convert models to single-shard safetensors for faster merge",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Maximum time to run the optimization in seconds",
)
@click.option(
    "--baseline/--no-baseline",
    "run_baseline",
    is_flag=True,
    default=True,
    help="Run baseline evaluations for source models before GA search",
)
@click.option(
    "--hf-model-id",
    type=str,
    default=None,
    help="Hugging Face model ID to push final model to (e.g. username/model-name)",
)
def main(
    genome_config_path: str,
    max_fevals: int,
    population_size: Optional[int],
    elite_fraction: Optional[float],
    mutation_rate: Optional[float],
    mutation_sigma: Optional[float],
    crossover: str,
    tournament_size: Optional[int],
    vllm: bool,
    strategy: str,
    in_memory: bool,
    storage_path: Optional[str],
    num_gpus: Optional[int],
    num_workers: Optional[int],
    merge_cuda: bool,
    trust_remote_code: bool,
    allow_crimes: bool,
    random_seed: int,
    batch_size: Optional[int],
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    use_mlflow: bool,
    mlflow_experiment: Optional[str],
    mlflow_tracking_uri: Optional[str],
    task_search_path: List[str],
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    reshard: bool,
    timeout: Optional[float],
    run_baseline: bool,
    hf_model_id: Optional[str],
):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)

    stage_log("Stage-Init", f"Seeding RNG with value {random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    stage_log("Stage-Init", f"Loading genome configuration from {genome_config_path}")
    with open(genome_config_path, "r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    config = EvolMergeConfiguration.model_validate(raw_config)

    stage_log("Stage-Init", "Validating configuration settings...")
    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    storage_path = os.path.abspath(storage_path)
    os.makedirs(storage_path, exist_ok=True)
    stage_log("Stage-Init", f"Storage path: {storage_path}")

    task_search_path = list(task_search_path)

    baseline_csv_path = None
    if run_baseline:
        baseline_csv_path = run_baseline_evaluations(
            config,
            storage_path,
            batch_size,
            merge_cuda,
            num_gpus,
            task_search_path,
            trust_remote_code,
        )
    else:
        stage_log("Stage-Baseline", "Skipping baseline evaluation (--no-baseline).")

    # Initialize experiment tracking
    stage_log("Stage-Tracking", "Initializing experiment tracker...")
    tracker = None
    if use_wandb and use_mlflow:
        raise ValueError(
            "Cannot use both wandb and mlflow at the same time. Choose one."
        )
    elif use_wandb:
        stage_log("Stage-Tracking", "Using Weights & Biases for experiment tracking.")
        tracker = create_tracker("wandb")
        tracker.initialize(
            project_name=wandb_project or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            entity=wandb_entity,
        )
    elif use_mlflow:
        stage_log("Stage-Tracking", "Using MLflow for experiment tracking.")
        tracker = create_tracker("mlflow")
        tracker.initialize(
            project_name=mlflow_experiment or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            tracking_uri=mlflow_tracking_uri,
        )
    else:
        stage_log("Stage-Tracking", "Experiment tracking disabled (logging to console only).")
        tracker = create_tracker("none")
        tracker.initialize(project_name="no-tracking", config={})

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    # convert models to single-shard safetensors
    if reshard:
        stage_log(
            "Stage-Reshard",
            "Converting source models to single-shard safetensors...",
        )
        resharded_models = []
        resharded_base = None
        for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):
            resharded_models.append(
                _reshard_model(
                    model,
                    storage_path,
                    merge_options.lora_merge_cache,
                    trust_remote_code,
                )
            )
        if config.genome.base_model is not None:
            resharded_base = _reshard_model(
                config.genome.base_model,
                storage_path,
                merge_options.lora_merge_cache,
                trust_remote_code,
            )
        stage_log("Stage-Reshard", "Resharding complete.")
    else:
        stage_log("Stage-Reshard", "Skipping reshard step (--no-reshard).")
        resharded_models = config.genome.models
        resharded_base = config.genome.base_model

    # Create genome based on type - check if it's MultiMethodGenomeDefinition
    from mergekit.evo.multi_method_genome import MultiMethodGenomeDefinition
    genome_config = config.genome
    
    if isinstance(genome_config, MultiMethodGenomeDefinition):
        genome_type = 'multi_method'
        # Create multi-method genome
        genome = MultiMethodGenome(
            MultiMethodGenomeDefinition.model_validate(
                {
                    **genome_config.model_dump(exclude=["models", "base_model"]),
                    "models": resharded_models,
                    "base_model": resharded_base,
                }
            ),
            trust_remote_code=trust_remote_code,
        )
    else:
        genome_type = 'standard'
        # Create traditional genome
        genome = ModelGenome(
            ModelGenomeDefinition.model_validate(
                {
                    **genome_config.model_dump(exclude=["models", "base_model"]),
                    "models": resharded_models,
                    "base_model": resharded_base,
                }
            ),
            trust_remote_code=trust_remote_code,
        )

    if strategy == "pool":
        strat_cls = ActorPoolEvaluationStrategy
    elif strategy == "buffered":
        strat_cls = BufferedRayEvaluationStrategy
    elif strategy == "serial":
        strat_cls = SerialEvaluationStrategy
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Validate crossover if provided
    if crossover is not None and crossover not in {"arithmetic", "uniform", "sbx"}:
        raise click.BadParameter("--crossover must be one of: arithmetic, uniform, sbx")

    stage_log("Stage-GA", f"Initializing evaluation strategy '{strategy}'...")
    strat = strat_cls(
        config,
        genome,
        merge_options,
        num_gpus=num_gpus,
        num_workers=num_workers,
        vllm=vllm,
        in_memory=in_memory,
        model_storage_path=os.path.join(storage_path, "merged"),
        batch_size=batch_size,
        task_search_path=task_search_path,
    )

    def log_population(res_list: List[dict], step: int):
        tracker.log_population_stats(res_list, step)

    def log_best(x: np.ndarray, score: float, step: int):
        tracker.log_best_individual(x, score, step, genome)

    def save_best_config(x: np.ndarray):
        best_yaml = genome.genotype_merge_config(x).to_yaml()
        config_path = os.path.join(storage_path, "best_config.yaml")
        with open(config_path, "w") as f:
            f.write(best_yaml)
        print(f"Merge configuration:\n{best_yaml}")
        tracker.log_artifact(config_path, "best_config")

    # Build GA optimizer with callbacks
    # Resolve GA parameters: CLI overrides YAML; fallback to GAParams defaults
    defaults = GAParams()
    yaml_ga = getattr(config, "ga", None)
    ga_params = GAParams(
        population_size=(
            population_size
            if population_size is not None
            else (yaml_ga.population_size if yaml_ga else defaults.population_size)
        ),
        elite_fraction=(
            elite_fraction
            if elite_fraction is not None
            else (yaml_ga.elite_fraction if yaml_ga else defaults.elite_fraction)
        ),
        mutation_rate=(
            mutation_rate
            if mutation_rate is not None
            else (yaml_ga.mutation_rate if yaml_ga else defaults.mutation_rate)
        ),
        mutation_sigma=(
            mutation_sigma
            if mutation_sigma is not None
            else (yaml_ga.mutation_sigma if yaml_ga else defaults.mutation_sigma)
        ),
        crossover=(
            crossover
            if crossover is not None
            else (yaml_ga.crossover if yaml_ga else defaults.crossover)
        ),
        tournament_size=(
            tournament_size
            if tournament_size is not None
            else (yaml_ga.tournament_size if yaml_ga else defaults.tournament_size)
        ),
    )

    # Log resolved GA params
    tracker.log_metrics(
        {
            "ga/population_size": ga_params.population_size,
            "ga/elite_fraction": ga_params.elite_fraction,
            "ga/mutation_rate": ga_params.mutation_rate,
            "ga/mutation_sigma": ga_params.mutation_sigma,
            "ga/tournament_size": ga_params.tournament_size,
        }
    )

    best_x = None
    best_score = -np.inf

    def on_pop(res_list: List[dict], pop_arr: np.ndarray, step: int, info: dict):
        # population stats
        log_population(res_list, step)

        # Compute CSV row values
        generation = int(
            info.get("generation", max(1, step // ga_params.population_size))
        )
        gen_best = info.get("gen_best")
        gen_mean = info.get("gen_mean")
        gen_std = info.get("gen_std")
        best_so_far = info.get("best_so_far")
        eval_seconds = info.get("eval_seconds", 0.0)
        timestamp = info.get("timestamp", "")
        evaluations = int(info.get("evaluations", 0))
        cache_hits = int(info.get("cache_hits", 0))
        failed_evals = int(info.get("failed_evals", 0))
        crossover_children = int(info.get("crossover_children", 0))
        crossover_type = info.get("crossover_type", ga_params.crossover)
        immigrants = int(info.get("immigrants", 0))

        print(
            f"[GA] gen={generation} best={gen_best} mean={gen_mean} std={gen_std} "
            f"evaluated={evaluations} cache_hits={cache_hits} failed={failed_evals} "
            f"crossover_children={crossover_children} type={crossover_type} immigrants={immigrants}"
        )

        # Write/append CSV history for offline tracking
        try:
            hist_path = os.path.join(storage_path, "ga_history.csv")
            header = (
                "generation,fevals,gen_best,gen_mean,gen_std,best_so_far,mutation_sigma,"
                "eval_seconds,timestamp,evaluations,cache_hits,failed_evals,crossover_children,"
                "crossover_type,immigrants\n"
            )
            line = (
                f"{generation},{step},{gen_best},{gen_mean},{gen_std},{best_so_far},"
                f"{ga_params.mutation_sigma},{eval_seconds},{timestamp},{evaluations},"
                f"{cache_hits},{failed_evals},{crossover_children},{crossover_type},{immigrants}\n"
            )
            if not os.path.exists(hist_path):
                with open(hist_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write(line)
            else:
                with open(hist_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logging.warning("Failed to write ga_history.csv", exc_info=e)

        # Log per-generation aggregates and extras
        tracker.log_metrics(
            {
                "ga/generation": generation,
                "ga/mutation_sigma": float(ga_params.mutation_sigma),
                "population/eval_seconds": float(eval_seconds),
                "population/evaluations": float(evaluations),
                "population/cache_hits": float(cache_hits),
                "population/failed_evals": float(failed_evals),
                "population/gen_best": (
                    float(gen_best) if gen_best is not None else None
                ),
                "population/gen_mean": (
                    float(gen_mean) if gen_mean is not None else None
                ),
                "population/gen_std": float(gen_std) if gen_std is not None else None,
                "global/best_so_far": (
                    float(best_so_far) if best_so_far is not None else None
                ),
                "ga/crossover_children": float(crossover_children),
                "ga/immigrants": float(immigrants),
            },
            step=step,
        )

        # Log top-5 scores
        scores = [r["score"] for r in res_list if r["score"] is not None]
        if scores:
            scores_sorted = sorted(scores, reverse=True)[:5]
            top_scores = {"population/top1": float(scores_sorted[0])}
            for k, v in enumerate(scores_sorted[1:], start=2):
                top_scores[f"population/top{k}"] = float(v)
            tracker.log_metrics(top_scores, step=step)

    def on_best(x: np.ndarray, score: float, step: int):
        nonlocal best_x, best_score
        best_x = x.copy()
        best_score = score
        print(f"New best score: {best_score:.4f}")
        save_best_config(best_x)
        log_best(best_x, best_score, step=step)

    # Choose optimizer based on genome type and parameters
    use_enhanced = (
        genome_type == 'multi_method' or 
        hasattr(config.ga, 'semantic_crossover_prob') or
        getattr(config.ga, 'crossover', None) == 'semantic'
    )
    
    if use_enhanced:
        # Convert GAParams to EnhancedGAParams for new features
        enhanced_params = EnhancedGAParams()
        for key, value in ga_params.__dict__.items():
            if hasattr(enhanced_params, key):
                setattr(enhanced_params, key, value)
        
        # Set additional enhanced parameters from config
        if hasattr(config.ga, 'semantic_crossover_prob') and config.ga.semantic_crossover_prob is not None:
            enhanced_params.semantic_crossover_prob = config.ga.semantic_crossover_prob
        if hasattr(config.ga, 'method_mutation_rate') and config.ga.method_mutation_rate is not None:
            enhanced_params.method_mutation_rate = config.ga.method_mutation_rate
        if hasattr(config.ga, 'model_mutation_rate') and config.ga.model_mutation_rate is not None:
            enhanced_params.model_mutation_rate = config.ga.model_mutation_rate
        if hasattr(config.ga, 'parameter_mutation_rate') and config.ga.parameter_mutation_rate is not None:
            enhanced_params.parameter_mutation_rate = config.ga.parameter_mutation_rate
            
        optimizer = EnhancedGAOptimizer(
            genome=genome,
            strategy=strat,
            params=enhanced_params,
            random_init=config.random_init,
            seed=random_seed,
            on_population_evaluated=on_pop,
            on_new_best=on_best,
        )
    else:
        # Use traditional optimizer
        optimizer = GAOptimizer(
            genome=genome,
            strategy=strat,
            params=ga_params,
            random_init=config.random_init,
            seed=random_seed,
            on_population_evaluated=on_pop,
            on_new_best=on_best,
        )

    if baseline_csv_path:
        stage_log(
            "Stage-GA",
            f"Baseline metrics located at {baseline_csv_path}",
        )
    else:
        stage_log("Stage-GA", "No baseline metrics available for this run.")

    stage_log("Stage-GA", "Starting GA optimization loop...")
    try:
        best_x, best_score = optimizer.run(max_fevals=max_fevals, timeout=timeout)
    except KeyboardInterrupt:
        ray.shutdown()

    stage_log("Stage-GA", "Optimization complete.")
    stage_log("Stage-GA", f"Best score achieved: {best_score:.4f}")

    # pause for a bit to let any CUDA-using processes clean up
    time.sleep(1.0)

    # save the best merge configuration using original model references
    if best_x is not None:
        if genome_type == 'multi_method':
            genome_pretty = MultiMethodGenome(
                MultiMethodGenomeDefinition.model_validate(config.genome.model_dump()),
                trust_remote_code=trust_remote_code
            )
            best_config = genome_pretty.genotype_to_merge_config(best_x)
        else:
            genome_pretty = ModelGenome(config.genome, trust_remote_code=trust_remote_code)
            best_config = genome_pretty.genotype_merge_config(best_x)
            
        stage_log("Stage-GA", "Best merge configuration computed.")
        print(best_config.to_yaml())

        if save_final_model:
            stage_log("Stage-GA", "Saving final merged model artifacts...")
            run_merge(best_config, os.path.join(storage_path, "final_model"), merge_options)
            
            # Upload to Hugging Face if requested
            if hf_model_id:
                stage_log("Stage-GA", f"Uploading final model to Hugging Face: {hf_model_id}")
                try:
                    from huggingface_hub import upload_folder
                    
                    final_model_path = os.path.join(storage_path, "final_model")
                    upload_folder(
                        repo_id=hf_model_id,
                        folder_path=final_model_path,
                        repo_type="model",
                    )
                    stage_log("Stage-GA", f"Model successfully uploaded to {hf_model_id}")
                except Exception as e:
                    stage_log(
                        "Stage-GA",
                        f"Failed to upload model to Hugging Face: {e}",
                        level=logging.ERROR,
                    )
                    stage_log(
                        "Stage-GA",
                        f"You can manually upload from: {os.path.join(storage_path, 'final_model')}",
                    )
    else:
        stage_log(
            "Stage-GA",
            "No valid solution found. All evaluations failed.",
            level=logging.ERROR,
        )
        stage_log("Stage-GA", "Possible causes:")
        stage_log("Stage-GA", "- Model compatibility issues")
        stage_log("Stage-GA", "- Evaluation environment problems")
        stage_log("Stage-GA", "- Insufficient population size or evaluations")


def run_baseline_evaluations(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
) -> Optional[str]:
    """Execute baseline evaluations for all models defined in the genome."""
    stage_log("Stage-Baseline", "Starting baseline evaluation phase...")

    storage_dir = os.path.abspath(storage_path)
    os.makedirs(storage_dir, exist_ok=True)

    models: List[ModelReference] = list(config.genome.models)
    if getattr(config.genome, "base_model", None) is not None:
        models.append(config.genome.base_model)

    if not models:
        stage_log("Stage-Baseline", "No models found in the genome; skipping baselines.")
        return None

    task_manager = lm_eval.tasks.TaskManager(
        include_path=list(task_search_path) or None
    )

    use_cuda = torch.cuda.is_available() and (merge_cuda or (num_gpus or 0) > 0)
    device = "cuda" if use_cuda else "cpu"
    stage_log(
        "Stage-Baseline",
        f"Using {'CUDA' if use_cuda else 'CPU'} for baseline evaluations.",
    )

    metric_columns = [f"{task.name}:{task.metric}" for task in config.tasks]
    baseline_rows: List[Dict[str, Optional[float]]] = []
    successes = 0
    failures = 0

    for model_ref in models:
        model_name = str(model_ref)
        row: Dict[str, Optional[float]] = {
            "model": model_name,
            "weighted_score": None,
            "error": None,
        }
        for column in metric_columns:
            row.setdefault(column, None)

        stage_log("Stage-Baseline", f"Evaluating {model_name}...")
        model_args = {
            "pretrained": model_name,
            "dtype": "float32",
            "use_cache": True,
            "trust_remote_code": trust_remote_code,
        }
        eval_kwargs = {"device": device}

        try:
            result = _eval_model(
                "huggingface",
                config.tasks,
                model_args,
                num_fewshot=config.num_fewshot,
                limit=config.limit,
                batch_size=batch_size,
                task_manager=task_manager,
                **eval_kwargs,
            )
        except Exception as exc:  # pragma: no cover - evaluation depends on environment
            failures += 1
            row["error"] = str(exc)
            stage_log(
                "Stage-Baseline",
                f"Evaluation failed for {model_name}: {exc}",
                level=logging.ERROR,
            )
            LOGGER.debug("Baseline evaluation error", exc_info=exc)
            baseline_rows.append(row)
            continue

        successes += 1
        weighted_score = result.get("score")
        row["weighted_score"] = weighted_score

        for task_cfg in config.tasks:
            task_results = result["results"].get(task_cfg.name, {})
            metric_value = task_results.get(task_cfg.metric)

            if metric_value is None:
                metric_alternatives = {
                    "ppl,none": [
                        "word_perplexity,none",
                        "perplexity,none",
                        "byte_perplexity,none",
                    ],
                    "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
                    "acc_norm,none": [
                        "acc_norm,none",
                        "acc,none",
                        "accuracy,none",
                    ],
                }
                for alt_metric in metric_alternatives.get(task_cfg.metric, []):
                    if alt_metric in task_results:
                        metric_value = task_results[alt_metric]
                        break

                if metric_value is None:
                    lowered_metric = task_cfg.metric.lower()
                    for metric_name, value in task_results.items():
                        lowered_name = metric_name.lower()
                        if "stderr" in lowered_name:
                            continue
                        if (
                            ("ppl" in lowered_metric or "perplexity" in lowered_metric)
                            and "perplexity" in lowered_name
                        ):
                            metric_value = value
                            break
                        if "acc" in lowered_metric and "acc" in lowered_name:
                            metric_value = value
                            break

            if isinstance(metric_value, float) and math.isnan(metric_value):
                metric_value = None

            row[f"{task_cfg.name}:{task_cfg.metric}"] = metric_value

        baseline_rows.append(row)

    if not baseline_rows:
        stage_log("Stage-Baseline", "No baseline results recorded; skipping CSV output.")
        return None

    baseline_df = pandas.DataFrame(baseline_rows)
    ordered_columns = ["model", "weighted_score", *metric_columns, "error"]
    # Ensure DataFrame includes expected columns even if absent from rows
    for column in ordered_columns:
        if column not in baseline_df.columns:
            baseline_df[column] = None
    baseline_df = baseline_df[ordered_columns]
    baseline_df.sort_values(
        "weighted_score",
        ascending=False,
        inplace=True,
        na_position="last",
    )

    baseline_csv_path = os.path.join(storage_dir, "baseline_results.csv")
    baseline_df.to_csv(baseline_csv_path, index=False)

    stage_log(
        "Stage-Baseline",
        f"Baseline metrics saved to {baseline_csv_path}",
    )
    stage_log(
        "Stage-Baseline",
        f"Completed evaluations: {successes}; failures: {failures}",
    )

    return baseline_csv_path


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    merged = model.merged(
        cache_dir=merge_cache,
        trust_remote_code=trust_remote_code,
    )
    out_path = os.path.join(
        storage_path,
        "input_models",
        merged.model._unique_id(),
    )

    if os.path.exists(out_path):
        logging.info(f"Using existing resharded model at {out_path}")
        return ModelReference(model=out_path)

    model_hf = call_with_dtype(
        transformers.AutoModelForCausalLM.from_pretrained,
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
        cache_dir=os.path.join(storage_path, "transformers_cache"),
    )
    model_hf.save_pretrained(
        out_path, safe_serialization=True, out_shard_size=1_000_000_000_000
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.model.path,
            revision=model.model.revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logging.warning(f"Could not save tokenizer for {model.model}", exc_info=e)

    return ModelReference(model=out_path)


if __name__ == "__main__":
    main()
