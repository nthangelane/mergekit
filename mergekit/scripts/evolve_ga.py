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
import os
import time
from typing import List, Optional, Tuple

import click
import numpy as np
import pandas
import ray
import torch
import tqdm
import transformers
import yaml

try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference
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
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )

    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    # Initialize experiment tracking
    tracker = None
    if use_wandb and use_mlflow:
        raise ValueError(
            "Cannot use both wandb and mlflow at the same time. Choose one."
        )
    elif use_wandb:
        tracker = create_tracker("wandb")
        tracker.initialize(
            project_name=wandb_project or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            entity=wandb_entity,
        )
    elif use_mlflow:
        tracker = create_tracker("mlflow")
        tracker.initialize(
            project_name=mlflow_experiment or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            tracking_uri=mlflow_tracking_uri,
        )
    else:
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
    else:
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

        # Write/append CSV history for offline tracking
        try:
            hist_path = os.path.join(storage_path, "ga_history.csv")
            header = "generation,fevals,gen_best,gen_mean,gen_std,best_so_far,mutation_sigma,eval_seconds\n"
            line = f"{generation},{step},{gen_best},{gen_mean},{gen_std},{best_so_far},{ga_params.mutation_sigma},{eval_seconds}\n"
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

    try:
        best_x, best_score = optimizer.run(max_fevals=max_fevals, timeout=timeout)
    except KeyboardInterrupt:
        ray.shutdown()

    print("!!! OPTIMIZATION COMPLETE (GA) !!!")
    print(f"Best score: {best_score:.4f}")
    print()

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
            
        print("Best merge configuration:")
        print(best_config.to_yaml())

        if save_final_model:
            print("Saving final model...")
            run_merge(best_config, os.path.join(storage_path, "final_model"), merge_options)
    else:
        print("No valid solution found. All evaluations failed.")
        print("This may indicate:")
        print("- Model compatibility issues")
        print("- Evaluation environment problems")
        print("- Insufficient population size or evaluations")


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

    model_hf = transformers.AutoModelForCausalLM.from_pretrained(
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
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
