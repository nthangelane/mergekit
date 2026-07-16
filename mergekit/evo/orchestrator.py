# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from mergekit.common import ModelReference
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.ga import GAParams
from mergekit.evo.genome import ModelGenome, ModelGenomeDefinition
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)


def resolve_stop_configuration(
    config: EvolMergeConfiguration,
    *,
    max_fevals_cli: Optional[int],
    timeout_cli: Optional[float],
) -> Dict[str, Any]:
    stop_config = config.stop
    return {
        "max_fevals": (
            int(max_fevals_cli)
            if max_fevals_cli is not None
            else (
                int(stop_config.max_fevals)
                if stop_config is not None and stop_config.max_fevals is not None
                else 100
            )
        ),
        "timeout_seconds": (
            float(timeout_cli)
            if timeout_cli is not None
            else (
                float(stop_config.max_time_seconds)
                if stop_config is not None and stop_config.max_time_seconds is not None
                else None
            )
        ),
        "target_improvement_abs": (
            float(stop_config.target_improvement_abs)
            if stop_config is not None
            and stop_config.target_improvement_abs is not None
            else None
        ),
        "target_improvement_pct": (
            float(stop_config.target_improvement_pct)
            if stop_config is not None
            and stop_config.target_improvement_pct is not None
            else None
        ),
        "target_reference": (
            str(stop_config.target_reference)
            if stop_config is not None
            else "best_baseline"
        ),
        "min_generations_before_target_stop": (
            int(stop_config.min_generations_before_target_stop)
            if stop_config is not None
            else 0
        ),
        "require_stage2_for_target": bool(
            stop_config.require_stage2_for_target if stop_config is not None else False
        ),
        "stagnation_patience_generations": (
            int(stop_config.stagnation_patience_generations)
            if stop_config is not None
            and stop_config.stagnation_patience_generations is not None
            else 0
        ),
        "stagnation_min_delta": (
            float(stop_config.stagnation_min_delta) if stop_config is not None else 0.0
        ),
    }


def resolve_device(device: str, num_gpus: Optional[int]) -> str:
    normalized = str(device or "auto").lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if normalized == "cpu":
        if num_gpus is not None and num_gpus > 0:
            raise ValueError("--device cpu conflicts with --num-gpus > 0.")
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "--device cuda was requested, but torch.cuda.is_available() is false."
            )
        if num_gpus == 0:
            raise ValueError("--device cuda conflicts with --num-gpus 0.")
        return "cuda"
    if num_gpus is not None and num_gpus <= 0:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_merge_cuda(merge_cuda: bool, num_gpus: Optional[int]) -> bool:
    if not merge_cuda:
        return False
    if num_gpus is not None and num_gpus <= 0:
        return False
    return bool(torch.cuda.is_available())


def build_genome(
    config: EvolMergeConfiguration,
    models: List[ModelReference],
    base_model: Optional[ModelReference],
    *,
    trust_remote_code: bool,
) -> Tuple[str, Any]:
    genome_config = config.genome
    if isinstance(genome_config, MultiMethodGenomeDefinition):
        definition = MultiMethodGenomeDefinition.model_validate(
            {
                **genome_config.model_dump(exclude=["models", "base_model"]),
                "models": models,
                "base_model": base_model,
            }
        )
        return "multi_method", MultiMethodGenome(
            definition,
            trust_remote_code=trust_remote_code,
        )

    definition = ModelGenomeDefinition.model_validate(
        {
            **genome_config.model_dump(exclude=["models", "base_model"]),
            "models": models,
            "base_model": base_model,
        }
    )
    return "standard", ModelGenome(
        definition,
        trust_remote_code=trust_remote_code,
    )


def resolve_evaluation_strategy(strategy: str):
    if strategy == "pool":
        from mergekit.evo.strategy import ActorPoolEvaluationStrategy

        return ActorPoolEvaluationStrategy
    if strategy == "buffered":
        from mergekit.evo.strategy import BufferedRayEvaluationStrategy

        return BufferedRayEvaluationStrategy
    if strategy == "serial":
        from mergekit.evo.strategy import SerialEvaluationStrategy

        return SerialEvaluationStrategy
    raise ValueError(f"Unknown strategy {strategy}")


def resolve_ga_params(
    config: EvolMergeConfiguration,
    *,
    population_size: Optional[int],
    elite_fraction: Optional[float],
    mutation_rate: Optional[float],
    mutation_sigma: Optional[float],
    crossover: Optional[str],
    tournament_size: Optional[int],
    resolved_stop: Dict[str, Any],
    baseline_best_score: Optional[float],
) -> GAParams:
    defaults = GAParams()
    yaml_ga = config.ga

    def select(cli_value, field_name: str):
        if cli_value is not None:
            return cli_value
        return (
            getattr(yaml_ga, field_name)
            if yaml_ga is not None
            else getattr(defaults, field_name)
        )

    baseline_score = (
        float(baseline_best_score)
        if baseline_best_score is not None and math.isfinite(baseline_best_score)
        else None
    )
    return GAParams(
        population_size=select(population_size, "population_size"),
        elite_fraction=select(elite_fraction, "elite_fraction"),
        mutation_rate=select(mutation_rate, "mutation_rate"),
        mutation_sigma=select(mutation_sigma, "mutation_sigma"),
        crossover=select(crossover, "crossover"),
        tournament_size=select(tournament_size, "tournament_size"),
        target_improvement_abs=resolved_stop.get("target_improvement_abs"),
        target_improvement_pct=resolved_stop.get("target_improvement_pct"),
        target_reference=str(resolved_stop.get("target_reference") or "best_baseline"),
        target_reference_score=baseline_score,
        min_generations_before_target_stop=int(
            resolved_stop.get("min_generations_before_target_stop") or 0
        ),
        require_stage2_for_target=bool(
            resolved_stop.get("require_stage2_for_target", False)
        ),
        stagnation_patience_generations=int(
            resolved_stop.get("stagnation_patience_generations") or 0
        ),
        stagnation_min_delta=float(resolved_stop.get("stagnation_min_delta") or 0.0),
    )


def build_run_signature(
    config: EvolMergeConfiguration,
    ga_params: GAParams,
    *,
    random_seed: int,
    strategy: str,
    device: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    trust_remote_code: bool,
    vllm: bool,
    tensor_parallel_size: int,
    num_gpus: Optional[int],
) -> str:
    payload = {
        "config": config.model_dump(mode="json"),
        "ga_params": ga_params.__dict__,
        "random_seed": random_seed,
        "strategy": strategy,
        "device": device,
        "batch_size": batch_size,
        "merge_cuda": merge_cuda,
        "trust_remote_code": trust_remote_code,
        "vllm": vllm,
        "tensor_parallel_size": tensor_parallel_size,
        "num_gpus": num_gpus,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
