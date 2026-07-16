# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from copy import deepcopy
from dataclasses import fields
from typing import List, Literal

from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.enhanced_ga import EnhancedGAParams
from mergekit.evo.ga import GAParams

OptimizerKind = Literal["standard", "enhanced"]

_ENHANCED_CONFIG_FIELDS = (
    "semantic_crossover_prob",
    "method_mutation_rate",
    "model_mutation_rate",
    "parameter_mutation_rate",
    "adaptive_method_sampling",
    "initial_method_probs",
    "operator_temperature",
    "operator_update_smoothing",
    "operator_avg_child_weight",
    "operator_parent_improvement_weight",
    "operator_survival_weight",
    "passthrough_penalty",
    "passthrough_max_fraction",
    "explorer_fraction",
    "diversity_parent_selection",
    "diversity_parent_weight",
    "gene_diversity_bonus_weight",
    "behavior_diversity_bonus_weight",
    "archive_novelty_bonus_weight",
    "novelty_archive_size",
    "rank_objective_weights",
)


def enhanced_optimizer_requirements(
    config: EvolMergeConfiguration,
    genome_type: str,
) -> List[str]:
    """Describe features that cannot run on the standard optimizer."""
    requirements: List[str] = []
    if genome_type == "multi_method":
        requirements.append("multi_method genome")
    if config.fitness_mode == "weighted_rank":
        requirements.append("weighted_rank fitness")

    ga_config = config.ga
    if ga_config is not None:
        for field_name in _ENHANCED_CONFIG_FIELDS:
            value = getattr(ga_config, field_name, None)
            if value is None or value is False:
                continue
            requirements.append(f"ga.{field_name}")
        if ga_config.crossover == "semantic":
            requirements.append("ga.crossover=semantic")
    return requirements


def resolve_optimizer_kind(
    config: EvolMergeConfiguration,
    genome_type: str,
) -> OptimizerKind:
    """Resolve the optimizer while retaining the historical `auto` behavior."""
    requirements = enhanced_optimizer_requirements(config, genome_type)
    requested = config.optimizer
    if requested == "standard":
        if requirements:
            raise ValueError(
                "optimizer='standard' is incompatible with: " + ", ".join(requirements)
            )
        return "standard"
    if requested == "enhanced":
        return "enhanced"

    # Before optimizer selection became explicit, any YAML `ga` block selected
    # EnhancedGAOptimizer. Keep that behavior for existing configurations.
    if requirements or config.ga is not None:
        return "enhanced"
    return "standard"


def build_enhanced_ga_params(
    base_params: GAParams,
    config: EvolMergeConfiguration,
) -> EnhancedGAParams:
    """Build enhanced parameters from resolved CLI/base and YAML settings."""
    enhanced = EnhancedGAParams()
    enhanced_field_names = {field.name for field in fields(EnhancedGAParams)}
    for field in fields(GAParams):
        if field.name in enhanced_field_names:
            setattr(enhanced, field.name, deepcopy(getattr(base_params, field.name)))

    if config.ga is not None:
        for field_name in _ENHANCED_CONFIG_FIELDS:
            value = getattr(config.ga, field_name, None)
            if value is not None:
                setattr(enhanced, field_name, deepcopy(value))

    enhanced.fitness_mode = config.fitness_mode
    return enhanced
