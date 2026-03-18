# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, model_validator

from mergekit.evo.genome import ModelGenomeDefinition
from mergekit.evo.multi_method_genome import MultiMethodGenomeDefinition

PHASE1_TASK_MIX_PROFILES: Dict[str, Dict[str, Union[float, Tuple[str, ...]]]] = {
    "tiny_local_default": {
        "allowed_methods": ("passthrough", "linear", "slerp"),
        "max_mutation_rate": 0.15,
        "max_mutation_sigma": 0.02,
        "task_score_weight": 0.55,
        "language_quality_weight": 0.35,
        "stability_weight": 0.10,
        "diversity_bonus_weight": 0.05,
    },
    "pythia70m_phase1": {
        "allowed_methods": ("passthrough", "linear", "slerp"),
        "max_mutation_rate": 0.15,
        "max_mutation_sigma": 0.02,
        "task_score_weight": 0.55,
        "language_quality_weight": 0.35,
        "stability_weight": 0.10,
        "diversity_bonus_weight": 0.05,
    },
    "pythia70m_phase3": {
        "allowed_methods": (
            "passthrough",
            "linear",
            "slerp",
            "ties",
            "dare_linear",
            "dare_ties",
        ),
        "max_mutation_rate": 0.15,
        "max_mutation_sigma": 0.02,
        "task_score_weight": 0.45,
        "language_quality_weight": 0.25,
        "stability_weight": 0.15,
        "diversity_bonus_weight": 0.15,
    },
}


class GAOptimizerConfiguration(BaseModel, frozen=True):
    model_config = {"protected_namespaces": ()}  # Allow model_ fields

    population_size: int = 32
    elite_fraction: float = 0.125
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.05
    crossover: Literal["arithmetic", "uniform", "sbx", "semantic"] = "arithmetic"
    tournament_size: int = 4

    # Enhanced GA parameters for semantic operations
    semantic_crossover_prob: Optional[float] = None
    method_mutation_rate: Optional[float] = None
    model_mutation_rate: Optional[float] = None
    parameter_mutation_rate: Optional[float] = None
    adaptive_method_sampling: bool = False
    initial_method_probs: Optional[Dict[str, float]] = None
    operator_temperature: Optional[float] = None
    operator_update_smoothing: Optional[float] = None
    operator_avg_child_weight: Optional[float] = None
    operator_parent_improvement_weight: Optional[float] = None
    operator_survival_weight: Optional[float] = None
    passthrough_penalty: Optional[float] = None
    passthrough_max_fraction: Optional[float] = None
    explorer_fraction: Optional[float] = None
    diversity_parent_selection: bool = False
    diversity_parent_weight: Optional[float] = None
    gene_diversity_bonus_weight: Optional[float] = None
    behavior_diversity_bonus_weight: Optional[float] = None
    archive_novelty_bonus_weight: Optional[float] = None
    novelty_archive_size: Optional[int] = None
    rank_objective_weights: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def validate_adaptive_settings(self):
        if self.initial_method_probs is not None:
            if not self.initial_method_probs:
                raise ValueError("initial_method_probs must not be empty")
            if any(value < 0 for value in self.initial_method_probs.values()):
                raise ValueError("initial_method_probs values must be non-negative")
            if sum(float(value) for value in self.initial_method_probs.values()) <= 0:
                raise ValueError("initial_method_probs must contain a positive mass")

        if self.operator_temperature is not None and self.operator_temperature <= 0:
            raise ValueError("operator_temperature must be > 0")

        if self.operator_update_smoothing is not None and not (
            0.0 <= self.operator_update_smoothing <= 1.0
        ):
            raise ValueError("operator_update_smoothing must be in [0, 1]")

        if self.passthrough_penalty is not None and self.passthrough_penalty < 0:
            raise ValueError("passthrough_penalty must be >= 0")

        if self.passthrough_max_fraction is not None and not (
            0.0 <= self.passthrough_max_fraction <= 1.0
        ):
            raise ValueError("passthrough_max_fraction must be in [0, 1]")

        if self.explorer_fraction is not None and not (
            0.0 <= self.explorer_fraction <= 1.0
        ):
            raise ValueError("explorer_fraction must be in [0, 1]")

        if (
            self.diversity_parent_weight is not None
            and self.diversity_parent_weight < 0
        ):
            raise ValueError("diversity_parent_weight must be >= 0")
        if (
            self.gene_diversity_bonus_weight is not None
            and self.gene_diversity_bonus_weight < 0
        ):
            raise ValueError("gene_diversity_bonus_weight must be >= 0")
        if (
            self.behavior_diversity_bonus_weight is not None
            and self.behavior_diversity_bonus_weight < 0
        ):
            raise ValueError("behavior_diversity_bonus_weight must be >= 0")
        if (
            self.archive_novelty_bonus_weight is not None
            and self.archive_novelty_bonus_weight < 0
        ):
            raise ValueError("archive_novelty_bonus_weight must be >= 0")
        if self.novelty_archive_size is not None and self.novelty_archive_size <= 0:
            raise ValueError("novelty_archive_size must be > 0")
        if self.rank_objective_weights is not None:
            if not self.rank_objective_weights:
                raise ValueError("rank_objective_weights must not be empty")
            if any(value < 0 for value in self.rank_objective_weights.values()):
                raise ValueError("rank_objective_weights values must be non-negative")
            if sum(float(value) for value in self.rank_objective_weights.values()) <= 0:
                raise ValueError("rank_objective_weights must contain a positive mass")

        weights = [
            self.operator_avg_child_weight,
            self.operator_parent_improvement_weight,
            self.operator_survival_weight,
        ]
        provided = [weight for weight in weights if weight is not None]
        if provided and any(weight < 0 for weight in provided):
            raise ValueError("operator score weights must be non-negative")

        return self


class TaskConfiguration(BaseModel, frozen=True):
    name: str
    weight: float = 1.0
    metric: str = "acc,none"

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class EvolMergeConfiguration(BaseModel, frozen=True):
    genome: Union[
        MultiMethodGenomeDefinition, ModelGenomeDefinition
    ]  # Try multi-method first
    tasks: List[TaskConfiguration]
    two_stage: bool = False
    stage1_tasks: Optional[List[TaskConfiguration]] = None
    stage1_limit: Optional[int] = None
    stage2_limit: Optional[int] = None
    stage2_top_k: Optional[int] = None
    fitness_mode: Literal["weighted_sum", "structured_phase1_tiny", "weighted_rank"] = (
        "weighted_sum"
    )
    task_mix_profile: Optional[str] = None
    behavior_prompts: Optional[List[str]] = None
    behavior_probe_max_new_tokens: int = 24
    behavior_repetition_ngram_size: int = 4
    behavior_min_distinct_ratio: float = 0.2
    behavior_reject_on_degenerate: bool = False
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    shuffle: bool = False
    random_init: bool = False
    ga: Optional[GAOptimizerConfiguration] = None
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = True

    @model_validator(mode="after")
    def validate_stage_settings(self):
        if self.stage1_limit is not None and self.stage1_limit <= 0:
            raise ValueError("stage1_limit must be > 0")
        if self.stage2_limit is not None and self.stage2_limit <= 0:
            raise ValueError("stage2_limit must be > 0")
        if self.stage2_top_k is not None and self.stage2_top_k <= 0:
            raise ValueError("stage2_top_k must be > 0")
        if self.behavior_probe_max_new_tokens <= 0:
            raise ValueError("behavior_probe_max_new_tokens must be > 0")
        if self.behavior_repetition_ngram_size <= 0:
            raise ValueError("behavior_repetition_ngram_size must be > 0")
        if not 0.0 <= self.behavior_min_distinct_ratio <= 1.0:
            raise ValueError("behavior_min_distinct_ratio must be in [0, 1]")
        if self.two_stage:
            if self.stage2_top_k is not None and self.stage2_top_k < 1:
                raise ValueError("two_stage requires stage2_top_k >= 1")
        if self.fitness_mode == "structured_phase1_tiny" and not self.task_mix_profile:
            raise ValueError(
                "structured_phase1_tiny requires task_mix_profile to be set"
            )
        if self.fitness_mode == "weighted_rank" and self.ga is not None:
            if (
                self.ga.rank_objective_weights is not None
                and not self.ga.rank_objective_weights
            ):
                raise ValueError(
                    "weighted_rank requires rank_objective_weights to be non-empty"
                )
        if self.task_mix_profile is not None:
            profile = PHASE1_TASK_MIX_PROFILES.get(self.task_mix_profile)
            if profile is None:
                raise ValueError(f"Unknown task_mix_profile: {self.task_mix_profile!r}")
            allowed_methods = tuple(
                str(method)
                for method in getattr(self.genome, "allowed_methods", []) or []
            )
            invalid_methods = sorted(
                set(allowed_methods) - set(profile["allowed_methods"])
            )
            if invalid_methods:
                raise ValueError(
                    "task_mix_profile "
                    f"{self.task_mix_profile!r} only supports methods "
                    f"{profile['allowed_methods']}; got {invalid_methods}"
                )
            if self.ga is not None:
                max_mutation_rate = float(profile["max_mutation_rate"])
                max_mutation_sigma = float(profile["max_mutation_sigma"])
                if self.ga.mutation_rate > max_mutation_rate:
                    raise ValueError(
                        "task_mix_profile "
                        f"{self.task_mix_profile!r} requires mutation_rate <= "
                        f"{max_mutation_rate}"
                    )
                if self.ga.mutation_sigma > max_mutation_sigma:
                    raise ValueError(
                        "task_mix_profile "
                        f"{self.task_mix_profile!r} requires mutation_sigma <= "
                        f"{max_mutation_sigma}"
                    )
        return self


NAUGHTY_PREFIXES = [
    "mmlu",
    "hendrycks",
    "agieval",
    "gsm8k",
    "hellaswag",
    "winogrande",
    "arc_",
    "ai2_arc",
    "truthfulqa",
    "bigbench",
    "piqa",
    "openbookqa",
    "leaderboard",
]


def check_for_naughty_config(config: EvolMergeConfiguration, allow: bool = False):
    """
    Check if the given configuration is naughty and should be disallowed.

    mergekit-evolve is perfectly set up to directly optimize against the test set
    of common benchmarks, which just makes the world a worse place. There are
    cases where this is useful but it deserves a giant honking warning.
    """
    suffix = ""
    if not allow:
        suffix = (
            " To proceed, set the "
            "--i-understand-the-depths-of-the-evils-i-am-unleashing flag."
        )
    for task in config.tasks:
        for prefix in NAUGHTY_PREFIXES:
            if task.name.startswith(prefix):
                if task.name.endswith("_train"):
                    # there aren't any tasks that match this pattern in base
                    # lm-eval, but it'd be a sane thing to do to add tasks for
                    # the training sets of these benchmarks. don't warn about
                    # them
                    continue

                message = (
                    f"Task {task.name} is a common benchmark task. "
                    "Optimizing against this task directly is unsporting at best "
                    "and outright malicious at worst. Using mergekit-evolve to "
                    "game benchmarks will be a black mark on your name for a "
                    f"thousand generations.{suffix}"
                )
                if not allow:
                    raise ValueError(message)
                else:
                    logging.warning(message)
