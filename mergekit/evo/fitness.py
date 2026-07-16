# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import os
from typing import Any, Dict

from mergekit.evo.checkpoint import atomic_write_json
from mergekit.evo.config import PHASE1_TASK_MIX_PROFILES, EvolMergeConfiguration

FITNESS_DEFINITION_FILENAME = "fitness_definition.json"

_TRANSFORM_FORMULAS = {
    "legacy_reciprocal": "1 / (1 + max(0, x))",
    "log_reciprocal": "1 / (1 + log1p(max(0, x)))",
}


def fitness_definition_payload(config: EvolMergeConfiguration) -> Dict[str, Any]:
    """Return the complete, serializable fitness definition for a run."""
    profile = PHASE1_TASK_MIX_PROFILES.get(config.task_mix_profile or "")
    profile_weights = None
    if profile is not None:
        profile_weights = {
            key: float(profile[key])
            for key in (
                "task_score_weight",
                "language_quality_weight",
                "stability_weight",
                "diversity_bonus_weight",
            )
        }

    transform = config.fitness.lower_is_better_transform
    return {
        "schema_version": 1,
        "fitness_version": config.fitness.version,
        "fitness_mode": config.fitness_mode,
        "lower_is_better_transform": transform,
        "lower_is_better_formula": _TRANSFORM_FORMULAS[transform],
        "task_mix_profile": config.task_mix_profile,
        "task_mix_weights": profile_weights,
    }


def write_fitness_definition(
    storage_path: str,
    config: EvolMergeConfiguration,
) -> str:
    return atomic_write_json(
        os.path.join(storage_path, FITNESS_DEFINITION_FILENAME),
        fitness_definition_payload(config),
    )


def ensure_fitness_definition(
    storage_path: str,
    config: EvolMergeConfiguration,
    *,
    resume: bool,
) -> str:
    """Write the definition, refusing to relabel an existing resumed run."""
    definition_path = os.path.join(storage_path, FITNESS_DEFINITION_FILENAME)
    expected = fitness_definition_payload(config)
    if resume and os.path.isfile(definition_path):
        with open(definition_path, "r", encoding="utf-8") as definition_file:
            existing = json.load(definition_file)
        if existing != expected:
            raise ValueError(
                "Fitness definition does not match the existing resumed run: "
                f"{definition_path}"
            )
    return atomic_write_json(definition_path, expected)
