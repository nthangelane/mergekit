# Copyright (C) 2025 Nkululeko Thangelane
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_RANK_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "raw_weighted_score": 0.20,
    "task_score": 0.25,
    "language_quality": 0.20,
    "stability_score": 0.15,
    "gene_diversity_score": 0.10,
    "behavior_diversity_score": 0.05,
    "archive_novelty_score": 0.05,
}


def _normalize_objective_weights(
    objective_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    weights = {
        key: float(value)
        for key, value in (
            objective_weights.items()
            if objective_weights is not None
            else DEFAULT_RANK_OBJECTIVE_WEIGHTS.items()
        )
        if float(value) > 0.0
    }
    total = float(sum(weights.values()))
    if total <= 0.0:
        return {"score": 1.0}
    return {key: value / total for key, value in weights.items()}


def _result_objectives(result: Dict[str, Any]) -> Dict[str, float]:
    objectives = {}
    for key in ("fitness_objectives", "fitness_components"):
        payload = result.get(key) or {}
        if isinstance(payload, dict):
            for obj_name, obj_value in payload.items():
                try:
                    objectives[str(obj_name)] = float(obj_value)
                except (TypeError, ValueError):
                    continue
    if "score" in result and result.get("score") is not None:
        objectives.setdefault("score", float(result["score"]))
    if "raw_score" in result and result.get("raw_score") is not None:
        objectives.setdefault("raw_score", float(result["raw_score"]))
    return objectives


def _normalized_rank(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    unique_values = {float(value) for value in values.values()}
    if len(values) == 1 or len(unique_values) == 1:
        return {idx: 1.0 for idx in values}
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    denom = float(max(len(ordered) - 1, 1))
    return {
        idx: float(1.0 - (rank / denom)) for rank, (idx, _value) in enumerate(ordered)
    }


def weighted_rank_scores(
    results: List[Dict[str, Any]],
    objective_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    weights = _normalize_objective_weights(objective_weights)
    valid_indices = [
        idx for idx, result in enumerate(results) if result.get("score") is not None
    ]
    scores = np.full(len(results), -np.inf, dtype=np.float32)
    details: List[Dict[str, Any]] = [{} for _ in results]
    if not valid_indices:
        return scores, details

    total_scores = {idx: 0.0 for idx in valid_indices}
    per_objective_ranks = {idx: {} for idx in valid_indices}
    per_objective_values = {idx: {} for idx in valid_indices}
    objective_maps = {idx: _result_objectives(results[idx]) for idx in valid_indices}

    any_objective_used = False
    for objective_name, objective_weight in weights.items():
        values = {}
        for idx in valid_indices:
            objective_value = objective_maps[idx].get(objective_name)
            if objective_value is None or not np.isfinite(objective_value):
                continue
            values[idx] = float(objective_value)
            per_objective_values[idx][objective_name] = float(objective_value)

        if not values:
            continue

        objective_ranks = _normalized_rank(values)
        any_objective_used = True
        for idx, rank_score in objective_ranks.items():
            total_scores[idx] += float(objective_weight) * float(rank_score)
            per_objective_ranks[idx][objective_name] = float(rank_score)

    if not any_objective_used:
        fallback = _normalized_rank(
            {idx: float(results[idx]["score"]) for idx in valid_indices}
        )
        for idx, rank_score in fallback.items():
            total_scores[idx] = float(rank_score)
            per_objective_ranks[idx]["score"] = float(rank_score)
            per_objective_values[idx]["score"] = float(results[idx]["score"])

    for idx in valid_indices:
        scores[idx] = float(total_scores[idx])
        details[idx] = {
            "fitness_proxy": "weighted_rank",
            "weighted_rank_score": float(total_scores[idx]),
            "objective_values": per_objective_values[idx],
            "objective_rank_scores": per_objective_ranks[idx],
        }

    return scores, details
