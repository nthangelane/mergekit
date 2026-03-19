import numpy as np
import pytest

from mergekit.evo.ranking import weighted_rank_scores


def test_weighted_rank_scores_prefers_best_weighted_objective_mix():
    results = [
        {
            "score": 0.9,
            "fitness_components": {
                "raw_weighted_score": 0.9,
                "task_score": 0.1,
                "language_quality": 0.1,
            },
        },
        {
            "score": 0.8,
            "fitness_components": {
                "raw_weighted_score": 0.8,
                "task_score": 0.9,
                "language_quality": 0.9,
            },
        },
    ]

    scores, details = weighted_rank_scores(
        results,
        objective_weights={
            "raw_weighted_score": 0.1,
            "task_score": 0.6,
            "language_quality": 0.3,
        },
    )

    assert scores[1] > scores[0]
    assert details[1]["fitness_proxy"] == "weighted_rank"
    assert details[1]["weighted_rank_score"] == pytest.approx(float(scores[1]))
    assert (
        details[1]["objective_rank_scores"]["task_score"]
        > details[0]["objective_rank_scores"]["task_score"]
    )


def test_weighted_rank_scores_falls_back_to_score_when_named_objectives_missing():
    results = [
        {"score": 0.3},
        {"score": 0.7},
        {"score": 0.5},
    ]

    scores, details = weighted_rank_scores(
        results,
        objective_weights={"task_score": 1.0},
    )

    assert np.all(np.isfinite(scores))
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.5)
    assert scores[0] == pytest.approx(0.0)
    assert details[1]["objective_rank_scores"] == {"score": 1.0}
    assert details[2]["objective_values"] == {"score": 0.5}
