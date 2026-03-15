import math

import pandas

from mergekit.scripts.evolve_ga import (
    _best_weighted_score_from_frame,
    _meets_improvement_thresholds,
    _score_improvement,
)


def test_best_weighted_score_from_frame_uses_max_normalized_score():
    frame = pandas.DataFrame(
        {"weighted_score": ["-64.27", -63.73, None, "not-a-number", float("nan")]}
    )

    assert _best_weighted_score_from_frame(frame) == -63.73


def test_score_improvement_uses_baseline_magnitude_for_negative_scores():
    delta, pct = _score_improvement(-63.73, -64.27)

    assert math.isclose(delta, 0.54)
    assert math.isclose(pct, 0.54 / 64.27 * 100.0)


def test_score_improvement_handles_positive_scores():
    delta, pct = _score_improvement(0.84, 0.80)

    assert math.isclose(delta, 0.04)
    assert math.isclose(pct, 5.0)


def test_score_improvement_returns_undefined_pct_for_zero_baseline():
    delta, pct = _score_improvement(0.25, 0.0)

    assert math.isclose(delta, 0.25)
    assert pct is None


def test_improvement_thresholds_accept_negative_baseline_improvement():
    delta, pct = _score_improvement(-63.73, -64.27)

    assert _meets_improvement_thresholds(delta, pct, min_abs=0.1, min_pct=0.5)


def test_improvement_thresholds_reject_undefined_pct_when_pct_required():
    delta, pct = _score_improvement(0.25, 0.0)

    assert not _meets_improvement_thresholds(delta, pct, min_abs=0.1, min_pct=1.0)
