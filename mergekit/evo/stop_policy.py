import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple

TargetReference = Literal["best_baseline"]


@dataclass
class StopPolicyParams:
    target_improvement_abs: Optional[float] = None
    target_improvement_pct: Optional[float] = None
    target_reference: TargetReference = "best_baseline"
    target_reference_score: Optional[float] = None
    min_generations_before_target_stop: int = 0
    require_stage2_for_target: bool = False
    stagnation_patience_generations: int = 0
    stagnation_min_delta: float = 0.0


@dataclass
class StopDetails:
    reason: str
    generation: int
    fevals: int
    elapsed_seconds: float
    best_score: Optional[float]
    best_generation: Optional[int] = None
    best_score_source: Optional[str] = None
    target_reference: Optional[str] = None
    target_reference_score: Optional[float] = None
    best_improvement_abs: Optional[float] = None
    best_improvement_pct: Optional[float] = None
    require_stage2_for_target: Optional[bool] = None
    min_generations_before_target_stop: Optional[int] = None
    stagnation_generations: Optional[int] = None
    stagnation_patience_generations: Optional[int] = None
    stagnation_min_delta: Optional[float] = None
    max_fevals: Optional[int] = None
    timeout_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def score_improvement(
    candidate_score: float, reference_score: float
) -> Tuple[float, Optional[float]]:
    delta = float(candidate_score) - float(reference_score)
    denominator = abs(float(reference_score))
    if denominator <= 0.0:
        return delta, None
    return delta, (delta / denominator) * 100.0


def score_source_is_target_eligible(score_source: Optional[str]) -> bool:
    return score_source != "stage1"


def evaluate_target_stop(
    *,
    best_score: float,
    best_score_source: Optional[str],
    generation: int,
    fevals: int,
    elapsed_seconds: float,
    best_generation: Optional[int],
    policy: StopPolicyParams,
) -> Optional[StopDetails]:
    if policy.target_improvement_abs is None and policy.target_improvement_pct is None:
        return None
    if not math.isfinite(float(best_score)):
        return None
    if generation < max(0, int(policy.min_generations_before_target_stop)):
        return None
    if policy.require_stage2_for_target and not score_source_is_target_eligible(
        best_score_source
    ):
        return None

    reference_score = policy.target_reference_score
    if reference_score is None or not math.isfinite(float(reference_score)):
        return None

    delta, pct = score_improvement(float(best_score), float(reference_score))
    abs_met = policy.target_improvement_abs is None or delta >= float(
        policy.target_improvement_abs
    )
    pct_met = policy.target_improvement_pct is None or (
        pct is not None and pct >= float(policy.target_improvement_pct)
    )
    if not (abs_met and pct_met):
        return None

    return StopDetails(
        reason="target_improvement",
        generation=int(generation),
        fevals=int(fevals),
        elapsed_seconds=float(elapsed_seconds),
        best_score=float(best_score),
        best_generation=None if best_generation is None else int(best_generation),
        best_score_source=best_score_source,
        target_reference=str(policy.target_reference),
        target_reference_score=float(reference_score),
        best_improvement_abs=float(delta),
        best_improvement_pct=None if pct is None else float(pct),
        require_stage2_for_target=bool(policy.require_stage2_for_target),
        min_generations_before_target_stop=int(
            policy.min_generations_before_target_stop
        ),
    )


def evaluate_stagnation_stop(
    *,
    best_score: float,
    best_score_source: Optional[str],
    generation: int,
    fevals: int,
    elapsed_seconds: float,
    best_generation: Optional[int],
    stagnation_generations: int,
    policy: StopPolicyParams,
) -> Optional[StopDetails]:
    patience = max(0, int(policy.stagnation_patience_generations))
    if patience <= 0 or stagnation_generations < patience:
        return None

    return StopDetails(
        reason="stagnation",
        generation=int(generation),
        fevals=int(fevals),
        elapsed_seconds=float(elapsed_seconds),
        best_score=float(best_score) if math.isfinite(float(best_score)) else None,
        best_generation=None if best_generation is None else int(best_generation),
        best_score_source=best_score_source,
        stagnation_generations=int(stagnation_generations),
        stagnation_patience_generations=patience,
        stagnation_min_delta=float(policy.stagnation_min_delta),
    )
