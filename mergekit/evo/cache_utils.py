import hashlib
from typing import Dict, Optional, Tuple

import numpy as np


def genotype_cache_key(x: np.ndarray, round_step: float) -> Tuple[int, ...]:
    arr = np.asarray(x, dtype=np.float32).ravel()
    step = max(float(round_step), 1e-9)
    quantized = np.round(arr / step).astype(np.int64)
    return tuple(quantized.tolist())


def genotype_exact_hash(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float32).ravel()
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def persisted_failure_result(
    failures_by_hash: Dict[str, dict],
    x: np.ndarray,
) -> Optional[dict]:
    if not failures_by_hash:
        return None

    genotype_hash = genotype_exact_hash(x)
    entry = failures_by_hash.get(genotype_hash)
    if entry is None:
        return None

    result = dict(entry)
    result.setdefault("score", None)
    result.setdefault("results", None)
    result.setdefault("error_stage", "merge")
    result.setdefault("error_type", "persisted_failed_genotype")
    result["error_message"] = str(
        result.get("error_message")
        or "Skipped due to persisted failed-genotype blacklist"
    )
    result.setdefault("genotype_hash", genotype_hash)
    result.setdefault("blacklisted", True)
    return result
