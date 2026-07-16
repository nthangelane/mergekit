import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

GA_STATE_FILENAME = "ga_state.json"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _encode_numpy_state(state) -> Dict[str, Any]:
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _decode_numpy_state(state: Dict[str, Any]):
    return (
        str(state["bit_generator"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def capture_rng_state(
    random_state: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    python_state = random.getstate()
    result = {
        "python": {
            "version": int(python_state[0]),
            "state": list(python_state[1]),
            "gauss": python_state[2],
        },
        "numpy": _encode_numpy_state(np.random.get_state()),
        "torch": torch.get_rng_state().tolist(),
        "torch_cuda": [],
    }
    if random_state is not None:
        result["optimizer_numpy"] = _encode_numpy_state(random_state.get_state())
    if torch.cuda.is_available():
        result["torch_cuda"] = [
            state.tolist() for state in torch.cuda.get_rng_state_all()
        ]
    return result


def restore_rng_state(
    state: Dict[str, Any],
    random_state: Optional[np.random.RandomState] = None,
) -> None:
    python_state = state["python"]
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"]),
            python_state.get("gauss"),
        )
    )
    np.random.set_state(_decode_numpy_state(state["numpy"]))
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if random_state is not None and state.get("optimizer_numpy") is not None:
        random_state.set_state(_decode_numpy_state(state["optimizer_numpy"]))
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [torch.tensor(item, dtype=torch.uint8) for item in state["torch_cuda"]]
        )


def atomic_write_json(path: str, payload: Dict[str, Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(payload), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, output_path)
    return str(output_path)


def load_ga_state(run_dir: str) -> Dict[str, Any]:
    state_path = Path(run_dir) / GA_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"GA checkpoint not found: {state_path}")
    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    if int(state.get("schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported GA checkpoint schema: {state.get('schema_version')!r}"
        )
    return state
