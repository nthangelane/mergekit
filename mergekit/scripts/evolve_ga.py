# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import click
import numpy as np
import pandas
import ray
import torch
import tqdm
import yaml

# Default to disabling tokenizer parallelism to avoid fork-safety warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference, call_with_dtype
from mergekit.config import MergeConfiguration
from mergekit.evo.cache_utils import genotype_exact_hash
from mergekit.evo.config import (
    EvolMergeConfiguration,
    ModelGenomeDefinition,
    TaskConfiguration,
    check_for_naughty_config,
)
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer, EnhancedGAParams
from mergekit.evo.ga import GAOptimizer, GAParams
from mergekit.evo.genome import ModelGenome
from mergekit.evo.helpers import (
    _eval_model,
    merge_model_with_details,
    validate_input_model_architecture,
)
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)
from mergekit.evo.ray_observability import RayRunObserver, default_run_label
from mergekit.evo.strategy import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
)
from mergekit.evo.task_utils import create_task_manager
from mergekit.evo.tracking import create_tracker
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

LOGGER = logging.getLogger("mergekit.evolve_ga.cli")
FAILED_BLACKLIST_FILENAME = "failed_genotype_blacklist.csv"


def stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    """Emit a structured log message for high-level run stages."""
    LOGGER.log(level, "[%s] %s", stage, message)


def _best_weighted_score_from_frame(frame: "pandas.DataFrame") -> Optional[float]:
    """Return the best normalized score from a baseline/comparison table."""
    if "weighted_score" not in frame.columns:
        return None

    numeric_scores = pandas.to_numeric(
        frame["weighted_score"], errors="coerce"
    ).dropna()
    if numeric_scores.empty:
        return None
    return float(numeric_scores.max())


def _reusable_baseline_csv_path(
    baseline_csv_path: str,
    expected_models: List[str],
) -> Optional[str]:
    if not os.path.exists(baseline_csv_path):
        return None

    try:
        frame = pandas.read_csv(baseline_csv_path)
    except Exception as exc:
        stage_log(
            "Stage-Baseline",
            f"Existing baseline_results.csv could not be read; rerunning baselines: {exc}",
            level=logging.WARNING,
        )
        return None

    if "model" not in frame.columns or "weighted_score" not in frame.columns:
        return None

    reusable_models = set(
        frame.loc[
            pandas.to_numeric(frame["weighted_score"], errors="coerce").notna(), "model"
        ]
        .dropna()
        .astype(str)
    )
    if not set(expected_models).issubset(reusable_models):
        return None

    return baseline_csv_path


def _unique_model_refs(model_refs: List[ModelReference]) -> List[ModelReference]:
    unique: List[ModelReference] = []
    seen: set[str] = set()
    for model_ref in model_refs:
        model_name = str(model_ref)
        if model_name in seen:
            continue
        seen.add(model_name)
        unique.append(model_ref)
    return unique


def _sanitize_metric_key_fragment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return cleaned or "unknown"


def _configured_merge_methods(config: EvolMergeConfiguration) -> List[str]:
    genome_cfg = config.genome
    if hasattr(genome_cfg, "allowed_methods"):
        return [str(method) for method in getattr(genome_cfg, "allowed_methods")]
    merge_method = getattr(genome_cfg, "merge_method", None)
    return [str(merge_method)] if merge_method else []


def _configured_task_names(config: EvolMergeConfiguration) -> List[str]:
    task_names: List[str] = []
    for task in config.tasks:
        if task.name not in task_names:
            task_names.append(task.name)
    for task in getattr(config, "stage1_tasks", None) or []:
        if task.name not in task_names:
            task_names.append(task.name)
    return task_names


def _collect_merge_method_outcomes(
    genotype_iterable: List[np.ndarray],
    results: List[dict],
    genome: Union[ModelGenome, MultiMethodGenome],
    configured_methods: List[str],
) -> Dict[str, Any]:
    method_counter: Counter[str] = Counter()
    method_success_counter: Counter[str] = Counter()
    method_failure_counter: Counter[str] = Counter()
    method_score_values: Dict[str, List[float]] = defaultdict(list)
    all_methods = set(str(method) for method in configured_methods)

    for genotype_candidate, result in zip(genotype_iterable, results):
        try:
            if hasattr(genome, "method_label_for_genotype"):
                method_name = str(genome.method_label_for_genotype(genotype_candidate))
            else:
                cfg = (
                    genome.genotype_to_merge_config(genotype_candidate)
                    if hasattr(genome, "genotype_to_merge_config")
                    else genome.genotype_merge_config(genotype_candidate)
                )
                method_name = str(getattr(cfg, "merge_method", None) or "unknown")
        except Exception:  # pragma: no cover - diagnostic path only
            logging.debug(
                "Unable to decode genotype for merge-method outcome stats",
                exc_info=True,
            )
            method_name = "decode_error"

        all_methods.add(method_name)
        method_counter[method_name] += 1
        score = result.get("score")
        if score is None:
            method_failure_counter[method_name] += 1
            continue

        score_value = float(score)
        method_success_counter[method_name] += 1
        method_score_values[method_name].append(score_value)

    metrics: Dict[str, float] = {}
    history_rows: List[Dict[str, Union[str, float, int]]] = []
    for method_name in sorted(all_methods):
        total = int(method_counter.get(method_name, 0))
        successes = int(method_success_counter.get(method_name, 0))
        failures = int(method_failure_counter.get(method_name, 0))
        success_rate = float(successes / total) if total else 0.0
        metric_key = _sanitize_metric_key_fragment(method_name)
        metrics[f"merge_method/{metric_key}/count"] = float(total)
        metrics[f"merge_method/{metric_key}/success_count"] = float(successes)
        metrics[f"merge_method/{metric_key}/failure_count"] = float(failures)
        metrics[f"merge_method/{metric_key}/success_rate"] = success_rate

        score_values = method_score_values.get(method_name, [])
        mean_score = None
        best_score = None
        if score_values:
            mean_score = float(sum(score_values) / len(score_values))
            best_score = float(max(score_values))
            metrics[f"merge_method/{metric_key}/mean_score"] = mean_score
            metrics[f"merge_method/{metric_key}/best_score"] = best_score

        history_rows.append(
            {
                "merge_method": method_name,
                "count": total,
                "success_count": successes,
                "failure_count": failures,
                "success_rate": success_rate,
                "mean_score": mean_score,
                "best_score": best_score,
            }
        )

    return {
        "method_counts": method_counter,
        "method_success_counts": method_success_counter,
        "method_failure_counts": method_failure_counter,
        "metrics": metrics,
        "history_rows": history_rows,
    }


def _write_merge_method_history(
    storage_path: str,
    generation: int,
    fevals: int,
    rows: List[Dict[str, Union[str, float, int]]],
) -> None:
    history_path = os.path.join(storage_path, "ga_method_history.csv")
    file_exists = os.path.exists(history_path)
    base_fields = ["generation", "fevals"]
    dynamic_fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in dynamic_fields:
                dynamic_fields.append(key)
    with open(history_path, "a", encoding="utf-8", newline="") as history_file:
        fieldnames = base_fields + dynamic_fields
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generation": generation,
                    "fevals": fevals,
                    **row,
                }
            )


def _write_candidate_history(
    storage_path: str,
    generation: int,
    fevals: int,
    rows: List[Dict[str, Any]],
) -> None:
    history_path = os.path.join(storage_path, "ga_candidate_history.csv")
    file_exists = os.path.exists(history_path)
    base_fields = ["generation", "fevals"]
    dynamic_fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in dynamic_fields:
                dynamic_fields.append(key)
    with open(history_path, "a", encoding="utf-8", newline="") as history_file:
        fieldnames = base_fields + dynamic_fields
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generation": generation,
                    "fevals": fevals,
                    **row,
                }
            )


def _write_mlflow_run_info(
    storage_path: str,
    tracker,
    project_name: Optional[str] = None,
    mlflow_ui_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = getattr(tracker, "get_run_metadata", lambda: {})()
    if not metadata or metadata.get("tracker_type") != "mlflow":
        return None

    info_path = os.path.join(storage_path, "mlflow_run_info.md")
    lines = [
        "# MLflow Run Info",
        "",
    ]
    if project_name:
        lines.append(f"- Experiment Name: {project_name}")
    experiment_id = metadata.get("experiment_id")
    run_id = metadata.get("run_id")
    tracking_uri = metadata.get("tracking_uri")
    local_store_path = metadata.get("local_store_path")
    run_url = metadata.get("run_url")
    ui_base_url = metadata.get("ui_base_url")

    if experiment_id:
        lines.append(f"- Experiment ID: `{experiment_id}`")
    if run_id:
        lines.append(f"- Run ID: `{run_id}`")
    if tracking_uri:
        lines.append(f"- Tracking URI: `{tracking_uri}`")
    if local_store_path:
        lines.append(f"- Local Store Path: `{local_store_path}`")
    if ui_base_url:
        lines.append(f"- MLflow UI Base URL: `{ui_base_url}`")
    if run_url:
        lines.append(f"- MLflow Review URL: {run_url}")
    if mlflow_ui_info:
        if mlflow_ui_info.get("ui_url"):
            lines.append(f"- MLflow UI URL: {mlflow_ui_info['ui_url']}")
        if mlflow_ui_info.get("pid"):
            lines.append(f"- MLflow UI PID: `{mlflow_ui_info['pid']}`")
        if mlflow_ui_info.get("log_path"):
            lines.append(f"- MLflow UI Log: `{mlflow_ui_info['log_path']}`")
    if local_store_path:
        ui_port = "5000"
        if ui_base_url:
            parsed = urlparse(str(ui_base_url))
            if parsed.port is not None:
                ui_port = str(parsed.port)
        lines.extend(
            [
                "",
                "## Local UI",
                "",
                "Start the UI with:",
                "",
                "```bash",
                f"mlflow ui --backend-store-uri '{local_store_path}' --port {ui_port}",
                "```",
            ]
        )

    with open(info_path, "w", encoding="utf-8") as info_file:
        info_file.write("\n".join(lines) + "\n")

    return info_path


def _is_local_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def _mlflow_ui_responds(ui_url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(ui_url, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except Exception:
        return False


def _start_mlflow_ui_if_needed(
    storage_path: str,
    tracker,
    *,
    enabled: bool,
    host: str = "127.0.0.1",
    port: int = 5001,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None

    metadata = getattr(tracker, "get_run_metadata", lambda: {})()
    if not metadata or metadata.get("tracker_type") != "mlflow":
        return None

    local_store_path = metadata.get("local_store_path")
    if not local_store_path:
        return None

    ui_url = os.getenv("MLFLOW_UI_URL")
    if ui_url:
        ui_url = ui_url.rstrip("/")
        if _mlflow_ui_responds(ui_url):
            return {"ui_url": ui_url, "started": False}
        os.environ.pop("MLFLOW_UI_URL", None)

    ui_url = f"http://{host}:{int(port)}"
    if _is_local_port_open(host, port):
        if _mlflow_ui_responds(ui_url):
            os.environ["MLFLOW_UI_URL"] = ui_url
            return {"ui_url": ui_url, "started": False}
        stage_log(
            "Stage-Tracking",
            (
                f"MLflow UI port {port} is open but not responding at {ui_url}; "
                "manual restart may be required."
            ),
            level=logging.WARNING,
        )
        return None

    log_path = os.path.join(storage_path, "mlflow_ui.log")
    pid_path = os.path.join(storage_path, "mlflow_ui.pid")
    cmd = [
        os.environ.get("PYTHON", os.sys.executable),
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        str(local_store_path),
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    env = os.environ.copy()
    env.setdefault("GUNICORN_CMD_ARGS", "--workers=1 --timeout 120")
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        with open(pid_path, "w", encoding="utf-8") as pid_file:
            pid_file.write(f"{process.pid}\n")
        os.environ["MLFLOW_UI_URL"] = ui_url
        return {
            "ui_url": ui_url,
            "pid": process.pid,
            "log_path": log_path,
            "pid_path": pid_path,
            "started": True,
        }
    except Exception as exc:  # pragma: no cover - runtime-only safety
        stage_log(
            "Stage-Tracking",
            f"Failed to start MLflow UI automatically: {exc}",
            level=logging.WARNING,
        )
        return None


def _log_run_artifacts(tracker, storage_path: str) -> None:
    artifacts = {
        "ga_history": "ga_history.csv",
        "ga_method_history": "ga_method_history.csv",
        "ga_summary": "ga_summary.txt",
        "ga_stop_details": "ga_stop_details.json",
        "ga_history_plot": "ga_history_plot.png",
        "baseline_results": "baseline_results.csv",
        "failed_genotypes": "failed_genotypes.csv",
        "failed_genotype_blacklist": FAILED_BLACKLIST_FILENAME,
        "final_comparison": "ga_final_comparison.csv",
        "final_comparison_plot": "ga_final_comparison.png",
        "mlflow_run_info": "mlflow_run_info.md",
        "mlflow_ui_log": "mlflow_ui.log",
        "ray_observability": "ray_observability.json",
    }
    for artifact_name, file_name in artifacts.items():
        file_path = os.path.join(storage_path, file_name)
        if os.path.exists(file_path):
            tracker.log_artifact(file_path, artifact_name)


def _score_improvement(
    current_score: float,
    baseline_score: float,
) -> Tuple[float, Optional[float]]:
    """Return absolute and percentage improvement over a baseline score.

    Scores are already normalized so that larger is always better. Percentage
    improvement is therefore measured against the baseline magnitude, not the
    raw baseline sign, which keeps loss-derived negative scores intuitive.
    """
    delta = current_score - baseline_score
    baseline_magnitude = abs(float(baseline_score))
    if baseline_magnitude == 0.0:
        return delta, None
    return delta, (delta / baseline_magnitude) * 100.0


def _meets_improvement_thresholds(
    delta: float,
    pct: Optional[float],
    min_abs: float,
    min_pct: float,
) -> bool:
    if delta < min_abs:
        return False
    if min_pct <= 0.0:
        return True
    if pct is None:
        return False
    return pct >= min_pct


def _resolve_stop_configuration(
    config: EvolMergeConfiguration,
    *,
    max_fevals_cli: Optional[int],
    timeout_cli: Optional[float],
) -> Dict[str, Any]:
    stop_cfg = getattr(config, "stop", None)
    resolved_max_fevals = (
        int(max_fevals_cli)
        if max_fevals_cli is not None
        else (
            int(stop_cfg.max_fevals)
            if stop_cfg is not None and stop_cfg.max_fevals is not None
            else 100
        )
    )
    resolved_timeout = (
        float(timeout_cli)
        if timeout_cli is not None
        else (
            float(stop_cfg.max_time_seconds)
            if stop_cfg is not None and stop_cfg.max_time_seconds is not None
            else None
        )
    )
    return {
        "max_fevals": resolved_max_fevals,
        "timeout_seconds": resolved_timeout,
        "target_improvement_abs": (
            float(stop_cfg.target_improvement_abs)
            if stop_cfg is not None and stop_cfg.target_improvement_abs is not None
            else None
        ),
        "target_improvement_pct": (
            float(stop_cfg.target_improvement_pct)
            if stop_cfg is not None and stop_cfg.target_improvement_pct is not None
            else None
        ),
        "target_reference": (
            str(stop_cfg.target_reference) if stop_cfg is not None else "best_baseline"
        ),
        "min_generations_before_target_stop": (
            int(stop_cfg.min_generations_before_target_stop)
            if stop_cfg is not None
            else 0
        ),
        "require_stage2_for_target": bool(
            stop_cfg.require_stage2_for_target if stop_cfg is not None else False
        ),
        "stagnation_patience_generations": (
            int(stop_cfg.stagnation_patience_generations)
            if stop_cfg is not None
            and stop_cfg.stagnation_patience_generations is not None
            else 0
        ),
        "stagnation_min_delta": (
            float(stop_cfg.stagnation_min_delta) if stop_cfg is not None else 0.0
        ),
    }


def _write_stop_details(
    storage_path: str,
    *,
    resolved_stop: Dict[str, Any],
    stop_details: Optional[Dict[str, Any]],
) -> str:
    output_path = os.path.join(storage_path, "ga_stop_details.json")
    payload = {
        "resolved_stop": resolved_stop,
        "final_stop": stop_details or {},
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def _failed_blacklist_scope(config: EvolMergeConfiguration) -> str:
    payload = json.dumps(
        config.genome.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _load_failed_genotype_blacklist(
    storage_path: str,
    genome_scope: str,
) -> Dict[str, dict]:
    blacklist_path = os.path.join(storage_path, FAILED_BLACKLIST_FILENAME)
    if not os.path.exists(blacklist_path):
        return {}

    blacklist: Dict[str, dict] = {}
    with open(blacklist_path, "r", encoding="utf-8", newline="") as blacklist_file:
        reader = csv.DictReader(blacklist_file)
        for row in reader:
            row_scope = str(row.get("genome_scope") or "").strip()
            if row_scope and row_scope != genome_scope:
                continue

            genotype_hash = str(row.get("genotype_hash") or "").strip()
            if not genotype_hash:
                continue

            blacklist[genotype_hash] = {
                "score": None,
                "results": None,
                "error_stage": str(row.get("error_stage") or "merge"),
                "error_type": str(row.get("error_type") or "unknown"),
                "error_message": str(
                    row.get("error_message")
                    or "Skipped due to persisted failed-genotype blacklist"
                ),
            }

    return blacklist


def _resolve_merge_cuda(merge_cuda: bool, num_gpus: Optional[int]) -> bool:
    if num_gpus == 0 and merge_cuda:
        stage_log(
            "Stage-Init",
            (
                "--num-gpus 0 requested; disabling CUDA merges automatically. "
                "Use --merge-cuda only when GPU workers are allocated."
            ),
            level=logging.WARNING,
        )
        return False
    return merge_cuda


def prune_stale_merged_artifacts(
    storage_path: str, *, keep: Optional[List[Path]] = None
) -> None:
    """Purge transient merged model directories to keep disk usage in check."""

    merged_dir = Path(storage_path) / "merged"
    if not merged_dir.exists():
        return

    keep_resolved = {p.resolve() for p in (keep or [])}
    for entry in merged_dir.iterdir():
        try:
            resolved = entry.resolve()
        except FileNotFoundError:  # pragma: no cover - concurrent cleanup window
            continue

        if resolved in keep_resolved:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _init_ray_for_baselines() -> None:
    """Initialize Ray for baseline evaluation.

    Prefer attaching to an existing cluster when `RAY_ADDRESS` is provided or when
    a local auto-discovered instance exists. Fall back to a local runtime for
    standalone local experiments.
    """

    if ray.is_initialized():
        return

    ray_address = os.environ.get("RAY_ADDRESS")
    try:
        ray.init(
            address=ray_address or "auto",
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )
    except ConnectionError:
        if ray_address:
            raise

        stage_log(
            "Stage-Baseline",
            "No running Ray instance found; starting a local Ray runtime for baselines.",
            level=logging.WARNING,
        )
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )


@click.command("mergekit-evolve-ga")
@click.argument("genome-config-path", type=str)
@click.option(
    "--max-fevals",
    type=int,
    default=None,
    help="Maximum function evaluations (overrides YAML stop.max_fevals if set)",
)
@click.option(
    "--population-size",
    type=int,
    default=None,
    help="Population size (overrides YAML if set)",
)
@click.option(
    "--elite-fraction",
    type=float,
    default=None,
    help="Elitism fraction [0,1] (overrides YAML if set)",
)
@click.option(
    "--mutation-rate",
    type=float,
    default=None,
    help="Per-gene mutation probability (overrides YAML if set)",
)
@click.option(
    "--mutation-sigma",
    type=float,
    default=None,
    help="Stddev for Gaussian mutation noise (overrides YAML if set)",
)
@click.option(
    "--crossover",
    type=str,
    default=None,
    help="Crossover operator: arithmetic | uniform | sbx (overrides YAML if set)",
)
@click.option(
    "--tournament-size",
    type=int,
    default=None,
    help="Tournament size for selection (overrides YAML if set)",
)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    help="Evaluation scheduling strategy",
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    default=False,
    help="Use in-memory merge & evaluation",
)
@click.option(
    "--storage-path",
    type=str,
    help="Path to storage accessible to all nodes for model storage",
    required=True,
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option(
    "--tensor-parallel-size",
    type=click.IntRange(1, 10),
    default=1,
    show_default=True,
    help="GPUs per vLLM evaluation replica (requires multi-GPU Ray workers)",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of CPU workers when GPUs=0 (pool/buffered)",
)
@click.option("--merge-cuda/--no-merge-cuda", is_flag=True, default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option("--allow-crimes/--no-allow-crimes", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=0)
@click.option("--batch-size", type=int, default=None, help="Batch size for evaluation")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Evaluation sample limit (overrides YAML if set)",
)
@click.option("use_wandb", "--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--wandb-project", type=str, help="Wandb project name")
@click.option("--wandb-entity", type=str, help="Wandb entity name")
@click.option("use_mlflow", "--mlflow/--no-mlflow", is_flag=True, default=False)
@click.option("--mlflow-experiment", type=str, help="MLflow experiment name")
@click.option(
    "--mlflow-tracking-uri", type=str, help="MLflow tracking URI (default: ./mlruns)"
)
@click.option(
    "--mlflow-ui/--no-mlflow-ui",
    "auto_mlflow_ui",
    is_flag=True,
    default=True,
    help="Start or reuse a local MLflow UI while the run is active",
)
@click.option(
    "--mlflow-ui-port",
    type=int,
    default=5001,
    show_default=True,
    help="Local port for the auto-started MLflow UI",
)
@click.option(
    "--task-search-path",
    type=str,
    multiple=True,
    help="Path to search for lmeval tasks",
)
@click.option(
    "--i-understand-the-depths-of-the-evils-i-am-unleashing",
    "allow_benchmark_tasks",
    is_flag=True,
    default=False,
    help="Allow benchmark tasks as objectives",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    is_flag=True,
    default=True,
    help="Save the final merged model",
)
@click.option(
    "--reshard/--no-reshard",
    is_flag=True,
    default=True,
    help="Convert models to single-shard safetensors for faster merge",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Maximum time to run the optimization in seconds",
)
@click.option(
    "--baseline/--no-baseline",
    "run_baseline",
    is_flag=True,
    default=True,
    help="Run baseline evaluations for source models before GA search",
)
@click.option(
    "--hf-model-id",
    type=str,
    default=None,
    help="Hugging Face model ID to push final model to (e.g. username/model-name)",
)
@click.option(
    "--hf-username",
    type=str,
    default=None,
    help="Hugging Face username to auto-generate a repo name if --hf-model-id is not set",
)
@click.option(
    "--hf-min-improvement",
    type=float,
    default=0.0,
    help="Minimum absolute improvement over best baseline required to upload",
)
@click.option(
    "--hf-min-improvement-pct",
    type=float,
    default=0.0,
    help="Minimum percentage improvement over best baseline required to upload",
)
def main(
    genome_config_path: str,
    max_fevals: Optional[int],
    population_size: Optional[int],
    elite_fraction: Optional[float],
    mutation_rate: Optional[float],
    mutation_sigma: Optional[float],
    crossover: str,
    tournament_size: Optional[int],
    vllm: bool,
    strategy: str,
    in_memory: bool,
    storage_path: Optional[str],
    num_gpus: Optional[int],
    tensor_parallel_size: int,
    num_workers: Optional[int],
    merge_cuda: bool,
    trust_remote_code: bool,
    allow_crimes: bool,
    random_seed: int,
    batch_size: Optional[int],
    limit: Optional[int],
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    use_mlflow: bool,
    mlflow_experiment: Optional[str],
    mlflow_tracking_uri: Optional[str],
    auto_mlflow_ui: bool,
    mlflow_ui_port: int,
    task_search_path: List[str],
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    reshard: bool,
    timeout: Optional[float],
    run_baseline: bool,
    hf_model_id: Optional[str],
    hf_username: Optional[str],
    hf_min_improvement: float,
    hf_min_improvement_pct: float,
):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)

    stage_log("Stage-Init", f"Seeding RNG with value {random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    stage_log("Stage-Init", f"Loading genome configuration from {genome_config_path}")
    with open(genome_config_path, "r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    config = EvolMergeConfiguration.model_validate(raw_config)
    if limit is not None:
        config = config.model_copy(update={"limit": limit})
        stage_log("Stage-Init", f"Overriding evaluation limit from CLI: {limit}")

    stage_log("Stage-Init", "Validating configuration settings...")
    check_for_naughty_config(config, allow=allow_benchmark_tasks)
    resolved_stop = _resolve_stop_configuration(
        config,
        max_fevals_cli=max_fevals,
        timeout_cli=timeout,
    )
    max_fevals = int(resolved_stop["max_fevals"])
    timeout = resolved_stop["timeout_seconds"]
    stop_summary_parts = [f"max_fevals={max_fevals}"]
    if timeout is not None:
        stop_summary_parts.append(f"max_time_seconds={timeout:.0f}")
    if resolved_stop.get("target_improvement_abs") is not None:
        stop_summary_parts.append(
            f"target_improvement_abs={resolved_stop['target_improvement_abs']:.6f}"
        )
    if resolved_stop.get("target_improvement_pct") is not None:
        stop_summary_parts.append(
            f"target_improvement_pct={resolved_stop['target_improvement_pct']:.2f}%"
        )
    if resolved_stop.get("stagnation_patience_generations"):
        stop_summary_parts.append(
            "stagnation="
            f"{resolved_stop['stagnation_patience_generations']} gens"
            f" @ delta<{resolved_stop['stagnation_min_delta']:.6f}"
        )
    stage_log("Stage-Init", "Resolved stop policy: " + ", ".join(stop_summary_parts))

    storage_path = os.path.abspath(storage_path)
    os.makedirs(storage_path, exist_ok=True)
    stage_log("Stage-Init", f"Storage path: {storage_path}")
    run_label = default_run_label(storage_path, strategy)
    ray_observer = RayRunObserver(
        run_label=run_label,
        strategy=strategy,
        role="driver",
        storage_path=storage_path,
        snapshot_enabled=True,
    )
    ray_observer.set_phase(
        "init",
        generation=0,
        fevals_completed=0,
    )
    merge_cuda = _resolve_merge_cuda(merge_cuda, num_gpus)
    if tensor_parallel_size > 1:
        if not vllm:
            raise click.ClickException(
                "--tensor-parallel-size > 1 requires --vllm because the HF backend is single-GPU here."
            )
        if in_memory:
            raise click.ClickException(
                "--tensor-parallel-size > 1 is not supported with --in-memory."
            )
        if num_gpus is not None and num_gpus <= 0:
            raise click.ClickException(
                "--tensor-parallel-size > 1 requires --num-gpus to be greater than 0."
            )
        if num_gpus is not None and tensor_parallel_size > num_gpus:
            raise click.ClickException(
                "--tensor-parallel-size cannot exceed --num-gpus for a run."
            )
        stage_log(
            "Stage-Init",
            (
                "Using vLLM tensor parallel evaluation with "
                f"{tensor_parallel_size} GPU(s) per candidate."
            ),
        )
    failed_blacklist_scope = _failed_blacklist_scope(config)
    persisted_failed_genotypes = _load_failed_genotype_blacklist(
        storage_path, failed_blacklist_scope
    )
    if persisted_failed_genotypes:
        stage_log(
            "Stage-Init",
            (
                "Loaded "
                f"{len(persisted_failed_genotypes)} failed genotype hashes from "
                f"{os.path.join(storage_path, FAILED_BLACKLIST_FILENAME)}; "
                "exact matches will be skipped."
            ),
        )

    if not hf_model_id and hf_username:
        base_ref = config.genome.base_model or config.genome.models[0]
        base_short = (
            re.sub(r"[^a-zA-Z0-9]+", "-", str(base_ref).split("/")[-1])
            .strip("-")
            .lower()
        )
        date_stamp = datetime.now().strftime("%d%b").lower()
        hf_model_id = f"{hf_username}/gaevolve-{date_stamp}-{base_short}"
        stage_log("Stage-Init", f"Auto-generated Hugging Face repo: {hf_model_id}")

    task_search_path = list(task_search_path)

    # Initialize experiment tracking before baselines so MLflow/W&B covers the full run.
    ray_observer.set_phase("tracking", generation=0, fevals_completed=0)
    stage_log("Stage-Tracking", "Initializing experiment tracker...")
    tracker = None
    if use_wandb and use_mlflow:
        raise ValueError(
            "Cannot use both wandb and mlflow at the same time. Choose one."
        )
    elif use_wandb:
        stage_log("Stage-Tracking", "Using Weights & Biases for experiment tracking.")
        tracker = create_tracker("wandb")
        tracker.initialize(
            project_name=wandb_project or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            entity=wandb_entity,
        )
    elif use_mlflow:
        stage_log("Stage-Tracking", "Using MLflow for experiment tracking.")
        tracker = create_tracker("mlflow")
        resolved_mlflow_tracking_uri = (
            mlflow_tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or f"file://{os.path.join(storage_path, 'mlruns')}"
        )
        tracker.initialize(
            project_name=mlflow_experiment or "mergekit-evolve-ga",
            config=config.model_dump(mode="json"),
            tracking_uri=resolved_mlflow_tracking_uri,
        )
    else:
        stage_log(
            "Stage-Tracking", "Experiment tracking disabled (logging to console only)."
        )
        tracker = create_tracker("none")
        tracker.initialize(project_name="no-tracking", config={})

    mlflow_ui_info = _start_mlflow_ui_if_needed(
        storage_path,
        tracker,
        enabled=bool(use_mlflow and auto_mlflow_ui),
        port=int(mlflow_ui_port),
    )
    ray_observer.set_mlflow_ui(bool(mlflow_ui_info and mlflow_ui_info.get("ui_url")))
    mlflow_info_path = _write_mlflow_run_info(
        storage_path,
        tracker,
        project_name=mlflow_experiment or "mergekit-evolve-ga",
        mlflow_ui_info=mlflow_ui_info,
    )
    if mlflow_info_path:
        stage_log("Stage-Tracking", f"MLflow run info written to {mlflow_info_path}")
        metadata = getattr(tracker, "get_run_metadata", lambda: {})()
        if metadata.get("run_url"):
            stage_log("Stage-Tracking", f"MLflow review URL: {metadata['run_url']}")
        if mlflow_ui_info and mlflow_ui_info.get("ui_url"):
            stage_log(
                "Stage-Tracking",
                f"MLflow UI available at {mlflow_ui_info['ui_url']}",
            )

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    stage_log("Stage-Init", "Checking source-model base architecture compatibility...")
    source_models: List[ModelReference] = list(config.genome.models)
    if getattr(config.genome, "base_model", None) is not None:
        source_models.append(config.genome.base_model)
    try:
        validate_input_model_architecture(source_models, merge_options)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    stage_log("Stage-Init", "Source-model base architecture check complete.")

    baseline_csv_path = None
    baseline_best_score: Optional[float] = None
    if run_baseline:
        ray_observer.set_phase("baseline", baseline_model_index=0)
        baseline_csv_path = run_baseline_evaluations(
            config,
            storage_path,
            batch_size,
            merge_cuda,
            num_gpus,
            task_search_path,
            trust_remote_code,
            ray_observer=ray_observer,
        )
        if baseline_csv_path:
            try:
                baseline_df = pandas.read_csv(baseline_csv_path)
                baseline_best_score = _best_weighted_score_from_frame(baseline_df)
            except Exception as exc:  # pragma: no cover - defensive logging only
                stage_log(
                    "Stage-Baseline",
                    f"Failed to parse baseline_results.csv for summary metrics: {exc}",
                    level=logging.WARNING,
                )
    else:
        stage_log("Stage-Baseline", "Skipping baseline evaluation (--no-baseline).")
        ray_observer.set_phase(
            "baseline", baseline_model_index=0, baseline_model_total=0
        )
    if (
        (
            resolved_stop.get("target_improvement_abs") is not None
            or resolved_stop.get("target_improvement_pct") is not None
        )
        and resolved_stop.get("target_reference") == "best_baseline"
        and (baseline_best_score is None or not math.isfinite(baseline_best_score))
    ):
        stage_log(
            "Stage-Baseline",
            "Stop target is configured against best_baseline, but baseline score is unavailable; target-based stopping will remain disabled for this run.",
            level=logging.WARNING,
        )

    # convert models to single-shard safetensors
    if reshard:
        ray_observer.set_phase("reshard", generation=0, fevals_completed=0)
        stage_log(
            "Stage-Reshard",
            "Converting source models to single-shard safetensors...",
        )
        resharded_models = []
        resharded_base = None
        for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):
            resharded_models.append(
                _reshard_model(
                    model,
                    storage_path,
                    merge_options.lora_merge_cache,
                    trust_remote_code,
                )
            )
        if config.genome.base_model is not None:
            resharded_base = _reshard_model(
                config.genome.base_model,
                storage_path,
                merge_options.lora_merge_cache,
                trust_remote_code,
            )
        stage_log("Stage-Reshard", "Resharding complete.")
    else:
        stage_log("Stage-Reshard", "Skipping reshard step (--no-reshard).")
        resharded_models = config.genome.models
        resharded_base = config.genome.base_model

    # Create genome based on type - check if it's MultiMethodGenomeDefinition
    from mergekit.evo.multi_method_genome import MultiMethodGenomeDefinition

    genome_config = config.genome

    if isinstance(genome_config, MultiMethodGenomeDefinition):
        genome_type = "multi_method"
        # Create multi-method genome
        genome = MultiMethodGenome(
            MultiMethodGenomeDefinition.model_validate(
                {
                    **genome_config.model_dump(exclude=["models", "base_model"]),
                    "models": resharded_models,
                    "base_model": resharded_base,
                }
            ),
            trust_remote_code=trust_remote_code,
        )
    else:
        genome_type = "standard"
        # Create traditional genome
        genome = ModelGenome(
            ModelGenomeDefinition.model_validate(
                {
                    **genome_config.model_dump(exclude=["models", "base_model"]),
                    "models": resharded_models,
                    "base_model": resharded_base,
                }
            ),
            trust_remote_code=trust_remote_code,
        )

    if strategy == "pool":
        strat_cls = ActorPoolEvaluationStrategy
    elif strategy == "buffered":
        strat_cls = BufferedRayEvaluationStrategy
    elif strategy == "serial":
        strat_cls = SerialEvaluationStrategy
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Validate crossover if provided
    if crossover is not None and crossover not in {"arithmetic", "uniform", "sbx"}:
        raise click.BadParameter("--crossover must be one of: arithmetic, uniform, sbx")

    stage_log("Stage-GA", f"Initializing evaluation strategy '{strategy}'...")
    strat = strat_cls(
        config,
        genome,
        merge_options,
        num_gpus=num_gpus,
        tensor_parallel_size=tensor_parallel_size,
        num_workers=num_workers,
        vllm=vllm,
        in_memory=in_memory,
        model_storage_path=os.path.join(storage_path, "merged"),
        batch_size=batch_size,
        task_search_path=task_search_path,
        run_label=run_label,
        run_observer=ray_observer,
    )

    def log_population(res_list: List[dict], step: int):
        tracker.log_population_stats(res_list, step)

    def log_best(x: np.ndarray, score: float, step: int):
        tracker.log_best_individual(x, score, step, genome)

    def save_best_config(x: np.ndarray):
        best_yaml: str
        if genome_type == "multi_method" and hasattr(genome, "execution_plan_dict"):
            plan_dict = genome.execution_plan_dict(x)
            if plan_dict.get("kind") == "config":
                merge_config = MergeConfiguration.model_validate(plan_dict["config"])
                best_yaml = merge_config.to_yaml()
            else:
                best_yaml = yaml.safe_dump(
                    {"layered_execution_plan": plan_dict},
                    sort_keys=False,
                ).rstrip()
        else:
            merge_config = (
                genome.genotype_to_merge_config(x)
                if genome_type == "multi_method"
                else genome.genotype_merge_config(x)
            )
            best_yaml = merge_config.to_yaml()
        config_path = os.path.join(storage_path, "best_config.yaml")
        with open(config_path, "w") as f:
            f.write(best_yaml)
        print(f"Merge configuration:\n{best_yaml}")
        tracker.log_artifact(config_path, "best_config")

    # Build GA optimizer with callbacks
    # Resolve GA parameters: CLI overrides YAML; fallback to GAParams defaults
    defaults = GAParams()
    yaml_ga = getattr(config, "ga", None)
    ga_params = GAParams(
        population_size=(
            population_size
            if population_size is not None
            else (yaml_ga.population_size if yaml_ga else defaults.population_size)
        ),
        elite_fraction=(
            elite_fraction
            if elite_fraction is not None
            else (yaml_ga.elite_fraction if yaml_ga else defaults.elite_fraction)
        ),
        mutation_rate=(
            mutation_rate
            if mutation_rate is not None
            else (yaml_ga.mutation_rate if yaml_ga else defaults.mutation_rate)
        ),
        mutation_sigma=(
            mutation_sigma
            if mutation_sigma is not None
            else (yaml_ga.mutation_sigma if yaml_ga else defaults.mutation_sigma)
        ),
        crossover=(
            crossover
            if crossover is not None
            else (yaml_ga.crossover if yaml_ga else defaults.crossover)
        ),
        tournament_size=(
            tournament_size
            if tournament_size is not None
            else (yaml_ga.tournament_size if yaml_ga else defaults.tournament_size)
        ),
        target_improvement_abs=resolved_stop.get("target_improvement_abs"),
        target_improvement_pct=resolved_stop.get("target_improvement_pct"),
        target_reference=str(resolved_stop.get("target_reference") or "best_baseline"),
        target_reference_score=(
            float(baseline_best_score)
            if baseline_best_score is not None and math.isfinite(baseline_best_score)
            else None
        ),
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

    # Log resolved GA params
    tracker.log_metrics(
        {
            "ga/population_size": ga_params.population_size,
            "ga/elite_fraction": ga_params.elite_fraction,
            "ga/mutation_rate": ga_params.mutation_rate,
            "ga/mutation_sigma": ga_params.mutation_sigma,
            "ga/tournament_size": ga_params.tournament_size,
            "ga/adaptive_method_sampling": float(
                1.0 if (yaml_ga and yaml_ga.adaptive_method_sampling) else 0.0
            ),
            "eval/two_stage": float(1.0 if config.two_stage else 0.0),
            "eval/stage1_limit": (
                float(config.stage1_limit) if config.stage1_limit is not None else None
            ),
            "eval/stage2_limit": (
                float(config.stage2_limit) if config.stage2_limit is not None else None
            ),
            "eval/stage2_top_k": (
                float(config.stage2_top_k) if config.stage2_top_k is not None else None
            ),
            "eval/fitness_mode": config.fitness_mode,
            "eval/task_mix_profile": config.task_mix_profile,
            "eval/behavior_probe_enabled": float(
                1.0 if config.behavior_prompts else 0.0
            ),
            "eval/behavior_probe_max_new_tokens": float(
                config.behavior_probe_max_new_tokens
            ),
            "eval/behavior_min_distinct_ratio": float(
                config.behavior_min_distinct_ratio
            ),
            "ga/gene_diversity_bonus_weight": (
                float(getattr(yaml_ga, "gene_diversity_bonus_weight"))
                if yaml_ga
                and getattr(yaml_ga, "gene_diversity_bonus_weight", None) is not None
                else None
            ),
            "ga/behavior_diversity_bonus_weight": (
                float(getattr(yaml_ga, "behavior_diversity_bonus_weight"))
                if yaml_ga
                and getattr(yaml_ga, "behavior_diversity_bonus_weight", None)
                is not None
                else None
            ),
            "ga/archive_novelty_bonus_weight": (
                float(getattr(yaml_ga, "archive_novelty_bonus_weight"))
                if yaml_ga
                and getattr(yaml_ga, "archive_novelty_bonus_weight", None) is not None
                else None
            ),
            "ga/novelty_archive_size": (
                float(getattr(yaml_ga, "novelty_archive_size"))
                if yaml_ga
                and getattr(yaml_ga, "novelty_archive_size", None) is not None
                else None
            ),
            "stop/max_fevals": float(max_fevals),
            "stop/max_time_seconds": (float(timeout) if timeout is not None else None),
            "stop/target_improvement_abs": (
                float(resolved_stop["target_improvement_abs"])
                if resolved_stop.get("target_improvement_abs") is not None
                else None
            ),
            "stop/target_improvement_pct": (
                float(resolved_stop["target_improvement_pct"])
                if resolved_stop.get("target_improvement_pct") is not None
                else None
            ),
            "stop/min_generations_before_target_stop": float(
                resolved_stop.get("min_generations_before_target_stop", 0)
            ),
            "stop/require_stage2_for_target": float(
                1.0 if resolved_stop.get("require_stage2_for_target") else 0.0
            ),
            "stop/stagnation_patience_generations": float(
                resolved_stop.get("stagnation_patience_generations", 0)
            ),
            "stop/stagnation_min_delta": float(
                resolved_stop.get("stagnation_min_delta", 0.0)
            ),
        }
    )

    best_x = None
    best_score = -np.inf
    generation_durations: List[float] = []
    generation_best_history: List[float] = []
    last_global_best = float("-inf")
    logged_failed_hashes: set[str] = set(persisted_failed_genotypes)
    total_generations = max(
        1, math.ceil(max_fevals / max(ga_params.population_size, 1))
    )
    configured_methods = _configured_merge_methods(config)

    def _format_time(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            return "--"
        if seconds >= 3600:
            hours = seconds / 3600.0
            return f"{hours:.1f}h"
        if seconds >= 60:
            minutes = seconds / 60.0
            return f"{minutes:.1f}m"
        return f"{seconds:.0f}s"

    def on_generation_start(
        generation_idx: int,
        fevals_completed: int,
        fevals_limit: int,
        population_size: int,
        current_best: float,
    ) -> None:
        completed = len(generation_durations)
        avg_seconds = sum(generation_durations) / completed if completed > 0 else None
        remaining_generations = max(total_generations - completed, 0)
        eta_seconds = (
            avg_seconds * remaining_generations if avg_seconds is not None else None
        )

        current_best_str = (
            f"{current_best:.4f}"
            if math.isfinite(current_best) and current_best > float("-inf")
            else "--"
        )
        strat.set_runtime_context(generation=generation_idx, phase="ga")
        ray_observer.record_generation_start(
            generation=generation_idx,
            fevals_completed=fevals_completed,
            fevals_limit=fevals_limit,
            population_size=population_size,
            best_score=current_best,
        )
        print(f"[GA] === Generation {generation_idx}/{total_generations} ===")
        print(
            f"[GA] Progress: fevals={fevals_completed}/{fevals_limit} | "
            f"current best={current_best_str} | avg/gen={_format_time(avg_seconds)} | "
            f"ETA~{_format_time(eta_seconds)}"
        )

    def on_pop(res_list: List[dict], pop_arr: np.ndarray, step: int, info: dict):
        # population stats
        log_population(res_list, step)

        base_model_counter: Counter[str] = Counter()

        def _record_base_model(model_ref) -> None:
            if model_ref:
                try:
                    base_model_counter[str(model_ref)] += 1
                except Exception:  # pragma: no cover - defensive string conversion
                    base_model_counter[repr(model_ref)] += 1

        def _tally_config(genotype_candidate: np.ndarray) -> None:
            try:
                cfg = (
                    genome.genotype_to_merge_config(genotype_candidate)
                    if hasattr(genome, "genotype_to_merge_config")
                    else genome.genotype_merge_config(genotype_candidate)
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logging.debug(
                    "Unable to decode genotype for GA history counters", exc_info=exc
                )
                return

            _record_base_model(getattr(cfg, "base_model", None))

            if cfg.slices:
                for slice_def in cfg.slices:
                    _record_base_model(getattr(slice_def, "base_model", None))

            if cfg.modules:
                for module_def in cfg.modules.values():
                    if module_def.slices:
                        for slice_def in module_def.slices:
                            _record_base_model(getattr(slice_def, "base_model", None))

        if isinstance(pop_arr, np.ndarray):
            if pop_arr.ndim <= 1:
                genotype_iterable = [pop_arr]
            else:
                genotype_iterable = [pop_arr[i] for i in range(pop_arr.shape[0])]
        else:
            genotype_iterable = list(pop_arr)

        merge_method_outcomes = _collect_merge_method_outcomes(
            genotype_iterable,
            res_list,
            genome,
            configured_methods,
        )
        population_metadata = info.get("population_metadata") or []
        operator_summary = info.get("operator_summary") or {}
        if operator_summary.get("metrics"):
            merge_method_outcomes["metrics"].update(operator_summary["metrics"])
        if operator_summary.get("history_rows"):
            history_rows_by_method = {
                str(row.get("merge_method")): dict(row)
                for row in merge_method_outcomes["history_rows"]
            }
            for row in operator_summary["history_rows"]:
                method_name = str(row.get("merge_method"))
                merged_row = history_rows_by_method.get(method_name, {})
                merged_row.update(row)
                history_rows_by_method[method_name] = merged_row
            merge_method_outcomes["history_rows"] = [
                history_rows_by_method[key]
                for key in sorted(history_rows_by_method.keys())
            ]
        method_counter = merge_method_outcomes["method_counts"]
        method_success_counter = merge_method_outcomes["method_success_counts"]
        method_failure_counter = merge_method_outcomes["method_failure_counts"]
        candidate_rows: List[Dict[str, Any]] = []
        gene_diversity_values: List[float] = []
        behavior_diversity_values: List[float] = []
        archive_novelty_values: List[float] = []
        stability_values: List[float] = []

        for genotype_candidate in genotype_iterable:
            _tally_config(genotype_candidate)

        for idx, (genotype_candidate, result) in enumerate(
            zip(genotype_iterable, res_list)
        ):
            try:
                if hasattr(genome, "method_label_for_genotype"):
                    candidate_method = str(
                        genome.method_label_for_genotype(genotype_candidate)
                    )
                else:
                    cfg = (
                        genome.genotype_to_merge_config(genotype_candidate)
                        if hasattr(genome, "genotype_to_merge_config")
                        else genome.genotype_merge_config(genotype_candidate)
                    )
                    candidate_method = str(
                        getattr(cfg, "merge_method", None) or "unknown"
                    )
            except Exception:
                candidate_method = "decode_error"

            metadata = (
                population_metadata[idx] if idx < len(population_metadata) else {}
            )
            fitness_components = dict(result.get("fitness_components") or {})
            behavior_probe = dict(result.get("behavior_probe") or {})
            gene_diversity = fitness_components.get("gene_diversity_score")
            behavior_diversity = fitness_components.get(
                "behavior_diversity_score",
                behavior_probe.get("behavior_diversity_score"),
            )
            archive_novelty = fitness_components.get("archive_novelty_score")
            stability_score = fitness_components.get(
                "stability_score", behavior_probe.get("stability_score")
            )
            if gene_diversity is not None:
                gene_diversity_values.append(float(gene_diversity))
            if behavior_diversity is not None:
                behavior_diversity_values.append(float(behavior_diversity))
            if archive_novelty is not None:
                archive_novelty_values.append(float(archive_novelty))
            if stability_score is not None:
                stability_values.append(float(stability_score))

            candidate_rows.append(
                {
                    "candidate_index": idx,
                    "merge_method": candidate_method,
                    "sampled_method": metadata.get("sampled_method"),
                    "role": metadata.get("role") or metadata.get("origin"),
                    "score": result.get("score"),
                    "raw_score": result.get("raw_score"),
                    "score_source": result.get("score_source"),
                    "stage1_score": result.get("stage1_score"),
                    "stage2_score": result.get("stage2_score"),
                    "stage2_skipped": result.get("stage2_skipped"),
                    "task_score": fitness_components.get("task_score"),
                    "language_quality": fitness_components.get("language_quality"),
                    "stability_score": stability_score,
                    "gene_diversity_score": gene_diversity,
                    "behavior_diversity_score": behavior_diversity,
                    "archive_novelty_score": archive_novelty,
                    "passthrough_penalty": fitness_components.get(
                        "passthrough_penalty"
                    ),
                    "gene_diversity_bonus": fitness_components.get(
                        "gene_diversity_bonus"
                    ),
                    "behavior_diversity_bonus": fitness_components.get(
                        "behavior_diversity_bonus"
                    ),
                    "archive_novelty_bonus": fitness_components.get(
                        "archive_novelty_bonus"
                    ),
                    "fitness_proxy": result.get("fitness_proxy"),
                    "error_stage": result.get("error_stage"),
                    "error_type": result.get("error_type"),
                    "error_message": result.get("error_message"),
                    "behavior_rejected": behavior_probe.get("rejected"),
                    "behavior_reject_reason": behavior_probe.get("reject_reason"),
                    "parent_scores": ";".join(
                        str(score) for score in metadata.get("parent_scores", [])
                    ),
                }
            )

        def _format_counter(counter: Counter[str]) -> str:
            if not counter:
                return ""
            items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            return ";".join(f"{key}:{value}" for key, value in items)

        base_model_counts_str = _format_counter(base_model_counter)
        merge_method_counts_str = _format_counter(method_counter)
        merge_method_success_counts_str = _format_counter(method_success_counter)
        merge_method_failure_counts_str = _format_counter(method_failure_counter)
        sampled_method_counts = Counter(info.get("sampled_method_counts") or {})
        sampled_method_counts_str = _format_counter(sampled_method_counts)
        role_counts = Counter(info.get("role_counts") or {})
        role_counts_str = _format_counter(role_counts)
        candidate_score_values = [
            float(row["score"])
            for row in candidate_rows
            if row.get("score") is not None and math.isfinite(float(row["score"]))
        ]
        median_score = (
            float(np.median(candidate_score_values)) if candidate_score_values else None
        )
        passthrough_fraction = (
            float(method_counter.get("passthrough", 0) / len(candidate_rows))
            if candidate_rows
            else 0.0
        )
        gene_diversity_mean = (
            float(np.mean(gene_diversity_values)) if gene_diversity_values else None
        )
        behavior_diversity_mean = (
            float(np.mean(behavior_diversity_values))
            if behavior_diversity_values
            else None
        )
        archive_novelty_mean = (
            float(np.mean(archive_novelty_values)) if archive_novelty_values else None
        )
        stability_mean = float(np.mean(stability_values)) if stability_values else None

        # Compute CSV row values
        generation = int(
            info.get("generation", max(1, step // ga_params.population_size))
        )
        strat.set_runtime_context(generation=generation, phase="ga")
        gen_best = info.get("gen_best")
        gen_mean = info.get("gen_mean")
        gen_std = info.get("gen_std")
        best_so_far = info.get("best_so_far")
        eval_seconds = info.get("eval_seconds", 0.0)
        timestamp = info.get("timestamp", "")
        evaluations = int(info.get("evaluations", 0))
        cache_hits = int(info.get("cache_hits", 0))
        failed_evals = int(info.get("failed_evals", 0))
        failure_reasons = str(info.get("failure_reasons", "") or "")
        crossover_children = int(info.get("crossover_children", 0))
        crossover_type = info.get("crossover_type", ga_params.crossover)
        immigrants = int(info.get("immigrants", 0))

        nonlocal last_global_best

        if gen_best is not None:
            generation_best_history.append(gen_best)
        prev_best = last_global_best if math.isfinite(last_global_best) else None
        gen_best_val = gen_best if gen_best is not None else float("-inf")
        new_global_best = (
            gen_best_val if prev_best is None else max(prev_best, gen_best_val)
        )
        improvement = None if prev_best is None else new_global_best - prev_best
        last_global_best = new_global_best

        gen_best_str = f"{gen_best:.6f}" if gen_best is not None else "None"
        gen_mean_str = f"{gen_mean:.6f}" if gen_mean is not None else "None"
        gen_std_str = f"{gen_std:.6f}" if gen_std is not None else "None"
        global_best_str = (
            f"{new_global_best:.6f}"
            if math.isfinite(new_global_best) and new_global_best > float("-inf")
            else "None"
        )
        if improvement is None:
            delta_str = "init"
        else:
            delta_str = f"{improvement:+.6f}"

        print(
            f"[GA] gen={generation} best={gen_best_str} mean={gen_mean_str} std={gen_std_str} "
            f"global_best={global_best_str} Δbest={delta_str} "
            f"evaluated={evaluations} cache_hits={cache_hits} failed={failed_evals} "
            f"crossover_children={crossover_children} type={crossover_type} immigrants={immigrants}"
            + (f" methods={merge_method_counts_str}" if merge_method_counts_str else "")
            + (
                f" sampled_methods={sampled_method_counts_str}"
                if sampled_method_counts_str
                else ""
            )
            + (f" roles={role_counts_str}" if role_counts_str else "")
            + (
                f" method_success={merge_method_success_counts_str}"
                if merge_method_success_counts_str
                else ""
            )
            + (f" failure_reasons={failure_reasons}" if failure_reasons else "")
        )

        generation_durations.append(max(float(eval_seconds), 0.0))
        completed_generations = len(generation_durations)
        avg_seconds = (
            sum(generation_durations) / completed_generations
            if completed_generations > 0
            else None
        )
        remaining_generations = max(total_generations - completed_generations, 0)
        eta_seconds = (
            avg_seconds * remaining_generations if avg_seconds is not None else None
        )
        print(
            f"[GA] Progress update: completed={completed_generations}/{total_generations} "
            f"avg/gen={_format_time(avg_seconds)} | ETA~{_format_time(eta_seconds)}"
        )
        ray_observer.record_generation_end(
            generation=generation,
            fevals_completed=step,
            generation_best=gen_best,
            generation_mean=gen_mean,
            best_score=best_so_far,
            cache_hits=cache_hits,
            failed_evals=failed_evals,
        )

        # Write/append CSV history for offline tracking
        try:
            hist_path = os.path.join(storage_path, "ga_history.csv")
            header = (
                "generation,fevals,gen_best,gen_mean,gen_std,best_so_far,mutation_sigma,"
                "eval_seconds,timestamp,evaluations,cache_hits,failed_evals,crossover_children,"
                "crossover_type,immigrants,base_model_counts,merge_method_counts,failure_reasons\n"
            )
            line = (
                f"{generation},{step},{gen_best},{gen_mean},{gen_std},{best_so_far},"
                f"{ga_params.mutation_sigma},{eval_seconds},{timestamp},{evaluations},"
                f"{cache_hits},{failed_evals},{crossover_children},{crossover_type},{immigrants},"
                f"{base_model_counts_str},{merge_method_counts_str},{failure_reasons}\n"
            )
            if not os.path.exists(hist_path):
                with open(hist_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write(line)
            else:
                with open(hist_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logging.warning("Failed to write ga_history.csv", exc_info=e)

        try:
            _write_merge_method_history(
                storage_path,
                generation,
                step,
                merge_method_outcomes["history_rows"],
            )
        except Exception as e:
            logging.warning("Failed to write ga_method_history.csv", exc_info=e)

        try:
            _write_candidate_history(storage_path, generation, step, candidate_rows)
        except Exception as e:
            logging.warning("Failed to write ga_candidate_history.csv", exc_info=e)

        new_failed_rows = []
        for genotype_candidate, result in zip(genotype_iterable, res_list):
            if result.get("score") is not None:
                continue
            genotype_hash = genotype_exact_hash(np.asarray(genotype_candidate))
            if genotype_hash in logged_failed_hashes:
                continue
            logged_failed_hashes.add(genotype_hash)
            new_failed_rows.append(
                [
                    generation,
                    step,
                    genotype_hash,
                    result.get("error_stage", "unknown"),
                    result.get("error_type", "unknown"),
                    result.get("error_message", ""),
                ]
            )

        if new_failed_rows:
            try:
                failed_path = os.path.join(storage_path, "failed_genotypes.csv")
                file_exists = os.path.exists(failed_path)
                with open(
                    failed_path, "a", encoding="utf-8", newline=""
                ) as failed_file:
                    writer = csv.writer(failed_file)
                    if not file_exists:
                        writer.writerow(
                            [
                                "generation",
                                "fevals",
                                "genotype_hash",
                                "error_stage",
                                "error_type",
                                "error_message",
                            ]
                        )
                    writer.writerows(new_failed_rows)
            except Exception as e:
                logging.warning("Failed to write failed_genotypes.csv", exc_info=e)

            try:
                blacklist_path = os.path.join(storage_path, FAILED_BLACKLIST_FILENAME)
                file_exists = os.path.exists(blacklist_path)
                with open(
                    blacklist_path, "a", encoding="utf-8", newline=""
                ) as blacklist_file:
                    writer = csv.writer(blacklist_file)
                    if not file_exists:
                        writer.writerow(
                            [
                                "genome_scope",
                                "genotype_hash",
                                "error_stage",
                                "error_type",
                                "error_message",
                            ]
                        )
                    for (
                        _,
                        _,
                        genotype_hash,
                        error_stage,
                        error_type,
                        error_message,
                    ) in new_failed_rows:
                        writer.writerow(
                            [
                                failed_blacklist_scope,
                                genotype_hash,
                                error_stage,
                                error_type,
                                error_message,
                            ]
                        )
            except Exception as e:
                logging.warning(
                    "Failed to write failed_genotype_blacklist.csv", exc_info=e
                )

        # Log per-generation aggregates and extras
        tracker.log_metrics(
            {
                "ga/generation": generation,
                "ga/mutation_sigma": float(ga_params.mutation_sigma),
                "population/eval_seconds": float(eval_seconds),
                "population/evaluations": float(evaluations),
                "population/cache_hits": float(cache_hits),
                "population/failed_evals": float(failed_evals),
                "population/failure_reason_kinds": float(
                    len([x for x in failure_reasons.split(";") if x])
                ),
                "population/gen_best": (
                    float(gen_best) if gen_best is not None else None
                ),
                "population/gen_mean": (
                    float(gen_mean) if gen_mean is not None else None
                ),
                "population/gen_median": median_score,
                "population/gen_std": float(gen_std) if gen_std is not None else None,
                "global/best_so_far": (
                    float(best_so_far) if best_so_far is not None else None
                ),
                "ga/crossover_children": float(crossover_children),
                "ga/immigrants": float(immigrants),
                "population/passthrough_fraction": passthrough_fraction,
                "population/gene_diversity_mean": gene_diversity_mean,
                "population/behavior_diversity_mean": behavior_diversity_mean,
                "population/archive_novelty_mean": archive_novelty_mean,
                "population/stability_mean": stability_mean,
                "ga/adaptive_method_sampling": float(
                    info.get("adaptive_method_sampling", 0.0)
                ),
            },
            step=step,
        )
        tracker.log_metrics(merge_method_outcomes["metrics"], step=step)
        method_probabilities = info.get("method_probabilities") or {}
        if method_probabilities:
            tracker.log_metrics(
                {
                    f"adaptive_method/{_sanitize_metric_key_fragment(method)}/probability_active": float(
                        probability
                    )
                    for method, probability in method_probabilities.items()
                },
                step=step,
            )
        if role_counts:
            tracker.log_metrics(
                {
                    f"population_role/{role}": float(count)
                    for role, count in role_counts.items()
                },
                step=step,
            )

        # Log top-5 scores
        scores = [r["score"] for r in res_list if r["score"] is not None]
        if scores:
            scores_sorted = sorted(scores, reverse=True)[:5]
            top_scores = {"population/top1": float(scores_sorted[0])}
            for k, v in enumerate(scores_sorted[1:], start=2):
                top_scores[f"population/top{k}"] = float(v)
            tracker.log_metrics(top_scores, step=step)

        prune_stale_merged_artifacts(storage_path)

    def on_best(x: np.ndarray, score: float, step: int):
        nonlocal best_x, best_score
        best_x = x.copy()
        best_score = score
        ray_observer.set_phase(
            "ga",
            best_score=float(score),
            fevals_completed=step,
        )
        print(f"New best score: {best_score:.4f}")
        save_best_config(best_x)
        log_best(best_x, best_score, step=step)

    # Choose optimizer based on genome type and parameters
    use_enhanced = (
        genome_type == "multi_method"
        or hasattr(config.ga, "semantic_crossover_prob")
        or getattr(config.ga, "crossover", None) == "semantic"
        or bool(getattr(config.ga, "adaptive_method_sampling", False))
        or getattr(config.ga, "initial_method_probs", None) is not None
        or getattr(config.ga, "passthrough_penalty", None) is not None
        or getattr(config.ga, "passthrough_max_fraction", None) is not None
        or getattr(config.ga, "explorer_fraction", None) is not None
        or bool(getattr(config.ga, "diversity_parent_selection", False))
        or getattr(config.ga, "diversity_parent_weight", None) is not None
        or getattr(config.ga, "gene_diversity_bonus_weight", None) is not None
        or getattr(config.ga, "behavior_diversity_bonus_weight", None) is not None
        or getattr(config.ga, "archive_novelty_bonus_weight", None) is not None
        or getattr(config.ga, "rank_objective_weights", None) is not None
        or config.fitness_mode == "weighted_rank"
    )

    if use_enhanced:
        # Convert GAParams to EnhancedGAParams for new features
        enhanced_params = EnhancedGAParams()
        for key, value in ga_params.__dict__.items():
            if hasattr(enhanced_params, key):
                setattr(enhanced_params, key, value)

        # Set additional enhanced parameters from config
        if (
            hasattr(config.ga, "semantic_crossover_prob")
            and config.ga.semantic_crossover_prob is not None
        ):
            enhanced_params.semantic_crossover_prob = config.ga.semantic_crossover_prob
        if (
            hasattr(config.ga, "method_mutation_rate")
            and config.ga.method_mutation_rate is not None
        ):
            enhanced_params.method_mutation_rate = config.ga.method_mutation_rate
        if (
            hasattr(config.ga, "model_mutation_rate")
            and config.ga.model_mutation_rate is not None
        ):
            enhanced_params.model_mutation_rate = config.ga.model_mutation_rate
        if (
            hasattr(config.ga, "parameter_mutation_rate")
            and config.ga.parameter_mutation_rate is not None
        ):
            enhanced_params.parameter_mutation_rate = config.ga.parameter_mutation_rate
        if (
            hasattr(config.ga, "adaptive_method_sampling")
            and config.ga.adaptive_method_sampling is not None
        ):
            enhanced_params.adaptive_method_sampling = (
                config.ga.adaptive_method_sampling
            )
        if (
            hasattr(config.ga, "initial_method_probs")
            and config.ga.initial_method_probs is not None
        ):
            enhanced_params.initial_method_probs = dict(config.ga.initial_method_probs)
        if (
            hasattr(config.ga, "operator_temperature")
            and config.ga.operator_temperature is not None
        ):
            enhanced_params.operator_temperature = config.ga.operator_temperature
        if (
            hasattr(config.ga, "operator_update_smoothing")
            and config.ga.operator_update_smoothing is not None
        ):
            enhanced_params.operator_update_smoothing = (
                config.ga.operator_update_smoothing
            )
        if (
            hasattr(config.ga, "operator_avg_child_weight")
            and config.ga.operator_avg_child_weight is not None
        ):
            enhanced_params.operator_avg_child_weight = (
                config.ga.operator_avg_child_weight
            )
        if (
            hasattr(config.ga, "operator_parent_improvement_weight")
            and config.ga.operator_parent_improvement_weight is not None
        ):
            enhanced_params.operator_parent_improvement_weight = (
                config.ga.operator_parent_improvement_weight
            )
        if (
            hasattr(config.ga, "operator_survival_weight")
            and config.ga.operator_survival_weight is not None
        ):
            enhanced_params.operator_survival_weight = (
                config.ga.operator_survival_weight
            )
        if (
            hasattr(config.ga, "passthrough_penalty")
            and config.ga.passthrough_penalty is not None
        ):
            enhanced_params.passthrough_penalty = config.ga.passthrough_penalty
        if (
            hasattr(config.ga, "passthrough_max_fraction")
            and config.ga.passthrough_max_fraction is not None
        ):
            enhanced_params.passthrough_max_fraction = (
                config.ga.passthrough_max_fraction
            )
        if (
            hasattr(config.ga, "explorer_fraction")
            and config.ga.explorer_fraction is not None
        ):
            enhanced_params.explorer_fraction = config.ga.explorer_fraction
        if (
            hasattr(config.ga, "diversity_parent_selection")
            and config.ga.diversity_parent_selection is not None
        ):
            enhanced_params.diversity_parent_selection = (
                config.ga.diversity_parent_selection
            )
        if (
            hasattr(config.ga, "diversity_parent_weight")
            and config.ga.diversity_parent_weight is not None
        ):
            enhanced_params.diversity_parent_weight = config.ga.diversity_parent_weight
        if (
            hasattr(config.ga, "gene_diversity_bonus_weight")
            and config.ga.gene_diversity_bonus_weight is not None
        ):
            enhanced_params.gene_diversity_bonus_weight = (
                config.ga.gene_diversity_bonus_weight
            )
        if (
            hasattr(config.ga, "behavior_diversity_bonus_weight")
            and config.ga.behavior_diversity_bonus_weight is not None
        ):
            enhanced_params.behavior_diversity_bonus_weight = (
                config.ga.behavior_diversity_bonus_weight
            )
        if (
            hasattr(config.ga, "archive_novelty_bonus_weight")
            and config.ga.archive_novelty_bonus_weight is not None
        ):
            enhanced_params.archive_novelty_bonus_weight = (
                config.ga.archive_novelty_bonus_weight
            )
        if (
            hasattr(config.ga, "novelty_archive_size")
            and config.ga.novelty_archive_size is not None
        ):
            enhanced_params.novelty_archive_size = config.ga.novelty_archive_size
        if (
            hasattr(config.ga, "rank_objective_weights")
            and config.ga.rank_objective_weights is not None
        ):
            enhanced_params.rank_objective_weights = dict(
                config.ga.rank_objective_weights
            )
        enhanced_params.fitness_mode = config.fitness_mode

        optimizer = EnhancedGAOptimizer(
            genome=genome,
            strategy=strat,
            params=enhanced_params,
            random_init=config.random_init,
            seed=random_seed,
            persisted_failed_genotypes=persisted_failed_genotypes,
            on_population_evaluated=on_pop,
            on_new_best=on_best,
            on_generation_start=on_generation_start,
        )
    else:
        # Use traditional optimizer
        optimizer = GAOptimizer(
            genome=genome,
            strategy=strat,
            params=ga_params,
            random_init=config.random_init,
            seed=random_seed,
            persisted_failed_genotypes=persisted_failed_genotypes,
            on_population_evaluated=on_pop,
            on_new_best=on_best,
            on_generation_start=on_generation_start,
        )

    if baseline_csv_path:
        stage_log(
            "Stage-GA",
            f"Baseline metrics located at {baseline_csv_path}",
        )
        if baseline_best_score is not None and math.isfinite(baseline_best_score):
            stage_log(
                "Stage-GA",
                f"Best baseline weighted_score: {baseline_best_score:.4f}",
            )
    else:
        stage_log("Stage-GA", "No baseline metrics available for this run.")

    stage_log("Stage-GA", "Starting GA optimization loop...")
    ray_observer.set_phase(
        "ga",
        generation=0,
        fevals_completed=0,
        best_score=(
            float(baseline_best_score)
            if baseline_best_score is not None and math.isfinite(baseline_best_score)
            else None
        ),
    )
    strat.set_runtime_context(generation=0, phase="ga")
    prune_stale_merged_artifacts(storage_path)
    try:
        best_x, best_score = optimizer.run(max_fevals=max_fevals, timeout=timeout)
    except KeyboardInterrupt:
        ray_observer.set_phase(
            "failed",
            best_score=(
                float(best_score)
                if best_score is not None and math.isfinite(best_score)
                else None
            ),
            fevals_completed=0,
        )
        ray.shutdown()
        raise

    stage_log("Stage-GA", "Optimization complete.")
    stage_log("Stage-GA", f"Best score achieved: {best_score:.4f}")
    stop_details = getattr(optimizer, "last_stop_details", None) or {}
    stop_details_path = _write_stop_details(
        storage_path,
        resolved_stop=resolved_stop,
        stop_details=stop_details,
    )
    stage_log("Stage-GA", f"Stop details written to {stop_details_path}")
    if stop_details.get("reason"):
        stop_reason = str(stop_details["reason"])
        stop_bits = [f"reason={stop_reason}"]
        if stop_details.get("generation") is not None:
            stop_bits.append(f"generation={stop_details['generation']}")
        if stop_details.get("fevals") is not None:
            stop_bits.append(f"fevals={stop_details['fevals']}")
        if stop_details.get("best_improvement_pct") is not None:
            stop_bits.append(
                f"improvement_pct={float(stop_details['best_improvement_pct']):+.2f}%"
            )
        stage_log("Stage-GA", "Stop outcome: " + " ".join(stop_bits))

    _write_ga_outputs(storage_path, stop_details=stop_details)

    if generation_best_history:
        initial_best = generation_best_history[0]
        final_best = max(generation_best_history)
        delta_overall = final_best - initial_best
        summary_parts = [
            f"generations={len(generation_best_history)}",
            f"initial_best={initial_best:.4f}",
            f"final_best={final_best:.4f}",
            f"Δbest={delta_overall:+.4f}",
        ]
        if baseline_best_score is not None and math.isfinite(baseline_best_score):
            baseline_delta, baseline_pct_change = _score_improvement(
                final_best,
                baseline_best_score,
            )
            summary_parts.append(f"baseline_best={baseline_best_score:.4f}")
            summary_parts.append(f"Δvs_baseline={baseline_delta:+.4f}")
            if baseline_pct_change is not None:
                pct_str = (
                    f"Δvs_baseline_pct={baseline_pct_change:+.2f}%"
                    if math.isfinite(baseline_pct_change)
                    else "Δvs_baseline_pct=undefined"
                )
            else:  # pragma: no cover - guard against zero baseline best
                pct_str = "Δvs_baseline_pct=undefined"
            summary_parts.append(pct_str)
        if stop_details.get("reason"):
            summary_parts.append(f"stop_reason={stop_details['reason']}")

        stage_log(
            "Stage-GA",
            "Run summary: " + " ".join(summary_parts),
        )
    else:
        stage_log("Stage-GA", "Run summary: no successful generations recorded.")

    # pause for a bit to let any CUDA-using processes clean up
    time.sleep(1.0)

    # save the best merge configuration using original model references
    if best_x is not None:
        best_config = None
        best_plan_dict = None
        if genome_type == "multi_method":
            genome_pretty = MultiMethodGenome(
                MultiMethodGenomeDefinition.model_validate(config.genome.model_dump()),
                trust_remote_code=trust_remote_code,
            )
            if hasattr(genome_pretty, "execution_plan_dict"):
                best_plan_dict = genome_pretty.execution_plan_dict(best_x)
            if not best_plan_dict or best_plan_dict.get("kind") == "config":
                best_config = genome_pretty.genotype_to_merge_config(best_x)
        else:
            genome_pretty = ModelGenome(
                config.genome, trust_remote_code=trust_remote_code
            )
            best_config = genome_pretty.genotype_merge_config(best_x)

        stage_log("Stage-GA", "Best merge configuration computed.")
        if best_config is not None:
            print(best_config.to_yaml())
        elif best_plan_dict is not None:
            print(
                yaml.safe_dump(
                    {"layered_execution_plan": best_plan_dict},
                    sort_keys=False,
                ).rstrip()
            )

        if save_final_model:
            stage_log("Stage-GA", "Saving final merged model artifacts...")
            final_model_path = os.path.join(storage_path, "final_model")
            if best_config is not None:
                run_merge(best_config, final_model_path, merge_options)
            else:
                merge_result = merge_model_with_details(
                    best_x,
                    genome_pretty,
                    os.path.join(storage_path, "merged"),
                    merge_options,
                )
                merged_path = merge_result.get("merged_path")
                if not merged_path:
                    raise RuntimeError(
                        merge_result.get(
                            "error_message", "Failed to materialize layered final model"
                        )
                    )
                if os.path.exists(final_model_path):
                    shutil.rmtree(final_model_path, ignore_errors=True)
                shutil.copytree(merged_path, final_model_path)
                shutil.rmtree(merged_path, ignore_errors=True)

            ray_observer.set_phase(
                "final_compare",
                best_score=float(best_score) if math.isfinite(best_score) else None,
            )
            _evaluate_and_write_final_comparison(
                config,
                storage_path,
                batch_size,
                merge_cuda,
                num_gpus,
                task_search_path,
                trust_remote_code,
            )

            # Append GA run details to the model card (README.md)
            try:
                readme_path = os.path.join(storage_path, "final_model", "README.md")
                task_lines = [
                    f"- {task.name} (metric: {task.metric}, weight: {task.weight})"
                    for task in config.tasks
                ]
                summary_lines = [
                    "\n## Merge Run Details",
                    "This model was produced via a GA-driven merge search.",
                    f"- Run timestamp: {datetime.now().isoformat()}",
                    f"- Max evaluations: {max_fevals}",
                    f"- Population size: {ga_params.population_size}",
                    f"- Elite fraction: {ga_params.elite_fraction}",
                    f"- Mutation rate: {ga_params.mutation_rate}",
                    f"- Mutation sigma: {ga_params.mutation_sigma}",
                    f"- Crossover: {ga_params.crossover}",
                    f"- Tournament size: {ga_params.tournament_size}",
                    f"- Generations completed: {len(generation_best_history)}",
                    f"- Best score: {best_score:.6f}",
                ]
                if stop_details.get("reason"):
                    summary_lines.append(
                        f"- Stop reason: {str(stop_details['reason'])}"
                    )
                if stop_details.get("fevals") is not None:
                    summary_lines.append(
                        f"- Function evaluations completed: {int(stop_details['fevals'])}"
                    )
                if baseline_best_score is not None and math.isfinite(
                    baseline_best_score
                ):
                    baseline_delta, baseline_pct_change = _score_improvement(
                        best_score,
                        baseline_best_score,
                    )
                    summary_lines.append(
                        f"- Best baseline score: {baseline_best_score:.6f}"
                    )
                    summary_lines.append(f"- Δ vs baseline: {baseline_delta:+.6f}")
                    if baseline_pct_change is not None and math.isfinite(
                        baseline_pct_change
                    ):
                        summary_lines.append(
                            f"- Δ vs baseline (%): {baseline_pct_change:+.2f}%"
                        )
                summary_lines.append("- Tasks:")
                summary_lines.extend(task_lines)
                with open(readme_path, "a", encoding="utf-8") as fp:
                    fp.write("\n" + "\n".join(summary_lines) + "\n")
            except Exception as exc:  # pragma: no cover - best-effort card update
                stage_log(
                    "Stage-GA",
                    f"Unable to append GA details to README.md: {exc}",
                    level=logging.WARNING,
                )

            # Upload to Hugging Face if requested
            if hf_model_id:
                allow_upload = True
                if baseline_best_score is None or not math.isfinite(
                    baseline_best_score
                ):
                    stage_log(
                        "Stage-GA",
                        "Baseline score unavailable; skipping upload due to improvement thresholds.",
                        level=logging.WARNING,
                    )
                    allow_upload = False
                else:
                    delta, pct = _score_improvement(best_score, baseline_best_score)
                    if not _meets_improvement_thresholds(
                        delta,
                        pct,
                        hf_min_improvement,
                        hf_min_improvement_pct,
                    ):
                        pct_display = (
                            f"{pct:+.2f}%"
                            if pct is not None and math.isfinite(pct)
                            else "undefined"
                        )
                        stage_log(
                            "Stage-GA",
                            "Upload skipped: improvement thresholds not met. "
                            f"Δ={delta:+.6f} (min {hf_min_improvement:+.6f}), "
                            f"Δ%={pct_display} (min {hf_min_improvement_pct:.2f}%).",
                            level=logging.WARNING,
                        )
                        allow_upload = False

                if allow_upload:
                    stage_log(
                        "Stage-GA",
                        f"Uploading final model to Hugging Face: {hf_model_id}",
                    )
                    try:
                        from huggingface_hub import upload_folder

                        final_model_path = os.path.join(storage_path, "final_model")
                        upload_folder(
                            repo_id=hf_model_id,
                            folder_path=final_model_path,
                            repo_type="model",
                        )
                        stage_log(
                            "Stage-GA",
                            f"Model successfully uploaded to {hf_model_id}",
                        )
                    except Exception as e:
                        stage_log(
                            "Stage-GA",
                            f"Failed to upload model to Hugging Face: {e}",
                            level=logging.ERROR,
                        )
                        stage_log(
                            "Stage-GA",
                            f"You can manually upload from: {os.path.join(storage_path, 'final_model')}",
                        )
        prune_stale_merged_artifacts(storage_path)
        ray_observer.set_phase(
            "finished",
            best_score=float(best_score) if math.isfinite(best_score) else None,
            fevals_completed=int(stop_details.get("fevals") or max_fevals),
        )
    else:
        stage_log(
            "Stage-GA",
            "No valid solution found. All evaluations failed.",
            level=logging.ERROR,
        )
        stage_log("Stage-GA", "Possible causes:")
        stage_log("Stage-GA", "- Model compatibility issues")
        stage_log("Stage-GA", "- Evaluation environment problems")
        stage_log("Stage-GA", "- Insufficient population size or evaluations")
        prune_stale_merged_artifacts(storage_path)
        ray_observer.set_phase(
            "failed",
            fevals_completed=int(stop_details.get("fevals") or 0),
        )

    _log_run_artifacts(tracker, storage_path)
    tracker.finish()


def run_baseline_evaluations(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
    ray_observer: Optional[RayRunObserver] = None,
) -> Optional[str]:
    """Execute baseline evaluations for all models defined in the genome."""
    stage_log("Stage-Baseline", "Starting baseline evaluation phase...")

    storage_dir = os.path.abspath(storage_path)
    os.makedirs(storage_dir, exist_ok=True)

    models: List[ModelReference] = list(config.genome.models)
    if getattr(config.genome, "base_model", None) is not None:
        models.append(config.genome.base_model)
    models = _unique_model_refs(models)

    if not models:
        stage_log(
            "Stage-Baseline", "No models found in the genome; skipping baselines."
        )
        if ray_observer is not None:
            ray_observer.set_phase(
                "baseline", baseline_model_index=0, baseline_model_total=0
            )
        return None

    if ray_observer is not None:
        ray_observer.set_phase(
            "baseline",
            baseline_model_index=0,
            baseline_model_total=len(models),
        )

    baseline_csv_path = _reusable_baseline_csv_path(
        os.path.join(storage_dir, "baseline_results.csv"),
        [str(model_ref) for model_ref in models],
    )
    if baseline_csv_path is not None:
        stage_log(
            "Stage-Baseline",
            f"Reusing existing baseline metrics from {baseline_csv_path}",
        )
        if ray_observer is not None:
            ray_observer.set_phase(
                "baseline",
                baseline_model_index=len(models),
                baseline_model_total=len(models),
            )
        return baseline_csv_path

    required_task_names = _configured_task_names(config)
    task_manager = create_task_manager(
        task_search_path,
        required_tasks=required_task_names,
    )

    _init_ray_for_baselines()

    use_cuda = (merge_cuda or (num_gpus or 0) > 0) and (num_gpus or 0) > 0
    device = "cuda" if use_cuda else "cpu"
    stage_log(
        "Stage-Baseline",
        f"Using Ray {'GPU' if use_cuda else 'CPU'} workers for baseline evaluations.",
    )

    metric_columns = [f"{task.name}:{task.metric}" for task in config.tasks]
    baseline_rows: List[Dict[str, Union[str, float, None]]] = []
    successes = 0
    failures = 0

    @ray.remote(num_cpus=1, num_gpus=1.0)
    def _baseline_eval_gpu(
        model_name: str,
        tasks: List[TaskConfiguration],
        num_fewshot: int,
        limit: Optional[int],
        batch_size: Optional[int],
        task_search_path: List[str],
        required_tasks: List[str],
        trust_remote_code: bool,
        fitness_mode: str,
        task_mix_profile: Optional[str],
    ) -> Dict[str, Any]:
        task_manager = create_task_manager(
            task_search_path,
            required_tasks=required_tasks,
        )
        model_args: Dict[str, Any] = {
            "pretrained": model_name,
            "dtype": "bfloat16",
            "use_cache": True,
            "trust_remote_code": trust_remote_code,
        }
        try:
            result = _eval_model(
                "huggingface",
                tasks,
                model_args,
                num_fewshot=num_fewshot,
                limit=limit,
                batch_size=batch_size,
                task_manager=task_manager,
                fitness_mode=fitness_mode,
                task_mix_profile=task_mix_profile,
                bootstrap_iters=0,
                device="cuda",
            )
            return {
                "score": result.get("score"),
                "results": result.get("results"),
                "error": None,
            }
        except Exception as exc:
            return {
                "score": None,
                "results": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @ray.remote(num_cpus=1)
    def _baseline_eval_cpu(
        model_name: str,
        tasks: List[TaskConfiguration],
        num_fewshot: int,
        limit: Optional[int],
        batch_size: Optional[int],
        task_search_path: List[str],
        required_tasks: List[str],
        trust_remote_code: bool,
        fitness_mode: str,
        task_mix_profile: Optional[str],
    ) -> Dict[str, Any]:
        task_manager = create_task_manager(
            task_search_path,
            required_tasks=required_tasks,
        )
        model_args: Dict[str, Any] = {
            "pretrained": model_name,
            "dtype": "float32",
            "use_cache": True,
            "trust_remote_code": trust_remote_code,
        }
        try:
            result = _eval_model(
                "huggingface",
                tasks,
                model_args,
                num_fewshot=num_fewshot,
                limit=limit,
                batch_size=batch_size,
                task_manager=task_manager,
                fitness_mode=fitness_mode,
                task_mix_profile=task_mix_profile,
                bootstrap_iters=0,
                device="cpu",
            )
            return {
                "score": result.get("score"),
                "results": result.get("results"),
                "error": None,
            }
        except Exception as exc:
            return {
                "score": None,
                "results": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    baseline_remote = _baseline_eval_gpu if use_cuda else _baseline_eval_cpu
    baseline_refs = {
        str(model_ref): baseline_remote.remote(
            str(model_ref),
            config.tasks,
            config.num_fewshot,
            config.limit,
            batch_size,
            task_search_path,
            required_task_names,
            trust_remote_code,
            config.fitness_mode,
            config.task_mix_profile,
        )
        for model_ref in models
    }

    for model_index, model_ref in enumerate(models, start=1):
        model_name = str(model_ref)
        row: Dict[str, Union[str, float, None]] = {
            "model": model_name,
            "weighted_score": None,
            "error": None,
        }
        for column in metric_columns:
            row.setdefault(column, None)

        stage_log("Stage-Baseline", f"Evaluating {model_name}...")
        try:
            result = ray.get(baseline_refs[model_name])
        except (
            Exception
        ) as exc:  # pragma: no cover - remote environment depends on cluster state
            failures += 1
            row["error"] = str(exc)
            stage_log(
                "Stage-Baseline",
                f"Evaluation failed for {model_name}: {exc}",
                level=logging.ERROR,
            )
            if ray_observer is not None:
                ray_observer.record_baseline_progress(
                    model_index=model_index,
                    model_total=len(models),
                    model_name=model_name,
                    score=None,
                    failed=True,
                )
            LOGGER.debug("Baseline evaluation error", exc_info=exc)
            baseline_rows.append(row)
            continue

        if not result or result.get("error"):
            failures += 1
            row["error"] = (
                result.get("error")
                if result
                else "Baseline evaluation returned no result"
            )
            stage_log(
                "Stage-Baseline",
                f"Evaluation failed for {model_name}: {row['error']}",
                level=logging.ERROR,
            )
            if ray_observer is not None:
                ray_observer.record_baseline_progress(
                    model_index=model_index,
                    model_total=len(models),
                    model_name=model_name,
                    score=None,
                    failed=True,
                )
            baseline_rows.append(row)
            continue

        successes += 1
        weighted_score = result.get("score")
        row["weighted_score"] = weighted_score
        if ray_observer is not None:
            ray_observer.record_baseline_progress(
                model_index=model_index,
                model_total=len(models),
                model_name=model_name,
                score=(
                    float(weighted_score)
                    if weighted_score is not None and math.isfinite(weighted_score)
                    else None
                ),
                failed=False,
            )

        row.update(_collect_task_metrics(result, config.tasks))

        baseline_rows.append(row)

    if not baseline_rows:
        stage_log(
            "Stage-Baseline", "No baseline results recorded; skipping CSV output."
        )
        return None

    baseline_df = pandas.DataFrame(baseline_rows)
    ordered_columns = ["model", "weighted_score", *metric_columns, "error"]
    # Ensure DataFrame includes expected columns even if absent from rows
    for column in ordered_columns:
        if column not in baseline_df.columns:
            baseline_df[column] = None
    baseline_df = baseline_df[ordered_columns]
    baseline_df.sort_values(
        "weighted_score",
        ascending=False,
        inplace=True,
        na_position="last",
    )

    baseline_csv_path = os.path.join(storage_dir, "baseline_results.csv")
    baseline_df.to_csv(baseline_csv_path, index=False)

    stage_log(
        "Stage-Baseline",
        f"Baseline metrics saved to {baseline_csv_path}",
    )
    stage_log(
        "Stage-Baseline",
        f"Completed evaluations: {successes}; failures: {failures}",
    )
    if ray_observer is not None:
        ray_observer.set_phase(
            "baseline",
            baseline_model_index=len(models),
            baseline_model_total=len(models),
            failed_evals=failures,
        )

    return baseline_csv_path


def _collect_task_metrics(
    result: Dict[str, Any],
    tasks: List[TaskConfiguration],
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for task_cfg in tasks:
        task_results = result.get("results", {}).get(task_cfg.name, {})
        metric_value = task_results.get(task_cfg.metric)

        if metric_value is None:
            metric_alternatives = {
                "ppl,none": [
                    "word_perplexity,none",
                    "perplexity,none",
                    "byte_perplexity,none",
                ],
                "acc,none": ["acc,none", "acc_norm,none", "accuracy,none"],
                "acc_norm,none": [
                    "acc_norm,none",
                    "acc,none",
                    "accuracy,none",
                ],
            }
            for alt_metric in metric_alternatives.get(task_cfg.metric, []):
                if alt_metric in task_results:
                    metric_value = task_results[alt_metric]
                    break

            if metric_value is None:
                lowered_metric = task_cfg.metric.lower()
                for metric_name, value in task_results.items():
                    lowered_name = metric_name.lower()
                    if "stderr" in lowered_name:
                        continue
                    if (
                        "ppl" in lowered_metric or "perplexity" in lowered_metric
                    ) and "perplexity" in lowered_name:
                        metric_value = value
                        break
                    if "acc" in lowered_metric and "acc" in lowered_name:
                        metric_value = value
                        break

        if isinstance(metric_value, float) and math.isnan(metric_value):
            metric_value = None

        metrics[f"{task_cfg.name}:{task_cfg.metric}"] = metric_value

    return metrics


def _evaluate_and_write_final_comparison(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
) -> None:
    baseline_csv = os.path.join(storage_path, "baseline_results.csv")
    final_model_path = os.path.join(storage_path, "final_model")
    comparison_csv = os.path.join(storage_path, "ga_final_comparison.csv")

    if not os.path.exists(final_model_path):
        stage_log("Stage-GA", "final_model not found; skipping final comparison table.")
        return

    baseline_rows: List[Dict[str, Union[str, float, None]]] = []
    if os.path.exists(baseline_csv):
        baseline_rows = pandas.read_csv(baseline_csv).to_dict(orient="records")
    else:
        stage_log(
            "Stage-GA",
            "baseline_results.csv not found; comparison will include only final model.",
            level=logging.WARNING,
        )

    task_manager = create_task_manager(
        task_search_path,
        required_tasks=_configured_task_names(config),
    )
    use_cuda = torch.cuda.is_available() and (merge_cuda or (num_gpus or 0) > 0)
    device = "cuda" if use_cuda else "cpu"

    model_args: Dict[str, Any] = {
        "pretrained": final_model_path,
        "dtype": "float32",
        "use_cache": True,
        "trust_remote_code": trust_remote_code,
    }
    eval_kwargs: Dict[str, Any] = {"device": device}

    stage_log("Stage-GA", "Evaluating final merged model for comparison table...")
    try:
        result = _eval_model(
            "huggingface",
            config.tasks,
            model_args,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            batch_size=batch_size,
            task_manager=task_manager,
            fitness_mode=config.fitness_mode,
            task_mix_profile=config.task_mix_profile,
            bootstrap_iters=0,
            **eval_kwargs,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime
        stage_log(
            "Stage-GA",
            f"Final model evaluation failed; skipping comparison table: {exc}",
            level=logging.ERROR,
        )
        return

    metric_columns = [f"{task.name}:{task.metric}" for task in config.tasks]
    final_row: Dict[str, Union[str, float, None]] = {
        "model": "final_merged",
        "weighted_score": result.get("score"),
        "error": None,
    }
    final_row.update(_collect_task_metrics(result, config.tasks))

    all_rows = baseline_rows + [final_row]
    comparison_df = pandas.DataFrame(all_rows)
    ordered_columns = ["model", "weighted_score", *metric_columns, "error"]
    for column in ordered_columns:
        if column not in comparison_df.columns:
            comparison_df[column] = None
    comparison_df = comparison_df[ordered_columns]
    comparison_df.to_csv(comparison_csv, index=False)

    _write_comparison_plot(
        comparison_df, os.path.join(storage_path, "ga_final_comparison.png")
    )


def _write_comparison_plot(table: "pandas.DataFrame", output_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_height = max(2.5, 0.35 * (len(table) + 1))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis("off")

        display_df = table.copy()
        if "weighted_score" in display_df.columns:
            display_df["weighted_score"] = display_df["weighted_score"].map(
                lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v
            )

        tbl = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.2)

        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional plotting
        stage_log(
            "Stage-GA",
            f"Skipping comparison plot generation: {exc}",
            level=logging.WARNING,
        )


def _write_ga_outputs(
    storage_path: str, *, stop_details: Optional[Dict[str, Any]] = None
) -> None:
    """Write a compact GA summary table and optional plot to the run outputs."""
    history_path = os.path.join(storage_path, "ga_history.csv")
    if not os.path.exists(history_path):
        stage_log("Stage-GA", "ga_history.csv not found; skipping summary outputs.")
        return

    rows: List[Dict[str, str]] = []
    with open(history_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        stage_log("Stage-GA", "ga_history.csv is empty; skipping summary outputs.")
        return

    summary_path = os.path.join(storage_path, "ga_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("gen  fevals  gen_best   best_so_far  eval_s  cache_hits  crossover\n")
        for r in rows:
            gen = int(r.get("generation", "0") or 0)
            fevals = int(r.get("fevals", "0") or 0)
            gen_best = float(r.get("gen_best", "nan") or float("nan"))
            best_so_far_raw = r.get("best_so_far", "")
            best_so_far = (
                "NaN" if best_so_far_raw in ("", "None", "-inf") else best_so_far_raw
            )
            eval_s = float(r.get("eval_seconds", "0") or 0)
            cache_hits = int(r.get("cache_hits", "0") or 0)
            crossover = r.get("crossover_type", "")
            f.write(
                f"{gen:>2}  {fevals:>6}  {gen_best:>8.5f}  {best_so_far:>10}  "
                f"{eval_s:>6.1f}     {cache_hits:>3}       {crossover}\n"
            )

        gen_best_vals = [float(r.get("gen_best", 0.0) or 0.0) for r in rows]
        finite_vals = [v for v in gen_best_vals if math.isfinite(v)]
        blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        f.write("\nGen-best sparkline:\n")
        if not finite_vals:
            f.write("(insufficient finite values)\n")
            f.write("min=N/A max=N/A\n")
        else:
            min_v = min(finite_vals)
            max_v = max(finite_vals)
            if max_v == min_v:
                spark = "".join(blocks[0] for _ in gen_best_vals)
            else:
                spark = "".join(
                    blocks[
                        min(
                            len(blocks) - 1,
                            max(
                                0,
                                int(
                                    ((v if math.isfinite(v) else min_v) - min_v)
                                    / (max_v - min_v)
                                    * (len(blocks) - 1)
                                ),
                            ),
                        )
                    ]
                    for v in gen_best_vals
                )
            f.write(spark + "\n")
            f.write(f"min={min_v:.5f} max={max_v:.5f}\n")

        if stop_details:
            f.write("\nStop details:\n")
            for key in sorted(stop_details.keys()):
                f.write(f"{key}={stop_details[key]}\n")

    # Optional plot (best/mean over generations)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens = [int(r.get("generation", "0") or 0) for r in rows]
        gen_best = [float(r.get("gen_best", 0.0) or 0.0) for r in rows]
        gen_mean = [float(r.get("gen_mean", 0.0) or 0.0) for r in rows]

        plot_path = os.path.join(storage_path, "ga_history_plot.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(gens, gen_best, marker="o", label="gen_best")
        plt.plot(gens, gen_mean, marker="x", label="gen_mean")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("GA Progress")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover - optional plotting
        stage_log(
            "Stage-GA",
            f"Skipping ga_history_plot.png generation: {exc}",
            level=logging.WARNING,
        )


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    merged = model.merged(
        cache_dir=merge_cache,
        trust_remote_code=trust_remote_code,
    )
    out_path = os.path.join(
        storage_path,
        "input_models",
        merged.model._unique_id(),
    )

    if os.path.exists(out_path):
        logging.info(f"Using existing resharded model at {out_path}")
        return ModelReference(model=out_path)

    model_hf = call_with_dtype(
        transformers.AutoModelForCausalLM.from_pretrained,
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
        cache_dir=os.path.join(storage_path, "transformers_cache"),
    )
    model_hf.save_pretrained(
        out_path, safe_serialization=True, out_shard_size=1_000_000_000_000
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.model.path,
            revision=model.model.revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logging.warning(f"Could not save tokenizer for {model.model}", exc_info=e)

    return ModelReference(model=out_path)


if __name__ == "__main__":
    main()
