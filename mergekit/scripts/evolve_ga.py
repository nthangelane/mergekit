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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import click
import numpy as np
import pandas
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
from mergekit.evo.baselines import (
    best_weighted_score_from_frame as _best_weighted_score_from_frame,
)
from mergekit.evo.baselines import collect_task_metrics as _collect_task_metrics
from mergekit.evo.baselines import configured_task_names as _configured_task_names
from mergekit.evo.baselines import (
    reusable_baseline_csv_path as _reusable_baseline_csv_path,
)
from mergekit.evo.baselines import run_baseline_evaluations as _run_baseline_evaluations
from mergekit.evo.baselines import unique_model_refs as _unique_model_refs
from mergekit.evo.cache_utils import genotype_exact_hash
from mergekit.evo.checkpoint import GA_STATE_FILENAME, atomic_write_json, load_ga_state
from mergekit.evo.config import EvolMergeConfiguration, check_for_naughty_config
from mergekit.evo.enhanced_ga import EnhancedGAOptimizer
from mergekit.evo.fitness import ensure_fitness_definition
from mergekit.evo.ga import GAOptimizer
from mergekit.evo.genome import ModelGenome
from mergekit.evo.multi_method_genome import (
    MultiMethodGenome,
    MultiMethodGenomeDefinition,
)
from mergekit.evo.optimizer_factory import (
    build_enhanced_ga_params,
    resolve_optimizer_kind,
)
from mergekit.evo.orchestrator import (
    build_genome,
    build_run_signature,
)
from mergekit.evo.orchestrator import resolve_device as _resolve_device
from mergekit.evo.orchestrator import (
    resolve_evaluation_strategy,
    resolve_ga_params,
)
from mergekit.evo.orchestrator import resolve_merge_cuda as _resolve_merge_cuda
from mergekit.evo.orchestrator import (
    resolve_stop_configuration as _resolve_stop_configuration,
)
from mergekit.evo.progress import ProgressLogger
from mergekit.evo.provenance import (
    annotate_lineage_risk,
    inspect_parent_lineage,
    lineage_policy_failed,
    write_parent_lineage_report,
)
from mergekit.evo.random_search import RandomSearchOptimizer
from mergekit.evo.ray_observability import RayRunObserver, default_run_label
from mergekit.evo.reporting import (
    classify_solution_novelty as _classify_solution_novelty,
)
from mergekit.evo.reporting import (
    collect_merge_method_outcomes as _collect_merge_method_outcomes,
)
from mergekit.evo.reporting import configured_merge_methods as _configured_merge_methods
from mergekit.evo.reporting import (
    evaluate_and_write_final_comparison as _evaluate_and_write_final_comparison,
)
from mergekit.evo.reporting import log_run_artifacts as _log_run_artifacts
from mergekit.evo.reporting import (
    meets_improvement_thresholds as _meets_improvement_thresholds,
)
from mergekit.evo.reporting import (
    sanitize_metric_key_fragment as _sanitize_metric_key_fragment,
)
from mergekit.evo.reporting import score_improvement as _score_improvement
from mergekit.evo.reporting import write_candidate_history as _write_candidate_history
from mergekit.evo.reporting import write_ga_outputs as _write_ga_outputs
from mergekit.evo.reporting import (
    write_merge_method_history as _write_merge_method_history,
)
from mergekit.evo.reporting import write_stop_details as _write_stop_details
from mergekit.evo.resources import InsufficientDiskSpaceError, ensure_free_disk
from mergekit.evo.tracking import create_tracker
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

LOGGER = logging.getLogger("mergekit.evolve_ga.cli")
FAILED_BLACKLIST_FILENAME = "failed_genotype_blacklist.csv"


def _require_ray():
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise click.ClickException(
            "Ray is not installed. Use --strategy serial --device cpu, or install "
            "the distributed experiment dependencies."
        ) from exc
    return ray


def merge_model_with_details(*args, **kwargs):
    from mergekit.evo.helpers import merge_model_with_details as implementation

    return implementation(*args, **kwargs)


def validate_input_model_architecture(*args, **kwargs):
    from mergekit.evo.helpers import validate_input_model_architecture as implementation

    return implementation(*args, **kwargs)


def stage_log(stage: str, message: str, *, level: int = logging.INFO) -> None:
    """Emit a structured log message for high-level run stages."""
    LOGGER.log(level, "[%s] %s", stage, message)


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


def _write_disk_abort(storage_path: str, exc: InsufficientDiskSpaceError) -> str:
    state_path = os.path.join(storage_path, GA_STATE_FILENAME)
    abort_path = os.path.join(storage_path, "run_abort.json")
    atomic_write_json(
        abort_path,
        {
            "reason": "insufficient_disk_space",
            "message": str(exc),
            "free_disk_gb": exc.free_gb,
            "required_disk_gb": exc.required_gb,
            "ga_state": state_path if os.path.isfile(state_path) else None,
        },
    )
    return abort_path


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

    ray = _require_ray()
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


def run_baseline_evaluations(
    config: EvolMergeConfiguration,
    storage_path: str,
    batch_size: Optional[int],
    merge_cuda: bool,
    num_gpus: Optional[int],
    task_search_path: List[str],
    trust_remote_code: bool,
    ray_observer: Optional[RayRunObserver] = None,
    use_ray: bool = True,
) -> Optional[str]:
    return _run_baseline_evaluations(
        config,
        storage_path,
        batch_size,
        merge_cuda,
        num_gpus,
        task_search_path,
        trust_remote_code,
        ray_observer=ray_observer,
        use_ray=use_ray,
        stage_logger=stage_log,
        ray_initializer=_init_ray_for_baselines,
        ray_loader=_require_ray,
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
    "--random-search",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="Evaluate N uniform random genotypes without selection or breeding",
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
    required=False,
)
@click.option(
    "--resume",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Resume an enhanced GA run from RUN_DIR/ga_state.json",
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
    help="Execution device; auto selects CUDA only when available",
)
@click.option(
    "--max-disk-gb-min",
    type=click.FloatRange(min=0.0),
    default=5.0,
    show_default=True,
    help="Minimum free disk space required before each candidate merge",
)
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
    random_search: Optional[int],
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
    resume: Optional[str],
    num_gpus: Optional[int],
    device: str,
    max_disk_gb_min: float,
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

    if resume is not None:
        resume = os.path.abspath(resume)
        if storage_path is not None and os.path.abspath(storage_path) != resume:
            raise click.ClickException(
                "--storage-path must match --resume when both are supplied."
            )
        storage_path = resume
    if storage_path is None:
        raise click.ClickException("Provide --storage-path or --resume RUN_DIR.")
    if resume is not None and random_search is not None:
        raise click.ClickException("--resume cannot be combined with --random-search.")

    stage_log("Stage-Init", "Validating configuration settings...")
    check_for_naughty_config(config, allow=allow_benchmark_tasks)
    resolved_stop = _resolve_stop_configuration(
        config,
        max_fevals_cli=max_fevals,
        timeout_cli=timeout,
    )
    if random_search is not None:
        resolved_stop["max_fevals"] = int(random_search)
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
    progress = ProgressLogger(storage_path, seed=random_seed)
    progress.write(
        "run_started",
        resumed=resume is not None,
        mode="random_search" if random_search is not None else "ga",
        max_fevals=max_fevals,
        config_path=os.path.abspath(genome_config_path),
    )
    try:
        resume_state = load_ga_state(storage_path) if resume is not None else None
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Unable to load GA checkpoint: {exc}") from exc
    stage_log("Stage-Init", f"Storage path: {storage_path}")
    if resume_state is not None:
        stage_log(
            "Stage-Init",
            "Loaded GA checkpoint at generation "
            f"{resume_state.get('generation', 0)} with "
            f"{resume_state.get('fevals', 0)} completed evaluations.",
        )
    try:
        fitness_definition_path = ensure_fitness_definition(
            storage_path,
            config,
            resume=resume is not None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    stage_log(
        "Stage-Init",
        "Fitness definition: "
        f"{config.fitness.version}/"
        f"{config.fitness.lower_is_better_transform} "
        f"({fitness_definition_path})",
    )
    stage_log("Stage-Init", "Checking source-model parent lineage...")
    lineage_models = list(config.genome.models)
    if config.genome.base_model is not None:
        lineage_models.append(config.genome.base_model)
    lineage_models = _unique_model_refs(lineage_models)
    merge_methods = _configured_merge_methods(config)
    if config.provenance == "off":
        parent_lineage = {
            "schema_version": 1,
            "status": "off",
            "common_lineage": [],
            "warning": None,
            "parents": [],
            "merge_methods": sorted(set(merge_methods)),
            "task_vector_methods": [],
            "risk_level": "none",
        }
    else:
        parent_lineage = annotate_lineage_risk(
            inspect_parent_lineage(lineage_models), merge_methods
        )
    parent_lineage["policy"] = config.provenance
    parent_lineage_path = write_parent_lineage_report(storage_path, parent_lineage)
    progress.write(
        "provenance_checked",
        status=parent_lineage.get("status"),
        risk_level=parent_lineage.get("risk_level"),
        policy=config.provenance,
    )
    if parent_lineage.get("warning"):
        stage_log(
            "Stage-Init",
            f"PARENT LINEAGE WARNING: {parent_lineage['warning']}",
            level=logging.WARNING,
        )
    elif config.provenance == "off":
        stage_log("Stage-Init", "Parent lineage check disabled by configuration.")
    else:
        common_lineage = parent_lineage.get("common_lineage") or []
        lineage_summary = (
            ", ".join(common_lineage) if common_lineage else "single parent"
        )
        stage_log("Stage-Init", f"Parent lineage check passed: {lineage_summary}")
    stage_log("Stage-Init", f"Parent lineage metadata: {parent_lineage_path}")
    if config.provenance == "fail" and lineage_policy_failed(parent_lineage):
        raise click.ClickException(
            "Parent provenance validation failed in strict mode. "
            f"See {parent_lineage_path} for evidence."
        )
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
    try:
        device = _resolve_device(device, num_gpus)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if device == "cpu":
        if merge_cuda:
            stage_log(
                "Stage-Init",
                "CPU execution selected; disabling CUDA merge operations.",
                level=logging.WARNING,
            )
        merge_cuda = False
        if num_gpus is None:
            num_gpus = 0
    else:
        merge_cuda = _resolve_merge_cuda(merge_cuda, num_gpus)
    stage_log("Stage-Init", f"Resolved execution device: {device}")
    repair_config = getattr(config, "repair", None)
    if repair_config is not None and repair_config.enabled:
        if strategy != "serial" or device != "cpu":
            raise click.ClickException(
                "repair.enabled currently requires --strategy serial --device cpu."
            )
        stage_log(
            "Stage-Init",
            "Gated parent-distillation repair enabled for Stage-2 finalists.",
        )
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
    tracking_config = config.model_dump(mode="json")
    tracking_config["parent_lineage"] = parent_lineage
    tracking_config["search_mode"] = (
        "random_search" if random_search is not None else "genetic_algorithm"
    )
    tracking_config["random_search_samples"] = random_search
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
            config=tracking_config,
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
            config=tracking_config,
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
        device=device,
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
        min_free_disk_gb=max_disk_gb_min,
        reuse_scratch_dir=True,
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
            use_ray=strategy != "serial",
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

    genome_type, genome = build_genome(
        config,
        list(resharded_models),
        resharded_base,
        trust_remote_code=trust_remote_code,
    )
    strat_cls = resolve_evaluation_strategy(strategy)

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

    ga_params = resolve_ga_params(
        config,
        population_size=population_size,
        elite_fraction=elite_fraction,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        crossover=crossover,
        tournament_size=tournament_size,
        resolved_stop=resolved_stop,
        baseline_best_score=baseline_best_score,
    )
    yaml_ga = config.ga

    # Log resolved GA params
    tracker.log_metrics(
        {
            "ga/population_size": ga_params.population_size,
            "search/random_search": float(1.0 if random_search is not None else 0.0),
            "search/random_samples": (
                float(random_search) if random_search is not None else None
            ),
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
            "eval/fitness_version": config.fitness.version,
            "eval/lower_is_better_transform": (
                config.fitness.lower_is_better_transform
            ),
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
            "repair/enabled": float(
                1.0
                if getattr(getattr(config, "repair", None), "enabled", False)
                else 0.0
            ),
            "repair/probe_steps": (
                float(config.repair.probe_steps) if config.repair else None
            ),
            "repair/max_steps": (
                float(config.repair.max_steps) if config.repair else None
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
        progress.write(
            "generation_started",
            idx=fevals_completed,
            generation=generation_idx,
            scores={
                "best": (float(current_best) if math.isfinite(current_best) else None)
            },
            fevals_limit=fevals_limit,
            population_size=population_size,
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
        novel_solution_count = 0
        novel_score_values: List[float] = []

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
            novelty = _classify_solution_novelty(
                candidate_method, np.asarray(genotype_candidate), genome
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
            score = result.get("score")
            if novelty["is_novel_solution"]:
                novel_solution_count += 1
                if score is not None and math.isfinite(float(score)):
                    novel_score_values.append(float(score))

            candidate_rows.append(
                {
                    "candidate_index": idx,
                    "genotype_hash": genotype_exact_hash(
                        np.asarray(genotype_candidate)
                    ),
                    "genotype": json.dumps(
                        np.asarray(genotype_candidate, dtype=np.float32)
                        .reshape(-1)
                        .tolist(),
                        separators=(",", ":"),
                    ),
                    "merge_method": candidate_method,
                    "is_novel_solution": novelty["is_novel_solution"],
                    "novelty_class": novelty["novelty_class"],
                    "novelty_reason": novelty["novelty_reason"],
                    "sampled_method": metadata.get("sampled_method"),
                    "role": metadata.get("role") or metadata.get("origin"),
                    "score": score,
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
                    "repair_repaired": (result.get("repair") or {}).get("repaired"),
                    "repair_probe_slope": (result.get("repair") or {}).get(
                        "probe_slope"
                    ),
                    "repair_steps_used": (result.get("repair") or {}).get("steps_used"),
                    "repair_initial_loss": (result.get("repair") or {}).get(
                        "initial_loss"
                    ),
                    "repair_final_loss": (result.get("repair") or {}).get("final_loss"),
                    "repair_pre_score": result.get("repair_pre_score"),
                    "repair_post_score": result.get("repair_post_score"),
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
        novel_solution_fraction = (
            float(novel_solution_count / len(candidate_rows)) if candidate_rows else 0.0
        )
        best_novel_solution_score = (
            float(max(novel_score_values)) if novel_score_values else None
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
        progress.write(
            "generation_completed",
            idx=step,
            generation=generation,
            scores={
                "generation_best": gen_best,
                "generation_mean": gen_mean,
                "generation_std": gen_std,
                "best_so_far": best_so_far,
            },
            secs=float(eval_seconds),
            compute=True,
            evaluations=evaluations,
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
                "population/novel_solution_fraction": novel_solution_fraction,
                "population/best_novel_solution_score": best_novel_solution_score,
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
        progress.write(
            "new_best",
            idx=step,
            scores={"best": float(best_score)},
        )
        save_best_config(best_x)
        log_best(best_x, best_score, step=step)

    try:
        optimizer_kind = resolve_optimizer_kind(config, genome_type)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    use_enhanced = optimizer_kind == "enhanced"
    stage_log(
        "Stage-GA",
        f"Resolved optimizer: {optimizer_kind} (configured={config.optimizer})",
    )
    run_signature = build_run_signature(
        config,
        ga_params,
        random_seed=random_seed,
        strategy=strategy,
        device=device,
        batch_size=batch_size,
        merge_cuda=merge_cuda,
        trust_remote_code=trust_remote_code,
        vllm=vllm,
        tensor_parallel_size=tensor_parallel_size,
        num_gpus=num_gpus,
    )
    if resume_state is not None and not use_enhanced:
        raise click.ClickException(
            "--resume currently requires an enhanced or multi-method GA configuration."
        )
    if (
        resume_state is not None
        and resume_state.get("config_signature")
        and resume_state["config_signature"] != run_signature
    ):
        raise click.ClickException(
            "Checkpoint configuration does not match the requested resumed run."
        )

    if random_search is not None:
        optimizer = RandomSearchOptimizer(
            genome=genome,
            strategy=strat,
            num_samples=int(random_search),
            seed=random_seed,
            population_size=ga_params.population_size,
            persisted_failed_genotypes=persisted_failed_genotypes,
            on_population_evaluated=on_pop,
            on_new_best=on_best,
            on_generation_start=on_generation_start,
        )
    elif use_enhanced:
        enhanced_params = build_enhanced_ga_params(ga_params, config)

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
            checkpoint_path=os.path.join(storage_path, GA_STATE_FILENAME),
            resume_state=resume_state,
            config_signature=run_signature,
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

    search_label = "random search" if random_search is not None else "GA optimization"
    stage_log("Stage-GA", f"Starting {search_label} loop...")
    progress.write(
        "search_started",
        mode="random_search" if random_search is not None else "ga",
        max_fevals=max_fevals,
    )
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
    except InsufficientDiskSpaceError as exc:
        _write_disk_abort(storage_path, exc)
        progress.write(
            "disk_abort",
            scores={
                "best": (
                    float(best_score)
                    if best_score is not None and math.isfinite(best_score)
                    else None
                )
            },
            free_disk_gb=exc.free_gb,
            required_disk_gb=exc.required_gb,
        )
        ray_observer.set_phase("failed")
        stage_log("Stage-GA", str(exc), level=logging.ERROR)
        state_path = os.path.join(storage_path, GA_STATE_FILENAME)
        if os.path.isfile(state_path):
            stage_log(
                "Stage-GA",
                f"Run stopped cleanly; resume state remains at {state_path}",
                level=logging.ERROR,
            )
        else:
            stage_log(
                "Stage-GA",
                "Run stopped cleanly; this optimizer has no resumable state file.",
                level=logging.ERROR,
            )
        _log_run_artifacts(tracker, storage_path)
        tracker.finish()
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        progress.write(
            "interrupted",
            scores={
                "best": (
                    float(best_score)
                    if best_score is not None and math.isfinite(best_score)
                    else None
                )
            },
        )
        ray_observer.set_phase(
            "failed",
            best_score=(
                float(best_score)
                if best_score is not None and math.isfinite(best_score)
                else None
            ),
            fevals_completed=0,
        )
        try:
            _require_ray().shutdown()
        except click.ClickException:
            pass
        raise

    stage_log("Stage-GA", f"{search_label.capitalize()} complete.")
    stage_log("Stage-GA", f"Best score achieved: {best_score:.4f}")
    stop_details = getattr(optimizer, "last_stop_details", None) or {}
    stop_details_path = _write_stop_details(
        storage_path,
        resolved_stop=resolved_stop,
        stop_details=stop_details,
    )
    progress.write(
        "search_completed",
        idx=int(stop_details.get("fevals") or 0),
        generation=int(stop_details.get("generation") or 0),
        scores={
            "best": float(best_score) if math.isfinite(best_score) else None,
        },
        secs=float(stop_details.get("elapsed_seconds") or 0.0),
        stop_reason=stop_details.get("reason"),
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
            try:
                if best_config is not None:
                    ensure_free_disk(storage_path, merge_options.min_free_disk_gb)
                    shutil.rmtree(final_model_path, ignore_errors=True)
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
                                "error_message",
                                "Failed to materialize layered final model",
                            )
                        )
                    if os.path.exists(final_model_path):
                        shutil.rmtree(final_model_path, ignore_errors=True)
                    shutil.copytree(merged_path, final_model_path)
                    shutil.rmtree(merged_path, ignore_errors=True)
            except InsufficientDiskSpaceError as exc:
                _write_disk_abort(storage_path, exc)
                ray_observer.set_phase("failed")
                stage_log("Stage-GA", str(exc), level=logging.ERROR)
                _log_run_artifacts(tracker, storage_path)
                tracker.finish()
                raise click.ClickException(str(exc)) from exc

            if repair_config is not None and repair_config.enabled:
                stage_log("Stage-GA", "Applying gated repair to exported winner...")
                final_repair = strat.repair_checkpoint(
                    final_model_path,
                    best_x,
                    config,
                )
                atomic_write_json(
                    os.path.join(storage_path, "final_repair.json"),
                    final_repair,
                )
                stage_log(
                    "Stage-GA",
                    "Export repair outcome: "
                    f"repaired={final_repair.get('repaired')} "
                    f"probe_slope={final_repair.get('probe_slope')}",
                )

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
        progress.write(
            "run_finished",
            idx=int(stop_details.get("fevals") or max_fevals),
            generation=int(stop_details.get("generation") or 0),
            scores={"best": float(best_score)},
            status="success",
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
        progress.write(
            "run_finished",
            idx=int(stop_details.get("fevals") or 0),
            generation=int(stop_details.get("generation") or 0),
            scores={"best": None},
            status="failed",
        )

    _log_run_artifacts(tracker, storage_path)
    tracker.finish()


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    import transformers

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
