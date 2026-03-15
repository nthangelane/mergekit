#!/usr/bin/env python3
"""Summarize thesis experiment status from manifests, local artifacts, and RayJobs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUITE_MANIFEST_PATH = ROOT / "experiments" / "thesis" / "manifest.yaml"


@dataclass
class ExperimentLocalStatus:
    state: str
    generations: int
    best_score: Optional[float]
    updated_at: Optional[str]
    artifacts: list[str]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_suite(target: str) -> tuple[dict, list[dict]]:
    suite = _load_yaml(SUITE_MANIFEST_PATH)
    manifests = []
    for rel_path in suite.get("manifests", []):
        path = (ROOT / rel_path).resolve()
        manifest = _load_yaml(path)
        manifest["_path"] = str(path.relative_to(ROOT))
        if target != "all" and manifest.get("group") != target:
            continue
        manifests.append(manifest)
    return suite, manifests


def _run_json(cmd: list[str]) -> Optional[dict]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _fmt_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _read_ga_history(path: Path) -> tuple[int, Optional[float]]:
    if not path.exists():
        return 0, None

    generations = 0
    best_score = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                generations = max(generations, int(row.get("generation", 0)))
            except ValueError:
                pass

            for key in ("best_so_far", "gen_best"):
                raw = row.get(key)
                if raw in (None, "", "None", "nan", "NaN", "-inf", "inf"):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                best_score = value if best_score is None else max(best_score, value)
                break

    return generations, best_score


def _local_status(storage_path: Path) -> ExperimentLocalStatus:
    if not storage_path.exists():
        return ExperimentLocalStatus(
            state="PLANNED",
            generations=0,
            best_score=None,
            updated_at=None,
            artifacts=[],
        )

    history = storage_path / "ga_history.csv"
    baseline = storage_path / "baseline_results.csv"
    best_config = storage_path / "best_config.yaml"
    final_model = storage_path / "final_model"
    summary = storage_path / "ga_summary.txt"

    generations, best_score = _read_ga_history(history)
    artifact_paths = [
        p for p in (history, baseline, best_config, summary) if p.exists()
    ]
    if final_model.exists():
        artifact_paths.append(final_model)

    artifacts = []
    if history.exists():
        artifacts.append("history")
    if baseline.exists():
        artifacts.append("baseline")
    if best_config.exists():
        artifacts.append("best_config")
    if final_model.exists():
        artifacts.append("final_model")

    latest_mtime = max((p.stat().st_mtime for p in artifact_paths), default=None)
    updated_at = _fmt_timestamp(latest_mtime) if latest_mtime else None

    if final_model.exists():
        state = "COMPLETE"
    elif history.exists():
        state = "STARTED"
    else:
        state = "SETUP"

    return ExperimentLocalStatus(
        state=state,
        generations=generations,
        best_score=best_score,
        updated_at=updated_at,
        artifacts=artifacts,
    )


def _cluster_snapshot(
    namespace: Optional[str], ray_cluster_name: Optional[str]
) -> dict:
    if not namespace or not ray_cluster_name:
        return {
            "rayjobs": {},
            "raycluster": {},
            "workload_counts": {},
            "instance_types": {},
        }

    rayjobs = _run_json(["kubectl", "get", "rayjobs", "-n", namespace, "-o", "json"])
    raycluster = _run_json(
        [
            "kubectl",
            "get",
            "raycluster",
            ray_cluster_name,
            "-n",
            namespace,
            "-o",
            "json",
        ]
    )
    nodes = _run_json(["kubectl", "get", "nodes", "-o", "json"])

    rayjob_map: Dict[str, dict] = {}
    if rayjobs:
        for item in rayjobs.get("items", []):
            rayjob_map[item["metadata"]["name"]] = item.get("status", {})

    workload_counts: Dict[str, int] = {}
    instance_types: Dict[str, int] = {}
    if nodes:
        for item in nodes.get("items", []):
            labels = item.get("metadata", {}).get("labels", {})
            workload = labels.get("workload", "unknown")
            instance_type = labels.get("node.kubernetes.io/instance-type", "unknown")
            workload_counts[workload] = workload_counts.get(workload, 0) + 1
            instance_types[instance_type] = instance_types.get(instance_type, 0) + 1

    cluster_status = raycluster.get("status", {}) if raycluster else {}
    return {
        "rayjobs": rayjob_map,
        "raycluster": cluster_status,
        "workload_counts": workload_counts,
        "instance_types": instance_types,
    }


def _cluster_total_gpus(snapshot: dict) -> int:
    raw = snapshot.get("raycluster", {}).get("desiredGPU", 0)
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return 0


def _cluster_ready_workers(snapshot: dict) -> Optional[int]:
    raw = snapshot.get("raycluster", {}).get("readyWorkerReplicas")
    if raw in (None, ""):
        return None
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _fit_status(exp: dict, manifest: dict, snapshot: dict) -> tuple[str, Optional[str]]:
    if manifest.get("execution_target") == "local_mac":
        return "ready_local", None

    if exp.get("requires_gated_hf_access"):
        return "gated", "requires gated Hugging Face access"

    required_gpus = int(exp.get("recommended_num_gpus", 0) or 0)
    available_gpus = _cluster_total_gpus(snapshot)
    has_cluster_data = bool(snapshot.get("raycluster") or snapshot.get("rayjobs"))

    if not has_cluster_data:
        return "cluster_unknown", "kubectl unavailable or cluster unreachable"
    if available_gpus >= required_gpus:
        return "ready_now", None
    if available_gpus > 0:
        return (
            "scale_required",
            f"needs {required_gpus} GPU(s), cluster advertises {available_gpus}",
        )
    return (
        "not_provisioned",
        f"needs {required_gpus} GPU(s), cluster advertises 0",
    )


def _row(
    job_status: Optional[dict],
    local: ExperimentLocalStatus,
    exp: dict,
    manifest: dict,
    fit: str,
) -> dict:
    ray_state = "-"
    if job_status:
        ray_state = (
            job_status.get("jobStatus") or job_status.get("jobDeploymentStatus") or "-"
        )

    return {
        "id": exp["id"],
        "target": manifest["group"],
        "fit": fit,
        "rayjob": ray_state,
        "local": local.state,
        "gpus": str(exp.get("recommended_num_gpus", "-")),
        "tp": str(exp.get("recommended_tensor_parallel_size", "-")),
        "gens": str(local.generations or "-"),
        "best": _fmt_float(local.best_score),
        "updated": local.updated_at or "-",
        "storage": exp["storage_path"],
    }


def _print_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    headers = [
        "id",
        "target",
        "fit",
        "rayjob",
        "local",
        "gpus",
        "tp",
        "gens",
        "best",
        "updated",
        "storage",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def _group_summary(
    manifest: dict, snapshot: dict, rows: list[dict], notes: list[str]
) -> dict:
    fit_counts = Counter(row["fit"] for row in rows)
    profile = manifest.get("target_cluster_profile", {})
    worker_target_min = profile.get("target_total_workers_min")
    worker_target_max = profile.get("target_total_workers_max")
    ready_workers = _cluster_ready_workers(snapshot)
    worker_target_status = None
    if (
        ready_workers is not None
        and worker_target_min is not None
        and worker_target_max is not None
    ):
        if ready_workers < worker_target_min:
            worker_target_status = (
                f"below_target ({ready_workers} < {worker_target_min})"
            )
        elif ready_workers > worker_target_max:
            worker_target_status = (
                f"above_target ({ready_workers} > {worker_target_max})"
            )
        else:
            worker_target_status = f"within_target ({ready_workers})"
    return {
        "fit_counts": dict(fit_counts),
        "notes": notes,
        "cluster_total_gpus": _cluster_total_gpus(snapshot),
        "ready_workers": ready_workers if ready_workers is not None else "-",
        "worker_target_min": worker_target_min,
        "worker_target_max": worker_target_max,
        "worker_target_status": worker_target_status,
    }


def _collect_status(target: str) -> dict:
    suite, manifests = _load_suite(target)
    snapshot_cache: dict[tuple[str, str], dict] = {}
    groups = []

    for manifest in manifests:
        snapshot = {
            "rayjobs": {},
            "raycluster": {},
            "workload_counts": {},
            "instance_types": {},
        }
        if manifest.get("execution_target") == "eks_gpu":
            cache_key = (
                manifest.get("namespace", ""),
                manifest.get("ray_cluster_name", ""),
            )
            if cache_key not in snapshot_cache:
                snapshot_cache[cache_key] = _cluster_snapshot(*cache_key)
            snapshot = snapshot_cache[cache_key]

        rows = []
        notes = []
        for exp in manifest["experiments"]:
            local = _local_status(ROOT / exp["storage_path"])
            fit, reason = _fit_status(exp, manifest, snapshot)
            job_status = snapshot["rayjobs"].get(exp.get("job_name", ""))
            rows.append(_row(job_status, local, exp, manifest, fit))
            if reason:
                notes.append(f"{exp['id']}: {reason}")

        groups.append(
            {
                "manifest": manifest,
                "cluster": snapshot,
                "summary": _group_summary(manifest, snapshot, rows, notes),
                "rows": rows,
            }
        )

    return {
        "suite": suite,
        "groups": groups,
    }


def _print_status(payload: dict) -> None:
    print(f"Suite: {payload['suite']['suite']}")
    print(f"Suite manifest: {SUITE_MANIFEST_PATH.relative_to(ROOT)}")

    for group in payload["groups"]:
        manifest = group["manifest"]
        cluster = group["cluster"]
        summary = group["summary"]

        print()
        print(f"Group: {manifest['group']} ({manifest['execution_target']})")
        print(f"Manifest: {manifest['_path']}")
        print(f"Workspace: {manifest.get('workspace_root', '-')}")

        if manifest.get("execution_target") == "eks_gpu":
            profile = manifest.get("target_cluster_profile", {})
            if profile:
                print(
                    "Target cluster: "
                    f"{profile.get('name', '-')} | "
                    f"workers={profile.get('target_total_workers_min', '-')}..{profile.get('target_total_workers_max', '-')} | "
                    f"per_run_gpus={profile.get('per_experiment_gpu_cap', '-')} | "
                    f"min_vram={profile.get('minimum_gpu_vram_gb', '-')}GiB"
                )
                node_types = profile.get("preferred_gpu_node_types") or []
                if node_types:
                    print("Preferred nodes: " + ", ".join(node_types))
            print(
                "Current cluster: "
                f"namespace={manifest.get('namespace', '-')} "
                f"ray_cluster={manifest.get('ray_cluster_name', '-')} "
                f"workers={summary['ready_workers']} "
                f"desired_gpus={summary['cluster_total_gpus']}"
            )
            if cluster["instance_types"]:
                instance_summary = ", ".join(
                    f"{k} x{v}" for k, v in sorted(cluster["instance_types"].items())
                )
                print(f"Nodes: {instance_summary}")
            if summary["worker_target_status"]:
                print("Worker scale: " f"{summary['worker_target_status']}")

        if summary["fit_counts"]:
            fit_summary = ", ".join(
                f"{status}={count}"
                for status, count in sorted(summary["fit_counts"].items())
            )
            print(f"Fit states: {fit_summary}")
        if summary["notes"]:
            print("Notes:")
            for note in summary["notes"]:
                print(f"  - {note}")

        print()
        _print_table(group["rows"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument(
        "--target",
        choices=["all", "local_mac", "eks_gpu"],
        default="all",
    )
    args = parser.parse_args()

    while True:
        payload = _collect_status(args.target)
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))
            _print_status(payload)

        if not args.watch:
            return 0
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    sys.exit(main())
