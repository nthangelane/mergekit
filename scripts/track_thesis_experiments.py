#!/usr/bin/env python3
"""Summarize thesis experiment status from manifest, local artifacts, and RayJobs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "thesis" / "manifest.yaml"


@dataclass
class ExperimentLocalStatus:
    state: str
    generations: int
    best_score: Optional[float]
    updated_at: Optional[str]
    artifacts: list[str]


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


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


def _cluster_snapshot(namespace: str, ray_cluster_name: str) -> dict:
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


def _cluster_summary(snapshot: dict, experiments: list[dict]) -> dict:
    raycluster = snapshot.get("raycluster", {})
    gpu_workers = int(snapshot.get("workload_counts", {}).get("gpu", 0))
    desired_gpu = int(str(raycluster.get("desiredGPU", "0")))
    gpu_experiments = sum(
        1 for exp in experiments if int(exp["recommended_num_gpus"]) > 0
    )
    blocked = [
        exp["id"] for exp in experiments if exp["current_readiness"] == "blocked"
    ]

    can_run_all = not blocked and gpu_workers >= gpu_experiments
    reasons = []
    if blocked:
        reasons.append(f"blocked experiments: {', '.join(blocked)}")
    if gpu_workers < gpu_experiments:
        reasons.append(
            f"only {gpu_workers} GPU node(s) for {gpu_experiments} GPU-backed experiments"
        )
    if desired_gpu < gpu_experiments:
        reasons.append(
            f"Ray cluster advertises {desired_gpu} total GPU(s), below the required {gpu_experiments}"
        )

    return {
        "can_run_all_now": can_run_all,
        "reasons": reasons,
    }


def _row(job_status: Optional[dict], local: ExperimentLocalStatus, exp: dict) -> dict:
    ray_state = "-"
    if job_status:
        ray_state = (
            job_status.get("jobStatus") or job_status.get("jobDeploymentStatus") or "-"
        )

    return {
        "id": exp["id"],
        "name": exp["name"],
        "readiness": exp["current_readiness"],
        "rayjob": ray_state,
        "local": local.state,
        "gens": str(local.generations or "-"),
        "best": _fmt_float(local.best_score),
        "updated": local.updated_at or "-",
        "storage": exp["storage_path"],
    }


def _print_table(rows: Iterable[dict]) -> None:
    rows = list(rows)
    headers = [
        "id",
        "readiness",
        "rayjob",
        "local",
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


def _collect_status() -> dict:
    manifest = _load_manifest()
    experiments = manifest["experiments"]
    snapshot = _cluster_snapshot(manifest["namespace"], manifest["ray_cluster_name"])
    summary = _cluster_summary(snapshot, experiments)

    rows = []
    for exp in experiments:
        local = _local_status(ROOT / exp["storage_path"])
        job_status = snapshot["rayjobs"].get(exp["job_name"])
        rows.append(_row(job_status, local, exp))

    return {
        "manifest": manifest,
        "cluster": snapshot,
        "summary": summary,
        "rows": rows,
    }


def _print_status(payload: dict) -> None:
    manifest = payload["manifest"]
    cluster = payload["cluster"]
    summary = payload["summary"]

    print(f"Suite: {manifest['suite']}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print(
        "Cluster: "
        f"namespace={manifest['namespace']} "
        f"ray_cluster={manifest['ray_cluster_name']} "
        f"workers={cluster['raycluster'].get('readyWorkerReplicas', '-')}"
    )
    if cluster["instance_types"]:
        instance_summary = ", ".join(
            f"{k} x{v}" for k, v in sorted(cluster["instance_types"].items())
        )
        print(f"Nodes: {instance_summary}")
    print(
        "All 5 runnable at once now: " + ("YES" if summary["can_run_all_now"] else "NO")
    )
    if summary["reasons"]:
        print("Why not:")
        for reason in summary["reasons"]:
            print(f"  - {reason}")
    print()
    _print_table(payload["rows"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()

    while True:
        payload = _collect_status()
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
