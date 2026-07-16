#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/experiments/thesis/local_mac/preflight_evo_cpu.yml"
COMMIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
RUN_LABEL="${1:-${COMMIT_SHA}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$ROOT_DIR/workspace/thesis/preflight/$RUN_LABEL"
SHARED_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"
REQUIRE_PUSHED="${REQUIRE_PUSHED:-1}"
SKIP_TESTS="${SKIP_TESTS:-0}"

if [[ "$REQUIRE_PUSHED" == "1" ]]; then
  git -C "$ROOT_DIR" diff --quiet
  git -C "$ROOT_DIR" diff --cached --quiet
  UPSTREAM="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref '@{upstream}')"
  LOCAL_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
  UPSTREAM_SHA="$(git -C "$ROOT_DIR" rev-parse "$UPSTREAM")"
  if [[ "$LOCAL_SHA" != "$UPSTREAM_SHA" ]]; then
    echo "Preflight requires the local commit to match $UPSTREAM." >&2
    exit 1
  fi
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing preflight directory: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$SHARED_CACHE"
ln -s "$SHARED_CACHE" "$RUN_DIR/transformers_cache"

if [[ "$SKIP_TESTS" != "1" ]]; then
  pytest -q "$ROOT_DIR/tests"
fi

python -m mergekit.scripts.run_campaign \
  exp26_aos_on exp26_aos_off exp27_native \
  exp28_adaptive exp28_random exp29_probe exp29_full \
  exp210_lora exp210_full_ft \
  --dry-run \
  --output-root "$RUN_DIR/campaign-dry-run"

python -m mergekit.scripts.evolve_ga \
  "$CONFIG_PATH" \
  --random-search 2 \
  --population-size 2 \
  --strategy serial \
  --storage-path "$RUN_DIR" \
  --device cpu \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --save-final-model \
  --reshard \
  --max-disk-gb-min "${MAX_DISK_GB_MIN:-5}" \
  --random-seed 11 \
  2>&1 | tee "$RUN_DIR/run.log"

python - "$RUN_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
required = [
    "parent_lineage.json",
    "fitness_definition.json",
    "ga_candidate_history.csv",
    "ga_history.csv",
    "ga_stop_details.json",
    "final_repair.json",
    "progress.log",
]
missing = [name for name in required if not (run_dir / name).is_file()]
if missing:
    raise SystemExit(f"Missing required preflight artifacts: {missing}")
if not (run_dir / "final_model").is_dir():
    raise SystemExit("Missing final_model directory")
if (run_dir / "run_abort.json").exists():
    raise SystemExit("Preflight produced run_abort.json")

input_models = run_dir / "input_models"
resharded_models = [path for path in input_models.iterdir() if path.is_dir()]
if len(resharded_models) < 2:
    raise SystemExit(f"Expected at least two resharded models, found {len(resharded_models)}")
for model_dir in resharded_models:
    if not list(model_dir.glob("*.safetensors")):
        raise SystemExit(f"Resharded model has no safetensors checkpoint: {model_dir}")

with (run_dir / "ga_candidate_history.csv").open(newline="") as handle:
    candidates = list(csv.DictReader(handle))
if len(candidates) != 2:
    raise SystemExit(f"Expected 2 candidate rows, found {len(candidates)}")
if any(not row.get("genotype_hash") or not row.get("genotype") for row in candidates):
    raise SystemExit("Candidate history is missing genotype provenance")

stop = json.loads((run_dir / "ga_stop_details.json").read_text())
final_stop = stop.get("final_stop", stop)
if final_stop.get("reason") != "random_search_complete" or final_stop.get("fevals") != 2:
    raise SystemExit(f"Unexpected stop details: {final_stop}")

lineage = json.loads((run_dir / "parent_lineage.json").read_text())
if lineage.get("status") not in {"common_lineage", "no_common_lineage"}:
    raise SystemExit(f"Lineage was not resolved: {lineage.get('status')}")

fitness = json.loads((run_dir / "fitness_definition.json").read_text())
if fitness.get("fitness_version") != "v2":
    raise SystemExit(f"Unexpected fitness version: {fitness}")
if fitness.get("lower_is_better_transform") != "log_reciprocal":
    raise SystemExit(f"Unexpected lower-is-better transform: {fitness}")

progress_events = [
    json.loads(line)
    for line in (run_dir / "progress.log").read_text().splitlines()
    if line.strip()
]
required_progress_keys = {"ts", "msg", "seed", "disk_free_gb"}
if any(not required_progress_keys.issubset(event) for event in progress_events):
    raise SystemExit("Progress log contains an event with missing required fields")
messages = [event.get("msg") for event in progress_events]
for required_message in ("run_started", "generation_completed", "run_finished"):
    if required_message not in messages:
        raise SystemExit(f"Progress log is missing {required_message!r}")
if progress_events[-1].get("msg") != "run_finished":
    raise SystemExit(f"Unexpected final progress event: {progress_events[-1]}")
if progress_events[-1].get("status") != "success":
    raise SystemExit(f"Preflight progress did not finish successfully: {progress_events[-1]}")

print(f"Evo preflight passed: {run_dir}")
PY
