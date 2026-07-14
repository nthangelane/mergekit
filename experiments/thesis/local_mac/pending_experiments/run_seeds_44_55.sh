#!/bin/bash
set -euo pipefail

# 5-Seed Validation Campaign: Seeds 44 and 55
# This script runs seeds 44 and 55 (the final two of the 5-seed extended validation)
# using the tuned configuration (thesis_run_pythia70m_tuned.yml) with pool strategy.
# Prior completed seeds: 11, 22, 33

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BATCH_LABEL="${1:-$(date +%Y%m%d)-thesis-pool-batch}"
CLI_NUM_WORKERS="${2:-}"
BATCH_DIR="$ROOT_DIR/workspace/thesis/local_mac/results/$BATCH_LABEL"
TRACKING_URI="file://$BATCH_DIR/mlruns"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT:-mergekit-thesis-pool-$BATCH_LABEL}"
UI_PORT="${MLFLOW_UI_PORT:-5013}"
RAY_TMP_ROOT="${RAY_TMPDIR:-/tmp/ray-thesis-pool}"
NUM_WORKERS="${CLI_NUM_WORKERS:-${NUM_WORKERS:-2}}"
SHARED_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"

# Use tuned config for extended validation.
CONFIG_PATH="${CONFIG_PATH_OVERRIDE:-$ROOT_DIR/experiments/thesis/local_mac/thesis_run_pythia70m_tuned.yml}"

mkdir -p "$BATCH_DIR"
mkdir -p "$RAY_TMP_ROOT"
mkdir -p "$SHARED_CACHE"

echo "Config: $CONFIG_PATH"
echo "Batch dir: $BATCH_DIR"
echo "MLflow: http://127.0.0.1:$UI_PORT"
echo "Seeds: 44 55"
echo "Workers: $NUM_WORKERS"
echo "Strategy: pool"

for seed in 44 55; do
  RUN_DIR="$BATCH_DIR/seed${seed}"
  if [[ -f "$RUN_DIR/ga_stop_details.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] skipping seed${seed}; already completed" | tee -a "$BATCH_DIR/batch.log"
    continue
  fi

  mkdir -p "$RUN_DIR"
  rm -rf "$RUN_DIR/transformers_cache"
  ln -s "$SHARED_CACHE" "$RUN_DIR/transformers_cache"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting seed${seed}" | tee -a "$BATCH_DIR/batch.log"
  env RAY_TMPDIR="$RAY_TMP_ROOT" \
    nice -n 10 \
    python -m mergekit.scripts.evolve_ga \
      "$CONFIG_PATH" \
      --strategy pool \
      --num-workers "$NUM_WORKERS" \
      --storage-path "$RUN_DIR" \
      --num-gpus 0 \
      --no-merge-cuda \
      --batch-size 1 \
      --baseline \
      --mlflow \
      --mlflow-experiment "$EXPERIMENT_NAME" \
      --mlflow-tracking-uri "$TRACKING_URI" \
      --mlflow-ui-port "$UI_PORT" \
      --no-reshard \
      --random-seed "$seed" \
      2>&1 | tee "$RUN_DIR/run.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished seed${seed}" | tee -a "$BATCH_DIR/batch.log"

  # Brief pause between seed runs
  if [[ "$seed" != "55" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sleeping 10 seconds before next seed" | tee -a "$BATCH_DIR/batch.log"
    sleep 10
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 5-seed validation campaign (seeds 44 and 55) complete" | tee -a "$BATCH_DIR/batch.log"
