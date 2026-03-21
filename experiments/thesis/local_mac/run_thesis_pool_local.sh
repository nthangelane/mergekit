#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH_OVERRIDE:-$ROOT_DIR/experiments/thesis/local_mac/thesis_run_pythia70m.yml}"
BATCH_LABEL="${1:-$(date +%Y%m%d)-thesis-pool-batch}"
BATCH_DIR="$ROOT_DIR/workspace/thesis/local_mac/results/$BATCH_LABEL"
TRACKING_URI="file://$BATCH_DIR/mlruns"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT:-mergekit-thesis-pool-$BATCH_LABEL}"
UI_PORT="${MLFLOW_UI_PORT:-5013}"
RAY_TMP_ROOT="${RAY_TMPDIR:-/tmp/ray-thesis-pool}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SHARED_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"
SEEDS="${SEEDS:-11 22 33}"

mkdir -p "$BATCH_DIR"
mkdir -p "$RAY_TMP_ROOT"
mkdir -p "$SHARED_CACHE"

echo "Config: $CONFIG_PATH"
echo "Batch dir: $BATCH_DIR"
echo "MLflow: http://127.0.0.1:$UI_PORT"
echo "Seeds: $SEEDS"

for seed in $SEEDS; do
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
done
