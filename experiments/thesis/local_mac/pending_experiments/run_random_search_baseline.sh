#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/experiments/thesis/local_mac/pending_experiments/thesis_random_search_baseline.yml"
BASE_OUTPUT_DIR="$ROOT_DIR/workspace/thesis/local_mac/results/random_search_baseline"
SHARED_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"
SEEDS="${SEEDS:-11 22 33}"
SAMPLES="${SAMPLES:-96}"
STRATEGY="${STRATEGY:-pool}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$BASE_OUTPUT_DIR" "$SHARED_CACHE"

echo "Native random-search baseline"
echo "Config: $CONFIG_PATH"
echo "Seeds: $SEEDS"
echo "Samples per seed: $SAMPLES"
echo "Strategy: $STRATEGY"

for seed in $SEEDS; do
  RUN_DIR="$BASE_OUTPUT_DIR/seed${seed}"
  if [[ -f "$RUN_DIR/ga_stop_details.json" ]]; then
    echo "Skipping seed${seed}; completion artifact already exists."
    continue
  fi

  mkdir -p "$RUN_DIR"
  rm -rf "$RUN_DIR/transformers_cache"
  ln -s "$SHARED_CACHE" "$RUN_DIR/transformers_cache"

  EXTRA_ARGS=()
  if [[ "$STRATEGY" != "serial" ]]; then
    EXTRA_ARGS+=(--num-workers "$NUM_WORKERS")
  fi

  python -m mergekit.scripts.evolve_ga \
    "$CONFIG_PATH" \
    --random-search "$SAMPLES" \
    --strategy "$STRATEGY" \
    "${EXTRA_ARGS[@]}" \
    --storage-path "$RUN_DIR" \
    --device cpu \
    --num-gpus 0 \
    --no-merge-cuda \
    --batch-size 1 \
    --baseline \
    --no-reshard \
    --random-seed "$seed" \
    2>&1 | tee "$RUN_DIR/run.log"
done

echo "Random-search runs completed: $BASE_OUTPUT_DIR"
