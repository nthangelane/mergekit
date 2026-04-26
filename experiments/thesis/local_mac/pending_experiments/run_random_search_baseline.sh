#!/bin/bash

# Random search baseline experiment runner for thesis GA comparison
# Executes 3 independent seeds (11, 22, 33) sequentially
# Each run: max 96 fevals, pure random candidate generation (no GA operators)
# Output: workspace/thesis/local_mac/results/random_search_baseline/seed{11,22,33}

set -e  # Exit on error

CONFIG="./thesis_random_search_baseline.yml"
BASE_OUTPUT_DIR="./workspace/thesis/local_mac/results/random_search_baseline"

# Seeds to run
SEEDS=(11 22 33)

# GA hyperparameters (same pool strategy as completed GA batch)
POOL_STRATEGY="multiprocessing"
WORKERS=4

echo "=========================================="
echo "Random Search Baseline Experiment"
echo "=========================================="
echo "Config: $CONFIG"
echo "Seeds: ${SEEDS[@]}"
echo "Pool strategy: $POOL_STRATEGY"
echo "Workers per seed: $WORKERS"
echo "Max fevals per seed: 96"
echo ""

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/seed$SEED"

    echo "=========================================="
    echo "Running seed $SEED"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="

    START_TIME=$(date +%s)

    # Run the search
    python -m thesis.run_search \
        --config "$CONFIG" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --pool-strategy "$POOL_STRATEGY" \
        --num-workers "$WORKERS"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    echo ""
    echo "Seed $SEED completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo ""
done

echo "=========================================="
echo "All seeds completed successfully"
echo "Results in: $BASE_OUTPUT_DIR"
echo "=========================================="
