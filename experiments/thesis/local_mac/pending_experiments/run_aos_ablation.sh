#!/bin/bash

# AOS Credit Weight Ablation Runner
# Runs three AOS configuration variants with seed=11
# Outputs results to workspace/thesis/local_mac/results/aos_ablation/{variant}/

set -e

BASE_DIR="/sessions/quirky-cool-bohr/mnt/local_mac"
CONFIG_DIR="${BASE_DIR}"
OUTPUT_BASE="workspace/thesis/local_mac/results/aos_ablation"
SEED=11

# Create output directories
mkdir -p "${OUTPUT_BASE}/equal_weights"
mkdir -p "${OUTPUT_BASE}/fitness_only"
mkdir -p "${OUTPUT_BASE}/current"

echo "=========================================="
echo "AOS Credit Weight Ablation Experiment"
echo "Seed: ${SEED}"
echo "Total variants: 3"
echo "Expected runtime: ~4.5 hours"
echo "=========================================="
echo ""

# Variant A: Equal weights
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Variant A: Equal weights (0.333/0.333/0.333)"
echo "Config: ${CONFIG_DIR}/thesis_aos_equal_weights.yml"
time python -m thesis_run \
  --config "${CONFIG_DIR}/thesis_aos_equal_weights.yml" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_BASE}/equal_weights"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Variant A complete"
echo ""

# Variant B: Fitness only
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Variant B: Fitness-only (1.0/0.0/0.0)"
echo "Config: ${CONFIG_DIR}/thesis_aos_fitness_only.yml"
time python -m thesis_run \
  --config "${CONFIG_DIR}/thesis_aos_fitness_only.yml" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_BASE}/fitness_only"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Variant B complete"
echo ""

# Variant C: Current (baseline)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Variant C: Current thesis weights (0.50/0.30/0.20) — baseline"
echo "Config: ${CONFIG_DIR}/thesis_aos_current.yml"
time python -m thesis_run \
  --config "${CONFIG_DIR}/thesis_aos_current.yml" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_BASE}/current"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Variant C complete"
echo ""

echo "=========================================="
echo "All variants complete!"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "=========================================="
