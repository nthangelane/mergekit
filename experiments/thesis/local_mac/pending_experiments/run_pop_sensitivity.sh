#!/bin/bash
# Population size sensitivity analysis runner
# Executes all four population size variants (4, 8, 16, 32) with seed=11 and same feval budget
# Seeds random number generators for reproducibility across runs
# Outputs timing information for each configuration
#
# Expected generation counts at N=96 fevals:
#   pop=4  -> ~24 generations  (many iterations, low diversity)
#   pop=8  -> ~12 generations  (thesis baseline)
#   pop=16 ->  ~6 generations  (higher diversity, fewer passes)
#   pop=32 ->  ~3 generations  (extreme diversity, minimal generational pressure)

set -e

# Define output directories
OUTPUT_BASE="workspace/thesis/local_mac/results/pop_sensitivity"
mkdir -p "$OUTPUT_BASE/pop4" "$OUTPUT_BASE/pop8" "$OUTPUT_BASE/pop16" "$OUTPUT_BASE/pop32"

echo "=========================================="
echo "Population Size Sensitivity Analysis"
echo "=========================================="
echo "Seed: 11"
echo "Max evals: 96 (all configs)"
echo "Output base: $OUTPUT_BASE"
echo ""

# Population size 4
echo "========== Running Pop=4 ==========="
start_time=$(date +%s)
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop4_sensitivity.yml \
  --seed 11 \
  --output "$OUTPUT_BASE/pop4" \
  --label "pop4_sensitivity"
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Pop=4 completed in ${duration}s (~$(( duration / 60 ))m)"
echo ""

# Population size 8 (baseline)
echo "========== Running Pop=8 (baseline) ==========="
start_time=$(date +%s)
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop8_sensitivity.yml \
  --seed 11 \
  --output "$OUTPUT_BASE/pop8" \
  --label "pop8_baseline"
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Pop=8 completed in ${duration}s (~$(( duration / 60 ))m)"
echo ""

# Population size 16
echo "========== Running Pop=16 ==========="
start_time=$(date +%s)
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop16_sensitivity.yml \
  --seed 11 \
  --output "$OUTPUT_BASE/pop16" \
  --label "pop16_sensitivity"
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Pop=16 completed in ${duration}s (~$(( duration / 60 ))m)"
echo ""

# Population size 32 (extreme upper bound)
echo "========== Running Pop=32 (upper bound) ==========="
echo "NOTE: Only ~3 complete generations expected at N=96. This tests diversity-vs-pressure tradeoff."
start_time=$(date +%s)
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop32_sensitivity.yml \
  --seed 11 \
  --output "$OUTPUT_BASE/pop32" \
  --label "pop32_sensitivity"
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Pop=32 completed in ${duration}s (~$(( duration / 60 ))m)"
echo ""

echo "=========================================="
echo "All runs complete. Results in: $OUTPUT_BASE"
echo "=========================================="
echo ""
echo "Key metric to compare: best_fitness_so_far at feval=96"
echo "Population-diversity tradeoff curve: pop4 -> pop8 -> pop16 -> pop32"
