#!/bin/bash
# Post-hoc evaluation: 4 models (2 baselines + 3 merged seeds) on 8 benchmarks
# Purpose: Validate whether 3-benchmark GA optimisation generalises to held-out tasks

set -e

TASKS="wikitext,lambada_openai,boolq,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2"
BASE_RESULTS_DIR="workspace/thesis/local_mac/results/posthoc_eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Models to evaluate: 2 baselines + 3 GA-optimised merged models
declare -A MODELS=(
    ["pythia-70m-deduped"]="EleutherAI/pythia-70m-deduped"
    ["pythia-70m-helpful-sft"]="lomahony/pythia-70m-helpful-sft"
    ["seed11-final"]="workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/final_model"
    ["seed22-final"]="workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/final_model"
    ["seed33-final"]="workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/final_model"
)

echo "=========================================="
echo "Post-hoc Evaluation: 8-Benchmark Suite"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Base results directory: $BASE_RESULTS_DIR"
echo "Tasks: $TASKS"
echo "Number of models: ${#MODELS[@]}"
echo ""

# Create base results directory
mkdir -p "$BASE_RESULTS_DIR"

# Evaluate each model
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    output_dir="${BASE_RESULTS_DIR}/${model_name}"

    echo "=========================================="
    echo "Evaluating: $model_name"
    echo "Model path: $model_path"
    echo "Output directory: $output_dir"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    mkdir -p "$output_dir"

    # Run lm_eval with specified configuration
    python -m lm_eval \
        --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True" \
        --tasks "$TASKS" \
        --num_fewshot 0 \
        --device cpu \
        --batch_size 1 \
        --output_path "$output_dir" \
        --log_samples

    echo "Completed: $model_name at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=========================================="
echo "All evaluations completed successfully!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results location: $BASE_RESULTS_DIR"
echo "=========================================="
