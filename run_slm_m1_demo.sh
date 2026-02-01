#!/bin/bash
# run_slm_m1_demo.sh - Launch the TinyLlama SLM GA demo on CPU-only Apple silicon.
# Usage: ./run_slm_m1_demo.sh [extra mergekit args]

set -euo pipefail

CONFIG_PATH=${SLM_CONFIG:-examples/slm_m1_demo.yml}
STORAGE_PATH=${SLM_STORAGE:-workspace/slm_m1_demo_run}
LOG_PATH=${SLM_LOG:-workspace/slm_m1_demo.log}
MAX_FEVALS=${SLM_MAX_FEVALS:-80}
EXTRA_ARGS=("$@")

cat <<EOF
🚀 Starting TinyLlama SLM GA demo
📁 Working directory: $(pwd)
🧪 Config: ${CONFIG_PATH}
🗂️  Storage path: ${STORAGE_PATH}
📝 Log file: ${LOG_PATH}
📈 Max function evals: ${MAX_FEVALS}
EOF

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "❌ Configuration file '${CONFIG_PATH}' not found. Set SLM_CONFIG to override." >&2
  exit 1
fi

mkdir -p "${STORAGE_PATH}"
mkdir -p "$(dirname "${LOG_PATH}")"

printf '\n🧹 Cleaning up lingering Ray state...\n'
ray stop --force >/dev/null 2>&1 || true
sleep 1

printf '\n▶️  Launching GA evolution (serial CPU strategy) ...\n'
set -x
python -m mergekit.scripts.evolve_ga "${CONFIG_PATH}" \
  --storage-path "${STORAGE_PATH}" \
  --strategy serial \
  --num-workers 1 \
  --num-gpus 0 \
  --no-merge-cuda \
  --max-fevals "${MAX_FEVALS}" \
  --save-final-model \
  --no-reshard \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} | tee "${LOG_PATH}"
set +x

echo "\n✅ GA run complete. Review ${STORAGE_PATH}/ga_history.csv for per-generation metrics."