#!/bin/bash
# run_slm_m1_demo.sh - Launch the TinyLlama SLM GA demo on CPU-only Apple silicon.
# Usage: ./run_slm_m1_demo.sh [extra mergekit args]

set -euo pipefail

CONFIG_PATH=${SLM_CONFIG:-examples/slm_m1_demo.yml}
STORAGE_PATH=${SLM_STORAGE:-workspace/slm_m1_demo_run}
LOG_PATH=${SLM_LOG:-workspace/slm_m1_demo.log}
MAX_FEVALS=${SLM_MAX_FEVALS:-80}
UPLOAD_MODEL=${SLM_UPLOAD:-0}
HF_USERNAME=${HF_USERNAME:-}
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

if [[ "${UPLOAD_MODEL}" == "1" ]]; then
  if [[ -z "${HF_USERNAME}" ]]; then
    echo "❌ SLM_UPLOAD=1 set but HF_USERNAME is empty. Set HF_USERNAME and retry." >&2
    exit 1
  fi

  echo "\n⬆️  Uploading final model to Hugging Face..."
  /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
    python /Users/nkululekothangelane/.vscode/extensions/ms-python.python-2026.0.0-darwin-arm64/python_files/get_output_via_markers.py -c "import datetime as _d; import re; from huggingface_hub import create_repo, upload_folder;\nimport yaml;\nfrom pathlib import Path;\nconf = yaml.safe_load(Path('${CONFIG_PATH}').read_text());\nbase = conf.get('genome', {}).get('base_model', 'model');\nshort = re.sub(r'[^a-zA-Z0-9]+', '-', str(base).split('/')[-1]).strip('-').lower();\nstamp = _d.datetime.now().strftime('%d%b').lower();\nrepo_id = f'${HF_USERNAME}/gaevolve-{stamp}-{short}';\ncreate_repo(repo_id, repo_type='model', exist_ok=True);\nupload_folder(repo_id=repo_id, folder_path='${STORAGE_PATH}/final_model', repo_type='model');\nprint(repo_id)" \
    || { echo "❌ Upload failed. Ensure you are logged in with huggingface-cli login." >&2; exit 1; }
fi