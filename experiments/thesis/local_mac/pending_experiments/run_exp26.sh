#!/bin/bash
# Experiment 2.6 — AOS ablation: adaptive (ON) vs static uniform (OFF), seeds 11/22/33 each.
# Both arms re-run with the identical thesis_aos_current base config so the comparison
# is controlled by construction. Resumable: re-run this script after any interruption.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
MON="$HOME/Documents/master_research/Master Research Paper/Research Project/experiments/m1_run_exp26"
OUT="$REPO/workspace/thesis/local_mac/results/exp26"
mkdir -p "$MON" "$OUT"

echo "=== Experiment 2.6 launcher (repo: $REPO) ==="
PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
if [ ! -d "$REPO/.venv-exp" ]; then
  echo "--- creating venv + installing (first run only, several minutes) ---"
  "$PY" -m venv "$REPO/.venv-exp"
  "$REPO/.venv-exp/bin/pip" install --quiet --upgrade pip
fi
PYBIN="$REPO/.venv-exp/bin/python"
"$PYBIN" -m pip install --quiet -e "$REPO" lm_eval ray matplotlib \
  --constraint "$HERE/m1_constraints.txt"

log() { echo "{\"ts\":\"$(date -Iseconds)\",\"msg\":\"$1\"}" >> "$MON/progress.log"; echo "[$(date '+%H:%M:%S')] $1"; }

on_exit() {
  STATUS=$?
  trap - EXIT
  if [ "$STATUS" -ne 0 ]; then log "EXP26 FAILED exit $STATUS"; fi
  exit "$STATUS"
}
trap on_exit EXIT

run_nicely() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}

validate_run() {
  "$PYBIN" -m mergekit.scripts.validate_evo_run "$1" --config "$2"
}

run_one () {  # arm config seed
  ARM=$1; CFG=$2; SEED=$3
  DIR="$OUT/${ARM}_seed${SEED}"
  if [ -f "$DIR/DONE" ]; then
    if validate_run "$DIR" "$HERE/$CFG" >/dev/null 2>&1; then
      log "skip ${ARM} seed ${SEED} (validated)"
      return
    fi
    rm -f "$DIR/DONE"
    log "reject stale DONE for ${ARM} seed ${SEED}"
  fi

  RESUME=0
  if [ -f "$DIR/ga_state.json" ]; then
    RESUME=1
  elif [ -d "$DIR" ] && [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    FAILED_DIR="${DIR}.failed-$(date '+%Y%m%d-%H%M%S')"
    mv "$DIR" "$FAILED_DIR"
    log "archive incomplete ${ARM} seed ${SEED} to $FAILED_DIR"
  fi

  mkdir -p "$DIR"
  if [ "$RESUME" -eq 1 ]; then
    STORAGE_ARGS=(--resume "$DIR")
    TEE_ARGS=(-a "$DIR/run.log")
    log "resume ${ARM} seed ${SEED}"
  else
    STORAGE_ARGS=(--storage-path "$DIR")
    TEE_ARGS=("$DIR/run.log")
    log "start ${ARM} seed ${SEED}"
  fi

  if run_nicely "$PYBIN" -m mergekit.scripts.evolve_ga "$HERE/$CFG" \
      --random-seed "$SEED" "${STORAGE_ARGS[@]}" --strategy serial \
      --no-vllm --no-merge-cuda --max-disk-gb-min 5 \
      2>&1 | tee "${TEE_ARGS[@]}" | tail -5; then
    :
  else
    STATUS=$?
    log "fail ${ARM} seed ${SEED} process exit $STATUS"
    return "$STATUS"
  fi

  if ! validate_run "$DIR" "$HERE/$CFG"; then
    log "fail ${ARM} seed ${SEED} artifact validation"
    return 1
  fi
  cp "$DIR/ga_method_history.csv" "$MON/${ARM}_seed${SEED}_method_history.csv"
  cp "$DIR/ga_candidate_history.csv" "$MON/${ARM}_seed${SEED}_candidate_history.csv"
  date -Iseconds > "$DIR/DONE"
  log "finish ${ARM} seed ${SEED}"
}

for SEED in 11 22 33; do run_one aos_off thesis_aos_off.yml   $SEED; done
for SEED in 11 22 33; do run_one aos_on  thesis_aos_current.yml $SEED; done
log "EXP26 ALL DONE"
echo "All six runs complete. Results in $OUT and mirrored CSVs in $MON."
