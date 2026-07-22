#!/bin/bash
# ============================================================================
# CAMPAIGN 2 — Audited fidelity + memetic repair. One resumable queue.
# Phases: 1) Exp 2.11a connected pair, audit+quarantine, seeds 11/22
#         2) Exp 2.11b disconnected control, audit+quarantine, seed 11
#         3) Exp 2.12 connected pair, audit+quarantine+memetic repair, seeds 11/22
# 5 runs, ~10-14 h (audits add ~1-2 h/run). Safe to Ctrl-C; re-paste to resume.
# Every validated run leaves a DONE marker; re-running skips verified work.
# Paste command:
#   bash "$HOME/Documents/master_research/mergekit/experiments/thesis/local_mac/pending_experiments/run_campaign2.sh"
# ============================================================================
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"
MON="$HOME/Documents/master_research/Master Research Paper/Research Project/experiments/m1_campaign2"
OUT="$REPO/workspace/thesis/local_mac/results"
SHARED_HF_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"
mkdir -p "$MON" "$OUT" "$SHARED_HF_CACHE"

# Venv lives OUTSIDE ~/Documents so iCloud cannot evict its compiled libraries
# (the pandas mmap errno=60 crashes came from iCloud evicting .so files in
# the old $REPO/.venv-exp).
VENV="$HOME/.venvs/mergekit-exp"
PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
if [ ! -d "$VENV" ]; then
  echo "--- creating venv at $VENV + installing (first run only, several minutes) ---"
  mkdir -p "$HOME/.venvs"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi
PYBIN="$VENV/bin/python"
"$PYBIN" -m pip install --quiet -e "$REPO" lm_eval ray matplotlib \
  --constraint "$HERE/m1_constraints.txt"
# Sanity check the interpreter can actually load its compiled deps before launching.
if ! "$PYBIN" -c "import pandas, torch, numpy" 2>/dev/null; then
  echo "--- venv sanity check failed; reinstalling binary packages ---"
  "$PYBIN" -m pip install --quiet --force-reinstall pandas numpy torch \
    --constraint "$HERE/m1_constraints.txt"
  "$PYBIN" -c "import pandas, torch, numpy" || { echo "venv still broken; aborting"; exit 1; }
fi

# Record code provenance for the campaign artefact trail.
GITHASH=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
GITDIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null | head -1)
log() { echo "{\"ts\":\"$(date -Iseconds)\",\"msg\":\"$1\"}" >> "$MON/progress.log"; echo "[$(date '+%H:%M:%S')] $1"; }
if [ -n "$GITDIRTY" ]; then
  echo "WARNING: working tree has uncommitted changes (campaign provenance will record ${GITHASH}-dirty)."
fi

on_exit() {
  STATUS=$?
  trap - EXIT
  if [ "$STATUS" -ne 0 ]; then
    log "CAMPAIGN2 FAILED exit $STATUS"
  fi
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

# The 12:51 attempt archived nothing of value (crashed at import); clean its stub.
if [ -d "$OUT/exp211a_connected_audit/seed11" ] && [ ! -f "$OUT/exp211a_connected_audit/seed11/ga_candidate_history.csv" ]; then
  rm -rf "$OUT/exp211a_connected_audit/seed11"
fi

mirror_csvs() {  # dir name seed
  local DIR=$1 NAME=$2 SEED=$3 f
  for f in ga_method_history ga_candidate_history ga_audit_history ga_reentry_history; do
    [ -f "$DIR/$f.csv" ] && cp "$DIR/$f.csv" "$MON/${NAME}_seed${SEED}_${f#ga_}.csv"
  done
  return 0
}

run_ga () {  # name config seed
  NAME=$1; CFG=$2; SEED=$3
  DIR="$OUT/$NAME/seed${SEED}"

  if [ -d "$DIR" ] && validate_run "$DIR" "$HERE/$CFG" >/dev/null 2>&1; then
    mirror_csvs "$DIR" "$NAME" "$SEED"
    date -Iseconds > "$DIR/DONE"
    log "skip $NAME seed $SEED (validated)"
    return
  fi
  if [ -f "$DIR/DONE" ]; then
    rm -f "$DIR/DONE"
    log "reject stale DONE for $NAME seed $SEED"
  fi

  RESUME=0
  if [ -f "$DIR/ga_state.json" ]; then
    RESUME=1
  elif [ -d "$DIR" ] && [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    FAILED_DIR="${DIR}.failed-$(date '+%Y%m%d-%H%M%S')"
    mv "$DIR" "$FAILED_DIR"
    log "archive incomplete $NAME seed $SEED to $FAILED_DIR"
  fi

  mkdir -p "$DIR"
  echo "{\"git\":\"$GITHASH\",\"dirty\":\"${GITDIRTY:+yes}\"}" > "$DIR/code_provenance.json"
  if [ ! -e "$DIR/transformers_cache" ] && [ ! -L "$DIR/transformers_cache" ]; then
    ln -s "$SHARED_HF_CACHE" "$DIR/transformers_cache"
  fi
  if [ "$RESUME" -eq 1 ]; then
    STORAGE_ARGS=(--resume "$DIR")
    TEE_ARGS=(-a "$DIR/run.log")
    log "resume $NAME seed $SEED"
  else
    STORAGE_ARGS=(--storage-path "$DIR")
    TEE_ARGS=("$DIR/run.log")
    log "start $NAME seed $SEED"
  fi

  if run_nicely "$PYBIN" -m mergekit.scripts.evolve_ga "$HERE/$CFG" \
      --random-seed "$SEED" "${STORAGE_ARGS[@]}" --strategy serial \
      --no-vllm --no-merge-cuda --max-disk-gb-min 5 \
      2>&1 | tee "${TEE_ARGS[@]}" | tail -3; then
    :
  else
    STATUS=$?
    log "fail $NAME seed $SEED process exit $STATUS"
    return "$STATUS"
  fi

  if ! validate_run "$DIR" "$HERE/$CFG"; then
    log "fail $NAME seed $SEED artifact validation"
    return 1
  fi
  mirror_csvs "$DIR" "$NAME" "$SEED"
  date -Iseconds > "$DIR/DONE"
  log "finish $NAME seed $SEED"
}

log "CAMPAIGN2 LAUNCH (code $GITHASH${GITDIRTY:+-dirty})"
# --- Phase 1: Exp 2.11a — connected pair, audited fidelity, no repair ---
for S in 11 22; do run_ga exp211a_connected_audit exp211_connected_audit.yml $S; done
log "PHASE 1 (Exp 2.11a) COMPLETE"

# --- Phase 2: Exp 2.11b — disconnected control ---
run_ga exp211b_disconnected_audit exp211b_disconnected_audit.yml 11
log "PHASE 2 (Exp 2.11b) COMPLETE"

# --- Phase 3: Exp 2.12 — memetic repair with re-entry ---
for S in 11 22; do run_ga exp212_memetic exp212_memetic.yml $S; done
log "PHASE 3 (Exp 2.12) COMPLETE"

log "CAMPAIGN2 ALL DONE"
echo ""
echo "============================================"
echo "Campaign 2 complete. Results under $OUT"
echo "Monitoring CSVs mirrored to $MON"
echo "============================================"
