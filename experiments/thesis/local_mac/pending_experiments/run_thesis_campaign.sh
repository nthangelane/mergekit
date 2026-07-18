#!/bin/bash
# ============================================================================
# THESIS EXPERIMENT CAMPAIGN — one resumable queue.
# Phases: 1) Exp 2.6 AOS ablation (6 runs)  2) Seeds 44/55 (2 runs)
#         3) Exp 2.7 native random-search parity (3 runs)
#         4) Exp 2.8 mode-connected pair, adaptive + random arms (6 runs)
#         5) Exp 2.9a gated-repair probe (1 run)
# Every validated run leaves a DONE marker; re-running skips only verified work.
# Total ~18 runs, ~15-20 h of compute. Safe to Ctrl-C anytime; re-paste to resume.
# ============================================================================
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"
MON="$HOME/Documents/master_research/Master Research Paper/Research Project/experiments/m1_campaign"
OUT="$REPO/workspace/thesis/local_mac/results"
SHARED_HF_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface/hub}"
mkdir -p "$MON" "$OUT" "$SHARED_HF_CACHE"

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
  if [ "$STATUS" -ne 0 ]; then
    log "CAMPAIGN FAILED exit $STATUS"
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

run_ga () {  # name config seed extra_flags...
  NAME=$1; CFG=$2; SEED=$3; shift 3
  DIR="$OUT/$NAME/seed${SEED}"
  IS_RANDOM=0
  for ARG in "$@"; do
    if [ "$ARG" = "--random-search" ]; then IS_RANDOM=1; fi
  done

  if [ -d "$DIR" ] && validate_run "$DIR" "$HERE/$CFG" >/dev/null 2>&1; then
    for f in ga_method_history ga_candidate_history; do
      cp "$DIR/$f.csv" "$MON/${NAME}_seed${SEED}_${f#ga_}.csv"
    done
    date -Iseconds > "$DIR/DONE"
    log "skip $NAME seed $SEED (validated)"
    return
  fi

  VALID_ARCHIVE=""
  for ARCHIVE in "$DIR".failed-*; do
    if [ -d "$ARCHIVE" ] && validate_run "$ARCHIVE" "$HERE/$CFG" >/dev/null 2>&1; then
      VALID_ARCHIVE="$ARCHIVE"
    fi
  done
  if [ -n "$VALID_ARCHIVE" ]; then
    if [ -d "$DIR" ]; then
      SUPERSEDED_DIR="${DIR}.superseded-$(date '+%Y%m%d-%H%M%S')"
      mv "$DIR" "$SUPERSEDED_DIR"
      log "archive duplicate $NAME seed $SEED to $SUPERSEDED_DIR"
    fi
    mv "$VALID_ARCHIVE" "$DIR"
    log "restore validated $NAME seed $SEED from $VALID_ARCHIVE"
    for f in ga_method_history ga_candidate_history; do
      cp "$DIR/$f.csv" "$MON/${NAME}_seed${SEED}_${f#ga_}.csv"
    done
    date -Iseconds > "$DIR/DONE"
    log "skip $NAME seed $SEED (restored and validated)"
    return
  fi
  if [ -f "$DIR/DONE" ]; then
    rm -f "$DIR/DONE"
    log "reject stale DONE for $NAME seed $SEED"
  fi

  if [ "$IS_RANDOM" -eq 1 ] && [ -f "$DIR/ga_stop_details.json" ]; then
    log "finalize completed search $NAME seed $SEED"
    if ! "$PYBIN" -m mergekit.scripts.finalize_evo_run \
        "$HERE/$CFG" "$DIR" --device cpu --min-free-disk-gb 5; then
      log "fail $NAME seed $SEED completed-search finalization"
      return 1
    fi
    if ! validate_run "$DIR" "$HERE/$CFG"; then
      log "fail $NAME seed $SEED recovered artifact validation"
      return 1
    fi
    for f in ga_method_history ga_candidate_history; do
      cp "$DIR/$f.csv" "$MON/${NAME}_seed${SEED}_${f#ga_}.csv"
    done
    date -Iseconds > "$DIR/DONE"
    log "recover $NAME seed $SEED without search rerun"
    return
  fi

  RESUME=0
  if [ "$IS_RANDOM" -eq 0 ] && [ -f "$DIR/ga_state.json" ]; then
    RESUME=1
  elif [ -d "$DIR" ] && [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    FAILED_DIR="${DIR}.failed-$(date '+%Y%m%d-%H%M%S')"
    mv "$DIR" "$FAILED_DIR"
    log "archive incomplete $NAME seed $SEED to $FAILED_DIR"
  fi

  mkdir -p "$DIR"
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
      --no-vllm --no-merge-cuda --max-disk-gb-min 5 "$@" \
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
  for f in ga_method_history ga_candidate_history; do
    cp "$DIR/$f.csv" "$MON/${NAME}_seed${SEED}_${f#ga_}.csv"
  done
  date -Iseconds > "$DIR/DONE"
  log "finish $NAME seed $SEED"
}

log "CAMPAIGN LAUNCH"
# --- Phase 1: Experiment 2.6 — AOS ablation (controlled, both arms) ---
for S in 11 22 33; do run_ga exp26_aos_off thesis_aos_off.yml     $S; done
for S in 11 22 33; do run_ga exp26_aos_on  thesis_aos_current.yml $S; done
log "PHASE 1 (Exp 2.6) COMPLETE"

# --- Phase 2: Seeds 44/55 with the TRUE campaign config (protocol-identical) ---
for S in 44 55;    do run_ga seeds4455    thesis_aos_current.yml $S; done
log "PHASE 2 (Seeds 44/55) COMPLETE"

# --- Phase 3: Experiment 2.7 parity — native random search, original pair ---
for S in 11 22 33; do run_ga exp27_native thesis_aos_current.yml $S --random-search 96; done
log "PHASE 3 (Exp 2.7 native parity) COMPLETE"

# --- Phase 4: Experiment 2.8 — mode-connected pair, both arms ---
for S in 11 22 33; do run_ga exp28_adaptive exp28_connected_adaptive.yml $S; done
for S in 11 22 33; do run_ga exp28_random   exp28_connected_adaptive.yml $S --random-search 96; done
log "PHASE 4 (Exp 2.8) COMPLETE"

# --- Phase 5: Experiment 2.9a — gated-repair probe (disconnected pair) ---
run_ga exp29a_probe exp29_repair_probe.yml 11
log "PHASE 5 (Exp 2.9a probe) COMPLETE"

log "CAMPAIGN ALL DONE"
echo ""
echo "============================================"
echo "Campaign complete. Results under $OUT"
echo "Monitoring CSVs mirrored to $MON"
echo "============================================"
