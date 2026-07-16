#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${MERGEKIT_CAMPAIGN_WRAPPED:-0}" != "1" ]]; then
  export MERGEKIT_CAMPAIGN_WRAPPED=1
  if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -dimsu nice -n "${MERGEKIT_NICE_LEVEL:-10}" "$0" "$@"
  fi
  exec nice -n "${MERGEKIT_NICE_LEVEL:-10}" "$0" "$@"
fi

cd "$ROOT"
exec "${PYTHON:-python}" -m mergekit.scripts.run_campaign "$@"
