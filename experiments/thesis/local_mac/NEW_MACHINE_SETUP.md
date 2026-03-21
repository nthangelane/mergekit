# New Machine Setup And Run Guide

This guide reproduces the local 70M adaptive GA thesis run from a fresh
machine.

## Scope

Use this path for the local tiny-model experiments only.

- Safe local pair:
  - `lomahony/pythia-70m-helpful-sft`
  - `EleutherAI/pythia-70m-deduped`
- Main thesis config:
  [`thesis_run_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m.yml)
- Branch used for the adaptive GA work:
  `codex/ga-evolve/adaptive-ga-phase-1`
- Current commit used when this guide was written:
  `90fe0d67a5ef51b05b9348c4dcac3023a491cac0`

## Prerequisites

Install these first:

- `git`
- `python 3.11` or newer
- enough free disk for model cache and run artifacts
  - recommended: at least `20 GB`

Optional but useful:

- Hugging Face login if you expect rate limits:

```bash
huggingface-cli login
```

## Clone The Repo

```bash
git clone https://github.com/nthangelane/mergekit.git
cd mergekit
git checkout codex/ga-evolve/adaptive-ga-phase-1
git rev-parse HEAD
```

The last command should print the commit you are actually on.

If you also want the original upstream remote available:

```bash
git remote add upstream https://github.com/arcee-ai/mergekit.git
git remote -v
```

## Create A Python Environment

Using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[evolve-ga,test]"
```

If you prefer Conda, the important part is still installing:

```bash
python -m pip install -e ".[evolve-ga,test]"
```

## Quick Verification

Confirm the GA entrypoint is available:

```bash
python -m mergekit.scripts.evolve_ga --help
```

If you want a clean local runtime before starting:

```bash
ray stop --force || true
pkill -f "mlflow ui" || true
```

## Recommended First Smoke Run

This checks the setup with a short pooled run before the full thesis batch.

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/thesis_run_pythia70m.yml \
  --strategy pool \
  --num-workers 2 \
  --storage-path workspace/thesis/local_mac/results/smoke-seed11 \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --mlflow \
  --mlflow-experiment mergekit-thesis-smoke \
  --mlflow-ui-port 5013 \
  --no-reshard \
  --random-seed 11 \
  --max-fevals 8
```

Expected outputs:

- `workspace/thesis/local_mac/results/smoke-seed11/baseline_results.csv`
- `workspace/thesis/local_mac/results/smoke-seed11/ga_history.csv`
- `workspace/thesis/local_mac/results/smoke-seed11/mlflow_run_info.md`
- `workspace/thesis/local_mac/results/smoke-seed11/ray_observability.json`

## Run The Full Local Thesis Batch

The repo now includes a reusable launcher:

- script:
  [`run_thesis_pool_local.sh`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/run_thesis_pool_local.sh)

Make it executable once:

```bash
chmod +x experiments/thesis/local_mac/run_thesis_pool_local.sh
```

Launch the full pooled batch:

```bash
./experiments/thesis/local_mac/run_thesis_pool_local.sh 20260321-thesis-pool-batch
```

What this does:

- runs seeds `11`, `22`, `33` sequentially
- uses `pool` strategy with `2` workers
- uses the thesis config
- writes results under `workspace/thesis/local_mac/results/<batch-label>`
- enables MLflow
- symlinks each run's `transformers_cache` to the shared local Hugging Face cache

### Useful Environment Overrides

You can override defaults without editing the script:

```bash
NUM_WORKERS=2 \
MLFLOW_UI_PORT=5014 \
SEEDS="11 22 33" \
HF_SHARED_CACHE="$HOME/.cache/huggingface/hub" \
./experiments/thesis/local_mac/run_thesis_pool_local.sh my-batch
```

## Monitor The Run

### Ray UI

Once the local Ray runtime is up, the dashboard is usually:

```text
http://127.0.0.1:8265
```

Check:

- `Jobs`
- `Actors`
- `Tasks`
- worker logs

### MLflow

The launcher starts MLflow locally. With the default port:

```text
http://127.0.0.1:5013
```

Each seed writes:

- `mlflow_run_info.md`
- `ray_observability.json`
- `run.log`

inside its run directory.

### Main Files To Watch

For each seed:

- `baseline_results.csv`
- `ga_history.csv`
- `ga_candidate_history.csv`
- `ga_method_history.csv`
- `ga_summary.txt`
- `ga_stop_details.json`
- `best_config.yaml`
- `ga_history_plot.png`
- `ga_final_comparison.csv`
- `ga_final_comparison.png`

## Output Layout

Example batch layout:

```text
workspace/thesis/local_mac/results/20260321-thesis-pool-batch/
  batch.log
  mlruns/
  seed11/
  seed22/
  seed33/
```

## Benchmarking The Winner

At the end of each seed run, `evolve_ga.py` already writes:

- `ga_final_comparison.csv`
- `ga_final_comparison.png`

Those files compare the final merged model against the baselines on the config's
task set.

If you want the same multi-seed thesis process we are using here:

1. let `seed11`, `seed22`, and `seed33` complete
2. compare the three `ga_final_comparison.csv` files
3. keep the strongest merged result
4. summarize the seeds with:
   - stop reason
   - best generation
   - best score
   - method mix
   - final comparison versus baselines

## Common Problems

### Ray UI not loading

Stop and restart local Ray:

```bash
ray stop --force
```

Then relaunch the batch.

### MLflow port already in use

Change the port:

```bash
MLFLOW_UI_PORT=5014 ./experiments/thesis/local_mac/run_thesis_pool_local.sh my-batch
```

### Cold-cache stalls

Make sure the Hugging Face cache directory exists and is reused:

```bash
ls ~/.cache/huggingface/hub
```

The launcher symlinks each run to that shared cache on purpose.

### Disk pressure

Check:

```bash
df -h .
du -sh ~/.cache/huggingface
```

## Recommended Thesis Path

For local reporting:

1. run the smoke
2. run the three-seed pooled batch
3. keep the outputs in `workspace/thesis/local_mac/results/...`
4. write the thesis tables from:
   - `baseline_results.csv`
   - `ga_history.csv`
   - `ga_method_history.csv`
   - `ga_final_comparison.csv`

## One-Line Summary

If you only want the shortest path on a new machine:

```bash
git clone https://github.com/nthangelane/mergekit.git && \
cd mergekit && \
git checkout codex/ga-evolve/adaptive-ga-phase-1 && \
python3.11 -m venv .venv && \
source .venv/bin/activate && \
python -m pip install --upgrade pip wheel setuptools && \
python -m pip install -e ".[evolve-ga,test]" && \
chmod +x experiments/thesis/local_mac/run_thesis_pool_local.sh && \
./experiments/thesis/local_mac/run_thesis_pool_local.sh
```
