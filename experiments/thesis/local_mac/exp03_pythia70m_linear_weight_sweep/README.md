# Experiment 3: Pythia-70M Deterministic Linear Weight Sweep

This experiment replaces GA search with a deterministic interpolation sweep over
the same two local Pythia-70M parents used in the earlier validation run.

## Purpose

The previous GA run showed that the best local answer was simply to preserve
`lomahony/pythia-70m-helpful-sft`. That makes a GA a poor first tool for the
next question. Before trying another evolutionary search, it is better to map
the interpolation landscape directly.

This experiment therefore evaluates a fixed grid of `linear` merge weights
between:

- `lomahony/pythia-70m-helpful-sft`
- `EleutherAI/pythia-70m-deduped`

## Why This Is Better Than Another GA First

- The search space is only one-dimensional.
- Deterministic coverage is cheaper than stochastic search.
- Endpoints double as baseline checks.
- It tells us whether any useful region exists between the two parents at all.

## Recommended Local Run

```bash
python -m mergekit.scripts.linear_sweep \
  experiments/thesis/local_mac/exp03_pythia70m_linear_weight_sweep/config.yml \
  --storage-path /tmp/mergekit-linear-sweep/pythia70m-alpha-grid \
  --alpha-step 0.05 \
  --coarse-limit 4 \
  --refine-limit 32 \
  --top-k 5 \
  --batch-size 1
```

## Staged Reevaluation

The script performs a cheap first pass at `--coarse-limit` for every alpha, then
reruns the top `--top-k` candidates at `--refine-limit`. This keeps the local
run tractable while reducing the chance that a noisy low-limit evaluation picks
the wrong winner.

## Expected Outputs

- `coarse_results.csv`
- `refine_results.csv`
- `summary.txt`
- `best_config.yaml`
- `final_model/` if `--save-final-model` is left enabled
