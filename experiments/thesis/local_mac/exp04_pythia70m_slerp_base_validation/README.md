# Experiment 4: Pythia-70M SLERP + Passthrough Base-Model Validation

This experiment is the next local GA after the linear-plus-passthrough run.

## Purpose

The earlier local GA suggested that plain global `linear` blending mostly
damaged the stronger parent. This follow-up keeps `passthrough` for baseline
preservation, but replaces `linear` with `slerp` and sets an explicit
`base_model`.

## Why SLERP Here

`task_arithmetic` is deferred for now because this local pair only gives us one
clear non-base tuned model. `slerp` is the cleaner next experiment with the
current assets because it supports two-model interpolation while still using a
valid `base_model`.

This config uses:

- source models:
  `lomahony/pythia-70m-helpful-sft` and `EleutherAI/pythia-70m-deduped`
- base model:
  `EleutherAI/pythia-70m-deduped`
- methods:
  `slerp` and `passthrough`

## Recommended Local Smoke Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp04_pythia70m_slerp_base_validation/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-slerp-base-smoke \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 4 \
  --max-fevals 8 \
  --limit 4 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```

## Recommended Full Local Validation Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp04_pythia70m_slerp_base_validation/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-slerp-base-pop12-g30 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 12 \
  --max-fevals 360 \
  --limit 32 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```

## Interpretation Goal

If this run still collapses to `passthrough`, the problem is probably the parent
pair or the task mix, not just the choice of merge operator.
