# Thesis Local Mac Track

This group is for the small validation run that should execute on the development machine before spending cloud GPU time.

## Included Experiment

- `exp01_tiny_controlled_merge`
- `exp02_pythia70m_linear_passthrough_validation`
- `exp03_pythia70m_linear_weight_sweep`
- `exp04_pythia70m_slerp_base_validation`
- `exp05_pythia70m_phase2_adaptive_probe`
- `exp06_pythia70m_phase3_layered_rank_probe`

## Suggested Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp01_tiny_controlled_merge/config.yml \
  --storage-path workspace/thesis/local_mac/exp01_tiny_controlled_merge \
  --max-fevals 480 \
  --strategy pool \
  --num-gpus 0 \
  --num-workers 4 \
  --no-vllm
```

## Pythia-70M Validation Narrative

The local Mac track exists to validate GA mechanics on tiny models before using
cloud GPUs. In practice this means tens of millions of parameters, or at most a
very small same-family pair such as Pythia-70M when the search space is tightly
constrained and the run stays CPU-only and serial.

The `exp02_pythia70m_linear_passthrough_validation` run is not a thesis-scale
performance experiment. It is a local validation experiment designed to answer a
more limited question: can the GA preserve a strong parent baseline while
exploring a narrow linear-plus-passthrough search space without tokenizer drift
or merge-path failures? The run uses only two source models, reuses the base
tokenizer, enables baseline evaluation, and weights `sciq` more heavily than
`wikitext` so the fitness function is not dominated by perplexity.

The next two local follow-ups are more diagnostic than the first GA. `exp03`
replaces the GA entirely with a deterministic linear interpolation sweep plus
staged reevaluation so the one-dimensional landscape can be mapped directly.
`exp04` then reintroduces the GA with an explicit `base_model` and `slerp` so
the next operator is tested without losing baseline preservation.

`exp05` is the first local phase-2 preset. It keeps the tiny-model safety
constraints but adds adaptive operator sampling, explorer slots,
diversity-aware parent selection, candidate-level history, and behavior probes
that can reject obviously broken children before the full evaluation pass.

`exp06` is the local phase-3 preset. It keeps the same tiny Pythia-70M pair,
but switches to nonzero layer granularity, weighted-rank fitness, a novelty
archive bonus, and the expanded method family so the structural path can be
smoke-tested without leaving the local machine.

For this machine, the safe operating rule is still the same: local runs only
use tiny models. Anything materially larger than Pythia-70M belongs on the EKS
track.

## Suggested 50-Generation Local Validation Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp02_pythia70m_linear_passthrough_validation/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-linear-passthrough-base-pop12-g50 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 12 \
  --max-fevals 600 \
  --limit 4 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```

## Suggested Deterministic Sweep

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

## Suggested Base-Model SLERP Validation

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

## Suggested Phase-2 Probe Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp05_pythia70m_phase2_adaptive_probe/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-phase2-adaptive-probe \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 8 \
  --max-fevals 96 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```

## Suggested Phase-3 Probe Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp06_pythia70m_phase3_layered_rank_probe/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-phase3-layered-rank-probe \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 6 \
  --max-fevals 12 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```
