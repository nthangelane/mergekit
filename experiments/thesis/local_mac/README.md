# Thesis Local Mac Track

This group is for the small validation run that should execute on the development machine before spending cloud GPU time.

## Included Experiment

- `exp01_tiny_controlled_merge`
- `exp02_pythia70m_linear_passthrough_validation`
- `exp03_pythia70m_linear_weight_sweep`
- `exp04_pythia70m_slerp_base_validation`
- `main_adaptive_pythia70m.yml`
- `phase3_ablation_pythia70m.yml`
- `thesis_run_pythia70m.yml`
- `BENCHMARK_PLAN_20260319.md`
- `RESULTS_20260321_THESIS_BATCH.md`
- `RESULTS_20260425_DEPENDENCY_PROBE.md`
- `PROJECT_IMPROVEMENT_TODO_20260426.md`
- `DEPENDENCY_COMPATIBILITY_NOTES_20260426.md`
- `NEW_MACHINE_SETUP.md`
- `run_thesis_pool_local.sh`
- `thesis_run_pythia70m_tuned.yml`
- `pending_experiments/`

## Current Local Defaults

- Main local adaptive preset:
  [`main_adaptive_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/main_adaptive_pythia70m.yml)
- Thesis local 70M preset:
  [`thesis_run_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m.yml)
- Phase-3 structural ablation:
  [`phase3_ablation_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/phase3_ablation_pythia70m.yml)
- Results write-up:
  [`RESULTS_20260319.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260319.md)
- Completed three-seed pooled thesis batch summary:
  [`RESULTS_20260321_THESIS_BATCH.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260321_THESIS_BATCH.md)
- Dependency-probe thesis validation:
  [`RESULTS_20260425_DEPENDENCY_PROBE.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260425_DEPENDENCY_PROBE.md)
- Project improvement checklist:
  [`PROJECT_IMPROVEMENT_TODO_20260426.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/PROJECT_IMPROVEMENT_TODO_20260426.md)
- Dependency compatibility notes:
  [`DEPENDENCY_COMPATIBILITY_NOTES_20260426.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/DEPENDENCY_COMPATIBILITY_NOTES_20260426.md)
- Thesis benchmark plan:
  [`BENCHMARK_PLAN_20260319.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/BENCHMARK_PLAN_20260319.md)
- New machine setup and runbook:
  [`NEW_MACHINE_SETUP.md`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/NEW_MACHINE_SETUP.md)
- Reusable pooled local launcher:
  [`run_thesis_pool_local.sh`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/run_thesis_pool_local.sh)
- Proposed tuned follow-up preset:
  [`thesis_run_pythia70m_tuned.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m_tuned.yml)
- Pending follow-up experiment specs:
  [`pending_experiments/`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/pending_experiments)

## Suggested Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp01_tiny_controlled_merge/config.yml \
  --storage-path workspace/thesis/local_mac/results/exp01_tiny_controlled_merge \
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

The main local adaptive preset is now
[`main_adaptive_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/main_adaptive_pythia70m.yml).
It is the recommended default because the short validation run beat both parent
baselines while keeping the search space narrow and stable.

For thesis-worthy local evidence, the longer preset is now
[`thesis_run_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m.yml).
It keeps the same stable search space, but raises evaluation fidelity, adds the
new stop-policy controls, and is intended to be run across multiple seeds for
reportable tables and narrative.

The richer structural path is now kept as the explicit ablation preset
[`phase3_ablation_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/phase3_ablation_pythia70m.yml).
It is useful for research comparison, but not yet the default local run,
because the first short validation run failed all sampled children in
generation 1.

For this machine, the safe operating rule is still the same: local runs only
use tiny models. Anything materially larger than Pythia-70M belongs on the EKS
track.

## Suggested 50-Generation Local Validation Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp02_pythia70m_linear_passthrough_validation/config.yml \
  --storage-path workspace/thesis/local_mac/results/exp02_pythia70m_linear_passthrough_validation \
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
  --storage-path workspace/thesis/local_mac/results/exp03_pythia70m_linear_weight_sweep \
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
  --storage-path workspace/thesis/local_mac/results/exp04_pythia70m_slerp_base_validation \
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

## Suggested Main Adaptive Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/main_adaptive_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/main_adaptive_pythia70m \
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

## Suggested Thesis 70M Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/thesis_run_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/20260319-thesis-run/seed11 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 11
```

## Suggested Phase-3 Ablation Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/phase3_ablation_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/phase3_ablation_pythia70m \
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
