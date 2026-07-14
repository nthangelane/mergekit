# Thesis Experiment Summary: Local Adaptive GA Track

This document summarizes the local thesis experiments run for the adaptive
genetic algorithm model-merging work. The local track uses small Pythia-70M
models so that the complete workflow can run on a Mac without GPU dependency:
baseline evaluation, merge generation, tokenizer handling, staged evaluation,
GA logging, final model export, and final comparison reporting.

The local experiments should be interpreted as validation and ablation evidence,
not as claims about large-scale model merging. Their purpose is to verify that
the adaptive GA is stable, observable, and able to preserve or improve candidate
quality before more expensive GPU/EKS experiments are attempted.

## Research Questions

The experiments were designed around four questions:

1. Can the local GA pipeline complete repeated model-merge runs without
   tokenizer, merge-path, or evaluation crashes?
2. Does the adaptive operator policy learn useful preferences between
   `linear`, `passthrough`, `slerp`, and richer structural operators?
3. Does search fitness translate into final exported model quality?
4. Which local configuration is safe enough to use as the main thesis preset?

## Experiment Timeline

| Date | Experiment | Purpose | Result |
| --- | --- | --- | --- |
| 2026-03-15 | Linear + passthrough validation | Prove the hardened local path can complete a longer CPU run | Completed 50 generations / 600 evaluations; preserved the strongest parent but did not improve it |
| 2026-03-19 | Main adaptive preset | Validate the safer adaptive GA preset after redesign | Completed cleanly; produced a short-run final model above local baselines |
| 2026-03-19 | Phase-3 structural ablation | Test richer layered structural search | Too brittle locally; all first-generation children failed |
| 2026-03-21 | Three-seed local thesis batch | Test repeatability across seeds 11, 22, and 33 | Stable repeated runs; search improved, but final exports did not clearly beat best baseline |
| 2026-04-25 | Dependency-upgrade probe | Test upgraded package stack against the full local workflow | Completed successfully; compatibility validated |
| 2026-04-26 | Tuned seed extension, seeds 44 and 55 | Extend tuned local validation with two more seeds | Both completed; stable convergence, but final raw scores remained below best base baseline |

Detailed sources:
- [Experiment 2 Results](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/exp02_pythia70m_linear_passthrough_validation/RESULTS.md)
- [2026-03-19 Results](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260319.md)
- [2026-03-21 Batch Results](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260321_THESIS_BATCH.md)
- [2026-04-25 Dependency Probe](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260425_DEPENDENCY_PROBE.md)
- [2026-04-26 Seeds 44/55 Results](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260426_SEEDS_44_55.md)

## Main Findings

### 1. The Local GA Pipeline Is Now Stable

Across the completed local experiments, the pipeline repeatedly completed the
full workflow: baseline evaluation, candidate generation, staged scoring,
history logging, final merge export, and final comparison. The important
engineering failures from earlier work, especially tokenizer-path instability
and merge-output brittleness, were not the dominant failure mode in the finished
local experiments.

The remaining failures are mostly research-quality failures, especially
`metric_guard` rejections where a candidate collapses on SciQ. This is a much
healthier failure mode because it means the system is rejecting poor candidates
rather than crashing.

### 2. The Safer Adaptive Preset Is The Right Local Default

The 2026-03-19 main adaptive run showed that the safer adaptive preset can
complete cleanly and produce a plausible final merged result. The richer
phase-3 structural ablation failed immediately under the local budget, with all
first-generation children rejected or failing.

The conclusion is that the local default should remain conservative:
`linear` and `passthrough` are suitable for the main local thesis track, while
phase-3 structural operators should remain ablations until they are made less
brittle.

### 3. `linear` Is The Strongest Local Search Operator

The three-seed March batch showed that `linear` was the strongest operator in
the Pythia-70M setting. `passthrough` was safe and useful as a floor, but it did
not drive the best search results. `slerp` was consistently weaker and more
failure-prone in the local setting.

This supports the adaptive-operator thesis claim: merge operators are not
equally useful, and the GA benefits from learning which operators are productive
under the current model pair, task mix, and evaluation budget.

### 4. Search Fitness And Final Exported Quality Are Not Fully Aligned

The most important research limitation is objective mismatch. Multiple runs
found candidates with improved GA search fitness, but those improvements did
not reliably become final exported benchmark wins over the strongest parent.

The March three-seed batch is the clearest example:

| Seed | Best search fitness | Final exported result |
| --- | ---: | --- |
| `11` | `0.6293291450` | near parity, slightly below best baseline |
| `22` | `0.6293291450` | near parity, slightly below best baseline |
| `33` | `0.6122599244` | below best baseline |

The April tuned seed extension repeated this pattern:

| Seed | Best GA score | Final raw score | Best raw baseline |
| --- | ---: | ---: | ---: |
| `44` | `0.6245083809` | `0.6155307821` | `0.6410625533` |
| `55` | `0.6250461340` | `0.6155377933` | `0.6410625533` |

This does not invalidate the adaptive GA work. It clarifies what the local
track proves: the system can search stably and adapt operator preferences, but
the current local objective still overestimates final exported model quality.

### 5. The Search Space Still Allows Near-Parent Solutions

The winning local candidates often look like near-parent preservation or
near-parent extrapolation rather than balanced merged hybrids. In the March
batch, the strongest configs were `linear` candidates with one parent weight
near or above `1.0` and the other near `0.0`. In the April tuned extension, the
best configs also leaned heavily toward one parent.

This suggests that the next search-space improvement should constrain or
penalize degenerate linear solutions. Options include convex weight constraints,
a dominance penalty, or a minimum second-parent contribution for multi-parent
linear merges.

## Latest Larger Experiment: Seeds 44 And 55

The most recent larger run used the tuned local configuration and completed
seeds `44` and `55`.

Run folder:
[20260426-thesis-tuned-seeds-44-55](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260426-thesis-tuned-seeds-44-55)

Both seeds stopped by stagnation:

| Seed | Stop generation | Fevals | Best GA score | Final raw score |
| --- | ---: | ---: | ---: | ---: |
| `44` | `10` | `80` | `0.6245083809` | `0.6155307821` |
| `55` | `10` | `80` | `0.6250461340` | `0.6155377933` |

The run supports a stability claim. Both seeds converged to the same narrow
score band and produced all expected artifacts and plots. It does not support a
new final-benchmark win claim, because the exported final models stayed below
the `EleutherAI/pythia-70m-deduped` raw baseline.

## Generated Artifacts

The latest run includes per-seed CSVs, exported final models, and plots:

- [seed44 artifacts](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260426-thesis-tuned-seeds-44-55/seed44)
- [seed55 artifacts](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260426-thesis-tuned-seeds-44-55/seed55)

Each seed folder includes:
- `ga_history.csv`
- `ga_candidate_history.csv`
- `ga_method_history.csv`
- `ga_final_comparison.csv`
- `ga_stop_details.json`
- `ga_summary.txt`
- `best_config.yaml`
- `final_model/`
- `ga_history_plot.png`
- `ga_final_comparison.png`
- `ga_method_history_plot.png`
- `ga_candidate_scores_plot.png`
- `ga_failure_breakdown_plot.png`
- `ga_raw_score_comparison_plot.png`

## Thesis Interpretation

The local adaptive GA experiments support the following thesis statement:

The adaptive GA redesign improves the reliability, observability, and
operator-selection behavior of local model-merge search. On the Pythia-70M
validation track, the system can complete repeated runs, reject poor candidates
through metric guards, adapt toward stronger merge operators, and preserve
competitive parent behavior. However, local search-fitness gains do not yet
reliably transfer into final exported benchmark wins, indicating that the
objective function and search-space constraints require further alignment.

## Implemented Follow-Up Changes

The next local preset has been updated to reflect these findings:

1. `MultiMethodGenome` now exposes a tabular `genotype_to_param_arrays()` path,
   so MLflow and W&B can log best individuals without assuming the older
   `ModelGenome` tensor layout.
2. The local multi-method genome supports `linear_min_source_weight` and
   `linear_max_scale`, allowing the thesis configs to prevent degenerate
   near-parent linear extrapolations.
3. The main local adaptive preset now uses `linear` plus `passthrough`, raises
   stage-2 fidelity, and reduces the number of promoted candidates.
4. `slerp` and phase-3 structural operators remain explicit ablations rather
   than default local operators.

## Remaining Next Steps

1. Re-run the tuned seed-extension experiment with the constrained linear
   search space.
2. Compare the constrained results against the unconstrained seeds `44` and
   `55` to measure whether final raw comparison improves.
3. Use the local track as a stability and methodology gate before moving larger
   experiments to the GPU/EKS track.
