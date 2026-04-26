# Dependency-Probe Thesis Run: 2026-04-25

This note summarizes the local thesis experiment used to validate the
dependency-upgrade branch. The goal was not to introduce a new search result as
the main thesis benchmark, but to check whether the upgraded package set still
supports the full local adaptive-GA workflow: baseline evaluation, model merge,
tokenizer construction, staged evaluation, GA history logging, final model
export, and final comparison reporting.

## Scope

Run folder:
[20260425-dependency-probe-main-adaptive](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive)

Configuration:
[main_adaptive_pythia70m.yml](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/main_adaptive_pythia70m.yml)

Branch/worktree:
- branch: `nk_env/dependency-probe`
- worktree:
  `/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe`
- base cleanup branch: `codex/ga-evolve/adaptive-ga-phase-1`

Runtime package set:
- `transformers==4.57.6`
- `torch==2.11.0`
- `accelerate==1.13.0`
- `pydantic==2.13.3`
- `safetensors==0.7.0`
- `ray==2.55.1`
- `lm_eval==0.4.11`
- `mlflow==3.11.1`

## Command

```bash
env TOKENIZERS_PARALLELISM=false RAY_TMPDIR=/tmp/ray-thesis-dependency-probe \
  /tmp/mergekit-dep-probe-venv/bin/python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/main_adaptive_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive \
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

The run completed with exit code `0`.

## Outcome

Stop details from
[ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_stop_details.json):
- stop reason: `max_fevals`
- completed generations: `12`
- completed function evaluations: `96/96`
- elapsed time: `3164.2` seconds, about `52.7` minutes
- best generation: `10`
- best GA objective score: `0.5601950288`
- score source: `stage2`

Summary from
[ga_summary.txt](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_summary.txt):
- initial best search score: `0.5383`
- final best search score: `0.5602`
- best baseline search score: `0.5390`
- search-objective improvement over best baseline: `+0.0212`
- reported percentage improvement over best baseline: `+3.94%`

This is a successful compatibility run for the upgraded dependency set. The
full local thesis pipeline completed without import failures, merge failures,
tokenizer-path failures, or final-export failures.

## Baseline And Final Comparison

Baseline and final merged raw comparison from
[ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_final_comparison.csv):

| Model | Raw weighted score | Wikitext byte perplexity | BoolQ accuracy | SciQ accuracy |
| --- | ---: | ---: | ---: | ---: |
| `EleutherAI/pythia-70m-deduped` | `0.5389575783` | `2.1678406307` | `0.6666666667` | `0.5` |
| `lomahony/pythia-70m-helpful-sft` | `0.5322374645` | `2.3729984381` | `0.6666666667` | `0.5` |
| `final_merged` | `0.5325731682` | `2.3621212259` | `0.6666666667` | `0.5` |

The important interpretation is that the GA objective and the raw final
comparison score are not the same metric. The best GA objective score
(`0.5601950288`) includes additional objective terms used by the optimizer.
The exported final model's raw weighted comparison score (`0.5325731682`) is
competitive with the helpful-SFT parent, but it does not beat the best raw
baseline in this run.

For thesis wording, this run should therefore be framed as evidence of
pipeline compatibility and stable adaptive search under upgraded dependencies,
not as a new raw benchmark win.

## Plots

Generation-level search progress:

![GA history plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_history_plot.png)

Raw baseline and final merged comparison:

![GA final comparison plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_final_comparison.png)

Adaptive operator probabilities and success rates:

![GA method history plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_method_history_plot.png)

Candidate score distribution by generation:

![GA candidate scores plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_candidate_scores_plot.png)

Failure count and failure-type breakdown:

![GA failure breakdown plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_failure_breakdown_plot.png)

Regenerated raw weighted-score comparison:

![GA raw score comparison plot](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_raw_score_comparison_plot.png)

The first two plots were generated by the GA runner. The method-history,
candidate-score, failure-breakdown, and raw-score comparison plots were
generated afterward with
[plot_thesis_results.py](/Users/nkululekothangelane/Documents/master_research/mergekit/scripts/research/plot_thesis_results.py).

## Best Merge Configuration

Best configuration from
[best_config.yaml](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/best_config.yaml):

```yaml
base_model: EleutherAI/pythia-70m-deduped
dtype: bfloat16
merge_method: linear
parameters:
  int8_mask: 1.0
  normalize: 1.0
slices:
- merge_method: linear
  parameters:
    int8_mask: 1.0
    normalize: 1.0
  sources:
  - layer_range: [0, 6]
    model: lomahony/pythia-70m-helpful-sft
    parameters:
      weight: 0.8374437093734741
  - layer_range: [0, 6]
    model: EleutherAI/pythia-70m-deduped
    parameters:
      weight: 0.003805825486779213
tokenizer_source: base
```

The winner is a `linear` candidate strongly weighted toward
`lomahony/pythia-70m-helpful-sft`, with only a very small contribution from the
base model. It is best described as a near-parent linear solution discovered
inside the constrained local search space, rather than a balanced two-parent
blend.

The exported merged model is available in
[final_model](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/final_model).

## Operator Behavior

Method history from
[ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_method_history.csv)
shows a clear local preference pattern:
- `linear` became the strongest operator after early guard failures and ended
  with probability about `0.485`
- `passthrough` stayed safe, but its probability drifted down to about `0.270`
- `slerp` was explored repeatedly but produced only guard failures in this run,
  ending at probability about `0.245`

This supports the local thesis observation that merge-operator usefulness is
not uniform. In this small Pythia-70M setting, adaptive selection correctly
favored `linear` after it became the only consistently productive operator.

## Failure Pattern

Failure records from
[failed_genotypes.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/failed_genotypes.csv):
- total failed candidates recorded: `47`
- failure type: `metric_guard`
- dominant guard: SciQ accuracy collapsing to `0`

This is a healthy failure pattern for a compatibility experiment. The failures
were quality-filter failures from the configured evaluation guard, not package
or runtime failures. The run continued through all generations and wrote all
expected artifacts.

## Generated Artifacts

Primary artifacts:
- [baseline_results.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/baseline_results.csv)
- [ga_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_history.csv)
- [ga_candidate_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_candidate_history.csv)
- [ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_method_history.csv)
- [ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_final_comparison.csv)
- [ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_stop_details.json)
- [ga_history_plot.png](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_history_plot.png)
- [ga_final_comparison.png](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ga_final_comparison.png)
- [ray_observability.json](/Users/nkululekothangelane/Documents/master_research/mergekit-dependency-probe/workspace/thesis/local_mac/results/20260425-dependency-probe-main-adaptive/ray_observability.json)

## Thesis Interpretation

This experiment provides positive evidence for the engineering side of the
thesis pipeline. The upgraded dependency set was able to run the local adaptive
GA end to end, including staged scoring and final model export. The search
behavior remained interpretable: `linear` became the preferred operator, weak
children were rejected by metric guards, and the final winner was a stable
near-parent linear merge.

The result should not be overstated as a raw benchmark improvement. The best
GA objective score exceeded the best baseline objective by `3.94%`, but the
exported final model's raw weighted comparison score remained below the best
raw parent baseline. The defensible thesis claim is therefore:

> Under the upgraded package stack, the local adaptive-GA pipeline remained
> stable and produced a coherent best candidate over a full 96-evaluation
> Pythia-70M run. The run validates compatibility and adaptive operator
> behavior, while raw benchmark superiority over the strongest parent remains
> unproven for this specific dependency-probe experiment.

## Recommendation

Use this run as a dependency-upgrade validation artifact. Keep the existing
March thesis batch as the stronger experimental evidence for search behavior,
and cite this April run when discussing reproducibility under newer package
versions.
