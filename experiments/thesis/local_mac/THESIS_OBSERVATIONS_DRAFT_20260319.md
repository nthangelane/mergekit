# Thesis Observations Draft: 2026-03-19

This note captures the current thesis-relevant observations from the local
Pythia-70M adaptive-GA track. It separates completed evidence from provisional
evidence because the longer multi-seed thesis batch is only partially complete:
`seed11` is finished, `seed22` is running, and `seed33` has not started yet.

## Scope

The local track is not intended to prove final absolute model quality. Its role
is to answer a narrower research question: can the adaptive GA search safely
explore tiny-model merges, preserve strong baselines, and produce evidence that
operator behavior changes over time under explicit diversity and stability
controls?

The key local artifacts at this point are:
- completed short adaptive validation in
  [exp05_pythia70m_phase2_run1](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-phase2-phase3/exp05_pythia70m_phase2_run1)
- ongoing pooled thesis batch in
  [20260319-thesis-pool-batch](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch)

## Completed Evidence

The short adaptive validation run completed cleanly and remains the strongest
completed local result so far.

Run:
[exp05_pythia70m_phase2_run1](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-phase2-phase3/exp05_pythia70m_phase2_run1)

Confirmed outcomes:
- `2` generations and `8` function evaluations completed
- best generation score: `0.5957933664`
- final merged comparison score: `0.5579598016`

Baseline comparison from
[baseline_results.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-phase2-phase3/exp05_pythia70m_phase2_run1/baseline_results.csv):
- `EleutherAI/pythia-70m-deduped`: `0.5389579233`
- `lomahony/pythia-70m-helpful-sft`: `0.5322375976`

Interpretation:
- the safer adaptive preset is mechanically stable on the tiny local setup
- the completed short run beat both parent baselines on the configured local
  objective
- the search did not collapse immediately into passthrough-only behavior

This is the strongest completed result currently available for the thesis
chapter on local validation.

## Evidence From The Longer Thesis Batch

The longer pooled thesis batch is now partially complete. `seed11` has finished
and provides the first long-run result, but the full three-seed batch is still
in progress.

Active batch:
[20260319-thesis-pool-batch](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch)

Completed seed:
[seed11](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11)

Final `seed11` outcome:
- stopped at generation `13` after `104/192` function evaluations in
  [ga_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_history.csv)
- best search fitness so far: `0.6293291450`
- this best fitness first appeared in generation `5` and persisted to the stop
  condition
- stop reason: `stagnation` in
  [ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_stop_details.json)

Baseline comparison from
[baseline_results.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/baseline_results.csv):
- `EleutherAI/pythia-70m-deduped`: `0.6033623793`
- `lomahony/pythia-70m-helpful-sft`: `0.5777276147`

At first glance the current best search fitness is about `4.30%` above the
best baseline. However, that number requires careful interpretation.

## Important Interpretation Note

The long thesis run optimizes a search fitness, not a pure benchmark score.
The candidate history and final comparison show that the current best
individual reached:
- search fitness: `0.6293291210`
- raw stage-2 weighted score: `0.6032261798`
- final merged weighted score after model export and comparison:
  `0.6032261825`

The source row is in
[ga_candidate_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_candidate_history.csv).
The exported-model comparison is in
[ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_final_comparison.csv).

This means:
- the best candidate is currently leading on the GA's internal search objective
- but its raw stage-2 weighted score is still about `0.02%` below the best
  baseline (`0.6033623793`)
- the present lead comes from the search fitness bonuses and controls, rather
  than from a clean raw weighted-score improvement

So the correct thesis wording at this point is:
- the longer run is showing encouraging search behavior
- the GA is finding candidates that are competitive with the best parent
- the run has not yet established a clear raw benchmark win over the best
  baseline in the longer thesis setting

That distinction matters and should be made explicit in the thesis tables.

## Operator Behavior

The method history already shows useful operator-learning behavior in the live
run.

Evidence from
[ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_method_history.csv)
through the completed `seed11` run:
- `linear`: `29` uses, `29` successes, best observed score `0.6293291210`
- `passthrough`: `16` uses, `16` successes, best observed score
  `0.5838305559`
- `slerp`: `35` uses, `30` successes, `5` failures, best observed score
  `0.3694414123`

Interpretation:
- `linear` is currently the most productive operator under this objective
- `passthrough` remains useful as a safe baseline-preservation operator, but it
  is not dominating the population
- `slerp` is being explored heavily, but on this seed it has been weaker and
  more failure-prone than `linear`
- the adaptive probabilities are now consistent with this pattern: `linear`
  remains near `0.45`, `passthrough` near `0.32`, and `slerp` has drifted down
  to about `0.22`

There is also an important nuance in the current winner. The saved best config
in
[best_config.yaml](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/best_config.yaml)
is labeled `linear`, but the second parent currently has weight `0.0` while
the first parent has weight greater than `1.0`. So the strongest candidate so
far is not yet a convincing two-parent blend. It is better described as a
near-parent extrapolation found inside the linear search space.

This is already thesis-relevant because it supports the claim that merge
operator usefulness is not uniform and can be learned adaptively.

## Stability Observations

The current failure mode is narrower than in earlier local experiments.

Evidence from
[failed_genotypes.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/failed_genotypes.csv):
- `5` failures so far
- all `5` were `eval:metric_guard`
- each failure was caused by `sciq` accuracy collapsing to `0`

Interpretation:
- the current run is not primarily failing because of merge crashes or
  tokenizer-path regressions
- the dominant failure mode is poor model quality on the guard task
- this is a healthier failure pattern for research, because it means the search
  is being filtered by evaluation quality rather than pipeline instability

## Convergence Observation

The run currently appears to have reached an early plateau.

Evidence from
[ga_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed11/ga_history.csv):
- the best score improved through generations `1`, `2`, `3`, and `5`
- no further improvement has been observed from generation `5` through
  generation `10`

Interpretation:
- the adaptive GA found a strong local region relatively quickly
- subsequent exploration has not yet produced a raw benchmark win over the best
  parent
- this is the exact kind of behavior the new stop-policy feature is intended to
  manage in longer thesis runs

## What Can Be Claimed Now

The following statements are already supported:
- the redesigned adaptive GA can run in pooled local mode with explicit MLflow
  and Ray observability
- the local adaptive search remains stable on tiny Pythia-70M runs
- operator behavior differs in practice, with `linear` currently stronger than
  `slerp` on the active seed
- passthrough is no longer the only safe behavior
- the longer run has reached near-baseline parity on raw stage-2 score while
  exceeding baseline on the search fitness used by the optimizer
- the strongest candidate so far still behaves more like a parent-preserving
  extrapolation than a genuinely mixed two-parent merge

## Remaining Batch Status

The next seed has started but is progressing slowly.

Current run:
[seed22](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/seed22)

Current state:
- the first `seed22` attempt stalled during first-merge cache refill and was
  restarted
- the restarted run now uses a shared warm cache in
  [shared_transformers_cache](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260319-thesis-pool-batch/shared_transformers_cache)
- the restart is progressing through baseline normally instead of appearing
  idle in file-download startup

This is an operational observation rather than a thesis result, but it matters
for interpreting wall-clock time on local pooled runs.

## What Should Wait Until The Batch Finishes

The following claims should remain provisional until seeds `22` and `33`
complete and the winners are benchmarked at higher fidelity:
- that the adaptive GA reliably beats the best parent on the long thesis run
- that the improvement is repeatable across seeds
- that the stop-policy target is routinely achievable
- that the operator ranking is stable across repeated runs

## Recommended Final Thesis Framing

If the later seeds behave similarly, the thesis should frame the local result
as follows:

The adaptive GA redesign improved search behavior and observability on tiny
local model merges. The search no longer collapsed immediately into
passthrough-only solutions, operator preferences shifted over time, and the
strongest candidates approached or exceeded parent baselines depending on the
scoring layer being examined. On the local Pythia-70M track, `linear` merging
appeared more reliable than `slerp`, while failures were dominated by benchmark
guard rejections rather than merge-pipeline crashes.
