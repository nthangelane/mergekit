# Local Thesis Batch Results: 2026-03-21

This note summarizes the completed three-seed local pooled thesis batch for the
adaptive GA on the tiny Pythia-70M pair.

Batch artifacts:
[20260321-thesis-pool-batch](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch)

Seeds:
- [seed11](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11)
- [seed22](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22)
- [seed33](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33)

## Setup

All three runs used:
- [`thesis_run_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m.yml)
- `pool` strategy with `2` workers
- the parent pair `lomahony/pythia-70m-helpful-sft` and
  `EleutherAI/pythia-70m-deduped`
- `passthrough`, `linear`, and `slerp`
- `layer_granularity: 0`
- stop policy with `max_fevals: 192`, `target_improvement_pct: 5.0`, and
  stagnation stopping

The best baseline was the same in all three seeds:
- `EleutherAI/pythia-70m-deduped`: `0.6033623793`
- `lomahony/pythia-70m-helpful-sft`: `0.5777276147`

Baseline source:
[seed11 baseline_results.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/baseline_results.csv)

## Search Summary

| Seed | Best Search Fitness | Delta Vs Best Baseline | Best Generation | Stop Generation | Fevals | Stop Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `seed11` | `0.6293291450` | `+4.30%` | `5` | `13` | `104` | `stagnation` |
| `seed22` | `0.6293291450` | `+4.30%` | `3` | `11` | `88` | `stagnation` |
| `seed33` | `0.6122599244` | `+1.47%` | `9` | `12` | `96` | `stagnation` |

Stop records:
- [seed11 ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/ga_stop_details.json)
- [seed22 ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/ga_stop_details.json)
- [seed33 ga_stop_details.json](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/ga_stop_details.json)

Interpretation:
- the adaptive GA consistently found competitive candidates quickly
- two seeds reached the same best search fitness
- none of the three seeds reached the configured `+5%` early-stop target
- all three seeds terminated on stagnation rather than timeout or crash

## Final Exported Model Comparison

The key caution is that the best search fitness did not translate into a clear
final exported benchmark win.

| Seed | Final Exported Weighted Score | Delta Vs Best Baseline | Result |
| --- | ---: | ---: | --- |
| `seed11` | `0.6032261825` | `-0.0226%` | near parity, slightly below baseline |
| `seed22` | `0.6032261825` | `-0.0226%` | near parity, slightly below baseline |
| `seed33` | `0.5812343054` | `-3.67%` | below baseline |

Comparison sources:
- [seed11 ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/ga_final_comparison.csv)
- [seed22 ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/ga_final_comparison.csv)
- [seed33 ga_final_comparison.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/ga_final_comparison.csv)

This is the most important result of the batch:
- the redesigned GA improved search behavior and stability
- but the saved best models did not beat the strongest parent on the final
  exported weighted benchmark

The search objective is therefore still optimistic relative to the final
evaluation path.

## Winner Shape

All three winning configs were `linear`, but the strongest two were not true
two-parent blends.

Best configs:
- [seed11 best_config.yaml](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/best_config.yaml)
- [seed22 best_config.yaml](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/best_config.yaml)
- [seed33 best_config.yaml](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/best_config.yaml)

Observed structure:
- `seed11`: `deduped` weight `1.1682`, `helpful-sft` weight `0.0`
- `seed22`: `deduped` weight `1.1731`, `helpful-sft` weight `0.0`
- `seed33`: `deduped` weight `0.9552`, `helpful-sft` weight `0.0273`

Interpretation:
- the search repeatedly converged to near-parent extrapolations around the
  stronger `deduped` parent
- the current local search space is still allowing solutions that behave more
  like amplified parent preservation than like useful merged hybrids

## Operator Behavior

Aggregate operator usage across all three seeds:

| Method | Uses | Successes | Failures | Success Rate | Best Search Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear` | `116` | `109` | `7` | `93.97%` | `0.6293291210` |
| `passthrough` | `61` | `61` | `0` | `100.00%` | `0.5838305559` |
| `slerp` | `111` | `91` | `20` | `81.98%` | `0.3694414123` |

Operator evidence:
- [seed11 ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/ga_method_history.csv)
- [seed22 ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/ga_method_history.csv)
- [seed33 ga_method_history.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/ga_method_history.csv)

Interpretation:
- `linear` is the strongest operator for this local 70M setup
- `passthrough` is a safe floor, but it is not the best search operator
- `slerp` is clearly weaker and more failure-prone on this pair

## Failure Pattern

The dominant failures were evaluation-quality failures, not pipeline crashes.

Failure files:
- [seed11 failed_genotypes.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/failed_genotypes.csv)
- [seed22 failed_genotypes.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/failed_genotypes.csv)
- [seed33 failed_genotypes.csv](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/failed_genotypes.csv)

Observed pattern:
- failures were `metric_guard` rejections on `sciq`
- there was no recurrent tokenizer crash pattern
- there was no repeated merge-pipeline crash pattern in the finished local batch

This is a healthier research failure mode than the earlier unstable runs.

## Plots

Per-seed search plots:
- ![seed11 history](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/ga_history_plot.png)
- ![seed22 history](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/ga_history_plot.png)
- ![seed33 history](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/ga_history_plot.png)

Per-seed final comparison plots:
- ![seed11 final comparison](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/ga_final_comparison.png)
- ![seed22 final comparison](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/ga_final_comparison.png)
- ![seed33 final comparison](/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/ga_final_comparison.png)

## Thesis Interpretation

The completed three-seed local batch supports the following claims:
- the adaptive GA framework is stable enough to run repeated pooled local
  experiments on the tiny Pythia-70M pair
- operator behavior is not uniform; `linear` is materially stronger than
  `slerp` on this setup
- the framework avoids immediate passthrough collapse
- the best local search results are reproducible across at least two seeds
- the local search objective is still misaligned with the final exported model
  evaluation, because search-fitness gains did not become final benchmark wins

The correct thesis phrasing is therefore:

The adaptive GA redesign improved local search behavior, observability, and
operator adaptation on the tiny Pythia-70M track, but the best candidates found
by the optimizer remained near-parent linear extrapolations and did not produce
a final exported model that clearly beat the strongest parent baseline.

## Parameter Adjustments For The Next Local Run

The current batch suggests four changes.

### Config-level changes

1. Raise stage-2 fidelity and reduce promotions.
   - Change `stage2_limit` from `12` to `24` or `32`
   - Change `stage2_top_k` from `4` to `2`
   - Rationale: the current search score is too optimistic relative to final
     exported evaluation

2. Narrow the operator set for the main local thesis run.
   - Move `slerp` out of the default thesis preset
   - Keep `linear` and `passthrough` in the main run
   - Keep `slerp` as an ablation
   - Rationale: `slerp` contributed many evaluations and failures while showing
     consistently weak best scores

3. Bias adaptive sampling harder toward `linear`.
   - Example next-run probabilities:
     - `linear: 0.75`
     - `passthrough: 0.25`
   - Rationale: `linear` was the only operator that found the best region

4. Reduce diversity bonus weights slightly.
   - Lower `gene_diversity_bonus_weight`
   - Lower `behavior_diversity_bonus_weight`
   - Rationale: the search objective should track final benchmark quality more
     closely once collapse is under control

### Code-level changes

The most important remaining issue is not configurable today.

The current search space allows `linear` winners with:
- parent weights above `1.0`
- a second parent weight of `0.0`

This means the optimizer can win by finding amplified parent extrapolations.

The next code-level search-space change should therefore be one of:
- constrain linear weights to a convex region
- add a penalty when one parent weight dominates beyond a threshold
- require a minimum non-zero contribution from the second parent for
  multi-parent `linear` candidates

Without that change, the search can continue to report improvements that are
not really merged-model improvements in the thesis sense.
