# Experiment 2.7: Native Random-Search Baseline

## Purpose

Test whether the adaptive GA finds better merges than independent uniform random
sampling under the same evaluation budget and two-stage protocol.

- Null hypothesis: GA and random search have equal best-found fitness at `N=96`.
- Alternative hypothesis: GA has higher best-found fitness at `N=96`.
- Seeds: `11`, `22`, and `33`.

## Protocol Parity

The baseline uses `mergekit.scripts.evolve_ga --random-search 96`. This mode
samples directly from the production genome and bypasses selection, crossover,
mutation, elitism, and adaptive operator updates. It still uses the production
evaluator, cache, candidate CSV, lineage report, and final-model path.

The preset matches the comparison GA on:

- parent models and search space
- task names, weights, metrics, and limits
- population cadence of `8`
- two-stage screening with `stage2_top_k: 2` per batch
- baseline evaluation and random seed

The configured GA operators are ignored in random-search mode. The GA block is
kept because `ga.population_size` controls batch cadence, which determines how
often Stage 2 promotion occurs. With `96` samples, both methods therefore run 12
batches and promote at most 24 candidates to Stage 2.

## Inputs

- Config: `thesis_random_search_baseline.yml`
- Runner: `run_random_search_baseline.sh`
- Comparison config: `../thesis_run_pythia70m.yml`

The lineage policy is `warn` because the current thesis pair is intentionally
retained for historical comparability even though its parent lineage is
disconnected. Every run must contain `parent_lineage.json` recording that fact.

## Run

From the repository root:

```bash
experiments/thesis/local_mac/pending_experiments/run_random_search_baseline.sh
```

Useful overrides:

```bash
SEEDS="11" SAMPLES=8 STRATEGY=serial \
  experiments/thesis/local_mac/pending_experiments/run_random_search_baseline.sh
```

The default pool strategy requires Ray. The serial override is the lightweight
CPU path and does not initialize Ray.

## Required Artifacts

Each `workspace/thesis/local_mac/results/random_search_baseline/seedNN` directory
must contain:

- `parent_lineage.json`
- `ga_candidate_history.csv` with exactly 96 candidate rows
- `ga_history.csv` with 12 batch rows
- `ga_stop_details.json` with `final_stop.reason: random_search_complete` and
  `final_stop.fevals: 96`
- `best_config.yaml` or the equivalent layered plan
- `final_model/` when final-model export is enabled

The candidate history must include genotype, exact genotype hash, merge method,
Stage 1 score, Stage 2 score or skip marker, and any repair metadata.

## Analysis

For each seed, extract best fitness and the cumulative best trajectory from
`ga_candidate_history.csv`. Compare random search with GA at the same cumulative
evaluation count.

Report:

- mean, standard deviation, and confidence interval of best fitness
- one-tailed Welch t-test for `GA > random search`
- effect size and the raw per-seed values
- Stage 2 promotion counts and failed-candidate counts
- wall-clock duration and peak resource usage

With only three seeds per method, emphasize effect sizes and raw observations;
a non-significant result is not strong evidence of equality.

## Pre-Run Gate

Do not start the three-seed campaign until the exact commit passes
`experiments/thesis/PRE_AWS_RUN_GATE.md`, including the native random-search CPU
smoke. Record the commit SHA in the experiment notes and do not modify code
between local validation and the cloud run.
