# Experiment 2 Results: Pythia-70M Linear + Passthrough Local Validation

## Summary

This experiment was a local validation run on the Apple M1 development machine.
It was deliberately constrained to tiny same-family models and a narrow search
space so that the GA could be tested safely on CPU without interfering with
other work.

The run completed successfully, but it did not find a merge better than the
strongest parent. Its main value was methodological: it confirmed that the
hardened local GA path can preserve a good baseline, avoid the earlier
tokenizer-path failures, and complete a longer serial run cleanly.

## Research Question

Can a constrained local GA run preserve a strong parent baseline while
exploring a small `linear` + `passthrough` search space, without tokenizer
instability or merge-path failures?

## Experimental Setup

The run used the tracked config in
[config.yml](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/exp02_pythia70m_linear_passthrough_validation/config.yml)
with these effective settings:

- Parent models: `lomahony/pythia-70m-helpful-sft` and `EleutherAI/pythia-70m-deduped`
- Search methods: `linear` and `passthrough`
- Tokenizer policy: `tokenizer_source: base`
- Baseline evaluation: enabled
- Execution mode: serial, CPU-only, no merge CUDA
- Population: `12`
- Generations: `50`
- Max function evaluations: `600`
- Runtime limit override: `--limit 4`
- Batch size: `1`
- Random seed: `42`

The fitness function was:

`score = 0.8 * sciq_acc - 0.2 * wikitext_byte_perplexity`

This weighting intentionally favored `sciq` so that local scoring would be less
dominated by `wikitext` perplexity.

## Why This Was a Local-Only Experiment

This was not a thesis-scale merge search. It was a local preflight. The
experiment was intentionally restricted to Pythia-70M because local runs only
remain practical and safe on tiny models. The goal was to validate mechanics,
not to claim that a laptop-class machine is an appropriate target for larger
model merging.

Anything materially larger than this setup should move to the EKS GPU track.

## Run Artifacts

- Run directory:
  [pythia70m-linear-passthrough-base-pop12-g50-20260315-2149](/tmp/mergekit-ga-runs/pythia70m-linear-passthrough-base-pop12-g50-20260315-2149)
- GA history:
  [ga_history.csv](/tmp/mergekit-ga-runs/pythia70m-linear-passthrough-base-pop12-g50-20260315-2149/ga_history.csv)
- Best config:
  [best_config.yaml](/tmp/mergekit-ga-runs/pythia70m-linear-passthrough-base-pop12-g50-20260315-2149/best_config.yaml)
- Baseline results:
  [baseline_results.csv](/tmp/mergekit-ga-runs/pythia70m-linear-passthrough-base-pop12-g50-20260315-2149/baseline_results.csv)

## Final Outcome

The run started on March 15, 2026 at `21:57:52 SAST` and finished on March 16,
2026 at `00:33:07 SAST`, for a wall-clock duration of about `2h 35m`.

Final headline results:

- Completed `50/50` generations
- Completed `600/600` evaluations
- Best score: `-0.0777601577`
- Total guarded failures: `3`
- Total cache hits: `266`
- Final run-directory size: about `431 MB`

The best configuration at the end of the run was:

```yaml
dtype: bfloat16
merge_method: passthrough
models:
  - model: lomahony/pythia-70m-helpful-sft
tokenizer_source: base
```

In other words, the best answer the search found was to keep the stronger
parent unchanged.

## Baseline Comparison

Baseline evaluation produced these reference scores:

- `lomahony/pythia-70m-helpful-sft`: `-0.0777601577`
- `EleutherAI/pythia-70m-deduped`: `-0.2374256055`

Relative to baseline:

- The final GA best matched the strongest parent baseline exactly
- The final GA best beat the weaker deduped base by `+0.1596654477`
- The GA never found a merged model better than the strongest parent

This means the experiment succeeded at baseline preservation but failed as a
search for a superior merge.

## Search Behavior

The run converged strongly toward `passthrough`.

Across all 600 evaluation slots recorded in generation summaries:

- `passthrough` appeared `517` times
- `linear` appeared `83` times

That behavior is consistent with the score landscape observed locally. Under
this small task mix and low evaluation limit, most linear combinations were
worse than simply preserving `lomahony/pythia-70m-helpful-sft`.

The back half of the run reinforced that conclusion. The global best score did
not improve beyond `-0.0777601577`, and many later generations either converged
fully to `passthrough` or reintroduced a small number of `linear` candidates
that failed to improve the frontier.

## Stability Outcome

From an engineering perspective, the run was successful.

- The hardened tokenizer path held for the full run
- The earlier tokenizer-fallback failures did not reappear
- There were no merge-path crashes in the completed experiment
- The only recorded failures were `metric_guard` rejections from poor candidates

This was the main practical purpose of the local experiment. It showed that the
local GA path is stable enough to act as a validation gate before more expensive
EKS runs.

## Interpretation

The result is not that the local experiment "failed" in every sense. It failed
to discover a better merge, but it succeeded in answering the narrower question
it was designed to answer.

The findings are:

1. `--baseline` and `passthrough` were necessary. Without them, the search had a
   strong tendency to wander into obviously worse merges.
2. Reusing the base tokenizer was the right local choice. It reduced avoidable
   tokenizer noise and let the run finish cleanly.
3. On this tiny local setup, the search pressure favored preserving a strong
   parent over blending parents.
4. The local Mac track is useful as a stability gate, not as a serious search
   environment for better merges.

## Limitations

This write-up should be read with the experiment's constraints in mind:

- The run used only two parents
- The search methods were intentionally narrow
- The evaluation budget was low enough to remain laptop-friendly
- The runtime used `--limit 4`, so metric estimates were noisy by design
- The task mix was intentionally reweighted for local practicality

Because of those limits, this result should not be generalized into a claim
that model merging is ineffective. It only shows that this particular local
search setup did not beat the best parent.

## Recommendation

Keep this experiment as a local validation template, not as a benchmark for
merge quality. It is useful for checking that:

- baseline preservation works
- tokenizer handling is stable
- GA mutations and crossover are not crashing
- the local environment can complete a bounded serial run

For actual merge discovery, move to the EKS track with a larger evaluation
budget, broader task coverage, and models that are too large for safe local
iteration.
