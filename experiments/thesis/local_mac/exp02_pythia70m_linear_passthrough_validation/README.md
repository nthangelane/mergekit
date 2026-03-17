# Experiment 2: Pythia-70M Linear + Passthrough Validation

This experiment is a local validation run for the Apple M1 development machine.
It deliberately stays in the tiny-model regime: two same-family 70M parents,
CPU-only execution, serial evaluation, and a narrow merge-method search space.

## Narrative

The purpose of this run is not to prove that local hardware can carry thesis-
scale model merging. It cannot. The purpose is to validate that the GA behaves
sensibly on a constrained local problem before spending time or money on larger
cloud runs.

The experiment asks a focused question: if one parent already has the best score
under the local task mix, can the GA preserve that parent through a
`passthrough` option while still exploring small `linear` merges around it? To
make that question meaningful on local hardware, the config does four things:

1. It limits the search to `linear` and `passthrough`.
2. It keeps the population serial and CPU-only to avoid disturbing other work.
3. It reuses the base tokenizer to avoid local tokenizer-union edge cases.
4. It reweights the local fitness toward `sciq` so `wikitext` perplexity does
   not dominate the search.

This run is therefore a stability and search-behavior experiment. It is a gate
before larger EKS experiments, not a substitute for them.

## Local Guardrail

Local runs only work with tiny models. Pythia-70M is acceptable here because it
is still small enough for controlled CPU-only validation. Anything substantially
larger should move to the EKS GPU track.

## Final Result

The completed 50-generation run confirmed the local guardrail rather than
overturning it. The GA remained stable, converged toward `passthrough`, and did
not discover a merge better than the strongest parent baseline.

See [RESULTS.md](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/exp02_pythia70m_linear_passthrough_validation/RESULTS.md) for the full write-up.
