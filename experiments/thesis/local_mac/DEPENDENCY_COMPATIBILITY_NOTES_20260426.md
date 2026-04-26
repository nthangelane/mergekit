# Dependency Compatibility Notes: 2026-04-26

This note records the dependency boundary established during the package-upgrade
probe and the follow-up thesis experiment.

## Validated Stack

The local adaptive-GA thesis experiment completed successfully with:

- `transformers==4.57.6`
- `torch==2.11.0`
- `accelerate==1.13.0`
- `pydantic==2.13.3`
- `safetensors==0.7.0`
- `ray==2.55.1`
- `lm_eval==0.4.11`
- `mlflow==3.11.1`

Validation artifact:
[RESULTS_20260425_DEPENDENCY_PROBE.md](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/RESULTS_20260425_DEPENDENCY_PROBE.md)

## Current Transformer Boundary

The validated branch should stay on the Transformers 4.x line for now. The
successful thesis run used `transformers==4.57.6`, and that is the highest
tested working version in this local compatibility pass.

## Transformers 5 Follow-Up

A temporary experiment with Transformers 5 showed import progress, but focused
tests hit a Pydantic forward-reference issue involving configured architecture
models. That should be handled as a separate compatibility branch because it is
not just a dependency pin update; it requires code changes and compatibility
tests.

Recommended follow-up:

- Create a dedicated Transformers 5 compatibility branch.
- Reproduce the Pydantic forward-reference failure in a focused test.
- Apply the smallest model-rebuild or schema-resolution fix.
- Re-run import checks, focused architecture tests, and the local GA smoke
  path before considering a wider package bound.

## Thesis Framing

For thesis documentation, cite the April dependency-probe run as evidence that
the local adaptive-GA pipeline remains reproducible on a newer Transformers 4.x
stack. Do not claim Transformers 5 support until the separate compatibility
branch passes focused and end-to-end checks.
