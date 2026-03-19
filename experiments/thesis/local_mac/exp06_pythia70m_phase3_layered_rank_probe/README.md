# Experiment 6: Pythia-70M Phase-3 Layered Rank Probe

This is the local phase-3 validation preset.

It keeps the run on a tiny same-family pair, but it turns on the structural
features that phase 3 is supposed to prove:

- nonzero `layer_granularity` so the genome can emit real mixed-method slice
  configs
- `weighted_rank` fitness so survivor choice is not tied to one raw scalar
- a novelty archive bonus in addition to the phase-2 diversity terms
- the expanded local method family: `passthrough`, `linear`, `slerp`, `ties`,
  `dare_linear`, and `dare_ties`

Current SLERP note:
When a genotype uses `slerp` with multiple layer groups enabled, the generated
config currently expands SLERP across the full model layer range instead of
restricting it to one narrow block. That keeps the emitted config compatible
with the base architecture while the narrower per-block SLERP path remains a
future refinement.

This is still a smoke/validation preset, not a thesis-scale performance run.
The intent is to prove that the structural path works locally before spending
more resources on larger cloud experiments.
