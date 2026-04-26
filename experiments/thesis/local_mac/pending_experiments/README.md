# Pending Local Thesis Experiments

This directory contains planned follow-up experiments for the local Pythia-70M
thesis track. These files are intentionally tracked as experiment designs and
candidate runbooks, while generated outputs should remain under `workspace/`.

## Experiment Groups

- `EXPERIMENT_SPEC_AOS_ABLATION.md`: sensitivity of adaptive operator
  selection to credit weighting.
- `EXPERIMENT_SPEC_POP_SENSITIVITY.md`: population-size trade-off under a
  fixed evaluation budget.
- `EXPERIMENT_SPEC_RANDOM_BASELINE.md`: random-search control for GA value.
- `EXPERIMENT_SPEC_POSTHOC_EVAL.md`: held-out benchmark evaluation for final
  merged models.
- `EXPERIMENT_SPEC_SEEDS_44_55.md`: extended five-seed validation campaign.

## Configs

- `thesis_aos_current.yml`
- `thesis_aos_equal_weights.yml`
- `thesis_aos_fitness_only.yml`
- `thesis_pop4_sensitivity.yml`
- `thesis_pop8_sensitivity.yml`
- `thesis_pop16_sensitivity.yml`
- `thesis_pop32_sensitivity.yml`
- `thesis_random_search_baseline.yml`
- `thesis_run_seed44.yml`
- `thesis_run_seed55.yml`

## Launchers

- `run_aos_ablation.sh`
- `run_pop_sensitivity.sh`
- `run_posthoc_eval.sh`
- `run_random_search_baseline.sh`
- `run_seeds_44_55.sh`

Before running these launchers on a new machine, review any local path
assumptions and prefer repository-relative paths where possible.
