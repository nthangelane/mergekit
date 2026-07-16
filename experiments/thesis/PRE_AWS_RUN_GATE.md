# Pre-AWS Evo Run Gate

Do not start a scaled AWS experiment until this gate passes on the exact commit
that will be deployed. A passing unit suite is necessary but not sufficient: the
gate also executes a real Pythia-70M CPU run through the production CLI.

## One-Command Gate

From the repository root, after committing and pushing the branch:

```bash
scripts/preflight_evo_aws.sh
```

The script refuses a dirty worktree, requires the local commit to equal its
upstream commit, runs the full test suite, then performs a two-candidate serial
CPU smoke with the exact thesis parent pair.

The smoke exercises:

- Hugging Face parent-lineage lookup and `parent_lineage.json`
- source-model architecture validation
- real source-model loading and single-shard safetensors conversion
- CPU execution without Ray initialization
- native seeded random sampling
- the `v2` log-reciprocal fitness definition and `fitness_definition.json`
- two-stage evaluation and Stage 2 top-K promotion
- gated parent-distillation repair on a WikiText train slice
- per-merge free-disk checks and scratch-directory reuse
- fsynced JSON-lines progress monitoring with elapsed time and free disk
- candidate and generation CSV logging
- final winner merge, repair replay, and comparison evaluation
- validation of every Experiment 2.6-2.10 campaign preset

The script validates the required artifacts and exits non-zero on any mismatch.
Results are written beneath `workspace/thesis/preflight/` and are ignored by Git.

## Automated Failure Tests

The full suite includes deterministic checks for paths that are awkward to
exercise manually:

```bash
pytest -q \
  tests/test_evo_checkpoint.py \
  tests/test_evo_resources.py \
  tests/test_random_search.py \
  tests/test_evo_repair.py \
  tests/test_parent_provenance.py \
  tests/test_finetune_baseline.py \
  tests/test_campaign.py
```

These cover interrupted/resumed equivalence, low-disk abort state, seed
reproducibility, repair gating and restoration, repair-split contamination, and
lineage warning escalation.

The fine-tuning tests include a real 50-step LoRA CPU run against a generated
tiny GPT-NeoX checkpoint, while campaign tests validate all pinned protocols and
the Ctrl-C/rerun resume decision.

## Promotion Checklist

- Record the passing commit SHA and preflight output directory.
- Confirm `parent_lineage.json` contains the expected disconnected-lineage
  warning for the historical thesis pair.
- Confirm `fitness_definition.json` records `fitness_version` as `v2` and
  `lower_is_better_transform` as `log_reciprocal`.
- Confirm `ga_candidate_history.csv` has two rows with genotype and exact hash.
- Confirm `ga_stop_details.json` records `final_stop.reason` as
  `random_search_complete` and `final_stop.fevals` as `2`.
- Confirm `final_repair.json` and `final_model/` exist.
- Confirm `progress.log` ends with `run_finished` and `status: success`.
- Build the AWS image from that same SHA; do not patch code in the cluster.
- Start with one AWS worker and `--random-search 2` before increasing workers or
  evaluation budget.
- Preserve the smoke and scaled-run directories as immutable evidence.

Passing this gate removes known local code-path failures. It cannot prove that
AWS IAM, networking, quotas, image contents, or node storage are correct, so the
single-worker AWS smoke remains mandatory before the full campaign.
