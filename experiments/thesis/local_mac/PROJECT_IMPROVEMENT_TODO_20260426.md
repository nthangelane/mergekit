# Project Improvement TODO: 2026-04-26

This checklist tracks the next cleanup and thesis-support work items. Work
through these step by step, with separate commits where behavior or review
scope changes.

## 1. Address Current Review Finding

- [x] Replace manual `Task` abstract-method clearing in
  `mergekit/merge_methods/easy_define.py` with `abc.update_abstractmethods`.
- [x] Add or adjust a focused test if the existing suite does not cover dynamic
  task instantiation.
- [x] Run focused tests for merge-method task construction.
- [ ] Keep this as a small code-review fix commit.

Progress:
- Completed code fix in `mergekit/merge_methods/easy_define.py`.
- Added `tests/test_easy_define.py`.
- Verified with:
  `PYTHONPATH=/Users/nkululekothangelane/Documents/master_research/mergekit /tmp/mergekit-dep-probe-venv/bin/python -m pytest tests/test_easy_define.py tests/test_basic_merges.py::TestBasicMerges::test_linear_merge -q`
- Result: `2 passed`, with unrelated deprecation warnings.
- Formatting note: `black` was not installed in the available temp venv, so
  formatting was checked manually.

## 2. Improve Thesis Result Visualization

- [x] Confirm the existing generated plots:
  - `ga_history_plot.png`
  - `ga_final_comparison.png`
- [x] Add a lightweight result-plotting script for the remaining structured
  CSV outputs:
  - `ga_method_history.csv`: operator probability and success-rate trends
  - `ga_candidate_history.csv`: candidate score distribution by generation
  - `failed_genotypes.csv`: failure type and guard-failure breakdown
  - `baseline_results.csv` and `ga_final_comparison.csv`: raw score comparison
- [x] Make the script accept a results directory and write plots beside the
  CSVs.
- [x] Document which plots are thesis-facing and which are debugging aids.
- [x] Re-run the script on the 2026-04-25 dependency-probe result folder.

Progress:
- Added `scripts/research/plot_thesis_results.py`.
- Generated:
  - `ga_method_history_plot.png`
  - `ga_candidate_scores_plot.png`
  - `ga_failure_breakdown_plot.png`
  - `ga_raw_score_comparison_plot.png`
- Embedded the new plots in
  `experiments/thesis/local_mac/RESULTS_20260425_DEPENDENCY_PROBE.md`.
- Verified script syntax with `python -m py_compile`.

## 3. Clean Deployment Manifest Duplicates

- [x] Decide whether the tracked `deploy/* 2.yaml` files are obsolete or legacy
  references.
- [x] Delete obsolete duplicates, or move useful legacy variants into
  `deploy/legacy/` with a short README.
- [x] Run `tests/test_run_on_eks.py` after changing deploy manifests.

Progress:
- Deleted tracked duplicate manifests:
  - `deploy/ray-cluster 2.yaml`
  - `deploy/ray-job 2.yaml`
  - `deploy/storage 2.yaml`
- Verified references only point to canonical manifest names.
- Verified with:
  `PYTHONPATH=/Users/nkululekothangelane/Documents/master_research/mergekit /tmp/mergekit-dep-probe-venv/bin/python -m pytest tests/test_run_on_eks.py -q`
- Result: `19 passed`.

## 4. Remove Local Generated Clutter

- [x] Remove local `.DS_Store` files from the working tree.
- [x] Check whether any generated local files are accidentally tracked.
- [x] Strengthen `.gitignore` if new local patterns are recurring.
- [ ] Keep this as a no-behavior-change cleanup commit.

Progress:
- Removed all local `.DS_Store` files found in the repository tree.
- Confirmed `.gitignore` already covers `.DS_Store`, `**/.DS_Store`,
  `.venv`, `workspace/`, model weights, caches, MLflow, and W&B outputs.
- Found `notebook.ipynb` is tracked; left it untouched because it is not
  ignored local clutter.

## 5. Curate Thesis Artifacts

- [x] Decide which untracked thesis files should be committed:
  - `RESULTS_20260321_THESIS_BATCH.md`
  - `RESULTS_20260425_DEPENDENCY_PROBE.md`
  - `thesis_run_pythia70m_tuned.yml`
  - `pending_experiments/`
- [x] Keep curated write-ups and configs tracked.
- [x] Keep large generated outputs under `workspace/` untracked.
- [x] Update `experiments/thesis/local_mac/README.md` after deciding what is
  canonical.

Progress:
- Decided the listed thesis write-ups, tuned config, and pending experiment
  designs should be tracked as source artifacts.
- Added `pending_experiments/README.md` to label the directory as planned
  experiment design work.
- Updated `experiments/thesis/local_mac/README.md` with the April dependency
  probe write-up, project TODO, tuned preset, and pending experiments.
- Confirmed YAML configs parse successfully.
- Kept generated result outputs under `workspace/` untracked.

## 6. Record Dependency Upgrade Boundaries

- [x] Document that the current validated dependency-probe stack uses
  `transformers==4.57.6`.
- [x] Record that Transformers `>5` needs a separate compatibility branch due
  to the observed Pydantic forward-reference issue.
- [x] Keep this documentation separate from the cleanup PR if it discusses
  compatibility behavior changes.

Progress:
- Added `DEPENDENCY_COMPATIBILITY_NOTES_20260426.md`.
- Linked it from `experiments/thesis/local_mac/README.md`.
- Documented the validated Transformers 4.x stack and the separate
  Transformers 5 follow-up boundary.

## 7. Add Reproducible Thesis Summary Generation

- [x] Add a script that reads:
  - `ga_summary.txt`
  - `ga_stop_details.json`
  - `ga_final_comparison.csv`
  - `ga_method_history.csv`
  - `failed_genotypes.csv`
- [x] Generate a Markdown summary scaffold from a result directory.
- [x] Include the important interpretation note distinguishing GA objective
  score from raw final comparison score.
- [x] Add a tiny fixture-based test for the summary generator.

Progress:
- Added `scripts/research/write_thesis_result_summary.py`.
- Added `tests/test_thesis_summary.py`.
- Verified fixture test with:
  `PYTHONPATH=/Users/nkululekothangelane/Documents/master_research/mergekit /tmp/mergekit-dep-probe-venv/bin/python -m pytest tests/test_thesis_summary.py -q`
- Result: `1 passed`.
- Generated a real summary for the dependency-probe result folder at
  `generated_thesis_summary.md` under that result directory.

## 8. Final Verification Pass

- [x] Run focused tests after each code change.
- [x] Run full import checks after cleanup and docs changes.
- [x] Run the full test suite before opening or updating the PR.
- [x] Confirm `git status` contains only intended files before staging.

Progress:
- Ran syntax checks with `python -m py_compile` for touched scripts and tests.
- Ran focused verification:
  `PYTHONPATH=/Users/nkululekothangelane/Documents/master_research/mergekit /tmp/mergekit-dep-probe-venv/bin/python -m pytest tests/test_easy_define.py tests/test_thesis_summary.py tests/test_run_on_eks.py tests/test_basic_merges.py::TestBasicMerges::test_linear_merge -q`
- Result: `22 passed`.
- First full-suite run found a real compatibility issue in
  `mergekit/evo/task_utils.py`; fixed logger initialization for newer
  `lm_eval` task indexing.
- Re-ran full suite:
  `PYTHONPATH=/Users/nkululekothangelane/Documents/master_research/mergekit /tmp/mergekit-dep-probe-venv/bin/python -m pytest -q`
- Result: `204 passed`, with `4` unrelated deprecation warnings.
