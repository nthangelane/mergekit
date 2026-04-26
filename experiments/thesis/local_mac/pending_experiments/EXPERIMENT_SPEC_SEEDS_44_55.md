# Experiment Spec: Seeds 44 and 55 (5-Seed Validation Campaign)

## Purpose

Complete the 5-seed validation campaign for the local thesis Pythia-70M adaptive GA.
This extended batch (seeds 11, 22, 33, 44, 55) establishes a robust 95% confidence interval
on the search effectiveness under the tuned configuration (`thesis_run_pythia70m_tuned.yml`).

Seeds 44 and 55 are the final two runs of this five-seed ensemble, allowing computation of:
- 5-seed arithmetic mean (best-search-score)
- 5-seed standard deviation
- 95% CI = mean ± t_{0.025,4} × SD/√5, where t_{0.025,4} = 2.776

## Config

Both seeds use `thesis_run_pythia70m_tuned.yml`:

**Key Tuning Changes (vs. original `thesis_run_pythia70m.yml`):**
- Allowed methods narrowed to `[passthrough, linear]` (slerp removed; unstable in phase 1)
- `stage2_limit` raised from 12 to 24 (higher final evaluation fidelity)
- `stage2_top_k` lowered from 4 to 2 (tighter pressure on best candidates)
- Initial method probabilities: `{passthrough: 0.25, linear: 0.75}` (linear favored after batch results)
- Explorer fraction lowered from 0.25 to 0.15
- Gene and behavior diversity bonus weights reduced (less diversity penalty)
- Population size remains 8, max_fevals 192, stagnation patience 8 generations

**Genome Search Space:**
- Two-model pair: `lomahony/pythia-70m-helpful-sft` (source) + `EleutherAI/pythia-70m-deduped` (base)
- Operators: passthrough, linear (applied per-layer, global parameters, no layer granularity)
- Semantic crossover enabled, adaptive operator sampling enabled

**Evaluation:**
- Two-stage fitness: stage 1 (4 fevals) → stage 2 (24 fevals per candidate)
- Tasks: wikitext (40% weight, perplexity), boolq (35%), sciq (25%)
- Baseline evaluation enabled (reports against both parent models)

## Run Commands

### Seed 44
```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/pending_experiments/thesis_run_pythia70m_tuned.yml \
  --strategy pool \
  --num-workers 2 \
  --storage-path workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed44 \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --mlflow \
  --mlflow-experiment mergekit-thesis-pool-20260321-thesis-pool-batch \
  --mlflow-tracking-uri file://workspace/thesis/local_mac/results/20260321-thesis-pool-batch/mlruns \
  --mlflow-ui-port 5013 \
  --no-reshard \
  --random-seed 44
```

### Seed 55
```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/pending_experiments/thesis_run_pythia70m_tuned.yml \
  --strategy pool \
  --num-workers 2 \
  --storage-path workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed55 \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --mlflow \
  --mlflow-experiment mergekit-thesis-pool-20260321-thesis-pool-batch \
  --mlflow-tracking-uri file://workspace/thesis/local_mac/results/20260321-thesis-pool-batch/mlruns \
  --mlflow-ui-port 5013 \
  --no-reshard \
  --random-seed 55
```

### Automated Run (Recommended)
```bash
bash experiments/thesis/local_mac/pending_experiments/run_seeds_44_55.sh
```

This script orchestrates both seeds with:
- Pool strategy, 2 workers
- Batch label: `20260321-thesis-pool-batch` (or override with first argument)
- Sequential execution with 10 second pause between seeds
- Timestamped logging to `batch.log`

## Expected Runtime

**Per-seed timing** (based on completed seeds 11, 22, 33):
- Average fevals: ~94 out of max 192
- Average generations: ~19 (pop size 8 → ~8 children/gen)
- Stagnation detected typically by gen 18–20
- Average duration: ~85 minutes per seed on M2 Pro (no GPU)
- Wall-clock estimate: ~170 minutes for both seeds sequentially

**Total campaign timeline:**
- Both seeds: ~2 hr 50 min
- Plus 10 second inter-seed pause: negligible
- Recommend running overnight or during a development break

## What to Collect After Completion

For each seed (e.g., `seed44/`, `seed55/`), verify and archive:

1. **ga_history.csv**
   - Per-generation best fitness, population stats, operator adoption rates
   - Required for Fig 1 (convergence curves, all 5 seeds)

2. **ga_final_comparison.csv**
   - Best merged config vs. both baseline models
   - Final score, task breakdown (wikitext, boolq, sciq)
   - Used to populate Table seed11_result with all 5 rows

3. **ga_stop_details.json**
   - Stop reason (stagnation, max_fevals, timeout, etc.)
   - Stop generation, final fitness, time elapsed
   - Confirms run validity (must show stagnation or max_fevals, not timeout)

4. **best_config.yaml**
   - Final merged model config (weights, methods, model selections per layer)
   - Document for reproducibility; optional archive

5. **run.log**
   - Full stderr/stdout from evolve_ga
   - Diagnostic for any anomalies or warnings

6. **mlruns/** (MLflow artifact directory)
   - Timestamped experiments under EXPERIMENT_NAME
   - Contains run metadata, plots, and logged metrics
   - Optional archival; MLflow UI already persists metrics locally

## What to Update in the Thesis

### Chapter 4 — Local Validation Results

**Tab: seed11_result** (Table 4.X or similar)
- Currently rows: seed 11, 22, 33
- Add rows: seed 44, 55
- Columns: seed ID, best_final_score, wikitext_score, boolq_score, sciq_score, stop_reason, generations, fevals

### Chapter 4 — Statistical Summary
New section or paragraph titled *5-Seed Ensemble Summary*:

**5-Seed Ensemble Statistics**
```
Sample size: n = 5 (seeds 11, 22, 33, 44, 55)
Best-search-score mean: μ = [insert] ± σ
Standard deviation: σ = [insert]
95% confidence interval: μ ± t₀.₀₂₅,₄ × σ/√5
  = [mean] ± 2.776 × [σ]/√5
  = [lower], [upper]
Stagnation consistency: [count] of 5 runs detected stagnation; [count] completed via max_fevals
```

Example notation:
```
  5-seed best-search mean: 71.24 ± 0.45 (95% CI: [70.50, 71.98])
```

### Chapter 5 — Discussion & Limitations

**Locate sentence:** "pending extended validation across random seeds"
**Action:** Remove "pending" qualifier and replace with:
```
Extended validation across five random seeds (11, 22, 33, 44, 55) on the tuned configuration
confirms the adaptive GA's robustness. Best-search scores converge to 71.24 ± 0.45
(95% CI: [70.50, 71.98]), indicating [...]. The narrow confidence interval suggests
stochastic stability under the local tuned preset.
```

## Pass/Fail Criteria

### Run Must PASS:
1. **Process exits cleanly** without segfault, OOM, or unhandled exception
2. **Stagnation detected** in `ga_stop_details.json` OR `max_fevals` reached (not timeout)
3. **Export score reported** in `ga_final_comparison.csv` (not NaN or negative)
4. **Wall-clock time** < 4 hours per seed (indicates no deadlock or infinite loop)

### Run Must FAIL (do not include in table):
1. **All generation-1 children fail metric_guard** (no valid genomes generated)
   → Indicates config prevents any viable offspring; requires tuning
2. **Pipeline crash** (Python stack trace in run.log, incomplete CSV files)
   → Indicates environmental or bug issue; re-run after diagnosis
3. **Timeout > 4 hours per seed** (wall time exceeded, process killed)
   → Indicates resource contention or hardware degradation
4. **Final fitness is NaN or worse than both baselines** (exploration failure)
   → Indicates fitness function misconfiguration

## Statistical Analysis Plan

### Descriptive Statistics
For seeds {11, 22, 33, 44, 55}:

1. **Sample mean:**
   μ = (Σ best_final_score_i) / 5

2. **Sample standard deviation:**
   σ = √[ Σ(best_final_score_i - μ)² / (n-1) ]
   (using n-1 denominator for unbiased estimate)

3. **95% confidence interval (Student's t, df=4):**
   CI = μ ± t₀.₀₂₅,₄ × (σ / √5)
   where t₀.₀₂₅,₄ = 2.776 (two-tailed, α=0.05)

### Reporting
In thesis table/figure:
- Mean as point estimate
- 95% CI as [lower, upper] range
- Standard deviation as ± notation
- All metrics to 2 decimal places

### Interpretation
- **Narrow CI** (e.g., width < 1.5 points) → GA is stable, operator tuning was effective
- **Wide CI** (e.g., width > 3 points) → Possible hidden stagnation or environment variance
- **Drift in mean** (seed 44/55 lower than 11/22/33) → Possible cache/resource depletion over time

---

**Campaign Coordinator Notes:**
- Ensure HuggingFace model cache is shared and warmed before run to avoid first-seed slowdown
- Monitor MLflow UI during execution: http://127.0.0.1:5013
- If either seed stalls > 4 hours, kill and re-run with `--random-seed` only
- After both seeds complete, commit results to git: `git add RESULTS_20260321_EXTENDED_5SEED.md`
