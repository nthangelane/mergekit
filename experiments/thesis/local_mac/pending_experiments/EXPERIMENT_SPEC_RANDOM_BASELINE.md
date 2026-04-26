# Experiment Spec: Random Search Baseline for GA Validation

## Purpose

Establish a rigorous baseline to validate whether evolutionary operators (selection, crossover, mutation) in the adaptive GA provide genuine search advantage over naive random candidate generation at identical computational budget. The GA achieved fitness 0.6293 vs. previous baseline 0.6034 (+4.3%), but without random search comparison at N=96 fevals, we cannot distinguish operator value from random variation.

This experiment answers: **Are GA operators justified, or does the search space favor random sampling?**

---

## Hypotheses

### Null Hypothesis (H₀)
**GA search fitness = random search fitness**

There is no significant difference in best-found fitness after 96 evaluations between adaptive GA search and pure random candidate generation. Any observed difference is within random variation.

### Alternative Hypothesis (H₁)
**GA search fitness > random search fitness** (one-tailed)

Adaptive GA's selection, crossover, and mutation operators provide statistically significant improvement in best-found fitness over random search at identical budget (p < 0.05, Welch t-test, one-tailed).

---

## Experimental Design

### Search Method: Pure Random Baseline
- **Population size:** 1 (no population, evaluate each candidate independently)
- **Initialization:** `random_init: true` (each candidate randomly sampled from genotype space)
- **Crossover:** `uniform` (schema-required; never fires at population_size=1 — no second parent exists)
- **Mutation:** `mutation_rate: 0.0` (no within-candidate perturbation)
- **Selection:** `tournament_size: 1` (no selection pressure)
- **Adaptive Operator Sampling:** `false` (no online operator frequency adaptation)
- **Elitism:** `elite_fraction: 0.0` (no carry-over of best individuals)
- **Diversity constraints:** All disabled (no diversity bonuses)

### Search Budget
- **Max fevals:** 96 (identical to mean GA fevals from completed batch)
- **Stage2 limit:** 12 (two-stage evaluation preserved, same as GA)
- **Stage2_top_k:** 4 (stage 1 selects 4 best for full evaluation)

### Task Configuration (identical to GA)
- **wikitext** (w=0.40): byte_perplexity
- **boolq** (w=0.35): accuracy
- **sciq** (w=0.25): accuracy

### Seeds
- **11, 22, 33** (3 independent runs for statistical power)

### Hardware & Parallelism (match GA batch)
- **Pool strategy:** multiprocessing
- **Workers:** 4
- **Max runtime per seed:** 60 minutes (budget: no time limit constraint, but expect ~50 min due to no crossover overhead)

---

## Config File

**Location:** `./thesis_random_search_baseline.yml`

Key differences from adaptive GA config (`thesis_run_pythia70m.yml`):

```yaml
random_init: true          # Each candidate randomly initialised

ga:
  population_size: 1       # Evaluate one candidate at a time
  elite_fraction: 0.0      # No elitism
  mutation_rate: 0.0       # No mutation
  crossover: uniform        # schema requires valid type; dead code at pop=1
  tournament_size: 1       # No selection pressure
  adaptive_method_sampling: false  # No operator adaptation
  diversity_parent_selection: false
  diversity_parent_weight: 0.0
  # ... all mutation/crossover rates set to 0.0

stop:
  max_fevals: 96           # Identical to GA budget
```

See full config: `/sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/thesis_random_search_baseline.yml`

---

## Run Commands

### Prerequisites
```bash
cd /sessions/quirky-cool-bohr/mnt/local_mac
chmod +x run_random_search_baseline.sh
```

### Execute all seeds sequentially
```bash
./run_random_search_baseline.sh
```

This runs:
1. Seed 11 → `workspace/thesis/local_mac/results/random_search_baseline/seed11`
2. Seed 22 → `workspace/thesis/local_mac/results/random_search_baseline/seed22`
3. Seed 33 → `workspace/thesis/local_mac/results/random_search_baseline/seed33`

### Or run individual seeds manually
```bash
# Seed 11
python -m thesis.run_search \
    --config thesis_random_search_baseline.yml \
    --seed 11 \
    --output-dir workspace/thesis/local_mac/results/random_search_baseline/seed11 \
    --pool-strategy multiprocessing \
    --num-workers 4

# Seed 22
python -m thesis.run_search \
    --config thesis_random_search_baseline.yml \
    --seed 22 \
    --output-dir workspace/thesis/local_mac/results/random_search_baseline/seed22 \
    --pool-strategy multiprocessing \
    --num-workers 4

# Seed 33
python -m thesis.run_search \
    --config thesis_random_search_baseline.yml \
    --seed 33 \
    --output-dir workspace/thesis/local_mac/results/random_search_baseline/seed33 \
    --pool-strategy multiprocessing \
    --num-workers 4
```

---

## Expected Runtime

| Seed | Expected Duration | Notes |
|------|-------------------|-------|
| 11 | ~50 minutes | No crossover overhead; pure random sampling |
| 22 | ~50 minutes | No crossover overhead; pure random sampling |
| 33 | ~50 minutes | No crossover overhead; pure random sampling |
| **Total** | **~150 minutes** | Sequential execution (2.5 hours wall-clock) |

Random search should be **faster** than adaptive GA (which took ~60 min/seed at 192 fevals) because:
- No semantic crossover computation (expensive vector similarity)
- No genetic recombination overhead
- No operator adaptation bookkeeping
- Smaller population (size=1 vs 8) reduces scheduling overhead

---

## What to Collect After Completion

### 1. Best Fitness Per Seed
Extract from each seed's output directory:
- **seed11/best_fitness.txt** or **seed11/results.json** → `best_fitness_11`
- **seed22/best_fitness.txt** or **seed22/results.json** → `best_fitness_22`
- **seed33/best_fitness.txt** or **seed33/results.json** → `best_fitness_33`

### 2. Full Evaluation History
For each seed, collect the complete fitness trajectory (all 96 evals):
- Evaluation index, candidate encoding, fitness score, method (passthrough/linear/slerp), stage (1 or 2)
- Used to plot convergence curves and assess search trajectory

### 3. Candidate Pool Statistics
- Total unique candidates evaluated per seed
- Method distribution (% passthrough, % linear, % slerp) — should be roughly 0.25/0.45/0.30 from `initial_method_probs`
- Parameter range coverage (min/max weights sampled)

### 4. Runtime Metrics
- Wall-clock time per seed (captured by run script)
- Total CPU time
- Peak memory usage

---

## What to Update in the Thesis

### Chapter 4 (Experiments & Results)
**Location:** `Chapter 4 → Section: Comparative Baselines → Subsection: Random Search Baseline`

Current status: placeholder section waiting for results.

**Content to add:**
- Describe random search methodology (pure random initialization, N=96 fevals, 3 seeds)
- Report best fitness per seed and aggregated statistics (mean, SD)
- Plot convergence curves: GA vs random search fitness over 96 evaluations
- Report statistical test result (Welch t-test, p-value, effect size if significant)

### Chapter 5 (Discussion & Limitations)
**Location:** `Chapter 5 → Section: Limitations → Ninth limitation (GA Operator Validation)`

Current placeholder: "...comparison against random search baseline pending..."

**Content to add (depending on results):**

**If GA significantly outperforms random (p < 0.05):**
> Random search baseline validation confirms that GA's adaptive selection, crossover, and mutation operators provide genuine search advantage beyond random sampling at equivalent budget (N=96 fevals). Mean best fitness: GA 0.6236 vs. random 0.61XX, p = 0.0XX. This justifies the computational overhead of GA operations and demonstrates that the search space benefits from evolutionary recombination rather than naive exploration alone.

**If GA does not significantly outperform random (p ≥ 0.05):**
> Random search baseline reveals that evolutionary operators provide limited advantage in this constrained search space (N=96 fevals). Mean best fitness: GA 0.6236 vs. random 0.62XX, p = 0.XXX (not significant). This suggests the model merging landscape at Pythia-70M scale may be dominated by initialization noise or exhibit narrow fitness ridges where crossover/mutation struggle to escape. Increasing feval budget or adopting diversity-preserving selection mechanisms may unlock evolutionary advantage. This finding limits generalization of the approach to larger models or more diverse model pairs.

---

## Statistical Test

### Test Design
- **Type:** Welch's t-test (unequal variances, two-sample, one-tailed)
- **Null:** μ_GA = μ_random
- **Alternative:** μ_GA > μ_random (one-tailed)
- **Significance level:** α = 0.05
- **Sample size:** n=3 per group (small sample; interpret with caution)

### Effect Size (if significant)
- **Cohen's d:** (mean_GA - mean_random) / pooled_SD
- Report alongside p-value: "GA outperforms random by 0.XX ± 0.XX (p=0.0XX, Cohen's d = X.XX)"

### Critical Note on Sample Size
With n=3 per group, statistical power is limited (approx 40-50% to detect medium effect at α=0.05). **Do not over-interpret non-significant results as evidence of no difference.** Report observed effect sizes and confidence intervals alongside p-values.

---

## Interpretation Guide

### Scenario 1: GA Significantly Better (p < 0.05, observed effect d > 0.5)
**Conclusion:** Evolutionary operators are justified.
- **Action:** Emphasize in Ch4 results that GA operators unlock search value.
- **Discussion:** Evolutionary search explores model merging space more effectively than random sampling; selection + crossover + mutation are not redundant.
- **Generalization:** Support the value of GA-based merging for similar-scale models and comparable task distributions.

### Scenario 2: GA Numerically Better but Not Significant (0.05 ≤ p < 0.20, d = 0.2–0.5)
**Conclusion:** Weak or inconclusive evidence for GA advantage; higher feval budget may be needed.
- **Action:** Report effect size prominently; acknowledge statistical limitation due to small n.
- **Discussion:** GA shows promise but benefit marginal at N=96 fevals. This is a **major limitation**: GA may require larger search budgets (192+ fevals) to fully leverage evolutionary recombination. Alternatively, the 70M model pair may be too constrained (low model diversity) to benefit from crossover.
- **Recommendation:** Flag in Ch5 limitations as motivation for scaling to 1B+B parameter models or more diverse model pairs.

### Scenario 3: Random Comparable or Better (p ≥ 0.20)
**Conclusion:** Random sampling is competitively effective; GA operators do not add value at this budget.
- **Action:** Substantial revision needed. This undermines the thesis claim.
- **Discussion:** Model merging at Pythia-70M may be inherently random due to:
  - Shallow search space (limited room for crossover to exploit structure)
  - High initialization noise obscuring operator effects
  - Population size (8) too small for effective recombination in 96 evals
  - Possible issue: GA config not exploiting problem structure (may need domain-aware crossover)
- **Recommendation:** Either (a) increase feval budget substantially to show GA benefits accrue over time, (b) switch to larger models where diversity enables crossover, or (c) redesign GA operators (e.g., layer-wise crossover instead of semantic).

---

## Results Table Template

**For insertion into Chapter 4 results section:**

| Method | Seed 11 Best | Seed 22 Best | Seed 33 Best | Mean | SD | 95% CI |
|--------|--------------|--------------|--------------|------|----|---------|
| Random search | ? | ? | ? | ? | ? | [?, ?] |
| GA (adaptive) | 0.6293 | 0.6293 | 0.6123 | 0.6236 | 0.0098 | [0.6101, 0.6371] |

**Fill-in instructions:**
1. After seed11, seed22, seed33 complete, extract best_fitness from each result directory
2. Compute Mean = (f11 + f22 + f33) / 3
3. Compute SD = sqrt(sum((fi - mean)² / 2))  [uses n-1 for unbiased estimate with n=3]
4. Compute 95% CI using t-distribution (df=2): mean ± t₀.₀₂₅(2) × SE, where SE = SD / sqrt(3)

**For insertion into Chapter 5 limitations section:**

| Comparison | GA vs Random | p-value | Effect Size (Cohen's d) | Interpretation |
|------------|--------------|---------|------------------------|-----------------|
| Best found fitness (Welch t-test) | 0.6236 vs 0.?? | 0.??? | ? | [See Interpretation Guide above] |

---

## Key Assumptions & Caveats

1. **Identical random seed sequences:** Both GA and random search use same RNG seeds (11, 22, 33) for model initialization and task sampling. This ensures fair comparison of stochasticity orthogonal to search strategy.

2. **Same evaluation protocol:** Both use two-stage evaluation (4 stage1 limit, stage2_top_k=4) to match computational fairness.

3. **Random search does not "learn":** Unlike GA, random search does not adapt operator frequencies or selection pressures. This is intentional — it isolates the contribution of adaptive operators.

4. **Search space assumptions:** Config assumes genotype space is roughly uniform in quality. If fitness landscape is highly multimodal or has sparse optima, crossover may struggle at small populations (size=1).

5. **Generalization scope:** Results are specific to Pythia-70M + 3 tasks. Random search baseline for larger models (1B+, 7B+) or different task distributions may show different GA advantage.

---

## Checklist: Before Running

- [ ] `/sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/thesis_random_search_baseline.yml` exists and is readable
- [ ] `/sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/run_random_search_baseline.sh` is executable (`chmod +x`)
- [ ] Output directory structure pre-created: `workspace/thesis/local_mac/results/random_search_baseline/`
- [ ] GA completed runs available for comparison (seeds 11, 22, 33 with best fitness reported)
- [ ] Python environment configured with thesis dependencies (thesis package, pytorch, evaluation tasks)
- [ ] Disk space: ~20GB free (3 seeds × 6GB per seed approximate size)
- [ ] No other experiments running on shared compute resource

---

## Checklist: After Completion

- [ ] All 3 seeds completed without error
- [ ] Best fitness extracted and tabulated per seed
- [ ] Convergence curves generated (fitness vs. feval for random search)
- [ ] Welch t-test computed and p-value reported
- [ ] Effect size (Cohen's d) computed
- [ ] Chapter 4 results section updated with random search results
- [ ] Chapter 5 limitations section updated with interpretation
- [ ] Statistical test interpretation recorded (see Interpretation Guide)
- [ ] Thesis results table completed (both rows: GA and Random)
- [ ] Backup: all result files copied to secondary storage

---

## References & Related Sections

- **Thesis Chapter 4, Section: Comparative Baselines** → main results location
- **Thesis Chapter 5, Section: Limitations, Ninth item** → discussion of operator validation
- **GA Config:** `thesis_run_pythia70m.yml` (adaptive GA with all operators enabled)
- **Random Config:** `thesis_random_search_baseline.yml` (pure random sampling)
- **Related Work:** No evolution-free baseline has been published for model merging at this scale; this experiment establishes first formal comparison
