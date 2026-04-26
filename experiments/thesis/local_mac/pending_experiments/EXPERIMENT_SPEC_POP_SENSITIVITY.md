# Experiment Spec: Population Size Sensitivity Analysis

## Purpose

Determine whether population size 8 (chosen for M1 memory constraints) is near-optimal for the search space, or whether smaller/larger populations would meaningfully change search convergence and final fitness. This addresses examiner questions about whether the baseline configuration is well-justified or merely convenient. A comparison across pop=4, pop=8, pop=16, pop=32 at fixed feval budget provides empirical evidence of the full trade-off curve between population diversity and generation depth.

## Experimental Design

**Variables:**
- Population sizes: 4, 8 (baseline), 16, 32
- Fixed feval budget: N=96 (identical across all four runs)
- Fixed random seed: 11 (ensures fair comparison)
- All other GA parameters: identical

**Proportional Scaling:**
- `elite_fraction` and `explorer_fraction` scale with population size to maintain a constant 1-elite policy and a +1 explorer per doubling:
  - Pop=4:  elite_fraction=0.25   (1 elite), explorer_fraction=0.25   (1 explorer)
  - Pop=8:  elite_fraction=0.125  (1 elite), explorer_fraction=0.25   (2 explorers)
  - Pop=16: elite_fraction=0.0625 (1 elite), explorer_fraction=0.1875 (3 explorers)
  - Pop=32: elite_fraction=0.03125 (1 elite), explorer_fraction=0.125 (4 explorers)

**Generation Expectations:**
- Pop=4:  ~24 complete generations (4  evals/gen × 24 = 96)
- Pop=8:  ~12 complete generations (8  evals/gen × 12 = 96)
- Pop=16: ~6  complete generations (16 evals/gen ×  6 = 96)
- Pop=32: ~3  complete generations (32 evals/gen ×  3 = 96) — extreme upper bound

The trade-off is explicit: smaller populations allow more generations but less intra-generational diversity; larger populations provide richer genetic material per generation but fewer opportunities for selection over time. Pop=32 at N=96 is deliberately extreme — 3 generations tests whether initial diversity alone (without iterative selection) can produce competitive candidates.

## Configs

All configs located in `local_mac/`:

| Config | Population | Gens @N=96 | Elite | Explorer | Purpose |
|--------|-----------|-----------|-------|----------|---------|
| `thesis_pop4_sensitivity.yml` | 4 | ~24 | 0.25 | 0.25 | Shallow pop, high-generation depth |
| `thesis_pop8_sensitivity.yml` | 8 | ~12 | 0.125 | 0.25 | Standard baseline (M1 optimized) |
| `thesis_pop16_sensitivity.yml` | 16 | ~6 | 0.0625 | 0.1875 | Double pop, half generations |
| `thesis_pop32_sensitivity.yml` | 32 | ~3 | 0.03125 | 0.125 | Extreme diversity, near-zero selection pressure |

All inherit the multi-method merging setup, task definitions, two-stage evaluation, and behavior diversity constraints from `thesis_run_pythia70m.yml`.

## Run Commands

Execute all four runs sequentially with fixed seed and budget:

```bash
bash /sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/run_pop_sensitivity.sh
```

Individual runs (if needed):

```bash
# Pop=4
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop4_sensitivity.yml \
  --seed 11 \
  --output workspace/thesis/local_mac/results/pop_sensitivity/pop4

# Pop=8
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop8_sensitivity.yml \
  --seed 11 \
  --output workspace/thesis/local_mac/results/pop_sensitivity/pop8

# Pop=16
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop16_sensitivity.yml \
  --seed 11 \
  --output workspace/thesis/local_mac/results/pop_sensitivity/pop16

# Pop=32 (extreme upper bound — only ~3 generations)
python -m thesis.ga_runner \
  --config mnt/local_mac/pending_experiments/thesis_pop32_sensitivity.yml \
  --seed 11 \
  --output workspace/thesis/local_mac/results/pop_sensitivity/pop32
```

## Expected Runtime

- **Pop=4**: ~50 minutes (most generations, fastest per-generation eval)
- **Pop=8**: ~85 minutes (baseline, moderate generations)
- **Pop=16**: ~110 minutes (fewer generations, higher per-generation cost due to stage2 expansion)
- **Pop=32**: ~130 minutes (only 3 generations, highest per-generation cost; stage2 processes 32 candidates)
- **Total**: ~375 minutes (~6 hours) for all four runs sequentially

Times assume M1 Pro with typical model load/cache patterns. Actual runtime may vary ±15% based on system load.

## What to Collect

For each run, extract and tabulate:

1. **Best fitness trajectory**
   - best_so_far per generation (0–6 for pop=16, 0–12 for pop=8, 0–24 for pop=4)
   - Plot all three on same axes (generation on x-axis, fitness on y-axis)
   - Observe convergence rate differences

2. **Generation-5 fitness (standardized comparison point)**
   - Record `best_so_far` at generation 5 for all three runs
   - Provides apples-to-apples snapshot since gen 5 occurs in all configs
   - Use as primary quantitative comparison metric

3. **Final fitness (gen N)**
   - Pop=4 gen 24, Pop=8 gen 12, Pop=16 gen 6
   - Documents asymptotic performance at budget exhaustion

4. **Best candidate**
   - Record merging method (passthrough, linear, slerp) and weights
   - Note which candidate genome achieved best fitness
   - Check for diversity in best candidates across runs (suggests robustness)

5. **Stage2 participation**
   - Count how many individuals entered stage 2 in each run
   - Higher stage2 rate may indicate better early-generation diversity

6. **Logs & artifacts**
   - Preserve full stdout logs (timing, parameter snapshots)
   - Save fitness_history.json for each run
   - Archive final merged models (for safety/reproducibility)

## Thesis Update

**Section:** Chapter 3, GA Configuration (Population Size Justification)

**If null result** (all three similar, pop differences ≤2%):
> "We compared population sizes 4, 8, and 16 under identical feval budget (N=96) and found marginal differences in final fitness (<2%). Population size 8 was selected pragmatically for M1 memory compatibility; the sensitivity analysis confirms it is not sub-optimal within the tested range, though larger populations might benefit searches with longer time budgets."

**If pop=16 clearly better** (>3% improvement over pop=8):
> "Sensitivity analysis across population sizes 4, 8, and 16 (equal feval budget) reveals that larger populations improve search convergence, with pop=16 achieving X% better fitness than pop=8. However, M1 memory constraints limit viable population size for full thesis runs; we adopt pop=8 as a pragmatic trade-off between diversity and hardware feasibility, acknowledging this as a potential source of sub-optimality."

**If pop=4 clearly worse** (>3% degradation vs pop=8):
> "Population size 4 showed X% fitness degradation vs. pop=8 despite more generations, indicating that diversity-limited selection impaired convergence quality. This justifies the pop=8 baseline as a minimal threshold for adequate intra-generational genetic variation."

**Expected length:** 2–3 sentences, positioned in the GA Configuration subsection immediately after population_size parameter definition.

## Analysis

**Visualization & Comparison:**

1. **Fitness vs. generation** (line plot)
   - X-axis: Generation number (standardized to index 0, 1, 2, ..., 3 for fair multi-population display)
   - Y-axis: Fitness (same scale across all runs)
   - Four lines: pop=4 (blue), pop=8 (green), pop=16 (orange), pop=32 (red)
   - Overlay vertical line at gen=3 to highlight standardized comparison point (all configs reach gen 3)

2. **Generation-3 fitness bar chart**
   - X-axis: Population size (4, 8, 16, 32)
   - Y-axis: best_so_far fitness at gen=3
   - Error bars: none (single seed=11)

3. **Final fitness (budget exhaustion)**
   - Record fitness at gen 24, 12, 6, 3 respectively
   - Plot as a curve: population size vs. final fitness at N=96 fevals
   - This is the population-diversity tradeoff curve

**Statistical Interpretation:**

- **Primary metric:** gen-5 fitness (equal generation index across all configs)
- **Secondary metrics:** final fitness (generation N), diversity in best candidates
- **Null hypothesis:** All three population sizes yield equivalent fitness (within measurement noise ~0.5%)
- **Alternative:** Population size significantly affects convergence quality

**Robustness Check:**

If time permits, re-run pop=8 with seed=12 (same config, different seed). If pop=8 gen-5 results are consistent across seeds, confidence in baseline is higher.

## Expected Outcomes

### Pop=4 (High-generation, low-diversity scenario)
- **Pros:** Many generations allow long evolutionary trajectory; may discover niche optima
- **Cons:** Selection at each generation bottlenecks diversity; risk of premature convergence
- **Prediction:** Fitness at gen 5 likely **below** pop=8, possibly by 1–3%; may recover somewhat by gen 24 if diversity mechanisms (behavior bonus) sustain exploration

### Pop=8 (Balanced baseline)
- **Pros:** Moderate diversity, moderate generational depth; proven M1-feasible
- **Cons:** Fewer total generations than pop=4; larger per-generation variance
- **Prediction:** **Reference point** for comparison; expect middle-ground convergence trajectory

### Pop=16 (Low-generation, high-diversity scenario)
- **Pros:** Rich intra-generational genetic variation; more selection targets per generation
- **Cons:** Only ~6 generations may limit cumulative selection pressure; stage2 evaluation cost multiplied by pop size
- **Prediction:** Fitness at gen 3 may be **comparable or slightly better** than pop=8 (richer material); final fitness may plateau early if 6 generations insufficient for convergence

### Pop=32 (Extreme upper bound — near-zero selection pressure)
- **Pros:** Maximum initial diversity; the best of 32 candidates after 3 rounds of selection is a strong upper bound on "what the search space contains"
- **Cons:** 3 generations provides almost no iterative selection pressure; essentially an informed random search with 3 rounds of elitism
- **Prediction:** Gen-3 fitness likely **below pop=16** (GA provides no real convergence benefit); if pop=32 at gen=3 approaches pop=8 at gen=12, it suggests the thesis GA gains little from generational iteration and the search space is easy to exploit with diversity alone
- **This is the most scientifically interesting result:** a near-random-search with pop=32 outperforming the tuned GA would strengthen the random baseline critique in Ch4

### Overall Hypothesis
- **Convergence speed:** pop=4 fastest per generation, pop=32 slowest
- **Final quality at N=96:** pop=8 ≈ pop=16 > pop=4 ≈ pop=32 (bell-curve tradeoff)
- **Tradeoff curve:** Fitness should peak somewhere between pop=8 and pop=16; pop=4 and pop=32 should both underperform the middle
- **Implication:** pop=8 is a defensible pragmatic choice; pop=32 serves as an upper-diversity bound that frames the GA's generational benefit

## Key Metric: Generation-3 Best Fitness

Use **best_so_far at generation 3** as the primary standardized comparison across all four runs.

**Why generation 3?**
- Occurs in all configs including pop=32 (the most constrained)
- Equal-generation comparison removes the confounding effect of generation depth
- Independent of final generation count

**Reporting format:**

```
Gen-3 Best Fitness Comparison (Seed 11, N=96 fevals)
=====================================================
Pop=4:  fitness = X.XXX  (~72 evals used by gen 3)
Pop=8:  fitness = Y.YYY  (~24 evals used by gen 3)  [baseline]
Pop=16: fitness = Z.ZZZ  (~48 evals used by gen 3)
Pop=32: fitness = W.WWW  (~96 evals used by gen 3)  [budget exhausted]

Final fitness at feval=96:
Pop=4:  fitness = X2.XXX (gen 24)
Pop=8:  fitness = Y2.YYY (gen 12)  [baseline]
Pop=16: fitness = Z2.ZZZ (gen 6)
Pop=32: fitness = W2.WWW (gen 3, same as gen-3 — no more generations)
```

If all gen-3 values within 1%: "Population size has minimal impact on early selection quality; generational depth is the differentiating factor."

If pop=32 gen-3 approaches pop=8 final (Y2): "Diversity alone nearly matches the full GA benefit — generational selection provides limited additional value."

If pop=4 final (X2) < pop=8 final (Y2) by >3%: "Low-population diversity deprivation creates persistent convergence deficit."

---

**Generated:** 2026-03-21
**Thesis:** GA-based Model Merging, Pythia-70M, Stellenbosch University
**Author:** [Your Name]
**Status:** Ready for execution
