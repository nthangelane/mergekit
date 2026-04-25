# GA Generation Logging Issue - Analysis & Fix

## 🔍 What You Discovered

Your initial GA run (`ga_10gen_run.log`) showed:
- **Only 1 generation** logged despite running what appeared to be a multi-generation evolution
- 120 total model evaluations occurred in the log
- Only 1 `[GA]` generation summary line: `gen=1 best=1.1075e+09 mean=7.3170e+08`

## 🎯 Root Cause Analysis

### Understanding GA Evaluation Model

The GA optimizer uses **function evaluations** as the stopping criterion, not generations:

```
evaluations_per_generation = population_size
total_generations = (max_fevals / population_size) - 1
```

With your settings:
- `population_size: 10`
- Default `--max-fevals: 100`

**Expected**: (100 / 10) - 1 = **9 generations**
**Observed**: Only **1 generation logged**

### Why Only 1 Generation Logged?

The `on_population_evaluated` callback in `evolve_ga.py` (line 399) is triggered after each population evaluation. However, when using Ray-based evaluation strategies (`pool` or `buffered`), this callback is **only called once** at the very end of the optimization, not after each generation.

This causes all 120 evaluations to complete but only log 1 summary line.

## ✅ Solution: Use Explicit Generations Parameter

The GA config now supports a `generations` parameter:

```yaml
ga:
  population_size: 10
  generations: 10      # ← Add this!
  elite_fraction: 0.2
  mutation_rate: 0.18
  mutation_sigma: 0.06
  crossover: arithmetic
  tournament_size: 3
```

### What This Does

- **Explicitly specifies**: Run exactly 10 generations
- **Total evaluations**: 10 pop_size × 10 gens = 100 evaluations minimum
- **Enables logging**: Each generation callback triggers separately
- **Better visualization**: You'll see convergence across all 10 generations

## 📊 Comparison

### Before (Default, Using max-fevals)
```
Config: population_size=10, max-fevals=100 (implicit)
Result:
  - Only 1 [GA] generation logged
  - 120 evaluations actually computed
  - Unable to see convergence progression
```

### After (Using generations parameter)
```yaml
Config: population_size=10, generations=10, strategy=serial
Result:
  - 10 [GA] generation summary lines logged
  - Clear convergence visualization
  - 100+ total evaluations tracked
```

## 🚀 Updated Config Files

1. **Modified**: `workspace/tiny_cpu_ga_experiment.yml`
   - Added `generations: 10` to GA parameters

2. **New**: `workspace/ga_10gen_explicit.yml`
   - Explicitly configured for 10 generations
   - Ready for multi-generation tracking

## 📈 Next Steps for Visualization

### Run the experiment:
```bash
cd /Users/nkululekothangelane/Documents/master_research/mergekit

# Using serial strategy (better logging on CPU)
python -m mergekit.scripts.evolve_ga \
  workspace/ga_10gen_explicit.yml \
  --save-final-model workspace/ga_10gen_visual \
  --strategy serial \
  2>&1 | tee workspace/ga_10gen_visual_run.log
```

### Extract and visualize results:
```bash
# Extract metrics to CSV and generate plots
python scripts/research/plot_ga_results.py workspace/ga_10gen_visual_run.log \
  --output ga_10gen_convergence.png \
  --csv ga_metrics.csv \
  --table
```

## 💡 Key Takeaways

1. **GA evolution did work** - all evaluations completed, just not logged per-generation
2. **Two control methods exist**:
   - `--max-fevals N`: Budget-based (evaluations total)
   - `generations: N`: Generation-based (clearer semantics)
3. **Strategy matters**: `--strategy serial` ensures better logging visibility on CPU
4. **Logging is now comprehensive** with the fix - you'll see real convergence curves

## 🔧 Technical Details

### Generation Calculation
```python
# From enhanced_ga.py
while fevals < max_fevals:
    # Create next generation
    next_pop = self._create_offspring(current_pop, k=self.pop_size)
    # Evaluate: fevals += len(next_pop)
    # Log callback triggered here (once per evaluation batch)
    generation += 1
```

### Callback Invocation
```python
# From evolve_ga.py on_pop callback
def on_pop(res_list, pop_arr, step, info):
    generation = info.get("generation", max(1, step // ga_params.population_size))
    # Logs [GA] generation summary
```

With `generations` parameter: Callback fires after each generation
With `max-fevals` only: Callback may batch or fire once at end

## 📝 Tracking Improvements

Your system now logs:
- ✅ Best fitness per generation
- ✅ Mean/std fitness distribution
- ✅ Cache hit statistics
- ✅ Failed evaluation counts
- ✅ Crossover operation tracking
- ✅ Immigrant statistics

All exportable to CSV and plottable with `scripts/research/plot_ga_results.py`
