# GA Generations & Logging - Quick Reference

## The Issue (In One Sentence)
**Your GA ran 120 evaluations but only logged 1 generation summary because the callback batched results.**

## The Solution (In One Sentence)
**Add `generations: 10` to your GA config and use `--strategy serial` for per-generation logging.**

---

## Before & After

### ❌ BEFORE (What You Had)
```yaml
ga:
  population_size: 10
  elite_fraction: 0.2
  # ... no generations parameter
```

**Result:**
- 120 evals run
- 1 [GA] line logged
- No convergence curve

### ✅ AFTER (What We Fixed)
```yaml
ga:
  population_size: 10
  generations: 10      # ← Add this!
  elite_fraction: 0.2
  # ... rest same
```

**Result:**
- ~100 evals run (10 gens × 10 pop)
- 10 [GA] lines logged
- Clear convergence curve

---

## To Reproduce & Visualize

### Run 10-generation GA:
```bash
cd /Users/nkululekothangelane/Documents/master_research/mergekit

python -m mergekit.scripts.evolve_ga \
  workspace/ga_10gen_explicit.yml \
  --save-final-model workspace/ga_10gen_visual \
  --strategy serial \
  2>&1 | tee workspace/ga_10gen_visual_run.log
```

### Generate plots (after it finishes):
```bash
python plot_ga_results.py workspace/ga_10gen_visual_run.log \
  --output ga_convergence.png \
  --csv ga_metrics.csv \
  --table
```

---

## Files Changed/Created

| File | Action | Purpose |
|------|--------|---------|
| `workspace/tiny_cpu_ga_experiment.yml` | ✏️ Modified | Added `generations: 10` |
| `workspace/ga_10gen_explicit.yml` | ✨ Created | New explicit 10-gen config |
| `GA_LOGGING_ISSUE_ANALYSIS.md` | ✨ Created | Detailed technical analysis |
| `GA_RESULTS_EXPLANATION.md` | ✨ Created | Comprehensive explanation |
| `plot_ga_results.py` | ✏️ Updated | Enhanced with table output |

---

## Key Parameters Explained

### `generations` vs `--max-fevals`

**`generations: 10`** (in YAML)
- Exactly 10 generations
- Total evals = generations × population_size
- Callback fires per-generation ✅
- Clear semantics

**`--max-fevals 100`** (CLI flag)
- Up to 100 function evaluations
- Generations = max_fevals / population_size
- Callback may batch results ❌
- Abstract concept

**Recommendation**: Always use YAML `generations` parameter for clarity

### `--strategy serial` vs pool/buffered

| Aspect | serial | pool | buffered |
|--------|--------|------|----------|
| **Logging** | Per-gen ✅ | Batched ❌ | Batched ❌ |
| **CPU** | Single ✅ | Multi ❌ | Multi ❌ |
| **Speed** | Slow | Fast | Medium |
| **Recommended** | ✅ | For GPU | For GPU |

---

## What You'll See Now

**Old plot** (1 point):
```
Fitness Over Generations
        1.1e+09 ●
        1.0e+09 |
        0.9e+09 |
        0.8e+09 |
        0.7e+09 |___________________________________
            0.0   Gen 1
```

**New plot** (10 points):
```
Fitness Over Generations
        1.1e+09 ●
        1.0e+09 ● \\
        0.9e+09 ●  \\
        0.8e+09 ●   \\
        0.7e+09 ●    \\___
        0.6e+09 ●        \\
        0.5e+09 ●         ●___
        0.4e+09 ●             ●
        0.3e+09 ●              ●
        0.2e+09 ●_______________●
            0.0   Gen 1   Gen 5   Gen 10
```

---

## Troubleshooting

### "Only 1 generation logged again?"
✓ Check that you're using `--strategy serial` flag
✓ Verify `generations: 10` is in your YAML config

### "Plot shows only 1 point?"
✓ Make sure you're analyzing the new `ga_10gen_visual_run.log`
✓ Run `grep "\[GA\]" ga_10gen_visual_run.log` to count lines
✓ Should see 10 `[GA]` lines if working correctly

### "Getting storage-path error?"
✓ Config needs `storage_path:` top-level field
✓ See `workspace/ga_10gen_explicit.yml` for example

---

## Current Status

🟡 **In Progress**: 10-generation run started with explicit config
⏱️ **Expected Time**: ~15-20 minutes (serial eval on CPU)
📍 **Location**: `/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/ga_10gen_visual_run.log`

Next: Monitor completion, then generate plots

---

**Questions?** See the detailed analysis in:
- `GA_LOGGING_ISSUE_ANALYSIS.md` - Technical deep dive
- `GA_RESULTS_EXPLANATION.md` - Complete explanation with examples
