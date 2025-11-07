# ⏱️ Live Monitoring Guide

## How to Monitor Your GA Run

The GA run is **currently in progress**. Use these commands to watch it evolve:

---

## Watch Generation Progress (RECOMMENDED)

### Real-Time Generation Counter
```bash
# Watch [GA] lines appear as each generation completes
tail -f workspace/ga_10gen_mac_cpu_run.log | grep "\[GA\]"
```

This will show each generation as it completes:
```
[GA] gen=1 best=1.1075e+09 mean=7.3170e+08 std=2.0462e+08 ...
[GA] gen=2 best=9.8760e+08 mean=6.8900e+08 std=1.9876e+08 ...
[GA] gen=3 best=8.5430e+08 mean=6.2100e+08 std=1.8765e+08 ...
# ... continues until gen=10
```

### Quick Status Check (Run this every few minutes)
```bash
# How many generations logged so far?
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log | wc -l
```

Expected progression:
- After 2-3 minutes: 1-2 lines
- After 5 minutes: 2-3 lines  
- After 10 minutes: 5-6 lines
- After 15 minutes: 8-10 lines

### All Generations So Far
```bash
# Display all completed generations
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log
```

---

## Full Log Monitoring

### Last 50 Lines
```bash
tail -50 workspace/ga_10gen_mac_cpu_run.log
```

### Search for Errors
```bash
# Check for any problems (should return empty)
grep -i "error\|exception\|traceback" workspace/ga_10gen_mac_cpu_run.log

# Check CUDA specifically (should return empty)
grep -i "cuda" workspace/ga_10gen_mac_cpu_run.log
```

### Count Evaluations Done
```bash
# Each evaluation generates output
grep -c "Evaluation" workspace/ga_10gen_mac_cpu_run.log
```

---

## Performance Metrics

### Time Elapsed
```bash
# From file creation time
ls -lah workspace/ga_10gen_mac_cpu_run.log | awk '{print $6, $7, $8}'

# Or check modification time
stat workspace/ga_10gen_mac_cpu_run.log | grep -i modify
```

### Estimated Time Remaining
```bash
# If you see gen=5 after 8 minutes:
# - 5 gens in 8 min = 1.6 min/gen
# - 5 gens remaining × 1.6 = ~8 min left
# Formula: (remaining_gens) × (minutes_per_gen)
```

---

## Completion Indicators

### ✅ Success Signs
- `grep "[GA]" workspace/ga_10gen_mac_cpu_run.log | wc -l` returns **10**
- No CUDA errors appear
- No `AssertionError` or `Traceback` in log
- File keeps growing with evaluation output

### ❌ Problem Signs
- Stuck at same generation for >3 minutes
- `AssertionError: Torch not compiled with CUDA enabled` appears
- Ray error messages appear
- Process seems frozen (no new output for >2 min)

### What to Do If Stuck
```bash
# Check if process is still running
ps aux | grep evolve_ga | grep -v grep

# If stuck, you can safely Ctrl+C and restart with:
python -m mergekit.scripts.evolve_ga workspace/ga_10gen_mac_cpu.yml \
  --storage-path workspace/ga_10gen_mac_storage \
  --save-final-model \
  --strategy serial \
  --num-workers 1 \
  --no-merge-cuda \
  2>&1 | tee -a workspace/ga_10gen_mac_cpu_run.log
```

---

## When Run Completes

### You'll See This in Terminal
```
# After 15-20 minutes:
[GA] gen=10 best=4.2340e+08 mean=4.3800e+08 std=1.5432e+08 ...
# No more output = run complete
```

### Verification (Run these commands right after)

**1. Count Generations**
```bash
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log | wc -l
# Should return: 10
```

**2. Verify No Errors**
```bash
grep -i "error\|cuda\|exception" workspace/ga_10gen_mac_cpu_run.log
# Should return: (empty)
```

**3. View All Results**
```bash
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log
# Should show 10 lines with improving best scores
```

**4. Extract Metrics**
```bash
python plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log \
  --table --csv metrics.csv --output convergence.png
```

---

## Comparing Old vs New

### Generation Count
```bash
echo "=== OLD RUN (PROBLEM) ==="
grep "\[GA\]" workspace/ga_10gen_run.log | wc -l
# Expected: 1 ❌

echo "=== NEW RUN (FIXED) ==="
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log | wc -l
# Expected: 10 ✅
```

### Generation Details
```bash
echo "=== OLD RUN ==="
grep "\[GA\]" workspace/ga_10gen_run.log

echo "=== NEW RUN ==="
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log
```

### Plot Comparison
```bash
# After run completes:
echo "Opening old plot (1 point, flat):"
open ga_results.png

echo "Opening new plot (10 points, convergence):"
open convergence.png
```

---

## Quick Command Reference

```bash
# MONITOR (Run these frequently)
tail -f workspace/ga_10gen_mac_cpu_run.log | grep "\[GA\]"  # Watch real-time
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log | wc -l     # Quick count
tail -20 workspace/ga_10gen_mac_cpu_run.log                   # Last 20 lines

# CHECK FOR PROBLEMS
grep -i "error\|cuda" workspace/ga_10gen_mac_cpu_run.log     # Error scan
ps aux | grep evolve_ga | grep -v grep                       # Process check

# AFTER COMPLETION
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log             # View all
python plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log --table  # Extract table
python plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log \
  --output convergence.png --csv metrics.csv                 # Full analysis
```

---

## Troubleshooting

### If You See CUDA Error
```
AssertionError: Torch not compiled with CUDA enabled
```
**Solution**: This shouldn't happen! The `--no-merge-cuda` flag should prevent it. If it appears:
- Stop the run (Ctrl+C)
- Make sure command includes `--no-merge-cuda`
- Restart with full command

### If Generation Takes >5 minutes
```
# Normal: 1-2 min per generation on Mac CPU
# Slow: > 2 min per generation
# Check if other programs using CPU:
top -n 1 | head -15
```

### If Terminal Seems Frozen
```bash
# Check if process still running:
ps aux | grep evolve_ga

# If running but no output, it may be evaluating (normal)
# If not running, run stopped unexpectedly - check log:
tail -100 workspace/ga_10gen_mac_cpu_run.log | tail -20
```

---

## Summary

| Task | Command | Expected Output |
|------|---------|-----------------|
| Watch live | `tail -f workspace/ga_10gen_mac_cpu_run.log \| grep "[GA]"` | New [GA] lines every 1-2 min |
| Quick check | `grep "[GA]" workspace/ga_10gen_mac_cpu_run.log \| wc -l` | Number 1-10 (increasing) |
| Check errors | `grep -i "error\|cuda" workspace/ga_10gen_mac_cpu_run.log` | (empty = good) |
| Completion sign | Run generates new [GA] line, then stops | See gen=1 through gen=10 |
| View results | `grep "[GA]" workspace/ga_10gen_mac_cpu_run.log` | 10 lines showing improvement |

**Expected Timeline**:
- Currently: ~5 min elapsed, ~10-15 min remaining
- Around gen 5: Should appear in next ~5-10 min
- Around gen 10: Should complete in ~15-20 min total

🎯 **Next Action**: Use the real-time monitor command to watch generations appear!
