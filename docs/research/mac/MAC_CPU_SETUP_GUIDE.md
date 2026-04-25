# 🍎 Mac M1/M2 CPU-Only GA Execution Guide

## ⚠️ The CUDA Error You Hit

```
AssertionError: Torch not compiled with CUDA enabled
```

**Cause**: Mac doesn't have CUDA, but the code was trying to load tensors with CUDA support

**Solution**: Use `--no-merge-cuda` flag to force CPU-only execution

---

## ✅ What We Fixed

### The Problem
```bash
# This failed because it tried to use CUDA
python -m mergekit.scripts.evolve_ga config.yml \
  --strategy serial
  # Missing: --no-merge-cuda
```

### The Solution
```bash
# This works on Mac - forces CPU-only execution
python -m mergekit.scripts.evolve_ga config.yml \
  --storage-path workspace/ga_storage \
  --save-final-model \
  --no-merge-cuda          # ← KEY FLAG FOR MAC!
  --strategy serial \
  --num-workers 1          # ← CPU worker count
```

---

## 🔧 Key Flags for Mac M1/M2

| Flag | Purpose | Value |
|------|---------|-------|
| `--no-merge-cuda` | **Disable CUDA** (essential for Mac) | No value (it's a flag) |
| `--strategy serial` | Per-generation logging | `serial` |
| `--num-workers` | CPU worker threads | `1` (or more if you want) |
| `--save-final-model` | Save best model | Flag (no value) |

---

## 📝 Command Template for Mac

```bash
python -m mergekit.scripts.evolve_ga \
  workspace/ga_10gen_mac_cpu.yml \
  --storage-path workspace/ga_10gen_mac_storage \
  --save-final-model \
  --no-merge-cuda \
  --strategy serial \
  --num-workers 1 \
  2>&1 | tee workspace/ga_10gen_mac_cpu_run.log
```

---

## 🏃 Running Now

We've started the corrected run:

```bash
✅ Config: workspace/ga_10gen_mac_cpu.yml (CPU-optimized)
✅ Flags: --no-merge-cuda --strategy serial
✅ Workers: 1 (CPU-only)
✅ Log: workspace/ga_10gen_mac_cpu_run.log
⏳ Status: Running...
⏱️ Expected: 15-20 minutes
```

---

## 📊 How to Monitor

### Watch progress (live)
```bash
tail -f workspace/ga_10gen_mac_cpu_run.log | grep "\[GA\]"
# Should see [GA] lines appear as generations complete
```

### Check for errors
```bash
tail workspace/ga_10gen_mac_cpu_run.log
# Should show merge and evaluation progress, NOT CUDA errors
```

### Verify no CUDA errors
```bash
grep -i "cuda\|gpu" workspace/ga_10gen_mac_cpu_run.log
# Should return nothing (empty = good!)
```

---

## ✨ What Changed

### Files Created/Updated
- ✨ `workspace/ga_10gen_mac_cpu.yml` - New Mac-specific config
- ⏳ `workspace/ga_10gen_mac_storage/` - Will be created during run
- ⏳ `workspace/ga_10gen_mac_cpu_run.log` - Run log (in progress)

### Previous Attempt
- ❌ Used Ray pool strategy (batches callbacks)
- ❌ Missing `--no-merge-cuda` flag (caused CUDA error)
- ❌ Config was generic (not Mac-optimized)

### This Attempt
- ✅ Using serial strategy (per-generation logging)
- ✅ Added `--no-merge-cuda` (CPU-only execution)
- ✅ Mac-optimized config and flags

---

## 🎯 Expected Behavior

### During the Run
```
[Resharding models: Progress...]
[Model preparation...]
[GA] gen=1 best=... mean=... std=...
[GA] gen=2 best=... mean=... std=...
... (more generations)
[GA] gen=10 best=... mean=... std=...
[Saving final model...]
```

### After Completion
```bash
✅ No CUDA errors
✅ 10 [GA] lines in log
✅ Final model saved
✅ Ready for analysis
```

---

## 🔍 Troubleshooting

### If you see "CUDA" error again:
```bash
# Make sure you used: --no-merge-cuda
grep "no-merge-cuda" your_command_history
# Should find it
```

### If you see "GPU" messages:
```bash
# Check that --num-gpus is NOT specified
# (It defaults to 0 for CPU-only)
```

### If only 1 generation logs:
```bash
# Make sure you used: --strategy serial
# --strategy pool would batch callbacks
```

---

## 📈 Next Steps

### While Run Completes (15-20 min)
1. Monitor: `tail -f workspace/ga_10gen_mac_cpu_run.log | grep "\[GA\]"`
2. Check: Verify [GA] lines appear without CUDA errors
3. Read: Check the documentation we created

### After Run Completes
```bash
# Verify success
grep "\[GA\]" workspace/ga_10gen_mac_cpu_run.log | wc -l
# Should show: 10

# Extract metrics
python scripts/research/plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log \
  --output convergence.png \
  --csv metrics.csv \
  --table

# View results
cat ga_metrics.csv | column -t -s,
```

---

## 💡 Mac vs GPU Systems

| Aspect | Mac M1/M2 | GPU System |
|--------|-----------|-----------|
| **Merge device** | CPU | GPU (CUDA) |
| **Eval device** | CPU | GPU |
| **Key flag** | `--no-merge-cuda` | (omit it) |
| **Workers** | 1-2 | Multiple |
| **Speed** | Slower (~20 min) | Faster (~5 min) |
| **Memory** | ~8 GB | More available |

---

## 🎓 What You Learned

1. **Mac doesn't have CUDA** - Need `--no-merge-cuda` flag
2. **PyTorch detects CUDA at runtime** - Error happens during merge
3. **Strategy matters** - `--strategy serial` for logging on CPU
4. **Config is generic** - Works on any system with right flags

---

## 📋 Complete Mac Setup Checklist

When running GA on Mac, always use:

- [ ] `--no-merge-cuda` ← CPU-only execution
- [ ] `--strategy serial` ← Per-generation logging
- [ ] `--num-workers 1` ← Single CPU worker
- [ ] `--storage-path` ← Required
- [ ] `--save-final-model` ← Optional but recommended

---

## 🚀 Quick Reference

```bash
# ✅ CORRECT: Full command for Mac
python -m mergekit.scripts.evolve_ga config.yml \
  --storage-path workspace/storage \
  --save-final-model \
  --no-merge-cuda \
  --strategy serial \
  --num-workers 1

# ❌ WRONG: Would fail with CUDA error
python -m mergekit.scripts.evolve_ga config.yml \
  --strategy serial
  # Missing --no-merge-cuda for Mac!
```

---

## 📊 Current Run Status

**Command**:
```
evolve_ga workspace/ga_10gen_mac_cpu.yml \
  --no-merge-cuda \
  --strategy serial
```

**Started**: 2025-10-19 ~12:40 UTC
**Expected Finish**: ~13:00 UTC (~15-20 min)
**Log**: `workspace/ga_10gen_mac_cpu_run.log`
**Status**: ⏳ Running...

---

**Key Takeaway**: Always add `--no-merge-cuda` when running mergekit on Mac!
