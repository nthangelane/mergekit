# Experiment 5: Pythia-70M Phase-2 Adaptive Probe

This preset is the first local run that exercises the phase-2 search-control
features on the known-safe Pythia-70M pair.

It keeps the local-machine rule intact:
- tiny models only
- serial CPU-only evaluation
- narrow method family: `passthrough`, `linear`, `slerp`

What it adds on top of the earlier local validation presets:
- staged evaluation (`stage1` + `stage2`)
- adaptive method probabilities
- explorer slots and diversity-aware parent selection
- intrinsic gene-diversity bonuses
- behavior probes with early rejection for degenerate outputs
- candidate-level history in the workspace

Suggested smoke run:

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp05_pythia70m_phase2_adaptive_probe/config.yml \
  --storage-path /tmp/mergekit-ga-runs/pythia70m-phase2-probe-smoke \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --population-size 4 \
  --max-fevals 8 \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 42
```
