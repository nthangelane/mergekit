# Thesis Experiment Suite

This directory is the canonical home for the formal thesis experiment presets.

## Layout

- `manifest.yaml`: suite manifest that points at the execution-group manifests
- `local_mac/`: experiments intended for the development Mac
- `eks_gpu/`: GPU-backed experiments intended for the EKS Ray cluster

## Execution Split

- `local_mac`: cheap validation runs on the development machine
- `eks_gpu`: cloud GPU runs with a target scale-out budget of `10` GPUs and `24 GiB+` VRAM

Tensor parallel evaluation now exists in the GA runner, but it only works when a Ray GPU worker pod advertises more than one GPU on a single node. The EKS bootstrap flow supports that with `--gpu-gpus-per-node` and `--gpu-worker-gpus`.

## Pre-AWS Gate

Before deploying a new evo commit, run:

```bash
scripts/preflight_evo_aws.sh
```

The required checks, artifacts, and one-worker promotion rule are documented in
[`PRE_AWS_RUN_GATE.md`](PRE_AWS_RUN_GATE.md).

## Status Tracking

Use the tracker script from the repo root:

```bash
python scripts/track_thesis_experiments.py
python scripts/track_thesis_experiments.py --target local_mac
python scripts/track_thesis_experiments.py --target eks_gpu --watch
python scripts/track_thesis_experiments.py --json
```

The tracker combines:

- static metadata from the suite and group manifests
- local artifact status from `workspace/thesis/...`
- live Kubernetes and RayJob state for the EKS group when `kubectl` is configured
