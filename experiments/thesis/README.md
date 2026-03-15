# Thesis Experiment Suite

This directory is the canonical home for the formal thesis experiment presets.

## Layout

- `manifest.yaml`: experiment registry with IDs, job names, storage paths, and run settings
- `exp01_tiny_controlled_merge/config.yml`
- `exp02_pythia28b_base_chat/config.yml`
- `exp03_qwen25_3b_multilingual_merge/config.yml`
- `exp04_mistral7b_general_code/config.yml`
- `exp05_llama3_8b_same_family/config.yml`

## Status Tracking

Use the tracker script from the repo root:

```bash
python scripts/track_thesis_experiments.py
python scripts/track_thesis_experiments.py --watch
python scripts/track_thesis_experiments.py --json
```

The tracker combines:

- static experiment metadata from `manifest.yaml`
- local artifact status from `workspace/thesis/...`
- live Kubernetes and RayJob state from the `mergekit` namespace when `kubectl` is configured

## Current Concurrency Assessment

Assessment date: `2026-03-15`

Observed cluster shape:

- Ray head: `m6i.xlarge`
- Ray CPU worker pool: `1 x m6i.2xlarge`, `4 CPU`, `16-24 GiB`
- Ray GPU worker pool: `1 x g6.xlarge`, `1 x NVIDIA L4`, `24 GiB VRAM`, container request `12 GiB`, limit `14 GiB`

Conclusion:

- Running all 5 thesis experiments at the same time is not realistic on the current cluster.
- At most 1 CPU-oriented run and 1 GPU-backed small or 3B run can be active without immediate contention.
- Experiments 4 and 5 should be treated as blocked on larger GPU capacity and, for experiment 5, gated-model access.

Recommended schedule:

1. Run `exp01` locally or on CPU.
2. Run `exp02` and `exp03` one at a time on the current EKS cluster.
3. Scale GPU capacity before attempting `exp04`.
4. Confirm Hugging Face access and larger GPU capacity before `exp05`.
