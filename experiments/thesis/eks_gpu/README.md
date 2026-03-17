# Thesis EKS GPU Track

This group contains the cloud GPU experiments. The target cluster profile for the thesis is `20-30` Ray workers on `24 GiB+` VRAM GPUs, while keeping a per-experiment cap of `10` GPUs.

## Included Experiments

- `exp01_tiny_controlled_merge`: fast cloud smoke baseline before larger GPU runs
- `exp02_pythia28b_base_chat`
- `exp03_qwen25_3b_multilingual_merge`
- `exp04_mistral7b_general_code`
- `exp05_llama3_8b_same_family`
- `exp06_pythia160m_phase2_ties_starter`

## Target Cluster Profile

- Preferred minimum profile: `g6e.12xlarge` nodes (`4` GPUs per node, `48 GiB` each) or better
- Acceptable lower-VRAM floor for the smaller runs: `g6` / NVIDIA L4 with `24 GiB`
- Tensor parallel runs require multi-GPU Ray worker pods on one node

## Throughput Profiles

Use these profiles when the goal is population throughput and broad parallel evaluation.

### 20 worker profile

```bash
python -m mergekit.scripts.run_on_eks bootstrap \
  --cluster-name mergekit-ga \
  --ray-cluster-name mergekit-ga \
  --region us-east-1 \
  --scale-profile throughput-20
```

That yields `20` GPU Ray workers.

### 28 worker profile

```bash
python -m mergekit.scripts.run_on_eks bootstrap \
  --cluster-name mergekit-ga \
  --ray-cluster-name mergekit-ga \
  --region us-east-1 \
  --scale-profile throughput-28
```

That yields `28` GPU Ray workers.

## Tensor Parallel Profile

Use this when you need fewer but wider workers for larger 7B or 8B runs:

```bash
python -m mergekit.scripts.run_on_eks bootstrap \
  --cluster-name mergekit-ga \
  --ray-cluster-name mergekit-ga \
  --region us-east-1 \
  --scale-profile tp2-large-model
```

That shape exposes `10-14` Ray GPU workers, each with `2` GPUs available for tensor parallel evaluation. Keep each submitted experiment capped at `--num-gpus 10`.

You can still override individual values after the profile. For example:

```bash
python -m mergekit.scripts.run_on_eks bootstrap \
  --cluster-name mergekit-ga \
  --ray-cluster-name mergekit-ga \
  --region us-east-1 \
  --scale-profile throughput-20 \
  --gpu-max-nodes 7
```

## Suggested Submit Pattern

```bash
python -m mergekit.scripts.run_on_eks submit \
  --ray-cluster-name mergekit-ga \
  --job-name thesis-exp05-llama3-8b \
  --config-path /app/experiments/thesis/eks_gpu/exp05_llama3_8b_same_family/config.yml \
  --storage-subpath thesis/eks_gpu/exp05_llama3_8b_same_family \
  --max-fevals 80 \
  --strategy pool \
  --num-gpus 10 \
  --vllm \
  --tensor-parallel-size 2 \
  --merge-cuda \
  --baseline \
  --extra-arg=--i-understand-the-depths-of-the-evils-i-am-unleashing
```

## Tracking

The tracker reports the current worker count against the manifest target:

```bash
python scripts/track_thesis_experiments.py --target eks_gpu
python scripts/track_thesis_experiments.py --target eks_gpu --watch
```
