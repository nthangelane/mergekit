# Thesis Local Mac Track

This group is for the small validation run that should execute on the development machine before spending cloud GPU time.

## Included Experiment

- `exp01_tiny_controlled_merge`

## Suggested Run

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/exp01_tiny_controlled_merge/config.yml \
  --storage-path workspace/thesis/local_mac/exp01_tiny_controlled_merge \
  --max-fevals 480 \
  --strategy pool \
  --num-gpus 0 \
  --num-workers 4 \
  --no-vllm
```
