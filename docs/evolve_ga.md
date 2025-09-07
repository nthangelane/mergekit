# mergekit-evolve-ga

`mergekit-evolve-ga` is a script that uses a Genetic Algorithm (GA) to optimize the parameters of a merge against model metrics measured by EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). It mirrors `mergekit-evolve` (which uses CMA-ES) in how it merges, schedules work (Ray), and evaluates models, differing only in the optimizer.

## CLI

```
mergekit-evolve-ga [OPTIONS] --storage-path PATH GENOME_CONFIG_PATH
```

Key options:

- `--population-size`: population size (default 32)
- `--elite-fraction`: fraction of top individuals carried over (default 0.125)
- `--mutation-rate`: per-gene mutation probability (default 0.15)
- `--mutation-sigma`: stddev of Gaussian mutation noise (default 0.05)
- `--crossover`: `arithmetic` or `uniform` (default `arithmetic`)
- `--tournament-size`: tournament size for selection (default 4)
- `--max-fevals`: maximum evaluations before stopping
- `--timeout`: optional time budget in seconds

Shared options with `mergekit-evolve`:

- `--strategy {pool,buffered,serial}`: evaluation scheduling
- `--vllm`: evaluate with vLLM backend
- `--in-memory`: in-memory merges (pool strategy only)
- `--wandb`: enable Weights & Biases logging
- `--num-gpus`, `--batch-size`, `--reshard`, `--trust-remote-code`, etc.

## Notes

- Writes the best-so-far configuration to `storage_path/best_config.yaml`.
- If `--save-final-model` is set (default true), saves the final best merge to `storage_path/final_model`.
- `--reshard` converts inputs to single-shard safetensors for faster merges and is enabled by default.

## Installation

Install the package and GA extras:

```
pip install -e .[evolve-ga]
```

To use CMA-ES instead (or both), install:

```
pip install -e .[evolve]
```
