# Experiment Tracking Quick Reference

## Installation

```bash
# Install with experiment tracking support
pip install mergekit[evolve-ga]
```

## Usage Examples

### No Tracking (Default)
```bash
python -m mergekit.scripts.evolve_ga config.yml output
```

### Local MLflow
```bash
python -m mergekit.scripts.evolve_ga config.yml output --mlflow
mlflow ui  # View results at http://localhost:5000
```

### Remote MLflow Server
```bash
python -m mergekit.scripts.evolve_ga config.yml output \
    --mlflow --mlflow-tracking-uri "http://server:5000"
```

### W&B Cloud
```bash
wandb login  # One-time setup
python -m mergekit.scripts.evolve_ga config.yml output \
    --wandb --wandb-project "my-project"
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mlflow` | Enable MLflow tracking | disabled |
| `--mlflow-experiment` | MLflow experiment name | "mergekit-evolve-ga" |
| `--mlflow-tracking-uri` | MLflow server URI | "./mlruns" |
| `--wandb` | Enable W&B tracking | disabled |
| `--wandb-project` | W&B project name | "mergekit-evolve-ga" |
| `--wandb-entity` | W&B username/entity | None |

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://server:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Default experiment name | `my-experiment` |
| `WANDB_PROJECT` | Default W&B project | `my-project` |
| `WANDB_ENTITY` | Default W&B entity | `username` |

## What Gets Tracked

- Configuration parameters
- Population statistics per generation
- Best individual performance
- All evaluation metrics
- Best merge configuration files

## MLflow Server Setup

### Basic Server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### With Database Backend
```bash
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --host 0.0.0.0 --port 5000
```

### Docker
```bash
docker run -p 5000:5000 \
    -v $(pwd)/mlruns:/mlflow/mlruns \
    python:3.11-slim \
    sh -c "pip install mlflow && mlflow server --host 0.0.0.0"
```
