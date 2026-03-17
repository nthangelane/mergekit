# Unified Experiment Tracking for Evolve GA

The mergekit evolve GA feature now supports flexible experiment tracking that allows users to choose between different tracking backends or no tracking at all. This addresses the previous dependency on Weights & Biases (W&B) that required external hosting.

## Supported Tracking Backends

### 1. No Tracking (Default)
When no tracking is specified, the system runs without logging experiments.

```bash
python -m mergekit.scripts.evolve_ga config.yml output_dir --no-wandb --no-mlflow
```

### 2. Weights & Biases (W&B)
Traditional cloud-based experiment tracking (requires W&B account).

```bash
python -m mergekit.scripts.evolve_ga config.yml output_dir \
    --wandb \
    --wandb-project "my-merge-project" \
    --wandb-entity "my-username"
```

### 3. MLflow (Local/Self-hosted)
Local or self-hosted experiment tracking without external dependencies.

```bash
# Local tracking (creates ./mlruns directory)
python -m mergekit.scripts.evolve_ga config.yml output_dir \
    --mlflow \
    --mlflow-experiment "tiny-model-merge"

# Custom tracking URI
python -m mergekit.scripts.evolve_ga config.yml output_dir \
    --mlflow \
    --mlflow-experiment "tiny-model-merge" \
    --mlflow-tracking-uri "http://localhost:5000"
```

## CLI Options

### W&B Options
- `--wandb/--no-wandb`: Enable/disable W&B tracking
- `--wandb-project`: W&B project name (default: "mergekit-evolve-ga")
- `--wandb-entity`: W&B entity/username

### MLflow Options
- `--mlflow/--no-mlflow`: Enable/disable MLflow tracking
- `--mlflow-experiment`: MLflow experiment name (default: "mergekit-evolve-ga")
- `--mlflow-tracking-uri`: MLflow tracking server URI (default: "./mlruns")

## What Gets Tracked

Both tracking backends capture:

1. **Configuration**: Complete merge configuration and GA parameters
2. **Population Statistics**: Generation-by-generation statistics
   - Population size
   - Mean fitness
   - Best fitness
   - Merge-method counts, failure counts, and success rates for multi-method runs
3. **Best Individual**: Details of the best-performing merge configuration
4. **Metrics**: All evaluation metrics from lm-eval tasks
5. **Artifacts**: Best merge configuration YAML files

For GA runs, mergekit now also writes:

- `ga_history.csv`: generation summary table
- `ga_method_history.csv`: per-generation merge-method counts, successes, failures, and success rates
- `ga_summary.txt`: compact human-readable history summary
- `mlflow_run_info.md`: tracking URI, run ID, local store path, and MLflow review URL

## Installation Requirements

### For W&B Tracking
```bash
pip install wandb
# or install with optional dependencies
pip install mergekit[evolve-ga]
```

### For MLflow Tracking
```bash
pip install mlflow
# or install with optional dependencies
pip install mergekit[evolve-ga]
```

Note: If you encounter pyarrow compatibility issues with MLflow, try:
```bash
pip install --upgrade pyarrow mlflow
```

## MLflow Server Hosting Options

### 1. Local Tracking (Default)
MLflow runs locally and stores experiments in a `./mlruns` directory:

```bash
python -m mergekit.scripts.evolve_ga config.yml output --mlflow
# Creates ./mlruns directory for experiment storage
```

To view results:
```bash
mlflow ui --backend-store-uri ./mlruns
# Opens web UI at http://localhost:5000
```

### 2. Remote MLflow Server
For team collaboration or centralized tracking, you can host MLflow on a server:

#### Server Requirements
- **Python 3.8+** with MLflow installed
- **Database backend** (optional but recommended for production):
  - PostgreSQL, MySQL, or SQLite
- **Artifact storage** (optional for large artifacts):
  - AWS S3, Google Cloud Storage, Azure Blob Storage, or local filesystem
- **Network access** from client machines

#### Basic Server Setup
```bash
# On the server machine
pip install mlflow psycopg2-binary  # Add database driver if using PostgreSQL

# Start MLflow server
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-mlflow-bucket/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

#### Client Configuration
```bash
# Point evolve_ga to your MLflow server
python -m mergekit.scripts.evolve_ga config.yml output \
    --mlflow \
    --mlflow-tracking-uri "http://your-server:5000" \
    --mlflow-experiment "distributed-merge-experiment"
```

#### Docker Deployment Example
```dockerfile
# Dockerfile for MLflow server
FROM python:3.11-slim

RUN pip install mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://user:password@db:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

#### Production Considerations
- **Authentication**: Use MLflow's built-in authentication or reverse proxy with auth
- **HTTPS**: Enable SSL/TLS for secure communication
- **Backup**: Regular backup of backend database and artifacts
- **Monitoring**: Monitor server health and storage usage
- **Scaling**: Consider load balancing for high-traffic scenarios

## Configuration

### Environment Variables

Both tracking backends support configuration via environment variables:

#### W&B Environment Variables
```bash
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="my-username"
export WANDB_API_KEY="your-api-key"
```

#### MLflow Environment Variables
```bash
export MLFLOW_TRACKING_URI="http://your-server:5000"
export MLFLOW_EXPERIMENT_NAME="my-experiment"
export MLFLOW_UI_URL="http://127.0.0.1:5001"  # Optional local UI base URL override
# For S3 artifact storage
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### Configuration Precedence
Settings are applied in this order (higher priority overrides lower):
1. CLI arguments (`--wandb-project`, `--mlflow-tracking-uri`)
2. Environment variables (`WANDB_PROJECT`, `MLFLOW_TRACKING_URI`)
3. Default values

## Architecture

The unified tracking system is implemented through an abstract `ExperimentTracker` base class with three concrete implementations:

- `WandBTracker`: Integrates with Weights & Biases
- `MLflowTracker`: Integrates with MLflow
- `NoOpTracker`: No-operation tracker for when tracking is disabled

This design allows easy extension to support additional tracking backends in the future.

## Quick Setup Guide

### Scenario 1: Local Development (No External Dependencies)
```bash
# Install mergekit with MLflow support
pip install mergekit[evolve-ga]

# Run with local MLflow tracking
python -m mergekit.scripts.evolve_ga config.yml output --mlflow

# View results
mlflow ui
```

Each tracked run also drops an `mlflow_run_info.md` file into its storage path.
When local file-backed MLflow is used, that file includes:

- the workspace `mlruns` path
- the suggested `mlflow ui --backend-store-uri ...` command
- a direct review URL such as `http://127.0.0.1:5000/#/experiments/<id>/runs/<run_id>`

### Scenario 2: Team Collaboration with Remote MLflow
```bash
# Install mergekit
pip install mergekit[evolve-ga]

# Run with remote server
python -m mergekit.scripts.evolve_ga config.yml output \
    --mlflow --mlflow-tracking-uri "http://team-server:5000"
```

### Scenario 3: Cloud-based with W&B
```bash
# Install mergekit
pip install mergekit[evolve-ga]

# Login to W&B
wandb login

# Run with W&B tracking
python -m mergekit.scripts.evolve_ga config.yml output \
    --wandb --wandb-project "research-project"
```

### Scenario 4: No Tracking (Fastest)
```bash
# Install base mergekit
pip install mergekit

# Run without tracking
python -m mergekit.scripts.evolve_ga config.yml output
```

## Examples

### Basic Usage (No Tracking)
```bash
python -m mergekit.scripts.evolve_ga examples/evolve_ga_tiny.yml storage/output
```

### Local MLflow Tracking
```bash
python -m mergekit.scripts.evolve_ga examples/evolve_ga_tiny.yml storage/output \
    --mlflow --mlflow-experiment "tiny-merge-test"

# View results in MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

### W&B Cloud Tracking
```bash
python -m mergekit.scripts.evolve_ga examples/evolve_ga_tiny.yml storage/output \
    --wandb --wandb-project "research-merges"
```

## Migration from W&B-only

Previous versions of mergekit required W&B for experiment tracking. The new unified system maintains backward compatibility:

- Old configs with W&B settings still work with `--wandb` flag
- New users can choose MLflow for local tracking without external accounts
- Researchers can disable tracking entirely for quick experiments

This change makes mergekit's evolutionary optimization more accessible to users who prefer local or self-hosted experiment tracking solutions.
