# EKS Experiment Write-Up (2026-03-15)

## What The Ray UI Is

The Ray dashboard is exposed only inside the Kubernetes cluster through the head service:

- service: `mergekit-ga-head-svc`
- namespace: `mergekit`
- dashboard port: `8265`
- Ray client port: `10001`

Use a local port-forward from the machine where `kubectl` is configured:

```bash
kubectl port-forward -n mergekit svc/mergekit-ga-head-svc 8265:8265 10001:10001
```

Then open:

- dashboard: `http://127.0.0.1:8265`

Useful live commands while a job is running:

```bash
kubectl get rayjobs -n mergekit -w
kubectl get pods -n mergekit -o wide
ray job list --address http://127.0.0.1:8265
kubectl logs -n mergekit <rayjob-pod-name> -f
kubectl exec -n mergekit <head-pod> -c ray-head -- ray status --address=127.0.0.1:6379
```

## Where Results Are Stored

All experiment artifacts are written to the shared EFS-backed mount at:

- `/data/artifacts/configs` for submitted config files
- `/data/artifacts/runs/<run-name>` for run outputs

Inside each run directory, the important files are typically:

- `ga_history.csv`
- `ga_summary.txt`
- `baseline_results.csv` when baseline mode is enabled
- `failed_genotypes.csv`
- `failed_genotype_blacklist.csv`
- `best_config.yaml` when a best config is emitted
- `merged/` for merged model outputs if saving is enabled
- `transformers_cache/` for model downloads cached onto shared storage

Current live examples:

- `/data/artifacts/runs/qwen25-3b-stable-gpu`
- `/data/artifacts/runs/qwen25-3b-stable-cpu-final`

To copy a finished run out of the cluster:

```bash
kubectl cp mergekit/<head-pod>:/data/artifacts/runs/<run-name> ./workspace/<run-name> -c ray-head
```

## Current Runtime Finding

The GPU path is operational and writes artifacts. The stable GPU smoke completed as a RayJob in about 16.3 minutes:

- start: `2026-03-15T12:38:56Z`
- end: `2026-03-15T12:55:16Z`
- run path: `/data/artifacts/runs/qwen25-3b-stable-gpu`
- `ga_history.csv` reported `eval_seconds=960.6743400096893`

That run is not thesis-ready because all four evaluations still failed inside the GA with `eval:IndexError:4`.

The CPU comparison path is not yet complete. After fixing scheduling so the head does not take CPU work, the CPU worker still failed before emitting `ga_history.csv`. The main finding is operational:

- `m6i.xlarge` is too small for this 3B CPU reference path
- the head must advertise `num-cpus: 0`
- the stable config should be submitted from shared storage, not from `/app/examples`, to avoid image drift

## Required Cluster Shape

### Minimum credible 3B research shape

Use this shape for thesis-facing 3B experiments:

- CPU nodes: `2 x m6i.2xlarge`
- GPU nodes: `2 x g6.2xlarge`
- head pod: `num-cpus: 0`
- CPU worker memory limit: at least `24Gi`
- GPU workers: `--num-gpus 2` or more for the main GA runs

This shape is enough for clean 3B pilot work, but it is not the right shape if the goal is to finish the full 3B experiment suite inside a strict 3-hour window.

### Shape to finish the 3B suite in under 3 hours

Use this if the target is to finish the full 3B must-complete suite in one session:

- CPU nodes: `2 x m6i.2xlarge`
- GPU nodes: `4 x g6.2xlarge`
- one CPU worker reserved for the CPU reference run
- up to `4` GPU workers available for GA evaluation parallelism or paired seeded runs

Why this shape:

- the account currently has a `5` instance quota for on-demand `G/VT` instances, so `4` GPU nodes fit
- a single `g6.xlarge` smoke took about 16 minutes for `4` fevals
- the 3-hour target is realistic only if the pilot and main seeded runs use multiple GPU workers instead of one

### 7B pilot shape

Treat 7B as a follow-on phase, not part of the under-3-hour 3B package:

- CPU nodes: `2 x m6i.4xlarge`
- GPU nodes: `2 x g6e.2xlarge` minimum

That is a viable 7B pilot shape, but it should be scheduled only after the 3B suite is stable.

## Experiment Set

### Must-complete 3B experiments

1. GPU uplift confirmation
   - config: [`examples/qwen25_3b_eks_stable_smoke.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_3b_eks_stable_smoke.yml)
   - goal: prove the GPU path is faster than the CPU path on the same workload
   - completion: both runs finish cleanly, GPU wall-clock is lower, and GPU `eval_seconds` is lower

2. Benchmark-aligned 3B smoke
   - config: [`examples/qwen25_3b_eks_scale.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_3b_eks_scale.yml)
   - goal: validate the thesis-facing metric mix with no evaluator crashes
   - completion: one clean smoke run with no `IndexError`, no `ActorUnavailableError`, and complete artifacts

3. 3B pilot seed A
   - config: [`examples/qwen25_3b_eks_scale.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_3b_eks_scale.yml)
   - suggested shape: `population_size=8`, `max-fevals=8`, `limit=16`, baseline enabled

4. 3B pilot seed B
   - same as experiment 3 with a second seed

5. 3B main seed A
   - config: [`examples/qwen25_3b_eks_scale.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_3b_eks_scale.yml)
   - suggested shape: `population_size=8`, `max-fevals=16`, `limit=32`, baseline enabled

6. 3B main seed B
   - same as experiment 5 with a second seed

7. Thesis export
   - aggregate `ga_history.csv`, `baseline_results.csv`, and selected best configs into thesis tables and plots

### Deferred 7B experiment

8. 7B pilot
   - config: [`examples/qwen25_7b_eks_scale.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_7b_eks_scale.yml)
   - completion: one clean 7B smoke or pilot after the 3B suite is stable

## Under-3-Hour Schedule

The under-3-hour target is realistic for the 3B suite only, not for the combined 3B and 7B program.

### Feasible 3B schedule on `2 x m6i.2xlarge` and `4 x g6.2xlarge`

1. Experiment 1, CPU and GPU uplift confirmation
   - estimated wall-clock: `45-60 min`
   - run CPU and GPU reference jobs in parallel where possible

2. Experiment 2, benchmark-aligned 3B smoke
   - estimated wall-clock: `20-25 min`

3. Experiments 3 and 4, pilot seeds
   - estimated wall-clock: `35-40 min` if parallelized across available GPU workers

4. Experiments 5 and 6, main seeds
   - estimated wall-clock: `50-60 min` if parallelized across available GPU workers

5. Experiment 7, export and artifact collation
   - estimated wall-clock: `10-15 min`

Estimated total:

- `160-200 min` if run strictly one after another
- `145-175 min` if the seeded GPU runs are paired or parallelized

That means the under-3-hour target is realistic only with the `4 x g6.2xlarge` shape and only for the 3B suite.

## Cost Estimate

### Pricing inputs used

- EKS control plane: `$0.10` per cluster hour
- EC2 on-demand prices were queried from the AWS Pricing API for `us-east-1`
- gp3 EBS estimate: `$0.08` per GB-month
- EFS standard estimate: `$0.30` per GB-month

The dominant cost is EC2. EBS and EFS are included as small approximations for a 3-hour run; data transfer and taxes are excluded.

### 3-hour cluster cost envelopes

1. Current smoke shape
   - shape: `2 x m6i.xlarge` and `1 x g6.xlarge`
   - estimated 3-hour cost: about `$4.15`

2. Minimum credible 3B shape
   - shape: `2 x m6i.2xlarge` and `2 x g6.2xlarge`
   - estimated 3-hour cost: about `$8.90`
   - good for pilot work, not ideal for finishing the whole 3B suite under 3 hours

3. Under-3-hour 3B shape
   - shape: `2 x m6i.2xlarge` and `4 x g6.2xlarge`
   - estimated 3-hour cost: about `$15.01`
   - this is the recommended shape if the goal is to finish the whole 3B must-complete suite in a single 3-hour session

4. 7B pilot shape
   - shape: `2 x m6i.4xlarge` and `2 x g6e.2xlarge`
   - estimated 3-hour cost: about `$18.84`
   - use this only after the 3B suite is stable

## Recommendation

Use the 3B under-3-hour shape for the next research session:

- `2 x m6i.2xlarge`
- `4 x g6.2xlarge`
- head `num-cpus: 0`
- CPU worker memory limit increased beyond the current `14Gi`
- submit configs from `/data/artifacts/configs`

Do not include the 7B pilot in the same 3-hour window.
The 7B phase should stay blocked until the 3B suite is producing clean, repeatable artifacts and the benchmark stack is stable.
