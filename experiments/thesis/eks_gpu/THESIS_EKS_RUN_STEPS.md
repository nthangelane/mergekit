# Thesis EKS Run Steps

This document records the operational steps for running the adaptive GA thesis experiment on EKS with pooled Ray workers.

## Scope

- Experiment config: `/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/config.yml`
- Cluster name: `mergekit-ga`
- Namespace: `mergekit`
- Region: `us-east-1`
- Scale profile used here: `quota-safe-g6x4`
- Strategy: `pool`

## Why This Run Exists

`exp07` is the cloud thesis run for the adaptive GA redesign. It uses:

- adaptive operator probabilities
- two-stage evaluation
- semantic crossover
- layered multi-method genome
- stop policy controls
- pooled Ray execution on GPU workers

The intended thesis comparison is across three replication seeds plus a smoke run:

- smoke run for cluster/runtime validation
- `seed11`
- `seed22`
- `seed33`

## Preconditions

Before launching:

- AWS credentials must be valid for the target account
- `aws`, `eksctl`, `kubectl`, `helm`, and `docker` must be installed locally
- ECR login must succeed
- the overlay image must include `ray[default]`
- the EKS storage path `/data/artifacts` must be writable from the Ray head/worker pods
- MLflow should use SQLite on shared storage, not the plain file backend

## 1. Build And Push The Image

Build from the current branch and push to ECR.

```bash
docker build -f deploy/Dockerfile.overlay \
  -t 214109453209.dkr.ecr.us-east-1.amazonaws.com/mergekit-ga:20260320-adaptive-ga-overlay-a214b1a-raydefault .

docker push 214109453209.dkr.ecr.us-east-1.amazonaws.com/mergekit-ga:20260320-adaptive-ga-overlay-a214b1a-raydefault
```

The image used for the current run is:

`214109453209.dkr.ecr.us-east-1.amazonaws.com/mergekit-ga:20260320-adaptive-ga-overlay-a214b1a-raydefault`

## 2. Bootstrap The Cluster

Use the quota-safe profile when the account cannot launch the larger `g6.12xlarge` profile.

```bash
python -m mergekit.scripts.run_on_eks bootstrap \
  --cluster-name mergekit-ga \
  --ray-cluster-name mergekit-ga \
  --region us-east-1 \
  --scale-profile quota-safe-g6x4
```

This profile is designed to yield:

- `4` single-GPU Ray workers on `g6.xlarge`
- `1` GPU per worker pod
- `maxReplicas=5` if quota increases later

## 3. Verify Cluster Health

Check the Ray cluster and the GPU workers before submitting jobs.

```bash
kubectl get raycluster -n mergekit -o wide
kubectl get pods -n mergekit -o wide
kubectl get nodes -o wide
```

Healthy signs:

- Ray head pod is `Running`
- GPU worker pods are `Running`
- `AVAILABLE WORKERS` is nonzero
- GPU nodes advertise `nvidia.com/gpu`

For this run, the useful minimum was `4` available workers.

## 4. Prepare MLflow Tracking

Do not use `file:///data/artifacts/mlruns` for the EKS jobs. Use SQLite on shared storage:

`sqlite:////data/artifacts/mlflow.db`

The experiment should exist before submit:

```bash
kubectl exec -n mergekit mergekit-ga-head-74wmx -- sh -lc 'python - <<\"PY\"
import mlflow
mlflow.set_tracking_uri(\"sqlite:////data/artifacts/mlflow.db\")
exp = mlflow.get_experiment_by_name(\"eks-exp07-pythia160m-adaptive\")
if exp is None:
    exp_id = mlflow.create_experiment(\"eks-exp07-pythia160m-adaptive\")
    print(f\"created:{exp_id}\")
else:
    print(f\"existing:{exp.experiment_id}\")
PY'
```

### Optional: start MLflow UI and port-forward it locally

Inside the Ray head pod:

```bash
kubectl exec -n mergekit mergekit-ga-head-74wmx -- sh -lc \
  'nohup mlflow ui --backend-store-uri sqlite:////data/artifacts/mlflow.db --host 0.0.0.0 --port 5001 >/tmp/mlflow-ui.log 2>&1 &'
```

Local port-forward:

```bash
kubectl port-forward -n mergekit pod/mergekit-ga-head-74wmx 5011:5001
```

Then open:

- [MLflow UI](http://127.0.0.1:5011)
- [MLflow Experiment 1](http://127.0.0.1:5011/#/experiments/1)

## 5. Submit The Smoke Run

Use the EKS submit path with:

- `--strategy pool`
- `--baseline`
- the benchmark-guard override because `piqa` is included
- the SQLite MLflow tracking URI

```bash
python -m mergekit.scripts.run_on_eks submit \
  --ray-cluster-name mergekit-ga \
  --job-name thesis-exp07-pythia160m-smoke \
  --config-path /app/experiments/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/config.yml \
  --storage-subpath thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/smoke-20260320 \
  --max-fevals 24 \
  --strategy pool \
  --num-gpus 4 \
  --baseline \
  --no-save-final-model \
  --extra-arg=--i-understand-the-depths-of-the-evils-i-am-unleashing \
  --extra-arg=--mlflow-tracking-uri \
  --extra-arg=sqlite:////data/artifacts/mlflow.db
```

## 6. Submit The Final Seeded Runs

The three seeded runs are identical except for `--random-seed` and storage subpath.

On the current quota-safe cluster, submit them one at a time. Do not run all three seeds concurrently on the small head node.

```bash
python -m mergekit.scripts.run_on_eks submit \
  --ray-cluster-name mergekit-ga \
  --job-name thesis-exp07-pythia160m-seed11 \
  --config-path /app/experiments/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/config.yml \
  --storage-subpath thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/seed11 \
  --max-fevals 320 \
  --strategy pool \
  --num-gpus 4 \
  --baseline \
  --no-save-final-model \
  --extra-arg=--random-seed \
  --extra-arg=11 \
  --extra-arg=--i-understand-the-depths-of-the-evils-i-am-unleashing \
  --extra-arg=--mlflow-tracking-uri \
  --extra-arg=sqlite:////data/artifacts/mlflow.db
```

Repeat for `seed22` and `seed33`.

The batch helper used in this repo is sequential and waits for each RayJob to finish before submitting the next one:

`/Users/nkululekothangelane/Documents/master_research/mergekit/workspace/thesis/eks_gpu/results/20260320-exp07-adaptive/submit_exp07_pool.sh`

## 7. Monitor Progress

### Cluster and RayJobs

```bash
kubectl get rayjobs -n mergekit -o wide
kubectl get pods -n mergekit -o wide
kubectl get raycluster -n mergekit -o wide
```

### Launcher logs

```bash
kubectl logs -n mergekit thesis-exp07-pythia160m-seed11-pwr2k --tail=120
kubectl logs -n mergekit thesis-exp07-pythia160m-seed22-7sb5v --tail=120
kubectl logs -n mergekit thesis-exp07-pythia160m-seed33-xhmpd --tail=120
```

### Shared-storage artifact inspection

```bash
kubectl exec -n mergekit mergekit-ga-head-74wmx -- sh -lc \
  'ls -R /data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis | sed -n "1,200p"'
```

### MLflow run metadata

Each run writes `mlflow_run_info.md` under its storage directory. Example:

`/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/seed11/mlflow_run_info.md`

## 8. Artifact Locations

Shared artifact root:

`/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis`

Per-run directories:

- `/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/smoke-20260320`
- `/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/seed11`
- `/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/seed22`
- `/data/artifacts/thesis/eks_gpu/exp07_pythia160m_adaptive_thesis/seed33`

Expected files:

- `baseline_results.csv`
- `ga_history.csv`
- `ga_candidate_history.csv`
- `ga_method_history.csv`
- `ga_summary.txt`
- `ga_stop_details.json`
- `mlflow_run_info.md`
- `ray_observability.json`

## 9. Current Observations For This Batch

As of `2026-03-20 13:26 SAST`:

- the smoke run failed after baseline during the first pooled GA evaluation
- the smoke failure surfaced as `ActorDiedError` with worker logs showing `SIGBUS` / CUDA copy issues
- `seed11`, `seed22`, and `seed33` all completed baseline successfully
- all three seeded runs entered `Generation 1/20`, but none finished a useful search run
- `seed11` completed generation 1 with `16` failed evaluations, all rejected as `smoke_test:degenerate_output`
- `seed11` then crashed on `ValueError: Method layered_mixed is not enabled in allowed_methods`
- `seed22` and `seed33` were killed by Ray OOM prevention on the head node because three job supervisors plus MLflow UI exceeded the small head-node memory budget
- MLflow tracking is working through SQLite
- MLflow UI is available locally through the head-pod port-forward at [http://127.0.0.1:5011](http://127.0.0.1:5011)

This means the next EKS attempt should change two things before re-submit:

- run one seed at a time or increase the head-node memory budget
- patch the adaptive sampling path so it cannot emit `layered_mixed` unless it is explicitly enabled in `allowed_methods`

## 10. Cost Control And Teardown

GPU nodes incur cost while the Ray cluster is up, even if jobs are stuck or idle.

Minimum cost-control actions after the batch:

- delete or scale down the GPU nodegroup
- delete the RayJobs if they are no longer needed
- stop any local `kubectl port-forward` sessions

Useful checks:

```bash
kubectl get rayjobs -n mergekit -o wide
kubectl get raycluster -n mergekit -o wide
aws eks list-nodegroups --cluster-name mergekit-ga --region us-east-1
```
