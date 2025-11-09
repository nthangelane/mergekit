# Running mergekit GA on Amazon EKS

This guide describes an optional workflow for launching the genetic algorithm (`mergekit-evolve-ga`) on a Ray cluster that runs inside an Amazon Elastic Kubernetes Service (EKS) environment. The deployment uses only open-source tooling and keeps the core `mergekit` code untouched—everything lives under the `deploy/` directory so it can be adopted or ignored as needed.

## Prerequisites

Before you begin, make sure you have the following:

- An AWS account with IAM permissions to create and delete EKS clusters, VPC components, and CloudFormation stacks.
- Local tooling: [`aws`](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [`eksctl`](https://eksctl.io/introduction/installation/), [`kubectl`](https://kubernetes.io/docs/tasks/tools/), [`helm`](https://helm.sh/docs/intro/install/), and [`docker`](https://docs.docker.com/engine/install/).
- A container registry that your EKS nodes can reach (Amazon ECR recommended). You’ll push the MergeKit GA image there and set `MERGEKIT_IMAGE=<registry>/mergekit-ga:tag` before bootstrapping.
- Python dependencies installed via `pip install -e .[eks]` (installs `boto3`, `kubernetes`, and `python-dotenv`).
- Optional: create a `.env.eks` file in the repository root to centralise configuration (e.g. `AWS_REGION=us-east-1`).

The helper CLI automatically checks for the required binaries before it runs anything. When a tool is missing it stops immediately and prints OS-aware installation hints so you can remediate without digging through docs. Typical install commands:

- **macOS (Homebrew):**
	```sh
	brew install awscli eksctl kubernetes-cli helm
	```
- **Ubuntu/Debian:**
	```sh
	sudo apt-get update && sudo apt-get install -y curl unzip
	curl -sS https://raw.githubusercontent.com/weaveworks/eksctl/main/install.sh | sudo bash
	curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
	curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
	unzip awscliv2.zip
	sudo ./aws/install
	sudo apt-get install -y kubectl  # or follow https://kubernetes.io/docs/tasks/tools/
	```
- **Windows (PowerShell + Chocolatey):**
	```powershell
	choco install -y awscli eksctl kubernetes-cli kubernetes-helm
	```

Feel free to install tools manually if you already have organisation-specific workflows; the CLI simply verifies that `aws`, `eksctl`, `kubectl`, and `helm` are discoverable in your `PATH`.

> **Costs:** Running EKS incurs AWS charges for the control plane, managed nodes, EBS volumes, and any load balancers you create. Shut everything down with the teardown command when you are done.

## Build and push the MergeKit image

The provided Dockerfile bundles mergekit with the GA extras. Build it locally and push to a registry your EKS cluster can access (Amazon ECR recommended):

```sh
# From the repo root
export AWS_ACCOUNT_ID=<aws_account_id>
export AWS_REGION=us-east-1  # or your preferred region
aws ecr get-login-password --region "$AWS_REGION" \
	| docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

./deploy/build_push.sh mergekit-ga latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

export MERGEKIT_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mergekit-ga:latest"
```

If the `registry-prefix` argument is omitted the script only performs a local build. Make sure you authenticate to your registry (for ECR: `aws ecr get-login-password | docker login --username AWS --password-stdin <registry>`).

> **Architecture tip:** EKS managed node groups are typically `linux/amd64`. The build script now targets that platform by default so images built on Apple Silicon (arm64) still run on the cluster. If you need a different architecture, set `PLATFORM=<os/arch>` before running the script (for example `PLATFORM=linux/arm64 ./deploy/build_push.sh …`). Seeing `exec format error` in Ray pod logs almost always means the image architecture and node architecture don’t match.

## Deploy Ray into EKS

All Kubernetes manifests live in `deploy/`:

- `ray-values.yaml` – Helm values for the Ray chart (head CPU nodes, optional GPU workers, LoadBalancer service).
- `ray-cluster.yaml` – Creates a `RayCluster` custom resource that references the MergeKit image.
- `ray-job.yaml` – Submits a GA run using the `examples/evolve_ga_m1_micro.yml` config.
- `storage.yaml` – PersistentVolumeClaims for Hugging Face cache data and GA artifacts (defaults assume an AWS EFS CSI storage class `efs-sc`; adjust to suit your cluster).

You can apply each resource manually (Helm + kubectl) or use the helper CLI.

### Option A — Helper CLI (`mergekit-eks`)

The new console script wraps the essential steps so you can bootstrap, submit work, and tear down when finished.

```sh
# Install Python dependencies (once)
pip install -e .[eks]

# Create cluster resources
MERGEKIT_IMAGE=$MERGEKIT_IMAGE mergekit-eks bootstrap --cluster-name mergekit-ga --region us-east-1

# Submit the GA workload
MERGEKIT_IMAGE=$MERGEKIT_IMAGE mergekit-eks submit --region us-east-1

# Inspect status
mergekit-eks status --region us-east-1

# Remove everything
MERGEKIT_IMAGE=$MERGEKIT_IMAGE mergekit-eks teardown --cluster-name mergekit-ga --region us-east-1
```

All commands accept `--dry-run` to print the exact `eksctl`, `helm`, and `kubectl` invocations without executing them. Namespaces default to `mergekit`; override with `--namespace` if needed. The status command prints the Ray pods and the EKS cluster metadata (via the AWS SDK) when not in dry-run mode.

### Option B — Manual commands

If you prefer to manage resources yourself, run the following (matching the helper CLI sequence):

```sh
helm repo add kuberay https://ray-project.github.io/kuberay-helm
helm repo update
helm upgrade --install kuberay-operator kuberay/kuberay-operator -f deploy/ray-values.yaml --namespace mergekit --create-namespace
kubectl apply -n mergekit -f deploy/storage.yaml
MERGEKIT_IMAGE=$MERGEKIT_IMAGE envsubst < deploy/ray-cluster.yaml | kubectl apply -n mergekit -f -
MERGEKIT_IMAGE=$MERGEKIT_IMAGE envsubst < deploy/ray-job.yaml | kubectl apply -n mergekit -f -
```

Check progress with:

```sh
kubectl get rayclusters -n mergekit
kubectl get rayjobs -n mergekit
```

## Storage considerations

The manifests reference two persistent volume claims—`hf-cache-pvc` and `artifacts-pvc`—so that model checkpoints and GA outputs survive restarts. Create them ahead of time or adjust the manifests to match your storage class (e.g. by enabling AWS EBS CSI dynamic provisioning).

## Customising jobs

- Update `deploy/ray-job.yaml` to point at a different GA config or to pass additional CLI flags.
- Adjust `deploy/ray-values.yaml` if you need to override the operator image or feature flags (defaults should work for most flows).
- Swap the Dockerfile base image or extras if you only require a subset of the tooling.

## Cleanup checklist

1. Delete the Ray job (`kubectl delete -n mergekit -f deploy/ray-job.yaml`).
2. Remove the Ray cluster (`kubectl delete -n mergekit -f deploy/ray-cluster.yaml`).
3. Uninstall the Helm release (`helm uninstall kuberay-operator -n mergekit`).
4. Delete the PVCs (`kubectl delete -n mergekit -f deploy/storage.yaml`).
5. Tear down the EKS cluster (`eksctl delete cluster --name mergekit-ga --region us-east-1`).
6. Remove any persistent volumes or ECR images you no longer need.

Keeping a `dry-run` first ensures you are comfortable with every action before executing it against your AWS account.
