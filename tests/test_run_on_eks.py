import types
from pathlib import Path

import yaml
from click.testing import CliRunner

import mergekit.scripts.run_on_eks as run_on_eks

DEPLOY_DIR = run_on_eks.DEPLOY_DIR
cli = run_on_eks.cli


def test_bootstrap_includes_storage_apply():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "bootstrap",
            "--cluster-name",
            "demo",
            "--region",
            "us-west-2",
        ],
    )

    assert result.exit_code == 0, result.output
    storage_cmd = f"kubectl apply -n test-ns -f {DEPLOY_DIR / 'storage.yaml'}"
    assert storage_cmd in result.output
    assert "aws eks update-kubeconfig --name demo --region us-west-2" in result.output
    assert (
        "helm repo add kuberay https://ray-project.github.io/kuberay-helm"
        in result.output
    )
    assert (
        "helm upgrade --install kuberay-operator kuberay/kuberay-operator"
        in result.output
    )
    assert (
        f"ensure managed nodegroup roles include {run_on_eks.EFS_UTILS_POLICY_ARN}"
        in result.output
    )


def test_bootstrap_skips_direct_namespace_creation():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "bootstrap",
            "--cluster-name",
            "demo",
            "--region",
            "us-west-2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "kubectl create namespace" not in result.output
    assert "eksctl create cluster" in result.output
    assert "aws eks update-kubeconfig --name demo --region us-west-2" in result.output
    assert (
        "helm repo add kuberay https://ray-project.github.io/kuberay-helm"
        in result.output
    )
    assert (
        "helm upgrade --install kuberay-operator kuberay/kuberay-operator"
        in result.output
    )


def test_teardown_deletes_storage():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "teardown",
            "--cluster-name",
            "demo",
            "--region",
            "us-west-2",
        ],
    )

    assert result.exit_code == 0, result.output
    delete_cmd = (
        f"kubectl delete -n test-ns -f {DEPLOY_DIR / 'storage.yaml'} --ignore-not-found"
    )
    assert delete_cmd in result.output
    assert "helm uninstall kuberay-operator -n test-ns" in result.output


def test_status_reports_nodes_pods_and_ray_resources():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "status",
            "--cluster-name",
            "demo",
            "--ray-cluster-name",
            "demo-ray",
            "--region",
            "us-west-2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        "kubectl get nodes -o custom-columns=NAME:.metadata.name,INSTANCE:.metadata.labels.node\\.kubernetes\\.io/instance-type,GPU:.status.allocatable.nvidia\\.com/gpu"
        in result.output
    )
    assert "kubectl get pods -n test-ns -o wide" in result.output
    assert "kubectl get rayclusters -n test-ns" in result.output
    assert "kubectl get rayjobs -n test-ns" in result.output
    assert (
        "ray job list --address ray://demo-ray-head-svc.test-ns.svc.cluster.local:10001"
        in result.output
    )


def test_deploy_manifests_are_valid_yaml():
    deploy_dir = Path(DEPLOY_DIR)
    manifests = [
        deploy_dir / "ray-cluster.yaml",
        deploy_dir / "ray-job.yaml",
        deploy_dir / "storage.yaml",
    ]

    for manifest in manifests:
        assert manifest.exists(), f"Missing manifest: {manifest}"
        docs = list(yaml.safe_load_all(manifest.read_text()))
        assert docs, f"Manifest {manifest} is empty"

    cluster = yaml.safe_load((deploy_dir / "ray-cluster.yaml").read_text())
    assert cluster.get("kind") == "RayCluster"
    head_container = cluster["spec"]["headGroupSpec"]["template"]["spec"]["containers"][
        0
    ]
    head_volumes = cluster["spec"]["headGroupSpec"]["template"]["spec"]["volumes"]
    assert head_volumes
    shared_claims = [
        volume["persistentVolumeClaim"]["claimName"]
        for volume in head_volumes
        if "persistentVolumeClaim" in volume
    ]
    assert "mergekit-shared-pvc" in shared_claims
    worker_specs = cluster["spec"]["workerGroupSpecs"]
    worker_commands = [
        spec["template"]["spec"]["containers"][0].get("command")
        for spec in worker_specs
    ]
    assert "command" not in head_container
    assert all(command is None for command in worker_commands)
    assert (
        "curl --fail --silent"
        in head_container["readinessProbe"]["exec"]["command"][-1]
    )
    assert (
        "curl --fail --silent" in head_container["livenessProbe"]["exec"]["command"][-1]
    )
    worker_containers = [
        spec["template"]["spec"]["containers"][0] for spec in worker_specs
    ]
    assert all(
        "curl --fail --silent" in container["readinessProbe"]["exec"]["command"][-1]
        for container in worker_containers
    )
    assert all(
        "curl --fail --silent" in container["livenessProbe"]["exec"]["command"][-1]
        for container in worker_containers
    )

    storage_docs = list(yaml.safe_load_all((deploy_dir / "storage.yaml").read_text()))
    claim_names = {doc["metadata"]["name"] for doc in storage_docs if doc}
    assert {"mergekit-shared-pv", "mergekit-shared-pvc"}.issubset(claim_names)

    ray_job = yaml.safe_load((deploy_dir / "ray-job.yaml").read_text())
    assert ray_job.get("kind") == "RayJob"
    assert ray_job["spec"]["clusterSelector"] == {
        "ray.io/cluster": "__RAY_CLUSTER_NAME__"
    }
    assert ray_job["spec"]["shutdownAfterJobFinishes"] is False
    assert "ttlSecondsAfterFinished" not in ray_job["spec"]


def test_submit_renders_valid_mergekit_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "submit",
            "--ray-cluster-name",
            "demo-ray",
            "--job-name",
            "demo-job",
            "--config-path",
            "/app/examples/evolve_ga_m1_micro.yml",
            "--storage-subpath",
            "runs/demo",
            "--max-fevals",
            "8",
            "--strategy",
            "pool",
            "--num-gpus",
            "1",
            "--limit",
            "16",
            "--random-seed",
            "7",
            "--no-merge-cuda",
            "--no-save-final-model",
            "--no-reshard",
            "--no-baseline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"kubectl apply -n test-ns -f {DEPLOY_DIR / 'ray-job.yaml'}" in result.output


def test_ensure_efs_node_role_permissions_attaches_only_when_missing(
    monkeypatch, capsys
):
    attached = []

    class FakeEksClient:
        def list_nodegroups(self, clusterName):
            assert clusterName == "demo"
            return {"nodegroups": ["cpu", "gpu"]}

        def describe_nodegroup(self, clusterName, nodegroupName):
            roles = {
                "cpu": "arn:aws:iam::123456789012:role/demo-cpu-role",
                "gpu": "arn:aws:iam::123456789012:role/demo-gpu-role",
            }
            return {"nodegroup": {"nodeRole": roles[nodegroupName]}}

    class FakeIamClient:
        def list_attached_role_policies(self, RoleName):
            if RoleName == "demo-cpu-role":
                return {"AttachedPolicies": []}
            return {
                "AttachedPolicies": [{"PolicyArn": run_on_eks.EFS_UTILS_POLICY_ARN}]
            }

        def attach_role_policy(self, RoleName, PolicyArn):
            attached.append((RoleName, PolicyArn))

    fake_boto3 = types.SimpleNamespace(
        client=lambda service_name, region_name=None: (
            FakeEksClient() if service_name == "eks" else FakeIamClient()
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)

    run_on_eks._ensure_efs_node_role_permissions("demo", "us-east-1", dry_run=False)

    assert attached == [("demo-cpu-role", run_on_eks.EFS_UTILS_POLICY_ARN)]
    assert "Attached" in capsys.readouterr().out
