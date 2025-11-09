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
    assert "helm repo add kuberay https://ray-project.github.io/kuberay-helm" in result.output
    assert "helm upgrade --install kuberay-operator kuberay/kuberay-operator" in result.output


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
    assert "helm repo add kuberay https://ray-project.github.io/kuberay-helm" in result.output
    assert "helm upgrade --install kuberay-operator kuberay/kuberay-operator" in result.output


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
    delete_cmd = f"kubectl delete -n test-ns -f {DEPLOY_DIR / 'storage.yaml'} --ignore-not-found"
    assert delete_cmd in result.output
    assert "helm uninstall kuberay-operator -n test-ns" in result.output


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
    head_volumes = cluster["spec"]["headGroupSpec"]["template"]["spec"]["volumes"]
    assert head_volumes

    storage_docs = list(yaml.safe_load_all((deploy_dir / "storage.yaml").read_text()))
    claim_names = {doc["metadata"]["name"] for doc in storage_docs if doc}
    assert {"hf-cache-pvc", "artifacts-pvc"}.issubset(claim_names)
