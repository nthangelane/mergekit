"""Utilities for orchestrating MergeKit GA runs on an Amazon EKS backed Ray cluster.

The helpers here assume that eksctl, kubectl, helm, and aws CLI are already installed and
configured in the executing environment. Use the ``mergekit[eks]`` optional extra to install
Python dependencies (boto3, kubernetes, python-dotenv) that are leveraged here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
import platform
import shutil

import click

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_ENV_FILE = Path(".env.eks")
DEFAULT_NAMESPACE = "mergekit"
REPO_ROOT = Path(os.environ.get("MERGEKIT_REPO_ROOT", Path(__file__).resolve().parents[2]))
DEPLOY_DIR = Path(os.environ.get("MERGEKIT_DEPLOY_DIR", REPO_ROOT / "deploy"))
DEFAULT_IMAGE = os.environ.get("MERGEKIT_IMAGE", "mergekit-ga:latest")

REQUIRED_BINARIES = {
    "kubectl": "Kubernetes CLI",
    "eksctl": "EKS cluster manager",
    "helm": "Helm package manager",
    "aws": "AWS CLI",
}


def _ensure_kubeconfig(cluster_name: str, region: str, *, dry_run: bool) -> None:
    """Write or refresh the local kubeconfig so kubectl can reach the EKS cluster."""

    _run_command(
        [
            "aws",
            "eks",
            "update-kubeconfig",
            "--name",
            cluster_name,
            "--region",
            region,
        ],
        dry_run=dry_run,
    )


def _cluster_exists(cluster_name: str, region: str) -> bool:
    result = subprocess.run(
        [
            "eksctl",
            "get",
            "cluster",
            "--name",
            cluster_name,
            "--region",
            region,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _load_env_file(env_file: Path) -> None:
    if load_dotenv is None:
        return

    if env_file.exists():
        load_dotenv(dotenv_path=env_file)


def _run_command(
    cmd: Iterable[str],
    *,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> None:
    display = " ".join(cmd)
    click.echo(f"→ {display}")
    if dry_run:
        return

    subprocess.run(list(cmd), check=True, env={**os.environ, **(env or {})})


def _render_manifest(path: Path, image: str) -> str:
    content = path.read_text()
    if "__MERGEKIT_IMAGE__" in content:
        return content.replace("__MERGEKIT_IMAGE__", image)
    return content


def _kubectl_manifest(
    action: str,
    path: Path,
    namespace: str,
    *,
    image: str,
    dry_run: bool,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    extra_args = list(extra_args or [])
    display_cmd = ["kubectl", action, "-n", namespace, "-f", str(path), *extra_args]
    click.echo(f"→ {' '.join(display_cmd)}")
    if dry_run:
        return

    manifest = _render_manifest(path, image)
    stdin_cmd = ["kubectl", action, "-n", namespace, "-f", "-", *extra_args]
    subprocess.run(stdin_cmd, input=manifest.encode("utf-8"), check=True)


def _install_hint(command: str) -> str:
    system = platform.system().lower()

    mac_commands = {
        "kubectl": "brew install kubectl",
        "eksctl": "brew install eksctl",
        "helm": "brew install helm",
        "aws": "brew install awscli",
    }

    linux_commands = {
        "kubectl": "sudo apt-get install -y kubectl  # or follow https://kubernetes.io/docs/tasks/tools/",
        "eksctl": "curl -sS https://raw.githubusercontent.com/weaveworks/eksctl/main/install.sh | sudo bash",
        "helm": "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash",
        "aws": 'curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip && unzip awscliv2.zip && sudo ./aws/install',
    }

    windows_commands = {
        "kubectl": "choco install kubernetes-cli",
        "eksctl": "choco install eksctl",
        "helm": "choco install kubernetes-helm",
        "aws": "choco install awscli",
    }

    if system == "darwin":
        return mac_commands.get(command, "Use Homebrew to install the required CLI.")
    if system == "linux":
        return linux_commands.get(command, "Install the package via your distribution or follow upstream docs.")
    if system == "windows":
        return windows_commands.get(command, "Install the package with Chocolatey or the official installer.")

    return "Refer to the official installation guide for your operating system."


def _verify_prerequisites(*, skip: bool = False) -> None:
    if skip:
        return

    missing = []
    for command, description in REQUIRED_BINARIES.items():
        if shutil.which(command) is None:
            missing.append((command, description, _install_hint(command)))

    if not missing:
        return

    lines = [
        "Missing required command-line tools to manage EKS deployments:",
    ]
    for command, description, hint in missing:
        lines.append(f"  • {command} ({description})")
        lines.append(f"    Install: {hint}")

    raise click.ClickException("\n".join(lines))


@click.group()
@click.option("--env-file", type=click.Path(path_type=Path), default=DEFAULT_ENV_FILE, help="Optional .env file with AWS_* overrides")
@click.option("--namespace", default=DEFAULT_NAMESPACE, show_default=True, help="Kubernetes namespace for Ray components")
@click.option("--dry-run/--no-dry-run", default=False, show_default=True, help="Print commands without executing them")
@click.option(
    "--image",
    default=lambda: os.environ.get("MERGEKIT_IMAGE", DEFAULT_IMAGE),
    show_default="env[MERGEKIT_IMAGE] or 'mergekit-ga:latest'",
    help="Container image to use for the Ray head/workers",
)
@click.pass_context
def cli(ctx: click.Context, env_file: Path, namespace: str, dry_run: bool, image: str) -> None:
    """Manage Ray on EKS resources for MergeKit GA."""

    _load_env_file(env_file)
    ctx.ensure_object(dict)
    ctx.obj["namespace"] = namespace
    ctx.obj["dry_run"] = dry_run
    ctx.obj["image"] = image


@cli.command()
@click.option("--cluster-name", default="mergekit-ga", show_default=True)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", "us-east-1"), show_default="env[AWS_REGION] or 'us-east-1'")
@click.option("--node-type", default="m6i.xlarge", show_default=True)
@click.option("--nodes", default=2, show_default=True)
@click.pass_context
def bootstrap(
    ctx: click.Context,
    cluster_name: str,
    region: str,
    node_type: str,
    nodes: int,
) -> None:
    """Create or upgrade cluster dependencies and deploy Ray cluster resources."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]
    image = ctx.obj["image"]

    _verify_prerequisites(skip=dry_run)

    if dry_run or not _cluster_exists(cluster_name, region):
        _run_command(
            [
                "eksctl",
                "create",
                "cluster",
                "--name",
                cluster_name,
                "--region",
                region,
                "--nodegroup-name",
                f"{cluster_name}-ng",
                "--nodes",
                str(nodes),
                "--node-type",
                node_type,
                "--managed",
            ],
            dry_run=dry_run,
        )
    else:
        click.echo(f"EKS cluster '{cluster_name}' already exists in {region}; skipping creation.")

    _ensure_kubeconfig(cluster_name, region, dry_run=dry_run)

    _run_command(["helm", "repo", "add", "kuberay", "https://ray-project.github.io/kuberay-helm"], dry_run=dry_run)
    _run_command(["helm", "repo", "update"], dry_run=dry_run)

    values_file = DEPLOY_DIR / "ray-values.yaml"
    helm_cmd = [
        "helm",
        "upgrade",
        "--install",
        "kuberay-operator",
        "kuberay/kuberay-operator",
        "--namespace",
        namespace,
        "--create-namespace",
    ]
    if values_file.exists():
        helm_cmd.extend(["-f", str(values_file)])

    _run_command(helm_cmd, dry_run=dry_run)

    storage_manifest = DEPLOY_DIR / "storage.yaml"
    if storage_manifest.exists():
        _kubectl_manifest("apply", storage_manifest, namespace, image=image, dry_run=dry_run)

    cluster_manifest = DEPLOY_DIR / "ray-cluster.yaml"
    _kubectl_manifest("apply", cluster_manifest, namespace, image=image, dry_run=dry_run)


@cli.command()
@click.option("--region", default=lambda: os.getenv("AWS_REGION", "us-east-1"), show_default="env[AWS_REGION] or 'us-east-1'")
@click.pass_context
def submit(ctx: click.Context, region: str) -> None:
    """Submit the GA Ray job manifest."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]
    image = ctx.obj["image"]

    _verify_prerequisites(skip=dry_run)

    job_manifest = DEPLOY_DIR / "ray-job.yaml"

    _kubectl_manifest("apply", job_manifest, namespace, image=image, dry_run=dry_run)

    if not dry_run:
        click.echo("RayJob submitted. Inspect with `kubectl get rayjobs -n %s`." % namespace)


@cli.command()
@click.option("--cluster-name", default="mergekit-ga", show_default=True)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", "us-east-1"), show_default="env[AWS_REGION] or 'us-east-1'")
@click.pass_context
def teardown(ctx: click.Context, cluster_name: str, region: str) -> None:
    """Remove Ray jobs, Ray chart, and the backing EKS cluster."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]
    image = ctx.obj["image"]

    _verify_prerequisites(skip=dry_run)

    _kubectl_manifest(
        "delete",
        DEPLOY_DIR / "ray-job.yaml",
        namespace,
        image=image,
        dry_run=dry_run,
        extra_args=["--ignore-not-found"],
    )
    _kubectl_manifest(
        "delete",
        DEPLOY_DIR / "ray-cluster.yaml",
        namespace,
        image=image,
        dry_run=dry_run,
        extra_args=["--ignore-not-found"],
    )
    _run_command(["helm", "uninstall", "kuberay-operator", "-n", namespace], dry_run=dry_run)

    storage_manifest = DEPLOY_DIR / "storage.yaml"
    if storage_manifest.exists():
        _kubectl_manifest(
            "delete",
            storage_manifest,
            namespace,
            image=image,
            dry_run=dry_run,
            extra_args=["--ignore-not-found"],
        )

    _run_command(
        [
            "eksctl",
            "delete",
            "cluster",
            "--name",
            cluster_name,
            "--region",
            region,
        ],
        dry_run=dry_run,
    )


@cli.command()
@click.option("--cluster-name", default="mergekit-ga", show_default=True)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", "us-east-1"), show_default="env[AWS_REGION] or 'us-east-1'")
@click.pass_context
def status(ctx: click.Context, cluster_name: str, region: str) -> None:
    """Show a short Ray cluster status report."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]

    _verify_prerequisites(skip=dry_run)

    _run_command(["kubectl", "get", "pods", "-n", namespace], dry_run=dry_run)
    _run_command(["ray", "job", "list", "--address", "ray://mergekit-ga-head-svc.mergekit.svc.cluster.local:10001"], dry_run=dry_run)

    if not dry_run:
        import boto3

        eks = boto3.client("eks", region_name=region)
        cluster = eks.describe_cluster(name=cluster_name)["cluster"]
        click.echo(json.dumps(cluster, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    try:
        cli(prog_name="mergekit-eks")
    except subprocess.CalledProcessError as exc:
        click.echo(f"Command failed with exit code {exc.returncode}", err=True)
        sys.exit(exc.returncode)
