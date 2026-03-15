"""Utilities for orchestrating MergeKit GA runs on an Amazon EKS backed Ray cluster.

The helpers here assume that eksctl, kubectl, helm, and aws CLI are already installed and
configured in the executing environment. Use the ``mergekit[eks]`` optional extra to install
Python dependencies (boto3, kubernetes, python-dotenv) that are leveraged here.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import click

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_ENV_FILE = Path(".env.eks")
DEFAULT_NAMESPACE = "mergekit"
DEFAULT_RAY_CLUSTER_NAME = "mergekit-ga"
DEFAULT_JOB_NAME = "mergekit-ga-job"
REPO_ROOT = Path(
    os.environ.get("MERGEKIT_REPO_ROOT", Path(__file__).resolve().parents[2])
)
DEPLOY_DIR = Path(os.environ.get("MERGEKIT_DEPLOY_DIR", REPO_ROOT / "deploy"))
DEFAULT_IMAGE = os.environ.get("MERGEKIT_IMAGE", "mergekit-ga:latest")
DEFAULT_CONFIG_PATH = "/app/examples/evolve_ga_m1_micro.yml"

REQUIRED_BINARIES = {
    "kubectl": "Kubernetes CLI",
    "eksctl": "EKS cluster manager",
    "helm": "Helm package manager",
    "aws": "AWS CLI",
}
EFS_UTILS_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonElasticFileSystemsUtils"
DEFAULT_COST_TAGS = {
    "Project": "master-research",
    "Application": "mergekit-ga",
    "Environment": "research",
    "Owner": "nkululekothangelane",
    "ManagedBy": "mergekit-eks",
}


def _cost_tags(
    cluster_name: str,
    *,
    workload: Optional[str] = None,
    nodegroup_name: Optional[str] = None,
    resource_kind: Optional[str] = None,
) -> Dict[str, str]:
    tags = {
        **DEFAULT_COST_TAGS,
        "mergekit-cluster": cluster_name,
    }
    if workload:
        tags["Workload"] = workload
    if nodegroup_name:
        tags["NodeGroup"] = nodegroup_name
    if resource_kind:
        tags["ResourceKind"] = resource_kind
    return tags


def _aws_tags(tags: Mapping[str, str]) -> list[dict]:
    return [{"Key": key, "Value": value} for key, value in sorted(tags.items())]


def _nodegroup_workload(nodegroup_name: str) -> str:
    lowered = nodegroup_name.lower()
    if "gpu" in lowered:
        return "gpu"
    if "cpu" in lowered:
        return "cpu"
    return "general"


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


def _render_template(
    path: Path, replacements: Optional[Mapping[str, str]] = None
) -> str:
    content = path.read_text()
    for key, value in (replacements or {}).items():
        content = content.replace(f"__{key}__", value)
    return content


def _render_to_tempfile(
    path: Path,
    replacements: Optional[Mapping[str, str]] = None,
    *,
    suffix: str = ".yaml",
) -> Path:
    rendered = _render_template(path, replacements)
    handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix)
    try:
        handle.write(rendered)
    finally:
        handle.close()
    return Path(handle.name)


def _kubectl_manifest(
    action: str,
    path: Path,
    namespace: str,
    *,
    dry_run: bool,
    replacements: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    extra_args = list(extra_args or [])
    display_cmd = ["kubectl", action, "-n", namespace, "-f", str(path), *extra_args]
    click.echo(f"→ {' '.join(display_cmd)}")
    if dry_run:
        return

    manifest = _render_template(path, replacements)
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
        return linux_commands.get(
            command,
            "Install the package via your distribution or follow upstream docs.",
        )
    if system == "windows":
        return windows_commands.get(
            command, "Install the package with Chocolatey or the official installer."
        )

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


def _describe_cluster(cluster_name: str, region: str) -> dict:
    import boto3

    eks = boto3.client("eks", region_name=region)
    return eks.describe_cluster(name=cluster_name)["cluster"]


def _ensure_efs_node_role_permissions(
    cluster_name: str, region: str, *, dry_run: bool
) -> None:
    if dry_run:
        click.echo(f"→ ensure managed nodegroup roles include {EFS_UTILS_POLICY_ARN}")
        return

    import boto3

    eks = boto3.client("eks", region_name=region)
    iam = boto3.client("iam")

    nodegroup_names = eks.list_nodegroups(clusterName=cluster_name).get(
        "nodegroups", []
    )
    for nodegroup_name in nodegroup_names:
        nodegroup = eks.describe_nodegroup(
            clusterName=cluster_name, nodegroupName=nodegroup_name
        )["nodegroup"]
        role_arn = nodegroup["nodeRole"]
        role_name = role_arn.rsplit("/", 1)[-1]

        attached = iam.list_attached_role_policies(RoleName=role_name).get(
            "AttachedPolicies", []
        )
        if any(policy.get("PolicyArn") == EFS_UTILS_POLICY_ARN for policy in attached):
            continue

        iam.attach_role_policy(RoleName=role_name, PolicyArn=EFS_UTILS_POLICY_ARN)
        click.echo(f"Attached {EFS_UTILS_POLICY_ARN} to node role {role_name}.")


def _wait_for_efs_mount_targets(
    efs_client, filesystem_id: str, expected: int, timeout_s: int = 600
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        mount_targets = efs_client.describe_mount_targets(
            FileSystemId=filesystem_id
        ).get("MountTargets", [])
        available = sum(
            1 for mt in mount_targets if mt.get("LifeCycleState") == "available"
        )
        if available >= expected:
            return
        time.sleep(10)
    raise click.ClickException(
        f"EFS mount targets for {filesystem_id} did not become available within {timeout_s} seconds."
    )


def _ensure_efs(cluster_name: str, region: str, *, dry_run: bool) -> str:
    if dry_run:
        click.echo("→ ensure shared EFS filesystem for Ray cache/artifacts")
        return "fs-dryrun"

    import boto3

    cluster = _describe_cluster(cluster_name, region)
    vpc_id = cluster["resourcesVpcConfig"]["vpcId"]
    subnet_ids = cluster["resourcesVpcConfig"]["subnetIds"]

    ec2 = boto3.client("ec2", region_name=region)
    efs = boto3.client("efs", region_name=region)

    filesystem_id: Optional[str] = None
    for filesystem in efs.describe_file_systems().get("FileSystems", []):
        tags = efs.describe_tags(FileSystemId=filesystem["FileSystemId"]).get(
            "Tags", []
        )
        for tag in tags:
            if (
                tag.get("Key") == "mergekit-cluster"
                and tag.get("Value") == cluster_name
            ):
                filesystem_id = filesystem["FileSystemId"]
                break
        if filesystem_id:
            break

    if filesystem_id is None:
        created = efs.create_file_system(
            CreationToken=f"{cluster_name}-mergekit-efs",
            PerformanceMode="generalPurpose",
            Encrypted=True,
            Tags=_aws_tags(
                {
                    **_cost_tags(cluster_name, resource_kind="efs"),
                    "Name": f"{cluster_name}-mergekit-efs",
                }
            ),
        )
        filesystem_id = created["FileSystemId"]
        click.echo(f"Created EFS filesystem {filesystem_id}.")
    else:
        efs.create_tags(
            FileSystemId=filesystem_id,
            Tags=_aws_tags(_cost_tags(cluster_name, resource_kind="efs")),
        )

    vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    vpc_cidr = vpc["CidrBlock"]

    sg_name = f"{cluster_name}-efs-sg"
    security_groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [sg_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    ).get("SecurityGroups", [])
    if security_groups:
        security_group_id = security_groups[0]["GroupId"]
    else:
        created_sg = ec2.create_security_group(
            GroupName=sg_name,
            Description=f"EFS access for {cluster_name}",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": _aws_tags(
                        {
                            **_cost_tags(cluster_name, resource_kind="security-group"),
                            "Name": sg_name,
                        }
                    ),
                }
            ],
        )
        security_group_id = created_sg["GroupId"]
        try:
            ec2.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 2049,
                        "ToPort": 2049,
                        "IpRanges": [
                            {"CidrIp": vpc_cidr, "Description": "VPC NFS access"}
                        ],
                    }
                ],
            )
        except Exception:
            pass

    subnets = ec2.describe_subnets(SubnetIds=subnet_ids)["Subnets"]
    az_to_subnet: Dict[str, str] = {}
    for subnet in subnets:
        az_to_subnet.setdefault(subnet["AvailabilityZone"], subnet["SubnetId"])
    desired_subnets = list(az_to_subnet.values())

    existing_mount_targets = efs.describe_mount_targets(FileSystemId=filesystem_id).get(
        "MountTargets", []
    )
    existing_subnets = {mt["SubnetId"] for mt in existing_mount_targets}
    for subnet_id in desired_subnets:
        if subnet_id in existing_subnets:
            continue
        efs.create_mount_target(
            FileSystemId=filesystem_id,
            SubnetId=subnet_id,
            SecurityGroups=[security_group_id],
        )

    _wait_for_efs_mount_targets(efs, filesystem_id, expected=len(desired_subnets))
    return filesystem_id


def _ensure_cost_allocation_tags(
    cluster_name: str, region: str, *, dry_run: bool
) -> None:
    if dry_run:
        click.echo(
            "→ ensure cost-allocation tags on EKS cluster, nodegroups, ASGs, EC2 instances, EBS volumes, and EFS"
        )
        return

    import boto3

    cluster = _describe_cluster(cluster_name, region)

    eks = boto3.client("eks", region_name=region)
    autoscaling = boto3.client("autoscaling", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)
    efs = boto3.client("efs", region_name=region)

    eks.tag_resource(
        resourceArn=cluster["arn"],
        tags=_cost_tags(cluster_name, resource_kind="eks-cluster"),
    )

    nodegroup_names = eks.list_nodegroups(clusterName=cluster_name).get(
        "nodegroups", []
    )
    for nodegroup_name in nodegroup_names:
        nodegroup = eks.describe_nodegroup(
            clusterName=cluster_name, nodegroupName=nodegroup_name
        )["nodegroup"]
        workload = _nodegroup_workload(nodegroup_name)
        nodegroup_tags = _cost_tags(
            cluster_name,
            workload=workload,
            nodegroup_name=nodegroup_name,
            resource_kind="eks-nodegroup",
        )
        eks.tag_resource(resourceArn=nodegroup["nodegroupArn"], tags=nodegroup_tags)

        asg_names = [
            group["name"]
            for group in nodegroup.get("resources", {}).get("autoScalingGroups", [])
            if group.get("name")
        ]
        if asg_names:
            autoscaling.create_or_update_tags(
                Tags=[
                    {
                        "ResourceId": asg_name,
                        "ResourceType": "auto-scaling-group",
                        "Key": key,
                        "Value": value,
                        "PropagateAtLaunch": True,
                    }
                    for asg_name in asg_names
                    for key, value in nodegroup_tags.items()
                ]
            )

        reservations = ec2.describe_instances(
            Filters=[
                {"Name": "tag:eks:cluster-name", "Values": [cluster_name]},
                {"Name": "tag:eks:nodegroup-name", "Values": [nodegroup_name]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                },
            ]
        ).get("Reservations", [])
        instances = [instance for res in reservations for instance in res["Instances"]]
        instance_ids = [instance["InstanceId"] for instance in instances]
        volume_ids = [
            mapping["Ebs"]["VolumeId"]
            for instance in instances
            for mapping in instance.get("BlockDeviceMappings", [])
            if "Ebs" in mapping and mapping["Ebs"].get("VolumeId")
        ]

        if instance_ids:
            ec2.create_tags(
                Resources=instance_ids,
                Tags=_aws_tags(
                    {
                        **nodegroup_tags,
                        "ResourceKind": "ec2-instance",
                    }
                ),
            )
        if volume_ids:
            ec2.create_tags(
                Resources=volume_ids,
                Tags=_aws_tags(
                    {
                        **nodegroup_tags,
                        "ResourceKind": "ebs-volume",
                    }
                ),
            )

    for filesystem in efs.describe_file_systems().get("FileSystems", []):
        filesystem_id = filesystem["FileSystemId"]
        tags = {
            tag["Key"]: tag["Value"]
            for tag in efs.describe_tags(FileSystemId=filesystem_id).get("Tags", [])
        }
        if tags.get("mergekit-cluster") != cluster_name:
            continue
        efs.create_tags(
            FileSystemId=filesystem_id,
            Tags=_aws_tags(_cost_tags(cluster_name, resource_kind="efs")),
        )

    security_groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [f"{cluster_name}-efs-sg"]},
            {
                "Name": "vpc-id",
                "Values": [cluster["resourcesVpcConfig"]["vpcId"]],
            },
        ]
    ).get("SecurityGroups", [])
    if security_groups:
        ec2.create_tags(
            Resources=[security_group["GroupId"] for security_group in security_groups],
            Tags=_aws_tags(_cost_tags(cluster_name, resource_kind="security-group")),
        )

    click.echo(
        "Ensured cost-allocation tags on cluster, nodegroups, autoscaling groups, EC2 instances, EBS volumes, and EFS."
    )


def _ensure_efs_csi_driver(cluster_name: str, region: str, *, dry_run: bool) -> None:
    addon_cmd = [
        "aws",
        "eks",
        "create-addon",
        "--cluster-name",
        cluster_name,
        "--region",
        region,
        "--addon-name",
        "aws-efs-csi-driver",
        "--resolve-conflicts",
        "OVERWRITE",
    ]

    if dry_run:
        _run_command(addon_cmd, dry_run=True)
        return

    import boto3
    from botocore.exceptions import ClientError

    eks = boto3.client("eks", region_name=region)
    addons = eks.list_addons(clusterName=cluster_name).get("addons", [])
    if "aws-efs-csi-driver" not in addons:
        _run_command(addon_cmd, dry_run=False)
    else:
        click.echo("EFS CSI driver addon already installed; ensuring it is active.")

    try:
        eks.get_waiter("addon_active").wait(
            clusterName=cluster_name, addonName="aws-efs-csi-driver"
        )
    except ClientError as exc:
        raise click.ClickException(
            f"Failed waiting for aws-efs-csi-driver addon: {exc}"
        ) from exc


def _build_entrypoint(
    *,
    config_path: str,
    storage_subpath: str,
    max_fevals: int,
    strategy: str,
    num_gpus: int,
    random_seed: int,
    limit: Optional[int],
    merge_cuda: bool,
    save_final_model: bool,
    reshard: bool,
    run_baseline: bool,
    extra_args: Sequence[str],
) -> str:
    cmd = [
        "python",
        "-m",
        "mergekit.scripts.evolve_ga",
        config_path,
        "--storage-path",
        f"/data/artifacts/{storage_subpath}",
        "--max-fevals",
        str(max_fevals),
        "--strategy",
        strategy,
        "--num-gpus",
        str(num_gpus),
        "--random-seed",
        str(random_seed),
        "--merge-cuda" if merge_cuda else "--no-merge-cuda",
        "--save-final-model" if save_final_model else "--no-save-final-model",
        "--reshard" if reshard else "--no-reshard",
        "--baseline" if run_baseline else "--no-baseline",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(extra_args)
    return shlex.join(cmd)


@click.group()
@click.option(
    "--env-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_ENV_FILE,
    help="Optional .env file with AWS_* overrides",
)
@click.option(
    "--namespace",
    default=DEFAULT_NAMESPACE,
    show_default=True,
    help="Kubernetes namespace for Ray components",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    show_default=True,
    help="Print commands without executing them",
)
@click.option(
    "--image",
    default=lambda: os.environ.get("MERGEKIT_IMAGE", DEFAULT_IMAGE),
    show_default="env[MERGEKIT_IMAGE] or 'mergekit-ga:latest'",
    help="Container image to use for the Ray head/workers",
)
@click.pass_context
def cli(
    ctx: click.Context, env_file: Path, namespace: str, dry_run: bool, image: str
) -> None:
    """Manage Ray on EKS resources for MergeKit GA."""

    _load_env_file(env_file)
    ctx.ensure_object(dict)
    ctx.obj["namespace"] = namespace
    ctx.obj["dry_run"] = dry_run
    ctx.obj["image"] = image


@cli.command()
@click.option("--cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option("--ray-cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option(
    "--region",
    default=lambda: os.getenv("AWS_REGION", "us-east-1"),
    show_default="env[AWS_REGION] or 'us-east-1'",
)
@click.option("--cpu-node-type", default="m6i.xlarge", show_default=True)
@click.option(
    "--cpu-nodes",
    default=2,
    show_default=True,
    help="Total CPU nodes, including the node that hosts the Ray head pod",
)
@click.option("--cpu-max-nodes", default=3, show_default=True)
@click.option("--gpu-node-type", default="g6.xlarge", show_default=True)
@click.option("--gpu-nodes", default=0, show_default=True)
@click.option("--gpu-max-nodes", default=1, show_default=True)
@click.pass_context
def bootstrap(
    ctx: click.Context,
    cluster_name: str,
    ray_cluster_name: str,
    region: str,
    cpu_node_type: str,
    cpu_nodes: int,
    cpu_max_nodes: int,
    gpu_node_type: str,
    gpu_nodes: int,
    gpu_max_nodes: int,
) -> None:
    """Create or upgrade cluster dependencies and deploy Ray cluster resources."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]
    image = ctx.obj["image"]

    _verify_prerequisites(skip=dry_run)

    cluster_config_path = DEPLOY_DIR / "eks-cluster-config.yaml"
    rendered_cluster_config = _render_to_tempfile(
        cluster_config_path,
        {
            "CLUSTER_NAME": cluster_name,
            "AWS_REGION": region,
            "CPU_NODE_TYPE": cpu_node_type,
            "CPU_NODES": str(cpu_nodes),
            "CPU_MAX_NODES": str(cpu_max_nodes),
            "GPU_NODE_TYPE": gpu_node_type,
            "GPU_NODES": str(gpu_nodes),
            "GPU_MAX_NODES": str(gpu_max_nodes),
        },
    )
    try:
        if dry_run or not _cluster_exists(cluster_name, region):
            _run_command(
                [
                    "eksctl",
                    "create",
                    "cluster",
                    "-f",
                    str(rendered_cluster_config),
                ],
                dry_run=dry_run,
            )
        else:
            click.echo(
                f"EKS cluster '{cluster_name}' already exists in {region}; skipping creation."
            )
    finally:
        rendered_cluster_config.unlink(missing_ok=True)

    _ensure_kubeconfig(cluster_name, region, dry_run=dry_run)

    _run_command(
        [
            "helm",
            "repo",
            "add",
            "kuberay",
            "https://ray-project.github.io/kuberay-helm",
        ],
        dry_run=dry_run,
    )
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

    _ensure_efs_csi_driver(cluster_name, region, dry_run=dry_run)
    _ensure_efs_node_role_permissions(cluster_name, region, dry_run=dry_run)
    efs_file_system_id = _ensure_efs(cluster_name, region, dry_run=dry_run)
    _ensure_cost_allocation_tags(cluster_name, region, dry_run=dry_run)

    replacements = {
        "MERGEKIT_IMAGE": image,
        "AWS_REGION": region,
        "RAY_CLUSTER_NAME": ray_cluster_name,
        "CPU_WORKER_REPLICAS": str(max(cpu_nodes - 1, 0)),
        "CPU_WORKER_MAX_REPLICAS": str(max(cpu_max_nodes - 1, 0)),
        "GPU_WORKER_REPLICAS": str(gpu_nodes),
        "GPU_WORKER_MAX_REPLICAS": str(gpu_max_nodes),
        "EFS_FILE_SYSTEM_ID": efs_file_system_id,
    }

    storage_manifest = DEPLOY_DIR / "storage.yaml"
    if storage_manifest.exists():
        _kubectl_manifest(
            "apply",
            storage_manifest,
            namespace,
            replacements=replacements,
            dry_run=dry_run,
        )

    cluster_manifest = DEPLOY_DIR / "ray-cluster.yaml"
    _kubectl_manifest(
        "apply",
        cluster_manifest,
        namespace,
        replacements=replacements,
        dry_run=dry_run,
    )


@cli.command()
@click.option("--ray-cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option("--job-name", default=DEFAULT_JOB_NAME, show_default=True)
@click.option(
    "--region",
    default=lambda: os.getenv("AWS_REGION", "us-east-1"),
    show_default="env[AWS_REGION] or 'us-east-1'",
)
@click.option("--config-path", default=DEFAULT_CONFIG_PATH, show_default=True)
@click.option("--storage-subpath", default="runs/aws-smoke", show_default=True)
@click.option("--max-fevals", default=4, show_default=True)
@click.option(
    "--strategy",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    show_default=True,
)
@click.option("--num-gpus", default=0, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--random-seed", default=42, show_default=True)
@click.option("--merge-cuda/--no-merge-cuda", default=False, show_default=True)
@click.option(
    "--save-final-model/--no-save-final-model", default=False, show_default=True
)
@click.option("--reshard/--no-reshard", default=False, show_default=True)
@click.option(
    "--baseline/--no-baseline", "run_baseline", default=False, show_default=True
)
@click.option(
    "--extra-arg",
    "extra_args",
    multiple=True,
    help="Additional CLI args to append to mergekit-evolve-ga",
)
@click.pass_context
def submit(
    ctx: click.Context,
    ray_cluster_name: str,
    job_name: str,
    region: str,
    config_path: str,
    storage_subpath: str,
    max_fevals: int,
    strategy: str,
    num_gpus: int,
    limit: Optional[int],
    random_seed: int,
    merge_cuda: bool,
    save_final_model: bool,
    reshard: bool,
    run_baseline: bool,
    extra_args: Sequence[str],
) -> None:
    """Submit a MergeKit GA Ray job manifest."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]

    _verify_prerequisites(skip=dry_run)

    entrypoint = _build_entrypoint(
        config_path=config_path,
        storage_subpath=storage_subpath,
        max_fevals=max_fevals,
        strategy=strategy,
        num_gpus=num_gpus,
        random_seed=random_seed,
        limit=limit,
        merge_cuda=merge_cuda,
        save_final_model=save_final_model,
        reshard=reshard,
        run_baseline=run_baseline,
        extra_args=extra_args,
    )

    job_manifest = DEPLOY_DIR / "ray-job.yaml"
    _kubectl_manifest(
        "apply",
        job_manifest,
        namespace,
        replacements={
            "AWS_REGION": region,
            "RAY_CLUSTER_NAME": ray_cluster_name,
            "RAY_JOB_NAME": job_name,
            "MERGEKIT_ENTRYPOINT": entrypoint,
        },
        dry_run=dry_run,
    )

    if not dry_run:
        click.echo(
            f"RayJob submitted. Inspect with `kubectl get rayjobs -n {namespace}`."
        )


@cli.command()
@click.option("--cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option("--ray-cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option("--job-name", default=DEFAULT_JOB_NAME, show_default=True)
@click.option(
    "--region",
    default=lambda: os.getenv("AWS_REGION", "us-east-1"),
    show_default="env[AWS_REGION] or 'us-east-1'",
)
@click.pass_context
def teardown(
    ctx: click.Context,
    cluster_name: str,
    ray_cluster_name: str,
    job_name: str,
    region: str,
) -> None:
    """Remove Ray jobs, Ray chart, storage resources, and the backing EKS cluster."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]

    _verify_prerequisites(skip=dry_run)

    replacements = {
        "AWS_REGION": region,
        "RAY_CLUSTER_NAME": ray_cluster_name,
        "RAY_JOB_NAME": job_name,
        "MERGEKIT_ENTRYPOINT": "true",
        "EFS_FILE_SYSTEM_ID": "fs-dryrun",
    }

    _kubectl_manifest(
        "delete",
        DEPLOY_DIR / "ray-job.yaml",
        namespace,
        replacements=replacements,
        dry_run=dry_run,
        extra_args=["--ignore-not-found"],
    )
    _kubectl_manifest(
        "delete",
        DEPLOY_DIR / "ray-cluster.yaml",
        namespace,
        replacements=replacements,
        dry_run=dry_run,
        extra_args=["--ignore-not-found"],
    )
    _run_command(
        ["helm", "uninstall", "kuberay-operator", "-n", namespace], dry_run=dry_run
    )

    storage_manifest = DEPLOY_DIR / "storage.yaml"
    if storage_manifest.exists():
        _kubectl_manifest(
            "delete",
            storage_manifest,
            namespace,
            replacements=replacements,
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
@click.option("--cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option("--ray-cluster-name", default=DEFAULT_RAY_CLUSTER_NAME, show_default=True)
@click.option(
    "--region",
    default=lambda: os.getenv("AWS_REGION", "us-east-1"),
    show_default="env[AWS_REGION] or 'us-east-1'",
)
@click.pass_context
def status(
    ctx: click.Context, cluster_name: str, ray_cluster_name: str, region: str
) -> None:
    """Show a short Ray cluster status report."""

    dry_run = ctx.obj["dry_run"]
    namespace = ctx.obj["namespace"]

    _verify_prerequisites(skip=dry_run)

    _run_command(
        [
            "kubectl",
            "get",
            "nodes",
            "-o",
            r"custom-columns=NAME:.metadata.name,INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu",
        ],
        dry_run=dry_run,
    )
    _run_command(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"], dry_run=dry_run
    )
    _run_command(["kubectl", "get", "rayclusters", "-n", namespace], dry_run=dry_run)
    _run_command(["kubectl", "get", "rayjobs", "-n", namespace], dry_run=dry_run)
    _run_command(
        [
            "ray",
            "job",
            "list",
            "--address",
            f"ray://{ray_cluster_name}-head-svc.{namespace}.svc.cluster.local:10001",
        ],
        dry_run=dry_run,
    )

    if not dry_run:
        cluster = _describe_cluster(cluster_name, region)
        click.echo(json.dumps(cluster, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    try:
        cli(prog_name="mergekit-eks")
    except subprocess.CalledProcessError as exc:
        click.echo(f"Command failed with exit code {exc.returncode}", err=True)
        sys.exit(exc.returncode)
