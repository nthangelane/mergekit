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
    assert (
        "ensure cost-allocation tags on EKS cluster, nodegroups, ASGs, EC2 instances, EBS volumes, and EFS"
        in result.output
    )
    assert "Planned Ray worker scale:" in result.output


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


def test_dashboard_prints_local_url_and_port_forward_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "dashboard",
            "--ray-cluster-name",
            "demo-ray",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Ray dashboard: http://127.0.0.1:8265" in result.output
    assert (
        "kubectl port-forward -n test-ns svc/demo-ray-head-svc 8265:8265 10001:10001"
        in result.output
    )


def test_deploy_manifests_are_valid_yaml():
    deploy_dir = Path(DEPLOY_DIR)
    manifests = [
        deploy_dir / "ray-cluster.yaml",
        deploy_dir / "ray-job.yaml",
        deploy_dir / "storage.yaml",
        deploy_dir / "fluent-bit.yaml",
    ]

    for manifest in manifests:
        assert manifest.exists(), f"Missing manifest: {manifest}"
        docs = list(yaml.safe_load_all(manifest.read_text()))
        assert docs, f"Manifest {manifest} is empty"

    cluster = yaml.safe_load((deploy_dir / "ray-cluster.yaml").read_text())
    assert cluster.get("kind") == "RayCluster"
    assert cluster["spec"]["headGroupSpec"]["rayStartParams"]["num-cpus"] == "0"
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

    fluent_docs = list(yaml.safe_load_all((deploy_dir / "fluent-bit.yaml").read_text()))
    fluent_kinds = {doc["kind"] for doc in fluent_docs if doc}
    assert {"Namespace", "ConfigMap", "DaemonSet"}.issubset(fluent_kinds)

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
    assert (
        "verify Ray gpu-workers pods are Ready and pass CUDA preflight" in result.output
    )


def test_build_entrypoint_includes_vllm_tensor_parallel_flags():
    entrypoint = run_on_eks._build_entrypoint(
        config_path="/app/experiments/thesis/eks_gpu/exp05_llama3_8b_same_family/config.yml",
        storage_subpath="thesis/eks_gpu/exp05_llama3_8b_same_family",
        max_fevals=80,
        strategy="pool",
        num_gpus=10,
        vllm=True,
        tensor_parallel_size=2,
        random_seed=11,
        limit=32,
        merge_cuda=True,
        save_final_model=True,
        reshard=True,
        run_baseline=True,
        extra_args=("--i-understand-the-depths-of-the-evils-i-am-unleashing",),
    )

    assert "--vllm" in entrypoint
    assert "--tensor-parallel-size 2" in entrypoint
    assert "--num-gpus 10" in entrypoint


def test_build_finalist_entrypoint_uses_ray_remote_gpu_worker():
    entrypoint = run_on_eks._build_finalist_entrypoint(
        storage_subpath="runs/demo",
        trust_remote_code=True,
        merge_cuda=True,
    )

    assert 'ray.init(address="auto")' in entrypoint
    assert "@ray.remote(num_cpus=1, num_gpus=1)" in entrypoint
    assert "/data/artifacts/runs/demo" in entrypoint
    assert "final_model" in entrypoint


def test_submit_finalist_renders_ray_job_and_preflight():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "submit-finalist",
            "--ray-cluster-name",
            "demo-ray",
            "--job-name",
            "demo-finalist",
            "--storage-subpath",
            "runs/demo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"kubectl apply -n test-ns -f {DEPLOY_DIR / 'ray-job.yaml'}" in result.output
    assert (
        "verify Ray gpu-workers pods are Ready and pass CUDA preflight" in result.output
    )


def test_submit_rejects_tensor_parallel_without_vllm():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "submit",
            "--tensor-parallel-size",
            "2",
            "--num-gpus",
            "4",
        ],
    )

    assert result.exit_code != 0
    assert "--tensor-parallel-size > 1 requires --vllm" in result.output


def test_submit_rejects_inline_final_model_save_for_gpu_search():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dry-run",
            "--namespace",
            "test-ns",
            "submit",
            "--num-gpus",
            "1",
            "--save-final-model",
        ],
    )

    assert result.exit_code != 0
    assert "must use --no-save-final-model" in result.output


def test_worker_scale_bounds_reports_20_to_30_worker_shapes():
    scale = run_on_eks._ray_worker_scale_bounds(
        cpu_nodes=2,
        cpu_max_nodes=2,
        gpu_nodes=5,
        gpu_max_nodes=7,
        gpu_worker_pods_per_node=4,
    )

    assert scale["gpu_workers"] == 20
    assert scale["gpu_workers_max"] == 28
    assert scale["total_workers"] == 21
    assert scale["total_workers_max"] == 29


def test_scale_profile_throughput_28_is_available_from_bootstrap():
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
            "--scale-profile",
            "throughput-28",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using scale profile 'throughput-28'" in result.output
    assert "Cloud scale target: 20-30 workers is supported" in result.output
    assert "gpu=28..28" in result.output


def test_explicit_cli_override_beats_scale_profile():
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
            "--scale-profile",
            "throughput-20",
            "--gpu-max-nodes",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using scale profile 'throughput-20'" in result.output
    assert "Explicit CLI overrides kept: gpu_max_nodes" in result.output
    assert "gpu=20..28" in result.output


def test_research_3b_scale_profile_sets_four_l4_workers():
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
            "--scale-profile",
            "research-3b-l4x4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using scale profile 'research-3b-l4x4'" in result.output
    assert "gpu=4..4" in result.output
    assert (
        f"kubectl apply -n amazon-cloudwatch -f {DEPLOY_DIR / 'fluent-bit.yaml'}"
        in result.output
    )
    assert run_on_eks.CLOUDWATCH_AGENT_POLICY_ARN in result.output


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


def test_ensure_cloudwatch_node_role_permissions_attaches_only_when_missing(
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
                "AttachedPolicies": [
                    {"PolicyArn": run_on_eks.CLOUDWATCH_AGENT_POLICY_ARN}
                ]
            }

        def attach_role_policy(self, RoleName, PolicyArn):
            attached.append((RoleName, PolicyArn))

    fake_boto3 = types.SimpleNamespace(
        client=lambda service_name, region_name=None: (
            FakeEksClient() if service_name == "eks" else FakeIamClient()
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)

    run_on_eks._ensure_cloudwatch_node_role_permissions(
        "demo", "us-east-1", dry_run=False
    )

    assert attached == [("demo-cpu-role", run_on_eks.CLOUDWATCH_AGENT_POLICY_ARN)]
    assert "Attached" in capsys.readouterr().out


def test_ensure_cost_allocation_tags_covers_cluster_and_storage(monkeypatch, capsys):
    eks_tag_calls = []
    asg_tag_calls = []
    ec2_tag_calls = []
    efs_tag_calls = []

    class FakeEksClient:
        def list_nodegroups(self, clusterName):
            assert clusterName == "demo"
            return {"nodegroups": ["demo-cpu", "demo-gpu"]}

        def describe_nodegroup(self, clusterName, nodegroupName):
            resources = {
                "demo-cpu": {
                    "nodegroupArn": "arn:aws:eks:cpu",
                    "resources": {
                        "autoScalingGroups": [{"name": "asg-cpu"}],
                    },
                },
                "demo-gpu": {
                    "nodegroupArn": "arn:aws:eks:gpu",
                    "resources": {
                        "autoScalingGroups": [{"name": "asg-gpu"}],
                    },
                },
            }
            return {"nodegroup": resources[nodegroupName]}

        def tag_resource(self, resourceArn, tags):
            eks_tag_calls.append((resourceArn, tags))

    class FakeAutoScalingClient:
        def create_or_update_tags(self, Tags):
            asg_tag_calls.extend(Tags)

    class FakeEc2Client:
        def describe_instances(self, Filters):
            nodegroup = next(
                f["Values"][0] for f in Filters if f["Name"] == "tag:eks:nodegroup-name"
            )
            instance_id = f"i-{nodegroup}"
            volume_id = f"vol-{nodegroup}"
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": instance_id,
                                "BlockDeviceMappings": [
                                    {"Ebs": {"VolumeId": volume_id}}
                                ],
                            }
                        ]
                    }
                ]
            }

        def create_tags(self, Resources, Tags):
            ec2_tag_calls.append((Resources, Tags))

        def describe_security_groups(self, Filters):
            return {"SecurityGroups": [{"GroupId": "sg-efs"}]}

    class FakeEfsClient:
        def describe_file_systems(self):
            return {"FileSystems": [{"FileSystemId": "fs-123"}]}

        def describe_tags(self, FileSystemId):
            assert FileSystemId == "fs-123"
            return {"Tags": [{"Key": "mergekit-cluster", "Value": "demo"}]}

        def create_tags(self, FileSystemId, Tags):
            efs_tag_calls.append((FileSystemId, Tags))

    fake_boto3 = types.SimpleNamespace(
        client=lambda service_name, region_name=None: {
            "eks": FakeEksClient(),
            "autoscaling": FakeAutoScalingClient(),
            "ec2": FakeEc2Client(),
            "efs": FakeEfsClient(),
        }[service_name]
    )
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
    monkeypatch.setattr(
        run_on_eks,
        "_describe_cluster",
        lambda cluster_name, region: {
            "arn": "arn:aws:eks:cluster/demo",
            "resourcesVpcConfig": {"vpcId": "vpc-123"},
        },
    )

    run_on_eks._ensure_cost_allocation_tags("demo", "us-east-1", dry_run=False)

    assert eks_tag_calls[0][0] == "arn:aws:eks:cluster/demo"
    assert any(call[0] == "arn:aws:eks:cpu" for call in eks_tag_calls)
    assert any(call[0] == "arn:aws:eks:gpu" for call in eks_tag_calls)
    assert any(tag["ResourceId"] == "asg-cpu" for tag in asg_tag_calls)
    assert any(tag["ResourceId"] == "asg-gpu" for tag in asg_tag_calls)
    assert any(resources == ["i-demo-cpu"] for resources, _ in ec2_tag_calls)
    assert any(resources == ["vol-demo-cpu"] for resources, _ in ec2_tag_calls)
    assert any(resources == ["sg-efs"] for resources, _ in ec2_tag_calls)
    assert efs_tag_calls == [
        (
            "fs-123",
            run_on_eks._aws_tags(run_on_eks._cost_tags("demo", resource_kind="efs")),
        )
    ]
    assert "Ensured cost-allocation tags" in capsys.readouterr().out
