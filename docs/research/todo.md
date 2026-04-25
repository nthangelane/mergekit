- [x] Fix score direction for loss-like metrics in GA evaluation.
- [x] Audit baseline/upload comparison logic after score normalization.
- [x] Make the CPU `serial` strategy actually serial or rename it.
- [x] Fix merge timing and merge-failure reporting in serial CPU evaluation.
- [x] Add a CLI override for evaluation `limit`.
- [x] Clean up repeated `lm_eval` and Hugging Face warning noise.
- [x] Add an invariance test for normalized merges of identical source models.
- [x] Reduce repeated fetch and tokenizer-copy overhead per genotype.
- [x] Persist an exact-hash failed-genotype blacklist across runs.
- [x] Make CPU-only GA runs automatically disable merge CUDA or fail fast with explicit guidance when `--num-gpus 0`.

## Scale-Out Work

- [x] Lock merge-compatible 3B candidate models and create a scale-ready experiment config.
- [x] Lock merge-compatible 7B candidate models and create a scale-ready experiment config.
- [x] Review the benchmark suite and align the 3B/7B scale configs to the research topic rather than generic smoke tasks.
- [x] Fix the Ray-on-EKS job submission path so it emits a valid `mergekit-evolve-ga` command.
- [x] Add EKS bootstrap support for separate CPU and GPU node groups.
- [x] Add shared storage for Ray pods so Hugging Face cache and GA artifacts survive across nodes.
- [x] Attach the EFS utilities policy to managed nodegroup roles during bootstrap so shared storage mounts successfully.
- [x] Build and push a linux/amd64 MergeKit image to ECR.
- [x] Bootstrap the EKS cluster in a GPU-capable region and validate KubeRay installation.
- [x] Validate that the EKS GPU node group advertises `nvidia.com/gpu`, and install the NVIDIA device plugin if the managed nodegroup does not provide it.
- [ ] Submit and monitor an AWS smoke run from the repo-managed Ray job path.
- [ ] Fix `ifeval` on the local HuggingFace evaluation path for Qwen2.5 GPU runs, or replace it with a stable instruction-following proxy for AWS smoke validation.
- [ ] Plan the quota increase or alternate node strategy required for 7B-scale runs in this AWS account.
- [ ] Reduce the CUDA container image size and pull latency so Ray pods can start quickly on EKS nodes.
