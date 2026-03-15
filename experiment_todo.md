# Experiment Todo

The checklist below is the minimum experiment set for the thesis-facing 3B EKS phase.
Do not mark Experiment 1 complete until both the CPU and GPU runs finish cleanly and the GPU run is faster on both wall-clock and `ga_history.csv` `eval_seconds`.

- [x] Experiment 0: Verify Ray-on-EKS submission, shared storage, and artifact persistence on the live cluster.
- [ ] Experiment 1: Confirm GPU throughput uplift on the stable 3B smoke by running the same config on a GPU worker and a resized CPU worker, then compare wall-clock and `eval_seconds`.
- [ ] Experiment 2: Restore an instruction-following metric for the thesis benchmark suite (`ifeval` fixed or an explicit stable proxy selected) and complete one clean benchmark-aligned 3B smoke run.
- [ ] Experiment 3: Run 3B pilot seed A with baseline enabled using [`examples/qwen25_3b_eks_scale.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/examples/qwen25_3b_eks_scale.yml).
- [ ] Experiment 4: Run 3B pilot seed B with baseline enabled using the same config and a different seed.
- [ ] Experiment 5: Run the 3B main experiment seed A after the pilot confirms clean completion and non-pathological scoring behavior.
- [ ] Experiment 6: Run the 3B main experiment seed B for repeatability.
- [ ] Experiment 7: Consolidate final 3B artifacts into thesis-ready tables, plots, and model/result summaries.

## Deferred / Blocked

- [ ] Experiment 8: Prepare the 7B pilot shape and quota plan, then run one 7B smoke after the 3B suite is stable.
