# Thesis campaign presets

Run one or more experiments from the repository root:

```bash
experiments/run_campaign.sh exp26_aos_off --seeds 11
experiments/run_campaign.sh exp26_aos_on exp27_native exp28_adaptive
```

Outputs are written to `workspace/thesis/campaigns/<preset>/seed-<seed>/`.
Every run writes `progress.log`; GA runs also write `ga_state.json` and resume from
it automatically when the same command is rerun. Use `--dry-run` to validate the
preset and print the exact command without starting model work.

The presets share fitness definition `v2`, 192 search evaluations, Stage 1 limit
2, Stage 2 top K 3, Stage 2 limit 6, and a 12-example export evaluation. The
fine-tuning controls use 192 optimizer steps and emit eight parent-versus-trained
rows across target/retention suites and search/export protocols.

| Preset | Thesis arm |
| --- | --- |
| `exp26_aos_on` | Adaptive operator sampling enabled |
| `exp26_aos_off` | Fixed uniform method sampling |
| `exp27_native` | Native random-search control |
| `exp28_adaptive` | Adaptive GA on the mode-connected parent pair |
| `exp28_random` | Random search on the mode-connected parent pair |
| `exp29_probe` | Probe-only parent-distillation repair |
| `exp29_full` | Gated repair with continuation to 500 steps |
| `exp210_lora` | LoRA fine-tuning baseline |
| `exp210_full_ft` | Full fine-tuning baseline |
