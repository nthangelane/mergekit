# Pythia-70M Thesis Run And Benchmark Plan

This plan defines the longer local 70M runs that are suitable for thesis
reporting on the development machine.

## Search Preset

- Config:
  [`thesis_run_pythia70m.yml`](/Users/nkululekothangelane/Documents/master_research/mergekit/experiments/thesis/local_mac/thesis_run_pythia70m.yml)
- Output root:
  `workspace/thesis/local_mac/results/20260319-thesis-run`
- Execution mode:
  serial, CPU-only, baseline enabled

Resolved search budget from the config:
- population size: `8`
- hard evaluation limit: `192`
- hard wall-clock limit: `4h`
- target stop: `+5%` over best baseline
- target gating: stage-2/full score only, not stage-1
- stagnation stop: `8` generations with improvement `< 0.005`

## Seeded Search Runs

Run the same preset three times to measure stability across random seeds:

1. Seed `11`
2. Seed `22`
3. Seed `33`

Recommended commands:

```bash
python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/thesis_run_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/20260319-thesis-run/seed11 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 11

python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/thesis_run_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/20260319-thesis-run/seed22 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 22

python -m mergekit.scripts.evolve_ga \
  experiments/thesis/local_mac/thesis_run_pythia70m.yml \
  --storage-path workspace/thesis/local_mac/results/20260319-thesis-run/seed33 \
  --strategy serial \
  --num-gpus 0 \
  --no-merge-cuda \
  --batch-size 1 \
  --baseline \
  --no-reshard \
  --random-seed 33
```

Each run should retain:
- `baseline_results.csv`
- `ga_history.csv`
- `ga_method_history.csv`
- `ga_candidate_history.csv`
- `ga_summary.txt`
- `ga_stop_details.json`
- `best_config.yaml`
- `final_model/`
- `ga_final_comparison.csv`
- `ga_history_plot.png`
- `ga_final_comparison.png`

## Higher-Fidelity Benchmark

After the three search runs finish:

1. Collect the winning `final_model` from each seed.
2. Keep both parent baselines for comparison:
   - `EleutherAI/pythia-70m-deduped`
   - `lomahony/pythia-70m-helpful-sft`
3. Re-evaluate the top merged models at a higher limit than the search loop.

Recommended confirmation limit:
- `32`

Recommended confirmation tasks:
- `wikitext`
- `boolq`
- `sciq`

The confirmation table should include:
- model label
- source seed
- merge method
- weighted score
- `wikitext` byte perplexity
- `boolq` accuracy
- `sciq` accuracy
- delta vs deduped base
- delta vs helpful-sft
- delta vs best baseline in percent

## Thesis Tables

Prepare three tables:

1. Search summary
   - seed
   - generations completed
   - fevals completed
   - stop reason
   - best score
   - best generation
   - best merge method
   - delta vs best baseline
   - delta vs best baseline percent

2. Operator behavior
   - method
   - usage count
   - success rate
   - parent improvement rate
   - survival rate
   - final sampling probability

3. Final benchmark comparison
   - model
   - weighted score
   - `wikitext` byte perplexity
   - `boolq` accuracy
   - `sciq` accuracy
   - delta vs deduped
   - delta vs helpful-sft
   - delta vs best baseline percent

## Narrative Focus

The write-up should answer:

- Did adaptive local GA find a merged model better than both parent baselines?
- Did the new stop policy stop because of target gain, stagnation, time, or hard budget?
- Were gains consistent across seeds?
- Which merge methods dominated or survived longest?
- Did `linear` or `slerp` deliver the strongest merged model under this local budget?
