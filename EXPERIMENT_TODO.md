# Experiment TODO — MergeKit GA Thesis Experiments
**Thesis:** Evolving Large Language Models using Model Merging through Genetic Algorithms
**Author:** Nkululeko Thangelane | **Supervisor:** Prof AP Engelbrecht
**Hardware:** Apple M1 MacBook Pro, 16GB Unified Memory
**Last Updated:** 2026-03-15

---

## Current Status

### EKS Scale-Out Track
- [ ] Confirm GPU uplift on the aligned 3B benchmark suite against a valid CPU reference.
- [ ] Complete `qwen25-3b-aligned-seed11` on EKS with baseline enabled.
- [ ] Complete `qwen25-3b-aligned-seed22` on EKS with baseline enabled.
- [ ] Compare seed-to-seed throughput and score stability from `ga_history.csv`, `baseline_results.csv`, and `ga_summary.txt`.
- [ ] Freeze the 3B benchmark suite and experiment shape for thesis reporting.
- [ ] Request AWS GPU quota increase for 7B (`g6e` or larger) only after the 3B runs are stable.
- [ ] Run the first 7B pilot after quota approval and larger GPU workers are in place.

### Active EKS Configuration
- Cluster: `mergekit-ga-scale` (`us-east-1`)
- Active 3B CPU nodes: `m6i.2xlarge` x2 (`scale=3b`)
- Active GPU node: `g6.xlarge` x1 (`NVIDIA L4`)
- Aligned benchmark suite: `ifeval`, `xnli_vi`, `boolq`, `multirc`
- Current runtime guardrails: baseline enabled, `apply_chat_template: false`, `limit` overridden from CLI for pilot runs

### Completed Runs
| Run | Models | Pop | Gens | Best Score | Method | Tasks |
|-----|--------|-----|------|-----------|--------|-------|
| `pythia70m_ga_m1_experiment_run_50gen` | Pythia-70M (2 models) | 8 | 50 | 0.44375 | TIES | lambada, sciq |
| `slm-merge-1000gen6` | GPT-2 variants | 36 | 1000 | 0.421875 | ties/dare_ties/nuslerp | lambada, sciq |

### Baseline Scores (from `baseline_results.csv`)
| Model | Weighted Score | LAMBADA | SciQ |
|-------|---------------|---------|------|
| lomahony/pythia-70m-helpful-sft | 0.44375 | 0.28125 | 0.6875 |
| EleutherAI/pythia-70m-deduped | 0.41875 | 0.28125 | 0.625 |
| mlabonne/chesspythia-70m | 0.21875 | 0.03125 | 0.5 |
| vineetsharma/databricks-dolly-15k-pythia-70m-deduped-v1 | 0.1375 | 0.0 | 0.34375 |

---

## Planned Experiment Matrix

| Exp # | Experiment Name | Recommended Size | Model A | HF Link A | Model B | HF Link B | Architecture | Merge Motivation | Recommended Population | Recommended Generations | Allowed Merge Operations | Evaluation Metrics | Preset |
|-------|-----------------|------------------|---------|-----------|---------|-----------|--------------|------------------|------------------------|--------------------------|--------------------------|--------------------|--------|
| 1 | Tiny Controlled Merge Baseline | 33M-160M | TinyStories-33M | [roneneldan/TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) | TinyStories-Instruct-33M | [roneneldan/TinyStories-Instruct-33M](https://huggingface.co/roneneldan/TinyStories-Instruct-33M) | GPT-Neo | Fast, cheap baseline to validate GA logic, chromosome design, mutation strategy, and fitness behavior before moving to larger models. | 20-30 | 15-25 | Linear, SLERP, TIES, DARE variants, layer-wise crossover, coefficient mutation | Perplexity, held-out generation quality, lightweight QA accuracy, convergence, merge runtime | [`experiments/thesis/exp01_tiny_controlled_merge/config.yml`](experiments/thesis/exp01_tiny_controlled_merge/config.yml) |
| 2 | Controlled Base + Chat Merge | 2.8B | Pythia-2.8B | [EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b) | Pythia-2.8B Synthetic Instruct | [lambdalabs/pythia-2.8b-deduped-synthetic-instruct](https://huggingface.co/lambdalabs/pythia-2.8b-deduped-synthetic-instruct) | GPT-NeoX | Same-family base vs instruct experiment to study knowledge retention versus alignment and instruction behavior. | 16-24 | 12-20 | Linear, SLERP, TIES, DARE variants, task arithmetic, layer-range crossover, weighted block merge | Perplexity, MMLU, GSM8K, IFEval proxy for instruction quality, inference memory | [`experiments/thesis/exp02_pythia28b_base_chat/config.yml`](experiments/thesis/exp02_pythia28b_base_chat/config.yml) |
| 3 | Multilingual Base + Instruct Merge | 3B | Qwen2.5-3B | [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) | Qwen2.5-3B-Instruct | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Qwen2 | Tests whether merging base and instruct models improves instruction following while keeping multilingual and general reasoning capability. | 12-20 | 10-18 | Linear, SLERP, TIES, DARE variants, layer-wise crossover, coefficient mutation | MMLU, XNLI, GSM8K, IFEval proxy for instruction quality, perplexity | [`experiments/thesis/exp03_qwen25_3b_multilingual_merge/config.yml`](experiments/thesis/exp03_qwen25_3b_multilingual_merge/config.yml) |
| 4 | General + Code Specialist Merge | 7B | Mistral-7B-v0.1 | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | Mistral-7B-Instruct-v0.2 | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | Mistral | Stronger practical experiment to test whether GA merging preserves general reasoning while improving structured or code-adjacent behavior. | 10-16 | 8-15 | Linear, SLERP, TIES, DARE variants, block merge by transformer layers, residual-weight interpolation | MMLU, GSM8K, IFEval, ARC-Easy, perplexity, external code eval, tokens/sec, VRAM usage | [`experiments/thesis/exp04_mistral7b_general_code/config.yml`](experiments/thesis/exp04_mistral7b_general_code/config.yml) |
| 5 | Large-Scale Same-Family Merge | 8B | Llama-3-8B | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | Llama-3-8B-Instruct | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Llama 3 | Strong thesis-scale same-family merge to test whether GA can recover base-model breadth while preserving instruct alignment. | 8-12 | 6-12 | Linear, SLERP, TIES, DARE variants, layer-band crossover, merge-weight mutation | MMLU, GSM8K, TruthfulQA, IFEval proxy for chat quality, perplexity, latency, memory footprint | [`experiments/thesis/exp05_llama3_8b_same_family/config.yml`](experiments/thesis/exp05_llama3_8b_same_family/config.yml) |

### Notes For The Matrix

- Hugging Face repo IDs were verified on `2026-03-15` against model configs before creating the presets.
- `togethercomputer/Pythia-Chat-Base-2.8B` did not resolve on `2026-03-15`, so the 2.8B preset uses [`lambdalabs/pythia-2.8b-deduped-synthetic-instruct`](https://huggingface.co/lambdalabs/pythia-2.8b-deduped-synthetic-instruct) as the closest same-family instruct model.
- `Qwen/Qwen2-3B` and `Qwen/Qwen2-3B-Instruct` did not resolve on `2026-03-15`, so the 3B preset uses the verified [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B) pair.
- Recommended generations are expressed operationally via `--max-fevals`, not a YAML field. Suggested mappings are: Exp 1 `24 x 20 = 480`, Exp 2 `20 x 16 = 320`, Exp 3 `16 x 14 = 224`, Exp 4 `12 x 10 = 120`, Exp 5 `10 x 8 = 80`.
- The GA YAMLs encode the lm-eval metrics that are directly supported in this repo. BLEU/ROUGE, AlpacaEval or MT-Bench, HumanEval or MBPP, runtime, latency, tokens per second, VRAM, and memory footprint should be collected as post-run analyses for the best checkpoints.
- Experiments 2-5 include benchmark tasks such as `mmlu`, `gsm8k`, and `truthfulqa_mc`; run them with `--i-understand-the-depths-of-the-evils-i-am-unleashing`.

---

## Phase 1: Core Experiments (REQUIRED FOR THESIS SUBMISSION)

### Experiment 1.1 — Full Benchmark Evaluation of Parent Models
**Priority:** CRITICAL (populates Table 4.1 in thesis)
**Status:** NOT STARTED

Run `lm-eval-harness` on ALL parent models individually with the FULL task suite:

```bash
# For each parent model:
lm_eval --model hf \
  --model_args pretrained=EleutherAI/pythia-70m-deduped,trust_remote_code=True \
  --tasks wikitext,lambada_openai,sst2,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
  --batch_size 1 \
  --output_path results/parent_pythia70m_deduped.json
```

**Models to evaluate:**
- [ ] `EleutherAI/pythia-70m-deduped` (base)
- [ ] `lomahony/pythia-70m-helpful-sft` (instruction-tuned)
- [ ] `mlabonne/chesspythia-70m` (domain-specialised)
- [ ] `vineetsharma/databricks-dolly-15k-pythia-70m-deduped-v1` (dolly fine-tune)

**Benchmarks (8 total):**
- [ ] WikiText (perplexity ↓)
- [ ] LAMBADA (accuracy ↑)
- [ ] SST-2 (accuracy ↑)
- [ ] SciQ (accuracy ↑)
- [ ] PIQA (accuracy ↑)
- [ ] Winogrande (accuracy ↑)
- [ ] ARC-Easy (accuracy ↑)
- [ ] TruthfulQA-MC2 (accuracy ↑)

**Output:** JSON files → parse into LaTeX table → update `05_results.tex` Table 4.1

---

### Experiment 1.2 — Baseline Merges (Non-GA)
**Priority:** CRITICAL (provides baseline comparison)
**Status:** NOT STARTED

Merge parent models using EACH heuristic method WITHOUT GA optimisation:

```bash
# Example: Simple Average
mergekit-yaml merge \
  --config configs/naive_avg.yaml \
  --out-path workspace/baseline_naive_avg/

# Then evaluate:
lm_eval --model hf \
  --model_args pretrained=workspace/baseline_naive_avg/ \
  --tasks wikitext,lambada_openai,sst2,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
  --batch_size 1 \
  --output_path results/baseline_naive_avg.json
```

**Merge methods to test:**
- [ ] `linear` (simple averaging with equal weights)
- [ ] `slerp` (spherical linear interpolation, t=0.5)
- [ ] `ties` (default density=0.5, no weight masks)
- [ ] `task_arithmetic` (default scaling)
- [ ] `dare_linear` (p=0.5 drop rate)
- [ ] `dare_ties` (combined DARE + TIES)

**Output:** 6 baseline merge scores across all 8 benchmarks

---

### Experiment 1.3 — GA-Optimised Merge (Full Benchmark Suite)
**Priority:** CRITICAL
**Status:** PARTIALLY DONE (50gen run exists but only on 2 tasks)

Re-run the GA with the FULL 8-benchmark fitness function:

```yaml
# ga_full_benchmark.yml
genome_type: multi_method
models:
  - EleutherAI/pythia-70m-deduped
  - lomahony/pythia-70m-helpful-sft
  - mlabonne/chesspythia-70m
allowed_methods:
  - linear
  - ties
  - task_arithmetic
  - dare_ties
  - slerp
  - nuslerp
tasks:
  - lambada_openai
  - sciq
  - sst2
  - piqa
  - winogrande
  - arc_easy
  - truthfulqa_mc2
task_weights:
  lambada_openai: 0.15
  sciq: 0.15
  sst2: 0.10
  piqa: 0.15
  winogrande: 0.15
  arc_easy: 0.15
  truthfulqa_mc2: 0.15
ga:
  population_size: 8
  generations: 50
  tournament_size: 2
  crossover_prob: 0.5
  mutation_sigma: 0.06
  sigma_decay: 0.5
  patience: 5
  elite_fraction: 0.125
```

**IMPORTANT M1 CONSTRAINTS:**
- `batch_size: 1` to avoid OOM
- Use `limit` per task if evaluation is too slow (e.g., `limit: 100`)
- Enable hash-based caching to skip duplicate genotypes
- Expect ~1 minute per evaluation, ~8 minutes per generation, ~6.5 hours for 50 gens

**Output:** `ga_history.csv`, `best_config.yaml`, full benchmark scores for best merged model

---

### Experiment 1.4 — Evaluate Best GA-Merged Model on Full Suite
**Priority:** CRITICAL
**Status:** NOT STARTED

After Experiment 1.3 completes, evaluate the best merged model on ALL 8 benchmarks:

```bash
lm_eval --model hf \
  --model_args pretrained=workspace/ga_full_benchmark/final_model/ \
  --tasks wikitext,lambada_openai,sst2,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
  --batch_size 1 \
  --output_path results/ga_merged_best.json
```

Also evaluate the best config from the existing 50-gen run:
```bash
lm_eval --model hf \
  --model_args pretrained=workspace/pythia70m_ga_m1_experiment_run_50gen/final_model/ \
  --tasks wikitext,lambada_openai,sst2,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
  --batch_size 1 \
  --output_path results/ga_merged_50gen.json
```

---

## Phase 2: Ablation Studies (STRONGLY RECOMMENDED)

### Experiment 2.1 — Population Size Ablation
**Priority:** HIGH (addresses GAP-16 — "population too small")

Run the GA with different population sizes, keeping other params fixed:
- [ ] Pop = 4, Gens = 50
- [ ] Pop = 8, Gens = 50 (current default)
- [ ] Pop = 16, Gens = 50
- [ ] Pop = 32, Gens = 25 (same total evals as pop=16×50)

**Expected output:** Table showing convergence speed and final fitness vs. population size.

---

### Experiment 2.2 — Generation Count Ablation
**Priority:** HIGH (addresses GAP-16)

Run the GA with fixed pop=8 but varying generations:
- [ ] 10 generations
- [ ] 25 generations
- [ ] 50 generations
- [ ] 100 generations

**Expected output:** Convergence curves showing when GA plateaus.

---

### Experiment 2.3 — Merge Method Comparison
**Priority:** HIGH

Run single-method GA (restricting `allowed_methods` to one) for each method:
- [ ] GA + linear only
- [ ] GA + ties only
- [ ] GA + task_arithmetic only
- [ ] GA + slerp only
- [ ] GA + dare_ties only
- [ ] GA + nuslerp only
- [ ] GA + multi-method (all allowed — current default)

**Expected output:** Table showing which method benefits most from GA optimisation.

---

### Experiment 2.4 — Fitness Weight Sensitivity Analysis
**Priority:** MEDIUM (addresses GAP-18)

Vary task weights in the fitness function:
- [ ] Equal weights (1/K for all tasks)
- [ ] Reasoning-heavy (2× weight on PIQA, Winogrande, ARC)
- [ ] Language-heavy (2× weight on LAMBADA, WikiText)
- [ ] Truthfulness-heavy (2× weight on TruthfulQA-MC2)

**Expected output:** Table showing how fitness weighting affects task-level performance.

---

### Experiment 2.5 — Crossover Operator Comparison
**Priority:** MEDIUM

Run GA with each crossover operator:
- [ ] Arithmetic crossover (default)
- [ ] Uniform crossover
- [ ] SBX crossover

**Expected output:** Convergence curves comparing operators.

---

## Phase 3: Extended Experiments (FUTURE WORK)

### Experiment 3.1 — African Language Evaluation
**Priority:** HIGH (addresses GAP-19 — most critical societal motivation gap)
**Status:** REQUIRES AFRICAN-LANGUAGE MODELS

**Step 1: Identify compatible African-language Pythia/GPT-2 fine-tunes:**
- Search HuggingFace for Pythia-70M models fine-tuned on African-language data
- If none exist, fine-tune Pythia-70M on a small African-language corpus (e.g., MasakhaNER training set)

**Step 2: Merge African-language model with English-specialised model using GA**

**Step 3: Evaluate on:**
- [ ] MasakhaNER (NER accuracy across 10 African languages)
- [ ] AfriSenti (sentiment analysis, 14 African languages)
- [ ] SIB-200 subset (topic classification)

---

### Experiment 3.2 — Scaling to Larger Models
**Priority:** MEDIUM (may require cloud GPU)

- [ ] Pythia-160M variants (if M1 memory allows)
- [ ] Pythia-410M variants (may need quantisation)

---

### Experiment 3.3 — Multi-Objective Optimisation (NSGA-II)
**Priority:** MEDIUM (addresses GAP-3)

Replace weighted-sum fitness with NSGA-II to produce Pareto front of:
- Accuracy (averaged across tasks)
- Efficiency (latency / throughput)
- (Optionally) Fairness metric

**Implementation:** Extend `GAOptimizer` with NSGA-II selection and crowding distance.

---

### Experiment 3.4 — Knowledge Distillation (Phase 3 of Thesis Design)
**Priority:** LOW (described in methodology but no results yet)

Take the best GA-merged model and distil into a smaller student:
- Teacher: best GA-merged Pythia-70M
- Student: custom smaller Pythia variant (e.g., 4 layers instead of 6)
- Distillation method: KL-divergence on logits

---

## Data Collection Checklist

After each experiment, collect and save:

- [ ] `ga_history.csv` (generation-level metrics)
- [ ] `best_config.yaml` (winning merge recipe)
- [ ] `baseline_results.csv` (individual model scores)
- [ ] Full `lm-eval` JSON output files for each model
- [ ] Timing data (seconds per generation, total wall clock)
- [ ] System info (M1 memory usage, CPU temp if available)
- [ ] Screenshots of convergence plots from MLflow

---

## Results → Thesis Mapping

| Experiment | Thesis Table/Figure |
|-----------|-------------------|
| 1.1 (Parent baselines) | Table 4.1 `tab:lmeval_results` — parent model rows |
| 1.2 (Heuristic baselines) | Table 4.1 rows for M0, and new rows for SLERP/TIES/TaskArith |
| 1.3 + 1.4 (GA merge) | Table 4.1 rows for M1, M2; Table 4.2 `tab:sim-results` |
| 1.3 convergence | Figure 4.1 `fig:ga_convergence` — real convergence curve |
| 2.1 (Pop ablation) | New Table 4.X — Population size vs. fitness |
| 2.2 (Gen ablation) | New Figure 4.X — Convergence vs. generation budget |
| 2.3 (Method comparison) | New Table 4.X — Per-method GA performance |
| 2.4 (Fitness weights) | New Table 4.X — Sensitivity analysis |
| Efficiency data | Table 4.3 `tab:sim-efficiency` — latency, throughput |

---

## Quick Start Commands

```bash
# Navigate to mergekit
cd /Users/nkululekothangelane/Documents/master_research/mergekit

# Activate environment
source venv/bin/activate  # or conda activate mergekit

# Run parent model evaluation (Experiment 1.1)
python -m lm_eval --model hf \
  --model_args pretrained=EleutherAI/pythia-70m-deduped \
  --tasks wikitext,lambada_openai,sst2,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
  --batch_size 1 \
  --output_path results/parent_pythia70m_deduped.json

# Run GA experiment (Experiment 1.3)
python -m mergekit.evo.run \
  --config workspace/ga_full_benchmark.yml \
  --output workspace/ga_full_benchmark/ \
  --storage workspace/ga_full_benchmark/storage/

# Check progress
tail -f workspace/ga_full_benchmark/ga_history.csv
```

---

## Priority Order (Time-Constrained Path)

If time is limited, complete experiments in this order:

1. **Experiment 1.1** (2-3 hours) — Parent baselines on full suite
2. **Experiment 1.2** (3-4 hours) — Heuristic merge baselines
3. **Experiment 1.4** (30 min) — Evaluate existing best merge on full suite
4. **Experiment 1.3** (6-8 hours) — Full GA run with 8 benchmarks
5. **Experiment 2.3** (12-24 hours) — Method comparison (most publishable ablation)
6. **Experiment 2.1** (12-24 hours) — Population size ablation

This gets you a defensible thesis with real results in ~2-3 days of compute.

---

*Generated for Nkululeko Thangelane's Master's Thesis, Stellenbosch University*
*Hardware: Apple M1 MacBook Pro, 16GB Unified Memory*
