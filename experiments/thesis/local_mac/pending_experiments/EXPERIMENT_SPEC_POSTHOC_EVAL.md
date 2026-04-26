# Experiment Spec: Full 8-Benchmark Post-Hoc Evaluation

## Purpose

Determine whether the 3-benchmark GA-based model merging optimisation (WikiText, BoolQ, SciQ) generalises to held-out benchmark tasks. This post-hoc evaluation populates the placeholder rows in Chapter 4's `tab:lmeval_results` and tests for benchmark overfitting—a critical validity check for the thesis claim of "broader capability beyond the optimisation objectives."

**Key Question:** Do merged models achieve comparable or superior performance on the three optimised benchmarks while avoiding catastrophic regression on the five held-out tasks?

---

## Models to Evaluate

| Model Identifier | Model Path | Role | Notes |
|---|---|---|---|
| `pythia-70m-deduped` | `EleutherAI/pythia-70m-deduped` | Parent baseline 1 | Stock Pythia-70M (base reference) |
| `pythia-70m-helpful-sft` | `lomahony/pythia-70m-helpful-sft` | Parent baseline 2 | Instruction-tuned parent (represents capability ceiling) |
| `seed11-final` | `workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed11/final_model` | GA-optimised merge (Seed 11) | **Best performer** on 3-benchmark fitness (from `best_config.yaml`) |
| `seed22-final` | `workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed22/final_model` | GA-optimised merge (Seed 22) | Mid-range performer for generalization check |
| `seed33-final` | `workspace/thesis/local_mac/results/20260321-thesis-pool-batch/seed33/final_model` | GA-optimised merge (Seed 33) | Lower fitness seed; tests robustness of approach |

**Total Models: 5**

---

## Tasks and Metrics

| Task | Task ID (lm_eval) | Metric | Few-Shot | Purpose | Optimisation Status |
|---|---|---|---|---|---|
| WikiText | `wikitext` | `byte_perplexity` | 0 | Language modeling on held-out Wikipedia | **OPTIMISED** (GA weight: 0.40) |
| LAMBADA | `lambada_openai` | `acc` | 0 | Zero-shot cloze; out-of-domain text | Held-out |
| BoolQ | `boolq` | `acc` | 0 | Boolean QA; reading comprehension | **OPTIMISED** (GA weight: 0.35) |
| SciQ | `sciq` | `acc` | 0 | Science QA; multiple choice | **OPTIMISED** (GA weight: 0.25) |
| PIQA | `piqa` | `acc` | 0 | Physical reasoning (commonsense) | Held-out |
| WinoGrande | `winogrande` | `acc` | 0 | Coreference resolution (commonsense) | Held-out |
| ARC Easy | `arc_easy` | `acc` | 0 | Science reasoning (easy subset) | Held-out |
| TruthfulQA MC2 | `truthfulqa_mc2` | `mc2` | 0 | Truthfulness and knowledge | Held-out |

**Total Tasks: 8**
**Optimised Tasks: 3** (wikitext, boolq, sciq)
**Held-Out Tasks: 5** (lambada_openai, piqa, winogrande, arc_easy, truthfulqa_mc2)

---

## Evaluation Configuration

### lm_eval Parameters
```bash
python -m lm_eval \
    --model hf \
    --model_args "pretrained=MODEL_PATH,trust_remote_code=True" \
    --tasks wikitext,lambada_openai,boolq,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2 \
    --num_fewshot 0 \
    --device cpu \
    --batch_size 1 \
    --output_path RESULTS_PATH \
    --log_samples
```

### Rationale for Configuration
- **num_fewshot=0**: Matches GA training regime (zero-shot); no data leakage from optimisation
- **device=cpu**: Ensures reproducibility across environments (no device-specific variance)
- **batch_size=1**: Conservative for Pythia-70M (~335M parameters); reduces memory variance
- **log_samples**: Enables post-hoc error analysis (debugging regressions if they occur)
- **trust_remote_code=True**: Required for custom merged model architectures

---

## Run Command

```bash
bash /sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/run_posthoc_eval.sh
```

**Output Structure:**
```
workspace/thesis/local_mac/results/posthoc_eval/
├── pythia-70m-deduped/
│   ├── results.json
│   └── results_samples.json
├── pythia-70m-helpful-sft/
│   ├── results.json
│   └── results_samples.json
├── seed11-final/
│   ├── results.json
│   └── results_samples.json
├── seed22-final/
│   ├── results.json
│   └── results_samples.json
└── seed33-final/
    ├── results.json
    └── results_samples.json
```

Each `results.json` contains metrics for all 8 tasks.

---

## Expected Runtime

| Model | Est. Time per Model | Notes |
|---|---|---|
| **Per-model** | 30–45 min | M1 CPU; batch_size=1 is conservative; wikitext dominates due to large eval set |
| **Total (5 models)** | ~3–3.75 hours | Sequential evaluation (no parallelisation); can run overnight |

**Timeline:**
- Start: ~1 model/45 min
- Recommend: Launch late afternoon, collect results by morning
- Alternative: Parallelise across 2–3 shells on separate cores if time-critical

---

## What to Collect

### Primary Results
1. **Per-Model Metrics** (from each `results.json`):
   - WikiText: `byte_perplexity`
   - LAMBADA: `acc` (accuracy)
   - BoolQ: `acc`
   - SciQ: `acc`
   - PIQA: `acc`
   - WinoGrande: `acc`
   - ARC Easy: `acc`
   - TruthfulQA MC2: `mc2` (macro-averaged)

### Secondary Results (for appendix)
2. **Variance** (if lm_eval reports stderr or confidence intervals)
3. **Sample logs** (from `results_samples.json`; useful for debugging pathological failures)
4. **Compute metadata** (execution time per task; useful for reproducibility claim)

### Synthesis
- **Raw table**: All 5 models × 8 tasks → 40 cells of data
- **Delta analysis**: Compare each merged model against both parents (element-wise % difference)
- **Aggregate metrics**: Mean performance on optimised vs. held-out task groups

---

## Thesis Update: tab:lmeval_results

The experiment directly populates Chapter 4's benchmark table with empirical results. The table structure is:

```latex
\begin{table}[H]
\caption{Full 8-Task Benchmark Evaluation: Baseline and GA-Merged Models}
\label{tab:lmeval_results}
\small
\begin{tabular}{lccccccccc}
\toprule
Model & WikiText (↓) & LAMBADA & BoolQ & SciQ & PIQA & WinoGrande & ARC-Easy & TruthfulQA MC2 \\
  & perp & acc & acc & acc & acc & acc & acc & mc2 \\
\midrule
EleutherAI/pythia-70m-deduped & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
lomahony/pythia-70m-helpful-sft & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 11 (Final Merge) & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 22 (Final Merge) & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 33 (Final Merge) & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
\bottomrule
\end{tabular}
\end{table}
```

**Legend:**
- ↓ = lower-is-better (WikiText perplexity)
- All other metrics are accuracy-like (higher-is-better)
- Mark statistically significant improvements (vs. best parent) with `*` or bold

---

## Analysis Questions to Answer

### Primary Hypotheses

**Q1: Do merged models match or exceed parents on optimised tasks?**
- *Hypothesis:* Seed 11 ≥ max(pythia-70m-deduped, pythia-70m-helpful-sft) on {wikitext, boolq, sciq}
- *Success criterion:* ≥90% of optimisation budget translated to test performance
- *Why it matters:* Validates GA optimiser worked (not a false positive from overfitting in the search)

**Q2: Do merged models regress on held-out tasks?**
- *Hypothesis:* Merged models show modest degradation (0–5% acc drop) on {lambada, piqa, winogrande, arc_easy, truthfulqa}
- *Success criterion:* Max single-task regression < 10% absolute (e.g., 65% → 59%)
- *Why it matters:* Tests whether 3-benchmark GA creates catastrophic negative transfer

**Q3: Does Seed 33 (lower fitness) also underperform equally on held-out tasks?**
- *Hypothesis:* All three seeds show similar hold-out regression patterns (not Seed 11 uniquely pathological)
- *Success criterion:* Seed 33 regression profile ≈ Seed 11 regression profile (within 2% per task)
- *Why it matters:* Disentangles "GA overfitting artefact" from "inherent merge geometry difficulty"

### Secondary Analyses

**Q4: Is there a Pareto frontier?**
- Plot all 5 models on 2D: (optimised-task-fitness, held-out-task-fitness)
- Do merged models strictly dominate parents on optimised tasks but sacrifice held-out performance?
- *Insight:* Characterises the trade-off; good narrative for thesis conclusion

**Q5: Which held-out task is hardest for merged models?**
- Rank held-out tasks by mean Δ(merged − parent)
- Identify any pathological tasks (e.g., WinoGrande coreference → syntax destroyed by merge?)
- *Insight:* Hypothesis for future work

---

## Results Table Template

Use this LaTeX block as your starting point for `tab:lmeval_results` in Chapter 4:

```latex
\begin{table}[H]
\centering
\caption{Post-Hoc Evaluation: Full 8-Task Benchmark Suite. Baseline Models and GA-Merged Pythia-70M. Lower = better for WikiText; higher = better for all other tasks.}
\label{tab:lmeval_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{WikiText} & \textbf{LAMBADA} & \textbf{BoolQ} & \textbf{SciQ} & \textbf{PIQA} & \textbf{WinoGrande} & \textbf{ARC-E} & \textbf{TruthfulQA} \\
 & \textit{ppl} & \textit{acc} & \textit{acc} & \textit{acc} & \textit{acc} & \textit{acc} & \textit{acc} & \textit{mc2} \\
\midrule
pythia-70m-deduped (Parent 1) & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
pythia-70m-helpful-sft (Parent 2) & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 11 Final Merge & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 22 Final Merge & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
Seed 33 Final Merge & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} & \placeholder{} \\
\bottomrule
\end{tabular}
\vspace{0.5em}
\footnotesize
\textit{Note:} Post-hoc evaluation on held-out benchmark suite. WikiText, BoolQ, and SciQ were GA optimisation objectives; LAMBADA, PIQA, WinoGrande, ARC-Easy, and TruthfulQA-MC2 are held-out generalisation tests. All evaluations zero-shot (\texttt{num\_fewshot=0}) on CPU.
\end{table}
```

**Formatting hints for final table:**
- Use `\textbf{}` or `\mathbf{}` for best-in-column entries (or highlight row-by-row)
- Add footnote explaining optimised vs. held-out task grouping
- Consider adding a "Δ (vs. Parent 2)" column if space permits (shows delta from pythia-70m-helpful-sft baseline)

---

## Key Risk: Benchmark Overfitting

**Scenario:** What if merged models regress substantially on held-out tasks (e.g., 5–15% absolute accuracy drop on PIQA, WinoGrande)?

**Why it could happen:**
- GA optimisation on 3 narrow benchmarks creates a "fitness valley" in weight space that is unfavourable for broader capabilities
- Model merge geometry (non-convex loss landscape) may trap the GA in a local optimum with poor generalisation
- Linear/SLERP merging methods are rigid; task-specific overfitting is well-documented in multi-task learning

**Why it's NOT a failure:**
- **Still publishable:** "GA-based model merging exhibits benchmark overfitting—a cautionary finding for practitioners"
- **Scientific value:** Negative results are valuable; they characterise the limitations of the approach
- **Actionable:** Generates hypotheses for mitigation (multi-task GA fitness; regularisation; task-aware merge methods)
- **Thesis narrative:** "We identified and quantified a critical trade-off in task-specific model merging"

**Mitigation (if observed):**
1. Report the overfitting honestly in Chapter 4
2. Propose explanations in Discussion (e.g., merge method limitations)
3. Add "Future Work" section: "Multi-objective GA with held-out task penalties" or "Pareto-optimal merging"
4. Conclude: *Despite overfitting, the approach advances understanding of model merge optimisation*

---

## Data Collection and Analysis Workflow

### Step 1: Run Evaluation
```bash
bash /sessions/quirky-cool-bohr/mnt/local_mac/pending_experiments/run_posthoc_eval.sh
# Monitor output for errors; expect ~3.5 hours total
```

### Step 2: Extract Results
For each model, parse `results.json` and extract the 8 task metrics into a CSV or JSON summary:
```
model,wikitext,lambada_openai,boolq,sciq,piqa,winogrande,arc_easy,truthfulqa_mc2
pythia-70m-deduped,XXX,XXX,XXX,XXX,XXX,XXX,XXX,XXX
pythia-70m-helpful-sft,XXX,XXX,XXX,XXX,XXX,XXX,XXX,XXX
seed11-final,XXX,XXX,XXX,XXX,XXX,XXX,XXX,XXX
seed22-final,XXX,XXX,XXX,XXX,XXX,XXX,XXX,XXX
seed33-final,XXX,XXX,XXX,XXX,XXX,XXX,XXX,XXX
```

### Step 3: Statistical Analysis (Optional)
- Compute Δ (merged − best parent) for each task
- Flag improvements ≥5% and regressions ≤−5%
- Report mean Δ on optimised vs. held-out task groups

### Step 4: Fill Thesis Table
Populate `tab:lmeval_results` in Chapter 4 with extracted values.

### Step 5: Narrative
- Discuss whether results support the thesis claim of "broad capability"
- Interpret any overfitting patterns in the Discussion section
- Update Abstract/Conclusion if results are surprising

---

## Version Info

- **Document Version:** 1.0
- **Date Created:** 2026-03-21
- **GA Config Source:** `thesis_run_pythia70m.yml` (3 optimised tasks, structured_phase1_tiny fitness mode)
- **Expected Result Date:** 2026-03-21 + ~4 hours (overnight run)
- **Thesis Chapter:** Chapter 4 (Empirical Evaluation)
