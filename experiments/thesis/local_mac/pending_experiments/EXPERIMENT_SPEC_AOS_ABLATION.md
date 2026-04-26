# Experiment Spec: AOS Credit Weight Ablation

## Purpose

Determine whether Adaptive Operator Selection (AOS) adaptation direction is robust across different credit weight parameterizations, or sensitive to the specific choice of weights (0.50/0.30/0.20). This ablation tests a core design choice in the AOS mechanism and addresses an anticipated examiner question: "Are your results dependent on those arbitrary weights?"

---

## Variants

| Variant | Config | Weights | Rationale |
|---------|--------|---------|-----------|
| **A: Equal** | `thesis_aos_equal_weights.yml` | 0.333 / 0.333 / 0.333 | Removes prioritization. All three credit signals (child fitness, parent improvement, survival) contribute equally. |
| **B: Fitness-only** | `thesis_aos_fitness_only.yml` | 1.0 / 0.0 / 0.0 | Isolates the effect of child fitness. Most aggressive variant; AOS responds only to recent operator performance. |
| **C: Current (baseline)** | `thesis_aos_current.yml` | 0.50 / 0.30 / 0.20 | Standard thesis configuration. Prioritizes child fitness (50%) over parent improvement (30%) and survival (20%). |

---

## Key Question

**Does slerp get de-emphasised in all three variants (indicating robust AOS and genuine slerp underperformance), or only with the current weights (indicating fragile AOS)?**

**Thesis-level implication:**
- **Convergent result (robust):** All three variants converge to similar final operator probabilities → AOS adaptation is a real, stable phenomenon. Weight choice (0.50/0.30/0.20) does not materially drive the result.
- **Divergent result (fragile):** Final probabilities differ significantly across variants → Thesis must discuss sensitivity and either justify the 0.50/0.30/0.20 choice more carefully or select a more robust parameterization.

---

## Run Commands

```bash
cd /sessions/quirky-cool-bohr/mnt/local_mac

# Run all three variants sequentially with logging
bash run_aos_ablation.sh

# Or run individually:
python -m thesis_run --config thesis_aos_equal_weights.yml --seed 11 --output_dir workspace/thesis/local_mac/results/aos_ablation/equal_weights

python -m thesis_run --config thesis_aos_fitness_only.yml --seed 11 --output_dir workspace/thesis/local_mac/results/aos_ablation/fitness_only

python -m thesis_run --config thesis_aos_current.yml --seed 11 --output_dir workspace/thesis/local_mac/results/aos_ablation/current
```

---

## Expected Runtime

- **Per variant:** ~85 minutes (same as seed=11 baseline run with 192 fevals, 13 generations)
- **Total:** ~4.5 hours (including I/O overhead)
- **Seed:** All three use `seed=11` for direct comparison

---

## What to Collect

From each variant's output directory, extract and compare:

```
workspace/thesis/local_mac/results/aos_ablation/
├── equal_weights/
│   ├── ga_method_history.csv          ← PRIMARY DATA
│   ├── fitness.csv
│   ├── generation_summary.json
│   └── final_genome.json
├── fitness_only/
│   ├── ga_method_history.csv          ← PRIMARY DATA
│   ├── fitness.csv
│   ├── generation_summary.json
│   └── final_genome.json
└── current/
    ├── ga_method_history.csv          ← PRIMARY DATA
    ├── fitness.csv
    ├── generation_summary.json
    └── final_genome.json
```

**Key file:** `ga_method_history.csv` contains per-generation probabilities for `passthrough`, `linear`, and `slerp`.

---

## What to Measure: AOS Adaptation Direction

### Primary Metric: Final Operator Probabilities

Extract the last row of each `ga_method_history.csv` and compare:

| Variant | p_passthrough(final) | p_linear(final) | p_slerp(final) | Notes |
|---------|---|---|---|---|
| Equal weights | ? | ? | ? | |
| Fitness-only | ? | ? | ? | |
| Current | ? | ? | ? | Baseline |

### Secondary Metrics (trajectory analysis)

For each variant, track:
- **Generation where slerp starts declining** (if at all)
- **Rate of slerp decline** (gradual vs. sharp)
- **Final plateau vs. continued drift** in later generations
- **Fitness improvement vs. method adaptation** (do method changes correlate with fitness gains?)

### Visualization Suggestion

Plot all three `ga_method_history.csv` curves on the same graph:
```
Operator Probability vs Generation
  Equal weights: ─ ─ ─ (dashed)
  Fitness-only:  ━ ━ ━ (bold)
  Current:       ─ ─ ─ (solid baseline)
```

---

## Thesis Update

### Chapter 3: AOS Mechanism & Adaptation

Add a subsection (≈3–5 sentences) after the main AOS results:

**Current text (placeholder):**
> "The AOS algorithm adapted method probabilities over 13 generations, de-emphasising slerp in favour of linear and passthrough. Operator credit weights were set to 0.50 (child fitness), 0.30 (parent improvement), and 0.20 (survival)."

**Revised text (with ablation result):**
> "To investigate the robustness of AOS adaptation, we conducted a weight ablation study with three parameterizations: (A) equal weights (0.333/0.333/0.333), (B) fitness-only (1.0/0.0/0.0), and (C) our standard choice (0.50/0.30/0.20). Despite substantial differences in credit signal composition, all three variants converged to similar final operator probabilities [RESULTS]. This indicates that slerp de-emphasis is a stable adaptation pattern, not an artefact of the specific weight choice. Consequently, the thesis's weight parameterization is robust to reasonable alternatives, and the result does not depend critically on weight tuning."

**Alternative (if result is mixed):**
> "Weight ablation revealed [SPECIFICS]. Equal-weight and fitness-only variants [DIVERGE/CONVERGE] relative to the baseline, suggesting [SENSITIVITY/ROBUSTNESS]. We therefore [ADOPT/JUSTIFY] the 0.50/0.30/0.20 weights as [ROBUST/NECESSARY] for stable AOS adaptation."

---

## Expected Outcomes

### Scenario 1: Robust AOS (high confidence result)
- **Equal weights variant:** Final p_slerp ≈ baseline p_slerp (within 5–10 percentage points)
- **Fitness-only variant:** Final p_slerp ≤ baseline p_slerp (faster or equal de-emphasis)
- **Interpretation:** AOS genuinely discovers slerp underperformance; weight choice is secondary.
- **Thesis implication:** Strengthen claim that AOS is a principled, stable mechanism.

### Scenario 2: Moderate sensitivity
- **Equal weights variant:** p_slerp intermediate (noticeably higher than baseline, but still de-emphasised)
- **Fitness-only variant:** p_slerp lower than baseline (more aggressive de-emphasis)
- **Interpretation:** Adaptation direction is consistent, but magnitude depends on weights.
- **Thesis implication:** Discuss trade-offs; justify the 0.50/0.30/0.20 choice as balancing stability and responsiveness.

### Scenario 3: Fragile AOS (rare, but actionable)
- **Equal weights or fitness-only:** Final p_slerp ≈ initial p_slerp (no de-emphasis)
- **Interpretation:** Baseline result is specific to the chosen weights; AOS may not be discovering slerp underperformance universally.
- **Thesis implication:** Revise AOS design; consider alternative credit signals or weight tuning based on validation set.

### Most Likely Outcome
**Scenario 1 or 2 is expected.** At 13 generations, slerp underperformance is likely "real" (backed by fitness data), so AOS should adapt consistently across reasonable weight ranges. A divergent result would be surprising and worth investigating.

---

## Viva Preparation Notes

### If all three variants converge (Scenario 1)
**Examiner:** "Your AOS weights are quite arbitrary. How sensitive are your results to those choices?"

**Response:** "We tested three weight parameterizations—equal weights, fitness-only, and our baseline. Despite substantial differences in how credit signals are combined, all three converged to nearly identical operator probabilities by generation 13. This suggests AOS is discovering a robust, stable adaptation pattern: slerp genuinely underperforms on this problem, regardless of weight tuning. The result is not an artefact of the specific 0.50/0.30/0.20 choice."

### If results show moderate sensitivity (Scenario 2)
**Examiner:** "The adaptation magnitude varies. Why should we trust the baseline weights?"

**Response:** "The adaptation *direction* is consistent across variants—slerp is consistently de-emphasised—but the *magnitude* varies with credit weighting. We chose 0.50/0.30/0.20 to balance three competing signals: immediate child performance (fitness), long-term operator track record (parent improvement), and generational persistence (survival). This weighting reflects a principled design choice, not arbitrary tuning. Alternative weights yield qualitatively similar results, confirming the underlying mechanism is sound."

### If results diverge sharply (Scenario 3 — unlikely)
**Examiner:** "Fascinating! It seems AOS only works with your specific weights. What does that tell you?"

**Response:** "This would be a valuable finding: it suggests the problem landscape is such that credit signal composition matters. We would investigate [A] whether the baseline weights better exploit problem structure, [B] whether alternative problems require different weights, or [C] whether a more sophisticated credit mechanism (e.g., adaptive weighting) is necessary. For this thesis, we would document the sensitivity and either justify the weights empirically or propose a more robust framework."

---

## Checklist

- [ ] All three configs created and verified
- [ ] `run_aos_ablation.sh` script created and made executable
- [ ] Expected output directories documented
- [ ] Primary metric (final operator probabilities) clearly defined
- [ ] Thesis update placeholder text prepared
- [ ] Viva talking points drafted
- [ ] Ablation run scheduled and completed
- [ ] Results extracted and compared
- [ ] Chapter 3 updated with ablation findings
- [ ] Final thesis compiled and proofread
