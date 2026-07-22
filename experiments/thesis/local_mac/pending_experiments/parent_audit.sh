#!/bin/bash
# Audit the four parent models at the SAME fidelity as the Campaign-2 audits
# (500 examples/task, WikiText full 62 docs, v2 composite), so child-vs-parent
# verdicts are like-for-like. ~10-15 min total on the M1.
# Output: Research Project/experiments/m1_campaign2/parent_audits.csv
set -u
OUT="$HOME/Documents/master_research/Master Research Paper/Research Project/experiments/m1_campaign2"
mkdir -p "$OUT"
PYBIN="$HOME/.venvs/mergekit-exp/bin/python"
caffeinate -i nice -n 10 "$PYBIN" - <<'EOF'
import csv, math, os
from lm_eval import simple_evaluate

MODELS = [
    ("EleutherAI/pythia-70m-deduped", "deduped"),
    ("lomahony/pythia-70m-helpful-sft", "sft"),
    ("mlabonne/chesspythia-70m", "chess"),
    ("vineetsharma/databricks-dolly-15k-pythia-70m-deduped-v1", "dolly"),
]
OUT = os.path.expanduser(
    "~/Documents/master_research/Master Research Paper/Research Project/experiments/m1_campaign2/parent_audits.csv"
)
rows = []
for repo, name in MODELS:
    print(f"=== auditing {name} ({repo}) at limit 500 ===", flush=True)
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={repo},trust_remote_code=True",
        tasks=["wikitext", "boolq", "sciq"],
        limit=500,
        num_fewshot=0,
        batch_size=1,
        device="cpu",
    )["results"]
    bp = float(res["wikitext"]["byte_perplexity,none"])
    boolq = float(res["boolq"]["acc,none"])
    sciq = float(res["sciq"]["acc,none"])
    comp = 0.40 * (1.0 / (1.0 + math.log1p(max(0.0, bp)))) + 0.35 * boolq + 0.25 * sciq
    print(f"{name}: byte_ppl={bp:.4f} boolq={boolq:.4f} sciq={sciq:.4f} composite={comp:.4f}", flush=True)
    rows.append({"model": name, "repo": repo, "byte_perplexity": f"{bp:.6f}",
                 "boolq_acc": f"{boolq:.6f}", "sciq_acc": f"{sciq:.6f}",
                 "composite_v2": f"{comp:.6f}", "limit": 500})
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWritten: {OUT}")
EOF
