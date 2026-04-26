import json

import pandas as pd

from scripts.research.write_thesis_result_summary import generate_summary


def test_generate_summary_distinguishes_ga_objective_from_raw_score(tmp_path):
    (tmp_path / "ga_stop_details.json").write_text(
        json.dumps(
            {
                "final_stop": {
                    "reason": "max_fevals",
                    "generation": 12,
                    "fevals": 96,
                    "best_generation": 10,
                    "best_score": 0.56,
                    "best_score_source": "stage2",
                }
            }
        )
    )
    (tmp_path / "ga_summary.txt").write_text("final_best=0.5600")
    pd.DataFrame(
        [
            {"model": "base", "weighted_score": 0.53},
            {"model": "final_merged", "weighted_score": 0.52},
        ]
    ).to_csv(tmp_path / "ga_final_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "generation": 1,
                "merge_method": "linear",
                "probability_after": 0.6,
                "success_rate": 1.0,
                "best_score": 0.56,
            }
        ]
    ).to_csv(tmp_path / "ga_method_history.csv", index=False)
    pd.DataFrame([{"error_type": "metric_guard"}]).to_csv(
        tmp_path / "failed_genotypes.csv", index=False
    )

    summary = generate_summary(tmp_path, title="Test Summary")

    assert "# Test Summary" in summary
    assert "best GA objective score: `0.5600000000`" in summary
    assert "`final_merged`" in summary
    assert "`metric_guard`" in summary
    assert (
        "GA objective score and the raw final comparison score are not the same"
        in summary
    )
