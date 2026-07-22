import json
from types import SimpleNamespace

import pandas

from mergekit.evo.reporting import evaluate_and_write_final_comparison, write_ga_outputs


def test_write_ga_outputs_handles_none_history_values(tmp_path):
    (tmp_path / "ga_history.csv").write_text(
        "generation,fevals,gen_best,gen_mean,best_so_far,eval_seconds,cache_hits,crossover_type\n"
        "1,8,0.4,0.4,None,12.5,None,none\n"
        "2,16,None,None,0.4,None,0,none\n",
        encoding="utf-8",
    )

    write_ga_outputs(
        str(tmp_path),
        stop_details={"reason": "random_search_complete", "fevals": 16},
    )

    summary = (tmp_path / "ga_summary.txt").read_text(encoding="utf-8")
    assert "Gen-best sparkline:" in summary
    assert "reason=random_search_complete" in summary
    assert "NaN" in summary


def test_final_comparison_uses_audit_limit_and_marks_winner_audited(
    monkeypatch, tmp_path
):
    (tmp_path / "final_model").mkdir()
    pandas.DataFrame(
        [{"model": "parent", "weighted_score": 0.3, "sciq:acc,none": 0.3}]
    ).to_csv(tmp_path / "baseline_results.csv", index=False)
    seen = {}

    monkeypatch.setattr(
        "mergekit.evo.reporting._create_task_manager", lambda *args, **kwargs: object()
    )

    def fake_eval(*args, **kwargs):
        seen["limit"] = kwargs["limit"]
        return {"score": 0.5, "results": {"sciq": {"acc,none": 0.5}}}

    monkeypatch.setattr("mergekit.evo.reporting._eval_model", fake_eval)
    monkeypatch.setattr(
        "mergekit.evo.reporting.write_comparison_plot", lambda *args, **kwargs: None
    )
    task = SimpleNamespace(name="sciq", metric="acc,none", weight=1.0)
    config = SimpleNamespace(
        tasks=[task],
        audit=SimpleNamespace(enabled=True, limit=50),
        limit=5,
        num_fewshot=0,
        fitness_mode="weighted_sum",
        fitness=SimpleNamespace(
            version="v2", lower_is_better_transform="log_reciprocal"
        ),
        task_mix_profile=None,
        stage1_tasks=None,
    )

    evaluate_and_write_final_comparison(
        config,
        str(tmp_path),
        batch_size=1,
        merge_cuda=False,
        num_gpus=0,
        task_search_path=[],
        trust_remote_code=False,
    )

    comparison = pandas.read_csv(tmp_path / "ga_final_comparison.csv")
    winner = comparison.loc[comparison["model"] == "final_merged"].iloc[0]
    assert seen["limit"] == 50
    assert bool(winner["audited"]) is True
