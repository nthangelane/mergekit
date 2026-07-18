import json

from mergekit.evo.reporting import write_ga_outputs


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
