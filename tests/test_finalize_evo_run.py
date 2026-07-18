import json

import pytest
import yaml

import mergekit.scripts.finalize_evo_run as finalizer
from mergekit.evo.progress import ProgressLogger, iter_progress_events


def _write_completed_search(tmp_path, *, candidate_rows=2):
    config_path = tmp_path / "evolve.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "genome": {
                    "models": ["author/model-a", "author/model-b"],
                    "merge_method": "linear",
                },
                "tasks": ["wikitext"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "best_config.yaml").write_text(
        yaml.safe_dump(
            {
                "merge_method": "linear",
                "models": [
                    {"model": "author/model-a", "parameters": {"weight": 0.5}},
                    {"model": "author/model-b", "parameters": {"weight": 0.5}},
                ],
            }
        ),
        encoding="utf-8",
    )
    csv_payload = "generation,fevals,value,score\n1,1,0,0.4\n1,2,1,0.5\n"
    (tmp_path / "ga_candidate_history.csv").write_text(
        "generation,fevals,value,score\n"
        + "".join(
            f"1,{index + 1},{index},{0.4 + index * 0.1}\n"
            for index in range(candidate_rows)
        ),
        encoding="utf-8",
    )
    for filename in ("baseline_results.csv", "ga_method_history.csv"):
        (tmp_path / filename).write_text(csv_payload, encoding="utf-8")
    (tmp_path / "ga_history.csv").write_text(
        "generation,fevals,gen_best,gen_mean,best_so_far,eval_seconds,cache_hits,crossover_type\n"
        "1,2,0.5,0.45,None,10.0,0,none\n",
        encoding="utf-8",
    )
    (tmp_path / "ga_stop_details.json").write_text(
        json.dumps(
            {
                "final_stop": {
                    "reason": "random_search_complete",
                    "fevals": 2,
                    "generation": 1,
                    "best_score": 0.5,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progress = ProgressLogger(str(tmp_path), seed=11)
    progress.write("run_started", mode="random_search", max_fevals=2)
    progress.write("generation_completed", generation=1, idx=2)
    progress.write(
        "search_completed",
        generation=1,
        idx=2,
        scores={"best": 0.5},
        stop_reason="random_search_complete",
    )
    return config_path


def test_finalize_evo_run_recovers_completed_random_search(monkeypatch, tmp_path):
    config_path = _write_completed_search(tmp_path)

    monkeypatch.setattr(finalizer, "ensure_free_disk", lambda *_args: 10.0)

    def fake_run_merge(_config, output_path, **_kwargs):
        output_path = finalizer.Path(output_path)
        output_path.mkdir(parents=True)
        (output_path / "model.safetensors").write_bytes(b"model")

    def fake_final_comparison(_config, storage_path, *_args):
        (finalizer.Path(storage_path) / "ga_final_comparison.csv").write_text(
            "model,weighted_score\nfinal_merged,0.5\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(finalizer, "run_merge", fake_run_merge)
    monkeypatch.setattr(
        finalizer,
        "evaluate_and_write_final_comparison",
        fake_final_comparison,
    )

    result = finalizer.finalize_evo_run(config_path, tmp_path, device="cpu")

    assert result.fevals == 2
    assert result.candidate_rows == 2
    terminal = list(iter_progress_events(str(tmp_path)))[-1]
    assert terminal["msg"] == "run_finished"
    assert terminal["status"] == "success"
    assert terminal["recovered_after_search"] is True


def test_finalize_evo_run_rejects_incomplete_candidate_budget(tmp_path):
    config_path = _write_completed_search(tmp_path, candidate_rows=1)

    with pytest.raises(finalizer.EvoRunFinalizationError, match="budget is incomplete"):
        finalizer.finalize_evo_run(config_path, tmp_path, device="cpu")
