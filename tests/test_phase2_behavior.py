import pytest

from mergekit.evo.config import TaskConfiguration
from mergekit.evo.helpers import evaluate_model_cpu, score_evaluation_payload


def test_structured_phase1_fitness_uses_task_language_and_stability():
    result = score_evaluation_payload(
        {
            "results": {
                "sciq": {"acc,none": 0.5},
                "wikitext": {"byte_perplexity,none": 2.0},
            },
            "higher_is_better": {
                "sciq": {"acc,none": True},
                "wikitext": {"byte_perplexity,none": False},
            },
        },
        [
            TaskConfiguration(name="sciq", weight=0.8, metric="acc,none"),
            TaskConfiguration(
                name="wikitext", weight=0.2, metric="byte_perplexity,none"
            ),
        ],
        fitness_mode="structured_phase1_tiny",
        task_mix_profile="pythia70m_phase1",
        behavior_probe={"stability_score": 0.8, "behavior_diversity_score": 0.6},
    )

    assert result["score"] == pytest.approx(0.4816666667, rel=1e-6)
    assert result["fitness_components"]["task_score"] == pytest.approx(0.5)
    assert result["fitness_components"]["language_quality"] == pytest.approx(1.0 / 3.0)
    assert result["fitness_components"]["stability_score"] == pytest.approx(0.9)
    assert result["fitness_components"]["behavior_diversity_score"] == pytest.approx(
        0.6
    )


def test_evaluate_model_cpu_rejects_degenerate_behavior_probe(monkeypatch, tmp_path):
    def fake_behavior_probe(*args, **kwargs):
        return {
            "rejected": True,
            "reject_reason": "repetition_loop",
            "behavior_diversity_score": 0.0,
            "stability_score": 0.0,
        }

    def fail_eval(*args, **kwargs):
        raise AssertionError("_eval_model should not run after smoke rejection")

    monkeypatch.setattr("mergekit.evo.helpers._run_behavior_probe", fake_behavior_probe)
    monkeypatch.setattr("mergekit.evo.helpers._eval_model", fail_eval)

    result = evaluate_model_cpu(
        str(tmp_path / "merged"),
        [TaskConfiguration(name="sciq", metric="acc,none")],
        num_fewshot=0,
        limit=2,
        behavior_prompts=["Hello"],
        behavior_reject_on_degenerate=True,
    )

    assert result["score"] is None
    assert result["error_stage"] == "smoke_test"
    assert result["error_type"] == "degenerate_output"
    assert result["results"]["behavior_probe"]["rejected"] is True
