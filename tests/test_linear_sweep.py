from mergekit.scripts.linear_sweep import (
    build_linear_merge_config,
    generate_alphas,
    select_top_candidates,
)


def test_generate_alphas_from_step_includes_endpoints():
    assert generate_alphas(alpha_step=0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_generate_alphas_from_explicit_values_sorts_and_deduplicates():
    assert generate_alphas(explicit_alphas=[1.0, 0.5, 0.5, 0.0]) == [0.0, 0.5, 1.0]


def test_select_top_candidates_filters_errors_and_keeps_score_order():
    rows = [
        {"candidate_index": 0, "alpha": 0.0, "weighted_score": -0.4, "error": None},
        {"candidate_index": 1, "alpha": 0.5, "weighted_score": -0.1, "error": None},
        {"candidate_index": 2, "alpha": 1.0, "weighted_score": None, "error": "boom"},
        {"candidate_index": 3, "alpha": 0.8, "weighted_score": -0.2, "error": None},
    ]
    top = select_top_candidates(rows, top_k=2)
    assert [row["alpha"] for row in top] == [0.5, 0.8]


def test_build_linear_merge_config_uses_complementary_weights():
    config = build_linear_merge_config(
        model_a="model-a",
        model_b="model-b",
        alpha=0.7,
        tokenizer_source="base",
    )
    dumped = config.model_dump(mode="json")
    assert dumped["merge_method"] == "linear"
    assert dumped["tokenizer_source"] == "base"
    assert dumped["models"][0]["parameters"]["weight"] == 0.7
    assert dumped["models"][1]["parameters"]["weight"] == 0.3
