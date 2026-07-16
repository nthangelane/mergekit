import pytest

from mergekit.evo.contamination import (
    assert_disjoint_from_evaluation,
    dataset_split_identity,
    find_evaluation_overlaps,
    repair_corpus_identity,
)


def test_wikitext_train_repair_is_disjoint_from_test_evaluation():
    source = repair_corpus_identity("wikitext-train-slice")

    assert source is not None
    assert find_evaluation_overlaps(source, ["wikitext"]) == []


def test_matching_boolq_validation_split_is_rejected():
    source = dataset_split_identity("super_glue", "validation", subset="boolq")

    with pytest.raises(ValueError, match="Training dataset contamination"):
        assert_disjoint_from_evaluation(
            source,
            ["boolq", "piqa"],
            source_label="Training dataset",
        )


def test_same_dataset_different_split_is_allowed():
    source = dataset_split_identity("sciq", "train")

    assert_disjoint_from_evaluation(
        source,
        ["sciq"],
        source_label="Training dataset",
    )


def test_dataset_aliases_preserve_subset_identity():
    source = dataset_split_identity("super_glue/boolq", "validation")

    overlaps = find_evaluation_overlaps(source, ["boolq"])

    assert overlaps[0][0] == "boolq"
