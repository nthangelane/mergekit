import logging

from mergekit.evo.config import TaskConfiguration
from mergekit.evo.helpers import _eval_model, _suppress_expected_eval_noise


def test_eval_model_negates_lower_is_better_metrics(monkeypatch):
    def fake_simple_evaluate(*args, **kwargs):
        return {
            "results": {
                "wikitext": {
                    "word_perplexity,none": 12.5,
                }
            },
            "higher_is_better": {
                "wikitext": {
                    "word_perplexity,none": False,
                }
            },
        }

    monkeypatch.setattr(
        "mergekit.evo.helpers.lm_eval.simple_evaluate", fake_simple_evaluate
    )

    result = _eval_model(
        "huggingface",
        [TaskConfiguration(name="wikitext", weight=1.0, metric="ppl,none")],
        model_args={"pretrained": "dummy"},
    )

    assert result["score"] == -12.5


def test_eval_model_preserves_higher_is_better_metrics(monkeypatch):
    def fake_simple_evaluate(*args, **kwargs):
        return {
            "results": {
                "sciq": {
                    "acc,none": 0.75,
                }
            },
            "higher_is_better": {
                "sciq": {
                    "acc,none": True,
                }
            },
        }

    monkeypatch.setattr(
        "mergekit.evo.helpers.lm_eval.simple_evaluate", fake_simple_evaluate
    )

    result = _eval_model(
        "huggingface",
        [TaskConfiguration(name="sciq", weight=1.0, metric="acc,none")],
        model_args={"pretrained": "dummy"},
    )

    assert result["score"] == 0.75


def test_eval_model_falls_back_to_metric_name_when_direction_is_missing(monkeypatch):
    def fake_simple_evaluate(*args, **kwargs):
        return {
            "results": {
                "wikitext": {
                    "perplexity,none": 8.0,
                }
            },
            "higher_is_better": {"wikitext": {}},
        }

    monkeypatch.setattr(
        "mergekit.evo.helpers.lm_eval.simple_evaluate", fake_simple_evaluate
    )

    result = _eval_model(
        "huggingface",
        [TaskConfiguration(name="wikitext", weight=0.5, metric="ppl,none")],
        model_args={"pretrained": "dummy"},
    )

    assert result["score"] == -4.0


def test_expected_eval_noise_filter_suppresses_datasets_trust_remote_code(caplog):
    logger = logging.getLogger("datasets.load")
    message = (
        "`trust_remote_code` is not supported anymore.\n"
        "Please check that the Hugging Face dataset 'wikitext' isn't based on a loading script."
    )

    with caplog.at_level(logging.ERROR):
        with _suppress_expected_eval_noise():
            logger.error(message)

    assert message not in caplog.text


def test_expected_eval_noise_filter_suppresses_local_model_sha_warning(caplog):
    logger = logging.getLogger("lm-eval")
    message = (
        "Failed to get model SHA for /tmp/fake-merged at revision main. "
        "Error: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/tmp/fake-merged'."
    )

    with caplog.at_level(logging.WARNING):
        with _suppress_expected_eval_noise():
            logger.warning(message)

    assert message not in caplog.text


def test_expected_eval_noise_filter_keeps_nonlocal_model_sha_warning(caplog):
    logger = logging.getLogger("lm-eval")
    message = (
        "Failed to get model SHA for author/model at revision main. "
        "Error: network down"
    )

    with caplog.at_level(logging.WARNING):
        with _suppress_expected_eval_noise():
            logger.warning(message)

    assert message in caplog.text
