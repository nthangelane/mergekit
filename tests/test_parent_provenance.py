import json
from types import SimpleNamespace

from mergekit.common import ModelReference
from mergekit.evo.provenance import (
    inspect_parent_lineage,
    write_parent_lineage_report,
)


class _FakeApi:
    def __init__(self, records):
        self.records = records

    def model_info(self, repo_id, revision=None, files_metadata=False, timeout=None):
        assert timeout == 10
        record = self.records[repo_id]
        if isinstance(record, Exception):
            raise record
        return SimpleNamespace(
            sha=record.get("sha", "abc123"),
            card_data=record.get("card_data"),
            config=record.get("config", {}),
        )


def _refs(*repo_ids):
    return [ModelReference.model_validate(repo_id) for repo_id in repo_ids]


def test_parent_lineage_detects_shared_structured_base():
    api = _FakeApi(
        {
            "author/finetune-a": {"card_data": {"base_model": "base/model"}},
            "author/finetune-b": {"card_data": {"base_model": ["base/model"]}},
        }
    )

    report = inspect_parent_lineage(
        _refs("author/finetune-a", "author/finetune-b"),
        api=api,
        card_loader=lambda *_args: "",
    )

    assert report["status"] == "compatible"
    assert report["common_lineage"] == ["base/model"]
    assert report["warning"] is None


def test_parent_lineage_reads_finetune_base_from_model_card_text():
    api = _FakeApi(
        {
            "lomahony/pythia-70m-helpful-sft": {},
            "EleutherAI/pythia-70m-deduped": {},
        }
    )
    cards = {
        "lomahony/pythia-70m-helpful-sft": (
            "[Pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) "
            "supervised finetuned using a helpful dataset."
        ),
        "EleutherAI/pythia-70m-deduped": "Pythia deduplicated checkpoint.",
    }

    report = inspect_parent_lineage(
        _refs(
            "lomahony/pythia-70m-helpful-sft",
            "EleutherAI/pythia-70m-deduped",
        ),
        api=api,
        card_loader=lambda repo_id, _revision: cards[repo_id],
    )

    assert report["status"] == "no_common_lineage"
    assert "No shared base checkpoint" in report["warning"]
    assert report["parents"][0]["declared_base_models"] == ["EleutherAI/pythia-70m"]


def test_parent_lineage_lookup_failure_is_warning_only():
    api = _FakeApi(
        {
            "author/model-a": RuntimeError("offline"),
            "author/model-b": {},
        }
    )

    report = inspect_parent_lineage(
        _refs("author/model-a", "author/model-b"),
        api=api,
        card_loader=lambda *_args: "",
    )

    assert report["status"] == "unknown"
    assert "metadata lookup failed" in report["warning"]


def test_parent_lineage_report_is_persisted(tmp_path):
    report = {
        "schema_version": 1,
        "status": "compatible",
        "common_lineage": ["base/model"],
        "warning": None,
        "parents": [],
    }

    output_path = write_parent_lineage_report(str(tmp_path), report)

    assert json.loads((tmp_path / "parent_lineage.json").read_text()) == report
    assert output_path == str(tmp_path / "parent_lineage.json")
