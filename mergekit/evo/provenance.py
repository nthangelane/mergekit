import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from huggingface_hub import HfApi, hf_hub_download

from mergekit.common import ModelReference

LOGGER = logging.getLogger(__name__)
PARENT_LINEAGE_FILENAME = "parent_lineage.json"

_HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_LINK_BEFORE_FINETUNE_RE = re.compile(
    r"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"[^.\n]{0,160}\b(?:supervised\s+)?fine[- ]?tun(?:ed|ing)\b",
    re.IGNORECASE,
)
_BASE_CONTEXT_RE = re.compile(
    r"(?:base(?:d)?\s+(?:model\s+)?(?:on|from)|derived\s+from|"
    r"fine[- ]?tun(?:ed|ing)\s+(?:on|from))[^.\n]{0,200}?"
    r"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)


def _normalize_hub_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("name") or value.get("model") or value.get("repo_id")
    if not isinstance(value, str):
        return None
    candidate = value.strip().removeprefix("https://huggingface.co/").strip("/")
    candidate = candidate.split("/tree/", 1)[0]
    return candidate if _HUB_ID_RE.fullmatch(candidate) else None


def _structured_base_models(card_data: Any, config: Any) -> List[str]:
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    card_data = card_data if isinstance(card_data, dict) else {}
    config = config if isinstance(config, dict) else {}

    values: List[Any] = []
    base_model = card_data.get("base_model")
    if isinstance(base_model, list):
        values.extend(base_model)
    elif base_model is not None:
        values.append(base_model)
    if config.get("_name_or_path"):
        values.append(config["_name_or_path"])

    result: List[str] = []
    for value in values:
        normalized = _normalize_hub_id(value)
        if normalized and normalized.lower() not in {item.lower() for item in result}:
            result.append(normalized)
    return result


def _text_base_models(card_text: str) -> List[str]:
    result: List[str] = []
    for pattern in (_LINK_BEFORE_FINETUNE_RE, _BASE_CONTEXT_RE):
        for match in pattern.finditer(card_text or ""):
            normalized = _normalize_hub_id(match.group(1))
            if normalized and normalized.lower() not in {
                item.lower() for item in result
            }:
                result.append(normalized)
    return result


def _default_card_loader(repo_id: str, revision: Optional[str]) -> str:
    path = hf_hub_download(repo_id, "README.md", revision=revision)
    return Path(path).read_text(encoding="utf-8")


def _inspect_parent(
    model: ModelReference,
    *,
    api: HfApi,
    card_loader: Callable[[str, Optional[str]], str],
) -> Dict[str, Any]:
    repo_id = str(model.model.path)
    revision = model.model.revision
    record: Dict[str, Any] = {
        "model": repo_id,
        "revision": revision,
        "resolved_sha": None,
        "declared_base_models": [],
        "lineage_candidates": [repo_id],
        "evidence": [],
        "lookup_error": None,
    }

    if os.path.exists(repo_id) or not _HUB_ID_RE.fullmatch(repo_id):
        record["lookup_error"] = "local_or_non_hub_model"
        return record

    try:
        info = api.model_info(
            repo_id,
            revision=revision,
            files_metadata=False,
            timeout=10,
        )
        record["resolved_sha"] = info.sha
        structured = _structured_base_models(info.card_data, info.config)
        if structured:
            record["evidence"].append("structured_metadata")

        card_text = ""
        try:
            card_text = card_loader(repo_id, revision)
        except Exception as exc:
            LOGGER.debug("Unable to load model card text for %s: %s", repo_id, exc)
        textual = _text_base_models(card_text)
        if textual:
            record["evidence"].append("model_card_text")

        declared: List[str] = []
        for candidate in [*structured, *textual]:
            if candidate.lower() != repo_id.lower() and candidate.lower() not in {
                item.lower() for item in declared
            }:
                declared.append(candidate)
        record["declared_base_models"] = declared
        record["lineage_candidates"] = [repo_id, *declared]
    except Exception as exc:
        record["lookup_error"] = f"{type(exc).__name__}: {exc}"
    return record


def inspect_parent_lineage(
    models: Iterable[ModelReference],
    *,
    api: Optional[HfApi] = None,
    card_loader: Optional[Callable[[str, Optional[str]], str]] = None,
) -> Dict[str, Any]:
    """Inspect source-model lineage without making missing metadata fatal."""
    unique_models: List[ModelReference] = []
    seen = set()
    for model in models:
        key = (str(model.model.path).lower(), model.model.revision)
        if key not in seen:
            seen.add(key)
            unique_models.append(model)

    api = api or HfApi()
    loader = card_loader or _default_card_loader
    parents = [
        _inspect_parent(model, api=api, card_loader=loader) for model in unique_models
    ]

    lineage_sets = [
        {str(value).lower() for value in parent["lineage_candidates"]}
        for parent in parents
    ]
    common = set.intersection(*lineage_sets) if lineage_sets else set()
    canonical = {
        str(value).lower(): str(value)
        for parent in parents
        for value in parent["lineage_candidates"]
    }
    common_lineage = [canonical[value] for value in sorted(common)]
    lookup_failures = [parent["model"] for parent in parents if parent["lookup_error"]]

    if len(parents) < 2:
        status = "single_parent"
        warning = None
    elif common_lineage:
        status = "compatible"
        warning = None
    elif lookup_failures:
        status = "unknown"
        warning = (
            "Parent lineage could not be fully verified because metadata lookup failed "
            f"for: {', '.join(lookup_failures)}."
        )
    else:
        status = "no_common_lineage"
        warning = (
            "No shared base checkpoint was found in parent identities or available "
            "Hugging Face metadata. Merge viability may be reduced by disconnected "
            "training lineages."
        )

    return {
        "schema_version": 1,
        "status": status,
        "common_lineage": common_lineage,
        "warning": warning,
        "parents": parents,
    }


def write_parent_lineage_report(storage_path: str, report: Dict[str, Any]) -> str:
    output_path = os.path.join(storage_path, PARENT_LINEAGE_FILENAME)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
