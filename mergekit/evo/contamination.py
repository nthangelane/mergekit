from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class DatasetSplitIdentity:
    dataset: str
    split: str
    subset: Optional[str] = None

    def label(self) -> str:
        subset = f"/{self.subset}" if self.subset else ""
        return f"{self.dataset}{subset}:{self.split}"


_DATASET_ALIASES = {
    "salesforce/wikitext": "wikitext",
    "eleutherai/wikitext_document_level": "wikitext",
    "wikitext": "wikitext",
    "super_glue": "super_glue",
    "boolq": "super_glue",
    "allenai/sciq": "sciq",
    "sciq": "sciq",
    "ybisk/piqa": "piqa",
    "piqa": "piqa",
    "eleutherai/lambada_openai": "lambada_openai",
    "lambada_openai": "lambada_openai",
    "lambada": "lambada",
}


_TASK_IDENTITIES = {
    "wikitext": DatasetSplitIdentity("wikitext", "test", subset="wikitext-2-raw-v1"),
    "boolq": DatasetSplitIdentity("super_glue", "validation", subset="boolq"),
    "sciq": DatasetSplitIdentity("sciq", "test"),
    "lambada_openai": DatasetSplitIdentity("lambada_openai", "test"),
    "lambada_standard": DatasetSplitIdentity("lambada", "test"),
    "piqa": DatasetSplitIdentity("piqa", "validation"),
}


def _normalized(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    return normalized or None


def _normalized_split(split: str) -> str:
    return str(split).strip().lower().split("[", 1)[0]


def dataset_split_identity(
    dataset: str,
    split: str,
    *,
    subset: Optional[str] = None,
) -> DatasetSplitIdentity:
    raw_dataset = str(dataset).strip().removeprefix("https://huggingface.co/")
    normalized_dataset = raw_dataset.lower().strip("/")
    normalized_subset = _normalized(subset)

    if normalized_dataset in {"super-glue/boolq", "super_glue/boolq"}:
        normalized_dataset = "super_glue"
        normalized_subset = "boolq"
    canonical_dataset = _DATASET_ALIASES.get(
        normalized_dataset.replace("-", "_"), normalized_dataset
    )
    canonical_subset = normalized_subset
    if canonical_dataset == "super_glue" and normalized_dataset == "boolq":
        canonical_subset = "boolq"
    if canonical_dataset == "wikitext" and canonical_subset is None:
        canonical_subset = "wikitext-2-raw-v1"

    return DatasetSplitIdentity(
        dataset=canonical_dataset,
        subset=canonical_subset,
        split=_normalized_split(split),
    )


def repair_corpus_identity(corpus: str) -> Optional[DatasetSplitIdentity]:
    normalized = str(corpus).strip().lower().replace("_", "-")
    if normalized == "wikitext-train-slice":
        return dataset_split_identity(
            "Salesforce/wikitext",
            "train",
            subset="wikitext-2-raw-v1",
        )
    return None


def evaluation_task_identity(task_name: str) -> Optional[DatasetSplitIdentity]:
    normalized = str(task_name).strip().lower().replace("-", "_")
    if normalized in _TASK_IDENTITIES:
        return _TASK_IDENTITIES[normalized]
    if normalized.startswith("wikitext_"):
        split = normalized.rsplit("_", 1)[-1]
        if split in {"train", "validation", "test"}:
            return DatasetSplitIdentity("wikitext", split, subset="wikitext-2-raw-v1")
    return None


def identities_overlap(
    source: DatasetSplitIdentity, evaluation: DatasetSplitIdentity
) -> bool:
    if source.dataset != evaluation.dataset or source.split != evaluation.split:
        return False
    return (
        source.subset is None
        or evaluation.subset is None
        or source.subset == evaluation.subset
    )


def find_evaluation_overlaps(
    source: DatasetSplitIdentity,
    evaluation_tasks: Iterable[object],
) -> Sequence[tuple[str, DatasetSplitIdentity]]:
    overlaps = []
    for task in evaluation_tasks:
        task_name = str(getattr(task, "name", task))
        identity = evaluation_task_identity(task_name)
        if identity is not None and identities_overlap(source, identity):
            overlaps.append((task_name, identity))
    return overlaps


def assert_disjoint_from_evaluation(
    source: DatasetSplitIdentity,
    evaluation_tasks: Iterable[object],
    *,
    source_label: str,
) -> None:
    overlaps = find_evaluation_overlaps(source, evaluation_tasks)
    if not overlaps:
        return
    task_names = ", ".join(sorted({task_name for task_name, _ in overlaps}))
    raise ValueError(
        f"{source_label} contamination: {source.label()!r} overlaps evaluation "
        f"task split(s): {task_names}."
    )
