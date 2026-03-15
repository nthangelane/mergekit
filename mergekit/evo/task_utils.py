"""Utility helpers for working with lm_eval tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union

import lm_eval.tasks
import yaml

LOGGER = logging.getLogger("mergekit.tasks")


class _LambadaFewshotWarningFilter(logging.Filter):
    def filter(
        self, record: logging.LogRecord
    ) -> bool:  # pragma: no cover - logging filter
        message = record.getMessage()
        if "[Task: lambada_openai]" in message and "fewshot_docs" in message:
            return False
        return True


def _iter_paths(paths: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _apply_task_overrides(
    task_manager: lm_eval.tasks.TaskManager, override_root: Path
) -> None:
    if not override_root.is_dir():
        return

    for override_file in override_root.rglob("*.yaml"):
        try:
            override_data = yaml.safe_load(override_file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOGGER.warning("Failed to read task override %s: %s", override_file, exc)
            continue

        task_name = override_data.get("task")
        if not task_name:
            continue

        LOGGER.debug("Applying task override for %s from %s", task_name, override_file)
        task_manager._task_index[task_name] = {
            "type": "task",
            "yaml_path": str(override_file),
        }


def create_task_manager(
    task_search_path: Optional[Union[str, Iterable[str]]] = None,
) -> lm_eval.tasks.TaskManager:
    """Construct a TaskManager that respects local override YAML files."""
    include_paths = [
        str(Path(path).resolve()) for path in _iter_paths(task_search_path) if path
    ]

    task_manager = lm_eval.tasks.TaskManager(include_path=include_paths or None)

    override_root = Path(__file__).resolve().parent.parent / "lm_eval_overrides"
    _apply_task_overrides(task_manager, override_root)

    task_logger = logging.getLogger("lm_eval.tasks.task")
    if not any(
        isinstance(f, _LambadaFewshotWarningFilter) for f in task_logger.filters
    ):
        task_logger.addFilter(_LambadaFewshotWarningFilter())
    return task_manager
