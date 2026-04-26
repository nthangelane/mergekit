"""Utility helpers for working with lm_eval tasks."""

from __future__ import annotations

import collections
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import lm_eval.tasks
import yaml
from lm_eval import utils as lm_eval_utils

LOGGER = logging.getLogger("mergekit.tasks")


class _LambadaFewshotWarningFilter(logging.Filter):
    def filter(
        self, record: logging.LogRecord
    ) -> bool:  # pragma: no cover - logging filter
        message = record.getMessage()
        if "[Task: lambada_openai]" in message and "fewshot_docs" in message:
            return False
        return True


class _ResilientTaskManager(lm_eval.tasks.TaskManager):
    """TaskManager variant that skips malformed task YAMLs instead of aborting."""

    def __init__(self, *args, required_tasks: Optional[Sequence[str]] = None, **kwargs):
        self.logger = LOGGER
        self.required_tasks = tuple(
            dict.fromkeys(str(task) for task in (required_tasks or []) if task)
        )
        super().__init__(*args, **kwargs)

    def initialize_tasks(
        self,
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ):
        if not self.required_tasks:
            return super().initialize_tasks(
                include_path=include_path,
                include_defaults=include_defaults,
            )

        task_index = self._initialize_required_tasks(
            include_path=include_path,
            include_defaults=include_defaults,
        )
        missing = [task for task in self.required_tasks if task not in task_index]
        if missing:
            self.logger.warning(
                "Required-task indexing could not resolve %s; falling back to full task index.",
                ", ".join(missing),
            )
            return super().initialize_tasks(
                include_path=include_path,
                include_defaults=include_defaults,
            )
        return task_index

    def _initialize_required_tasks(
        self,
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ):
        all_paths: List[str] = []
        if include_defaults:
            all_paths.append(
                os.path.dirname(os.path.abspath(lm_eval.tasks.__file__)) + "/"
            )
        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        task_index = {}
        remaining = set(self.required_tasks)
        for task_dir in all_paths:
            if not remaining:
                break
            tasks = self._get_required_tasks(task_dir, remaining)
            task_index = {**tasks, **task_index}
            remaining = set(self.required_tasks) - set(task_index.keys())

        return task_index

    def _get_required_tasks(self, task_dir: str, required_tasks: set[str]):
        if not required_tasks:
            return {}

        ignore_dirs = [
            "__pycache__",
            ".ipynb_checkpoints",
        ]
        tasks_and_groups = {}
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file_name in file_list:
                if not file_name.endswith(".yaml"):
                    continue
                yaml_path = os.path.join(root, file_name)
                try:
                    config = lm_eval_utils.load_yaml_config(yaml_path, mode="simple")
                    if config is None or not isinstance(config, dict):
                        self.logger.warning(
                            "Skipping invalid task config %s during task indexing.",
                            yaml_path,
                        )
                        continue
                except Exception as exc:
                    self.logger.warning(
                        "Skipping task config %s during task indexing: %s",
                        yaml_path,
                        exc,
                    )
                    continue

                if self._config_is_python_task(config):
                    task_name = config.get("task")
                    if isinstance(task_name, str) and task_name in required_tasks:
                        tasks_and_groups[task_name] = {
                            "type": "python_task",
                            "yaml_path": yaml_path,
                        }
                elif self._config_is_group(config):
                    group_name = config.get("group")
                    if isinstance(group_name, str) and group_name in required_tasks:
                        tasks_and_groups[group_name] = {
                            "type": "group",
                            "task": -1,
                            "yaml_path": yaml_path,
                        }
                elif self._config_is_task(config):
                    task_name = config["task"]
                    if task_name in required_tasks:
                        tasks_and_groups[task_name] = {
                            "type": "task",
                            "yaml_path": yaml_path,
                        }

                if required_tasks.issubset(tasks_and_groups.keys()):
                    return tasks_and_groups

        return tasks_and_groups

    def _get_task_and_group(self, task_dir: str):  # pragma: no cover - thin wrapper
        print_info = True
        ignore_dirs = [
            "__pycache__",
            ".ipynb_checkpoints",
        ]
        tasks_and_groups = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file_name in file_list:
                if not file_name.endswith(".yaml"):
                    continue
                yaml_path = os.path.join(root, file_name)
                try:
                    config = lm_eval_utils.load_yaml_config(yaml_path, mode="simple")
                    if config is None or not isinstance(config, dict):
                        self.logger.warning(
                            "Skipping invalid task config %s during task indexing.",
                            yaml_path,
                        )
                        continue
                except Exception as exc:
                    self.logger.warning(
                        "Skipping task config %s during task indexing: %s",
                        yaml_path,
                        exc,
                    )
                    continue

                if self._config_is_python_task(config):
                    tasks_and_groups[config["task"]] = {
                        "type": "python_task",
                        "yaml_path": yaml_path,
                    }
                elif self._config_is_group(config):
                    tasks_and_groups[config["group"]] = {
                        "type": "group",
                        "task": -1,
                        "yaml_path": yaml_path,
                    }
                elif self._config_is_task(config):
                    task = config["task"]
                    tasks_and_groups[task] = {
                        "type": "task",
                        "yaml_path": yaml_path,
                    }

                    for attr in ["tag", "group"]:
                        if attr not in config:
                            continue
                        if attr == "group" and print_info:
                            self.logger.info(
                                "`group` and `group_alias` keys in tasks' configs will no longer be used in the next release of lm-eval. "
                                "`tag` will be used to allow to call a collection of tasks just like `group`. "
                                "`group` will be removed in order to not cause confusion with the new ConfigurableGroup "
                                "which will be the offical way to create groups with addition of group-wide configuations."
                            )
                            print_info = False

                        attr_list = config[attr]
                        if isinstance(attr_list, str):
                            attr_list = [attr_list]

                        for tag in attr_list:
                            if tag not in tasks_and_groups:
                                tasks_and_groups[tag] = {
                                    "type": "tag",
                                    "task": [task],
                                    "yaml_path": -1,
                                }
                            elif tasks_and_groups[tag]["type"] != "tag":
                                self.logger.info(
                                    f"The tag {tag} is already registered as a group, this tag will not be registered. "
                                    "This may affect tasks you want to call."
                                )
                                break
                            else:
                                tasks_and_groups[tag]["task"].append(task)
                else:
                    self.logger.debug(
                        "File %s in %s could not be loaded as a task config",
                        file_name,
                        root,
                    )
        return tasks_and_groups


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
    *,
    include_defaults: bool = True,
    required_tasks: Optional[Sequence[str]] = None,
) -> lm_eval.tasks.TaskManager:
    """Construct a TaskManager that respects local override YAML files."""
    include_paths = [
        str(Path(path).resolve()) for path in _iter_paths(task_search_path) if path
    ]

    task_manager = _ResilientTaskManager(
        include_path=include_paths or None,
        include_defaults=include_defaults,
        required_tasks=required_tasks,
    )

    override_root = Path(__file__).resolve().parent.parent / "lm_eval_overrides"
    _apply_task_overrides(task_manager, override_root)

    task_logger = logging.getLogger("lm_eval.tasks.task")
    if not any(
        isinstance(f, _LambadaFewshotWarningFilter) for f in task_logger.filters
    ):
        task_logger.addFilter(_LambadaFewshotWarningFilter())
    return task_manager
