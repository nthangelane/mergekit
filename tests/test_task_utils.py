from pathlib import Path

from mergekit.evo.task_utils import create_task_manager


def test_create_task_manager_skips_empty_yaml(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    (tmp_path / "good.yaml").write_text(
        "\n".join(
            [
                "task: toy_boolq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = create_task_manager([str(tmp_path)], include_defaults=False)

    assert "toy_boolq" in manager.all_tasks


def test_create_task_manager_skips_unparseable_yaml(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "task: bad_task\n: not-valid-yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "good.yaml").write_text(
        "\n".join(
            [
                "task: toy_sciq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = create_task_manager([str(tmp_path)], include_defaults=False)

    assert "toy_sciq" in manager.all_tasks
    assert "bad_task" not in manager.all_tasks


def test_create_task_manager_required_tasks_indexes_only_selected(tmp_path):
    (tmp_path / "wanted.yaml").write_text(
        "\n".join(
            [
                "task: toy_boolq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "other.yaml").write_text(
        "\n".join(
            [
                "task: toy_sciq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = create_task_manager(
        [str(tmp_path)],
        include_defaults=False,
        required_tasks=["toy_boolq"],
    )

    assert manager.all_tasks == ["toy_boolq"]
    assert "toy_boolq" in manager.task_index
    assert "toy_sciq" not in manager.task_index


def test_create_task_manager_required_tasks_falls_back_when_missing(tmp_path):
    (tmp_path / "good.yaml").write_text(
        "\n".join(
            [
                "task: toy_boolq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "other.yaml").write_text(
        "\n".join(
            [
                "task: toy_sciq",
                "dataset_path: local/toy",
                "output_type: generate_until",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = create_task_manager(
        [str(tmp_path)],
        include_defaults=False,
        required_tasks=["missing_task"],
    )

    assert "toy_boolq" in manager.all_tasks
    assert "toy_sciq" in manager.all_tasks
