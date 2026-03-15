from pathlib import Path

from mergekit.evo.task_utils import create_task_manager


def test_lambada_override_applied():
    manager = create_task_manager()
    entry = manager._task_index.get("lambada_openai")
    assert entry is not None, "lambada_openai task missing from TaskManager"

    yaml_path = Path(entry["yaml_path"]).resolve()
    assert "lm_eval_overrides/lambada/lambada_openai.yaml" in str(yaml_path), (
        "Expected lambada_openai to use local override YAML, got " f"{yaml_path}"
    )
