import pytest

import mergekit.common as common
from mergekit.common import ModelPath, ModelReference


class TestModelReference:
    def test_parse_simple(self):
        text = "hf_user/model"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision=None)
        assert mr.lora is None
        assert str(mr) == text

    def test_parse_lora(self):
        text = "hf_user/model+hf_user/lora"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision=None)
        assert mr.lora == ModelPath(path="hf_user/lora", revision=None)
        assert str(mr) == text

    def test_parse_revision(self):
        text = "hf_user/model@v0.0.1"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision="v0.0.1")
        assert mr.lora is None
        assert str(mr) == text

    def test_parse_lora_plus_revision(self):
        text = "hf_user/model@v0.0.1+hf_user/lora@main"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision="v0.0.1")
        assert mr.lora == ModelPath(path="hf_user/lora", revision="main")
        assert str(mr) == text

    def test_parse_bad(self):
        with pytest.raises(RuntimeError):
            ModelReference.parse("@@@@@")

        with pytest.raises(RuntimeError):
            ModelReference.parse("a+b+c")

        with pytest.raises(RuntimeError):
            ModelReference.parse("a+b+c@d+e@f@g")

    def test_local_path_caches_snapshot_download(self, monkeypatch, tmp_path):
        common._snapshot_local_path.cache_clear()

        calls = {"list_repo_files": 0, "snapshot_download": 0}
        snapshot_path = tmp_path / "snapshot"
        snapshot_path.mkdir()

        def fake_list_repo_files(path, repo_type, revision):
            calls["list_repo_files"] += 1
            return ["config.json", "tokenizer.json", "model.safetensors"]

        def fake_snapshot_download(path, revision, cache_dir, allow_patterns):
            calls["snapshot_download"] += 1
            assert cache_dir == str((tmp_path / "hf-cache").resolve())
            assert "*.safetensors" in allow_patterns
            return str(snapshot_path)

        monkeypatch.setattr(
            common.huggingface_hub, "list_repo_files", fake_list_repo_files
        )
        monkeypatch.setattr(
            common.huggingface_hub,
            "snapshot_download",
            fake_snapshot_download,
        )

        model_ref = ModelReference.parse("author/model")
        cache_dir = str(tmp_path / "hf-cache")

        assert model_ref.local_path(cache_dir=cache_dir) == str(snapshot_path)
        assert model_ref.local_path(cache_dir=cache_dir) == str(snapshot_path)
        assert calls == {"list_repo_files": 1, "snapshot_download": 1}
