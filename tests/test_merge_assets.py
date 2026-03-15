import os
from types import SimpleNamespace

from mergekit.merge import (
    _ensure_tokenizer_assets,
    _link_or_copy_file,
    _materialize_symlinked_output_files,
)


def test_link_or_copy_file_prefers_hardlinks_on_same_filesystem(tmp_path):
    src = tmp_path / "tokenizer.json"
    dst = tmp_path / "out" / "tokenizer.json"
    dst.parent.mkdir()
    src.write_text('{"tokenizer": true}', encoding="utf-8")

    _link_or_copy_file(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert os.stat(src).st_ino == os.stat(dst).st_ino


def test_link_or_copy_file_dereferences_symlink_sources(tmp_path):
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    real_src = blobs_dir / "tokenizer.json"
    real_src.write_text('{"tokenizer": "blob"}', encoding="utf-8")

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    symlink_src = snapshot_dir / "tokenizer.json"
    symlink_src.symlink_to(real_src)

    dst = tmp_path / "out" / "tokenizer.json"
    dst.parent.mkdir()

    _link_or_copy_file(str(symlink_src), str(dst))

    assert dst.exists()
    assert not dst.is_symlink()
    assert dst.read_text(encoding="utf-8") == real_src.read_text(encoding="utf-8")
    assert os.stat(real_src).st_ino == os.stat(dst).st_ino


def test_materialize_symlinked_output_files_replaces_symlink_with_real_file(tmp_path):
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    real_src = blobs_dir / "tokenizer.json"
    real_src.write_text('{"tokenizer": "blob"}', encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    symlink_dst = out_dir / "tokenizer.json"
    symlink_dst.symlink_to(real_src)

    _materialize_symlinked_output_files(str(out_dir), ["tokenizer.json"])

    assert symlink_dst.exists()
    assert not symlink_dst.is_symlink()
    assert symlink_dst.read_text(encoding="utf-8") == real_src.read_text(
        encoding="utf-8"
    )
    assert os.stat(real_src).st_ino == os.stat(symlink_dst).st_ino


def test_link_or_copy_file_replaces_existing_broken_destination_symlink(tmp_path):
    src = tmp_path / "tokenizer.json"
    src.write_text('{"tokenizer": "real"}', encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    broken_dst = out_dir / "tokenizer.json"
    broken_dst.symlink_to(tmp_path / "missing-blobs" / "tokenizer.json")

    _link_or_copy_file(str(src), str(broken_dst))

    assert broken_dst.exists()
    assert not broken_dst.is_symlink()
    assert broken_dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert os.stat(src).st_ino == os.stat(broken_dst).st_ino


def test_ensure_tokenizer_assets_repairs_broken_links(monkeypatch, tmp_path):
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]:
        (donor_dir / name).write_text(f"{name}-real", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]:
        (out_dir / name).symlink_to(tmp_path / "missing" / name)

    def fake_copy_tokenizer(_merge_config, out_path, options):
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ]:
            _link_or_copy_file(str(donor_dir / name), str((out_dir / name)))

    monkeypatch.setattr("mergekit.merge._copy_tokenizer", fake_copy_tokenizer)

    _ensure_tokenizer_assets(
        merge_config=object(),
        out_path=str(out_dir),
        options=SimpleNamespace(copy_tokenizer=True),
        file_names=[
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]:
        path = out_dir / name
        assert path.exists()
        assert not path.is_symlink()
        assert path.read_text(encoding="utf-8") == f"{name}-real"
