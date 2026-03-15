import os

from mergekit.merge import _link_or_copy_file


def test_link_or_copy_file_prefers_hardlinks_on_same_filesystem(tmp_path):
    src = tmp_path / "tokenizer.json"
    dst = tmp_path / "out" / "tokenizer.json"
    dst.parent.mkdir()
    src.write_text('{"tokenizer": true}', encoding="utf-8")

    _link_or_copy_file(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert os.stat(src).st_ino == os.stat(dst).st_ino
