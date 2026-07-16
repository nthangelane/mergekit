import pytest

from mergekit.evo.resources import (
    FREE_DISK_OVERRIDE_ENV,
    InsufficientDiskSpaceError,
    ensure_free_disk,
)


def test_low_disk_override_raises_clean_error(monkeypatch, tmp_path):
    monkeypatch.setenv(FREE_DISK_OVERRIDE_ENV, "0.25")

    with pytest.raises(InsufficientDiskSpaceError) as exc_info:
        ensure_free_disk(str(tmp_path), 5.0)

    assert exc_info.value.free_gb == 0.25
    assert exc_info.value.required_gb == 5.0
    assert "Insufficient free disk space" in str(exc_info.value)


def test_disk_guard_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv(FREE_DISK_OVERRIDE_ENV, "0")
    assert ensure_free_disk(str(tmp_path), 0) == 0.0
