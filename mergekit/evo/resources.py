import os
import shutil
from pathlib import Path
from typing import Optional

FREE_DISK_OVERRIDE_ENV = "MERGEKIT_FREE_DISK_GB_OVERRIDE"


class InsufficientDiskSpaceError(RuntimeError):
    def __init__(self, path: str, free_gb: float, required_gb: float):
        self.path = path
        self.free_gb = float(free_gb)
        self.required_gb = float(required_gb)
        super().__init__(
            f"Insufficient free disk space at {path}: {free_gb:.3f} GiB available, "
            f"{required_gb:.3f} GiB required."
        )


def free_disk_gb(path: str) -> float:
    override = os.getenv(FREE_DISK_OVERRIDE_ENV)
    if override is not None:
        try:
            return float(override)
        except ValueError as exc:
            raise ValueError(
                f"{FREE_DISK_OVERRIDE_ENV} must contain a numeric GiB value"
            ) from exc

    probe = Path(path).expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return float(shutil.disk_usage(probe).free) / float(1024**3)


def ensure_free_disk(path: str, minimum_gb: Optional[float]) -> float:
    if minimum_gb is None or float(minimum_gb) <= 0:
        return free_disk_gb(path)
    available = free_disk_gb(path)
    if available < float(minimum_gb):
        raise InsufficientDiskSpaceError(path, available, float(minimum_gb))
    return available
