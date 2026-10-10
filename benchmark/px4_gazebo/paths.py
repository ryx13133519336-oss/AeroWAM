"""Central storage layout for the PX4-Gazebo benchmark.

Large, replaceable artifacts live outside the source tree by default.  The
same environment variables work from Windows and WSL; Windows drive paths are
translated to ``/mnt/<drive>`` when this module runs under Linux.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_MOUNT = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")


def _portable_path(value: str | Path) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    if os.name != "nt":
        match = _WINDOWS_DRIVE.match(raw)
        if match:
            drive, tail = match.groups()
            return Path("/mnt") / drive.lower() / Path(tail.replace("\\", "/"))
    else:
        match = _WSL_MOUNT.match(raw)
        if match:
            drive, tail = match.groups()
            return Path(f"{drive.upper()}:/") / (tail or "")
    return Path(raw)


def _default_root() -> Path:
    configured = os.getenv("AEROWAM_DATA_ROOT")
    if configured:
        return _portable_path(configured)
    if os.name == "nt":
        return Path("D:/AeroWAM-data")
    if Path("/mnt/d").exists():
        return Path("/mnt/d/AeroWAM-data")
    return Path.home() / ".local" / "share" / "aerowam"


def _configured(name: str, fallback: Path) -> Path:
    value = os.getenv(name)
    return _portable_path(value) if value else fallback


@dataclass(frozen=True)
class StoragePaths:
    root: Path
    datasets: Path
    checkpoints: Path
    px4_logs: Path
    episodes: Path
    gazebo_assets: Path
    cache: Path
    conda_envs: Path
    conda_pkgs: Path

    @classmethod
    def from_env(cls) -> "StoragePaths":
        root = _default_root()
        return cls(
            root=root,
            datasets=_configured("AEROWAM_DATASET_ROOT", root / "datasets"),
            checkpoints=_configured("AEROWAM_CHECKPOINT_ROOT", root / "checkpoints"),
            px4_logs=_configured("AEROWAM_LOG_ROOT", root / "px4_logs"),
            episodes=_configured("AEROWAM_EPISODE_ROOT", root / "episodes"),
            gazebo_assets=_configured("AEROWAM_GAZEBO_ASSET_ROOT", root / "gazebo_assets"),
            cache=_configured("AEROWAM_CACHE_ROOT", root / "cache"),
            conda_envs=_configured("AEROWAM_CONDA_ENV_ROOT", root / "conda_envs"),
            conda_pkgs=_configured("AEROWAM_CONDA_PKG_ROOT", root / "conda_pkgs"),
        )

    def create(self) -> None:
        for path in asdict(self).values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def serializable(self) -> dict[str, str]:
        return {name: str(path) for name, path in asdict(self).items()}


def get_storage_paths(create: bool = False) -> StoragePaths:
    paths = StoragePaths.from_env()
    if create:
        paths.create()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or create AeroWAM storage paths.")
    parser.add_argument("--create", action="store_true", help="Create every configured directory.")
    args = parser.parse_args()
    print(json.dumps(get_storage_paths(create=args.create).serializable(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
