"""Load the pinned track definition (configs/<track>.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
TRACK_FILES = {"HPC-01": "hpc01.yaml", "HPC-02": "hpc02.yaml"}


@dataclass
class Track:
    data: dict
    path: Path

    @property
    def id(self) -> str:
        return self.data["track"]

    def __getitem__(self, key):
        return self.data[key]

    @property
    def runtime_env(self) -> dict[str, str]:
        return {k: str(v) for k, v in (self.data["runtime"].get("env") or {}).items()}

    def clean_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """os.environ with every SPARKINFER_* knob removed, then the pinned ones applied."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKINFER_")}
        env.update(self.runtime_env)
        env.update(extra or {})
        return env


def load_track(name_or_path: str = "HPC-01") -> Track:
    p = Path(name_or_path)
    if not p.exists():
        fname = TRACK_FILES.get(name_or_path.upper())
        if not fname:
            raise FileNotFoundError(f"unknown track {name_or_path!r}")
        p = CONFIGS / fname
    return Track(yaml.safe_load(p.read_text()), p)
