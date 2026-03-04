from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class EngineConfig:
    tick_rate: int = 30
    physics_substeps: int = 1
    lockstep: bool = True
    battlefield_radius: float = 800_000.0
    detailed_logging: bool = False
    detail_log_file: str = "logs/sim_detail.log"
    log_merge_window_sec: float = 1.0


@dataclass(slots=True)
class UiConfig:
    width: int = 1400
    height: int = 900
    world_to_screen_scale: float = 0.0009


def resolve_pyfa_source_dir() -> Path:
    env = os.getenv("PYFA_SOURCE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    default = Path(__file__).resolve().parents[1] / "Pyfa-master"
    return default
