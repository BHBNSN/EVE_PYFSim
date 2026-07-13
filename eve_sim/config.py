from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class EngineConfig:
    tick_rate: int = 1
    physics_substeps: int = 1
    lockstep: bool = True
    isolate_systems: bool = True
    parallel_systems: bool = False
    parallel_system_workers: int = 0
    parallel_system_target_pressure: float = 96.0
    parallel_system_timeout_sec: float = 30.0
    parallel_system_preflight: bool = True
    parallel_system_disable_after_failure: bool = True
    parallel_system_worker_start_method: str = "spawn"
    parallel_system_strict_validation: bool = True
    simulation_seed: int = 0
    tidi_min_factor: float = 0.1
    detailed_logging: bool = False
    hotspot_logging: bool = False
    detail_log_file: str = "logs/sim_detail.log"
    hotspot_log_file: str = "logs/sim_hotspot.log"
    log_merge_window_sec: float = 1.0


@dataclass(slots=True)
class UiConfig:
    width: int = 1400
    height: int = 900
    world_to_screen_scale: float = 0.3


def resolve_pyfa_source_dir() -> Path:
    env = os.getenv("PYFA_SOURCE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    default = Path(__file__).resolve().parents[1] / "Pyfa-master"
    return default
