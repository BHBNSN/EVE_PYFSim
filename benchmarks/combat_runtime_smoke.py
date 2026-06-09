from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eve_sim.config import EngineConfig
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem
from eve_sim.world import WorldState


def run_empty_world_steps(steps: int = 300) -> float:
    engine = SimulationEngine(WorldState(), EngineConfig(tick_rate=30, physics_substeps=1), CombatSystem(PyfaBridge()))
    started = time.perf_counter()
    for _ in range(steps):
        engine.step()
    return time.perf_counter() - started


if __name__ == "__main__":
    elapsed = run_empty_world_steps()
    print(f"combat_runtime_smoke_seconds={elapsed:.6f}")
