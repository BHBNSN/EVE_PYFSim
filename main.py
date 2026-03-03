from __future__ import annotations

import argparse
import json

from eve_sim.agents import CommanderAgent
from eve_sim.config import EngineConfig
from eve_sim.gui_app import run_gui
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.scenario import build_default_world
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem


def build_engine(config: EngineConfig | None = None) -> SimulationEngine:
    cfg = config or EngineConfig()
    pyfa = PyfaBridge()
    world = build_default_world(pyfa)
    engine = SimulationEngine(world=world, config=cfg, combat_system=CombatSystem(pyfa))
    engine.register_commander(CommanderAgent(agent_id="cmd-blue", squad_ids=["BLUE-ALPHA", "BLUE-LOGI"]))
    engine.register_commander(CommanderAgent(agent_id="cmd-red", squad_ids=["RED-ALPHA"]))
    for ship_id in world.ships:
        engine.register_ship(ship_id)
    return engine


def run_headless(seconds: float, config: EngineConfig | None = None) -> None:
    engine = build_engine(config)
    snapshot = engine.run_blocking(seconds)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EVE SIM continuous-space wargame")
    p.add_argument(
        "--mode",
        choices=["gui", "headless"],
        default="gui",
        help="run mode (default: gui)"
    )
    p.add_argument("--seconds", type=float, default=20.0, help="seconds for headless mode")
    p.add_argument("--detail-log", action=argparse.BooleanOptionalAction, default=False, help="enable detailed per-round log")
    p.add_argument("--log-file", default="logs/sim_detail.log", help="detail log file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine_cfg = EngineConfig(detailed_logging=bool(args.detail_log), detail_log_file=args.log_file)
    if args.mode == "gui":
        run_gui(engine_config=engine_cfg)
        return
    if args.mode == "headless":
        run_headless(args.seconds, config=engine_cfg)
        return


if __name__ == "__main__":
    main()
