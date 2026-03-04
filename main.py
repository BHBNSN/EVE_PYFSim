from __future__ import annotations

import argparse
from eve_sim.config import EngineConfig
from eve_sim.gui_app import run_gui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EVE SIM continuous-space wargame")
    p.add_argument("--detail-log", action=argparse.BooleanOptionalAction, default=False, help="enable detailed per-round log")
    p.add_argument("--log-file", default="logs/sim_detail.log", help="detail log file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine_cfg = EngineConfig(detailed_logging=bool(args.detail_log), detail_log_file=args.log_file)
    run_gui(engine_config=engine_cfg)


if __name__ == "__main__":
    main()
