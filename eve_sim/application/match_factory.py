from __future__ import annotations

from collections.abc import Iterable

from ..agents import CommanderAgent
from ..config import EngineConfig
from ..fleet_setup import EftFitParser, ManualShipSetup, RuntimeFromEftFactory, build_world_from_manual_setup
from ..maps import MapDefinition
from ..models import Team
from ..pyfa_bridge import PyfaBridge
from ..simulation_engine import SimulationEngine
from ..systems import CombatSystem
from .match_application import MatchApplication


class MatchApplicationFactory:
    """Assemble a complete match without leaking kernel construction into presentation code."""

    @staticmethod
    def from_manual_setup(
        setup: Iterable[ManualShipSetup],
        config: EngineConfig,
        *,
        map_definition: MapDefinition | None = None,
        pyfa: PyfaBridge | None = None,
    ) -> MatchApplication:
        parser = EftFitParser()
        runtime_factory = RuntimeFromEftFactory()
        world = build_world_from_manual_setup(list(setup), map_definition=map_definition)
        engine = SimulationEngine(
            world=world,
            config=config,
            combat_system=CombatSystem(pyfa or PyfaBridge()),
        )
        engine.register_commander(CommanderAgent(agent_id="cmd-blue", team=Team.BLUE))
        engine.register_commander(CommanderAgent(agent_id="cmd-red", team=Team.RED))
        for ship_id in sorted(world.ships):
            engine.register_ship(ship_id)
        return MatchApplication.from_engine(
            engine,
            fit_parser=parser,
            fit_factory=runtime_factory,
        )


__all__ = ["MatchApplicationFactory"]
