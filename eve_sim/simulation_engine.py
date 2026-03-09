from __future__ import annotations

from dataclasses import asdict
import logging

from .agents import CommanderAgent, ShipAgent
from .config import EngineConfig
from .sim_logging import get_sim_logger
from .systems import CombatSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .world import WorldState


class SimulationEngine:
    def __init__(self, world: WorldState, config: EngineConfig, combat_system: CombatSystem) -> None:
        self.world = world
        self.config = config
        self._logger: logging.Logger = get_sim_logger(config)
        self.commanders: list[CommanderAgent] = []
        self.ship_agents: dict[str, ShipAgent] = {}

        self.perception = PerceptionSystem()
        self.movement = MovementSystem(config.battlefield_radius)
        self.combat = combat_system
        self.combat.attach_logger(self._logger, self.config.detailed_logging, self.config.log_merge_window_sec)
        self.logistics = LogisticsSystem()

        self._dt = 1.0 / config.tick_rate

    def register_commander(self, commander: CommanderAgent) -> None:
        self.commanders.append(commander)

    def register_ship(self, ship_id: str) -> None:
        self.ship_agents[ship_id] = ShipAgent(agent_id=f"agent:{ship_id}", ship_id=ship_id)

    def step(self) -> None:
        self.world.tick += 1
        self.world.now += self._dt

        self.perception.run(self.world)

        for commander in self.commanders:
            commander.think(self.world)

        for agent in self.ship_agents.values():
            agent.sense(self.world)
            agent.think(self.world)

        sub_dt = self._dt / max(1, self.config.physics_substeps)
        for _ in range(self.config.physics_substeps):
            self.movement.run(self.world, sub_dt)
            self.combat.run(self.world, sub_dt)
            self.logistics.run(self.world, sub_dt)

    def snapshot(self) -> dict:
        ships = {}
        for ship_id, ship in self.world.ships.items():
            module_states: dict[str, str] = {}
            if ship.runtime is not None:
                module_states = {module.module_id: module.state.value for module in ship.runtime.modules}
            ships[ship_id] = {
                "ship_id": ship_id,
                "team": ship.team.value,
                "squad_id": ship.squad_id,
                "ship_name": ship.fit.ship_name,
                "alive": ship.vital.alive,
                "position": {"x": ship.nav.position.x, "y": ship.nav.position.y},
                "velocity": {"x": ship.nav.velocity.x, "y": ship.nav.velocity.y},
                "facing_deg": ship.nav.facing_deg,
                "shield": ship.vital.shield,
                "armor": ship.vital.armor,
                "structure": ship.vital.structure,
                "shield_max": ship.vital.shield_max,
                "armor_max": ship.vital.armor_max,
                "structure_max": ship.vital.structure_max,
                "cap": ship.vital.cap,
                "cap_max": ship.vital.cap_max,
                "target": ship.combat.current_target,
                "projected_targets": {k: v for k, v in ship.combat.projected_targets.items()},
                "module_cycle_timers": {k: float(v) for k, v in ship.combat.module_cycle_timers.items()},
                "ecm_jam_sources": {k: float(v) for k, v in ship.combat.ecm_jam_sources.items()},
                "ecm_last_attempt_target": ship.combat.ecm_last_attempt_target,
                "ecm_last_attempt_module": ship.combat.ecm_last_attempt_module,
                "ecm_last_attempt_success": ship.combat.ecm_last_attempt_success,
                "ecm_last_attempt_chance": float(ship.combat.ecm_last_attempt_chance),
                "ecm_last_attempt_at": float(ship.combat.ecm_last_attempt_at),
                "ecm_last_attempt_target_by_module": {k: str(v) for k, v in ship.combat.ecm_last_attempt_target_by_module.items()},
                "ecm_last_attempt_success_by_module": {k: bool(v) for k, v in ship.combat.ecm_last_attempt_success_by_module.items()},
                "ecm_last_attempt_at_by_module": {k: float(v) for k, v in ship.combat.ecm_last_attempt_at_by_module.items()},
                "module_states": module_states,
            }
        return {
            "tick": self.world.tick,
            "now": self.world.now,
            "ships": ships,
            "intents": {k: asdict(v) for k, v in self.world.intents.items()},
            "squad_focus_queues": {k: list(v) for k, v in self.world.squad_focus_queues.items()},
        }
