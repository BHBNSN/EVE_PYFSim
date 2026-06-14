from __future__ import annotations

from dataclasses import asdict
import logging
import time

from .agents import CommanderAgent, ShipAgent
from .config import EngineConfig
from .fleet_setup import prewarm_runtime_base_cache, prewarm_world_base_cache
from .sim_logging import get_sim_logger, log_sim_event
from .systems import CombatSystem, DeployableSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .world import WorldState


class SimulationEngine:
    def __init__(self, world: WorldState, config: EngineConfig, combat_system: CombatSystem) -> None:
        self.world = world
        self.config = config
        self._logger: logging.Logger = get_sim_logger(config)
        self.commanders: list[CommanderAgent] = []
        self.ship_agents: dict[str, ShipAgent] = {}

        self.perception = PerceptionSystem()
        self.movement = MovementSystem()
        self.combat = combat_system
        self.combat.attach_logger(
            self._logger,
            self.config.detailed_logging,
            self.config.log_merge_window_sec,
            self.config.hotspot_logging,
        )
        self.deployables = DeployableSystem(self.combat, self.movement)
        self.logistics = LogisticsSystem()

        self.tidi_factor: float = 1.0
        self.last_step_ms: float = 0.0
        self.last_step_budget_ms: float = self.nominal_tick_interval_ms()
        self._dt = 1.0 / float(self._configured_tick_rate())
        prewarm_world_base_cache(world)

    def _configured_tick_rate(self) -> int:
        try:
            return max(1, int(float(self.config.tick_rate)))
        except Exception:
            return 1

    def refresh_timing_from_config(self) -> None:
        self._dt = 1.0 / float(self._configured_tick_rate())
        self.last_step_budget_ms = self.nominal_tick_interval_ms()

    def nominal_tick_interval_ms(self) -> float:
        return 1000.0 / float(self._configured_tick_rate())

    def tidi_tick_interval_ms(self) -> int:
        factor = max(self._configured_tidi_min_factor(), min(1.0, float(self.tidi_factor or 1.0)))
        return max(1, int(round(self.nominal_tick_interval_ms() / factor)))

    def next_tick_delay_ms(self) -> int:
        return max(1, int(round(float(self.tidi_tick_interval_ms()) - float(self.last_step_ms))))

    def _configured_tidi_min_factor(self) -> float:
        try:
            return max(0.01, min(1.0, float(self.config.tidi_min_factor)))
        except Exception:
            return 0.1

    def update_tidi_after_step(self, elapsed_ms: float) -> None:
        try:
            elapsed = max(0.0, float(elapsed_ms))
        except Exception:
            elapsed = 0.0
        budget = self.nominal_tick_interval_ms()
        self.last_step_ms = elapsed
        self.last_step_budget_ms = budget
        if elapsed <= 0.0 or elapsed <= budget:
            self.tidi_factor = 1.0
            return
        self.tidi_factor = max(self._configured_tidi_min_factor(), min(1.0, budget / elapsed))

    def _log_hotspot(self, name: str, start_time: float, **fields) -> None:
        if not bool(getattr(self.config, "hotspot_logging", False)):
            return
        if self._logger.disabled:
            return
        log_sim_event(
            self._logger,
            "hotspot",
            name=name,
            duration_ms=(time.perf_counter() - start_time) * 1000.0,
            **fields,
        )

    def register_commander(self, commander: CommanderAgent) -> None:
        self.commanders.append(commander)

    def register_ship(self, ship_id: str) -> None:
        self.ship_agents[ship_id] = ShipAgent(agent_id=f"agent:{ship_id}", ship_id=ship_id)
        ship = self.world.ships.get(ship_id)
        if ship is not None:
            prewarm_runtime_base_cache(getattr(ship, "runtime", None))

    def step(self) -> None:
        step_perf_started = time.perf_counter()
        self.world.tick += 1

        step_start = float(self.world.now)
        step_end = step_start + self._dt
        self.world.now = step_end

        perf_started = time.perf_counter()
        self.perception.run(self.world)
        self._log_hotspot("engine.perception", perf_started, tick=self.world.tick)

        perf_started = time.perf_counter()
        for commander in self.commanders:
            commander.think(self.world)
        self._log_hotspot("engine.commanders", perf_started, tick=self.world.tick, commanders=len(self.commanders))

        perf_started = time.perf_counter()
        for agent in self.ship_agents.values():
            agent.sense(self.world)
            agent.think(self.world)
        self._log_hotspot("engine.ship_agents", perf_started, tick=self.world.tick, agents=len(self.ship_agents))

        perf_started = time.perf_counter()
        self.deployables.run(self.world, self._dt, advance_physics=False, apply_effects=False)
        self._log_hotspot("engine.deployables.prepare", perf_started, tick=self.world.tick)

        substep_count = max(1, int(self.config.physics_substeps))
        base_slice_dt = self._dt / substep_count
        self.world.now = step_start

        for slice_index in range(substep_count):
            substep_start = step_start + (base_slice_dt * slice_index)
            if slice_index + 1 >= substep_count:
                substep_end = step_end
            else:
                substep_end = substep_start + base_slice_dt
            slice_dt = max(1e-6, float(substep_end) - float(substep_start))
            self.world.now = substep_end

            perf_started = time.perf_counter()
            self.movement.run(self.world, slice_dt)
            self._log_hotspot("engine.movement", perf_started, tick=self.world.tick, slice_index=slice_index, slice_dt=slice_dt)

            perf_started = time.perf_counter()
            self.deployables.run_physics(self.world, slice_dt)
            self._log_hotspot("engine.deployables.physics", perf_started, tick=self.world.tick, slice_index=slice_index, slice_dt=slice_dt)

        self.world.now = step_end

        perf_started = time.perf_counter()
        self.combat.run(self.world, self._dt)
        self._log_hotspot("engine.combat", perf_started, tick=self.world.tick, dt=self._dt)

        perf_started = time.perf_counter()
        self.deployables.run(self.world, self._dt, advance_physics=False, apply_effects=True)
        self._log_hotspot("engine.deployables", perf_started, tick=self.world.tick, dt=self._dt)

        perf_started = time.perf_counter()
        self.logistics.run(self.world, self._dt)
        self._log_hotspot("engine.logistics", perf_started, tick=self.world.tick, dt=self._dt)

        self._log_hotspot("engine.step_total", step_perf_started, tick=self.world.tick, external_dt=self._dt, slices=substep_count)

    def snapshot(self) -> dict:
        ships = {}
        for ship_id, ship in self.world.ships.items():
            module_states: dict[str, str] = {}
            if ship.runtime is not None:
                module_states = {
                    module.module_id: module.normalized_state().value
                    for module in ship.runtime.modules
                }
            ships[ship_id] = {
                "ship_id": ship_id,
                "team": ship.team.value,
                "squad_id": ship.squad_id,
                "ship_group_id": str(getattr(ship, "ship_group_id", "") or ""),
                "ship_name": ship.fit.ship_name,
                "alive": ship.vital.alive,
                "position": {"x": ship.nav.position.x, "y": ship.nav.position.y},
                "velocity": {"x": ship.nav.velocity.x, "y": ship.nav.velocity.y},
                "facing_deg": ship.nav.facing_deg,
                "system_id": str(getattr(ship.nav, "system_id", "") or ""),
                "gate_target_structure_id": str(getattr(getattr(ship.nav, "gate", None), "target_structure_id", "") or ""),
                "gate_cloak_active": bool(getattr(getattr(ship.nav, "cloak", None), "active", False)),
                "gate_cloak_expires_at": float(getattr(getattr(ship.nav, "cloak", None), "expires_at", 0.0) or 0.0),
                "gate_cloak_source": str(getattr(getattr(ship.nav, "cloak", None), "source", "") or ""),
                "follow_hold_active": bool(getattr(ship.nav, "follow_hold_active", False)),
                "follow_hold_leader_id": str(getattr(ship.nav, "follow_hold_leader_id", "") or ""),
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
            "drones": {
                drone_id: {
                    "ship_id": drone_id,
                    "owner_ship_id": drone.owner_ship_id,
                    "team": drone.team.value,
                    "squad_id": drone.squad_id,
                    "type_name": drone.definition.type_name,
                    "group_name": drone.definition.group_name,
                    "max_velocity": float(drone.definition.max_velocity),
                    "state": drone.state,
                    "target_id": drone.target_id,
                    "connected": bool(drone.connected),
                    "target_command_at": float(drone.target_command_at),
                    "alive": drone.vital.alive,
                    "is_sentry": bool(drone.definition.is_sentry),
                    "position": {"x": drone.nav.position.x, "y": drone.nav.position.y},
                    "velocity": {"x": drone.nav.velocity.x, "y": drone.nav.velocity.y},
                    "facing_deg": drone.nav.facing_deg,
                    "system_id": str(getattr(drone.nav, "system_id", "") or ""),
                    "shield": drone.vital.shield,
                    "armor": drone.vital.armor,
                    "structure": drone.vital.structure,
                    "shield_max": drone.vital.shield_max,
                    "armor_max": drone.vital.armor_max,
                    "structure_max": drone.vital.structure_max,
                    "cycle_timer": float(drone.cycle_timer),
                    "ewar_cycle_timer": float(drone.ewar_cycle_timer),
                }
                for drone_id, drone in self.world.drones.items()
            },
            "fighters": {
                fighter_id: {
                    "ship_id": fighter_id,
                    "owner_ship_id": fighter.owner_ship_id,
                    "team": fighter.team.value,
                    "squad_id": fighter.squad_id,
                    "owner_squad_id": fighter.owner_squad_id,
                    "type_name": fighter.definition.type_name,
                    "group_name": fighter.definition.group_name,
                    "slot_kind": fighter.definition.slot_kind,
                    "squadron_size": int(fighter.definition.squadron_size),
                    "max_velocity": float(fighter.definition.max_velocity),
                    "state": fighter.state,
                    "target_id": fighter.target_id,
                    "connected": bool(fighter.connected),
                    "target_command_at": float(fighter.target_command_at),
                    "alive": fighter.vital.alive,
                    "position": {"x": fighter.nav.position.x, "y": fighter.nav.position.y},
                    "velocity": {"x": fighter.nav.velocity.x, "y": fighter.nav.velocity.y},
                    "facing_deg": fighter.nav.facing_deg,
                    "system_id": str(getattr(fighter.nav, "system_id", "") or ""),
                    "shield": fighter.vital.shield,
                    "armor": fighter.vital.armor,
                    "structure": fighter.vital.structure,
                    "shield_max": fighter.vital.shield_max,
                    "armor_max": fighter.vital.armor_max,
                    "structure_max": fighter.vital.structure_max,
                    "ability_cycle_timers": {k: float(v) for k, v in fighter.ability_cycle_timers.items()},
                    "ability_ammo_remaining": {k: int(v) for k, v in fighter.ability_ammo_remaining.items()},
                    "ability_reload_timers": {k: float(v) for k, v in fighter.ability_reload_timers.items()},
                    "pending_manual_abilities": sorted(str(k) for k in fighter.pending_manual_abilities),
                    "mwd_active_timer": float(fighter.mwd_active_timer),
                    "mwd_cooldown_timer": float(fighter.mwd_cooldown_timer),
                }
                for fighter_id, fighter in self.world.fighters.items()
            },
            "projectiles": {
                projectile_id: {
                    "projectile_id": projectile.projectile_id,
                    "kind": projectile.kind,
                    "source_ship_id": projectile.source_ship_id,
                    "source_module_id": projectile.source_module_id,
                    "team": projectile.team.value,
                    "position": {"x": projectile.position.x, "y": projectile.position.y},
                    "velocity": {"x": projectile.velocity.x, "y": projectile.velocity.y},
                    "system_id": str(getattr(projectile, "system_id", "") or ""),
                    "target_ship_id": projectile.target_ship_id,
                    "speed": float(projectile.speed),
                    "max_speed": float(projectile.max_speed),
                    "distance_traveled": float(projectile.distance_traveled),
                    "flight_time": float(projectile.flight_time),
                    "age": float(projectile.age),
                    "blast_radius": float(projectile.blast_radius),
                }
                for projectile_id, projectile in self.world.projectiles.items()
            },
            "projectile_blasts": {
                blast_id: {
                    "blast_id": blast.blast_id,
                    "kind": blast.kind,
                    "position": {"x": blast.position.x, "y": blast.position.y},
                    "system_id": str(getattr(blast, "system_id", "") or ""),
                    "radius_m": float(blast.radius_m),
                    "expires_at": float(blast.expires_at),
                }
                for blast_id, blast in self.world.projectile_blasts.items()
            },
            "bubble_fields": {
                field_id: {
                    "field_id": field.field_id,
                    "kind": field.kind,
                    "interdiction_kind": field.interdiction_kind,
                    "source_ship_id": field.source_ship_id,
                    "source_module_id": field.source_module_id,
                    "team": field.team.value,
                    "position": {"x": field.position.x, "y": field.position.y},
                    "system_id": str(getattr(field, "system_id", "") or ""),
                    "radius_m": float(field.radius_m),
                    "expires_at": float(field.expires_at),
                    "blocks_warp": bool(field.blocks_warp),
                    "speed_factor_mult": float(field.speed_factor_mult),
                    "anchor_ship_id": field.anchor_ship_id,
                    "alive": bool(field.alive),
                }
                for field_id, field in self.world.bubble_fields.items()
            },
            "intents": {k: asdict(v) for k, v in self.world.intents.items()},
            "squad_focus_queues": {k: list(v) for k, v in self.world.squad_focus_queues.items()},
            "squad_focus_updated_at": {k: float(v) for k, v in self.world.squad_focus_updated_at.items()},
        }
