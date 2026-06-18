from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import logging
import time

from .agents import CommanderAgent, ShipAgent
from .config import EngineConfig
from .fleet_setup import prewarm_runtime_base_cache, prewarm_world_base_cache
from .pyfa_bridge import PyfaBridge
from .sim_logging import get_sim_logger, log_sim_event
from .systems import CombatSystem, DeployableSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .system_isolation import (
    SystemExecutionPlan,
    SystemTransferIn,
    build_system_shard,
    log_group_hotspot,
    has_unassigned_active_entities,
    merge_system_results,
    plan_system_execution,
    replace_task_combat,
    run_system_group,
    sanitize_combat_for_worker,
)
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
        self._system_combats: dict[str, CombatSystem] = {}
        self._system_executor: ProcessPoolExecutor | None = None
        self._system_executor_workers: int = 0
        self.last_system_execution_plan: SystemExecutionPlan = SystemExecutionPlan()
        self.last_system_parallel_error: str | None = None
        self.last_system_transfers: list[SystemTransferIn] = []
        prewarm_world_base_cache(world)

    def shutdown_parallel_workers(self) -> None:
        if self._system_executor is None:
            return
        self._system_executor.shutdown(wait=False, cancel_futures=True)
        self._system_executor = None
        self._system_executor_workers = 0

    def __del__(self) -> None:
        try:
            self.shutdown_parallel_workers()
        except Exception:
            pass

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

        if self._should_run_isolated_systems():
            self._step_isolated_systems(step_start, step_end, step_perf_started)
            return

        self.last_system_execution_plan = SystemExecutionPlan()

        self._step_global_systems(step_start, step_end, step_perf_started)

    def _step_global_systems(self, step_start: float, step_end: float, step_perf_started: float) -> None:
        self.shutdown_parallel_workers()
        self.last_system_transfers = []

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

    def _should_run_isolated_systems(self) -> bool:
        if not bool(getattr(self.config, "isolate_systems", True)):
            return False
        if not isinstance(self.combat, CombatSystem):
            return False
        plan = plan_system_execution(self.world, self.config)
        if has_unassigned_active_entities(self.world):
            self.last_system_execution_plan = plan
            return False
        if len(plan.active_systems) <= 1:
            self.last_system_execution_plan = plan
            return False
        self.last_system_execution_plan = plan
        return True

    def _clone_base_combat(self) -> CombatSystem:
        sink = getattr(self.combat, "_combat_event_sink", None)
        logger = getattr(self.combat, "logger", None)
        try:
            self.combat.attach_event_sink(None)
            self.combat.logger = None
            cloned = deepcopy(self.combat)
        except Exception:
            cloned = CombatSystem(PyfaBridge())
        finally:
            self.combat.attach_event_sink(sink)
            self.combat.logger = logger
        cloned.attach_event_sink(None)
        cloned.logger = None
        return cloned

    def _combat_for_system(self, system_id: str) -> CombatSystem:
        combat = self._system_combats.get(system_id)
        if combat is not None:
            return combat
        combat = self._clone_base_combat() if not self._system_combats else CombatSystem(PyfaBridge())
        self._system_combats[system_id] = combat
        return combat

    def _executor_for_plan(self, plan: SystemExecutionPlan) -> ProcessPoolExecutor:
        workers = max(1, int(plan.worker_count or len(plan.groups) or 1))
        if self._system_executor is not None and self._system_executor_workers == workers:
            return self._system_executor
        self.shutdown_parallel_workers()
        self._system_executor = ProcessPoolExecutor(max_workers=workers)
        self._system_executor_workers = workers
        return self._system_executor

    def _emit_isolated_events(self, events) -> None:
        sink = getattr(self.combat, "_combat_event_sink", None)
        if sink is None:
            return
        for event in events:
            sink(event)

    def _step_isolated_systems(self, step_start: float, step_end: float, step_perf_started: float) -> None:
        plan = self.last_system_execution_plan
        if not plan.groups:
            self._step_global_systems(step_start, step_end, step_perf_started)
            return

        started = time.perf_counter()
        tasks_by_system = {}
        for group in plan.groups:
            for system_id in group.system_ids:
                task = build_system_shard(self.world, system_id, self.commanders, self.ship_agents)
                tasks_by_system[system_id] = replace_task_combat(
                    task,
                    sanitize_combat_for_worker(self._combat_for_system(system_id)),
                )

        group_tasks = [
            [tasks_by_system[system_id] for system_id in group.system_ids if system_id in tasks_by_system]
            for group in plan.groups
        ]
        group_tasks = [tasks for tasks in group_tasks if tasks]

        group_results = []
        if plan.use_processes:
            executor = self._executor_for_plan(plan)
            futures = [
                executor.submit(run_system_group, tasks, self.config, step_start, step_end, self._dt)
                for tasks in group_tasks
            ]
            try:
                group_results = [future.result() for future in futures]
                self.last_system_parallel_error = None
            except Exception as exc:
                self.last_system_parallel_error = str(exc) or exc.__class__.__name__
                self._logger.exception("system parallel execution failed; falling back to serial shard execution")
                for future in futures:
                    future.cancel()
                self.shutdown_parallel_workers()
                group_results = [
                    run_system_group(tasks, self.config, step_start, step_end, self._dt)
                    for tasks in group_tasks
                ]
                plan = SystemExecutionPlan(
                    active_systems=plan.active_systems,
                    groups=plan.groups,
                    worker_count=1,
                    use_processes=False,
                )
                self.last_system_execution_plan = plan
        else:
            self.shutdown_parallel_workers()
            self.last_system_parallel_error = None
            group_results = [
                run_system_group(tasks, self.config, step_start, step_end, self._dt)
                for tasks in group_tasks
            ]

        shard_results = [result for group_result in group_results for result in group_result.results]
        for result in shard_results:
            self._system_combats[result.system_id] = result.combat

        transfer_ins: list[SystemTransferIn] = []
        events = merge_system_results(self.world, shard_results, self.ship_agents, transfer_sink=transfer_ins)
        self.last_system_transfers = transfer_ins
        self._emit_isolated_events(events)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log_group_hotspot(self.config, plan, elapsed_ms)
        self._log_hotspot(
            "engine.step_total",
            step_perf_started,
            tick=self.world.tick,
            external_dt=self._dt,
            slices=max(1, int(self.config.physics_substeps)),
            systems=len(plan.active_systems),
            groups=len(plan.groups),
            processes=bool(plan.use_processes),
        )

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
