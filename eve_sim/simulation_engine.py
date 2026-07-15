from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, fields
import logging
import multiprocessing
import random
import time

from .agents import CommanderAgent, ShipAgent
from .config import EngineConfig
from .domain.events import TickDiagnostics, TickResult
from .domain.squad_service import SquadLeadershipService
from .fleet_setup import prewarm_runtime_base_cache, prewarm_world_base_cache
from .sim_logging import get_sim_logger, log_sim_event
from .system_identity import stable_system_seed
from .systems import CombatStateCloneError, CombatSystem, DeployableSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .system_isolation import (
    SYSTEM_SHARD_PROTOCOL_VERSION,
    ParallelCapabilityError,
    SystemExecutionPlan,
    SystemExecutionMode,
    SystemGroupResult,
    SystemShardResult,
    SystemTransferIn,
    build_system_shard,
    commit_system_merge,
    log_group_hotspot,
    has_unassigned_active_entities,
    parallel_worker_probe,
    plan_system_execution,
    prepare_system_merge,
    replace_task_combat,
    run_system_group,
    sanitize_combat_for_worker,
    validate_parallel_capability,
    validate_shard_result,
)
from .world import WorldState


@dataclass(frozen=True, slots=True)
class SimulationCommandPorts:
    """Explicit command-side ports exposed only to the application composition root."""

    deployables: DeployableSystem
    module_metadata: CombatSystem
    fit_runtime: CombatSystem


class SimulationEngine:
    def __init__(self, world: WorldState, config: EngineConfig, combat_system: CombatSystem) -> None:
        self.world = world
        self.config = config
        self._logger: logging.Logger = get_sim_logger(config)
        self.commanders: list[CommanderAgent] = []
        self.ship_agents: dict[str, ShipAgent] = {}
        self.squad_leadership = SquadLeadershipService()
        self._last_tick_events = []

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
        self._system_random_states: dict[str, object] = {}
        self._system_executor: ProcessPoolExecutor | None = None
        self._system_executor_workers: int = 0
        self._parallel_preflight_complete: bool = False
        self._closed: bool = False
        self.last_system_execution_plan: SystemExecutionPlan = SystemExecutionPlan()
        self.last_system_parallel_error: str | None = None
        self.last_system_transfers: list[SystemTransferIn] = []
        if (
            not bool(getattr(config, "isolate_systems", True))
            or not isinstance(combat_system, CombatSystem)
        ):
            self.system_execution_mode = SystemExecutionMode.GLOBAL_SERIAL
        elif bool(getattr(config, "parallel_systems", False)):
            self.system_execution_mode = SystemExecutionMode.SHARD_PROCESS
        else:
            self.system_execution_mode = SystemExecutionMode.SHARD_SERIAL
        self.parallel_disabled_reason: str | None = None
        self.parallel_failure_count: int = 0
        self.parallel_disabled_at_tick: int | None = None
        self.last_parallel_tick_duration_ms: float | None = None
        self.last_serial_fallback_duration_ms: float | None = None
        self.last_effective_system_execution_mode: str = self.system_execution_mode.value
        self._fixed_system_executor_workers: int = 0
        self._isolated_commit_completed: bool = False
        prewarm_world_base_cache(world)

    @property
    def system_executor_workers(self) -> int:
        return self._system_executor_workers

    @staticmethod
    def _executor_process_snapshot(executor: ProcessPoolExecutor) -> list:
        processes = getattr(executor, "_processes", None)
        return list(processes.values()) if isinstance(processes, dict) else []

    @staticmethod
    def _process_is_alive(process) -> bool:
        try:
            return bool(process.is_alive())
        except Exception:
            return False

    def shutdown_parallel_workers(
        self,
        *,
        wait_for_workers: bool = True,
        force: bool = False,
        timeout_sec: float = 3.0,
    ) -> None:
        executor = self._system_executor
        if executor is None:
            return
        self._system_executor = None
        self._system_executor_workers = 0
        processes = self._executor_process_snapshot(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        if force:
            for process in processes:
                try:
                    process.terminate()
                except Exception:
                    pass
        elif wait_for_workers:
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            for process in processes:
                try:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                except Exception:
                    pass
        for process in processes:
            if self._process_is_alive(process):
                try:
                    process.terminate()
                except Exception:
                    pass
        for process in processes:
            if self._process_is_alive(process):
                try:
                    process.join(timeout=1.0)
                except Exception:
                    pass
        for process in processes:
            if self._process_is_alive(process) and hasattr(process, "kill"):
                try:
                    process.kill()
                    process.join(timeout=1.0)
                except Exception:
                    pass

    def close(self, timeout_sec: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        self.shutdown_parallel_workers(wait_for_workers=True, timeout_sec=timeout_sec)

    def __enter__(self) -> "SimulationEngine":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.shutdown_parallel_workers(wait_for_workers=False, force=True)
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

    def refresh_runtime_from_config(self) -> None:
        """Apply mutable runtime configuration at an application-controlled boundary."""
        self._logger = get_sim_logger(self.config)
        self.combat.attach_logger(
            self._logger,
            self.config.detailed_logging,
            self.config.log_merge_window_sec,
            self.config.hotspot_logging,
        )
        self.refresh_timing_from_config()

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

    def set_replica_tidi_factor(self, factor: float) -> None:
        self.tidi_factor = max(0.0, min(1.0, float(factor)))

    def current_tidi_factor(self) -> float:
        return float(self.tidi_factor)

    def runtime_diagnostics(self) -> dict[str, object]:
        """Return a detached diagnostics payload for application queries and snapshots."""
        return {
            "tick_rate": int(self.config.tick_rate),
            "execution_mode": self.system_execution_mode.value,
            "effective_execution_mode": self.last_effective_system_execution_mode,
            "parallel_disabled_reason": self.parallel_disabled_reason,
            "parallel_disabled_at_tick": self.parallel_disabled_at_tick,
            "parallel_failure_count": self.parallel_failure_count,
            "last_step_ms": self.last_step_ms,
            "tidi_factor": self.tidi_factor,
        }

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

    def command_ports(self) -> SimulationCommandPorts:
        return SimulationCommandPorts(
            deployables=self.deployables,
            module_metadata=self.combat,
            fit_runtime=self.combat,
        )

    def register_ship(self, ship_id: str) -> None:
        self.ship_agents[ship_id] = ShipAgent(agent_id=f"agent:{ship_id}", ship_id=ship_id)
        ship = self.world.ships.get(ship_id)
        if ship is not None:
            prewarm_runtime_base_cache(getattr(ship, "runtime", None))

    def unregister_ship(self, ship_id: str) -> None:
        self.ship_agents.pop(str(ship_id), None)

    def subscribe_combat_events(self, sink) -> None:
        previous_sink = getattr(self.combat, "_combat_event_sink", None)

        def chained(event) -> None:
            if previous_sink is not None:
                previous_sink(event)
            sink(event)

        self.combat.attach_event_sink(chained)

    def flush_pending_combat_events(self) -> None:
        self.combat.flush_pending_events()

    def step(self) -> TickResult:
        if self._closed:
            raise RuntimeError("SimulationEngine is closed")
        step_perf_started = time.perf_counter()
        previous_tick = int(self.world.tick)
        previous_now = float(self.world.now)
        self._isolated_commit_completed = False
        self._last_tick_events = []
        self.world.tick += 1

        step_start = float(self.world.now)
        step_end = step_start + self._dt
        self.world.now = step_end

        try:
            run_isolated = self._should_run_isolated_systems()
        except Exception:
            self.world.tick = previous_tick
            self.world.now = previous_now
            raise

        if run_isolated:
            authority_before_step = deepcopy(self.world)
            authority_before_step.tick = previous_tick
            authority_before_step.now = previous_now
            try:
                self._step_isolated_systems(step_start, step_end, step_perf_started)
            except Exception:
                if not self._isolated_commit_completed:
                    for state_field in fields(WorldState):
                        setattr(self.world, state_field.name, getattr(authority_before_step, state_field.name))
                raise
            return self._tick_result()

        self.last_system_execution_plan = SystemExecutionPlan()

        self._step_global_systems(step_start, step_end, step_perf_started)
        return self._tick_result()

    def _tick_result(self) -> TickResult:
        return TickResult(
            tick=int(self.world.tick),
            now=float(self.world.now),
            events=tuple(self._last_tick_events),
            transfers=tuple(self.last_system_transfers),
            diagnostics=TickDiagnostics(
                execution_mode=self.system_execution_mode.value,
                effective_execution_mode=str(self.last_effective_system_execution_mode),
                step_ms=float(self.last_step_ms),
                tidi_factor=float(self.tidi_factor),
                parallel_disabled_reason=self.parallel_disabled_reason,
            ),
        )

    def _refresh_squad_leaders(self) -> None:
        self._last_tick_events.extend(self.squad_leadership.refresh(self.world).events)

    def _step_global_systems(self, step_start: float, step_end: float, step_perf_started: float) -> None:
        self.shutdown_parallel_workers()
        self.last_system_transfers = []
        self._refresh_squad_leaders()

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
        if self.system_execution_mode is SystemExecutionMode.GLOBAL_SERIAL:
            return False
        if not isinstance(self.combat, CombatSystem):
            return False
        plan = plan_system_execution(self.world, self.config)
        if self.system_execution_mode is not SystemExecutionMode.SHARD_PROCESS:
            plan = SystemExecutionPlan(
                active_systems=plan.active_systems,
                groups=plan.groups,
                worker_count=1 if plan.groups else 0,
                use_processes=False,
            )
        if has_unassigned_active_entities(self.world):
            self.last_system_execution_plan = plan
            if self.system_execution_mode is SystemExecutionMode.SHARD_PROCESS:
                self.disable_parallel_for_match("active entity has an empty system_id", self.world.tick)
            raise ValueError("isolated execution requires every active entity to have a system_id")
        self.last_system_execution_plan = plan
        return True

    def _combat_for_system(self, system_id: str) -> CombatSystem:
        combat = self._system_combats.get(system_id)
        if combat is not None:
            return combat
        combat = self.combat.clone_for_system(system_id)
        combat.set_event_rng_context(stable_system_seed(self.config.simulation_seed, system_id), 0)
        self._system_combats[system_id] = combat
        return combat

    def _random_state_for_system(self, system_id: str) -> object:
        state = self._system_random_states.get(system_id)
        if state is not None:
            return state
        return random.Random(stable_system_seed(self.config.simulation_seed, system_id)).getstate()

    def _executor_for_plan(self, plan: SystemExecutionPlan) -> ProcessPoolExecutor:
        requested_workers = max(1, int(plan.worker_count or len(plan.groups) or 1))
        if self._fixed_system_executor_workers <= 0:
            self._fixed_system_executor_workers = requested_workers
        workers = self._fixed_system_executor_workers
        if self._system_executor is not None:
            return self._system_executor
        requested_method = str(getattr(self.config, "parallel_system_worker_start_method", "spawn") or "spawn")
        available_methods = multiprocessing.get_all_start_methods()
        if requested_method not in available_methods:
            requested_method = "spawn" if "spawn" in available_methods else available_methods[0]
            self.config.parallel_system_worker_start_method = requested_method
        self._system_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(requested_method),
        )
        self._system_executor_workers = workers
        return self._system_executor

    def _parallel_timeout_sec(self) -> float:
        try:
            timeout = float(getattr(self.config, "parallel_system_timeout_sec", 30.0))
        except Exception:
            timeout = 30.0
        if timeout <= 0.0:
            timeout = 30.0
            self.config.parallel_system_timeout_sec = timeout
        return timeout

    def disable_parallel_for_match(self, reason: str, tick: int) -> None:
        self.parallel_failure_count += 1
        if self.parallel_disabled_reason is None:
            self.parallel_disabled_reason = str(reason)
            self.parallel_disabled_at_tick = int(tick)
        self.system_execution_mode = SystemExecutionMode.SHARD_SERIAL_DEGRADED
        self.last_system_parallel_error = str(reason)
        self.shutdown_parallel_workers(wait_for_workers=False, force=True)

    def _preflight_parallel(self, plan: SystemExecutionPlan, tasks: list) -> None:
        if self._parallel_preflight_complete or not bool(getattr(self.config, "parallel_system_preflight", True)):
            return
        validate_parallel_capability(self.config, tasks, plan.worker_count)
        executor = self._executor_for_plan(plan)
        pid, protocol_version, _python_version = executor.submit(parallel_worker_probe).result(
            timeout=self._parallel_timeout_sec()
        )
        if pid <= 0 or protocol_version != SYSTEM_SHARD_PROTOCOL_VERSION:
            raise ParallelCapabilityError("worker probe returned an incompatible protocol")
        self._parallel_preflight_complete = True

    @staticmethod
    def _flatten_group_results(group_results: list[SystemGroupResult]) -> list[SystemShardResult]:
        return sorted(
            [result for group_result in group_results for result in group_result.results],
            key=lambda result: result.system_id,
        )

    def _validate_shard_results(self, tasks_by_system: dict, results: list[SystemShardResult]) -> None:
        result_ids = [result.system_id for result in results]
        expected_ids = sorted(tasks_by_system)
        if sorted(result_ids) != expected_ids or len(set(result_ids)) != len(result_ids):
            raise ValueError(f"worker result systems mismatch: expected {expected_ids}, got {result_ids}")
        if not bool(getattr(self.config, "parallel_system_strict_validation", True)):
            return
        for result in results:
            validate_shard_result(tasks_by_system[result.system_id], result, int(self.world.tick))

    def _run_parallel_groups(self, plan: SystemExecutionPlan, group_tasks: list[list], step_start: float, step_end: float):
        tasks = [task for group in group_tasks for task in group]
        self._preflight_parallel(plan, tasks)
        executor = self._executor_for_plan(plan)
        futures = [
            executor.submit(run_system_group, items, self.config, step_start, step_end, self._dt)
            for items in group_tasks
        ]
        done, not_done = wait(futures, timeout=self._parallel_timeout_sec())
        if not_done:
            for future in futures:
                future.cancel()
            raise TimeoutError(f"parallel system tick exceeded {self._parallel_timeout_sec():.3f}s")
        return [future.result() for future in futures]

    def _emit_isolated_events(self, events) -> None:
        sink = getattr(self.combat, "_combat_event_sink", None)
        if sink is None:
            return
        for event in events:
            sink(event)

    def _deliver_committed_events(self, events) -> None:
        try:
            self._emit_isolated_events(events)
        except Exception:
            self._logger.exception("committed tick event delivery failed")

    def _step_isolated_systems(self, step_start: float, step_end: float, step_perf_started: float) -> None:
        plan = self.last_system_execution_plan
        self._refresh_squad_leaders()
        for commander in self.commanders:
            commander.think(self.world)
        if not plan.groups:
            self.last_effective_system_execution_mode = "shard_idle"
            self.last_system_transfers = []
            self.last_system_parallel_error = None
            self._log_hotspot(
                "engine.step_total",
                step_perf_started,
                tick=self.world.tick,
                external_dt=self._dt,
                slices=0,
                systems=0,
                groups=0,
                processes=False,
            )
            return

        started = time.perf_counter()
        tasks_by_system = {}
        try:
            for group in plan.groups:
                for system_id in group.system_ids:
                    task = build_system_shard(self.world, system_id, self.ship_agents)
                    task.random_state = self._random_state_for_system(system_id)
                    tasks_by_system[system_id] = replace_task_combat(
                        task,
                        sanitize_combat_for_worker(self._combat_for_system(system_id)),
                    )
        except CombatStateCloneError as exc:
            if self.system_execution_mode is SystemExecutionMode.SHARD_PROCESS:
                self.disable_parallel_for_match(str(exc), self.world.tick)
            raise

        group_tasks = [
            [tasks_by_system[system_id] for system_id in group.system_ids if system_id in tasks_by_system]
            for group in plan.groups
        ]
        group_tasks = [tasks for tasks in group_tasks if tasks]

        group_results: list[SystemGroupResult]
        merge_plan = None
        if plan.use_processes:
            self.last_effective_system_execution_mode = SystemExecutionMode.SHARD_PROCESS.value
            parallel_started = time.perf_counter()
            try:
                group_results = self._run_parallel_groups(plan, group_tasks, step_start, step_end)
                shard_results = self._flatten_group_results(group_results)
                self._validate_shard_results(tasks_by_system, shard_results)
                merge_plan = prepare_system_merge(self.world, shard_results, self.ship_agents)
                self.last_system_parallel_error = None
                self.last_parallel_tick_duration_ms = (time.perf_counter() - parallel_started) * 1000.0
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__
                self.last_system_parallel_error = reason
                self._logger.exception("system parallel execution failed; falling back to serial shard execution")
                if bool(getattr(self.config, "parallel_system_disable_after_failure", True)):
                    self.disable_parallel_for_match(reason, self.world.tick)
                    self.last_effective_system_execution_mode = SystemExecutionMode.SHARD_SERIAL_DEGRADED.value
                else:
                    self.shutdown_parallel_workers(wait_for_workers=False, force=True)
                fallback_started = time.perf_counter()
                group_results = [
                    run_system_group(tasks, self.config, step_start, step_end, self._dt)
                    for tasks in group_tasks
                ]
                shard_results = self._flatten_group_results(group_results)
                self._validate_shard_results(tasks_by_system, shard_results)
                merge_plan = prepare_system_merge(self.world, shard_results, self.ship_agents)
                self.last_serial_fallback_duration_ms = (time.perf_counter() - fallback_started) * 1000.0
                plan = SystemExecutionPlan(
                    active_systems=plan.active_systems,
                    groups=plan.groups,
                    worker_count=1,
                    use_processes=False,
                )
                self.last_system_execution_plan = plan
        else:
            if self.system_execution_mode is SystemExecutionMode.SHARD_PROCESS:
                self.last_effective_system_execution_mode = "shard_serial_single_group"
            else:
                self.last_effective_system_execution_mode = self.system_execution_mode.value
            if self.system_execution_mode is not SystemExecutionMode.SHARD_PROCESS:
                self.shutdown_parallel_workers(wait_for_workers=True)
            group_results = [
                run_system_group(tasks, self.config, step_start, step_end, self._dt)
                for tasks in group_tasks
            ]
            shard_results = self._flatten_group_results(group_results)
            self._validate_shard_results(tasks_by_system, shard_results)
            merge_plan = prepare_system_merge(self.world, shard_results, self.ship_agents)

        assert merge_plan is not None
        transfer_ins: list[SystemTransferIn] = []
        for result in shard_results:
            if not isinstance(result.combat, CombatSystem) or result.combat._system_id != result.system_id:
                raise CombatStateCloneError(f"invalid committed combat state for {result.system_id!r}")
            if result.random_state is None:
                raise ValueError(f"missing committed random state for {result.system_id!r}")
        events = commit_system_merge(self.world, self.ship_agents, merge_plan, transfer_sink=transfer_ins)
        for result in shard_results:
            self._system_combats[result.system_id] = result.combat
            self._system_random_states[result.system_id] = result.random_state
        self._isolated_commit_completed = True
        self.last_system_transfers = transfer_ins
        self._deliver_committed_events(events)
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
