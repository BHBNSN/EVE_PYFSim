from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
import math
import os
import pickle
import random
import sys
import time
from typing import Any

from .agents import ShipAgent
from .config import EngineConfig
from .pyfa_bridge import PyfaBridge
from .replay.schema import CombatEvent
from .sim_logging import get_sim_logger, log_sim_event
from .squad_identity import squad_key
from .system_identity import normalize_system_namespace, stable_system_seed
from .systems import CombatSystem, DeployableSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .timing_wheel import TimingWheel
from .world import WorldState


SYSTEM_SHARD_PROTOCOL_VERSION = 2


class SystemExecutionMode(Enum):
    GLOBAL_SERIAL = "global_serial"
    SHARD_SERIAL = "shard_serial"
    SHARD_PROCESS = "shard_process"
    SHARD_SERIAL_DEGRADED = "shard_serial_degraded"


class DuplicateEntityIdError(RuntimeError):
    pass


class ShardResultValidationError(RuntimeError):
    pass


class ParallelCapabilityError(RuntimeError):
    pass


@dataclass(slots=True)
class SystemPressure:
    system_id: str
    pressure: float
    ship_count: int = 0
    deployable_count: int = 0
    projectile_count: int = 0
    bubble_count: int = 0


@dataclass(slots=True)
class SystemExecutionGroup:
    system_ids: tuple[str, ...]
    pressure: float


@dataclass(slots=True)
class SystemExecutionPlan:
    active_systems: tuple[SystemPressure, ...] = tuple()
    groups: tuple[SystemExecutionGroup, ...] = tuple()
    worker_count: int = 0
    use_processes: bool = False


@dataclass(slots=True)
class SystemTransferOut:
    collection_name: str
    entity_id: str
    source_system_id: str
    destination_system_id: str
    entity: Any
    reason: str = "system_id_changed"


@dataclass(slots=True)
class SystemTransferIn:
    collection_name: str
    entity_id: str
    source_system_id: str
    destination_system_id: str
    entity: Any
    reason: str = "system_id_changed"


@dataclass(slots=True)
class SystemShardTask:
    system_id: str
    world: WorldState
    combat: CombatSystem
    ship_agents: dict[str, ShipAgent]
    owned_entity_ids: dict[str, set[str]] = field(default_factory=dict)
    tick: int = 0
    protocol_version: int = SYSTEM_SHARD_PROTOCOL_VERSION
    random_state: object | None = None


@dataclass(slots=True)
class SystemShardResult:
    system_id: str
    world: WorldState
    combat: CombatSystem
    ship_agents: dict[str, ShipAgent] = field(default_factory=dict)
    events: list[CombatEvent] = field(default_factory=list)
    elapsed_ms: float = 0.0
    owned_entity_ids: dict[str, set[str]] = field(default_factory=dict)
    transfer_outs: list[SystemTransferOut] = field(default_factory=list)
    tick: int = 0
    protocol_version: int = SYSTEM_SHARD_PROTOCOL_VERSION
    random_state: object | None = None


@dataclass(slots=True)
class SystemGroupResult:
    results: list[SystemShardResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class SystemMergePlan:
    world: WorldState
    ship_agents: dict[str, ShipAgent]
    events: list[CombatEvent] = field(default_factory=list)
    transfer_ins: list[SystemTransferIn] = field(default_factory=list)


def entity_system_id(entity: Any) -> str:
    nav = getattr(entity, "nav", None)
    if nav is not None:
        return str(getattr(nav, "system_id", "") or "")
    return str(getattr(entity, "system_id", "") or "")


def _entity_is_active(entity: Any) -> bool:
    vital = getattr(entity, "vital", None)
    if vital is not None:
        return bool(getattr(vital, "alive", False))
    if hasattr(entity, "alive"):
        return bool(getattr(entity, "alive", True))
    return True


def has_unassigned_active_entities(world: WorldState) -> bool:
    for ship in world.ships.values():
        if _entity_is_active(ship) and not entity_system_id(ship):
            return True
    for collection_name in ("drones", "fighters", "projectiles", "projectile_blasts", "bubble_fields"):
        collection = getattr(world, collection_name, {}) or {}
        for entity in collection.values():
            if _entity_is_active(entity) and not entity_system_id(entity):
                return True
    return False


def _runtime_pressure(ship: Any) -> float:
    runtime = getattr(ship, "runtime", None)
    if runtime is None:
        return 0.0
    modules = list(getattr(runtime, "modules", ()) or ())
    active_modules = sum(1 for module in modules if str(getattr(getattr(module, "state", None), "value", "")) in {"ACTIVE", "OVERHEATED"})
    pending = len(getattr(getattr(ship, "combat", None), "module_decision_pending", ()) or ())
    projected = len(getattr(getattr(ship, "combat", None), "projected_targets", {}) or {})
    return float(len(modules)) * 0.75 + float(active_modules) * 2.0 + float(pending) * 0.35 + float(projected)


def active_system_pressures(world: WorldState) -> dict[str, SystemPressure]:
    pressures: dict[str, SystemPressure] = {}

    for ship in world.ships.values():
        if not _entity_is_active(ship):
            continue
        system_id = entity_system_id(ship)
        if not system_id:
            continue
        entry = pressures.setdefault(system_id, SystemPressure(system_id=system_id, pressure=0.0))
        entry.ship_count += 1
        lock_state = getattr(ship, "combat", None)
        lock_pressure = 0.0
        if lock_state is not None:
            lock_pressure += len(getattr(lock_state, "lock_targets", ()) or ()) * 0.4
            lock_pressure += len(getattr(lock_state, "lock_timers", {}) or {}) * 0.4
            lock_pressure += len(getattr(lock_state, "module_cycle_timers", {}) or {}) * 0.35
        entry.pressure += 8.0 + lock_pressure + _runtime_pressure(ship)

    for collection_name, weight, attr_name in (
        ("drones", 2.5, "deployable_count"),
        ("fighters", 4.0, "deployable_count"),
        ("projectiles", 2.0, "projectile_count"),
        ("projectile_blasts", 1.0, "projectile_count"),
        ("bubble_fields", 1.25, "bubble_count"),
    ):
        collection = getattr(world, collection_name, {}) or {}
        for entity in collection.values():
            if not _entity_is_active(entity):
                continue
            system_id = entity_system_id(entity)
            if not system_id:
                continue
            entry = pressures.setdefault(system_id, SystemPressure(system_id=system_id, pressure=0.0))
            setattr(entry, attr_name, int(getattr(entry, attr_name)) + 1)
            entry.pressure += weight

    return pressures


def plan_system_execution(
    world: WorldState,
    config: EngineConfig,
    *,
    cpu_count: int | None = None,
) -> SystemExecutionPlan:
    active = tuple(sorted(active_system_pressures(world).values(), key=lambda item: item.system_id))
    if not active:
        return SystemExecutionPlan()

    configured_workers = int(getattr(config, "parallel_system_workers", 0) or 0)
    available_cpus = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    auto_workers = max(1, available_cpus - 1) if available_cpus > 1 else 1
    max_workers = configured_workers if configured_workers > 0 else auto_workers
    max_workers = max(1, min(max_workers, len(active)))

    try:
        target_pressure = max(1.0, float(getattr(config, "parallel_system_target_pressure", 96.0)))
    except Exception:
        target_pressure = 96.0
    total_pressure = sum(max(0.0, item.pressure) for item in active)
    pressure_workers = max(1, int(math.ceil(total_pressure / target_pressure)))
    worker_count = max(1, min(max_workers, pressure_workers, len(active)))

    bins: list[list[SystemPressure]] = [[] for _ in range(worker_count)]
    bin_pressure = [0.0 for _ in range(worker_count)]
    for item in sorted(active, key=lambda entry: (-entry.pressure, entry.system_id)):
        index = min(range(worker_count), key=lambda idx: (bin_pressure[idx], len(bins[idx])))
        bins[index].append(item)
        bin_pressure[index] += max(0.0, item.pressure)

    groups = tuple(
        SystemExecutionGroup(
            system_ids=tuple(sorted(item.system_id for item in group)),
            pressure=sum(max(0.0, item.pressure) for item in group),
        )
        for group in bins
        if group
    )
    use_processes = bool(getattr(config, "parallel_systems", False)) and len(groups) > 1
    return SystemExecutionPlan(
        active_systems=active,
        groups=groups,
        worker_count=len(groups),
        use_processes=use_processes,
    )


def _system_structure_ids(world: WorldState, system_id: str) -> set[str]:
    structure_ids = {
        structure_id
        for structure_id, structure in world.structures.items()
        if str(getattr(structure, "system_id", "") or "") == system_id
        or str(getattr(structure, "kind", "") or "").upper() == "STARGATE"
    }
    return structure_ids


def _filter_focus_queue(world: WorldState, queue: list[str] | tuple[str, ...], local_entity_ids: set[str]) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_target_id in queue:
        target_id = str(raw_target_id)
        if target_id in seen or target_id not in local_entity_ids:
            continue
        target = world.combat_entity(target_id)
        if target is None or not bool(getattr(getattr(target, "vital", None), "alive", False)):
            continue
        filtered.append(target_id)
        seen.add(target_id)
    return filtered


def build_system_shard(
    world: WorldState,
    system_id: str,
    ship_agents: dict[str, ShipAgent],
) -> SystemShardTask:
    # Correctness-first snapshot runner; high-throughput workers should exchange deltas and own CombatSystem state.
    all_local_ship_ids = {
        ship_id
        for ship_id, ship in world.ships.items()
        if entity_system_id(ship) == system_id
    }
    local_entity_ids = set(all_local_ship_ids)
    local_entity_ids.update(
        entity_id
        for collection in (world.drones, world.fighters)
        for entity_id, entity in collection.items()
        if entity_system_id(entity) == system_id
    )

    shard_squad_keys = {
        squad_key(ship.team, ship.squad_id)
        for ship_id, ship in world.ships.items()
        if ship_id in all_local_ship_ids
    }
    owned_entity_ids = {
        "ships": set(all_local_ship_ids),
        "drones": {
            drone_id
            for drone_id, drone in world.drones.items()
            if entity_system_id(drone) == system_id
        },
        "fighters": {
            fighter_id
            for fighter_id, fighter in world.fighters.items()
            if entity_system_id(fighter) == system_id
        },
        "projectiles": {
            projectile_id
            for projectile_id, projectile in world.projectiles.items()
            if entity_system_id(projectile) == system_id
        },
        "projectile_blasts": {
            blast_id
            for blast_id, blast in world.projectile_blasts.items()
            if entity_system_id(blast) == system_id
        },
        "bubble_fields": {
            field_id
            for field_id, field in world.bubble_fields.items()
            if entity_system_id(field) == system_id
        },
    }

    structures = {
        structure_id: deepcopy(world.structures[structure_id])
        for structure_id in _system_structure_ids(world, system_id)
    }
    shard = WorldState(
        now=float(world.now),
        tick=int(world.tick),
        map_id=str(world.map_id),
        map_name=str(world.map_name),
        map_definition=world.map_definition,
        ships={
            ship_id: deepcopy(ship)
            for ship_id, ship in world.ships.items()
            if entity_system_id(ship) == system_id
        },
        structures=structures,
        intents={},
        squad_leaders={
            key: leader_id
            for key, leader_id in world.squad_leaders.items()
            if key in shard_squad_keys
        },
        squad_leader_locations={
            key: deepcopy(location)
            for key, location in world.squad_leader_locations.items()
            if key in shard_squad_keys
        },
        squad_leader_location_versions={
            key: int(version)
            for key, version in world.squad_leader_location_versions.items()
            if key in shard_squad_keys
        },
        squad_propulsion_commands={
            key: value
            for key, value in world.squad_propulsion_commands.items()
            if key in shard_squad_keys
        },
        squad_leader_speed_limits={
            key: value
            for key, value in world.squad_leader_speed_limits.items()
            if key in shard_squad_keys
        },
        squad_focus_queues={
            key: _filter_focus_queue(world, queue, local_entity_ids)
            for key, queue in world.squad_focus_queues.items()
            if key in shard_squad_keys
        },
        squad_focus_updated_at={
            key: float(value)
            for key, value in world.squad_focus_updated_at.items()
            if key in shard_squad_keys
        },
        drones={
            drone_id: deepcopy(drone)
            for drone_id, drone in world.drones.items()
            if entity_system_id(drone) == system_id
        },
        fighters={
            fighter_id: deepcopy(fighter)
            for fighter_id, fighter in world.fighters.items()
            if entity_system_id(fighter) == system_id
        },
        projectiles={
            projectile_id: deepcopy(projectile)
            for projectile_id, projectile in world.projectiles.items()
            if entity_system_id(projectile) == system_id
        },
        projectile_blasts={
            blast_id: deepcopy(blast)
            for blast_id, blast in world.projectile_blasts.items()
            if entity_system_id(blast) == system_id
        },
        bubble_fields={
            field_id: deepcopy(field)
            for field_id, field in world.bubble_fields.items()
            if entity_system_id(field) == system_id
        },
    )
    shard_agents = {
        ship_id: deepcopy(agent)
        for ship_id, agent in ship_agents.items()
        if ship_id in all_local_ship_ids
    }
    combat = CombatSystem(PyfaBridge())
    return SystemShardTask(
        system_id=system_id,
        world=shard,
        combat=combat,
        ship_agents=shard_agents,
        owned_entity_ids=owned_entity_ids,
        tick=int(world.tick),
    )


def replace_task_combat(task: SystemShardTask, combat: CombatSystem) -> SystemShardTask:
    return SystemShardTask(
        system_id=task.system_id,
        world=task.world,
        combat=combat,
        ship_agents=task.ship_agents,
        owned_entity_ids=task.owned_entity_ids,
        tick=task.tick,
        protocol_version=task.protocol_version,
        random_state=task.random_state,
    )


def _reset_combat_process_state(combat: CombatSystem) -> None:
    combat.attach_event_sink(None)
    combat.logger = None
    combat._timing_wheel = TimingWheel()


def sanitize_combat_for_worker(combat: CombatSystem) -> CombatSystem:
    cloned = combat.clone_for_system(combat._system_id)
    _reset_combat_process_state(cloned)
    return cloned


def _rebuild_timing_wheel(combat: CombatSystem, world: WorldState) -> None:
    combat._timing_wheel = TimingWheel()
    now = float(world.now)
    for ship in world.ships.values():
        if not bool(getattr(getattr(ship, "vital", None), "alive", False)):
            continue
        for target_id, deadline in list(ship.combat.lock_deadlines.items()):
            combat._schedule_lock_deadline(ship, str(target_id), deadline=float(deadline), now=now)
        for module_id, deadline in list(ship.combat.module_cycle_deadlines.items()):
            combat._schedule_module_cycle_deadline(ship, str(module_id), deadline=float(deadline), now=now)
        for module_id, deadline in list(ship.combat.module_ammo_reload_deadlines.items()):
            combat._schedule_module_reload_deadline(ship, str(module_id), deadline=float(deadline), now=now)
        for module_id, deadline in list(ship.combat.module_reactivation_deadlines.items()):
            combat._schedule_module_reactivation_deadline(ship, str(module_id), deadline=float(deadline), now=now)


def _detect_transfer_outs(task: SystemShardTask) -> list[SystemTransferOut]:
    transfers: list[SystemTransferOut] = []
    for collection_name, owned_ids in task.owned_entity_ids.items():
        collection = getattr(task.world, collection_name, {}) or {}
        for entity_id in sorted(owned_ids):
            entity = collection.get(entity_id)
            if entity is None:
                continue
            destination_system_id = entity_system_id(entity)
            if not destination_system_id or destination_system_id == task.system_id:
                continue
            transfers.append(
                SystemTransferOut(
                    collection_name=collection_name,
                    entity_id=str(entity_id),
                    source_system_id=task.system_id,
                    destination_system_id=destination_system_id,
                    entity=entity,
                )
            )
    return transfers


def _run_shard(task: SystemShardTask, config: EngineConfig, step_start: float, step_end: float, dt: float) -> SystemShardResult:
    started = time.perf_counter()
    process_random_state = random.getstate()
    if task.random_state is not None:
        random.setstate(task.random_state)
    world = task.world
    world.now = float(step_end)
    combat = task.combat
    events: list[CombatEvent] = []

    try:
        # Workers collect authority events in memory; only the parent process commits them.
        combat.logger = None
        combat.attach_event_sink(events.append)
        _rebuild_timing_wheel(combat, world)

        perception = PerceptionSystem()
        movement = MovementSystem()
        deployables = DeployableSystem(combat, movement)
        logistics = LogisticsSystem()

        perception.run(world)

        for agent in task.ship_agents.values():
            if agent.ship_id not in world.ships:
                continue
            agent.sense(world)
            agent.think(world)

        deployables.run(world, dt, advance_physics=False, apply_effects=False)

        substep_count = max(1, int(getattr(config, "physics_substeps", 1) or 1))
        base_slice_dt = dt / substep_count
        world.now = float(step_start)

        for slice_index in range(substep_count):
            substep_start = step_start + (base_slice_dt * slice_index)
            if slice_index + 1 >= substep_count:
                substep_end = step_end
            else:
                substep_end = substep_start + base_slice_dt
            slice_dt = max(1e-6, float(substep_end) - float(substep_start))
            world.now = float(substep_end)
            movement.run(world, slice_dt)
            deployables.run_physics(world, slice_dt)

        world.now = float(step_end)
        combat.run(world, dt)
        deployables.run(world, dt, advance_physics=False, apply_effects=True)
        logistics.run(world, dt)
        completed_random_state = random.getstate()
    finally:
        combat.attach_event_sink(None)
        combat.logger = None
        random.setstate(process_random_state)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SystemShardResult(
        system_id=task.system_id,
        world=world,
        combat=combat,
        ship_agents=task.ship_agents,
        events=events,
        elapsed_ms=elapsed_ms,
        owned_entity_ids={key: set(value) for key, value in task.owned_entity_ids.items()},
        transfer_outs=_detect_transfer_outs(task),
        tick=int(task.tick),
        protocol_version=task.protocol_version,
        random_state=completed_random_state,
    )


def run_system_group(tasks: list[SystemShardTask], config: EngineConfig, step_start: float, step_end: float, dt: float) -> SystemGroupResult:
    started = time.perf_counter()
    results = [_run_shard(task, config, step_start, step_end, dt) for task in sorted(tasks, key=lambda item: item.system_id)]
    return SystemGroupResult(results=results, elapsed_ms=(time.perf_counter() - started) * 1000.0)


def parallel_worker_probe() -> tuple[int, int, str]:
    return os.getpid(), SYSTEM_SHARD_PROTOCOL_VERSION, sys.version.split()[0]


def validate_parallel_capability(
    config: EngineConfig,
    tasks: list[SystemShardTask],
    worker_count: int,
) -> None:
    if os.getpid() <= 0:
        raise ParallelCapabilityError("multiprocessing is unavailable")
    import multiprocessing

    if multiprocessing.current_process().daemon:
        raise ParallelCapabilityError("daemon processes cannot create process workers")
    if worker_count <= 1 or len(tasks) <= 1:
        raise ParallelCapabilityError("parallel execution requires at least two worker groups")
    if has_unassigned_active_entities(tasks[0].world):
        raise ParallelCapabilityError("active entities must have a non-empty system_id")
    try:
        pickle.dumps(config)
        for task in tasks:
            pickle.dumps(task)
    except Exception as exc:
        raise ParallelCapabilityError(f"parallel task is not serializable: {exc}") from exc


def _finite_number(value: Any, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ShardResultValidationError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise ShardResultValidationError(f"{label} is not finite")


def validate_shard_result(
    task: SystemShardTask,
    result: SystemShardResult,
    expected_tick: int,
) -> None:
    if result.protocol_version != SYSTEM_SHARD_PROTOCOL_VERSION:
        raise ShardResultValidationError("unsupported shard result protocol")
    if result.system_id != task.system_id:
        raise ShardResultValidationError(
            f"worker returned system {result.system_id!r} for task {task.system_id!r}"
        )
    if result.tick != expected_tick or int(result.world.tick) != expected_tick:
        raise ShardResultValidationError(
            f"worker tick mismatch for {task.system_id!r}: {result.tick}/{result.world.tick} != {expected_tick}"
        )
    if not isinstance(result.combat, CombatSystem):
        raise ShardResultValidationError(f"missing CombatSystem for {task.system_id!r}")
    if result.random_state is None:
        raise ShardResultValidationError(f"missing random state for {task.system_id!r}")
    try:
        random.Random().setstate(result.random_state)
    except Exception as exc:
        raise ShardResultValidationError(f"invalid random state for {task.system_id!r}") from exc

    transfers = {(item.collection_name, item.entity_id): item for item in result.transfer_outs}
    if len(transfers) != len(result.transfer_outs):
        raise ShardResultValidationError(f"duplicate transfer in {task.system_id!r}")
    for transfer in result.transfer_outs:
        if transfer.source_system_id != task.system_id or not str(transfer.destination_system_id).strip():
            raise ShardResultValidationError(f"invalid transfer {transfer.entity_id!r}")

    for collection_name in ("ships", "drones", "fighters", "projectiles", "projectile_blasts", "bubble_fields"):
        collection = getattr(result.world, collection_name, {}) or {}
        for entity_id, entity in collection.items():
            system_id = entity_system_id(entity)
            is_transfer = (collection_name, str(entity_id)) in transfers
            if not system_id:
                raise ShardResultValidationError(f"{collection_name}:{entity_id} has an empty system_id")
            if not is_transfer and system_id != task.system_id:
                raise ShardResultValidationError(
                    f"{collection_name}:{entity_id} escaped shard {task.system_id!r}"
                )
            vital = getattr(entity, "vital", None)
            if vital is not None:
                for name in ("shield", "armor", "structure", "cap"):
                    if hasattr(vital, name):
                        _finite_number(getattr(vital, name), f"{collection_name}:{entity_id}.{name}")
            position = getattr(getattr(entity, "nav", None), "position", None) or getattr(entity, "position", None)
            if position is not None:
                _finite_number(getattr(position, "x", None), f"{collection_name}:{entity_id}.x")
                _finite_number(getattr(position, "y", None), f"{collection_name}:{entity_id}.y")
    _finite_number(result.world.now, f"{task.system_id}.now")

    seen_event_sequences: set[tuple[int, int]] = set()
    for event in result.events:
        key = (int(event.rng_seed), int(event.rng_counter))
        if key in seen_event_sequences:
            raise ShardResultValidationError(f"duplicate event sequence in {task.system_id!r}")
        seen_event_sequences.add(key)
        if event.system_id and event.system_id != task.system_id:
            raise ShardResultValidationError(f"event belongs to unexpected system {event.system_id!r}")
    for ship_id, agent in result.ship_agents.items():
        if ship_id not in result.world.ships or agent.ship_id != ship_id:
            raise ShardResultValidationError(f"agent {ship_id!r} does not belong to shard {task.system_id!r}")


def _merge_collection(
    world_collection: dict[str, Any],
    result_collection: dict[str, Any],
    owned_ids: set[str],
    transfer_out_ids: set[str],
) -> None:
    for entity_id in set(owned_ids):
        if entity_id in transfer_out_ids or entity_id not in result_collection:
            world_collection.pop(entity_id, None)
    for entity_id, entity in result_collection.items():
        if entity_id in transfer_out_ids:
            continue
        if entity_id in world_collection and entity_id not in owned_ids:
            raise DuplicateEntityIdError(f"entity ID collision: {entity_id}")
        world_collection[entity_id] = entity


def _transfer_ids_by_collection(transfers: list[SystemTransferOut]) -> dict[str, set[str]]:
    by_collection: dict[str, set[str]] = {}
    for transfer in transfers:
        by_collection.setdefault(transfer.collection_name, set()).add(transfer.entity_id)
    return by_collection


def _transfer_in_from_out(transfer: SystemTransferOut) -> SystemTransferIn:
    return SystemTransferIn(
        collection_name=transfer.collection_name,
        entity_id=transfer.entity_id,
        source_system_id=transfer.source_system_id,
        destination_system_id=transfer.destination_system_id,
        entity=transfer.entity,
        reason=transfer.reason,
    )


def _apply_transfer_ins(world: WorldState, transfer_ins: list[SystemTransferIn]) -> None:
    for transfer in sorted(
        transfer_ins,
        key=lambda item: (item.source_system_id, item.destination_system_id, item.collection_name, item.entity_id),
    ):
        if not transfer.destination_system_id:
            raise ShardResultValidationError(f"transfer {transfer.entity_id!r} has no destination")
        collection = getattr(world, transfer.collection_name, None)
        if not isinstance(collection, dict):
            raise ShardResultValidationError(f"unknown transfer collection {transfer.collection_name!r}")
        if transfer.entity_id in collection:
            raise DuplicateEntityIdError(f"transfer entity ID collision: {transfer.entity_id}")
        collection[transfer.entity_id] = transfer.entity


def _merge_system_results_in_place(
    world: WorldState,
    results: list[SystemShardResult],
    ship_agents: dict[str, ShipAgent],
) -> tuple[list[CombatEvent], list[SystemTransferIn]]:
    events: list[CombatEvent] = []
    transfer_ins: list[SystemTransferIn] = []
    for result in sorted(results, key=lambda item: item.system_id):
        owned = result.owned_entity_ids
        transfer_ids = _transfer_ids_by_collection(result.transfer_outs)
        _merge_collection(world.ships, result.world.ships, owned.get("ships", set()), transfer_ids.get("ships", set()))
        _merge_collection(world.drones, result.world.drones, owned.get("drones", set()), transfer_ids.get("drones", set()))
        _merge_collection(world.fighters, result.world.fighters, owned.get("fighters", set()), transfer_ids.get("fighters", set()))
        _merge_collection(world.projectiles, result.world.projectiles, owned.get("projectiles", set()), transfer_ids.get("projectiles", set()))
        _merge_collection(
            world.projectile_blasts,
            result.world.projectile_blasts,
            owned.get("projectile_blasts", set()),
            transfer_ids.get("projectile_blasts", set()),
        )
        _merge_collection(world.bubble_fields, result.world.bubble_fields, owned.get("bubble_fields", set()), transfer_ids.get("bubble_fields", set()))
        transfer_ins.extend(_transfer_in_from_out(transfer) for transfer in result.transfer_outs)

        for ship_id, agent in result.ship_agents.items():
            ship_agents[ship_id] = agent
        events.extend(result.events)

    transfer_ins.sort(
        key=lambda item: (item.source_system_id, item.destination_system_id, item.collection_name, item.entity_id)
    )
    _apply_transfer_ins(world, transfer_ins)
    return events, transfer_ins


def prepare_system_merge(
    world: WorldState,
    results: list[SystemShardResult],
    ship_agents: dict[str, ShipAgent],
) -> SystemMergePlan:
    staged_world = deepcopy(world)
    staged_agents = deepcopy(ship_agents)
    events, transfer_ins = _merge_system_results_in_place(staged_world, results, staged_agents)
    return SystemMergePlan(
        world=staged_world,
        ship_agents=staged_agents,
        events=events,
        transfer_ins=transfer_ins,
    )


def commit_system_merge(
    world: WorldState,
    ship_agents: dict[str, ShipAgent],
    plan: SystemMergePlan,
    transfer_sink: list[SystemTransferIn] | None = None,
) -> list[CombatEvent]:
    for world_field in fields(WorldState):
        setattr(world, world_field.name, getattr(plan.world, world_field.name))
    ship_agents.clear()
    ship_agents.update(plan.ship_agents)
    if transfer_sink is not None:
        transfer_sink.extend(plan.transfer_ins)
    return list(plan.events)


def merge_system_results(
    world: WorldState,
    results: list[SystemShardResult],
    ship_agents: dict[str, ShipAgent],
    transfer_sink: list[SystemTransferIn] | None = None,
) -> list[CombatEvent]:
    plan = prepare_system_merge(world, results, ship_agents)
    return commit_system_merge(world, ship_agents, plan, transfer_sink)


def log_group_hotspot(config: EngineConfig, plan: SystemExecutionPlan, elapsed_ms: float) -> None:
    if not bool(getattr(config, "hotspot_logging", False)):
        return
    logger = get_sim_logger(config)
    if logger.disabled:
        return
    log_sim_event(
        logger,
        "hotspot",
        name="engine.system_isolation",
        duration_ms=float(elapsed_ms),
        systems=len(plan.active_systems),
        groups=len(plan.groups),
        processes=bool(plan.use_processes),
    )


__all__ = [
    "DuplicateEntityIdError",
    "ParallelCapabilityError",
    "ShardResultValidationError",
    "SystemExecutionMode",
    "SystemExecutionPlan",
    "SystemMergePlan",
    "SystemPressure",
    "SystemShardResult",
    "SystemShardTask",
    "SystemTransferIn",
    "SystemTransferOut",
    "active_system_pressures",
    "build_system_shard",
    "commit_system_merge",
    "entity_system_id",
    "has_unassigned_active_entities",
    "log_group_hotspot",
    "merge_system_results",
    "normalize_system_namespace",
    "parallel_worker_probe",
    "plan_system_execution",
    "prepare_system_merge",
    "replace_task_combat",
    "run_system_group",
    "sanitize_combat_for_worker",
    "stable_system_seed",
    "validate_parallel_capability",
    "validate_shard_result",
]
