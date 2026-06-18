from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import os
import time
from typing import Any

from .agents import CommanderAgent, ShipAgent
from .config import EngineConfig
from .models import Team
from .pyfa_bridge import PyfaBridge
from .replay.schema import CombatEvent
from .sim_logging import get_sim_logger, log_sim_event
from .systems import CombatSystem, DeployableSystem, LogisticsSystem, MovementSystem, PerceptionSystem
from .timing_wheel import TimingWheel
from .world import WorldState


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
    commanders: list[CommanderAgent]
    ship_agents: dict[str, ShipAgent]
    owned_entity_ids: dict[str, set[str]] = field(default_factory=dict)
    intent_keys: set[str] = field(default_factory=set)
    squad_keys: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SystemShardResult:
    system_id: str
    world: WorldState
    combat: CombatSystem
    ship_agents: dict[str, ShipAgent] = field(default_factory=dict)
    events: list[CombatEvent] = field(default_factory=list)
    elapsed_ms: float = 0.0
    owned_entity_ids: dict[str, set[str]] = field(default_factory=dict)
    intent_keys: set[str] = field(default_factory=set)
    consumed_intent_keys: set[str] = field(default_factory=set)
    squad_keys: set[str] = field(default_factory=set)
    transfer_outs: list[SystemTransferOut] = field(default_factory=list)


@dataclass(slots=True)
class SystemGroupResult:
    results: list[SystemShardResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


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


def _focus_key(team: Team, squad_id: str) -> str:
    return f"{team.value}:{squad_id}"


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
    }
    linked_ids: set[str] = set()
    for structure_id in list(structure_ids):
        structure = world.structures.get(structure_id)
        linked_id = str(getattr(structure, "linked_structure_id", "") or "").strip() if structure is not None else ""
        if linked_id and linked_id in world.structures:
            linked_ids.add(linked_id)
    return structure_ids | linked_ids


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
    commanders: list[CommanderAgent],
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

    local_squad_keys = {
        _focus_key(ship.team, ship.squad_id)
        for ship_id, ship in world.ships.items()
        if ship_id in all_local_ship_ids
    }
    local_squad_ids_by_team: dict[Team, set[str]] = {}
    for ship_id, ship in world.ships.items():
        if ship_id not in all_local_ship_ids:
            continue
        local_squad_ids_by_team.setdefault(ship.team, set()).add(str(ship.squad_id))
    shard_intents = {
        intent_key: deepcopy(intent)
        for intent_key, intent in world.intents.items()
        if intent.squad_id in local_squad_ids_by_team.get(Team.BLUE, set())
        or intent.squad_id in local_squad_ids_by_team.get(Team.RED, set())
        or str(intent_key) in local_squad_keys
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
        intents=shard_intents,
        squad_leaders={
            key: leader_id
            for key, leader_id in world.squad_leaders.items()
            if key in local_squad_keys and str(leader_id) in all_local_ship_ids
        },
        squad_propulsion_commands={
            key: value
            for key, value in world.squad_propulsion_commands.items()
            if key in local_squad_keys
        },
        squad_leader_speed_limits={
            key: value
            for key, value in world.squad_leader_speed_limits.items()
            if key in local_squad_keys
        },
        squad_focus_queues={
            key: _filter_focus_queue(world, queue, local_entity_ids)
            for key, queue in world.squad_focus_queues.items()
            if key in local_squad_keys
        },
        squad_focus_updated_at={
            key: float(value)
            for key, value in world.squad_focus_updated_at.items()
            if key in local_squad_keys
        },
        squad_prelocked_targets={
            key: {
                ship_id: {target_id for target_id in targets if target_id in local_entity_ids}
                for ship_id, targets in by_ship.items()
                if ship_id in all_local_ship_ids
            }
            for key, by_ship in world.squad_prelocked_targets.items()
            if key in local_squad_keys
        },
        squad_prelock_timers={
            key: {
                ship_id: {
                    target_id: float(timer)
                    for target_id, timer in timers.items()
                    if target_id in local_entity_ids
                }
                for ship_id, timers in by_ship.items()
                if ship_id in all_local_ship_ids
            }
            for key, by_ship in world.squad_prelock_timers.items()
            if key in local_squad_keys
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
    shard.squad_prelocked_targets = {
        key: {ship_id: targets for ship_id, targets in by_ship.items() if targets}
        for key, by_ship in shard.squad_prelocked_targets.items()
    }
    shard.squad_prelock_timers = {
        key: {ship_id: timers for ship_id, timers in by_ship.items() if timers}
        for key, by_ship in shard.squad_prelock_timers.items()
    }
    shard_commanders = [
        deepcopy(commander)
        for commander in commanders
        if local_squad_ids_by_team.get(commander.team, set()).intersection(set(commander.squad_ids))
    ]
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
        commanders=shard_commanders,
        ship_agents=shard_agents,
        owned_entity_ids=owned_entity_ids,
        intent_keys=set(shard_intents.keys()),
        squad_keys=set(local_squad_keys),
    )


def replace_task_combat(task: SystemShardTask, combat: CombatSystem) -> SystemShardTask:
    return SystemShardTask(
        system_id=task.system_id,
        world=task.world,
        combat=combat,
        commanders=task.commanders,
        ship_agents=task.ship_agents,
        owned_entity_ids=task.owned_entity_ids,
        intent_keys=task.intent_keys,
        squad_keys=task.squad_keys,
    )


def _reset_combat_process_state(combat: CombatSystem) -> None:
    combat.attach_event_sink(None)
    combat.logger = None
    combat._timing_wheel = TimingWheel()


def sanitize_combat_for_worker(combat: CombatSystem) -> CombatSystem:
    cloned = deepcopy(combat)
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
    world = task.world
    world.now = float(step_end)
    combat = task.combat
    events: list[CombatEvent] = []

    logger = get_sim_logger(config)
    combat.attach_logger(
        logger,
        bool(getattr(config, "detailed_logging", False)),
        float(getattr(config, "log_merge_window_sec", 1.0) or 1.0),
        bool(getattr(config, "hotspot_logging", False)),
    )
    combat.attach_event_sink(events.append)
    _rebuild_timing_wheel(combat, world)

    perception = PerceptionSystem()
    movement = MovementSystem()
    deployables = DeployableSystem(combat, movement)
    logistics = LogisticsSystem()

    perception.run(world)

    for commander in task.commanders:
        commander.think(world)

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

    combat.attach_event_sink(None)
    combat.logger = None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SystemShardResult(
        system_id=task.system_id,
        world=world,
        combat=combat,
        ship_agents=task.ship_agents,
        events=events,
        elapsed_ms=elapsed_ms,
        owned_entity_ids={key: set(value) for key, value in task.owned_entity_ids.items()},
        intent_keys=set(task.intent_keys),
        consumed_intent_keys=set(task.intent_keys) - set(world.intents.keys()),
        squad_keys=set(task.squad_keys),
        transfer_outs=_detect_transfer_outs(task),
    )


def run_system_group(tasks: list[SystemShardTask], config: EngineConfig, step_start: float, step_end: float, dt: float) -> SystemGroupResult:
    started = time.perf_counter()
    results = [_run_shard(task, config, step_start, step_end, dt) for task in tasks]
    return SystemGroupResult(results=results, elapsed_ms=(time.perf_counter() - started) * 1000.0)


def _local_entity_ids(result: SystemShardResult) -> set[str]:
    ids = set(result.world.ships.keys())
    ids.update(result.world.drones.keys())
    ids.update(result.world.fighters.keys())
    return ids


def _merge_focus_state(world: WorldState, result: SystemShardResult) -> None:
    local_ids = _local_entity_ids(result)
    local_focus_keys = {
        _focus_key(ship.team, ship.squad_id)
        for ship in result.world.ships.values()
    }
    for focus_key in local_focus_keys:
        existing = [target_id for target_id in world.squad_focus_queues.get(focus_key, []) if str(target_id) not in local_ids]
        local_queue = [
            target_id
            for target_id in result.world.squad_focus_queues.get(focus_key, [])
            if str(target_id) in local_ids and str(target_id) not in existing
        ]
        merged = existing + local_queue
        if merged:
            world.squad_focus_queues[focus_key] = merged
            local_updated = result.world.squad_focus_updated_at.get(focus_key)
            if local_updated is not None:
                world.squad_focus_updated_at[focus_key] = max(
                    float(world.squad_focus_updated_at.get(focus_key, 0.0) or 0.0),
                    float(local_updated),
                )
        else:
            world.squad_focus_queues.pop(focus_key, None)
            world.squad_focus_updated_at.pop(focus_key, None)

        local_ship_ids = {ship.ship_id for ship in result.world.ships.values() if _focus_key(ship.team, ship.squad_id) == focus_key}
        prelocked = world.squad_prelocked_targets.setdefault(focus_key, {})
        timers = world.squad_prelock_timers.setdefault(focus_key, {})
        for ship_id in local_ship_ids:
            prelocked.pop(ship_id, None)
            timers.pop(ship_id, None)
        for ship_id, targets in result.world.squad_prelocked_targets.get(focus_key, {}).items():
            if ship_id in local_ship_ids and targets:
                prelocked[ship_id] = set(targets)
        for ship_id, target_timers in result.world.squad_prelock_timers.get(focus_key, {}).items():
            if ship_id in local_ship_ids and target_timers:
                timers[ship_id] = dict(target_timers)
        if not prelocked:
            world.squad_prelocked_targets.pop(focus_key, None)
        if not timers:
            world.squad_prelock_timers.pop(focus_key, None)


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
    for transfer in transfer_ins:
        if not transfer.destination_system_id:
            continue
        collection = getattr(world, transfer.collection_name, None)
        if not isinstance(collection, dict):
            continue
        collection[transfer.entity_id] = transfer.entity


def merge_system_results(
    world: WorldState,
    results: list[SystemShardResult],
    ship_agents: dict[str, ShipAgent],
    transfer_sink: list[SystemTransferIn] | None = None,
) -> list[CombatEvent]:
    events: list[CombatEvent] = []
    consumed_intent_keys: set[str] = set()
    transfer_ins: list[SystemTransferIn] = []
    for result in results:
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
        _merge_focus_state(world, result)
        for squad_key in result.squad_keys:
            if squad_key in result.world.squad_leaders:
                world.squad_leaders[squad_key] = result.world.squad_leaders[squad_key]
            if squad_key in result.world.squad_propulsion_commands:
                world.squad_propulsion_commands[squad_key] = bool(result.world.squad_propulsion_commands[squad_key])
            if squad_key in result.world.squad_leader_speed_limits:
                world.squad_leader_speed_limits[squad_key] = float(result.world.squad_leader_speed_limits[squad_key])
        consumed_intent_keys.update(result.consumed_intent_keys)
        events.extend(result.events)

    for intent_key in consumed_intent_keys:
        world.intents.pop(intent_key, None)
    _apply_transfer_ins(world, transfer_ins)
    if transfer_sink is not None:
        transfer_sink.extend(transfer_ins)
    return events


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
    "SystemExecutionPlan",
    "SystemPressure",
    "SystemTransferIn",
    "SystemTransferOut",
    "active_system_pressures",
    "build_system_shard",
    "entity_system_id",
    "has_unassigned_active_entities",
    "log_group_hotspot",
    "merge_system_results",
    "plan_system_execution",
    "replace_task_combat",
    "run_system_group",
    "sanitize_combat_for_worker",
]
