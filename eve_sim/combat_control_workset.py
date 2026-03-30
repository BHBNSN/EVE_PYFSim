from __future__ import annotations

from typing import Any


def runtime_controlled_module_ids(runtime, controlled_entries: tuple[tuple[Any, Any], ...]) -> tuple[str, ...]:
    """Cache the stable controlled-module ordering used by the control loop."""
    cached = runtime.diagnostics.get("runtime_controlled_module_ids")
    if isinstance(cached, tuple) and len(cached) == len(controlled_entries):
        return cached
    controlled_ids = tuple(str(module.module_id) for module, _metadata in controlled_entries)
    runtime.diagnostics["runtime_controlled_module_ids"] = controlled_ids
    return controlled_ids


def runtime_controlled_entry_lookup(
    runtime,
    controlled_entries: tuple[tuple[Any, Any], ...],
    controlled_ids: tuple[str, ...],
) -> dict[str, tuple[Any, Any]]:
    cached = runtime.diagnostics.get("runtime_controlled_entry_lookup")
    cached_signature = runtime.diagnostics.get("runtime_controlled_entry_lookup_signature")
    if isinstance(cached, dict) and cached_signature == controlled_ids:
        return cached
    lookup = {
        str(module.module_id): (module, metadata)
        for module, metadata in controlled_entries
    }
    runtime.diagnostics["runtime_controlled_entry_lookup"] = lookup
    runtime.diagnostics["runtime_controlled_entry_lookup_signature"] = controlled_ids
    return lookup


def runtime_decision_rule_groups(runtime, controlled_entries: tuple[tuple[Any, Any], ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    cached = runtime.diagnostics.get("runtime_decision_rule_groups")
    if isinstance(cached, dict):
        activation = cached.get("activation")
        target = cached.get("target")
        if isinstance(activation, dict) and isinstance(target, dict):
            return cached

    activation_groups: dict[str, list[str]] = {}
    target_groups: dict[str, list[str]] = {}
    for module, metadata in controlled_entries:
        module_id = str(module.module_id)
        decision_rule = metadata.decision_rule
        activation_groups.setdefault(str(decision_rule.activation_mode), []).append(module_id)
        target_groups.setdefault(str(decision_rule.target_mode), []).append(module_id)

    groups = {
        "activation": {key: tuple(values) for key, values in activation_groups.items()},
        "target": {key: tuple(values) for key, values in target_groups.items()},
    }
    runtime.diagnostics["runtime_decision_rule_groups"] = groups
    return groups


def ensure_ship_module_decision_pending(ship, controlled_ids: tuple[str, ...]) -> None:
    # When fit composition changes, rebuild the pending set from the new controlled-module order.
    if ship.combat.module_decision_pending_signature != controlled_ids:
        ship.combat.module_decision_pending_signature = controlled_ids
        ship.combat.module_decision_pending = set(controlled_ids)
        return
    ship.combat.module_decision_pending.intersection_update(controlled_ids)


def ship_candidate_module_ids(ship) -> set[str]:
    pending = ship.combat.module_decision_pending
    return pending if isinstance(pending, set) else set(pending)


def enqueue_control_signal_modules(
    ship,
    decision_rule_groups: dict[str, dict[str, tuple[str, ...]]],
    *,
    propulsion_active: bool,
    recent_enemy_weapon_damage_active: bool,
    focus_changed: bool,
) -> None:
    pending = ship.combat.module_decision_pending
    activation_groups = decision_rule_groups.get("activation", {})
    target_groups = decision_rule_groups.get("target", {})

    if ship.combat.module_decision_propulsion_active is None or ship.combat.module_decision_propulsion_active != propulsion_active:
        pending.update(activation_groups.get("propulsion_command", ()))
        ship.combat.module_decision_propulsion_active = propulsion_active

    if (
        ship.combat.module_decision_recent_enemy_damage_active is None
        or ship.combat.module_decision_recent_enemy_damage_active != recent_enemy_weapon_damage_active
    ):
        pending.update(activation_groups.get("recent_enemy_weapon_damage", ()))
        ship.combat.module_decision_recent_enemy_damage_active = recent_enemy_weapon_damage_active

    if focus_changed:
        # Focus-driven weapon modules can stay asleep while the queue is empty, but they must
        # wake immediately when squad focus changes.
        pending.update(activation_groups.get("weapon_focus_only", ()))
        pending.update(target_groups.get("weapon_focus_prefocus", ()))


def module_keeps_decision_pending(
    ship,
    module,
    *,
    cycle_time: float,
    activation_mode: str,
    target_mode: str,
    propulsion_active: bool,
    recent_enemy_weapon_damage_active: bool,
    has_focus_queue: bool,
) -> bool:
    module_id = str(module.module_id)
    if module.state == module.state.OFFLINE:
        return False
    if cycle_time > 0.0 and float(ship.combat.module_cycle_timers.get(module_id, 0.0) or 0.0) > 0.0:
        return False
    if float(ship.combat.module_reactivation_timers.get(module_id, 0.0) or 0.0) > 0.0:
        return False
    if float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0) > 0.0:
        return False
    if float(ship.combat.module_pending_ammo_reload_timers.get(module_id, 0.0) or 0.0) > 0.0:
        return False
    if activation_mode == "never" and target_mode == "none":
        return False
    if activation_mode == "propulsion_command" and not propulsion_active:
        return False
    if activation_mode == "recent_enemy_weapon_damage" and not recent_enemy_weapon_damage_active:
        return False
    if activation_mode == "weapon_focus_only" and not has_focus_queue:
        return False
    if target_mode == "weapon_focus_prefocus" and not has_focus_queue:
        return False
    return True
