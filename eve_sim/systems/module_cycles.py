from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Any

from ..combat_control_workset import (
    enqueue_control_signal_modules,
    ensure_ship_module_decision_pending,
    module_keeps_decision_pending,
    runtime_controlled_entry_lookup,
    runtime_controlled_module_ids,
    ship_candidate_module_ids,
)
from ..fit_runtime import EffectClass, ModuleState
from ..module_control import normalize_module_manual_mode, normalize_module_target_mode
from ..models import ShipProfile, Team
from ..squad_identity import squad_key
from ..timer_views import sync_deadline_view
from ..timing_wheel import EventType
from ..world import WorldState
from .constants import REPAIR_QUEUE_LAYERS
from .models import CycleTargetSnapshot, ModuleDecisionRule, ModuleStaticMetadata


class ModuleCyclesMixin:
    def _ensure_ship_module_decision_pending(self, ship, runtime) -> None:
        controlled_ids = self._runtime_controlled_module_ids(runtime)
        ensure_ship_module_decision_pending(ship, controlled_ids)

    def _enqueue_ship_control_signal_modules(
        self,
        world: WorldState,
        ship,
        runtime,
        *,
        focus_changed: bool,
        enemy_targets_active: bool = False,
        ally_targets_active: bool = False,
        now: float | None = None,
    ) -> None:
        now_value = self._decision_now(world, now)
        enqueue_control_signal_modules(
            ship,
            self._ship_decision_rule_groups(ship, runtime),
            propulsion_active=bool(ship.nav.propulsion_command_active),
            recent_enemy_weapon_damage_active=(
                (
                    now_value
                    - float(
                        getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9)
                        if getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9) is not None
                        else -1e9
                    )
                )
                <= 30.0
            ),
            enemy_targets_active=enemy_targets_active,
            ally_targets_active=ally_targets_active,
            focus_changed=focus_changed,
        )

    def _ship_candidate_control_entries(self, ship, runtime) -> tuple[tuple[Any, ModuleStaticMetadata], ...]:
        # Keep candidate selection outside the main control loop so the staged active-set refactor
        # can evolve independently from activation semantics.
        controlled_entries = self._runtime_module_buckets(runtime).controlled_entries
        controlled_ids = runtime_controlled_module_ids(runtime, controlled_entries)
        ensure_ship_module_decision_pending(ship, controlled_ids)
        candidate_ids = ship_candidate_module_ids(ship)

        if not candidate_ids:
            return ()
        if len(candidate_ids) >= len(controlled_ids):
            return controlled_entries

        lookup = runtime_controlled_entry_lookup(runtime, controlled_entries, controlled_ids)
        ordered_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        for module_id in controlled_ids:
            if module_id not in candidate_ids:
                continue
            entry = lookup.get(module_id)
            if entry is not None:
                ordered_entries.append(entry)
        return tuple(ordered_entries)

    def _module_keeps_decision_pending(self, ship, module, metadata: ModuleStaticMetadata) -> bool:
        return self._module_keeps_decision_pending_with_context(
            ship,
            module,
            metadata,
            propulsion_active=bool(ship.nav.propulsion_command_active),
            recent_enemy_weapon_damage_active=False,
            enemy_targets_active=True,
            ally_targets_active=True,
            has_focus_queue=False,
        )

    def _module_keeps_decision_pending_with_context(
        self,
        ship,
        module,
        metadata: ModuleStaticMetadata,
        *,
        propulsion_active: bool,
        recent_enemy_weapon_damage_active: bool,
        enemy_targets_active: bool,
        ally_targets_active: bool,
        has_focus_queue: bool,
    ) -> bool:
        module_id = str(module.module_id)
        module_max_state = self._runtime_module_max_state(ship.runtime, module_id)
        if (
            module.state != module.state.ACTIVE
            and self._runtime_state_rank(module_max_state) < self._runtime_state_rank(ModuleState.ACTIVE)
        ):
            # Dynamic pyfa constraints such as scram/group limits only change on refresh events.
            # Sleep blocked modules until a refresh updates the max-state map and re-queues them.
            return False

        manual_mode = self._manual_module_mode(ship, module_id)
        if manual_mode != "auto":
            if module.state == module.state.OFFLINE:
                return False
            if manual_mode == "online":
                return module.state == module.state.ACTIVE
            return True
        decision_rule = self._effective_module_decision_rule(ship, module, metadata)
        return module_keeps_decision_pending(
            ship,
            module,
            cycle_time=metadata.cycle_time,
            activation_mode=decision_rule.activation_mode,
            target_mode=decision_rule.target_mode,
            propulsion_active=propulsion_active,
            recent_enemy_weapon_damage_active=recent_enemy_weapon_damage_active,
            enemy_targets_active=enemy_targets_active,
            ally_targets_active=ally_targets_active,
            has_focus_queue=has_focus_queue,
        )

    @staticmethod
    def _manual_module_mode(ship, module_id: str) -> str:
        raw_modes = getattr(ship.combat, "module_manual_modes", {})
        if not isinstance(raw_modes, dict):
            return "auto"
        return normalize_module_manual_mode(raw_modes.get(str(module_id), "auto"))

    @staticmethod
    def _manual_module_target_mode(ship, module_id: str) -> str:
        raw_modes = getattr(ship.combat, "module_target_modes", {})
        if not isinstance(raw_modes, dict):
            return "auto"
        return normalize_module_target_mode(raw_modes.get(str(module_id), "auto"))

    def _module_target_mode_choices(self, module, metadata: ModuleStaticMetadata) -> tuple[str, ...]:
        if not metadata.has_projected or metadata.is_area_effect:
            return tuple()
        if metadata.target_side == "ally":
            choices: list[str] = []
            if metadata.has_projected_rep:
                choices.append("ally_repair_queue")
            if "ally_nearest" not in choices:
                choices.append("ally_nearest")
            return tuple(choices)
        return ("weapon_focus_prefocus", "enemy_nearest", "enemy_random")

    def _effective_module_decision_rule(self, ship, module, metadata: ModuleStaticMetadata) -> ModuleDecisionRule:
        override = self._manual_module_target_mode(ship, str(module.module_id))
        if override == "auto":
            return metadata.decision_rule
        if override not in self._module_target_mode_choices(module, metadata):
            return metadata.decision_rule
        if override == metadata.decision_rule.target_mode:
            return metadata.decision_rule
        return replace(metadata.decision_rule, target_mode=override)

    @staticmethod
    def _decision_rule_needs_enemy_targets(rule: ModuleDecisionRule, metadata: ModuleStaticMetadata) -> bool:
        if not metadata.has_projected or metadata.is_area_effect:
            return False
        if rule.target_mode in {"weapon_focus_prefocus", "enemy_nearest", "enemy_random"}:
            return True
        if rule.target_mode == "none":
            return False
        return metadata.target_side != "ally"

    @staticmethod
    def _decision_rule_needs_ally_targets(rule: ModuleDecisionRule, metadata: ModuleStaticMetadata) -> bool:
        if not metadata.has_projected or metadata.is_area_effect:
            return False
        if rule.target_mode in {"ally_nearest", "ally_repair_queue"}:
            return True
        if rule.target_mode == "none":
            return False
        return metadata.target_side == "ally"

    def _ship_decision_rule_groups(self, ship, runtime) -> dict[str, dict[str, tuple[str, ...]]]:
        activation_groups: dict[str, list[str]] = {}
        target_groups: dict[str, list[str]] = {}
        for module, metadata in self._runtime_module_buckets(runtime).controlled_entries:
            module_id = str(module.module_id)
            decision_rule = self._effective_module_decision_rule(ship, module, metadata)
            activation_groups.setdefault(str(decision_rule.activation_mode), []).append(module_id)
            target_groups.setdefault(str(decision_rule.target_mode), []).append(module_id)
        return {
            "activation": {key: tuple(values) for key, values in activation_groups.items()},
            "target": {key: tuple(values) for key, values in target_groups.items()},
        }

    def _requested_module_mode(
        self,
        ship,
        module,
        metadata: ModuleStaticMetadata,
        *,
        propulsion_active: bool,
    ) -> str:
        explicit_mode = self._manual_module_mode(ship, str(module.module_id))
        if explicit_mode != "auto":
            return explicit_mode
        if metadata.is_propulsion:
            return "active" if propulsion_active else "online"
        return "auto"

    def _manual_weapon_target(self, world: WorldState, source, module, previous_target_id: str | None) -> str | None:
        for candidate_id in (previous_target_id, getattr(source.combat, "current_target", None)):
            if not candidate_id:
                continue
            target = world.combat_entity(str(candidate_id))
            if target is None or not target.vital.alive or target.team == source.team:
                continue
            if not self._target_within_lock_range(source, target):
                continue
            if not self._module_in_projected_range(source, target, module):
                continue
            if not self._can_target_under_ecm(source, str(candidate_id), self._decision_now(world)):
                continue
            return str(candidate_id)
        return None

    @staticmethod
    def _clear_module_cycle_timer(ship, module_id: str) -> None:
        ship.combat.module_cycle_timers.pop(module_id, None)
        ship.combat.module_cycle_deadlines.pop(module_id, None)

    @staticmethod
    def _clear_module_reload_timer(ship, module_id: str, *, clear_pending: bool = False) -> None:
        ship.combat.module_ammo_reload_timers.pop(module_id, None)
        ship.combat.module_ammo_reload_deadlines.pop(module_id, None)
        if clear_pending:
            ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)

    @staticmethod
    def _clear_module_reactivation_timer(ship, module_id: str) -> None:
        ship.combat.module_reactivation_timers.pop(module_id, None)
        ship.combat.module_reactivation_deadlines.pop(module_id, None)

    def _schedule_timer_deadline(
        self,
        ship,
        key: str,
        *,
        deadline: float,
        now: float,
        deadline_map: dict[str, float],
        view_map: dict[str, float],
        event_type: EventType,
    ) -> None:
        normalized_key = str(key or "")
        if not normalized_key:
            return
        due_at = max(float(now), float(deadline))
        deadline_map[normalized_key] = due_at
        view_map[normalized_key] = max(0.0, due_at - float(now))
        self._timing_wheel.schedule(due_at, event_type, ship.ship_id, normalized_key)

    def _schedule_lock_deadline(self, ship, target_id: str, *, duration: float | None = None, deadline: float | None = None, now: float) -> None:
        due_at = float(deadline) if deadline is not None else float(now) + max(0.0, float(duration or 0.0))
        self._remember_lock_started(ship, target_id, now)
        self._schedule_timer_deadline(
            ship,
            target_id,
            deadline=due_at,
            now=now,
            deadline_map=ship.combat.lock_deadlines,
            view_map=ship.combat.lock_timers,
            event_type=EventType.LOCK_COMPLETE,
        )

    def _schedule_module_cycle_deadline(self, ship, module_id: str, *, duration: float | None = None, deadline: float | None = None, now: float) -> None:
        due_at = float(deadline) if deadline is not None else float(now) + max(0.0, float(duration or 0.0))
        self._schedule_timer_deadline(
            ship,
            module_id,
            deadline=due_at,
            now=now,
            deadline_map=ship.combat.module_cycle_deadlines,
            view_map=ship.combat.module_cycle_timers,
            event_type=EventType.CYCLE_END,
        )

    def _schedule_module_reload_deadline(self, ship, module_id: str, *, duration: float | None = None, deadline: float | None = None, now: float) -> None:
        due_at = float(deadline) if deadline is not None else float(now) + max(0.0, float(duration or 0.0))
        self._schedule_timer_deadline(
            ship,
            module_id,
            deadline=due_at,
            now=now,
            deadline_map=ship.combat.module_ammo_reload_deadlines,
            view_map=ship.combat.module_ammo_reload_timers,
            event_type=EventType.RELOAD_END,
        )

    def _schedule_module_reactivation_deadline(self, ship, module_id: str, *, duration: float | None = None, deadline: float | None = None, now: float) -> None:
        due_at = float(deadline) if deadline is not None else float(now) + max(0.0, float(duration or 0.0))
        self._schedule_timer_deadline(
            ship,
            module_id,
            deadline=due_at,
            now=now,
            deadline_map=ship.combat.module_reactivation_deadlines,
            view_map=ship.combat.module_reactivation_timers,
            event_type=EventType.REACTIVATION_END,
        )

    def _sync_timer_views_for_ship(self, ship, now: float) -> None:
        sync_deadline_view(ship.combat.lock_deadlines, ship.combat.lock_timers, now)
        sync_deadline_view(ship.combat.module_cycle_deadlines, ship.combat.module_cycle_timers, now)
        sync_deadline_view(ship.combat.module_ammo_reload_deadlines, ship.combat.module_ammo_reload_timers, now)
        sync_deadline_view(ship.combat.module_reactivation_deadlines, ship.combat.module_reactivation_timers, now)

    def _prepare_ship_timer_views(self, ship, now: float) -> None:
        self._sync_timer_views_for_ship(ship, now)

    @staticmethod
    def _event_deadline_map(ship, event_type: EventType) -> dict[str, float]:
        if event_type == EventType.LOCK_COMPLETE:
            return ship.combat.lock_deadlines
        if event_type == EventType.CYCLE_END:
            return ship.combat.module_cycle_deadlines
        if event_type == EventType.RELOAD_END:
            return ship.combat.module_ammo_reload_deadlines
        if event_type == EventType.REACTIVATION_END:
            return ship.combat.module_reactivation_deadlines
        return {}

    def _timer_event_is_stale(self, world: WorldState, event) -> bool:
        ship = world.ships.get(str(event.ship_id))
        if ship is None or not ship.vital.alive:
            return True
        key = str(event.module_id or "")
        if not key:
            return True
        deadline_map = self._event_deadline_map(ship, event.event_type)
        current_deadline = deadline_map.get(key)
        if current_deadline is None:
            return True
        return abs(float(current_deadline) - float(event.trigger_time)) > 1e-6

    def _next_timer_event_time(self, world: WorldState) -> float | None:
        while True:
            event = self._timing_wheel.peek_next_event()
            if event is None:
                return None
            if self._timer_event_is_stale(world, event):
                self._timing_wheel.pop_next_event()
                continue
            return float(event.trigger_time)

    def _process_due_timer_events(self, world: WorldState, current_time: float | None = None) -> None:
        due_time = self._decision_now(world, current_time)
        for event in self._timing_wheel.pop_due_events(due_time):
            if self._timer_event_is_stale(world, event):
                continue
            ship = world.ships.get(str(event.ship_id))
            if ship is None:
                continue
            key = str(event.module_id or "")
            if event.event_type == EventType.LOCK_COMPLETE:
                ship.combat.lock_deadlines.pop(key, None)
                ship.combat.lock_timers[key] = 0.0
                continue
            if event.event_type == EventType.CYCLE_END:
                ship.combat.module_cycle_deadlines.pop(key, None)
                ship.combat.module_cycle_timers[key] = 0.0
            elif event.event_type == EventType.RELOAD_END:
                ship.combat.module_ammo_reload_deadlines.pop(key, None)
                ship.combat.module_ammo_reload_timers[key] = 0.0
            elif event.event_type == EventType.REACTIVATION_END:
                ship.combat.module_reactivation_deadlines.pop(key, None)
                ship.combat.module_reactivation_timers[key] = 0.0
            else:
                continue
            ship.combat.module_decision_pending.add(key)

    def request_module_reload(self, ship, module_id: str, reload_seconds: float, *, now: float | None = None) -> None:
        now_value = float(now if now is not None else 0.0)
        reload_time = max(0.0, float(reload_seconds or 0.0))
        if reload_time <= 0.0:
            self._clear_module_reload_timer(ship, module_id, clear_pending=True)
            return
        self._prepare_ship_timer_views(ship, now_value)
        cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module_id, 0.0) or 0.0))
        active_reload_left = max(0.0, float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0))
        if cycle_left > 0.0 or active_reload_left > 0.0:
            ship.combat.module_pending_ammo_reload_timers[module_id] = reload_time
            return
        self._schedule_module_reload_deadline(ship, module_id, duration=reload_time, now=now_value)
        ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)

    def _add_projected_cycle_total(
        self,
        source_ship_id: str,
        module_id: str,
        target_ship_id: str,
        shield_repaired: float,
        armor_repaired: float,
        cap_drained: float,
        em_damage: float,
        thermal_damage: float,
        kinetic_damage: float,
        explosive_damage: float,
        total_damage: float,
    ) -> None:
        key = (source_ship_id, module_id, target_ship_id)
        entry = self._projected_cycle_totals.setdefault(
            key,
            {
                "shield_repaired": 0.0,
                "armor_repaired": 0.0,
                "cap_drained": 0.0,
                "em": 0.0,
                "thermal": 0.0,
                "kinetic": 0.0,
                "explosive": 0.0,
                "total_damage": 0.0,
            },
        )
        entry["shield_repaired"] += max(0.0, float(shield_repaired))
        entry["armor_repaired"] += max(0.0, float(armor_repaired))
        entry["cap_drained"] += max(0.0, float(cap_drained))
        entry["em"] += max(0.0, float(em_damage))
        entry["thermal"] += max(0.0, float(thermal_damage))
        entry["kinetic"] += max(0.0, float(kinetic_damage))
        entry["explosive"] += max(0.0, float(explosive_damage))
        entry["total_damage"] += max(0.0, float(total_damage))

    def _flush_projected_cycle_total(self, world: WorldState, source_ship_id: str, module, target_ship_id: str | None) -> None:
        if not target_ship_id:
            return
        key = (source_ship_id, module.module_id, target_ship_id)
        totals = self._projected_cycle_totals.pop(key, None)
        if not totals:
            return
        if (
            totals["shield_repaired"] <= 0.0
            and totals["armor_repaired"] <= 0.0
            and totals["cap_drained"] <= 0.0
            and totals["total_damage"] <= 0.0
        ):
            return
        source_ship = world.ships.get(source_ship_id)
        target_ship = world.ships.get(target_ship_id)
        self._queue_merged_event(
            "active_module_cycle_effect",
            merge_fields={
                "team": source_ship.team.value if source_ship is not None else "",
                "squad": source_ship.squad_id if source_ship is not None else "",
                "ship_type": source_ship.fit.ship_name if source_ship is not None else "",
                "module": module.module_id,
                "group": module.group,
                "target_type": target_ship.fit.ship_name if target_ship is not None else "",
            },
            sum_fields={
                "shield_repaired": totals["shield_repaired"],
                "armor_repaired": totals["armor_repaired"],
                "cap_drained": totals["cap_drained"],
                "em": totals["em"],
                "thermal": totals["thermal"],
                "kinetic": totals["kinetic"],
                "explosive": totals["explosive"],
                "total_damage": totals["total_damage"],
            },
        )

    @staticmethod
    def _module_cycle_snapshot_key(source_ship_id: str, module_id: str) -> tuple[str, str]:
        return source_ship_id, module_id

    @staticmethod
    def _uses_cycle_start_projected_application(metadata: ModuleStaticMetadata) -> bool:
        if metadata.is_missile_weapon or metadata.is_bomb_launcher:
            return False
        return metadata.is_area_effect or metadata.is_weapon or metadata.has_projected_rep or metadata.is_cap_warfare

    def _mark_projected_cycle_started(self, source_ship_id: str, module_id: str) -> None:
        self._projected_cycle_starts_this_tick.add(self._module_cycle_snapshot_key(source_ship_id, module_id))

    def _projected_cycle_started_this_tick(self, source_ship_id: str, module_id: str) -> bool:
        return self._module_cycle_snapshot_key(source_ship_id, module_id) in self._projected_cycle_starts_this_tick

    def _module_cycle_snapshots_for(self, source_ship_id: str, module_id: str) -> dict[str, CycleTargetSnapshot]:
        return self._module_cycle_target_snapshots.get(self._module_cycle_snapshot_key(source_ship_id, module_id), {})

    def _module_cycle_snapshot_for_target(
        self,
        source_ship_id: str,
        module_id: str,
        target_ship_id: str,
    ) -> CycleTargetSnapshot | None:
        return self._module_cycle_snapshots_for(source_ship_id, module_id).get(target_ship_id)

    def _live_cycle_snapshot_target_ids(
        self,
        world: WorldState,
        source_ship_id: str,
        module_id: str,
        *,
        team: Team | None = None,
        require_runtime: bool = False,
    ) -> tuple[str, ...]:
        retained_ids: list[str] = []
        for target_id in self._module_cycle_snapshots_for(source_ship_id, module_id):
            target = world.combat_entity(target_id)
            if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target):
                continue
            if team is not None and target.team != team:
                continue
            if require_runtime and target.runtime is None:
                continue
            retained_ids.append(str(target_id))
        retained_ids.sort()
        return tuple(retained_ids)

    def _clear_module_cycle_snapshots(self, source_ship_id: str, module_id: str) -> None:
        self._module_cycle_target_snapshots.pop(self._module_cycle_snapshot_key(source_ship_id, module_id), None)

    def _prune_cycle_effect_snapshots(self, world: WorldState) -> None:
        for key in list(self._module_cycle_target_snapshots.keys()):
            source_ship_id, module_id = key
            source = world.ships.get(source_ship_id)
            if source is None or not source.vital.alive or source.runtime is None:
                self._module_cycle_target_snapshots.pop(key, None)
                continue
            module = next((candidate for candidate in source.runtime.modules if candidate.module_id == module_id), None)
            if module is None or module.state != module.state.ACTIVE:
                self._module_cycle_target_snapshots.pop(key, None)

    @staticmethod
    def _effect_uses_cached_strength(effect) -> bool:
        return any(str(key).startswith("damage_") for key in effect.projected_add.keys())

    def _cycle_effect_strength(
        self,
        effect,
        effect_index: int,
        target_snapshot: CycleTargetSnapshot,
    ) -> float:
        if effect_index not in target_snapshot.active_effect_indices:
            return 0.0
        cached = target_snapshot.effect_strengths.get(effect_index)
        if cached is not None:
            return max(0.0, min(1.0, float(cached)))
        return max(0.0, min(1.0, self._projected_strength(effect, target_snapshot.distance)))

    def _capture_module_cycle_snapshots(
        self,
        world: WorldState,
        source,
        module,
        projected_target_id: str | None,
        *,
        area_candidates: list | None = None,
    ) -> None:
        metadata = self._module_static_metadata(module)
        snapshot_key = self._module_cycle_snapshot_key(source.ship_id, module.module_id)
        projected_effects = metadata.projected_effects
        if not projected_effects or metadata.is_missile_weapon or metadata.is_bomb_launcher:
            self._module_cycle_target_snapshots.pop(snapshot_key, None)
            return

        if not metadata.is_area_effect:
            if not projected_target_id:
                self._module_cycle_target_snapshots.pop(snapshot_key, None)
                return
            target = world.combat_entity(projected_target_id)
            if target is None or not target.vital.alive:
                self._module_cycle_target_snapshots.pop(snapshot_key, None)
                return

            distance = source.nav.position.distance_to(target.nav.position)
            target_snapshot = CycleTargetSnapshot(distance=distance)
            for effect_index, effect in projected_effects:
                max_range = self._projected_max_range(effect)
                if max_range > 0.0 and distance > max_range:
                    continue
                strength = self._projected_strength(effect, distance)
                if strength > 0.0:
                    target_snapshot.active_effect_indices.add(effect_index)
                    if self._effect_uses_cached_strength(effect):
                        target_snapshot.effect_strengths[effect_index] = max(0.0, min(1.0, strength))

            if target_snapshot.active_effect_indices:
                self._module_cycle_target_snapshots[snapshot_key] = {target.ship_id: target_snapshot}
            else:
                self._module_cycle_target_snapshots.pop(snapshot_key, None)
            return

        target_snapshots: dict[str, CycleTargetSnapshot] = {}

        if len(projected_effects) == 1:
            effect_index, effect = projected_effects[0]
            for target in self._iter_area_targets_in_range(world, source, module, effect, candidates=area_candidates):
                distance = source.nav.position.distance_to(target.nav.position)
                strength = self._projected_strength(effect, distance)
                if strength <= 0.0:
                    continue
                target_snapshot = CycleTargetSnapshot(distance=distance, active_effect_indices={effect_index})
                if self._effect_uses_cached_strength(effect):
                    target_snapshot.effect_strengths[effect_index] = max(0.0, min(1.0, strength))
                target_snapshots[target.ship_id] = target_snapshot

            if target_snapshots:
                self._module_cycle_target_snapshots[snapshot_key] = target_snapshots
            else:
                self._module_cycle_target_snapshots.pop(snapshot_key, None)
            return

        for _effect_index, effect in projected_effects:
            for target in self._iter_area_targets_in_range(world, source, module, effect, candidates=area_candidates):
                distance = source.nav.position.distance_to(target.nav.position)
                existing = target_snapshots.get(target.ship_id)
                if existing is None:
                    target_snapshots[target.ship_id] = CycleTargetSnapshot(distance=distance)
                else:
                    existing.distance = min(existing.distance, distance)

        if not target_snapshots:
            self._module_cycle_target_snapshots.pop(snapshot_key, None)
            return

        for effect_index, effect in projected_effects:
            max_range = self._projected_max_range(effect)
            for target_snapshot in target_snapshots.values():
                if max_range > 0.0 and target_snapshot.distance > max_range:
                    continue
                strength = self._projected_strength(effect, target_snapshot.distance)
                if strength > 0.0:
                    target_snapshot.active_effect_indices.add(effect_index)
                    if self._effect_uses_cached_strength(effect):
                        target_snapshot.effect_strengths[effect_index] = max(0.0, min(1.0, strength))

        filtered = {
            target_id: snapshot
            for target_id, snapshot in target_snapshots.items()
            if snapshot.active_effect_indices
        }
        if filtered:
            self._module_cycle_target_snapshots[snapshot_key] = filtered
        else:
            self._module_cycle_target_snapshots.pop(snapshot_key, None)

    def _resolve_cap_recharge(self, cap_now: float, cap_max: float, recharge_time: float, dt: float) -> float:
        if cap_max <= 0 or recharge_time <= 0:
            return cap_now
        cap = max(0.0, min(cap_max, cap_now))
        tau = recharge_time / 5.0
        if tau <= 0:
            return cap
        inner = 1.0 + (math.sqrt(max(cap / cap_max, 0.0)) - 1.0) * math.exp(-dt / tau)
        return max(0.0, min(cap_max, (inner * inner) * cap_max))

    @staticmethod
    def _clamp_ship_layer_hp(ship) -> None:
        ship.vital.shield_max = max(1.0, float(ship.vital.shield_max))
        ship.vital.armor_max = max(1.0, float(ship.vital.armor_max))
        ship.vital.structure_max = max(1.0, float(ship.vital.structure_max))
        ship.vital.shield = max(0.0, min(float(ship.vital.shield), ship.vital.shield_max))
        ship.vital.armor = max(0.0, min(float(ship.vital.armor), ship.vital.armor_max))
        ship.vital.structure = max(0.0, min(float(ship.vital.structure), ship.vital.structure_max))

    def _sync_vital_max_with_profile(self, ship, profile: ShipProfile) -> None:
        previous_values = {
            "shield": (float(ship.vital.shield), float(ship.vital.shield_max)),
            "armor": (float(ship.vital.armor), float(ship.vital.armor_max)),
            "structure": (float(ship.vital.structure), float(ship.vital.structure_max)),
        }
        ship.vital.shield_max = max(1.0, float(getattr(profile, "shield_hp", ship.vital.shield_max) or ship.vital.shield_max))
        ship.vital.armor_max = max(1.0, float(getattr(profile, "armor_hp", ship.vital.armor_max) or ship.vital.armor_max))
        ship.vital.structure_max = max(1.0, float(getattr(profile, "structure_hp", ship.vital.structure_max) or ship.vital.structure_max))
        self._clamp_ship_layer_hp(ship)
        changed_layers = [
            layer
            for layer in REPAIR_QUEUE_LAYERS
            if abs(self._ship_layer_values(ship, layer)[0] - previous_values[layer][0]) > 1e-6
            or abs(self._ship_layer_values(ship, layer)[1] - previous_values[layer][1]) > 1e-6
        ]
        if changed_layers:
            self._mark_team_repair_queues_dirty(ship.team, *changed_layers)

    @staticmethod
    def _projected_max_range(effect) -> float:
        if effect.falloff_m > 0.0:
            return max(0.0, effect.range_m) + 3.0 * max(0.0, effect.falloff_m)
        return max(0.0, effect.range_m)

    def _projected_strength(self, effect, distance: float) -> float:
        if effect.range_m > 0 or effect.falloff_m > 0:
            return self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
        return 1.0

    @staticmethod
    def _ship_system_id(ship) -> str:
        nav = getattr(ship, "nav", None)
        if nav is None:
            return ""
        system_id = getattr(nav, "system_id", "")
        return system_id if isinstance(system_id, str) else str(system_id or "")

    @classmethod
    def _same_system(cls, source, target) -> bool:
        source_system_id = cls._ship_system_id(source)
        target_system_id = cls._ship_system_id(target)
        if source_system_id and target_system_id:
            return source_system_id == target_system_id
        return True

    def _module_projected_max_range(
        self,
        module,
        metadata: ModuleStaticMetadata | None = None,
    ) -> float | None:
        if metadata is not None:
            if not metadata.has_projected:
                return None
            return metadata.projected_max_range

        has_projected = False
        projected_max_range = 0.0
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            has_projected = True
            effect_max_range = self._projected_max_range(effect)
            if effect_max_range <= 0.0:
                return 0.0
            projected_max_range = max(projected_max_range, effect_max_range)
        return projected_max_range if has_projected else None

    @staticmethod
    def _prefocus_fire_probability(ship) -> float:
        level = str(getattr(ship.quality.level, "value", "REGULAR")).upper()
        if level == "ELITE":
            base = 0.38
        elif level == "IRREGULAR":
            base = 0.10
        else:
            base = 0.22
        configured = float(getattr(ship.quality, "ignore_order_probability", 0.0) or 0.0)
        return max(0.0, min(1.0, max(base, configured)))

    @staticmethod
    def _sample_weapon_fire_delay(ship) -> float:
        base_delay = float(getattr(ship.quality, "reaction_delay", 0.0) or 0.0)
        if base_delay <= 0.0:
            level = str(getattr(ship.quality.level, "value", "REGULAR")).upper()
            if level == "ELITE":
                base_delay = random.uniform(0.05, 0.30)
            elif level == "IRREGULAR":
                base_delay = random.uniform(0.55, 1.60)
            else:
                base_delay = random.uniform(0.20, 0.85)
        jitter = max(0.0, float(getattr(ship.quality, "formation_jitter", 0.0) or 0.0))
        if jitter > 0.0:
            base_delay *= 1.0 + random.uniform(0.0, jitter)
        return max(0.0, base_delay)

    def _sync_weapon_fire_delay(self, ship, previous_target_id: str | None, new_target_id: str | None, now: float) -> None:
        if not new_target_id:
            ship.combat.fire_delay_timers.clear()
            return
        if previous_target_id == new_target_id:
            for stale in [target_id for target_id in ship.combat.fire_delay_timers if target_id != new_target_id]:
                ship.combat.fire_delay_timers.pop(stale, None)
            return
        delay = self._sample_weapon_fire_delay(ship)
        ship.combat.fire_delay_timers[new_target_id] = float(now) + delay
        for stale in [target_id for target_id in ship.combat.fire_delay_timers if target_id != new_target_id]:
            ship.combat.fire_delay_timers.pop(stale, None)

    @staticmethod
    def _weapon_fire_delay_ready(ship, target_id: str | None, now: float) -> bool:
        if not target_id:
            return False
        ready_at = ship.combat.fire_delay_timers.get(target_id)
        if ready_at is None:
            return True
        return float(now) >= float(ready_at)

    def _candidates_in_projected_range(
        self,
        source,
        module,
        candidates: list,
        *,
        metadata: ModuleStaticMetadata | None = None,
        source_system_id: str | None = None,
        candidate_cache: dict[tuple[str, float | None], list] | None = None,
        cache_bucket: str,
    ) -> list:
        projected_max_range = self._module_projected_max_range(module, metadata)
        cache_key = (cache_bucket, projected_max_range)
        if candidate_cache is not None:
            cached = candidate_cache.get(cache_key)
            if isinstance(cached, list):
                return cached

        if projected_max_range is None or projected_max_range <= 0.0:
            filtered = list(candidates)
        else:
            filtered = [
                candidate
                for candidate in candidates
                if source.nav.position.distance_to(candidate.nav.position) <= projected_max_range
            ]

        if candidate_cache is not None:
            candidate_cache[cache_key] = filtered
        return filtered

    def _module_has_area_enemies_in_range(self, world: WorldState, source, module) -> bool:
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            for candidate in self._iter_area_targets_in_range(world, source, module, effect):
                if candidate.team != source.team:
                    return True
        source_system_id = self._ship_system_id(source)
        for effect in self._module_bubble_effects(module):
            radius_m = max(0.0, float(effect.local_add.get("bubble_radius_m", 0.0) or 0.0))
            if radius_m <= 0.0:
                continue
            for candidate in world.ships.values():
                if candidate.ship_id == source.ship_id or candidate.team == source.team:
                    continue
                if not candidate.vital.alive or self._ship_in_warp(candidate):
                    continue
                if source_system_id and self._ship_system_id(candidate) != source_system_id:
                    continue
                if source.nav.position.distance_to(candidate.nav.position) <= radius_m:
                    return True
        return False

    @staticmethod
    def _ship_id_in_pool(ship_id: str, pool: list) -> bool:
        return any(candidate.ship_id == ship_id and candidate.vital.alive for candidate in pool)

    def _ally_candidates_in_projected_range(
        self,
        source,
        module,
        allies_pool: list,
        *,
        metadata: ModuleStaticMetadata | None = None,
        source_system_id: str | None = None,
        candidate_cache: dict[tuple[str, float | None], list] | None = None,
    ) -> list:
        return [
            ally
            for ally in self._candidates_in_projected_range(
                source,
                module,
                allies_pool,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
                cache_bucket="ally",
            )
            if ally.ship_id != source.ship_id
            and not self._ship_disallows_assistance(ally)
        ]

    def _can_reuse_projected_target(
        self,
        world: WorldState,
        source,
        module,
        metadata: ModuleStaticMetadata,
        rule: ModuleDecisionRule,
        target_id: str | None,
        allies_pool: list,
        enemies_pool: list,
        ally_ids: set[str],
        enemy_ids: set[str],
        *,
        source_system_id: str,
    ) -> bool:
        if not target_id:
            return False

        target = world.combat_entity(target_id)
        if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target):
            return False
        if not self._target_within_lock_range(source, target):
            return False
        if not self._module_in_projected_range(source, target, module, metadata=metadata, source_system_id=source_system_id):
            return False
        if not self._can_target_under_ecm(source, target_id, self._decision_now(world)):
            return False

        if rule.target_mode == "weapon_focus_prefocus":
            focus_queue = world.squad_focus_queues.get(squad_key(source.team, source.squad_id), [])
            if not focus_queue:
                return False
            allowed_ids: set[str] = {str(focus_queue[0])}
            if len(focus_queue) > 1:
                allowed_ids.add(str(focus_queue[1]))
            return target_id in allowed_ids and target_id in enemy_ids

        if rule.target_mode == "ally_repair_queue":
            return target_id == self._select_repair_queue_target(world, source, module, metadata)

        if rule.target_mode == "ally_nearest":
            return (
                target_id != source.ship_id
                and target_id in ally_ids
                and (target is not None)
                and not self._ship_disallows_assistance(target)
            )

        if rule.target_mode in {"enemy_random", "enemy_nearest"}:
            return target_id in enemy_ids

        side = metadata.target_side
        if side == "ally":
            if target_id == source.ship_id:
                return False
            return target_id in ally_ids and (target is not None) and not self._ship_disallows_assistance(target)
        return target_id in enemy_ids

    def _select_enemy_random_in_range(
        self,
        source,
        module,
        enemies_pool: list,
        existing_target_id: str | None,
        *,
        metadata: ModuleStaticMetadata,
        source_system_id: str,
        candidate_cache: dict[tuple[str, float | None], list] | None,
    ) -> str | None:
        candidates = self._candidates_in_projected_range(
            source,
            module,
            enemies_pool,
            metadata=metadata,
            source_system_id=source_system_id,
            candidate_cache=candidate_cache,
            cache_bucket="enemy",
        )
        if not candidates:
            return None
        if existing_target_id and any(enemy.ship_id == existing_target_id for enemy in candidates):
            return existing_target_id
        return random.choice(candidates).ship_id

    def _select_enemy_nearest_in_range(
        self,
        source,
        module,
        enemies_pool: list,
        existing_target_id: str | None,
        *,
        metadata: ModuleStaticMetadata,
        source_system_id: str,
        candidate_cache: dict[tuple[str, float | None], list] | None,
    ) -> str | None:
        candidates = self._candidates_in_projected_range(
            source,
            module,
            enemies_pool,
            metadata=metadata,
            source_system_id=source_system_id,
            candidate_cache=candidate_cache,
            cache_bucket="enemy",
        )
        if not candidates:
            return None
        if existing_target_id and any(enemy.ship_id == existing_target_id for enemy in candidates):
            return existing_target_id
        return min(candidates, key=lambda enemy: source.nav.position.distance_to(enemy.nav.position)).ship_id

    def _select_ally_nearest_in_range(
        self,
        source,
        module,
        allies_pool: list,
        existing_target_id: str | None,
        *,
        metadata: ModuleStaticMetadata,
        source_system_id: str,
        candidate_cache: dict[tuple[str, float | None], list] | None,
    ) -> str | None:
        candidates = self._ally_candidates_in_projected_range(
            source,
            module,
            allies_pool,
            metadata=metadata,
            source_system_id=source_system_id,
            candidate_cache=candidate_cache,
        )
        if not candidates:
            return None
        if existing_target_id and any(ally.ship_id == existing_target_id for ally in candidates):
            return existing_target_id
        return min(candidates, key=lambda ally: source.nav.position.distance_to(ally.nav.position)).ship_id

    def _select_weapon_focus_target(self, world: WorldState, source, module, existing_target_id: str | None) -> str | None:
        focus_queue = world.squad_focus_queues.get(squad_key(source.team, source.squad_id), [])
        if not focus_queue:
            return None

        valid_focus_id: str | None = None
        valid_prefocus_id: str | None = None
        for queue_index, raw_target_id in enumerate(focus_queue[:2]):
            target_id = str(raw_target_id)
            target = world.combat_entity(target_id)
            if target is None or not target.vital.alive or target.team == source.team:
                continue
            if not self._target_within_lock_range(source, target):
                continue
            if not self._module_in_projected_range(source, target, module):
                continue
            if queue_index == 0:
                valid_focus_id = target_id
            else:
                valid_prefocus_id = target_id

        valid_ids = {candidate_id for candidate_id in (valid_focus_id, valid_prefocus_id) if candidate_id}
        if not valid_ids:
            return None
        if existing_target_id in valid_ids:
            return existing_target_id

        if valid_focus_id and valid_prefocus_id:
            use_prefocus = random.random() < self._prefocus_fire_probability(source)
            return valid_prefocus_id if use_prefocus else valid_focus_id
        return valid_focus_id or valid_prefocus_id

    def _should_activate_module(self, world: WorldState, ship, module, rule: ModuleDecisionRule, target_id: str | None) -> bool:
        cap_ratio = self._cap_ratio(ship)
        hp_ratio = self._hp_ratio(ship)

        if rule.activation_mode == "always":
            return True
        if rule.activation_mode == "never":
            return False
        if cap_ratio < max(0.0, float(rule.cap_threshold)):
            return False
        if rule.activation_mode == "propulsion_command":
            return bool(ship.nav.propulsion_command_active)
        if rule.activation_mode == "cap_min":
            return cap_ratio >= max(0.0, float(rule.cap_threshold))
        if rule.activation_mode == "cap_max":
            return cap_ratio <= max(0.0, float(rule.cap_threshold))
        if rule.activation_mode == "cap_or_low_hp":
            return cap_ratio >= max(0.0, float(rule.cap_threshold)) or hp_ratio < 0.5
        if rule.activation_mode == "recent_enemy_weapon_damage":
            raw_last_hit_at = getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9)
            last_hit_at = float(raw_last_hit_at if raw_last_hit_at is not None else -1e9)
            return (self._decision_now(world) - last_hit_at) <= 30.0
        if rule.activation_mode == "enemy_in_area":
            return self._module_has_area_enemies_in_range(world, ship, module)
        if rule.activation_mode == "weapon_focus_only":
            if not target_id:
                return False
            return self._weapon_fire_delay_ready(ship, target_id, self._decision_now(world))
        return True

    def _select_projected_target(
        self,
        world: WorldState,
        source,
        module,
        metadata: ModuleStaticMetadata,
        allies_pool: list,
        enemies_pool: list,
        rule: ModuleDecisionRule,
        existing_target_id: str | None,
        *,
        source_system_id: str,
        candidate_cache: dict[tuple[str, float | None], list] | None,
        target_cache: dict[tuple[str, str, float | None], str | None] | None,
    ) -> str | None:
        # Central target selector: each target_mode maps to a reusable selection helper.
        if rule.target_mode == "none":
            return None
        cache_key = (
            str(rule.target_mode),
            str(metadata.target_side),
            metadata.projected_max_range,
        )
        if target_cache is not None and rule.target_mode != "enemy_random":
            cached_target = target_cache.get(cache_key)
            if cache_key in target_cache:
                return cached_target
        if rule.target_mode == "weapon_focus_prefocus":
            selected_target = self._select_weapon_focus_target(world, source, module, existing_target_id)
        elif rule.target_mode == "ally_repair_queue":
            selected_target = self._select_repair_queue_target(world, source, module, metadata)
        elif rule.target_mode == "ally_nearest":
            selected_target = self._select_ally_nearest_in_range(
                source,
                module,
                allies_pool,
                existing_target_id,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
            )
        elif rule.target_mode == "enemy_random":
            selected_target = self._select_enemy_random_in_range(
                source,
                module,
                enemies_pool,
                existing_target_id,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
            )
        elif rule.target_mode == "enemy_nearest":
            selected_target = self._select_enemy_nearest_in_range(
                source,
                module,
                enemies_pool,
                existing_target_id,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
            )
        elif metadata.target_side == "ally":
            selected_target = self._select_ally_nearest_in_range(
                source,
                module,
                allies_pool,
                existing_target_id,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
            )
        else:
            selected_target = self._select_enemy_nearest_in_range(
                source,
                module,
                enemies_pool,
                existing_target_id,
                metadata=metadata,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
            )
        if target_cache is not None and rule.target_mode != "enemy_random":
            target_cache[cache_key] = selected_target
        return selected_target

    def _resolve_projected_target_id(
        self,
        world: WorldState,
        ship,
        module,
        metadata: ModuleStaticMetadata,
        decision_rule: ModuleDecisionRule,
        *,
        requested_mode: str,
        previous_projected_target: str | None,
        allies_pool: list,
        enemies_alive: list,
        ally_ids: set[str],
        enemy_ids: set[str],
        force_target_reselect: bool,
        synced_weapon_fire_delay_pairs: set[tuple[str | None, str | None]],
        now: float,
        source_system_id: str,
        candidate_cache: dict[tuple[str, float | None], list] | None,
        target_cache: dict[tuple[str, str, float | None], str | None] | None,
    ) -> str | None:
        module_id = str(module.module_id)
        if (
            requested_mode == "active"
            and metadata.is_weapon
            and not metadata.is_area_effect
            and self._manual_module_target_mode(ship, module_id) == "auto"
        ):
            projected_target_id = self._manual_weapon_target(world, ship, module, previous_projected_target)
        elif (not force_target_reselect) and self._can_reuse_projected_target(
            world,
            ship,
            module,
            metadata,
            decision_rule,
            previous_projected_target,
            allies_pool,
            enemies_alive,
            ally_ids,
            enemy_ids,
            source_system_id=source_system_id,
        ):
            projected_target_id = previous_projected_target
        else:
            projected_target_id = self._select_projected_target(
                world,
                ship,
                module,
                metadata,
                allies_pool=allies_pool,
                enemies_pool=enemies_alive,
                rule=decision_rule,
                existing_target_id=None,
                source_system_id=source_system_id,
                candidate_cache=candidate_cache,
                target_cache=target_cache,
            )

        if decision_rule.target_mode == "weapon_focus_prefocus":
            delay_pair = (previous_projected_target, projected_target_id)
            if delay_pair not in synced_weapon_fire_delay_pairs:
                self._sync_weapon_fire_delay(
                    ship,
                    previous_target_id=previous_projected_target,
                    new_target_id=projected_target_id,
                    now=now,
                )
                synced_weapon_fire_delay_pairs.add(delay_pair)
        return projected_target_id

    def _apply_module_reload_gating(
        self,
        ship,
        module,
        module_id: str,
        *,
        desired_active: bool,
        now: float,
    ) -> bool:
        ammo_reload_left = max(
            0.0,
            float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0),
        )
        active_reload_timer_present = module_id in ship.combat.module_ammo_reload_timers
        if ammo_reload_left > 0.0:
            if module_id not in ship.combat.module_ammo_reload_deadlines:
                self._schedule_module_reload_deadline(
                    ship,
                    module_id,
                    duration=ammo_reload_left,
                    now=now,
                )
            desired_active = False
        elif active_reload_timer_present and module_id not in ship.combat.module_ammo_reload_deadlines:
            self._clear_module_reload_timer(ship, module_id)
            if module.charge_capacity > 0:
                module.charge_remaining = float(module.charge_capacity)

        if module_id in ship.combat.module_ammo_reload_timers:
            desired_active = False

        pending_ammo_reload_left = max(
            0.0,
            float(ship.combat.module_pending_ammo_reload_timers.get(module_id, 0.0) or 0.0),
        )
        active_ammo_reload_left = max(
            0.0,
            float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0),
        )
        current_cycle_left = max(
            0.0,
            float(ship.combat.module_cycle_timers.get(module_id, 0.0) or 0.0),
        )
        if active_ammo_reload_left <= 0.0 and pending_ammo_reload_left > 0.0 and current_cycle_left <= 0.0:
            self._schedule_module_reload_deadline(
                ship,
                module_id,
                duration=pending_ammo_reload_left,
                now=now,
            )
            ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)
            desired_active = False

        if module.charge_capacity > 0 and module.charge_rate > 0.0 and module.charge_remaining <= 0.0:
            if module_id not in ship.combat.module_ammo_reload_timers:
                auto_reload_time = max(0.0, float(module.charge_reload_time))
                if auto_reload_time > 0.0:
                    self._schedule_module_reload_deadline(
                        ship,
                        module_id,
                        duration=auto_reload_time,
                        now=now,
                    )
                else:
                    module.charge_remaining = float(module.charge_capacity)
            desired_active = False

        return desired_active

    def _apply_module_reactivation_gating(
        self,
        ship,
        module_id: str,
        *,
        desired_active: bool,
        now: float,
    ) -> bool:
        cooldown_left = ship.combat.module_reactivation_timers.get(module_id)
        if cooldown_left is None:
            return desired_active
        if module_id in ship.combat.module_reactivation_deadlines:
            if float(cooldown_left) > 0.0:
                return False
            self._clear_module_reactivation_timer(ship, module_id)
            return desired_active

        cooldown_left = max(0.0, float(cooldown_left) or 0.0)
        if cooldown_left > 0.0:
            self._schedule_module_reactivation_deadline(
                ship,
                module_id,
                duration=cooldown_left,
                now=now,
            )
            return False

        self._clear_module_reactivation_timer(ship, module_id)
        return desired_active

    def _update_module_states(self, world: WorldState, dt: float, now: float | None = None) -> bool:
        alive_by_team_system: dict[tuple[Team, str], list] = {}
        for candidate in world.ships.values():
            if candidate.vital.alive and not self._ship_hidden_from_targeting(candidate, now):
                system_id = self._ship_system_id(candidate)
                alive_by_team_system.setdefault((candidate.team, system_id), []).append(candidate)

        changed_focus_keys = self._changed_focus_queues(world)
        pyfa_remote_inputs_dirty = False
        now_value = self._decision_now(world, now)

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            runtime = ship.runtime
            if self._reconcile_external_module_state_changes(world, ship, runtime):
                pyfa_remote_inputs_dirty = True
            self._prepare_ship_timer_views(ship, now_value)
            if self._ship_combat_suppressed(ship, now_value):
                self._clear_ship_warp_engagement_state(ship, runtime)
                continue

            self._ensure_ship_module_decision_pending(ship, runtime)
            all_controlled_entries = self._runtime_module_buckets(runtime).controlled_entries
            if not all_controlled_entries:
                continue
            focus_key = squad_key(ship.team, ship.squad_id)
            focus_queue = tuple(str(target_id) for target_id in world.squad_focus_queues.get(focus_key, []))
            has_focus_queue = bool(focus_queue)
            propulsion_active = bool(ship.nav.propulsion_command_active)
            recent_enemy_weapon_damage_active = (
                (
                    now_value
                    - float(
                        getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9)
                        if getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9) is not None
                        else -1e9
                    )
                )
                <= 30.0
            )
            decision_rules_by_module_id: dict[str, ModuleDecisionRule] = {}
            include_allies = False
            include_enemies = False
            for module, metadata in all_controlled_entries:
                if not metadata.has_projected or metadata.is_area_effect:
                    continue
                decision_rule = self._effective_module_decision_rule(ship, module, metadata)
                decision_rules_by_module_id[str(module.module_id)] = decision_rule
                if self._decision_rule_needs_ally_targets(decision_rule, metadata):
                    include_allies = True
                if self._decision_rule_needs_enemy_targets(decision_rule, metadata):
                    include_enemies = True

            allies_pool, enemies_alive, ally_ids, enemy_ids = self._ship_target_candidate_pools(
                world,
                ship,
                focus_queue=focus_queue,
                include_allies=include_allies,
                include_enemies=include_enemies,
            )
            source_system_id = self._ship_system_id(ship)
            enemy_targets_active = include_enemies and bool(enemies_alive)
            ally_targets_active = include_allies and any(
                ally.ship_id != ship.ship_id and not self._ship_disallows_assistance(ally)
                for ally in allies_pool
            )
            force_target_reselect = focus_key in changed_focus_keys
            self._enqueue_ship_control_signal_modules(
                world,
                ship,
                runtime,
                focus_changed=force_target_reselect,
                enemy_targets_active=enemy_targets_active,
                ally_targets_active=ally_targets_active,
                now=now_value,
            )
            local_signature_dirty = False
            active_pyfa_remote_inputs_dirty = False
            synced_weapon_fire_delay_pairs: set[tuple[str | None, str | None]] = set()
            projected_candidate_cache: dict[tuple[str, float | None], list] = {}
            projected_target_cache: dict[tuple[str, str, float | None], str | None] = {}
            lock_ready_cache: dict[str, bool] = {}
            next_pending_modules: set[str] = set()

            controlled_entries = self._ship_candidate_control_entries(ship, runtime)
            if not controlled_entries:
                continue

            for module, metadata in controlled_entries:
                module_id = str(module.module_id)
                if module.state == module.state.ACTIVE:
                    active_timer = ship.combat.module_cycle_timers.get(module_id)
                    if active_timer is not None and module_id in ship.combat.module_cycle_deadlines and float(active_timer) > 0.0:
                        continue

                if module.state == module.state.OFFLINE:
                    if self._module_affects_pyfa_remote_inputs(module) and (
                        module_id in ship.combat.projected_targets
                        or module_id in ship.combat.module_cycle_timers
                        or bool(self._module_cycle_snapshots_for(ship.ship_id, module_id))
                    ):
                        pyfa_remote_inputs_dirty = True
                    self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                    self._clear_module_cycle_timer(ship, module_id)
                    self._clear_module_reactivation_timer(ship, module_id)
                    self._clear_module_reload_timer(ship, module_id, clear_pending=True)
                    ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)
                    continue

                previous_state = module.state
                previous_projected_target = ship.combat.projected_targets.get(module_id)
                active_timer = ship.combat.module_cycle_timers.get(module_id) if module.state == module.state.ACTIVE else None

                active_effects = metadata.active_effects
                if not active_effects:
                    if previous_state == module.state.ACTIVE:
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                    self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                    module.state = module.state.ONLINE
                    self._clear_module_cycle_timer(ship, module_id)
                    self._clear_module_reactivation_timer(ship, module_id)
                    self._clear_module_reload_timer(ship, module_id, clear_pending=True)
                    ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)
                    ship.combat.projected_targets.pop(module_id, None)
                    continue

                cycle_cost = metadata.cycle_cost
                cycle_time = metadata.cycle_time
                reactivation_delay = metadata.reactivation_delay

                if module.state == module.state.ACTIVE and cycle_time > 0:
                    if active_timer is not None:
                        if module_id in ship.combat.module_cycle_deadlines:
                            if float(active_timer) > 0.0:
                                continue
                        else:
                            timer_left = max(0.0, float(active_timer) or 0.0)
                            if timer_left > 0:
                                self._schedule_module_cycle_deadline(
                                    ship,
                                    module_id,
                                    duration=timer_left,
                                    now=now_value,
                                )
                                continue
                        self._clear_module_cycle_timer(ship, module_id)
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                        self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                        if reactivation_delay > 0.0:
                            self._schedule_module_reactivation_deadline(
                                ship,
                                module_id,
                                duration=reactivation_delay,
                                now=now_value,
                            )
                        pending_ammo_reload = max(
                            0.0,
                            float(ship.combat.module_pending_ammo_reload_timers.get(module_id, 0.0) or 0.0),
                        )

                        if module.charge_capacity > 0 and module.charge_rate > 0.0:
                            module.charge_remaining = max(0.0, float(module.charge_remaining) - float(module.charge_rate))
                            if module.charge_remaining <= 0.0:
                                module.charge_remaining = 0.0
                                if pending_ammo_reload <= 0.0:
                                    auto_reload_time = max(0.0, float(module.charge_reload_time))
                                    if auto_reload_time > 0.0:
                                        self._schedule_module_reload_deadline(
                                            ship,
                                            module_id,
                                            duration=auto_reload_time,
                                            now=now_value,
                                        )
                                    else:
                                        module.charge_remaining = float(module.charge_capacity)

                        if pending_ammo_reload > 0.0:
                            self._schedule_module_reload_deadline(
                                ship,
                                module_id,
                                duration=pending_ammo_reload,
                                now=now_value,
                            )
                            ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)

                decision_rule = decision_rules_by_module_id.get(module_id)
                if decision_rule is None:
                    decision_rule = self._effective_module_decision_rule(ship, module, metadata)
                requested_mode = self._requested_module_mode(
                    ship,
                    module,
                    metadata,
                    propulsion_active=propulsion_active,
                )
                desired_active = False
                projected_target_id: str | None = None
                has_projected = metadata.has_projected
                cycle_started = False

                if has_projected:
                    projected_target_id = self._resolve_projected_target_id(
                        world,
                        ship,
                        module,
                        metadata,
                        decision_rule,
                        requested_mode=requested_mode,
                        previous_projected_target=previous_projected_target,
                        allies_pool=allies_pool,
                        enemies_alive=enemies_alive,
                        ally_ids=ally_ids,
                        enemy_ids=enemy_ids,
                        force_target_reselect=force_target_reselect,
                        synced_weapon_fire_delay_pairs=synced_weapon_fire_delay_pairs,
                        now=now_value,
                        source_system_id=source_system_id,
                        candidate_cache=projected_candidate_cache,
                        target_cache=projected_target_cache,
                    )

                if requested_mode == "active":
                    desired_active = True
                elif requested_mode == "online":
                    desired_active = False
                else:
                    desired_active = self._should_activate_module(
                        world,
                        ship,
                        module,
                        decision_rule,
                        projected_target_id,
                    )
                if has_projected and projected_target_id is None and not metadata.is_area_effect:
                    desired_active = False

                desired_active = self._apply_module_reload_gating(
                    ship,
                    module,
                    module_id,
                    desired_active=desired_active,
                    now=now_value,
                )

                desired_active = self._apply_module_reactivation_gating(
                    ship,
                    module_id,
                    desired_active=desired_active,
                    now=now_value,
                )

                module_max_state = self._runtime_module_max_state(ship.runtime, module_id)
                if desired_active and self._runtime_state_rank(module_max_state) < self._runtime_state_rank(ModuleState.ACTIVE):
                    desired_active = False

                activation_target_id: str | None = (
                    projected_target_id
                    if has_projected and not metadata.is_area_effect
                    else None
                )

                if desired_active and activation_target_id is not None and not metadata.is_bomb_launcher:
                    lock_ready = lock_ready_cache.get(activation_target_id)
                    if lock_ready is None:
                        activation_target = world.combat_entity(activation_target_id)
                        lock_ready = self._ensure_target_lock(
                            world,
                            ship,
                            activation_target_id,
                            activation_target,
                            lock_context="module_lock",
                            now=now_value,
                        )
                        lock_ready_cache[activation_target_id] = lock_ready
                    if not lock_ready:
                        desired_active = False

                if has_projected and projected_target_id is None and not metadata.is_area_effect:
                    desired_active = False

                if desired_active:
                    if cycle_time > 0:
                        if cycle_cost > max(0.0, ship.vital.cap):
                            desired_active = False
                        else:
                            if cycle_cost > 0:
                                ship.vital.cap = max(0.0, ship.vital.cap - cycle_cost)
                            self._schedule_module_cycle_deadline(
                                ship,
                                module_id,
                                duration=cycle_time,
                                now=now_value,
                            )
                            cycle_started = True
                    else:
                        self._clear_module_cycle_timer(ship, module_id)
                else:
                    self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                    self._clear_module_cycle_timer(ship, module_id)

                inactive_state = self._runtime_inactive_module_state(ship.runtime, module_id)
                module.state = ModuleState.ACTIVE if desired_active else inactive_state
                if projected_target_id is not None:
                    ship.combat.projected_targets[module_id] = projected_target_id
                elif module_id in ship.combat.projected_targets:
                    ship.combat.projected_targets.pop(module_id, None)

                # ECM is resolved once at cycle start so first activation round shows immediate result.
                if cycle_started:
                    if metadata.is_burst_jammer:
                        self._resolve_area_ecm_cycle(world, ship, module)
                    elif metadata.is_ecm and projected_target_id is not None:
                        self._resolve_ecm_cycle(world, ship, module, projected_target_id)
                    elif metadata.is_missile_weapon or metadata.is_bomb_launcher:
                        self._spawn_cycle_projectiles(
                            world,
                            source=ship,
                            module=module,
                            metadata=metadata,
                            target_id=projected_target_id,
                        )
                    for effect in self._module_bubble_effects(module):
                        if not self._bubble_follows_owner(effect):
                            self._spawn_static_bubble_field(
                                world,
                                source=ship,
                                module=module,
                                effect=effect,
                            )

                if module.state == module.state.ACTIVE and (
                    cycle_started
                    or previous_state != module.state.ACTIVE
                    or previous_projected_target != projected_target_id
                ):
                    if not (metadata.is_missile_weapon or metadata.is_bomb_launcher):
                        self._capture_module_cycle_snapshots(
                            world,
                            ship,
                            module,
                            projected_target_id,
                            area_candidates=alive_by_team_system.get((ship.team, source_system_id), []) if metadata.is_command_burst else None,
                        )

                if cycle_started and self._uses_cycle_start_projected_application(metadata):
                    self._mark_projected_cycle_started(ship.ship_id, module_id)

                if previous_projected_target and (
                    module.state != module.state.ACTIVE or previous_projected_target != projected_target_id
                ):
                    self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)

                if previous_state != module.state:
                    if metadata.affects_local_pyfa_profile:
                        local_signature_dirty = True
                    if self._module_affects_pyfa_remote_inputs(module):
                        active_pyfa_remote_inputs_dirty = True
                    state_target_id = projected_target_id or previous_projected_target
                    state_target = world.combat_entity(state_target_id) if state_target_id else None
                    self._queue_merged_event(
                        "active_module_state_switch",
                        merge_fields={
                            "team": ship.team.value,
                            "squad": ship.squad_id,
                            "ship_type": ship.fit.ship_name,
                            "module": module_id,
                            "group": module.group,
                            "from_state": previous_state.value,
                            "to_state": module.state.value,
                            "target_type": state_target.fit.ship_name if state_target is not None else "",
                        },
                    )

                if cycle_started:
                    effects = ",".join(effect.name for effect in active_effects)
                    cycle_target = world.combat_entity(projected_target_id) if projected_target_id else None
                    self._queue_merged_event(
                        "active_module_cycle",
                        merge_fields={
                            "team": ship.team.value,
                            "squad": ship.squad_id,
                            "ship_type": ship.fit.ship_name,
                            "module": module_id,
                            "group": module.group,
                            "effects": effects,
                            "cycle_time": cycle_time,
                            "target_type": cycle_target.fit.ship_name if cycle_target is not None else "",
                        },
                        sum_fields={
                            "cap_cost": cycle_cost,
                        },
                    )

                if metadata.affects_local_pyfa_profile and previous_state != module.state:
                    if self._runtime_has_active_pyfa_remote_inputs(runtime):
                        pyfa_remote_inputs_dirty = True

                if self._module_affects_pyfa_remote_inputs(module) and (
                    previous_state != module.state
                    or previous_projected_target != projected_target_id
                    or cycle_started
                ):
                    pyfa_remote_inputs_dirty = True

                if self._module_keeps_decision_pending_with_context(
                    ship,
                    module,
                    metadata,
                    propulsion_active=propulsion_active,
                    recent_enemy_weapon_damage_active=recent_enemy_weapon_damage_active,
                    enemy_targets_active=enemy_targets_active,
                    ally_targets_active=ally_targets_active,
                    has_focus_queue=has_focus_queue,
                ):
                    next_pending_modules.add(module_id)

            ship.combat.module_decision_pending = next_pending_modules
            self._sync_timer_views_for_ship(ship, now_value)

            if local_signature_dirty:
                runtime.diagnostics.pop("runtime_local_state_signature", None)
                tracked_ids = runtime.diagnostics.get("runtime_local_stateful_module_ids")
                if isinstance(tracked_ids, tuple):
                    tracked_id_set = {str(module_id) for module_id in tracked_ids}
                    runtime.diagnostics["runtime_local_state_signature"] = tuple(
                        (str(module.module_id), str(module.state.value or "ONLINE").upper())
                        for module in runtime.modules
                        if str(module.module_id) in tracked_id_set
                    )
                else:
                    runtime.diagnostics["runtime_local_state_signature"] = tuple(
                        (str(module.module_id), str(module.state.value or "ONLINE").upper())
                        for module in runtime.modules
                        if self._module_static_metadata(module).affects_local_pyfa_profile
                    )
            if active_pyfa_remote_inputs_dirty:
                runtime.diagnostics.pop("runtime_has_active_pyfa_remote_inputs", None)
                runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = self._runtime_has_active_pyfa_remote_inputs(runtime)

        return pyfa_remote_inputs_dirty

    def _changed_focus_queues(self, world: WorldState) -> set[str]:
        changed: set[str] = set()
        active_focus_keys: set[str] = {
            squad_key(ship.team, ship.squad_id)
            for ship in world.ships.values()
            if ship.vital.alive
        }
        active_focus_keys.update(str(key) for key in world.squad_focus_queues.keys())

        for focus_key in active_focus_keys:
            current_queue = tuple(str(target_id) for target_id in world.squad_focus_queues.get(focus_key, []))
            previous_queue = self._last_focus_queue_by_squad.get(focus_key)
            if previous_queue != current_queue:
                changed.add(focus_key)
            self._last_focus_queue_by_squad[focus_key] = current_queue

        for stale_key in [key for key in self._last_focus_queue_by_squad.keys() if key not in active_focus_keys]:
            self._last_focus_queue_by_squad.pop(stale_key, None)

        return changed

    def _update_squad_prelocks(self, world: WorldState, dt: float, effective_profiles: dict[str, ShipProfile]) -> None:
        for ship in world.ships.values():
            if not ship.vital.alive or self._ship_hidden_from_targeting(ship):
                ship.combat.prelocked_targets.clear()
                ship.combat.prelock_timers.clear()
                continue
            focus_key = squad_key(ship.team, ship.squad_id)
            queue = world.squad_focus_queues.get(focus_key, [])
            seen: set[str] = set()
            cleaned: list[str] = []
            for target_id in queue:
                if target_id in seen:
                    continue
                target = world.combat_entity(target_id)
                if (
                    target is None
                    or not target.vital.alive
                    or self._ship_hidden_from_targeting(target)
                    or target.team == ship.team
                    or self._ship_system_id(target) != self._ship_system_id(ship)
                ):
                    continue
                seen.add(target_id)
                cleaned.append(target_id)

            pre_targets = cleaned[1:] if len(cleaned) > 1 else []
            valid_pre = set(pre_targets)
            ship_prelocked = ship.combat.prelocked_targets
            ship_timers = ship.combat.prelock_timers
            ship_prelocked.intersection_update(valid_pre)
            for target_id in list(ship_timers):
                if target_id not in valid_pre:
                    ship_timers.pop(target_id, None)

            if not pre_targets:
                continue

            attacker_profile = effective_profiles.get(ship.ship_id) or ship.profile
            for target_id in pre_targets:
                if target_id in ship_prelocked:
                    continue
                target = world.combat_entity(target_id)
                if target is None or not target.vital.alive:
                    continue
                if not self._target_within_lock_range(ship, target, source_profile=attacker_profile):
                    ship_timers.pop(target_id, None)
                    continue
                target_profile = effective_profiles.get(target_id) or target.profile
                left = ship_timers.get(target_id)
                if left is None:
                    ship_timers[target_id] = self._cached_lock_time(attacker_profile, target_profile)
                    continue
                left -= dt
                if left <= 0:
                    ship_prelocked.add(target_id)
                    ship_timers.pop(target_id, None)
                else:
                    ship_timers[target_id] = left
