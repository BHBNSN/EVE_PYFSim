from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import logging
import random
import time
import weakref
from typing import Any

import numpy as np

from ..combat_control_workset import (
    enqueue_control_signal_modules,
    ensure_ship_module_decision_pending,
    module_keeps_decision_pending,
    runtime_decision_rule_groups,
    runtime_controlled_entry_lookup,
    runtime_controlled_module_ids,
    ship_candidate_module_ids,
)
from ..fleet_setup import (
    _module_affects_local_pyfa_profile,
    _runtime_local_profile_state_signature,
    get_runtime_resolve_cache_key,
    resolve_runtime_from_pyfa_runtime,
)
from ..fit_runtime import EffectClass, FitRuntime, ModuleEffect, ModuleRuntime, ModuleState, ProjectedImpact, RuntimeStatEngine
from ..math2d import Vector2
from ..models import ProjectileBlast, ProjectileEntity, ShipProfile, Team
from ..pyfa_bridge import PyfaBridge
from ..sim_logging import log_sim_event
from ..timing_wheel import EventType, TimingWheel
from ..world import WorldState


DamageTuple = tuple[float, float, float, float]

_PROFILE_PASSTHROUGH_ATTRS = (
    "weapon_system",
    "optimal_sig",
    "turret_dps",
    "missile_dps",
    "turret_cycle",
    "missile_cycle",
    "damage_em",
    "damage_thermal",
    "damage_kinetic",
    "damage_explosive",
    "turret_em_dps",
    "turret_thermal_dps",
    "turret_kinetic_dps",
    "turret_explosive_dps",
    "missile_em_dps",
    "missile_thermal_dps",
    "missile_kinetic_dps",
    "missile_explosive_dps",
    "missile_damage_reduction_factor",
    "warp_speed_au_s",
    "warp_capacitor_need",
    "max_warp_distance_au",
)

_PYFA_PROJECTION_RANGE_BUCKET_M = 100.0
_REPAIR_QUEUE_LAYERS = ("shield", "armor", "structure")

_RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS = frozenset(
    {
        "runtime_module_static_metadata",
        "runtime_module_buckets",
        "runtime_controlled_module_ids",
        "runtime_controlled_entry_lookup",
        "runtime_decision_rule_groups",
    }
)

from .models import *
from .models import _FORMULA_PROJECTED_KEYS, _sum_damage, _scale_damage, _layer_effective_damage, _apply_damage_sequence


class CombatSystem:
    def __init__(
        self,
        pyfa: PyfaBridge,
    ) -> None:
        self.pyfa = pyfa
        self.runtime = RuntimeStatEngine()
        self.logger: logging.Logger | None = None
        self.detailed_logging: bool = False
        self.hotspot_logging_enabled: bool = False
        self.event_logging_enabled: bool = False
        self.event_merge_window_sec: float = 1.0
        self._diag_logged_ships: set[str] = set()
        self._lock_time_cache: dict[tuple[float, float], float] = {}
        self._projected_cycle_totals: dict[tuple[str, str, str], dict[str, float]] = {}
        self._projected_cycle_starts_this_tick: set[tuple[str, str]] = set()
        self._module_cycle_target_snapshots: dict[tuple[str, str], dict[str, CycleTargetSnapshot]] = {}
        self._merged_event_buckets: dict[tuple, dict[str, Any]] = {}
        self._merge_window_start_time: float | None = None
        self._merge_window_end_time: float | None = None
        self._last_focus_queue_by_squad: dict[str, tuple[str, ...]] = {}
        self._pyfa_remote_inputs_dirty: bool = True
        self._alive_runtime_ship_ids: set[str] = set()
        self._cached_command_booster_snapshots: dict[str, list[dict[str, Any]]] | None = None
        self._cached_projected_source_snapshots: dict[str, list[dict[str, Any]]] | None = None
        self._module_static_metadata_by_object_id: dict[int, tuple[weakref.ReferenceType[Any], ModuleStaticMetadata]] = {}
        self._repair_queue_cache: dict[tuple[Team, str], tuple[str, ...]] = {}
        self._repair_queue_dirty: set[tuple[Team, str]] = set()
        self._projectile_seq: int = 0
        self._projectile_blast_seq: int = 0
        self._timing_wheel = TimingWheel()
        self._decision_reference_time: float | None = None

    def attach_logger(
        self,
        logger: logging.Logger,
        detailed_logging: bool,
        merge_window_sec: float = 1.0,
        hotspot_logging: bool = False,
    ) -> None:
        self.logger = logger
        self.event_logging_enabled = bool(detailed_logging)
        self.detailed_logging = bool(detailed_logging)
        self.hotspot_logging_enabled = bool(hotspot_logging)
        try:
            self.event_merge_window_sec = max(0.1, float(merge_window_sec))
        except Exception:
            self.event_merge_window_sec = 1.0
        self._merge_window_start_time = None
        self._merge_window_end_time = None
        self._merged_event_buckets.clear()

    def _log_hotspot(self, name: str, start_time: float, **fields: Any) -> None:
        if not self.hotspot_logging_enabled:
            return
        if self.logger is None or self.logger.disabled:
            return
        log_sim_event(
            self.logger,
            "hotspot",
            name=name,
            duration_ms=(time.perf_counter() - start_time) * 1000.0,
            **fields,
        )

    @staticmethod
    def _copy_profile_passthrough_fields(base: ShipProfile, target: ShipProfile) -> None:
        for attr in _PROFILE_PASSTHROUGH_ATTRS:
            setattr(target, attr, getattr(base, attr, getattr(target, attr, 0.0)))

    @staticmethod
    def _copy_formula_base_fields(base: ShipProfile, target: ShipProfile) -> None:
        for attr in (
            "dps",
            "volley",
            "optimal",
            "falloff",
            "tracking",
            "sig_radius",
            "scan_resolution",
            "max_target_range",
            "max_speed",
            "max_cap",
            "cap_recharge_time",
            "shield_hp",
            "armor_hp",
            "structure_hp",
            "rep_amount",
            "rep_cycle",
            "mass",
            "agility",
            "warp_speed_au_s",
            "warp_capacitor_need",
            "max_warp_distance_au",
        ):
            setattr(target, attr, getattr(base, attr))

    @staticmethod
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    def _clear_ship_warp_engagement_state(self, ship, runtime: FitRuntime | None = None) -> None:
        self._break_all_locks(ship)
        ship.combat.last_attack_target = None
        if runtime is None:
            return

        for module, metadata in self._ship_candidate_control_entries(ship, runtime):
            if not (
                metadata.has_projected
                or metadata.is_weapon
                or metadata.is_command_burst
                or metadata.is_area_effect
                or metadata.has_projected_rep
                or metadata.is_cap_warfare
                or metadata.is_target_ewar
                or metadata.is_ecm
            ):
                continue
            module_id = str(module.module_id)
            if module.state in {module.state.ACTIVE, module.state.OVERHEATED}:
                module.state = module.state.ONLINE
            self._clear_module_cycle_snapshots(ship.ship_id, module_id)
            self._clear_module_cycle_timer(ship, module_id)
            self._clear_module_reactivation_timer(ship, module_id)

    def _apply_runtime_projected_impacts(self, base: ShipProfile, impacts: list[ProjectedImpact], runtime=None) -> ShipProfile:
        penalty_context = None
        weapon_penalty_context = None
        if runtime is not None:
            raw_context = getattr(runtime, "diagnostics", {}).get("pyfa_ship_attribute_penalty_context")
            if isinstance(raw_context, dict):
                penalty_context = raw_context
            raw_weapon_context = getattr(runtime, "diagnostics", {}).get("pyfa_weapon_attribute_penalty_context")
            if isinstance(raw_weapon_context, list):
                weapon_penalty_context = raw_weapon_context
        effective = self.runtime.apply_projected_effects(replace(base), impacts, base_penalty_context=penalty_context)
        self._apply_weighted_weapon_projection_context(effective, impacts, weapon_penalty_context)
        self._copy_profile_passthrough_fields(base, effective)
        return effective

    def _apply_weighted_weapon_projection_context(
        self,
        effective: ShipProfile,
        impacts: list[ProjectedImpact],
        weapon_penalty_context: list[dict[str, Any]] | None,
    ) -> None:
        if not weapon_penalty_context:
            return

        grouped_multipliers: dict[str, dict[str, list[float]]] = {
            "optimal": {},
            "falloff": {},
            "tracking": {},
        }
        affected_keys: set[str] = set()
        for impact in impacts:
            effect = impact.effect
            strength = max(0.0, min(1.0, float(impact.strength)))
            mult_groups = getattr(effect, "projected_mult_groups", {}) or {}
            for key in ("optimal", "falloff", "tracking"):
                value = effect.projected_mult.get(key)
                if value is None:
                    continue
                group_name = mult_groups.get(key, "default")
                if group_name is None:
                    group_name = f"__unstacked__:{len(grouped_multipliers[key])}"
                grouped_multipliers[key].setdefault(str(group_name), []).append(1.0 + (float(value) - 1.0) * strength)
                affected_keys.add(key)

        if not affected_keys:
            return

        attr_map = {
            "optimal": "optimal",
            "falloff": "falloff",
            "tracking": "tracking",
        }
        for key in affected_keys:
            weighted_total = 0.0
            total_weight = 0.0
            for entry in weapon_penalty_context:
                if str(entry.get("kind", "")) != "gunnery":
                    continue
                weight = float(entry.get("weight", 0.0) or 0.0)
                if weight <= 0.0:
                    continue
                context = entry.get(key)
                if isinstance(context, dict):
                    current_value = float(context.get("current", getattr(effective, attr_map[key])) or 0.0)
                else:
                    current_value = float(getattr(effective, attr_map[key]) or 0.0)
                    context = None
                value = self.runtime._apply_penalized_projection(
                    current_value,
                    0.0,
                    grouped_multipliers[key],
                    context,
                )
                weighted_total += weight * value
                total_weight += weight
            if total_weight <= 0.0:
                continue
            resolved_value = weighted_total / total_weight
            if key == "optimal":
                effective.optimal = max(1.0, resolved_value)
            elif key == "falloff":
                effective.falloff = max(1.0, resolved_value)
            elif key == "tracking":
                effective.tracking = max(0.0001, resolved_value)

    def _fallback_unprojected_profile(self, ship) -> ShipProfile:
        runtime = getattr(ship, "runtime", None)
        if runtime is None:
            return replace(ship.profile)

        fallback = replace(ship.profile)
        fit_base = replace(self.pyfa.build_profile(ship.fit))
        runtime_base = replace(self.runtime.compute_base_profile(runtime))
        self._copy_formula_base_fields(runtime_base, fallback)
        for attr in (
            "sensor_strength_gravimetric",
            "sensor_strength_ladar",
            "sensor_strength_magnetometric",
            "sensor_strength_radar",
            "shield_resonance_em",
            "shield_resonance_thermal",
            "shield_resonance_kinetic",
            "shield_resonance_explosive",
            "armor_resonance_em",
            "armor_resonance_thermal",
            "armor_resonance_kinetic",
            "armor_resonance_explosive",
            "structure_resonance_em",
            "structure_resonance_thermal",
            "structure_resonance_kinetic",
            "structure_resonance_explosive",
            "missile_explosion_radius",
            "missile_explosion_velocity",
            "missile_max_range",
            "warp_speed_au_s",
            "warp_capacitor_need",
            "max_warp_distance_au",
        ):
            setattr(fallback, attr, getattr(fit_base, attr))
        return fallback

    @staticmethod
    def _local_runtime_state_signature(runtime) -> tuple[tuple[str, str], ...] | None:
        blueprint = runtime.diagnostics.get("pyfa_blueprint")
        if not isinstance(blueprint, dict):
            return None
        return _runtime_local_profile_state_signature(runtime)

    def _local_runtime_state_signature_from_metadata(self, runtime) -> tuple[tuple[str, str], ...] | None:
        blueprint = runtime.diagnostics.get("pyfa_blueprint")
        if not isinstance(blueprint, dict):
            return None
        tracked_ids = runtime.diagnostics.get("runtime_local_stateful_module_ids")
        if isinstance(tracked_ids, tuple):
            tracked_id_set = {str(module_id) for module_id in tracked_ids}
            signature = tuple(
                (str(module.module_id), str(module.state.value or "ONLINE").upper())
                for module in runtime.modules
                if str(module.module_id) in tracked_id_set
            )
        else:
            signature = tuple(
                (str(module.module_id), str(module.state.value or "ONLINE").upper())
                for module in runtime.modules
                if self._module_static_metadata(module).affects_local_pyfa_profile
            )
        runtime.diagnostics["runtime_local_state_signature"] = signature
        return signature

    @staticmethod
    def _runtime_observed_module_state_signature(runtime) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(module.module_id), str(module.state.value or "ONLINE").upper())
            for module in runtime.modules
        )

    def _mark_pyfa_remote_inputs_dirty(self) -> None:
        self._pyfa_remote_inputs_dirty = True

    def _mark_all_repair_queues_dirty(self) -> None:
        self._repair_queue_cache.clear()
        for team in Team:
            for layer in _REPAIR_QUEUE_LAYERS:
                self._repair_queue_dirty.add((team, layer))

    def _mark_team_repair_queues_dirty(self, team: Team, *layers: str) -> None:
        if team is None:
            return
        dirty_layers = tuple(str(layer) for layer in (layers or _REPAIR_QUEUE_LAYERS) if str(layer) in _REPAIR_QUEUE_LAYERS)
        for layer in dirty_layers:
            cache_key = (team, layer)
            self._repair_queue_cache.pop(cache_key, None)
            self._repair_queue_dirty.add(cache_key)

    def _refresh_alive_runtime_ship_ids(self, world: WorldState) -> None:
        current_alive_runtime_ship_ids = {
            ship.ship_id
            for ship in world.ships.values()
            if ship.vital.alive and ship.runtime is not None and not self._ship_in_warp(ship)
        }
        if current_alive_runtime_ship_ids != self._alive_runtime_ship_ids:
            self._alive_runtime_ship_ids = current_alive_runtime_ship_ids
            self._mark_pyfa_remote_inputs_dirty()
            self._mark_all_repair_queues_dirty()

    def _cached_pyfa_remote_inputs_available(self) -> bool:
        return self._cached_command_booster_snapshots is not None and self._cached_projected_source_snapshots is not None

    @staticmethod
    def _module_tags(module) -> frozenset[str]:
        return frozenset(str(tag) for tag in (getattr(module, "tags", ()) or ()))

    def _module_static_metadata(self, module) -> ModuleStaticMetadata:
        key = id(module)
        cached_entry = self._module_static_metadata_by_object_id.get(key)
        if cached_entry is not None:
            cached_ref, cached_metadata = cached_entry
            if cached_ref() is module:
                return cached_metadata
            self._module_static_metadata_by_object_id.pop(key, None)

        tags = self._module_tags(module)
        active_effects = tuple(
            effect
            for effect in module.effects
            if str(effect.state_required.value).upper() == "ACTIVE"
        )
        projected_effects = tuple(
            (effect_index, effect)
            for effect_index, effect in enumerate(module.effects)
            if effect.effect_class == EffectClass.PROJECTED
        )
        has_projected = bool(projected_effects)
        target_side = "ally" if "support" in tags and "hostile" not in tags else "enemy"
        is_command_burst = "command_burst" in tags
        is_smart_bomb = "smart_bomb" in tags
        is_burst_jammer = "burst_jammer" in tags
        is_area_effect = "area_effect" in tags
        is_cap_booster = "cap_booster" in tags
        is_propulsion = "propulsion" in tags
        is_damage_control = "damage_control" in tags
        is_hardener = "hardener" in tags
        is_cap_warfare = "cap_warfare" in tags
        is_target_ewar = "target_ewar" in tags
        is_ecm = "ecm" in tags
        is_weapon = "weapon" in tags
        is_missile_weapon = any(
            float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5
            for _effect_index, effect in projected_effects
        )
        is_bomb_launcher = any(
            float(effect.projected_add.get("weapon_is_bomb", 0.0) or 0.0) > 0.5
            for _effect_index, effect in projected_effects
        )
        has_projected_rep = "remote_repair" in tags
        repair_layers: list[str] = []
        if has_projected_rep:
            for _effect_index, effect in projected_effects:
                if float(effect.projected_add.get("shield_rep", 0.0) or 0.0) > 0.0 and "shield" not in repair_layers:
                    repair_layers.append("shield")
                if float(effect.projected_add.get("armor_rep", 0.0) or 0.0) > 0.0 and "armor" not in repair_layers:
                    repair_layers.append("armor")
                if float(effect.projected_add.get("structure_rep", 0.0) or 0.0) > 0.0 and "structure" not in repair_layers:
                    repair_layers.append("structure")
        is_offensive_ewar = "offensive_ewar" in tags or is_ecm or is_cap_warfare
        supports_formula_projected_profile = (
            has_projected
            and any(effect.projected_mult or effect.projected_add for _effect_index, effect in projected_effects)
            and all(self._effect_supports_runtime_formula_projection(effect) for _effect_index, effect in projected_effects)
        )
        uses_pyfa_projected_profile = (
            has_projected
            and not is_command_burst
            and not is_smart_bomb
            and not is_burst_jammer
            and not is_ecm
            and not is_weapon
            and not has_projected_rep
            and not is_cap_warfare
            and not supports_formula_projected_profile
        )

        if is_command_burst:
            decision_rule = ModuleDecisionRule(
                rule_id="area_command_burst",
                activation_mode="always",
                target_mode="none",
            )
        elif is_smart_bomb:
            decision_rule = ModuleDecisionRule(
                rule_id="area_smart_bomb",
                activation_mode="enemy_in_area",
                target_mode="none",
                cap_threshold=0.15,
            )
        elif is_burst_jammer:
            decision_rule = ModuleDecisionRule(
                rule_id="area_burst_jammer",
                activation_mode="enemy_in_area",
                target_mode="none",
                cap_threshold=0.15,
            )
        elif is_weapon:
            decision_rule = ModuleDecisionRule(
                rule_id="weapon_focus_only",
                activation_mode="weapon_focus_only",
                target_mode="weapon_focus_prefocus",
            )
        elif has_projected:
            if has_projected_rep:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_remote_repair",
                    activation_mode="always",
                    target_mode="ally_repair_queue",
                )
            elif is_offensive_ewar:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_offensive_ewar",
                    activation_mode="cap_min",
                    target_mode="enemy_random",
                    cap_threshold=0.15,
                )
            elif is_target_ewar:
                decision_rule = ModuleDecisionRule(
                    rule_id="weapon_focus_only",
                    activation_mode="weapon_focus_only",
                    target_mode="weapon_focus_prefocus",
                )
            elif target_side == "ally":
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_support_generic",
                    activation_mode="always",
                    target_mode="ally_lowest_hp",
                )
            else:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_hostile_generic",
                    activation_mode="never",
                    target_mode="none",
                )
        elif is_propulsion:
            decision_rule = ModuleDecisionRule(
                rule_id="local_propulsion",
                activation_mode="propulsion_command",
                target_mode="none",
            )
        elif is_damage_control:
            decision_rule = ModuleDecisionRule(
                rule_id="local_damage_control",
                activation_mode="recent_enemy_weapon_damage",
                target_mode="none",
            )
        elif is_hardener:
            decision_rule = ModuleDecisionRule(
                rule_id="local_hardener",
                activation_mode="cap_or_low_hp",
                target_mode="none",
                cap_threshold=0.10,
            )
        elif is_cap_booster:
            decision_rule = ModuleDecisionRule(
                rule_id="local_cap_booster",
                activation_mode="cap_max",
                target_mode="none",
                cap_threshold=0.85,
            )
        else:
            decision_rule = ModuleDecisionRule(
                rule_id="local_active_default",
                activation_mode="never",
                target_mode="none",
            )

        metadata = ModuleStaticMetadata(
            active_effects=active_effects,
            projected_effects=projected_effects,
            cycle_cost=sum(max(0.0, effect.cap_need) for effect in active_effects),
            cycle_time=min((max(0.1, effect.cycle_time) for effect in active_effects if effect.cycle_time > 0), default=0.0),
            reactivation_delay=max((max(0.0, float(getattr(effect, "reactivation_delay", 0.0) or 0.0)) for effect in active_effects), default=0.0),
            has_projected=has_projected,
            target_side=target_side,
            is_command_burst=is_command_burst,
            is_smart_bomb=is_smart_bomb,
            is_burst_jammer=is_burst_jammer,
            is_area_effect=is_area_effect,
            is_weapon=is_weapon,
            is_missile_weapon=is_missile_weapon,
            is_bomb_launcher=is_bomb_launcher,
            has_projected_rep=has_projected_rep,
            is_cap_warfare=is_cap_warfare,
            is_target_ewar=is_target_ewar,
            is_ecm=is_ecm,
            uses_pyfa_projected_profile=uses_pyfa_projected_profile,
            is_hardener=is_hardener,
            is_cap_booster=is_cap_booster,
            is_propulsion=is_propulsion,
            is_damage_control=is_damage_control,
            affects_local_pyfa_profile=_module_affects_local_pyfa_profile(module),
            repair_layers=tuple(repair_layers),
            decision_rule=decision_rule,
        )
        module_ref = weakref.ref(
            module,
            lambda ref, *, cache=self._module_static_metadata_by_object_id, cache_key=key: (
                cache.pop(cache_key, None)
                if cache.get(cache_key, (None, None))[0] is ref
                else None
            ),
        )
        self._module_static_metadata_by_object_id[key] = (module_ref, metadata)
        return metadata

    @staticmethod
    def _effect_supports_runtime_formula_projection(effect) -> bool:
        projected_mult_keys = {str(key) for key in effect.projected_mult.keys()}
        projected_add_keys = {str(key) for key in effect.projected_add.keys()}
        modeled_keys = projected_mult_keys | projected_add_keys
        if not modeled_keys:
            return False
        return modeled_keys.issubset(_FORMULA_PROJECTED_KEYS)

    @staticmethod
    def _round_projection_signature_value(value: float) -> float:
        return round(float(value or 0.0), 6)

    @classmethod
    def _projected_effect_signature(cls, effect) -> tuple[Any, ...]:
        return (
            str(getattr(effect, "name", "") or ""),
            str(getattr(getattr(effect, "effect_class", None), "value", getattr(effect, "effect_class", "")) or ""),
            str(getattr(getattr(effect, "state_required", None), "value", getattr(effect, "state_required", "")) or ""),
            cls._round_projection_signature_value(float(getattr(effect, "range_m", 0.0) or 0.0)),
            cls._round_projection_signature_value(float(getattr(effect, "falloff_m", 0.0) or 0.0)),
            tuple(
                sorted(
                    (str(key), cls._round_projection_signature_value(float(value or 0.0)))
                    for key, value in effect.projected_mult.items()
                )
            ),
            tuple(
                sorted(
                    (str(key), cls._round_projection_signature_value(float(value or 0.0)))
                    for key, value in effect.projected_add.items()
                )
            ),
            tuple(
                sorted(
                    (str(key), None if value is None else str(value))
                    for key, value in getattr(effect, "projected_mult_groups", {}).items()
                )
            ),
            tuple(getattr(effect, "projected_signature", ()) or ()),
        )

    @classmethod
    def _projected_module_runtime_signature(
        cls,
        module,
        module_blueprint: dict[str, Any] | None,
        module_state: str,
        active_effect_indices: set[int] | None = None,
    ) -> tuple[Any, ...]:
        projected_effect_signatures: list[tuple[Any, ...]] = []
        for effect_index, effect in enumerate(module.effects):
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            if active_effect_indices and effect_index not in active_effect_indices:
                continue
            projected_effect_signatures.append(cls._projected_effect_signature(effect))

        blueprint_signature = (
            str((module_blueprint or {}).get("module_name", "") or ""),
            str((module_blueprint or {}).get("charge_name", "") or ""),
            tuple(sorted(str(name) for name in ((module_blueprint or {}).get("effect_names") or ()))),
        )
        return (
            "module_projection",
            blueprint_signature,
            str(getattr(module, "group", "") or ""),
            str(module_state or "ONLINE").upper(),
            tuple(projected_effect_signatures),
        )

    def _runtime_module_metadata_list(self, runtime) -> tuple[ModuleStaticMetadata, ...]:
        cached = runtime.diagnostics.get("runtime_module_static_metadata")
        if isinstance(cached, tuple) and len(cached) == len(runtime.modules):
            return cached
        metadata_list = tuple(self._module_static_metadata(module) for module in runtime.modules)
        runtime.diagnostics["runtime_module_static_metadata"] = metadata_list
        return metadata_list

    def _runtime_module_buckets(self, runtime) -> RuntimeModuleBuckets:
        cached = runtime.diagnostics.get("runtime_module_buckets")
        if isinstance(cached, RuntimeModuleBuckets) and cached.module_count == len(runtime.modules):
            return cached

        controlled_ids = runtime.diagnostics.get("runtime_controlled_module_ids")
        controlled_id_set = {str(module_id) for module_id in controlled_ids} if isinstance(controlled_ids, tuple) else None
        controlled_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        command_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        runtime_projected_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        pyfa_projected_entries: list[tuple[Any, ModuleStaticMetadata]] = []

        for module in runtime.modules:
            module_id = str(module.module_id)
            if controlled_id_set is not None and module_id not in controlled_id_set:
                if not any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects):
                    continue
            metadata = self._module_static_metadata(module)
            if controlled_id_set is not None:
                if module_id in controlled_id_set:
                    controlled_entries.append((module, metadata))
            elif metadata.active_effects:
                controlled_entries.append((module, metadata))
            if metadata.is_command_burst:
                command_entries.append((module, metadata))
            if metadata.projected_effects:
                if metadata.uses_pyfa_projected_profile:
                    pyfa_projected_entries.append((module, metadata))
                else:
                    runtime_projected_entries.append((module, metadata))

        buckets = RuntimeModuleBuckets(
            module_count=len(runtime.modules),
            controlled_entries=tuple(controlled_entries),
            command_entries=tuple(command_entries),
            runtime_projected_entries=tuple(runtime_projected_entries),
            pyfa_projected_entries=tuple(pyfa_projected_entries),
        )
        runtime.diagnostics["runtime_module_buckets"] = buckets
        return buckets

    def _runtime_controlled_module_ids(self, runtime) -> tuple[str, ...]:
        return runtime_controlled_module_ids(runtime, self._runtime_module_buckets(runtime).controlled_entries)

    def _runtime_controlled_entry_lookup(self, runtime) -> dict[str, tuple[Any, ModuleStaticMetadata]]:
        controlled_entries = self._runtime_module_buckets(runtime).controlled_entries
        controlled_ids = self._runtime_controlled_module_ids(runtime)
        return runtime_controlled_entry_lookup(runtime, controlled_entries, controlled_ids)

    def _runtime_decision_rule_groups(self, runtime) -> dict[str, dict[str, tuple[str, ...]]]:
        return runtime_decision_rule_groups(runtime, self._runtime_module_buckets(runtime).controlled_entries)

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
        now: float | None = None,
    ) -> None:
        now_value = self._decision_now(world, now)
        enqueue_control_signal_modules(
            ship,
            runtime_decision_rule_groups(runtime, self._runtime_module_buckets(runtime).controlled_entries),
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
        has_focus_queue: bool,
    ) -> bool:
        if self._manual_module_mode(ship, str(module.module_id)) != "auto":
            return module.state != module.state.OFFLINE
        decision_rule = metadata.decision_rule
        return module_keeps_decision_pending(
            ship,
            module,
            cycle_time=metadata.cycle_time,
            activation_mode=decision_rule.activation_mode,
            target_mode=decision_rule.target_mode,
            propulsion_active=propulsion_active,
            recent_enemy_weapon_damage_active=recent_enemy_weapon_damage_active,
            has_focus_queue=has_focus_queue,
        )

    @staticmethod
    def _manual_module_mode(ship, module_id: str) -> str:
        raw_modes = getattr(ship.combat, "module_manual_modes", {})
        if not isinstance(raw_modes, dict):
            return "auto"
        normalized = str(raw_modes.get(str(module_id), "auto") or "auto").strip().lower()
        return normalized if normalized in {"auto", "active", "online"} else "auto"

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
            target = world.ships.get(str(candidate_id))
            if target is None or not target.vital.alive or target.team == source.team:
                continue
            if not self._module_in_projected_range(source, target, module):
                continue
            if not self._can_target_under_ecm(source, str(candidate_id), self._decision_now(world)):
                continue
            return str(candidate_id)
        return None

    def _validate_cached_pyfa_base_profiles(
        self,
        world: WorldState,
    ) -> tuple[bool, bool, list[tuple[Any, ShipProfile]]]:
        reusable_profiles: list[tuple[Any, ShipProfile]] = []
        reusable = True
        remote_recollect_required = False

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            local_signature = self._local_runtime_state_signature_from_metadata(ship.runtime)
            cached_local_signature = ship.runtime.diagnostics.get("pyfa_local_state_signature")
            cached_base_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
            if local_signature != cached_local_signature or not isinstance(cached_base_profile, ShipProfile):
                reusable = False
                if (
                    local_signature != cached_local_signature
                    and self._runtime_has_active_pyfa_remote_inputs(ship.runtime)
                ):
                    remote_recollect_required = True
                continue

            reusable_profiles.append((ship, cached_base_profile))

        return reusable, remote_recollect_required, reusable_profiles

    @staticmethod
    def _command_snapshot_list_signature(snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        signature: list[tuple[Any, ...]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            state_raw = snapshot.get("state_by_module_id")
            state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
            signature.append(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple((str(module_id), str(state)) for module_id, state in state_by_module_id.items()),
                )
            )
        return tuple(signature)

    @classmethod
    def _projected_snapshot_list_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        signature: list[tuple[Any, ...]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            projection_key_mode, distance_signature = cls._normalized_snapshot_projection_signature(snapshot)
            signature.append(
                (
                    cls._projected_snapshot_module_signature(snapshot),
                    projection_key_mode,
                    distance_signature,
                )
            )
        return tuple(signature)

    @staticmethod
    def _projected_snapshot_module_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
        direct_signature = snapshot.get("pyfa_projection_module_signature")
        if isinstance(direct_signature, tuple):
            return direct_signature
        if isinstance(direct_signature, list):
            return tuple(direct_signature)

        state_raw = snapshot.get("state_by_module_id")
        state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
        command_raw = snapshot.get("command_booster_snapshots")
        command_snapshots = [snap for snap in command_raw if isinstance(snap, dict)] if isinstance(command_raw, list) else []
        return (
            "legacy_source",
            str(snapshot.get("fit_key", "") or ""),
            tuple((str(module_id), str(state)) for module_id, state in state_by_module_id.items()),
            CombatSystem._command_snapshot_list_signature(command_snapshots),
        )

    def _module_affects_pyfa_remote_inputs(self, module) -> bool:
        metadata = self._module_static_metadata(module)
        return metadata.is_command_burst or metadata.uses_pyfa_projected_profile

    def _runtime_has_active_pyfa_remote_inputs(self, runtime) -> bool:
        buckets = self._runtime_module_buckets(runtime)
        for module, _metadata in buckets.command_entries:
            if str(module.state.value or "ONLINE").upper() not in {"ACTIVE", "OVERHEATED"}:
                continue
            runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = True
            return True
        for module, _metadata in buckets.pyfa_projected_entries:
            if str(module.state.value or "ONLINE").upper() not in {"ACTIVE", "OVERHEATED"}:
                continue
            runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = True
            return True
        runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = False
        return False

    def _reconcile_external_module_state_changes(self, world: WorldState, ship, runtime) -> bool:
        current_signature = self._runtime_observed_module_state_signature(runtime)
        cached_signature = runtime.diagnostics.get("runtime_observed_module_state_signature")
        runtime.diagnostics["runtime_observed_module_state_signature"] = current_signature
        if not isinstance(cached_signature, tuple) or cached_signature == current_signature:
            return False

        previous_states = {str(module_id): str(state) for module_id, state in cached_signature}
        current_states = {str(module_id): str(state) for module_id, state in current_signature}
        changed_module_ids = {
            module_id
            for module_id, state in current_states.items()
            if previous_states.get(module_id) != state
        } | {
            module_id
            for module_id in previous_states.keys()
            if module_id not in current_states
        }
        if not changed_module_ids:
            return False

        runtime.diagnostics.pop("runtime_local_state_signature", None)
        runtime.diagnostics.pop("runtime_has_active_pyfa_remote_inputs", None)

        pyfa_remote_inputs_dirty = False
        for module in runtime.modules:
            module_id = str(module.module_id)
            if module_id not in changed_module_ids:
                continue

            metadata = self._module_static_metadata(module)
            state_name = str(module.state.value or "ONLINE").upper()
            previous_projected_target = ship.combat.projected_targets.get(module_id)

            if state_name not in {"ACTIVE", "OVERHEATED"}:
                self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                self._clear_module_cycle_timer(ship, module_id)
                self._clear_module_reactivation_timer(ship, module_id)
                if state_name == "OFFLINE":
                    ship.combat.projected_targets.pop(module_id, None)

            if metadata.affects_local_pyfa_profile:
                runtime.diagnostics.pop("pyfa_local_state_signature", None)

            if self._module_affects_pyfa_remote_inputs(module):
                pyfa_remote_inputs_dirty = True

        return pyfa_remote_inputs_dirty

    @classmethod
    def _projected_snapshot_structure_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(item[:-1] for item in cls._projected_snapshot_list_signature(snapshots))

    @staticmethod
    def _quantize_pyfa_projection_range(distance: float) -> float:
        safe_distance = max(0.0, float(distance or 0.0))
        if _PYFA_PROJECTION_RANGE_BUCKET_M <= 0.0:
            return safe_distance
        return math.floor(safe_distance / _PYFA_PROJECTION_RANGE_BUCKET_M) * _PYFA_PROJECTION_RANGE_BUCKET_M

    @classmethod
    def _normalized_snapshot_projection_signature(cls, snapshot: dict[str, Any]) -> tuple[str, Any]:
        projection_key_mode = str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range")
        if projection_key_mode == "exact_range":
            try:
                distance_signature: Any = round(
                    cls._quantize_pyfa_projection_range(
                        float(snapshot.get("pyfa_projection_range", snapshot.get("projection_range", 0.0)) or 0.0)
                    ),
                    3,
                )
            except Exception:
                distance_signature = 0.0
            return "exact_range", distance_signature
        return "in_range", None

    def _pyfa_projection_snapshot_params(self, module, target_snapshot: CycleTargetSnapshot) -> tuple[str, float]:
        projected_effects = [
            effect
            for effect_index, effect in enumerate(module.effects)
            if effect.effect_class == EffectClass.PROJECTED and effect_index in target_snapshot.active_effect_indices
        ]
        if not projected_effects:
            projected_effects = [effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED]
        falloff_effects = [
            effect
            for effect in projected_effects
            if max(0.0, float(getattr(effect, "falloff_m", 0.0) or 0.0)) > 0.0
        ]
        if not falloff_effects:
            return "in_range", 0.0
        if all(target_snapshot.distance <= max(0.0, float(getattr(effect, "range_m", 0.0) or 0.0)) for effect in falloff_effects):
            return "in_range", 0.0
        return "exact_range", self._quantize_pyfa_projection_range(target_snapshot.distance)

    @staticmethod
    def _runtime_state_rank(state: ModuleState) -> int:
        return {
            ModuleState.OFFLINE: 0,
            ModuleState.ONLINE: 1,
            ModuleState.ACTIVE: 2,
            ModuleState.OVERHEATED: 3,
        }.get(state, 0)

    @classmethod
    def _runtime_module_max_state(cls, runtime: FitRuntime | None, module_id: str) -> ModuleState:
        if runtime is None:
            return ModuleState.OVERHEATED
        raw_map = runtime.diagnostics.get("pyfa_max_state_by_module_id")
        if not isinstance(raw_map, dict):
            return ModuleState.OVERHEATED
        state_name = str(raw_map.get(str(module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
        if state_name in ModuleState.__members__:
            return ModuleState[state_name]
        return ModuleState.OVERHEATED

    @classmethod
    def _clamp_runtime_state_to_pyfa_max(cls, requested_state: ModuleState, max_state: ModuleState) -> ModuleState:
        return requested_state if cls._runtime_state_rank(requested_state) <= cls._runtime_state_rank(max_state) else max_state

    @classmethod
    def _runtime_inactive_module_state(cls, runtime: FitRuntime | None, module_id: str) -> ModuleState:
        max_state = cls._runtime_module_max_state(runtime, module_id)
        if cls._runtime_state_rank(max_state) < cls._runtime_state_rank(ModuleState.ONLINE):
            return ModuleState.OFFLINE
        return ModuleState.ONLINE

    @staticmethod
    def _copy_runtime_dynamic_state(source_runtime, target_runtime) -> None:
        raw_max_state_map = target_runtime.diagnostics.get("pyfa_max_state_by_module_id")
        max_state_map = raw_max_state_map if isinstance(raw_max_state_map, dict) else {}
        if len(source_runtime.modules) == len(target_runtime.modules):
            for source_module, target_module in zip(source_runtime.modules, target_runtime.modules):
                target_module.module_id = source_module.module_id
                max_state_name = str(max_state_map.get(str(target_module.module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
                max_state = ModuleState[max_state_name] if max_state_name in ModuleState.__members__ else ModuleState.OVERHEATED
                target_module.state = CombatSystem._clamp_runtime_state_to_pyfa_max(source_module.state, max_state)
                if source_module.charge_capacity > 0:
                    target_module.charge_remaining = max(
                        0.0,
                        min(float(source_module.charge_remaining), float(target_module.charge_capacity)),
                    )
            return

        source_by_module_id = {module.module_id: module for module in source_runtime.modules}
        for module in target_runtime.modules:
            source_module = source_by_module_id.get(module.module_id)
            if source_module is None:
                continue
            max_state_name = str(max_state_map.get(str(module.module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
            max_state = ModuleState[max_state_name] if max_state_name in ModuleState.__members__ else ModuleState.OVERHEATED
            module.state = CombatSystem._clamp_runtime_state_to_pyfa_max(source_module.state, max_state)
            if module.charge_capacity > 0:
                module.charge_remaining = max(0.0, min(float(source_module.charge_remaining), float(module.charge_capacity)))

    def _apply_runtime_activation_limit_transitions(
        self,
        world: WorldState,
        ship,
        source_runtime: FitRuntime,
        target_runtime: FitRuntime,
    ) -> bool:
        source_by_module_id = {str(module.module_id): module for module in source_runtime.modules}
        pyfa_remote_inputs_dirty = False

        for target_module in target_runtime.modules:
            module_id = str(target_module.module_id)
            source_module = source_by_module_id.get(module_id)
            if source_module is None:
                continue
            if source_module.state not in {ModuleState.ACTIVE, ModuleState.OVERHEATED}:
                continue
            if target_module.state in {ModuleState.ACTIVE, ModuleState.OVERHEATED}:
                continue

            previous_projected_target = ship.combat.projected_targets.get(module_id)
            self._flush_projected_cycle_total(world, ship.ship_id, target_module, previous_projected_target)
            self._clear_module_cycle_snapshots(ship.ship_id, module_id)
            self._clear_module_cycle_timer(ship, module_id)
            self._clear_module_reactivation_timer(ship, module_id)
            if target_module.state == ModuleState.OFFLINE:
                ship.combat.projected_targets.pop(module_id, None)
            if self._module_affects_pyfa_remote_inputs(target_module):
                pyfa_remote_inputs_dirty = True

        return pyfa_remote_inputs_dirty

    @staticmethod
    def _clone_resolved_runtime_for_ship(source_runtime, resolved_runtime) -> FitRuntime:
        # Clone only the mutable runtime shell. Avoid generic deepcopy so batched Pyfa refresh
        # does not recursively copy cached metadata and immutable fit graph fragments.
        diagnostics = {
            key: value
            for key, value in source_runtime.diagnostics.items()
            if key not in _RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS
        }
        diagnostics.update(
            {
                key: value
                for key, value in resolved_runtime.diagnostics.items()
                if key not in _RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS
            }
        )
        modules = [
            ModuleRuntime(
                module_id=str(module.module_id),
                group=str(module.group),
                state=module.state,
                effects=list(module.effects),
                charge_capacity=int(module.charge_capacity),
                charge_rate=float(module.charge_rate),
                charge_remaining=float(module.charge_remaining),
                charge_reload_time=float(module.charge_reload_time),
                tags=tuple(str(tag) for tag in getattr(module, "tags", ()) or ()),
            )
            for module in resolved_runtime.modules
        ]
        return FitRuntime(
            fit_key=str(source_runtime.fit_key),
            hull=resolved_runtime.hull,
            skills=resolved_runtime.skills,
            modules=modules,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _runtime_offline_module_signature(runtime) -> int:
        signature = 0
        for index, module in enumerate(runtime.modules):
            if module.state == module.state.OFFLINE:
                signature |= 1 << index
        return signature

    def _runtime_minimum_potential_cycle_time(self, runtime) -> float | None:
        signature = self._runtime_offline_module_signature(runtime)
        cached_signature = runtime.diagnostics.get("runtime_minimum_potential_cycle_signature")
        cached_minimum = runtime.diagnostics.get("runtime_minimum_potential_cycle_time")
        if cached_signature == signature:
            if cached_minimum is None:
                return None
            return float(cached_minimum)

        minimum: float | None = None
        for module, metadata in self._runtime_module_buckets(runtime).controlled_entries:
            if module.state == module.state.OFFLINE:
                continue
            cycle_time = metadata.cycle_time
            if cycle_time <= 0.0:
                continue
            if minimum is None or cycle_time < minimum:
                minimum = cycle_time

        runtime.diagnostics["runtime_minimum_potential_cycle_signature"] = signature
        runtime.diagnostics["runtime_minimum_potential_cycle_time"] = minimum
        return minimum

    def _minimum_potential_cycle_time(self, world: WorldState) -> float | None:
        minimum: float | None = None
        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue
            cycle_time = self._runtime_minimum_potential_cycle_time(ship.runtime)
            if cycle_time is None:
                continue
            if minimum is None or cycle_time < minimum:
                minimum = cycle_time
        return minimum

    def _decision_now(self, world: WorldState, fallback: float | None = None) -> float:
        if fallback is not None:
            return float(fallback)
        if self._decision_reference_time is not None:
            return float(self._decision_reference_time)
        return float(world.now)

    @staticmethod
    def _deadline_remaining(deadline: float | None, now: float) -> float | None:
        if deadline is None:
            return None
        try:
            return max(0.0, float(deadline) - float(now))
        except Exception:
            return None

    @staticmethod
    def _clear_lock_timer(ship, target_id: str) -> None:
        ship.combat.lock_timers.pop(target_id, None)
        ship.combat.lock_deadlines.pop(target_id, None)

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

    @staticmethod
    def _sync_deadline_view(deadline_map: dict[str, float], view_map: dict[str, float], now: float) -> None:
        for key, deadline in list(deadline_map.items()):
            remaining = CombatSystem._deadline_remaining(deadline, now)
            if remaining is None:
                continue
            view_map[str(key)] = remaining

    def _sync_timer_views_for_ship(self, ship, now: float) -> None:
        self._sync_deadline_view(ship.combat.lock_deadlines, ship.combat.lock_timers, now)
        self._sync_deadline_view(ship.combat.module_cycle_deadlines, ship.combat.module_cycle_timers, now)
        self._sync_deadline_view(ship.combat.module_ammo_reload_deadlines, ship.combat.module_ammo_reload_timers, now)
        self._sync_deadline_view(ship.combat.module_reactivation_deadlines, ship.combat.module_reactivation_timers, now)

    def _adopt_legacy_timer_views(self, ship, now: float) -> None:
        epsilon = 1e-6
        for target_id, remaining in list(ship.combat.lock_timers.items()):
            if target_id in ship.combat.lock_deadlines:
                continue
            try:
                remaining_float = max(0.0, float(remaining))
            except Exception:
                continue
            if remaining_float <= epsilon:
                continue
            self._schedule_lock_deadline(ship, target_id, duration=remaining_float, now=now)

        for module_id, remaining in list(ship.combat.module_cycle_timers.items()):
            if module_id in ship.combat.module_cycle_deadlines:
                continue
            try:
                remaining_float = max(0.0, float(remaining))
            except Exception:
                continue
            if remaining_float <= epsilon:
                continue
            self._schedule_module_cycle_deadline(ship, module_id, duration=remaining_float, now=now)

        for module_id, remaining in list(ship.combat.module_ammo_reload_timers.items()):
            if module_id in ship.combat.module_ammo_reload_deadlines:
                continue
            try:
                remaining_float = max(0.0, float(remaining))
            except Exception:
                continue
            if remaining_float <= epsilon:
                continue
            self._schedule_module_reload_deadline(ship, module_id, duration=remaining_float, now=now)

        for module_id, remaining in list(ship.combat.module_reactivation_timers.items()):
            if module_id in ship.combat.module_reactivation_deadlines:
                continue
            try:
                remaining_float = max(0.0, float(remaining))
            except Exception:
                continue
            if remaining_float <= epsilon:
                continue
            self._schedule_module_reactivation_deadline(ship, module_id, duration=remaining_float, now=now)

    def _prepare_ship_timer_views(self, ship, now: float) -> None:
        self._sync_timer_views_for_ship(ship, now)
        self._adopt_legacy_timer_views(ship, now)
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

    def recommended_time_slice(self, world: WorldState, max_dt: float) -> float:
        slice_dt = max(1e-6, float(max_dt))
        now = float(world.now)
        epsilon = 1e-6

        def note_duration(value: float | None) -> None:
            nonlocal slice_dt
            if value is None:
                return
            try:
                duration = float(value)
            except Exception:
                return
            if epsilon < duration < slice_dt:
                slice_dt = duration

        next_due_time = self._next_timer_event_time(world)
        if next_due_time is not None:
            note_duration(max(epsilon, float(next_due_time) - now))

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            for target_id, remaining in ship.combat.lock_timers.items():
                if target_id in ship.combat.lock_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_cycle_timers.items():
                if module_id in ship.combat.module_cycle_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_reactivation_timers.items():
                if module_id in ship.combat.module_reactivation_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_ammo_reload_timers.items():
                if module_id in ship.combat.module_ammo_reload_deadlines:
                    continue
                note_duration(remaining)

            for remaining in ship.combat.module_pending_ammo_reload_timers.values():
                note_duration(remaining)

            for ready_at in ship.combat.fire_delay_timers.values():
                note_duration(float(ready_at) - now)

            for jam_until in ship.combat.ecm_jam_sources.values():
                note_duration(float(jam_until) - now)

            raw_last_enemy_damage = getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9)
            last_enemy_damage = float(raw_last_enemy_damage if raw_last_enemy_damage is not None else -1e9)
            note_duration(30.0 - (now - last_enemy_damage))

        if abs(slice_dt - float(max_dt)) <= epsilon:
            note_duration(self._minimum_potential_cycle_time(world))

        return max(epsilon, min(slice_dt, float(max_dt)))

    def _log_event(self, event: str, **fields: Any) -> None:
        if not self.event_logging_enabled:
            return
        log_sim_event(self.logger, event, **fields)

    @staticmethod
    def _normalize_merge_value(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, (list, tuple, set)):
            return tuple(CombatSystem._normalize_merge_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), CombatSystem._normalize_merge_value(v)) for k, v in value.items()))
        return value

    def _queue_merged_event(
        self,
        event: str,
        merge_fields: dict[str, Any],
        sum_fields: dict[str, float] | None = None,
        count: int = 1,
    ) -> None:
        if not self.event_logging_enabled:
            return
        key = (event,) + tuple(
            (k, self._normalize_merge_value(v))
            for k, v in sorted(merge_fields.items())
        )
        bucket = self._merged_event_buckets.get(key)
        if bucket is None:
            bucket = {
                "event": event,
                "merge_fields": dict(merge_fields),
                "sum_fields": {},
                "count": 0,
            }
            self._merged_event_buckets[key] = bucket
        bucket["count"] = int(bucket["count"]) + max(1, int(count))
        if sum_fields:
            sums = bucket["sum_fields"]
            for field, value in sum_fields.items():
                sums[field] = float(sums.get(field, 0.0)) + float(value)

    def _flush_merged_events(self, window_start: float | None = None, window_end: float | None = None) -> None:
        if not self._merged_event_buckets:
            return
        for bucket in self._merged_event_buckets.values():
            payload = dict(bucket["merge_fields"])
            event_count = int(bucket.get("count", 0))
            if event_count > 1:
                payload["count"] = event_count
            for field, value in bucket.get("sum_fields", {}).items():
                payload[field] = float(value)
            if window_start is not None and window_end is not None:
                payload["window_start"] = float(window_start)
                payload["window_end"] = float(window_end)
            self._log_event(str(bucket["event"]), **payload)
        self._merged_event_buckets.clear()

    def _advance_merge_window(self, now: float) -> None:
        window = max(0.1, float(self.event_merge_window_sec))
        if self._merge_window_end_time is None or self._merge_window_start_time is None:
            self._merge_window_start_time = float(now)
            self._merge_window_end_time = float(now) + window
            return
        while now >= self._merge_window_end_time:
            self._flush_merged_events(self._merge_window_start_time, self._merge_window_end_time)
            self._merge_window_start_time = self._merge_window_end_time
            self._merge_window_end_time = self._merge_window_start_time + window

    def flush_pending_events(self) -> None:
        self._flush_merged_events(self._merge_window_start_time, self._merge_window_end_time)

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

    def _compute_projected_damage_factor(
        self,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        strength: float,
        distance: float,
    ) -> float:
        damage_factor = strength
        if float(effect.projected_add.get("weapon_is_turret", 0.0) or 0.0) > 0.5:
            relative_velocity = source.nav.velocity - target.nav.velocity
            radial = (target.nav.position - source.nav.position).normalized()
            tangential = Vector2(-radial.y, radial.x)
            transversal = abs(relative_velocity.x * tangential.x + relative_velocity.y * tangential.y)
            chance = self.pyfa.turret_chance_to_hit(
                tracking=max(0.0, float(effect.projected_add.get("weapon_tracking", 0.0) or 0.0)),
                optimal_sig=max(1.0, float(effect.projected_add.get("weapon_optimal_sig", 40_000.0) or 40_000.0)),
                distance=distance,
                optimal=effect.range_m,
                falloff=effect.falloff_m,
                transversal_speed=transversal,
                target_sig=target_profile.sig_radius,
                attacker_radius=source.nav.radius,
                target_radius=target.nav.radius,
            )
            damage_factor = max(0.0, self.pyfa.turret_damage_multiplier(chance))
        elif float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5:
            target_speed = target.nav.velocity.length()
            explosion_radius = max(0.0, float(effect.projected_add.get("weapon_explosion_radius", 0.0) or 0.0))
            explosion_velocity = max(0.0, float(effect.projected_add.get("weapon_explosion_velocity", 0.0) or 0.0))
            drf = max(0.1, float(effect.projected_add.get("weapon_drf", 0.5) or 0.5))
            if explosion_radius > 0.0:
                sig_factor = target_profile.sig_radius / max(1.0, explosion_radius)
                vel_term = (sig_factor * explosion_velocity) / max(1.0, target_speed)
                vel_factor = vel_term ** drf
                application = max(0.0, min(1.0, min(sig_factor, vel_factor, 1.0)))
            else:
                application = 1.0
            damage_factor = max(0.0, min(1.0, application * strength))
        return max(0.0, damage_factor)

    def _cycle_effect_damage_factor(
        self,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        effect_index: int,
        target_snapshot: CycleTargetSnapshot,
        strength: float,
    ) -> float | None:
        cached = target_snapshot.effect_damage_factors.get(effect_index)
        if cached is not None:
            return cached
        is_turret = float(effect.projected_add.get("weapon_is_turret", 0.0) or 0.0) > 0.5
        is_missile = float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5
        if not (is_turret or is_missile):
            return None
        damage_factor = self._compute_projected_damage_factor(
            source=source,
            target=target,
            target_profile=target_profile,
            effect=effect,
            strength=strength,
            distance=target_snapshot.distance,
        )
        target_snapshot.effect_damage_factors[effect_index] = damage_factor
        return damage_factor

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
            target = world.ships.get(projected_target_id)
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
            for layer in _REPAIR_QUEUE_LAYERS
            if abs(self._ship_layer_values(ship, layer)[0] - previous_values[layer][0]) > 1e-6
            or abs(self._ship_layer_values(ship, layer)[1] - previous_values[layer][1]) > 1e-6
        ]
        if changed_layers:
            self._mark_team_repair_queues_dirty(ship.team, *changed_layers)

    def _cached_lock_time(self, attacker_profile, defender_profile) -> float:
        key = (
            round(float(getattr(attacker_profile, "scan_resolution", 0.0) or 0.0), 4),
            round(float(getattr(defender_profile, "sig_radius", 0.0) or 0.0), 4),
        )
        cached = self._lock_time_cache.get(key)
        if cached is not None:
            return cached
        value = max(0.0, float(self.pyfa.calculate_lock_time(attacker_profile, defender_profile)))
        self._lock_time_cache[key] = value
        return value

    def _ensure_target_lock(
        self,
        world: WorldState,
        ship,
        target_id: str | None,
        target,
        *,
        lock_context: str,
        target_profile: ShipProfile | None = None,
        now: float | None = None,
    ) -> bool:
        if not target_id or target is None or not target.vital.alive:
            if target_id:
                ship.combat.lock_targets.discard(target_id)
                self._clear_lock_timer(ship, target_id)
            return False
        now_value = self._decision_now(world, now)
        if not self._can_target_under_ecm(ship, target_id, now_value):
            ship.combat.lock_targets.discard(target_id)
            self._clear_lock_timer(ship, target_id)
            return False
        if target_id in ship.combat.lock_targets:
            return True
        if target_id not in ship.combat.lock_deadlines and target_id not in ship.combat.lock_timers:
            profile_for_lock = target_profile if target_profile is not None else target.profile
            self._schedule_lock_deadline(
                ship,
                target_id,
                duration=self._cached_lock_time(ship.profile, profile_for_lock),
                now=now_value,
            )
            if self.detailed_logging and self.logger is not None:
                self.logger.debug(
                    f"{lock_context}_start source={ship.ship_id} target={target_id} lock_time={ship.combat.lock_timers[target_id]:.2f}"
                )
        elif target_id not in ship.combat.lock_deadlines:
            remaining = max(0.0, float(ship.combat.lock_timers.get(target_id, 0.0) or 0.0))
            if remaining > 0.0:
                self._schedule_lock_deadline(ship, target_id, duration=remaining, now=now_value)
        return False

    def _advance_target_locks(self, world: WorldState, dt: float, now: float | None = None) -> None:
        now_value = self._decision_now(world, now)
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            if self._ship_in_warp(ship):
                ship.combat.lock_targets.clear()
                ship.combat.lock_timers.clear()
                ship.combat.lock_deadlines.clear()
                continue
            self._prepare_ship_timer_views(ship, now_value)
            for target_id, left in list(ship.combat.lock_timers.items()):
                target = world.ships.get(target_id)
                if (
                    target is None
                    or not target.vital.alive
                    or self._ship_in_warp(target)
                    or not self._can_target_under_ecm(ship, target_id, now_value)
                ):
                    self._clear_lock_timer(ship, target_id)
                    ship.combat.lock_targets.discard(target_id)
                    continue
                if target_id in ship.combat.lock_deadlines:
                    if float(left) > 0.0:
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(
                                f"lock_progress attacker={ship.ship_id} target={target_id} remaining={float(left):.2f}"
                            )
                        continue
                else:
                    left = float(left) - dt
                    if left > 0.0:
                        ship.combat.lock_timers[target_id] = left
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(
                                f"lock_progress attacker={ship.ship_id} target={target_id} remaining={left:.2f}"
                            )
                        continue
                if float(ship.combat.lock_timers.get(target_id, 0.0) or 0.0) <= 0.0:
                    ship.combat.lock_targets.add(target_id)
                    self._clear_lock_timer(ship, target_id)
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(f"lock_complete attacker={ship.ship_id} target={target_id}")

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
    def _vector_from_facing_deg(facing_deg: float) -> Vector2:
        radians = math.radians(float(facing_deg or 0.0))
        return Vector2(math.cos(radians), math.sin(radians))

    @staticmethod
    def _projectile_damage_tuple(projectile: ProjectileEntity) -> DamageTuple:
        return (
            max(0.0, float(projectile.damage_em or 0.0)),
            max(0.0, float(projectile.damage_thermal or 0.0)),
            max(0.0, float(projectile.damage_kinetic or 0.0)),
            max(0.0, float(projectile.damage_explosive or 0.0)),
        )

    @staticmethod
    def _effect_damage_tuple(effect) -> DamageTuple:
        return (
            max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
        )

    @staticmethod
    def _projectile_acceleration_time(flight_time: float, mass: float, agility: float) -> float:
        if flight_time <= 0.0 or mass <= 0.0 or agility <= 0.0:
            return 0.0
        return max(0.0, min(float(flight_time), (float(mass) * float(agility)) / 1_000_000.0))

    @classmethod
    def _projectile_max_range(
        cls,
        *,
        max_speed: float,
        flight_time: float,
        acceleration_time: float,
        fallback_range: float,
    ) -> float:
        if max_speed <= 0.0 or flight_time <= 0.0:
            return max(0.0, float(fallback_range or 0.0))
        accel_time = max(0.0, min(float(acceleration_time), float(flight_time)))
        during_acceleration = max_speed * 0.5 * accel_time
        at_full_speed = max_speed * max(0.0, float(flight_time) - accel_time)
        return max(max(0.0, float(fallback_range or 0.0)), during_acceleration + at_full_speed)

    @classmethod
    def _projectile_distance_for_interval(
        cls,
        *,
        max_speed: float,
        acceleration_time: float,
        start_age: float,
        dt: float,
    ) -> float:
        if max_speed <= 0.0 or dt <= 0.0:
            return 0.0
        if acceleration_time <= 1e-9:
            return max_speed * dt

        start = max(0.0, float(start_age))
        end = start + max(0.0, float(dt))
        if start >= acceleration_time:
            return max_speed * (end - start)

        accel_end = min(end, acceleration_time)
        start_speed = max_speed * max(0.0, min(1.0, start / acceleration_time))
        end_speed = max_speed * max(0.0, min(1.0, accel_end / acceleration_time))
        distance = 0.5 * (start_speed + end_speed) * max(0.0, accel_end - start)
        if end > acceleration_time:
            distance += max_speed * (end - acceleration_time)
        return max(0.0, distance)

    @classmethod
    def _projectile_speed_for_age(cls, *, max_speed: float, acceleration_time: float, age: float) -> float:
        if max_speed <= 0.0:
            return 0.0
        if acceleration_time <= 1e-9:
            return max_speed
        return max_speed * max(0.0, min(1.0, float(age) / float(acceleration_time)))

    def _create_projectile_blast(
        self,
        world: WorldState,
        *,
        kind: str,
        position: Vector2,
        radius_m: float,
        duration: float = 1.0,
    ) -> None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return
        self._projectile_blast_seq += 1
        blast_id = f"blast:{self._projectile_blast_seq}"
        world.projectile_blasts[blast_id] = ProjectileBlast(
            blast_id=blast_id,
            kind=str(kind),
            position=Vector2(position.x, position.y),
            radius_m=radius,
            expires_at=float(world.now) + max(0.1, float(duration)),
        )

    @staticmethod
    def _clamp_projectile_layer_hp(projectile: ProjectileEntity) -> None:
        projectile.shield_max = max(0.0, float(projectile.shield_max or 0.0))
        projectile.armor_max = max(0.0, float(projectile.armor_max or 0.0))
        projectile.structure_max = max(0.0, float(projectile.structure_max or 0.0))
        projectile.shield = max(0.0, min(float(projectile.shield or 0.0), projectile.shield_max))
        projectile.armor = max(0.0, min(float(projectile.armor or 0.0), projectile.armor_max))
        projectile.structure = max(0.0, min(float(projectile.structure or 0.0), projectile.structure_max))
        projectile.alive = (projectile.shield + projectile.armor + projectile.structure) > 1e-6

    def _apply_damage_to_projectile(
        self,
        projectile: ProjectileEntity,
        damage: DamageTuple,
        *,
        damage_factor: float = 1.0,
    ) -> bool:
        scaled_damage = _scale_damage(damage, max(0.0, float(damage_factor)))
        if _sum_damage(scaled_damage) <= 0.0:
            return not projectile.alive

        self._clamp_projectile_layer_hp(projectile)
        remaining = scaled_damage
        layer_specs = (
            (
                "shield",
                float(projectile.shield),
                (
                    float(projectile.shield_resonance_em),
                    float(projectile.shield_resonance_thermal),
                    float(projectile.shield_resonance_kinetic),
                    float(projectile.shield_resonance_explosive),
                ),
            ),
            (
                "armor",
                float(projectile.armor),
                (
                    float(projectile.armor_resonance_em),
                    float(projectile.armor_resonance_thermal),
                    float(projectile.armor_resonance_kinetic),
                    float(projectile.armor_resonance_explosive),
                ),
            ),
            (
                "structure",
                float(projectile.structure),
                (
                    float(projectile.structure_resonance_em),
                    float(projectile.structure_resonance_thermal),
                    float(projectile.structure_resonance_kinetic),
                    float(projectile.structure_resonance_explosive),
                ),
            ),
        )
        new_values = {
            "shield": float(projectile.shield),
            "armor": float(projectile.armor),
            "structure": float(projectile.structure),
        }
        for layer_name, layer_hp, layer_resonances in layer_specs:
            if layer_hp <= 0.0:
                continue
            effective_damage = _layer_effective_damage(remaining, layer_resonances)
            if effective_damage <= 0.0:
                continue
            if effective_damage <= layer_hp:
                new_values[layer_name] = layer_hp - effective_damage
                remaining = (0.0, 0.0, 0.0, 0.0)
                break
            consumed_ratio = max(0.0, min(1.0, layer_hp / effective_damage))
            new_values[layer_name] = 0.0
            remaining = _scale_damage(remaining, 1.0 - consumed_ratio)

        projectile.shield = max(0.0, new_values["shield"])
        projectile.armor = max(0.0, new_values["armor"])
        projectile.structure = max(0.0, new_values["structure"])
        projectile.alive = (projectile.shield + projectile.armor + projectile.structure) > 1e-6
        return not projectile.alive

    def _destroy_projectiles_in_area(
        self,
        world: WorldState,
        *,
        center: Vector2,
        radius_m: float,
        exclude_ids: set[str] | None = None,
        damage: DamageTuple | None = None,
        damage_factor: float = 1.0,
    ) -> None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return
        excluded = exclude_ids or set()
        hit_radius = radius + 25.0
        for projectile_id, projectile in list(world.projectiles.items()):
            if projectile_id in excluded:
                continue
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)
                continue
            if projectile.position.distance_to(center) > hit_radius:
                continue
            if damage is None:
                projectile.alive = False
            else:
                self._apply_damage_to_projectile(
                    projectile,
                    damage,
                    damage_factor=damage_factor,
                )
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)

    def _apply_direct_damage(
        self,
        world: WorldState,
        *,
        source,
        target,
        target_profile: ShipProfile,
        damage: DamageTuple,
        damage_factor: float,
    ) -> float:
        self._clamp_ship_layer_hp(target)
        scaled_damage = _scale_damage(damage, max(0.0, float(damage_factor)))
        total_damage = _sum_damage(scaled_damage)
        if total_damage <= 0.0:
            return 0.0

        shield_before = target.vital.shield
        armor_before = target.vital.armor
        structure_before = target.vital.structure
        alive_before = bool(target.vital.alive)

        target.vital.shield, target.vital.armor, target.vital.structure = _apply_damage_sequence(
            target.vital.shield,
            target.vital.armor,
            target.vital.structure,
            scaled_damage,
            target_profile,
        )
        dirty_layers: list[str] = []
        if abs(target.vital.shield - shield_before) > 1e-6:
            dirty_layers.append("shield")
        if abs(target.vital.armor - armor_before) > 1e-6:
            dirty_layers.append("armor")
        if abs(target.vital.structure - structure_before) > 1e-6:
            dirty_layers.append("structure")

        applied = (shield_before + armor_before + structure_before) - (
            target.vital.shield + target.vital.armor + target.vital.structure
        )
        if applied > 0.0:
            target.combat.last_damaged_at = float(world.now)
            if source is not None and source.team != target.team:
                target.combat.last_enemy_weapon_damaged_at = float(world.now)
        if target.vital.structure <= 0.0:
            target.vital.alive = False
            target.nav.velocity = Vector2(0.0, 0.0)
        if dirty_layers or (alive_before and not target.vital.alive):
            self._mark_team_repair_queues_dirty(target.team, *(dirty_layers or _REPAIR_QUEUE_LAYERS))
        return max(0.0, applied)

    def _missile_damage_factor(self, projectile: ProjectileEntity, target, target_profile: ShipProfile) -> float:
        explosion_radius = max(0.0, float(projectile.explosion_radius or 0.0))
        if explosion_radius <= 0.0:
            return 1.0
        sig_factor = float(target_profile.sig_radius or 0.0) / max(1.0, explosion_radius)
        explosion_velocity = max(0.0, float(projectile.explosion_velocity or 0.0))
        target_speed = target.nav.velocity.length()
        if target_speed <= 1e-9 or explosion_velocity <= 0.0:
            velocity_factor = 1.0
        else:
            velocity_factor = ((sig_factor * explosion_velocity) / max(1.0, target_speed)) ** max(
                0.1,
                float(projectile.damage_reduction_factor or 0.5),
            )
        return max(0.0, min(1.0, min(1.0, sig_factor, velocity_factor)))

    def _bomb_damage_factor(self, projectile: ProjectileEntity, target_profile: ShipProfile) -> float:
        explosion_radius = max(0.0, float(projectile.explosion_radius or 0.0))
        if explosion_radius <= 0.0:
            return 1.0
        return max(0.0, min(1.0, float(target_profile.sig_radius or 0.0) / explosion_radius))

    def _spawn_projectile(
        self,
        world: WorldState,
        *,
        source,
        module,
        metadata: ModuleStaticMetadata,
        effect,
        target_id: str | None,
    ) -> None:
        max_speed = max(0.0, float(effect.projected_add.get("weapon_projectile_speed", 0.0) or 0.0))
        flight_time = max(0.0, float(effect.projected_add.get("weapon_projectile_flight_time", 0.0) or 0.0))
        acceleration_time = self._projectile_acceleration_time(
            flight_time,
            float(effect.projected_add.get("weapon_projectile_mass", 0.0) or 0.0),
            float(effect.projected_add.get("weapon_projectile_agility", 0.0) or 0.0),
        )
        max_range = self._projectile_max_range(
            max_speed=max_speed,
            flight_time=flight_time,
            acceleration_time=acceleration_time,
            fallback_range=float(effect.range_m or 0.0),
        )
        if max_speed <= 0.0 and flight_time > 0.0 and max_range > 0.0:
            max_speed = max_range / max(0.1, flight_time)
        if flight_time <= 0.0 and max_speed > 0.0 and max_range > 0.0:
            flight_time = max_range / max(1.0, max_speed)
        if max_range <= 0.0 and max_speed <= 0.0:
            return

        target = world.ships.get(target_id) if target_id else None
        if target is not None and target.vital.alive:
            direction = (target.nav.position - source.nav.position).normalized()
        else:
            direction = self._vector_from_facing_deg(source.nav.facing_deg)
        if direction.length() <= 1e-9:
            direction = Vector2(1.0, 0.0)

        initial_speed = self._projectile_speed_for_age(
            max_speed=max_speed,
            acceleration_time=acceleration_time,
            age=0.0,
        )
        kind = "bomb" if metadata.is_bomb_launcher else "missile"
        self._projectile_seq += 1
        projectile_id = f"proj:{self._projectile_seq}"
        shield_max = max(0.0, float(effect.projected_add.get("weapon_projectile_shield_hp", 0.0) or 0.0))
        armor_max = max(0.0, float(effect.projected_add.get("weapon_projectile_armor_hp", 0.0) or 0.0))
        structure_max = max(0.0, float(effect.projected_add.get("weapon_projectile_structure_hp", 0.0) or 0.0))
        if (shield_max + armor_max + structure_max) <= 0.0:
            structure_max = 1.0
        world.projectiles[projectile_id] = ProjectileEntity(
            projectile_id=projectile_id,
            kind=kind,
            source_ship_id=str(source.ship_id),
            source_module_id=str(module.module_id),
            team=source.team,
            position=Vector2(source.nav.position.x, source.nav.position.y),
            velocity=direction * initial_speed,
            facing_deg=direction.angle_deg(),
            target_ship_id=str(target_id) if target_id else None,
            speed=initial_speed,
            max_speed=max_speed,
            max_range=max_range,
            distance_traveled=0.0,
            flight_time=max(0.0, flight_time),
            age=0.0,
            acceleration_time=acceleration_time,
            damage_em=max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            damage_thermal=max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            damage_kinetic=max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            damage_explosive=max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
            explosion_radius=max(0.0, float(effect.projected_add.get("weapon_explosion_radius", 0.0) or 0.0)),
            explosion_velocity=max(0.0, float(effect.projected_add.get("weapon_explosion_velocity", 0.0) or 0.0)),
            damage_reduction_factor=max(0.1, float(effect.projected_add.get("weapon_drf", 0.5) or 0.5)),
            shield=shield_max,
            armor=armor_max,
            structure=structure_max,
            shield_max=shield_max,
            armor_max=armor_max,
            structure_max=structure_max,
            shield_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_em", 1.0) or 1.0))),
            shield_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_thermal", 1.0) or 1.0))),
            shield_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_kinetic", 1.0) or 1.0))),
            shield_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_explosive", 1.0) or 1.0))),
            armor_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_em", 1.0) or 1.0))),
            armor_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_thermal", 1.0) or 1.0))),
            armor_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_kinetic", 1.0) or 1.0))),
            armor_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_explosive", 1.0) or 1.0))),
            structure_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_em", 1.0) or 1.0))),
            structure_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_thermal", 1.0) or 1.0))),
            structure_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_kinetic", 1.0) or 1.0))),
            structure_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_explosive", 1.0) or 1.0))),
            blast_radius=max(0.0, float(effect.projected_add.get("weapon_blast_radius", 0.0) or 0.0)),
            alive=True,
        )

    def _spawn_cycle_projectiles(
        self,
        world: WorldState,
        *,
        source,
        module,
        metadata: ModuleStaticMetadata,
        target_id: str | None,
    ) -> None:
        if not (metadata.is_missile_weapon or metadata.is_bomb_launcher):
            return
        for _effect_index, effect in metadata.projected_effects:
            damage_total = sum(
                max(0.0, float(effect.projected_add.get(key, 0.0) or 0.0))
                for key in ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive")
            )
            if damage_total <= 0.0:
                continue
            self._spawn_projectile(
                world,
                source=source,
                module=module,
                metadata=metadata,
                effect=effect,
                target_id=target_id,
            )

    def _resolve_projectile_hit(self, world: WorldState, projectile: ProjectileEntity) -> None:
        target = world.ships.get(projectile.target_ship_id or "")
        if target is None or not target.vital.alive:
            return
        source = world.ships.get(projectile.source_ship_id)
        damage_factor = self._missile_damage_factor(projectile, target, target.profile)
        self._apply_direct_damage(
            world,
            source=source,
            target=target,
            target_profile=target.profile,
            damage=self._projectile_damage_tuple(projectile),
            damage_factor=damage_factor,
        )

    def _resolve_bomb_explosion(self, world: WorldState, projectile: ProjectileEntity) -> None:
        source = world.ships.get(projectile.source_ship_id)
        blast_radius = max(0.0, float(projectile.blast_radius or 0.0))
        if blast_radius <= 0.0:
            return
        self._create_projectile_blast(
            world,
            kind="bomb",
            position=projectile.position,
            radius_m=blast_radius,
        )
        for target in world.ships.values():
            if not target.vital.alive:
                continue
            if projectile.position.distance_to(target.nav.position) > (blast_radius + max(0.0, float(target.nav.radius or 0.0))):
                continue
            damage_factor = self._bomb_damage_factor(projectile, target.profile)
            self._apply_direct_damage(
                world,
                source=source,
                target=target,
                target_profile=target.profile,
                damage=self._projectile_damage_tuple(projectile),
                damage_factor=damage_factor,
            )
        self._destroy_projectiles_in_area(
            world,
            center=projectile.position,
            radius_m=blast_radius,
            exclude_ids={projectile.projectile_id},
            damage=self._projectile_damage_tuple(projectile),
        )

    def _advance_projectiles(self, world: WorldState, dt: float) -> None:
        if dt <= 0.0:
            return
        for blast_id, blast in list(world.projectile_blasts.items()):
            if float(blast.expires_at) <= float(world.now):
                world.projectile_blasts.pop(blast_id, None)

        for projectile_id, projectile in list(world.projectiles.items()):
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)
                continue

            target = world.ships.get(projectile.target_ship_id or "")
            if projectile.kind == "missile" and target is not None and target.vital.alive:
                direction = (target.nav.position - projectile.position).normalized()
                if direction.length() > 1e-9:
                    projectile.facing_deg = direction.angle_deg()
                else:
                    direction = self._vector_from_facing_deg(projectile.facing_deg)
            else:
                direction = projectile.velocity.normalized()
                if direction.length() <= 1e-9:
                    direction = self._vector_from_facing_deg(projectile.facing_deg)
            if direction.length() <= 1e-9:
                direction = Vector2(1.0, 0.0)

            step_distance = self._projectile_distance_for_interval(
                max_speed=float(projectile.max_speed or 0.0),
                acceleration_time=float(projectile.acceleration_time or 0.0),
                start_age=float(projectile.age or 0.0),
                dt=dt,
            )
            target_contact_distance = None
            if projectile.kind == "missile" and target is not None and target.vital.alive:
                target_contact_distance = max(
                    0.0,
                    projectile.position.distance_to(target.nav.position) - max(0.0, float(target.nav.radius or 0.0)),
                )

            if target_contact_distance is not None and target_contact_distance <= step_distance + 1e-6:
                projectile.position = Vector2(target.nav.position.x, target.nav.position.y)
                projectile.distance_traveled += max(0.0, target_contact_distance)
                projectile.age += dt
                self._resolve_projectile_hit(world, projectile)
                projectile.alive = False
                world.projectiles.pop(projectile_id, None)
                continue

            projectile.position = projectile.position + (direction * step_distance)
            projectile.distance_traveled += max(0.0, step_distance)
            projectile.age += dt
            projectile.speed = self._projectile_speed_for_age(
                max_speed=float(projectile.max_speed or 0.0),
                acceleration_time=float(projectile.acceleration_time or 0.0),
                age=float(projectile.age or 0.0),
            )
            projectile.velocity = direction * projectile.speed

            expired_by_time = projectile.flight_time > 0.0 and float(projectile.age) >= float(projectile.flight_time) - 1e-6
            expired_by_range = projectile.max_range > 0.0 and float(projectile.distance_traveled) >= float(projectile.max_range) - 1e-6
            if projectile.kind == "missile":
                if target is not None and target.vital.alive:
                    contact_distance = max(
                        0.0,
                        projectile.position.distance_to(target.nav.position) - max(0.0, float(target.nav.radius or 0.0)),
                    )
                    if contact_distance <= 1e-6:
                        self._resolve_projectile_hit(world, projectile)
                        projectile.alive = False
                        world.projectiles.pop(projectile_id, None)
                        continue
                if expired_by_time or expired_by_range:
                    projectile.alive = False
                    world.projectiles.pop(projectile_id, None)
                    continue
            else:
                if expired_by_time or expired_by_range:
                    self._resolve_bomb_explosion(world, projectile)
                    projectile.alive = False
                    world.projectiles.pop(projectile_id, None)

    def _collect_projected_impacts(self, world: WorldState, dt: float) -> dict[str, list[ProjectedImpact]]:
        del dt
        impacts: dict[str, list[ProjectedImpact]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_in_warp(source):
                continue
            for module, metadata in self._runtime_module_buckets(source.runtime).runtime_projected_entries:
                if metadata.is_missile_weapon or metadata.is_bomb_launcher:
                    continue
                for effect_index, effect in metadata.projected_effects:
                    if not module.is_active_for(effect.state_required):
                        continue

                    target_id = source.combat.projected_targets.get(module.module_id)
                    if not target_id:
                        continue
                    target = world.ships.get(target_id)
                    if target is None or not target.vital.alive or self._ship_in_warp(target):
                        continue

                    if not self._ensure_target_lock(
                        world,
                        source,
                        target_id,
                        target,
                        lock_context="projected_lock",
                    ):
                        continue

                    target_snapshot = self._module_cycle_snapshot_for_target(source.ship_id, module.module_id, target_id)
                    if target_snapshot is None:
                        continue
                    strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                    if strength <= 0:
                        continue
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(
                            f"projected_formula source={source.ship_id} target={target_id} module={module.module_id} dist={target_snapshot.distance:.1f} range={effect.range_m:.1f} falloff={effect.falloff_m:.1f} strength={strength:.4f}"
                        )
                    impacts.setdefault(target_id, []).append(
                        ProjectedImpact(source_ship_id=source.ship_id, target_ship_id=target_id, effect=effect, strength=strength)
                    )
        return impacts

    @staticmethod
    def _hp_ratio(ship) -> float:
        hp_max = max(1.0, ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max)
        hp_now = ship.vital.shield + ship.vital.armor + ship.vital.structure
        return hp_now / hp_max

    @staticmethod
    def _ship_layer_values(ship, layer: str) -> tuple[float, float]:
        if layer == "shield":
            return float(ship.vital.shield), max(1.0, float(ship.vital.shield_max))
        if layer == "armor":
            return float(ship.vital.armor), max(1.0, float(ship.vital.armor_max))
        return float(ship.vital.structure), max(1.0, float(ship.vital.structure_max))

    @classmethod
    def _ship_layer_fraction(cls, ship, layer: str) -> float:
        current, maximum = cls._ship_layer_values(ship, layer)
        return max(0.0, min(1.0, current / maximum))

    @classmethod
    def _ship_needs_layer_repair(cls, ship, layer: str) -> bool:
        if not ship.vital.alive:
            return False
        current, maximum = cls._ship_layer_values(ship, layer)
        return (maximum - current) > 1e-6

    def _team_repair_queue(self, world: WorldState, team: Team, layer: str) -> tuple[str, ...]:
        cache_key = (team, layer)
        if cache_key not in self._repair_queue_dirty:
            cached = self._repair_queue_cache.get(cache_key)
            if cached is not None:
                return cached

        ranked: list[tuple[float, str]] = []
        for ship in world.ships.values():
            if ship.team != team:
                continue
            if self._ship_in_warp(ship):
                continue
            if not self._ship_needs_layer_repair(ship, layer):
                continue
            ranked.append((self._ship_layer_fraction(ship, layer), str(ship.ship_id)))
        ranked.sort(key=lambda item: (item[0], item[1]))
        queue = tuple(ship_id for _fraction, ship_id in ranked)
        self._repair_queue_cache[cache_key] = queue
        self._repair_queue_dirty.discard(cache_key)
        return queue

    def _select_repair_queue_target(self, world: WorldState, source, module, metadata: ModuleStaticMetadata) -> str | None:
        for layer in metadata.repair_layers:
            for target_id in self._team_repair_queue(world, source.team, layer):
                if target_id == source.ship_id:
                    continue
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive or self._ship_in_warp(target) or target.team != source.team:
                    continue
                if not self._ship_needs_layer_repair(target, layer):
                    continue
                if not self._module_in_projected_range(source, target, module):
                    continue
                return target_id
        return None

    @staticmethod
    def _module_in_projected_range(source, target, module) -> bool:
        distance = source.nav.position.distance_to(target.nav.position)
        has_projected = False
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            has_projected = True
            max_range = CombatSystem._projected_max_range(effect)
            if max_range <= 0 or distance <= max_range:
                return True
        return not has_projected

    @staticmethod
    def _cap_ratio(ship) -> float:
        return max(0.0, float(ship.vital.cap) / max(1.0, float(ship.vital.cap_max)))

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

    def _candidates_in_projected_range(self, source, module, candidates: list) -> list:
        return [
            candidate
            for candidate in candidates
            if candidate.vital.alive
            and not self._ship_in_warp(candidate)
            and self._module_in_projected_range(source, candidate, module)
        ]

    def _iter_area_targets_in_range(self, world: WorldState, source, module, effect, *, candidates: list | None = None) -> list:
        targets: list = []
        metadata = self._module_static_metadata(module)
        include_self = metadata.is_command_burst
        same_team_only = metadata.is_command_burst
        max_range = self._projected_max_range(effect)

        candidate_iterable = candidates if candidates is not None else world.ships.values()
        for candidate in candidate_iterable:
            if not candidate.vital.alive:
                continue
            if self._ship_in_warp(candidate):
                continue
            if candidate.ship_id == source.ship_id and not include_self:
                continue
            if same_team_only and candidate.team != source.team:
                continue
            distance = source.nav.position.distance_to(candidate.nav.position)
            if max_range > 0.0 and distance > max_range:
                continue
            targets.append(candidate)

        return targets

    def _collect_command_booster_snapshots(self, world: WorldState) -> dict[str, list[dict[str, Any]]]:
        snapshots_by_ship: dict[str, list[dict[str, Any]]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_in_warp(source):
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue

            command_entries = self._runtime_module_buckets(source.runtime).command_entries
            if not command_entries:
                continue

            base_state_by_module_id: dict[str, str] = {}
            active_state_by_module_id: dict[str, str] = {}
            active_targets_by_module_id: dict[str, set[str]] = {}
            covered_targets: set[str] = set()

            for module, _metadata in command_entries:
                state_value = str(module.state.value or "ONLINE").upper()
                base_state_by_module_id[module.module_id] = "ONLINE" if state_value in {"ACTIVE", "OVERHEATED"} else state_value
                if state_value not in {"ACTIVE", "OVERHEATED"}:
                    continue

                target_ids = {
                    target_id
                    for target_id in self._module_cycle_snapshots_for(source.ship_id, module.module_id)
                    if (target := world.ships.get(target_id)) is not None
                    and target.vital.alive
                    and not self._ship_in_warp(target)
                    and target.team == source.team
                    and target.runtime is not None
                }
                if not target_ids:
                    continue

                active_state_by_module_id[module.module_id] = state_value
                active_targets_by_module_id[module.module_id] = target_ids
                covered_targets.update(target_ids)

            for target_id in sorted(covered_targets):
                state_by_module_id = dict(base_state_by_module_id)
                has_active_in_range = False
                for module_id, target_ids in active_targets_by_module_id.items():
                    if target_id not in target_ids:
                        continue
                    state_by_module_id[module_id] = active_state_by_module_id[module_id]
                    has_active_in_range = True

                if not has_active_in_range:
                    continue

                snapshots_by_ship.setdefault(target_id, []).append(
                    {
                        "fit_key": str(source.runtime.fit_key or ""),
                        "blueprint": blueprint,
                        "state_by_module_id": state_by_module_id,
                    }
                )

        return snapshots_by_ship

    def _collect_projected_source_snapshots(
        self,
        world: WorldState,
        command_boosters_by_ship: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        snapshots_by_ship: dict[str, list[dict[str, Any]]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_in_warp(source):
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue
            blueprint_modules_raw = blueprint.get("modules")
            blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
            blueprint_modules_by_id = {
                str(raw.get("module_id", "") or ""): raw
                for raw in blueprint_modules
                if isinstance(raw, dict)
            }

            source_command_snapshots = command_boosters_by_ship.get(source.ship_id, [])
            base_state_by_module_id: dict[str, str] = {}
            active_projected_modules: list[tuple[Any, str]] = []
            projected_module_ids: set[str] = set()

            for module, metadata in zip(source.runtime.modules, self._runtime_module_metadata_list(source.runtime)):
                state_value = str(module.state.value or "ONLINE").upper()
                projected_state = state_value
                if metadata.has_projected:
                    projected_module_ids.add(str(module.module_id))

                if state_value in {"ACTIVE", "OVERHEATED"}:
                    if metadata.is_command_burst:
                        projected_state = state_value
                    elif metadata.uses_pyfa_projected_profile:
                        projected_state = "ONLINE"
                    elif metadata.is_area_effect or metadata.is_weapon or metadata.has_projected_rep or metadata.is_cap_warfare:
                        projected_state = "ONLINE"

                base_state_by_module_id[module.module_id] = projected_state

                if metadata.uses_pyfa_projected_profile and state_value in {"ACTIVE", "OVERHEATED"}:
                    active_projected_modules.append((module, state_value))

            for active_projected_module, active_state in active_projected_modules:
                target_id = source.combat.projected_targets.get(active_projected_module.module_id)
                if not target_id:
                    continue
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive or self._ship_in_warp(target) or target.runtime is None:
                    continue
                target_snapshot = self._module_cycle_snapshot_for_target(source.ship_id, active_projected_module.module_id, target_id)
                if target_snapshot is None:
                    continue
                projection_key_mode, projection_range = self._pyfa_projection_snapshot_params(
                    active_projected_module,
                    target_snapshot,
                )

                state_by_module_id = dict(base_state_by_module_id)
                for projected_module_id in projected_module_ids:
                    if projected_module_id != str(active_projected_module.module_id):
                        state_by_module_id[projected_module_id] = "OFFLINE"
                state_by_module_id[active_projected_module.module_id] = active_state
                snapshots_by_ship.setdefault(target_id, []).append(
                    {
                        "fit_key": f"{source.runtime.fit_key}:{active_projected_module.module_id}",
                        "blueprint": blueprint,
                        "state_by_module_id": state_by_module_id,
                        "command_booster_snapshots": source_command_snapshots,
                        "pyfa_projection_key_mode": projection_key_mode,
                        "pyfa_projection_range": projection_range,
                        "projection_range": projection_range,
                        "pyfa_projection_module_signature": self._projected_module_runtime_signature(
                            active_projected_module,
                            blueprint_modules_by_id.get(str(active_projected_module.module_id)),
                            active_state,
                            active_effect_indices=target_snapshot.active_effect_indices,
                        ),
                    }
                )

        return snapshots_by_ship

    def _refresh_effective_runtimes_from_pyfa(
        self,
        world: WorldState,
        command_boosters_by_ship: dict[str, list[dict[str, Any]]],
        projected_sources_by_ship: dict[str, list[dict[str, Any]]],
    ) -> None:
        pending_batches: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            booster_snapshots = command_boosters_by_ship.get(ship.ship_id, [])
            projected_snapshots = projected_sources_by_ship.get(ship.ship_id, [])
            cache_key = get_runtime_resolve_cache_key(ship.runtime, booster_snapshots, projected_snapshots)
            local_signature = self._local_runtime_state_signature_from_metadata(ship.runtime)
            booster_signature = self._command_snapshot_list_signature(booster_snapshots)
            projected_signature = self._projected_snapshot_list_signature(projected_snapshots)
            projected_structure_signature = self._projected_snapshot_structure_signature(projected_snapshots)
            cached_signature = ship.runtime.diagnostics.get("pyfa_resolve_signature")
            cached_base_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
            if cache_key is not None and cached_signature == cache_key and isinstance(cached_base_profile, ShipProfile):
                if local_signature is not None:
                    ship.runtime.diagnostics["pyfa_local_state_signature"] = local_signature
                ship.runtime.diagnostics["pyfa_command_boosters"] = booster_snapshots
                ship.runtime.diagnostics["pyfa_projected_sources"] = projected_snapshots
                ship.runtime.diagnostics["pyfa_command_booster_signature"] = booster_signature
                ship.runtime.diagnostics["pyfa_projected_sources_signature"] = projected_signature
                ship.runtime.diagnostics["pyfa_projected_sources_structure_signature"] = projected_structure_signature
                ship.profile = cached_base_profile
                continue

            batch_key = cache_key if cache_key is not None else ("ship", ship.ship_id)
            pending_batches.setdefault(batch_key, []).append(
                {
                    "ship": ship,
                    "runtime": ship.runtime,
                    "booster_snapshots": booster_snapshots,
                    "projected_snapshots": projected_snapshots,
                    "booster_signature": booster_signature,
                    "projected_signature": projected_signature,
                    "local_signature": local_signature,
                    "cache_key": cache_key,
                }
            )

        for pending_group in pending_batches.values():
            first_pending = pending_group[0]
            resolve_started = time.perf_counter()
            resolved = resolve_runtime_from_pyfa_runtime(
                first_pending["runtime"],
                first_pending["booster_snapshots"],
                first_pending["projected_snapshots"],
            )
            resolve_cache = "error"
            projected_fit_cache = "error"
            if resolved is not None:
                resolve_cache = str(resolved[0].diagnostics.get("pyfa_runtime_resolve_cache", "unknown") or "unknown")
                projected_fit_cache = str(resolved[0].diagnostics.get("pyfa_projected_target_fit_cache", "not_applicable") or "not_applicable")
            self._log_hotspot(
                "combat.pyfa_resolve_batch",
                resolve_started,
                tick=int(world.tick),
                batch_size=len(pending_group),
                ship_ids=tuple(str(pending["ship"].ship_id) for pending in pending_group),
                fit_key=str(first_pending["runtime"].fit_key or ""),
                command_sources=len(first_pending["booster_snapshots"]),
                projected_sources=len(first_pending["projected_snapshots"]),
                success=resolved is not None,
                resolve_cache=resolve_cache,
                projected_fit_cache=projected_fit_cache,
            )
            if resolved is None:
                for pending in pending_group:
                    cached_base_profile = pending["runtime"].diagnostics.get("pyfa_base_profile")
                    if isinstance(cached_base_profile, ShipProfile):
                        pending["ship"].profile = cached_base_profile
                    else:
                        pending["ship"].profile = self._fallback_unprojected_profile(pending["ship"])
                continue

            resolved_runtime, resolved_profile = resolved
            resolved_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

            for index, pending in enumerate(pending_group):
                source_runtime = pending["runtime"]
                ship = pending["ship"]
                target_runtime = resolved_runtime if index == 0 else self._clone_resolved_runtime_for_ship(source_runtime, resolved_runtime)
                target_runtime.fit_key = source_runtime.fit_key

                blueprint = source_runtime.diagnostics.get("pyfa_blueprint")
                if isinstance(blueprint, dict):
                    target_runtime.diagnostics["pyfa_blueprint"] = deepcopy(blueprint)

                target_runtime.diagnostics["pyfa_command_boosters"] = pending["booster_snapshots"]
                target_runtime.diagnostics["pyfa_projected_sources"] = pending["projected_snapshots"]
                target_runtime.diagnostics["pyfa_command_booster_signature"] = pending["booster_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_signature"] = pending["projected_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_structure_signature"] = self._projected_snapshot_structure_signature(pending["projected_snapshots"])
                target_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

                self._copy_runtime_dynamic_state(source_runtime, target_runtime)
                if self._apply_runtime_activation_limit_transitions(world, ship, source_runtime, target_runtime):
                    self._mark_pyfa_remote_inputs_dirty()
                resolved_local_signature = self._local_runtime_state_signature_from_metadata(target_runtime)
                resolved_cache_key = get_runtime_resolve_cache_key(
                    target_runtime,
                    pending["booster_snapshots"],
                    pending["projected_snapshots"],
                )
                if resolved_cache_key is not None:
                    target_runtime.diagnostics["pyfa_resolve_signature"] = resolved_cache_key
                else:
                    target_runtime.diagnostics.pop("pyfa_resolve_signature", None)
                if resolved_local_signature is not None:
                    target_runtime.diagnostics["pyfa_local_state_signature"] = resolved_local_signature
                else:
                    target_runtime.diagnostics.pop("pyfa_local_state_signature", None)
                target_runtime.diagnostics["runtime_local_state_signature"] = resolved_local_signature
                target_runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = self._runtime_has_active_pyfa_remote_inputs(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_minimum_potential_cycle_signature"] = self._runtime_offline_module_signature(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_minimum_potential_cycle_time"] = self._runtime_minimum_potential_cycle_time(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_observed_module_state_signature"] = self._runtime_observed_module_state_signature(
                    target_runtime
                )
                ship.runtime = target_runtime
                ship.profile = resolved_profile

    def _module_has_area_enemies_in_range(self, world: WorldState, source, module) -> bool:
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            for candidate in self._iter_area_targets_in_range(world, source, module, effect):
                if candidate.team != source.team:
                    return True
        return False

    @staticmethod
    def _ship_id_in_pool(ship_id: str, pool: list) -> bool:
        return any(candidate.ship_id == ship_id and candidate.vital.alive for candidate in pool)

    def _is_lowest_hp_ally_target(self, source, module, allies_pool: list, target_id: str) -> bool:
        candidates = [
            ally
            for ally in self._candidates_in_projected_range(source, module, allies_pool)
            if ally.ship_id != source.ship_id
        ]
        if not candidates:
            return False

        wounded = [ally for ally in candidates if self._hp_ratio(ally) < 0.999]
        pool = wounded if wounded else candidates
        target = next((ally for ally in pool if ally.ship_id == target_id), None)
        if target is None:
            return False

        lowest_hp_ratio = min(self._hp_ratio(ally) for ally in pool)
        return self._hp_ratio(target) <= lowest_hp_ratio + 1e-6

    def _can_reuse_projected_target(
        self,
        world: WorldState,
        source,
        module,
        rule: ModuleDecisionRule,
        target_id: str | None,
        allies_pool: list,
        enemies_pool: list,
        ally_ids: set[str],
        enemy_ids: set[str],
    ) -> bool:
        if not target_id:
            return False

        target = world.ships.get(target_id)
        if target is None or not target.vital.alive:
            return False
        if not self._module_in_projected_range(source, target, module):
            return False
        if not self._can_target_under_ecm(source, target_id, self._decision_now(world)):
            return False

        if rule.target_mode == "weapon_focus_prefocus":
            focus_queue = world.squad_focus_queues.get(self._focus_key(source.team, source.squad_id), [])
            if not focus_queue:
                return False
            allowed_ids: set[str] = {str(focus_queue[0])}
            if len(focus_queue) > 1:
                allowed_ids.add(str(focus_queue[1]))
            return target_id in allowed_ids and target_id in enemy_ids

        if rule.target_mode == "ally_repair_queue":
            metadata = self._module_static_metadata(module)
            return target_id == self._select_repair_queue_target(world, source, module, metadata)

        if rule.target_mode == "ally_lowest_hp":
            if target_id == source.ship_id:
                return False
            return self._is_lowest_hp_ally_target(source, module, allies_pool, target_id)

        if rule.target_mode in {"enemy_random", "enemy_nearest"}:
            return target_id in enemy_ids

        side = self._module_static_metadata(module).target_side
        if side == "ally":
            if target_id == source.ship_id:
                return False
            return target_id in ally_ids
        return target_id in enemy_ids

    def _select_enemy_random_in_range(self, source, module, enemies_pool: list, existing_target_id: str | None) -> str | None:
        candidates = self._candidates_in_projected_range(source, module, enemies_pool)
        if not candidates:
            return None
        if existing_target_id and any(enemy.ship_id == existing_target_id for enemy in candidates):
            return existing_target_id
        return random.choice(candidates).ship_id

    def _select_enemy_nearest_in_range(self, source, module, enemies_pool: list, existing_target_id: str | None) -> str | None:
        candidates = self._candidates_in_projected_range(source, module, enemies_pool)
        if not candidates:
            return None
        if existing_target_id and any(enemy.ship_id == existing_target_id for enemy in candidates):
            return existing_target_id
        return min(candidates, key=lambda enemy: source.nav.position.distance_to(enemy.nav.position)).ship_id

    def _select_ally_lowest_hp_in_range(self, source, module, allies_pool: list, existing_target_id: str | None) -> str | None:
        candidates = [
            ally
            for ally in self._candidates_in_projected_range(source, module, allies_pool)
            if ally.ship_id != source.ship_id
        ]
        if not candidates:
            return None
        wounded = [ally for ally in candidates if self._hp_ratio(ally) < 0.999]
        pool = wounded if wounded else candidates
        if existing_target_id and any(ally.ship_id == existing_target_id for ally in pool):
            return existing_target_id
        return min(pool, key=self._hp_ratio).ship_id

    def _select_weapon_focus_target(self, world: WorldState, source, module, existing_target_id: str | None) -> str | None:
        focus_queue = world.squad_focus_queues.get(self._focus_key(source.team, source.squad_id), [])
        if not focus_queue:
            return None

        valid_focus_id: str | None = None
        valid_prefocus_id: str | None = None
        for queue_index, raw_target_id in enumerate(focus_queue[:2]):
            target_id = str(raw_target_id)
            target = world.ships.get(target_id)
            if target is None or not target.vital.alive or target.team == source.team:
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
        allies_pool: list,
        enemies_pool: list,
        rule: ModuleDecisionRule,
        existing_target_id: str | None,
    ) -> str | None:
        # Central target selector: each target_mode maps to a reusable selection helper.
        if rule.target_mode == "none":
            return None
        if rule.target_mode == "weapon_focus_prefocus":
            return self._select_weapon_focus_target(world, source, module, existing_target_id)
        if rule.target_mode == "ally_repair_queue":
            return self._select_repair_queue_target(world, source, module, self._module_static_metadata(module))
        if rule.target_mode == "ally_lowest_hp":
            return self._select_ally_lowest_hp_in_range(source, module, allies_pool, existing_target_id)
        if rule.target_mode == "enemy_random":
            return self._select_enemy_random_in_range(source, module, enemies_pool, existing_target_id)
        if rule.target_mode == "enemy_nearest":
            return self._select_enemy_nearest_in_range(source, module, enemies_pool, existing_target_id)

        side = self._module_static_metadata(module).target_side
        if side == "ally":
            return self._select_ally_lowest_hp_in_range(source, module, allies_pool, existing_target_id)
        return self._select_enemy_nearest_in_range(source, module, enemies_pool, existing_target_id)

    @staticmethod
    def _ecm_strength_from_effect(effect) -> dict[str, float]:
        return {
            "gravimetric": max(0.0, float(effect.projected_add.get("ecm_gravimetric", 0.0) or 0.0)),
            "ladar": max(0.0, float(effect.projected_add.get("ecm_ladar", 0.0) or 0.0)),
            "magnetometric": max(0.0, float(effect.projected_add.get("ecm_magnetometric", 0.0) or 0.0)),
            "radar": max(0.0, float(effect.projected_add.get("ecm_radar", 0.0) or 0.0)),
        }

    @staticmethod
    def _target_sensor_type_and_strength(profile: ShipProfile) -> tuple[str, float, bool]:
        strengths = {
            "gravimetric": max(0.0, float(getattr(profile, "sensor_strength_gravimetric", 0.0) or 0.0)),
            "ladar": max(0.0, float(getattr(profile, "sensor_strength_ladar", 0.0) or 0.0)),
            "magnetometric": max(0.0, float(getattr(profile, "sensor_strength_magnetometric", 0.0) or 0.0)),
            "radar": max(0.0, float(getattr(profile, "sensor_strength_radar", 0.0) or 0.0)),
        }
        sensor_type, sensor_strength = max(strengths.items(), key=lambda item: item[1])
        has_known_sensor_type = sensor_strength > 0.0
        if sensor_strength <= 0.0:
            sensor_strength = 1.0
        return sensor_type, sensor_strength, has_known_sensor_type

    @staticmethod
    def _ecm_duration_seconds(module_group: str) -> float:
        group = (module_group or "").lower()
        if "burst jammer" in group:
            return 5.0
        if "drone" in group:
            return 5.0
        return 20.0

    @staticmethod
    def _prune_ecm_sources(ship, now: float) -> set[str]:
        active_sources: set[str] = set()
        for source_id, jam_until in list(ship.combat.ecm_jam_sources.items()):
            if float(jam_until) > now:
                active_sources.add(str(source_id))
                continue
            ship.combat.ecm_jam_sources.pop(source_id, None)
        return active_sources

    def _can_target_under_ecm(self, ship, target_id: str | None, now: float) -> bool:
        if not target_id:
            return False
        active_sources = self._prune_ecm_sources(ship, now)
        if not active_sources:
            return True
        return str(target_id) in active_sources

    def _enforce_ecm_restrictions(self, ship, now: float) -> None:
        active_sources = self._prune_ecm_sources(ship, now)
        if not active_sources:
            return
        ship.combat.lock_targets.intersection_update(active_sources)
        for target_id in list(ship.combat.lock_timers.keys()):
            if target_id not in active_sources:
                self._clear_lock_timer(ship, target_id)
        for target_id in list(ship.combat.fire_delay_timers.keys()):
            if target_id not in active_sources:
                ship.combat.fire_delay_timers.pop(target_id, None)
        for module_id, target_id in list(ship.combat.projected_targets.items()):
            if target_id not in active_sources:
                ship.combat.projected_targets.pop(module_id, None)
        if ship.combat.current_target and ship.combat.current_target not in active_sources:
            ship.combat.current_target = None

    def _update_ecm_restrictions(self, world: WorldState, now: float | None = None) -> None:
        now_value = self._decision_now(world, now)
        for ship in world.ships.values():
            if not ship.vital.alive:
                ship.combat.ecm_jam_sources.clear()
                continue
            self._enforce_ecm_restrictions(ship, now_value)

    @staticmethod
    def _break_all_locks(ship) -> None:
        ship.combat.lock_targets.clear()
        ship.combat.lock_timers.clear()
        ship.combat.lock_deadlines.clear()
        ship.combat.fire_delay_timers.clear()
        ship.combat.projected_targets.clear()
        ship.combat.current_target = None

    def _resolve_ecm_cycle(self, world: WorldState, source, module, target_id: str) -> None:
        target = world.ships.get(target_id)
        if target is None or not target.vital.alive:
            return
        if target_id not in source.combat.lock_targets:
            return
        now = float(world.now)
        distance = source.nav.position.distance_to(target.nav.position)
        target_sensor_type, target_sensor_strength, has_known_sensor_type = self._target_sensor_type_and_strength(target.profile)
        if target_sensor_strength <= 0.0:
            return

        jammed = False
        ecm_attempted = False
        jam_chance = 0.0
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            strengths = self._ecm_strength_from_effect(effect)
            module_jam_strength = strengths.get(target_sensor_type, 0.0)
            if module_jam_strength <= 0.0 and not has_known_sensor_type:
                module_jam_strength = max(strengths.values(), default=0.0)
            if module_jam_strength <= 0.0:
                continue
            ecm_attempted = True

            if effect.falloff_m > 0.0:
                max_range = effect.range_m + 3.0 * effect.falloff_m
            else:
                max_range = effect.range_m
            if max_range > 0 and distance > max_range:
                continue

            if effect.range_m > 0.0 or effect.falloff_m > 0.0:
                range_factor = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
            else:
                range_factor = 1.0

            effective_strength = module_jam_strength * max(0.0, min(1.0, range_factor))
            chance = max(0.0, min(1.0, effective_strength / max(1e-9, target_sensor_strength)))
            jam_chance = max(jam_chance, chance)
            if random.random() < chance:
                jammed = True
                break

        if not ecm_attempted:
            return

        source.combat.ecm_last_attempt_target = target_id
        source.combat.ecm_last_attempt_module = module.module_id
        source.combat.ecm_last_attempt_success = jammed
        source.combat.ecm_last_attempt_chance = max(0.0, min(1.0, float(jam_chance)))
        source.combat.ecm_last_attempt_at = now
        source.combat.ecm_last_attempt_target_by_module[module.module_id] = target_id
        source.combat.ecm_last_attempt_success_by_module[module.module_id] = bool(jammed)
        source.combat.ecm_last_attempt_at_by_module[module.module_id] = now

        if not jammed:
            return

        jam_until = now + self._ecm_duration_seconds(module.group)
        target.combat.ecm_jam_sources[source.ship_id] = max(
            float(target.combat.ecm_jam_sources.get(source.ship_id, 0.0) or 0.0),
            jam_until,
        )
        self._enforce_ecm_restrictions(target, now)
        self._queue_merged_event(
            "ecm_jam_applied",
            merge_fields={
                "source": source.ship_id,
                "target": target.ship_id,
                "module": module.module_id,
                "sensor_type": target_sensor_type,
            },
            sum_fields={
                "chance": jam_chance,
            },
        )

    def _resolve_area_ecm_cycle(self, world: WorldState, source, module) -> None:
        now = float(world.now)

        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            strengths = self._ecm_strength_from_effect(effect)
            if max(strengths.values(), default=0.0) <= 0.0:
                continue

            for target in self._iter_area_targets_in_range(world, source, module, effect):
                if target.ship_id == source.ship_id:
                    continue

                distance = source.nav.position.distance_to(target.nav.position)
                target_sensor_type, target_sensor_strength, has_known_sensor_type = self._target_sensor_type_and_strength(target.profile)
                if target_sensor_strength <= 0.0:
                    continue

                module_jam_strength = strengths.get(target_sensor_type, 0.0)
                if module_jam_strength <= 0.0 and not has_known_sensor_type:
                    module_jam_strength = max(strengths.values(), default=0.0)
                if module_jam_strength <= 0.0:
                    continue

                if effect.range_m > 0.0 or effect.falloff_m > 0.0:
                    range_factor = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
                else:
                    range_factor = 1.0

                effective_strength = module_jam_strength * max(0.0, min(1.0, range_factor))
                jam_chance = max(0.0, min(1.0, effective_strength / max(1e-9, target_sensor_strength)))
                jammed = random.random() < jam_chance

                source.combat.ecm_last_attempt_target = target.ship_id
                source.combat.ecm_last_attempt_module = module.module_id
                source.combat.ecm_last_attempt_success = jammed
                source.combat.ecm_last_attempt_chance = jam_chance
                source.combat.ecm_last_attempt_at = now
                source.combat.ecm_last_attempt_target_by_module[module.module_id] = target.ship_id
                source.combat.ecm_last_attempt_success_by_module[module.module_id] = bool(jammed)
                source.combat.ecm_last_attempt_at_by_module[module.module_id] = now

                if not jammed:
                    continue

                self._break_all_locks(target)
                self._queue_merged_event(
                    "ecm_burst_lock_break",
                    merge_fields={
                        "source": source.ship_id,
                        "target": target.ship_id,
                        "module": module.module_id,
                        "sensor_type": target_sensor_type,
                    },
                    sum_fields={
                        "chance": jam_chance,
                    },
                )

    def _update_module_states(self, world: WorldState, dt: float, now: float | None = None) -> bool:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        alive_ids_by_team: dict[Team, set[str]] = {Team.BLUE: set(), Team.RED: set()}
        for candidate in world.ships.values():
            if candidate.vital.alive and not self._ship_in_warp(candidate):
                alive_by_team[candidate.team].append(candidate)
                alive_ids_by_team[candidate.team].add(candidate.ship_id)

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
            if self._ship_in_warp(ship):
                self._clear_ship_warp_engagement_state(ship, runtime)
                continue
            focus_key = self._focus_key(ship.team, ship.squad_id)
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
            allies_pool = alive_by_team.get(ship.team, [])
            enemies_alive = alive_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, [])
            ally_ids = alive_ids_by_team.get(ship.team, set())
            enemy_ids = alive_ids_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, set())
            force_target_reselect = focus_key in changed_focus_keys
            self._enqueue_ship_control_signal_modules(world, ship, runtime, focus_changed=force_target_reselect, now=now_value)
            local_signature_dirty = False
            active_pyfa_remote_inputs_dirty = False
            synced_weapon_fire_delay_pairs: set[tuple[str | None, str | None]] = set()
            controlled_entries = self._ship_candidate_control_entries(ship, runtime)
            if not controlled_entries:
                continue
            next_pending_modules: set[str] = set()

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
                cycle_just_completed = False
                ammo_reload_started_this_tick = False

                if module.state == module.state.ACTIVE and cycle_time > 0:
                    if active_timer is not None:
                        if module_id in ship.combat.module_cycle_deadlines:
                            if float(active_timer) > 0.0:
                                continue
                        else:
                            timer_left = float(active_timer) - dt
                            if timer_left > 0:
                                ship.combat.module_cycle_timers[module_id] = timer_left
                                continue
                        self._clear_module_cycle_timer(ship, module_id)
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                        self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                        cycle_just_completed = True
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
                                        ammo_reload_started_this_tick = True
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
                            ammo_reload_started_this_tick = True

                decision_rule = metadata.decision_rule
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
                    if requested_mode == "active" and metadata.is_weapon and not metadata.is_area_effect:
                        projected_target_id = self._manual_weapon_target(world, ship, module, previous_projected_target)
                    else:
                        if (not force_target_reselect) and self._can_reuse_projected_target(
                            world,
                            ship,
                            module,
                            decision_rule,
                            previous_projected_target,
                            allies_pool,
                            enemies_alive,
                            ally_ids,
                            enemy_ids,
                        ):
                            projected_target_id = previous_projected_target
                        else:
                            projected_target_id = self._select_projected_target(
                                world,
                                ship,
                                module,
                                allies_pool=allies_pool,
                                enemies_pool=enemies_alive,
                                rule=decision_rule,
                                existing_target_id=None,
                            )
                    if decision_rule.target_mode == "weapon_focus_prefocus":
                        delay_pair = (previous_projected_target, projected_target_id)
                        if delay_pair not in synced_weapon_fire_delay_pairs:
                            self._sync_weapon_fire_delay(
                                ship,
                                previous_target_id=previous_projected_target,
                                new_target_id=projected_target_id,
                                now=now_value,
                            )
                            synced_weapon_fire_delay_pairs.add(delay_pair)

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

                ammo_reload_left = max(
                    0.0,
                    float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0),
                )
                active_reload_timer_present = module_id in ship.combat.module_ammo_reload_timers
                if ammo_reload_left > 0.0:
                    if module_id in ship.combat.module_ammo_reload_deadlines:
                        desired_active = False
                    else:
                        if not ammo_reload_started_this_tick:
                            ammo_reload_left = max(0.0, ammo_reload_left - dt)
                        if ammo_reload_left > 0.0:
                            ship.combat.module_ammo_reload_timers[module_id] = ammo_reload_left
                            desired_active = False
                        else:
                            self._clear_module_reload_timer(ship, module_id)
                            if module.charge_capacity > 0:
                                module.charge_remaining = float(module.charge_capacity)
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
                        now=now_value,
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
                                now=now_value,
                            )
                        else:
                            module.charge_remaining = float(module.charge_capacity)
                    desired_active = False

                cooldown_left = ship.combat.module_reactivation_timers.get(module_id)
                if cooldown_left is not None:
                    if module_id in ship.combat.module_reactivation_deadlines:
                        if float(cooldown_left) > 0.0:
                            desired_active = False
                        else:
                            self._clear_module_reactivation_timer(ship, module_id)
                    else:
                        if not cycle_just_completed:
                            cooldown_left = float(cooldown_left) - dt
                        if cooldown_left > 0.0:
                            ship.combat.module_reactivation_timers[module_id] = cooldown_left
                            desired_active = False
                        else:
                            self._clear_module_reactivation_timer(ship, module_id)

                module_max_state = self._runtime_module_max_state(ship.runtime, module_id)
                if desired_active and self._runtime_state_rank(module_max_state) < self._runtime_state_rank(ModuleState.ACTIVE):
                    desired_active = False

                activation_target_id: str | None = (
                    projected_target_id
                    if has_projected and not metadata.is_area_effect
                    else None
                )

                if desired_active and activation_target_id is not None and not metadata.is_bomb_launcher:
                    activation_target = world.ships.get(activation_target_id)
                    if not self._ensure_target_lock(
                        world,
                        ship,
                        activation_target_id,
                        activation_target,
                        lock_context="module_lock",
                        now=now_value,
                    ):
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
                            area_candidates=allies_pool if metadata.is_command_burst else None,
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
                    state_target = world.ships.get(state_target_id) if state_target_id else None
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
                    cycle_target = world.ships.get(projected_target_id) if projected_target_id else None
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

    def _apply_projected_cycle_effects(
        self,
        world: WorldState,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        strength: float,
        damage_factor_override: float | None = None,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        if target is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Keep layer values bounded to prevent hidden overflow from masking later damage.
        self._clamp_ship_layer_hp(target)

        strength = max(0.0, min(1.0, strength))

        shield_repaired = 0.0
        armor_repaired = 0.0
        cap_drained = 0.0
        dirty_layers: set[str] = set()
        alive_before = bool(target.vital.alive)

        shield_rep = float(effect.projected_add.get("shield_rep", 0.0) or 0.0)
        if shield_rep > 0.0:
            amount = shield_rep * strength
            before = target.vital.shield
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + amount)
            shield_repaired = max(0.0, target.vital.shield - before)
            if shield_repaired > 0.0:
                dirty_layers.add("shield")

        armor_rep = float(effect.projected_add.get("armor_rep", 0.0) or 0.0)
        if armor_rep > 0.0:
            amount = armor_rep * strength
            before = target.vital.armor
            target.vital.armor = min(target.vital.armor_max, target.vital.armor + amount)
            armor_repaired = max(0.0, target.vital.armor - before)
            if armor_repaired > 0.0:
                dirty_layers.add("armor")

        cap_drain = float(effect.projected_add.get("cap_drain", 0.0) or 0.0)
        if cap_drain > 0.0:
            resistance = max(0.0, float(getattr(target_profile, "energy_warfare_resistance", 1.0) or 1.0))
            amount = cap_drain * strength * resistance
            before_cap = target.vital.cap
            target.vital.cap = max(0.0, target.vital.cap - amount)
            cap_drained = max(0.0, before_cap - target.vital.cap)

        base_damage = (
            max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
        )
        if _sum_damage(base_damage) <= 0.0:
            if dirty_layers:
                self._mark_team_repair_queues_dirty(target.team, *dirty_layers)
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        damage_factor = strength if damage_factor_override is None else max(0.0, float(damage_factor_override))

        dealt_damage = _scale_damage(base_damage, damage_factor)
        total_damage = _sum_damage(dealt_damage)
        if total_damage <= 0.0:
            if dirty_layers:
                self._mark_team_repair_queues_dirty(target.team, *dirty_layers)
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        shield_before = target.vital.shield
        armor_before = target.vital.armor
        structure_before = target.vital.structure
        target.vital.shield, target.vital.armor, target.vital.structure = _apply_damage_sequence(
            target.vital.shield,
            target.vital.armor,
            target.vital.structure,
            dealt_damage,
            target_profile,
        )
        if abs(target.vital.shield - shield_before) > 1e-6:
            dirty_layers.add("shield")
        if abs(target.vital.armor - armor_before) > 1e-6:
            dirty_layers.add("armor")
        if abs(target.vital.structure - structure_before) > 1e-6:
            dirty_layers.add("structure")
        applied = (shield_before + armor_before + structure_before) - (
            target.vital.shield + target.vital.armor + target.vital.structure
        )
        if applied > 0.0:
            target.combat.last_damaged_at = world.now
        if target.vital.structure <= 0:
            target.vital.alive = False
            target.nav.velocity = Vector2(0.0, 0.0)
        if alive_before and not target.vital.alive:
            dirty_layers.update(_REPAIR_QUEUE_LAYERS)
        if dirty_layers:
            self._mark_team_repair_queues_dirty(target.team, *dirty_layers)

        return (
            shield_repaired,
            armor_repaired,
            cap_drained,
            dealt_damage[0],
            dealt_damage[1],
            dealt_damage[2],
            dealt_damage[3],
            total_damage,
        )

    def _effective_profile(self, ship, impacts: dict[str, list[ProjectedImpact]]):
        if ship.runtime is None:
            return ship.profile

        applied = impacts.get(ship.ship_id)
        if not applied:
            return ship.profile
        return self._apply_runtime_projected_impacts(ship.profile, applied, runtime=ship.runtime)

    @staticmethod
    def _focus_key(team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    def _changed_focus_queues(self, world: WorldState) -> set[str]:
        changed: set[str] = set()
        active_focus_keys: set[str] = {
            self._focus_key(ship.team, ship.squad_id)
            for ship in world.ships.values()
            if ship.vital.alive
        }
        active_focus_keys.update(str(key) for key in world.squad_focus_queues.keys())

        for focus_key in active_focus_keys:
            current_queue = tuple(str(target_id) for target_id in world.squad_focus_queues.get(focus_key, []))
            previous_queue = self._last_focus_queue_by_squad.get(focus_key)
            if previous_queue is not None and previous_queue != current_queue:
                changed.add(focus_key)
            self._last_focus_queue_by_squad[focus_key] = current_queue

        for stale_key in [key for key in self._last_focus_queue_by_squad.keys() if key not in active_focus_keys]:
            self._last_focus_queue_by_squad.pop(stale_key, None)

        return changed

    def _update_squad_prelocks(self, world: WorldState, dt: float, effective_profiles: dict[str, ShipProfile]) -> None:
        squads: dict[str, list] = {}
        for ship in world.ships.values():
            if not ship.vital.alive or self._ship_in_warp(ship):
                continue
            squads.setdefault(self._focus_key(ship.team, ship.squad_id), []).append(ship)

        for focus_key, queue in list(world.squad_focus_queues.items()):
            members = squads.get(focus_key, [])
            if not members:
                world.squad_prelocked_targets.pop(focus_key, None)
                world.squad_prelock_timers.pop(focus_key, None)
                continue

            members.sort(key=lambda s: s.ship_id)
            own_team = members[0].team

            seen: set[str] = set()
            cleaned: list[str] = []
            for target_id in queue:
                if target_id in seen:
                    continue
                target = world.ships.get(target_id)
                if target is None or (not target.vital.alive) or self._ship_in_warp(target) or target.team == own_team:
                    continue
                seen.add(target_id)
                cleaned.append(target_id)
            world.squad_focus_queues[focus_key] = cleaned

            pre_targets = cleaned[1:] if len(cleaned) > 1 else []
            valid_pre = set(pre_targets)

            prelocked_by_ship = world.squad_prelocked_targets.setdefault(focus_key, {})
            timers_by_ship = world.squad_prelock_timers.setdefault(focus_key, {})
            member_ids = {ship.ship_id for ship in members}
            for ship_id in list(prelocked_by_ship.keys()):
                if ship_id not in member_ids:
                    prelocked_by_ship.pop(ship_id, None)
            for ship_id in list(timers_by_ship.keys()):
                if ship_id not in member_ids:
                    timers_by_ship.pop(ship_id, None)

            for ship in members:
                ship_prelocked = prelocked_by_ship.setdefault(ship.ship_id, set())
                ship_timers = timers_by_ship.setdefault(ship.ship_id, {})
                for target_id in list(ship_prelocked):
                    if target_id not in valid_pre:
                        ship_prelocked.discard(target_id)
                for target_id in list(ship_timers.keys()):
                    if target_id not in valid_pre:
                        ship_timers.pop(target_id, None)

                if not pre_targets:
                    if not ship_prelocked:
                        prelocked_by_ship.pop(ship.ship_id, None)
                    if not ship_timers:
                        timers_by_ship.pop(ship.ship_id, None)
                    continue

                attacker_profile = effective_profiles.get(ship.ship_id) or ship.profile
                for target_id in pre_targets:
                    if target_id in ship_prelocked:
                        continue
                    target = world.ships.get(target_id)
                    if target is None or not target.vital.alive:
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

                if not ship_prelocked:
                    prelocked_by_ship.pop(ship.ship_id, None)
                if not ship_timers:
                    timers_by_ship.pop(ship.ship_id, None)

            if not prelocked_by_ship:
                world.squad_prelocked_targets.pop(focus_key, None)
            if not timers_by_ship:
                world.squad_prelock_timers.pop(focus_key, None)

        for focus_key in list(world.squad_prelocked_targets.keys()):
            if focus_key not in world.squad_focus_queues:
                world.squad_prelocked_targets.pop(focus_key, None)
        for focus_key in list(world.squad_prelock_timers.keys()):
            if focus_key not in world.squad_focus_queues:
                world.squad_prelock_timers.pop(focus_key, None)

    def run(self, world: WorldState, dt: float) -> None:
        self._projected_cycle_starts_this_tick.clear()
        self._prune_cycle_effect_snapshots(world)
        self._refresh_alive_runtime_ship_ids(world)
        step_end = float(world.now)
        step_start = max(0.0, step_end - max(0.0, float(dt)))
        self._decision_reference_time = step_start
        if self.event_logging_enabled:
            self._advance_merge_window(world.now)
        else:
            self._merged_event_buckets.clear()
            self._merge_window_start_time = None
            self._merge_window_end_time = None

        started = time.perf_counter()
        self._update_ecm_restrictions(world, now=step_start)
        self._log_hotspot("combat.update_ecm_restrictions", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._process_due_timer_events(world, current_time=step_start)
        self._log_hotspot("combat.process_due_timer_events", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._advance_target_locks(world, dt, now=step_start)
        self._log_hotspot("combat.advance_target_locks", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        if self._update_module_states(world, dt, now=step_start):
            self._mark_pyfa_remote_inputs_dirty()
        self._log_hotspot("combat.update_module_states", started, tick=int(world.tick), dt=dt)

        reusable_cached_profiles: list[tuple[Any, ShipProfile]] = []
        can_restore_cached_pyfa_bases = False
        if (not self._pyfa_remote_inputs_dirty) and self._cached_pyfa_remote_inputs_available():
            can_restore_cached_pyfa_bases, remote_recollect_required, reusable_cached_profiles = self._validate_cached_pyfa_base_profiles(world)
            if remote_recollect_required:
                self._mark_pyfa_remote_inputs_dirty()
                can_restore_cached_pyfa_bases = False
                reusable_cached_profiles = []

        reuse_remote_pyfa_inputs = (not self._pyfa_remote_inputs_dirty) and self._cached_pyfa_remote_inputs_available()

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs:
            command_boosters = self._cached_command_booster_snapshots or {}
        else:
            command_boosters = self._collect_command_booster_snapshots(world)
        self._log_hotspot("combat.collect_command_boosters", started, tick=int(world.tick), ships=len(command_boosters))

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs:
            projected_sources = self._cached_projected_source_snapshots or {}
        else:
            projected_sources = self._collect_projected_source_snapshots(world, command_boosters)
            self._cached_command_booster_snapshots = command_boosters
            self._cached_projected_source_snapshots = projected_sources
            self._pyfa_remote_inputs_dirty = False
        self._log_hotspot("combat.collect_projected_sources", started, tick=int(world.tick), ships=len(projected_sources))

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs and can_restore_cached_pyfa_bases:
            for ship, cached_profile in reusable_cached_profiles:
                ship.profile = cached_profile
        else:
            self._refresh_effective_runtimes_from_pyfa(world, command_boosters, projected_sources)
        self._log_hotspot("combat.refresh_effective_runtimes", started, tick=int(world.tick), ships=len(world.ships))

        started = time.perf_counter()
        projected = self._collect_projected_impacts(world, dt)
        self._log_hotspot(
            "combat.collect_projected_impacts",
            started,
            tick=int(world.tick),
            targets=sum(len(v) for v in projected.values()),
        )

        started = time.perf_counter()
        effective_profiles: dict[str, ShipProfile] = {}
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            effective_profiles[ship.ship_id] = self._effective_profile(ship, projected)
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            effective = effective_profiles.get(ship.ship_id)
            if effective is not None:
                ship.profile = effective
        self._log_hotspot("combat.apply_effective_profiles", started, tick=int(world.tick), ships=len(effective_profiles))

        self._update_squad_prelocks(world, dt, effective_profiles)

        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_in_warp(source):
                continue
            for module in source.runtime.modules:
                metadata = self._module_static_metadata(module)
                if module.state != module.state.ACTIVE:
                    continue
                if metadata.is_command_burst or metadata.is_burst_jammer:
                    continue
                if (
                    metadata.cycle_time > 0.0
                    and self._uses_cycle_start_projected_application(metadata)
                    and not self._projected_cycle_started_this_tick(source.ship_id, module.module_id)
                ):
                    continue

                cycle_target_snapshots = self._module_cycle_snapshots_for(source.ship_id, module.module_id)
                if not cycle_target_snapshots and not metadata.is_smart_bomb:
                    continue

                for effect_index, effect in metadata.projected_effects:

                    targets: list[tuple[Any, CycleTargetSnapshot, float]] = []
                    if metadata.is_smart_bomb:
                        for target_id, target_snapshot in cycle_target_snapshots.items():
                            target = world.ships.get(target_id)
                            if target is None or not target.vital.alive or self._ship_in_warp(target):
                                continue
                            strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                            if strength > 0.0:
                                targets.append((target, target_snapshot, strength))
                    else:
                        tgt_id = source.combat.projected_targets.get(module.module_id)
                        if not tgt_id:
                            continue
                        target = world.ships.get(tgt_id)
                        if target is None or not target.vital.alive or self._ship_in_warp(target):
                            continue
                        got_snapshot = cycle_target_snapshots.get(tgt_id)
                        if got_snapshot is None:
                            continue
                        target_snapshot = got_snapshot
                        if False:
                            continue
                        strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                        if strength <= 0.0:
                            continue
                        targets.append((target, target_snapshot, strength))

                    for target, target_snapshot, strength in targets:
                        target_profile = effective_profiles.get(target.ship_id) or target.profile
                        damage_factor_override = self._cycle_effect_damage_factor(
                            source=source,
                            target=target,
                            target_profile=target_profile,
                            effect=effect,
                            effect_index=effect_index,
                            target_snapshot=target_snapshot,
                            strength=strength,
                        )
                        hp_before = target.vital.shield + target.vital.armor + target.vital.structure
                        (
                            shield_repaired,
                            armor_repaired,
                            cap_drained,
                            em_damage,
                            thermal_damage,
                            kinetic_damage,
                            explosive_damage,
                            total_damage,
                        ) = self._apply_projected_cycle_effects(
                            world=world,
                            source=source,
                            target=target,
                            target_profile=target_profile,
                            effect=effect,
                            strength=strength,
                            damage_factor_override=damage_factor_override,
                        )
                        hp_after = target.vital.shield + target.vital.armor + target.vital.structure
                        applied_damage = max(0.0, hp_before - hp_after)
                        if (
                            applied_damage > 0.0
                            and source.team != target.team
                            and metadata.is_weapon
                        ):
                            target.combat.last_enemy_weapon_damaged_at = float(world.now)
                        if (
                            shield_repaired > 0.0
                            or armor_repaired > 0.0
                            or cap_drained > 0.0
                            or total_damage > 0.0
                        ):
                            self._add_projected_cycle_total(
                                source_ship_id=source.ship_id,
                                module_id=module.module_id,
                                target_ship_id=target.ship_id,
                                shield_repaired=shield_repaired,
                                armor_repaired=armor_repaired,
                                cap_drained=cap_drained,
                                em_damage=em_damage,
                                thermal_damage=thermal_damage,
                                kinetic_damage=kinetic_damage,
                                explosive_damage=explosive_damage,
                                total_damage=total_damage,
                            )
                    if metadata.is_smart_bomb:
                        self._destroy_projectiles_in_area(
                            world,
                            center=Vector2(source.nav.position.x, source.nav.position.y),
                            radius_m=max(0.0, float(effect.range_m or 0.0)),
                            damage=self._effect_damage_tuple(effect),
                        )

        self._advance_projectiles(world, dt)

        if self.detailed_logging and self.logger is not None:
            total_impacts = sum(len(v) for v in projected.values())
            self.logger.debug(f"combat_tick dt={dt:.4f} projected_impacts={total_impacts}")

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            ship_profile = effective_profiles.get(ship.ship_id, ship.profile)
            self._sync_vital_max_with_profile(ship, ship_profile)
            if (
                self.detailed_logging
                and self.logger is not None
                and ship.runtime is not None
                and ship.ship_id not in self._diag_logged_ships
            ):
                unmodeled = ship.runtime.diagnostics.get("unmodeled_modules", [])
                if unmodeled:
                    self.logger.debug(
                        f"fit_diagnostics ship={ship.ship_id} unmodeled_modules={unmodeled}"
                    )
                self._diag_logged_ships.add(ship.ship_id)
            ship.nav.max_speed = ship_profile.max_speed

            ship.vital.cap_max = ship_profile.max_cap
            if ship.vital.cap > ship.vital.cap_max:
                ship.vital.cap = ship.vital.cap_max

            ship.vital.cap = self._resolve_cap_recharge(
                cap_now=ship.vital.cap,
                cap_max=ship.vital.cap_max,
                recharge_time=ship_profile.cap_recharge_time,
                dt=dt,
            )

            if self._ship_in_warp(ship):
                ship.combat.current_target = None
                continue

            current_target_id = ship.combat.current_target
            if current_target_id:
                current_target = world.ships.get(current_target_id)
                if (
                    current_target is None
                    or not current_target.vital.alive
                    or self._ship_in_warp(current_target)
                    or current_target.team == ship.team
                ):
                    ship.combat.current_target = None

            if not ship.combat.current_target:
                queue = list(world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), []))
                for candidate_id in queue:
                    candidate = world.ships.get(candidate_id)
                    if candidate is None or not candidate.vital.alive or self._ship_in_warp(candidate) or candidate.team == ship.team:
                        continue
                    ship.combat.current_target = candidate_id
                    break

        started = time.perf_counter()
        self._process_due_timer_events(world, current_time=step_end)
        self._log_hotspot("combat.commit_due_timer_events", started, tick=int(world.tick), dt=dt)
        self._decision_reference_time = None

        if self.event_logging_enabled:
            self._advance_merge_window(world.now)





