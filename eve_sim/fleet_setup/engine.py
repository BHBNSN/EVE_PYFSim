from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import math
import random
import re
from pathlib import Path
import sqlite3
import sys
import types
from typing import Any, cast

from ..config import resolve_pyfa_source_dir

from ..fit_runtime import (
    EffectClass,
    FitRuntime,
    HullProfile,
    ModuleEffect,
    ModuleRuntime,
    ModuleState,
    SkillProfile,
)
from ..math2d import Vector2
from ..models import (
    Beacon,
    CombatState,
    FitDescriptor,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipProfile,
    ShipEntity,
    Team,
    VitalState,
)
from ..world import WorldState


QUALITY_PRESETS = {
    QualityLevel.ELITE: QualityState(QualityLevel.ELITE, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.REGULAR: QualityState(QualityLevel.REGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.IRREGULAR: QualityState(QualityLevel.IRREGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
}

from .models import *
from .eft_parser import *

class RuntimeFromEftFactory:
    def __init__(self) -> None:
        self._runtime_cache: dict[str, FitRuntime] = {}
        self._fit_cache: dict[str, FitDescriptor] = {}
        self._profile_cache: dict[str, ShipProfile] = {}
        self._pyfa = _PyfaStaticBackend()
        self._charge_module_ammo_overrides: dict[str, str] = {}

    def set_charge_module_ammo_override(self, module_name: str, ammo_name: str) -> None:
        module = self._pyfa.resolve_type_name(module_name)
        ammo = self._pyfa.resolve_type_name(ammo_name)
        if module and ammo:
            self._charge_module_ammo_overrides[module.lower()] = ammo
            self._runtime_cache.clear()
            self._fit_cache.clear()
            self._profile_cache.clear()

    def clear_charge_module_ammo_overrides(self) -> None:
        if self._charge_module_ammo_overrides:
            self._charge_module_ammo_overrides.clear()
            self._runtime_cache.clear()
            self._fit_cache.clear()
            self._profile_cache.clear()

    def _resolve_module_charge_name(self, module_item, explicit_charge_name: str | None) -> str | None:
        if explicit_charge_name:
            return self._pyfa.resolve_type_name(explicit_charge_name)
        module_name = self._pyfa.resolve_type_name(str(getattr(module_item, "typeName", "") or ""))
        if not module_name:
            return None
        override = self._charge_module_ammo_overrides.get(module_name.lower())
        if override:
            return override
        ammo = self._pyfa.list_charge_options_for_module(module_name)
        if ammo:
            return ammo[0]
        return None

    @staticmethod
    def _skills_default() -> SkillProfile:
        return SkillProfile(levels={})

    @staticmethod
    def _is_weapon_like_group(group_name: str) -> bool:
        g = group_name.lower()
        if "disruptor" in g:
            return False
        return ("launcher" in g) or ("turret" in g) or ("weapon" in g)

    @staticmethod
    def _is_charge_compatible(module_item, charge_item) -> bool:
        if module_item is None or charge_item is None:
            return False
        module_capacity = module_item.getAttribute("capacity", None)
        charge_volume = charge_item.getAttribute("volume", None)
        if module_capacity is not None and charge_volume is not None:
            try:
                if float(charge_volume) > float(module_capacity):
                    return False
            except Exception:
                pass

        module_size = module_item.getAttribute("chargeSize", None)
        charge_size = charge_item.getAttribute("chargeSize", None)
        if module_size is not None and charge_size is not None:
            try:
                ws = int(float(module_size))
                cs = int(float(charge_size))
                if ws > 0 and ws != cs:
                    return False
            except Exception:
                pass

        charge_group = getattr(charge_item, "groupID", None)
        if charge_group is None:
            charge_group = getattr(getattr(charge_item, "group", None), "ID", None)
        try:
            charge_group_id = int(float(charge_group)) if charge_group is not None else 0
        except Exception:
            charge_group_id = 0
        if charge_group_id <= 0:
            return False

        for idx in range(0, 5):
            value = module_item.getAttribute(f"chargeGroup{idx}", None)
            if value is None:
                continue
            try:
                if int(float(value)) == charge_group_id:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _normalize_effect_name(name: str) -> str:
        return str(name or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    @staticmethod
    def _collect_effect_names(effect_holder: Any) -> set[str]:
        names: set[str] = set()
        if effect_holder is None:
            return names
        effects = getattr(effect_holder, "effects", None)
        if effects is None:
            return names

        raw_names: list[str] = []
        try:
            if hasattr(effects, "keys"):
                for key in effects.keys():
                    if key is not None:
                        raw_names.append(str(key))
        except Exception:
            pass

        effect_values: list[Any] = []
        try:
            if hasattr(effects, "values"):
                effect_values = list(effects.values())
            else:
                effect_values = list(effects)
        except Exception:
            effect_values = []

        for effect in effect_values:
            for candidate in (
                getattr(effect, "name", None),
                getattr(effect, "effectName", None),
                getattr(effect, "displayName", None),
            ):
                if candidate:
                    raw_names.append(str(candidate))

        for raw_name in raw_names:
            normalized = RuntimeFromEftFactory._normalize_effect_name(raw_name)
            if normalized:
                names.add(normalized)
        return names

    @classmethod
    def _module_effect_names(cls, fitted_module: Any) -> set[str]:
        names = cls._collect_effect_names(getattr(fitted_module, "item", None))
        names.update(cls._collect_effect_names(getattr(fitted_module, "charge", None)))
        return names

    @staticmethod
    def _item_flag(item: Any, flag_name: str) -> bool:
        try:
            return bool(getattr(item, flag_name, False))
        except Exception:
            return False

    @classmethod
    def _module_tags(
        cls,
        *,
        item: Any,
        has_projected: bool,
        is_active_module: bool,
        affects_local_pyfa_profile: bool,
        is_command_burst: bool,
        is_smart_bomb: bool,
        is_burst_jammer: bool,
        is_weapon: bool,
        has_projected_rep: bool,
        is_cap_warfare: bool,
        is_offensive_ewar: bool,
        is_target_ewar: bool,
        is_ecm: bool,
        is_propulsion: bool,
        is_damage_control: bool,
        is_hardener: bool,
        is_cap_booster: bool,
    ) -> tuple[str, ...]:
        tags: set[str] = set()

        if is_active_module:
            tags.add("controlled")
        if affects_local_pyfa_profile:
            tags.add("affects_local_pyfa_profile")
        if has_projected:
            tags.add("projected")

        if cls._item_flag(item, "assistive"):
            tags.add("support")
        if cls._item_flag(item, "offensive"):
            tags.add("hostile")

        if is_command_burst:
            tags.update(("command_burst", "area_effect", "support"))
        if is_smart_bomb:
            tags.update(("smart_bomb", "area_effect", "weapon", "hostile"))
        if is_burst_jammer:
            tags.update(("burst_jammer", "area_effect", "ecm", "offensive_ewar", "hostile"))
        if is_weapon:
            tags.update(("weapon", "hostile"))
        if has_projected_rep:
            tags.update(("remote_repair", "support"))
        if is_cap_warfare:
            tags.update(("cap_warfare", "offensive_ewar", "hostile"))
        if is_offensive_ewar:
            tags.update(("offensive_ewar", "hostile"))
        if is_target_ewar:
            tags.update(("target_ewar", "hostile"))
        if is_ecm:
            tags.update(("ecm", "offensive_ewar", "hostile"))
        if is_propulsion:
            tags.add("propulsion")
        if is_damage_control:
            tags.add("damage_control")
        if is_hardener:
            tags.add("hardener")
        if is_cap_booster:
            tags.add("cap_booster")

        return tuple(sorted(tags))

    def _fitted_module_projected_range(self, fitted_module: Any) -> float:
        def attr_opt(name: str) -> float | None:
            try:
                value = fitted_module.getModifiedItemAttr(name)
            except Exception:
                value = None
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        for attr_name in (
            "maxRange",
            "shieldTransferRange",
            "powerTransferRange",
            "energyDestabilizationRange",
            "empFieldRange",
            "ecmBurstRange",
            "warpScrambleRange",
            "cargoScanRange",
            "shipScanRange",
            "surveyScanRange",
        ):
            value = attr_opt(attr_name)
            if value is not None and abs(value) > 1e-9:
                return max(0.0, float(value))
        return 0.0

    @staticmethod
    def _projection_filter_signature(filter_func: Any, target_kind: str) -> tuple[Any, ...]:
        code = getattr(filter_func, "__code__", None)
        closure = getattr(filter_func, "__closure__", None) or ()
        closure_values: list[Any] = []
        for cell in closure:
            try:
                value = cell.cell_contents
            except Exception:
                value = None
            if isinstance(value, (str, int, float, bool, type(None))):
                closure_values.append(value)
            else:
                closure_values.append(type(value).__name__)

        class _ProbeReq:
            def __init__(self, skills: tuple[str, ...]) -> None:
                self._skills = set(skills)

            def requiresSkill(self, name: str) -> bool:
                return str(name) in self._skills

        class _ProbeGroup:
            def __init__(self, name: str) -> None:
                self.name = name

        class _ProbeMod:
            def __init__(self, item_skills: tuple[str, ...], charge_skills: tuple[str, ...], group_name: str) -> None:
                self.item = type("ProbeItem", (), {"requiresSkill": _ProbeReq(item_skills).requiresSkill, "group": _ProbeGroup(group_name)})()
                self.charge = type("ProbeCharge", (), {"requiresSkill": _ProbeReq(charge_skills).requiresSkill})()

        probes = (
            ("module_item_gunnery", _ProbeMod(("Gunnery",), tuple(), "Hybrid Weapon")),
            ("module_charge_missile", _ProbeMod(tuple(), ("Missile Launcher Operation",), "Missile Launcher Heavy")),
            ("fighter_item", _ProbeMod(("Fighters",), tuple(), "Fighter")),
        )
        semantic_matches: list[str] = []
        for probe_name, probe in probes:
            try:
                if bool(filter_func(probe)):
                    semantic_matches.append(probe_name)
            except Exception:
                continue

        return (
            str(target_kind),
            tuple(sorted(semantic_matches)),
            getattr(code, "co_name", ""),
            tuple(str(name) for name in (getattr(code, "co_names", None) or ())),
            tuple(repr(value) for value in (getattr(code, "co_consts", None) or ())),
            tuple(repr(value) for value in closure_values),
        )

    @staticmethod
    def _projection_operation_multiplier(op_kind: str, value: float) -> float | None:
        if op_kind == "boost":
            return 1.0 + float(value) / 100.0
        if op_kind == "multiply":
            return float(value)
        return None

    @staticmethod
    def _merge_projected_multiplier(
        projected_mult: dict[str, float],
        projected_mult_groups: dict[str, str | None],
        key: str,
        multiplier: float,
        group_name: str | None,
    ) -> bool:
        existing_group = projected_mult_groups.get(key, group_name)
        if key in projected_mult and existing_group != group_name:
            return False
        projected_mult[key] = float(projected_mult.get(key, 1.0) or 1.0) * float(multiplier)
        projected_mult_groups[key] = group_name
        return True

    def _record_projected_effect_operations(self, fitted_module: Any) -> tuple[list[tuple[Any, ...]], tuple[Any, ...]]:
        records: list[tuple[Any, ...]] = []

        class _AttrRecorder:
            def __init__(self, target_kind: str) -> None:
                self._target_kind = target_kind

            def _record(self, op_kind: str, attr_name: str, value: float, **kwargs) -> None:
                try:
                    numeric_value = float(value)
                except Exception:
                    return
                records.append(
                    (
                        self._target_kind,
                        (),
                        op_kind,
                        str(attr_name or ""),
                        numeric_value,
                        bool(kwargs.get("stackingPenalties", False)),
                        str(kwargs.get("penaltyGroup", "default")) if kwargs.get("stackingPenalties", False) else None,
                    )
                )

            def increaseItemAttr(self, attr_name: str, value: float, **kwargs) -> None:
                self._record("increase", attr_name, value, **kwargs)

            def multiplyItemAttr(self, attr_name: str, value: float, **kwargs) -> None:
                self._record("multiply", attr_name, value, **kwargs)

            def boostItemAttr(self, attr_name: str, value: float, **kwargs) -> None:
                self._record("boost", attr_name, value, **kwargs)

            def forceItemAttr(self, attr_name: str, value: float, **kwargs) -> None:
                self._record("force", attr_name, value, **kwargs)

            def getModifiedItemAttr(self, _name: str, default: float = 0.0) -> float:
                return float(default or 0.0)

        class _FilteredRecorder:
            def __init__(self, target_kind: str, factory: "RuntimeFromEftFactory") -> None:
                self._target_kind = target_kind
                self._factory = factory

            def _record(self, target_kind: str, filter_func: Any, op_kind: str, attr_name: str, value: float, **kwargs) -> None:
                try:
                    numeric_value = float(value)
                except Exception:
                    return
                records.append(
                    (
                        target_kind,
                        self._factory._projection_filter_signature(filter_func, target_kind),
                        op_kind,
                        str(attr_name or ""),
                        numeric_value,
                        bool(kwargs.get("stackingPenalties", False)),
                        str(kwargs.get("penaltyGroup", "default")) if kwargs.get("stackingPenalties", False) else None,
                    )
                )

            def filteredItemIncrease(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_item", filter_func, "increase", attr_name, value, **kwargs)

            def filteredItemMultiply(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_item", filter_func, "multiply", attr_name, value, **kwargs)

            def filteredItemBoost(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_item", filter_func, "boost", attr_name, value, **kwargs)

            def filteredChargeIncrease(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_charge", filter_func, "increase", attr_name, value, **kwargs)

            def filteredChargeMultiply(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_charge", filter_func, "multiply", attr_name, value, **kwargs)

            def filteredChargeBoost(self, filter_func: Any, attr_name: str, value: float, **kwargs) -> None:
                self._record(f"{self._target_kind}_charge", filter_func, "boost", attr_name, value, **kwargs)

            def __iter__(self):
                return iter(())

        class _FitRecorder:
            def __init__(self, factory: "RuntimeFromEftFactory") -> None:
                self.ship = _AttrRecorder("ship")
                self.modules = _FilteredRecorder("module", factory)
                self.drones = _FilteredRecorder("drone", factory)
                self.fighters = _FilteredRecorder("fighter", factory)

        recorder = _FitRecorder(self)
        original_state = getattr(fitted_module, "state", None)
        try:
            active_state = _pyfa_module_state_from_runtime_state(self, fitted_module, "ACTIVE")
            if active_state is not None:
                fitted_module.state = active_state
            for run_time in ("early", "normal", "late"):
                fitted_module.calculateModifiedAttributes(recorder, run_time, forceProjected=True, forcedProjRange=0.0)
        except Exception:
            return [], tuple()
        finally:
            if original_state is not None:
                fitted_module.state = original_state

        signature = tuple(
            sorted(
                (
                    str(target_kind),
                    tuple(filter_signature),
                    str(op_kind),
                    str(attr_name),
                    round(float(value or 0.0), 6),
                    bool(stacking_penalties),
                    None if penalty_group is None else str(penalty_group),
                )
                for target_kind, filter_signature, op_kind, attr_name, value, stacking_penalties, penalty_group in records
            )
        )
        return records, signature

    def _projected_effects_from_pyfa_handler(
        self,
        fitted_module: Any,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str | None], tuple[Any, ...]]:
        projected_mult: dict[str, float] = {}
        projected_add: dict[str, float] = {}
        projected_mult_groups: dict[str, str | None] = {}
        records, signature = self._record_projected_effect_operations(fitted_module)

        ship_mult_map = {
            "maxVelocity": "speed",
            "signatureRadius": "sig",
            "scanResolution": "scan",
            "maxTargetRange": "range",
            "scanGravimetricStrength": "sensor_strength_gravimetric",
            "scanLadarStrength": "sensor_strength_ladar",
            "scanMagnetometricStrength": "sensor_strength_magnetometric",
            "scanRadarStrength": "sensor_strength_radar",
            "shieldCapacity": "shield_hp",
            "armorHP": "armor_hp",
            "hp": "structure_hp",
            "capacitorCapacity": "cap_max",
            "rechargeRate": "cap_recharge",
            "agility": "agility",
            "mass": "mass",
            "shieldEmDamageResonance": "shield_resonance_em",
            "shieldThermalDamageResonance": "shield_resonance_thermal",
            "shieldKineticDamageResonance": "shield_resonance_kinetic",
            "shieldExplosiveDamageResonance": "shield_resonance_explosive",
            "armorEmDamageResonance": "armor_resonance_em",
            "armorThermalDamageResonance": "armor_resonance_thermal",
            "armorKineticDamageResonance": "armor_resonance_kinetic",
            "armorExplosiveDamageResonance": "armor_resonance_explosive",
            "structureEmDamageResonance": "structure_resonance_em",
            "structureThermalDamageResonance": "structure_resonance_thermal",
            "structureKineticDamageResonance": "structure_resonance_kinetic",
            "structureExplosiveDamageResonance": "structure_resonance_explosive",
        }
        ship_add_map = {
            "warpScrambleStatus": "warp_scramble_status",
        }
        turret_mult_map = {
            "trackingSpeed": "tracking",
            "maxRange": "optimal",
            "falloff": "falloff",
        }
        missile_mult_map = {
            "aoeCloudSize": "missile_explosion_radius",
            "aoeVelocity": "missile_explosion_velocity",
            "maxVelocity": "missile_range",
            "explosionDelay": "missile_range",
        }

        for target_kind, filter_signature, op_kind, attr_name, value, stacking_penalties, penalty_group in records:
            multiplier = self._projection_operation_multiplier(str(op_kind), float(value or 0.0))
            semantic_tags = set(str(tag) for tag in (filter_signature[1] if len(filter_signature) > 1 and isinstance(filter_signature[1], tuple) else ()))
            if target_kind == "ship":
                mapped_mult = ship_mult_map.get(str(attr_name))
                if mapped_mult and multiplier is not None:
                    if not self._merge_projected_multiplier(
                        projected_mult,
                        projected_mult_groups,
                        mapped_mult,
                        multiplier,
                        str(penalty_group) if stacking_penalties else None,
                    ):
                        return {}, {}, {}, signature
                    continue
                mapped_add = ship_add_map.get(str(attr_name))
                if mapped_add and str(op_kind) == "increase":
                    projected_add[mapped_add] = projected_add.get(mapped_add, 0.0) + float(value or 0.0)
                    continue
                if str(attr_name) in {"disallowAssistance", "disallowOffensiveModifiers"}:
                    continue
                return {}, {}, {}, signature

            if target_kind == "module_item" and "module_item_gunnery" in semantic_tags:
                mapped_mult = turret_mult_map.get(str(attr_name))
                if mapped_mult and multiplier is not None:
                    if not self._merge_projected_multiplier(
                        projected_mult,
                        projected_mult_groups,
                        mapped_mult,
                        multiplier,
                        str(penalty_group) if stacking_penalties else None,
                    ):
                        return {}, {}, {}, signature
                    continue
                return {}, {}, {}, signature

            if target_kind == "module_charge" and "module_charge_missile" in semantic_tags:
                mapped_mult = missile_mult_map.get(str(attr_name))
                if mapped_mult and multiplier is not None:
                    if not self._merge_projected_multiplier(
                        projected_mult,
                        projected_mult_groups,
                        mapped_mult,
                        multiplier,
                        str(penalty_group) if stacking_penalties else None,
                    ):
                        return {}, {}, {}, signature
                    continue
                return {}, {}, {}, signature

            if str(target_kind).startswith("fighter_") or str(target_kind).startswith("drone_"):
                continue
            return {}, {}, {}, signature

        return projected_mult, projected_add, projected_mult_groups, signature

    def _module_damage_components(
        self,
        fitted_module: Any,
        *,
        read_active: bool = False,
    ) -> tuple[float, float, float, float] | None:
        original_state = getattr(fitted_module, "state", None)
        state_switched = False
        if read_active:
            active_state = _pyfa_module_state_from_runtime_state(self, fitted_module, "ACTIVE")
            if active_state is not None and active_state != original_state:
                try:
                    fitted_module.state = active_state
                    state_switched = True
                except Exception:
                    state_switched = False

        try:
            dps_obj = fitted_module.getDps()
        except Exception:
            dps_obj = None
        finally:
            if state_switched and original_state is not None:
                try:
                    fitted_module.state = original_state
                except Exception:
                    pass

        if dps_obj is None:
            return None

        return (
            max(0.0, float(getattr(dps_obj, "em", 0.0) or 0.0)),
            max(0.0, float(getattr(dps_obj, "thermal", 0.0) or 0.0)),
            max(0.0, float(getattr(dps_obj, "kinetic", 0.0) or 0.0)),
            max(0.0, float(getattr(dps_obj, "explosive", 0.0) or 0.0)),
        )

    def _module_effect_pyfa(
        self,
        parsed: ParsedEftFit | Any,
        fit_ctx: Any = None,
        fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]] | None = None,
        fitted_module: Any | None = None,
        idx: int | None = None,
        state_by_module_id: dict[str, str] | None = None,
        command_booster_snapshots: list[dict[str, Any]] | None = None,
    ) -> ModuleRuntime | None:
        if idx is None and fitted_module is None and isinstance(fit_ctx, int):
            idx = int(fit_ctx)
            fitted_module = parsed
            parsed = ParsedEftFit(ship_name="", fit_name="", module_names=[], module_specs=[], cargo_item_names=[], fit_key="")
            fit_ctx = None
            fitted_modules = []
        del fit_ctx
        if fitted_module is None or idx is None:
            return None
        if fitted_modules is None:
            fitted_modules = []
        item = getattr(fitted_module, "item", None)
        if item is None:
            return None

        group_name = (item.group.name or "").lower()
        suffix = f"-{idx}"
        loaded_charge = getattr(fitted_module, "charge", None)
        def attr_opt(name: str) -> float | None:
            try:
                value = fitted_module.getModifiedItemAttr(name)
            except Exception:
                value = None
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        def charge_attr_opt(name: str) -> float | None:
            try:
                value = fitted_module.getModifiedChargeAttr(name)
            except Exception:
                value = None
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        def attr(name: str, default: float = 0.0) -> float:
            value = attr_opt(name)
            return float(default if value is None else value)

        def charge_attr(name: str, default: float = 0.0) -> float:
            value = charge_attr_opt(name)
            return float(default if value is None else value)

        def pct_to_mult(value: float) -> float:
            return max(0.01, 1.0 + value / 100.0)

        def first_attr(*names: str) -> float:
            for attr_name in names:
                value = attr_opt(attr_name)
                if value is None:
                    continue
                if abs(value) > 1e-9:
                    return float(value)
            return 0.0

        supports_active_state = False
        active_state = getattr(self._pyfa, "_fitting_module_state_active", None)
        if active_state is not None and hasattr(fitted_module, "isValidState"):
            try:
                supports_active_state = bool(fitted_module.isValidState(active_state))
            except Exception:
                supports_active_state = False
        if not supports_active_state:
            try:
                supports_active_state = bool(item.isType("active"))
            except Exception:
                supports_active_state = False

        duration_ms = attr("duration", 0.0)
        speed_ms = attr("speed", 0.0)
        cycle_ms = duration_ms if duration_ms > 0 else speed_ms
        cycle_sec = max(0.1, cycle_ms / 1000.0) if cycle_ms > 0 else 5.0
        cap_need = max(0.0, attr("capacitorNeed", 0.0))
        reactivation_delay_sec = max(0.0, attr("moduleReactivationDelay", 0.0) / 1000.0)

        range_m = max(
            0.0,
            first_attr(
                "maxRange",
                "shieldTransferRange",
                "powerTransferRange",
                "energyDestabilizationRange",
                "empFieldRange",
                "ecmBurstRange",
                "warpScrambleRange",
                "cargoScanRange",
                "shipScanRange",
                "surveyScanRange",
            ),
        )
        falloff_m = max(0.0, attr("falloffEffectiveness", 0.0))
        if falloff_m <= 0.0:
            falloff_m = max(0.0, attr("falloff", 0.0))
        pyfa_cycle_sec = 0.0
        try:
            cycle_params = fitted_module.getCycleParameters()
        except Exception:
            cycle_params = None
        if cycle_params is not None:
            try:
                pyfa_cycle_sec = max(0.0, float(getattr(cycle_params, "averageTime", 0.0) or 0.0) / 1000.0)
            except Exception:
                pyfa_cycle_sec = 0.0

        local_mult: dict[str, float] = {}
        local_add: dict[str, float] = {}
        projected_mult: dict[str, float] = {}
        projected_add: dict[str, float] = {}

        is_command_burst = "command burst" in group_name
        is_burst_jammer = "burst jammer" in group_name
        is_smart_bomb = ("smart bomb" in group_name) or ("structure area denial module" in group_name)
        is_bomb_launcher = ("bomb launcher" in group_name)
        if not is_bomb_launcher and loaded_charge is not None:
            try:
                is_bomb_launcher = bool(loaded_charge.requiresSkill("Bomb Deployment"))
            except Exception:
                is_bomb_launcher = False
        is_weapon_like = self._is_weapon_like_group(group_name)

        speed_factor = attr_opt("speedFactor")
        if speed_factor is not None and abs(float(speed_factor) - 1.0) > 1e-6 and range_m <= 0.0 and speed_factor > 0:
            local_mult["speed"] = max(local_mult.get("speed", 1.0), pct_to_mult(speed_factor))

        max_velocity_bonus = attr_opt("maxVelocityBonus")
        if max_velocity_bonus is not None and max_velocity_bonus > 0 and range_m <= 0.0:
            local_mult["speed"] = max(local_mult.get("speed", 1.0), pct_to_mult(max_velocity_bonus))

        signature_radius_bonus = attr_opt("signatureRadiusBonus")
        if signature_radius_bonus is not None and signature_radius_bonus != 0 and range_m <= 0.0:
            local_mult["sig"] = pct_to_mult(signature_radius_bonus)

        for attr_name, key in (
            ("scanResolutionBonus", "scan"),
            ("maxTargetRangeBonus", "range"),
            ("trackingSpeedBonus", "tracking"),
            ("maxRangeBonus", "optimal"),
            ("falloffBonus", "falloff"),
        ):
            value = attr_opt(attr_name)
            if value is None or abs(value) < 1e-9:
                continue
            if range_m <= 0.0 or value > 0.0:
                local_mult[key] = pct_to_mult(value)

        ecm_strengths = {
            "ecm_gravimetric": max(0.0, attr("scanGravimetricStrengthBonus", 0.0)),
            "ecm_ladar": max(0.0, attr("scanLadarStrengthBonus", 0.0)),
            "ecm_magnetometric": max(0.0, attr("scanMagnetometricStrengthBonus", 0.0)),
            "ecm_radar": max(0.0, attr("scanRadarStrengthBonus", 0.0)),
        }
        is_ecm = any(value > 0.0 for value in ecm_strengths.values()) or is_burst_jammer
        shield_rep = abs(attr("shieldBonus", 0.0))
        armor_rep = abs(attr("armorDamageAmount", 0.0))
        cap_drain = abs(attr("energyNeutralizerAmount", 0.0))
        nos_drain = abs(attr("powerTransferAmount", 0.0))
        if range_m > 0.0 and is_ecm:
            for key, value in ecm_strengths.items():
                if value > 0.0:
                    projected_add[key] = value

        if range_m > 0.0:
            if shield_rep > 0.0:
                projected_add["shield_rep"] = shield_rep
            if armor_rep > 0.0:
                projected_add["armor_rep"] = armor_rep
            if cap_drain > 0.0:
                projected_add["cap_drain"] = cap_drain
            if nos_drain > 0.0 and "nosferatu" in group_name:
                projected_add["cap_drain"] = max(projected_add.get("cap_drain", 0.0), nos_drain)
        else:
            local_rep = abs(attr("shieldBonus", 0.0)) + abs(attr("armorDamageAmount", 0.0))
            if local_rep > 0.0:
                local_add["rep"] = local_add.get("rep", 0.0) + local_rep

        if is_smart_bomb and range_m > 0.0:
            damage_components = self._module_damage_components(
                fitted_module,
                read_active=True,
            )
            if damage_components is not None:
                damage_cycle = max(0.1, cycle_sec)
                em_dps, thermal_dps, kinetic_dps, explosive_dps = damage_components
                if (em_dps + thermal_dps + kinetic_dps + explosive_dps) > 0.0:
                    projected_add["damage_em"] = em_dps * damage_cycle
                    projected_add["damage_thermal"] = thermal_dps * damage_cycle
                    projected_add["damage_kinetic"] = kinetic_dps * damage_cycle
                    projected_add["damage_explosive"] = explosive_dps * damage_cycle

        if is_weapon_like:
            damage_components = self._module_damage_components(
                fitted_module,
                read_active=True,
            )
            if damage_components is not None:
                damage_cycle = max(0.1, cycle_sec)
                em_dps, thermal_dps, kinetic_dps, explosive_dps = damage_components
                if (em_dps + thermal_dps + kinetic_dps + explosive_dps) > 0.0:
                    projected_add["damage_em"] = em_dps * damage_cycle
                    projected_add["damage_thermal"] = thermal_dps * damage_cycle
                    projected_add["damage_kinetic"] = kinetic_dps * damage_cycle
                    projected_add["damage_explosive"] = explosive_dps * damage_cycle

                    if "launcher" in group_name:
                        if is_bomb_launcher:
                            projected_add["weapon_is_bomb"] = 1.0
                            projected_add["weapon_blast_radius"] = max(0.0, charge_attr("explosionRange", 0.0))
                        else:
                            projected_add["weapon_is_missile"] = 1.0
                        explosion_radius = max(0.0, charge_attr("aoeCloudSize", 0.0))
                        explosion_velocity = max(0.0, charge_attr("aoeVelocity", 0.0))
                        damage_reduction_factor = max(
                            0.1,
                            charge_attr_opt("aoeDamageReductionFactor")
                            or charge_attr_opt("damageReductionFactor")
                            or 0.5,
                        )
                        charge_velocity = max(0.0, charge_attr("maxVelocity", 0.0))
                        charge_delay_sec = max(0.0, charge_attr("explosionDelay", 0.0) / 1000.0)
                        charge_mass = max(0.0, charge_attr("mass", 0.0))
                        charge_agility = max(0.0, charge_attr("agility", 0.0))
                        projectile_shield_hp = max(0.0, charge_attr("shieldCapacity", 0.0))
                        projectile_armor_hp = max(0.0, charge_attr("armorHP", 0.0))
                        projectile_structure_hp = max(0.0, charge_attr("hp", 0.0))
                        charge_range = 0.0
                        if charge_velocity > 0.0 and charge_delay_sec > 0.0:
                            accel_time = min(charge_delay_sec, (charge_mass * charge_agility) / 1_000_000.0) if charge_mass > 0.0 and charge_agility > 0.0 else 0.0
                            charge_range = (charge_velocity * 0.5 * accel_time) + (charge_velocity * max(0.0, charge_delay_sec - accel_time))
                        if range_m <= 0.0 and charge_range > 0.0:
                            range_m = charge_range
                        if falloff_m <= 1.0:
                            falloff_m = 0.0
                        projected_add["weapon_explosion_radius"] = explosion_radius
                        projected_add["weapon_explosion_velocity"] = explosion_velocity
                        projected_add["weapon_drf"] = damage_reduction_factor
                        projected_add["weapon_projectile_speed"] = charge_velocity
                        projected_add["weapon_projectile_flight_time"] = charge_delay_sec
                        projected_add["weapon_projectile_mass"] = charge_mass
                        projected_add["weapon_projectile_agility"] = charge_agility
                        projected_add["weapon_projectile_shield_hp"] = projectile_shield_hp
                        projected_add["weapon_projectile_armor_hp"] = projectile_armor_hp
                        projected_add["weapon_projectile_structure_hp"] = projectile_structure_hp
                        projected_add["weapon_projectile_shield_resonance_em"] = max(
                            0.0,
                            charge_attr_opt("shieldEmDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_shield_resonance_thermal"] = max(
                            0.0,
                            charge_attr_opt("shieldThermalDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_shield_resonance_kinetic"] = max(
                            0.0,
                            charge_attr_opt("shieldKineticDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_shield_resonance_explosive"] = max(
                            0.0,
                            charge_attr_opt("shieldExplosiveDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_armor_resonance_em"] = max(
                            0.0,
                            charge_attr_opt("armorEmDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_armor_resonance_thermal"] = max(
                            0.0,
                            charge_attr_opt("armorThermalDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_armor_resonance_kinetic"] = max(
                            0.0,
                            charge_attr_opt("armorKineticDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_armor_resonance_explosive"] = max(
                            0.0,
                            charge_attr_opt("armorExplosiveDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_structure_resonance_em"] = max(
                            0.0,
                            charge_attr_opt("emDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_structure_resonance_thermal"] = max(
                            0.0,
                            charge_attr_opt("thermalDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_structure_resonance_kinetic"] = max(
                            0.0,
                            charge_attr_opt("kineticDamageResonance") or 1.0,
                        )
                        projected_add["weapon_projectile_structure_resonance_explosive"] = max(
                            0.0,
                            charge_attr_opt("explosiveDamageResonance") or 1.0,
                        )
                    else:
                        projected_add["weapon_is_turret"] = 1.0
                        projected_add["weapon_tracking"] = max(0.0, attr("trackingSpeed", 0.0))
                        projected_add["weapon_optimal_sig"] = max(1.0, attr("optimalSigRadius", 40_000.0))

        projected_mult_groups: dict[str, str | None] = {}
        projected_signature: tuple[Any, ...] = tuple()
        should_record_projected_profile = (
            range_m > 0.0
            and not is_ecm
            and not is_command_burst
            and not is_smart_bomb
            and not is_weapon_like
            and shield_rep <= 0.0
            and armor_rep <= 0.0
            and cap_drain <= 0.0
            and not (nos_drain > 0.0 and "nosferatu" in group_name)
        )
        if should_record_projected_profile:
            recorded_mult, recorded_add, recorded_groups, recorded_signature = self._projected_effects_from_pyfa_handler(
                fitted_module,
            )
            projected_mult.update(recorded_mult)
            projected_add.update(recorded_add)
            projected_mult_groups.update(recorded_groups)
            projected_signature = recorded_signature

        cap_capacity_bonus = attr_opt("capacitorCapacityBonus")
        if cap_capacity_bonus is not None and abs(cap_capacity_bonus) > 1e-9:
            local_mult["cap_max"] = pct_to_mult(cap_capacity_bonus)

        cap_recharge_mult = attr_opt("capacitorRechargeRateMultiplier")
        if cap_recharge_mult is not None and cap_recharge_mult > 0 and abs(cap_recharge_mult - 1.0) > 1e-6:
            local_mult["cap_recharge"] = max(0.01, cap_recharge_mult)

        if not is_weapon_like:
            damage_scale = 1.0
            damage_multiplier_bonus = attr_opt("damageMultiplierBonus")
            if damage_multiplier_bonus is not None and abs(damage_multiplier_bonus) > 1e-9:
                damage_scale *= pct_to_mult(damage_multiplier_bonus)

            missile_damage_bonus = attr_opt("missileDamageMultiplierBonus")
            if missile_damage_bonus is not None and abs(missile_damage_bonus) > 1e-9:
                if missile_damage_bonus > 2.0:
                    damage_scale *= max(0.01, missile_damage_bonus)
                else:
                    damage_scale *= pct_to_mult(missile_damage_bonus)

            direct_damage_multiplier = attr_opt("damageMultiplier")
            if direct_damage_multiplier is not None and 0.0 < direct_damage_multiplier <= 2.0:
                damage_scale *= max(0.01, direct_damage_multiplier)

            rof_multiplier = attr_opt("speedMultiplier")
            if rof_multiplier is not None and rof_multiplier > 0 and abs(rof_multiplier - 1.0) > 1e-6:
                damage_scale *= 1.0 / max(0.01, rof_multiplier)

            if abs(damage_scale - 1.0) > 1e-6:
                local_mult["dps"] = max(0.01, damage_scale)

        if is_weapon_like and range_m <= 0.0:
            range_m = max(
                range_m,
                float(projected_add.get("weapon_projectile_speed", 0.0) or 0.0)
                * float(projected_add.get("weapon_projectile_flight_time", 0.0) or 0.0),
            )
        if pyfa_cycle_sec > 0.0:
            cycle_sec = max(0.1, pyfa_cycle_sec)

        has_projected_damage = any(str(key).startswith("damage_") for key in projected_add.keys())
        is_weapon = is_weapon_like and has_projected_damage
        has_projected_rep = shield_rep > 0.0 or armor_rep > 0.0
        is_cap_warfare = cap_drain > 0.0 or (nos_drain > 0.0 and "nosferatu" in group_name)
        is_offensive_ewar = any(token in group_name for token in ("weapon disruptor", "sensor damp", "warp scrambler"))
        is_target_ewar = any(token in group_name for token in ("target painter", "stasis web", "stasis grappler"))
        is_cap_booster = "capacitor booster" in group_name
        is_propulsion = "propulsion module" in group_name
        is_damage_control = group_name == "damage control"
        is_hardener = any(
            token in group_name
            for token in (
                "shield hardener",
                "armor hardener",
                "energized",
                "armor resistance shift hardener",
            )
        )
        has_projected = (
            range_m > 0.0
            and (
                bool(projected_mult or projected_add or projected_signature)
                or should_record_projected_profile
                or is_command_burst
                or is_smart_bomb
                or is_burst_jammer
                or is_ecm
                or is_weapon
                or has_projected_rep
                or is_cap_warfare
            )
        )
        is_active_module = supports_active_state and ((cap_need > 0.0) or (cycle_ms > 0.0) or has_projected)
        if self._is_weapon_like_group(group_name) and cycle_ms > 0.0:
            is_active_module = True

        state_required = ModuleState.ACTIVE if is_active_module else ModuleState.ONLINE
        module_state = ModuleState.ONLINE
        module_id = f"mod{suffix}"
        affects_local_pyfa_profile = is_command_burst or (not has_projected and state_required == ModuleState.ACTIVE)
        module_tags = self._module_tags(
            item=item,
            has_projected=has_projected,
            is_active_module=is_active_module,
            affects_local_pyfa_profile=affects_local_pyfa_profile,
            is_command_burst=is_command_burst,
            is_smart_bomb=is_smart_bomb,
            is_burst_jammer=is_burst_jammer,
            is_weapon=is_weapon,
            has_projected_rep=has_projected_rep,
            is_cap_warfare=is_cap_warfare,
            is_offensive_ewar=is_offensive_ewar,
            is_target_ewar=is_target_ewar,
            is_ecm=is_ecm,
            is_propulsion=is_propulsion,
            is_damage_control=is_damage_control,
            is_hardener=is_hardener,
            is_cap_booster=is_cap_booster,
        )

        charge_capacity = 0
        item_modified_attrs = getattr(fitted_module, "itemModifiedAttributes", None)
        has_charge_rate_attr = False
        if item_modified_attrs is not None:
            try:
                has_charge_rate_attr = "chargeRate" in item_modified_attrs
            except Exception:
                has_charge_rate_attr = False
        charge_rate = max(0.0, attr("chargeRate", 0.0)) if has_charge_rate_attr else 0.0
        charge_reload_time = max(0.0, attr("reloadTime", 0.0) / 1000.0)
        charge_remaining = 0.0
        if loaded_charge is not None:
            try:
                charge_capacity = int(float(getattr(fitted_module, "numCharges", 0) or 0))
            except Exception:
                charge_capacity = 0
            if charge_capacity <= 0:
                try:
                    module_capacity = float(item.getAttribute("capacity", None) or 0.0)
                except Exception:
                    module_capacity = 0.0
                try:
                    charge_volume = float(loaded_charge.getAttribute("volume", None) or 0.0)
                except Exception:
                    charge_volume = 0.0
                if module_capacity > 0.0 and charge_volume > 0.0:
                    charge_capacity = int(module_capacity / charge_volume)
            charge_remaining = float(max(0, charge_capacity))
        else:
            charge_rate = 0.0

        if charge_capacity <= 0:
            charge_rate = 0.0
            charge_remaining = 0.0

        if has_projected:
            effect = ModuleEffect(
                f"projected{suffix}",
                EffectClass.PROJECTED,
                ModuleState.ACTIVE,
                range_m,
                falloff_m,
                cycle_sec,
                cap_need,
                reactivation_delay_sec,
                {},
                {},
                projected_mult,
                projected_add,
                projected_mult_groups,
                projected_signature,
            )
        else:
            effect = ModuleEffect(
                f"local{suffix}",
                EffectClass.LOCAL,
                state_required,
                0.0,
                0.0,
                cycle_sec,
                cap_need if state_required == ModuleState.ACTIVE else 0.0,
                reactivation_delay_sec,
                local_mult,
                local_add,
                {},
                {},
                {},
                tuple(),
            )

        return ModuleRuntime(
            module_id=module_id,
            group=str(item.group.name or ""),
            state=module_state,
            effects=[effect],
            charge_capacity=charge_capacity,
            charge_rate=charge_rate,
            charge_remaining=charge_remaining,
            charge_reload_time=charge_reload_time,
            tags=module_tags,
        )

    @staticmethod
    def _collect_pyfa_weapon_stats(modules: list[Any], ship: Any, *, require_volley: bool) -> dict[str, float]:
        turret_dps = 0.0
        missile_dps = 0.0
        turret_volley = 0.0
        missile_volley = 0.0
        weighted_optimal_sig = 0.0
        turret_cycle_weighted = 0.0
        turret_cycle_weight = 0.0
        missile_cycle_weighted = 0.0
        missile_cycle_weight = 0.0
        turret_weighted_optimal = 0.0
        turret_weighted_falloff = 0.0
        turret_weighted_tracking = 0.0
        missile_weighted_optimal = 0.0
        missile_weighted_falloff = 0.0
        missile_weighted_tracking = 0.0
        turret_em = turret_th = turret_ki = turret_ex = 0.0
        missile_em = missile_th = missile_ki = missile_ex = 0.0
        missile_sig_radius = 0.0
        missile_explosion_velocity = 0.0
        missile_max_range = 0.0
        missile_drf = 0.5

        for module in modules:
            try:
                item = module.item
            except Exception:
                item = None
            if item is None:
                continue
            group_name = str(getattr(getattr(item, "group", None), "name", "") or "").lower()
            is_launcher = "launcher" in group_name
            is_turret = ("weapon" in group_name or "turret" in group_name) and not is_launcher
            if not (is_turret or is_launcher):
                continue

            try:
                dps_obj = module.getDps()
                volley_obj = module.getVolley() if require_volley else None
            except Exception:
                continue

            dps_total = float(getattr(dps_obj, "total", 0.0) or 0.0)
            if dps_total <= 0.0:
                continue
            volley_total = float(getattr(volley_obj, "total", 0.0) or 0.0) if volley_obj is not None else 0.0

            cycle = float(module.getModifiedItemAttr("speed") or 0.0) / 1000.0
            cycle = max(0.1, cycle)

            em_dps = float(getattr(dps_obj, "em", 0.0) or 0.0)
            th_dps = float(getattr(dps_obj, "thermal", 0.0) or 0.0)
            ki_dps = float(getattr(dps_obj, "kinetic", 0.0) or 0.0)
            ex_dps = float(getattr(dps_obj, "explosive", 0.0) or 0.0)

            optimal = float(module.getModifiedItemAttr("maxRange") or 0.0)
            falloff = float(module.getModifiedItemAttr("falloff") or 0.0)
            tracking = float(module.getModifiedItemAttr("trackingSpeed") or 0.0)

            if is_launcher:
                missile_dps += dps_total
                missile_volley += volley_total
                missile_cycle_weighted += cycle * dps_total
                missile_cycle_weight += dps_total
                missile_weighted_optimal += optimal * dps_total
                missile_weighted_falloff += falloff * dps_total
                missile_weighted_tracking += tracking * dps_total
                missile_em += em_dps
                missile_th += th_dps
                missile_ki += ki_dps
                missile_ex += ex_dps
                try:
                    explosion_radius = float(module.getModifiedChargeAttr("aoeCloudSize") or 0.0)
                    explosion_velocity = float(module.getModifiedChargeAttr("aoeVelocity") or 0.0)
                    damage_reduction_factor = float(module.getModifiedChargeAttr("damageReductionFactor") or 0.5)
                    charge_velocity = float(module.getModifiedChargeAttr("maxVelocity") or 0.0)
                    charge_delay_ms = float(module.getModifiedChargeAttr("explosionDelay") or 0.0)
                    charge_range = charge_velocity * (charge_delay_ms / 1000.0) if charge_velocity > 0 and charge_delay_ms > 0 else 0.0
                except Exception:
                    explosion_radius = 0.0
                    explosion_velocity = 0.0
                    damage_reduction_factor = 0.5
                    charge_range = 0.0
                missile_sig_radius = max(missile_sig_radius, explosion_radius)
                missile_explosion_velocity = max(missile_explosion_velocity, explosion_velocity)
                missile_drf = max(0.1, min(2.0, damage_reduction_factor))
                missile_max_range = max(missile_max_range, charge_range, optimal)
            else:
                turret_dps += dps_total
                turret_volley += volley_total
                turret_cycle_weighted += cycle * dps_total
                turret_cycle_weight += dps_total
                turret_weighted_optimal += optimal * dps_total
                turret_weighted_falloff += falloff * dps_total
                turret_weighted_tracking += tracking * dps_total
                turret_optimal_sig = float(module.getModifiedItemAttr("optimalSigRadius") or 40_000.0)
                weighted_optimal_sig += turret_optimal_sig * dps_total
                turret_em += em_dps
                turret_th += th_dps
                turret_ki += ki_dps
                turret_ex += ex_dps

        total_weapon_dps = turret_dps + missile_dps
        optimal = float(ship.getModifiedItemAttr("maxTargetRange") or 0.0)
        falloff = 1.0
        tracking = 0.0001
        if total_weapon_dps > 0:
            weighted_optimal = turret_weighted_optimal + missile_weighted_optimal
            weighted_falloff = turret_weighted_falloff + missile_weighted_falloff
            weighted_tracking = turret_weighted_tracking + missile_weighted_tracking
            optimal = max(1.0, weighted_optimal / total_weapon_dps)
            falloff = max(1.0, weighted_falloff / total_weapon_dps)
            tracking = max(0.0001, weighted_tracking / total_weapon_dps)

        return {
            "dps": total_weapon_dps,
            "volley": turret_volley + missile_volley,
            "turret_dps": turret_dps,
            "missile_dps": missile_dps,
            "turret_cycle": (turret_cycle_weighted / max(1e-6, turret_cycle_weight)) if turret_dps > 0 else 0.0,
            "missile_cycle": (missile_cycle_weighted / max(1e-6, missile_cycle_weight)) if missile_dps > 0 else 0.0,
            "damage_em": turret_em + missile_em,
            "damage_thermal": turret_th + missile_th,
            "damage_kinetic": turret_ki + missile_ki,
            "damage_explosive": turret_ex + missile_ex,
            "turret_em_dps": turret_em,
            "turret_thermal_dps": turret_th,
            "turret_kinetic_dps": turret_ki,
            "turret_explosive_dps": turret_ex,
            "missile_em_dps": missile_em,
            "missile_thermal_dps": missile_th,
            "missile_kinetic_dps": missile_ki,
            "missile_explosive_dps": missile_ex,
            "missile_explosion_radius": missile_sig_radius,
            "missile_explosion_velocity": missile_explosion_velocity,
            "missile_max_range": missile_max_range,
            "missile_damage_reduction_factor": missile_drf,
            "optimal": optimal,
            "falloff": falloff,
            "tracking": tracking,
            "optimal_sig": weighted_optimal_sig / max(1e-6, turret_dps) if turret_dps > 0 else 40_000.0,
        }

    def _build_pyfa_fit(
        self,
        parsed: ParsedEftFit,
        state_by_module_id: dict[str, str] | None = None,
        *,
        calculate_modified_attributes: bool = True,
        default_online: bool = False,
        ignore_offline_specs: bool = False,
    ) -> tuple[Any, list[tuple[ParsedModuleSpec, Any, str | None]]]:
        if not self._pyfa.fit_engine_ready:
            raise ValueError("pyfa static database unavailable")

        fit_cls = self._pyfa._fit_cls
        ship_cls = self._pyfa._ship_cls
        module_cls = self._pyfa._module_cls
        character_cls = self._pyfa._character_cls
        active_state = self._pyfa._fitting_module_state_active
        online_state = self._pyfa._fitting_module_state_online
        offline_state = self._pyfa._fitting_module_state_offline
        overheated_state = self._pyfa._fitting_module_state_overheated
        assert fit_cls is not None
        assert ship_cls is not None
        assert module_cls is not None
        assert character_cls is not None
        assert active_state is not None
        assert online_state is not None
        assert offline_state is not None

        ship_name = self._pyfa.resolve_type_name(parsed.ship_name)
        ship_item = self._pyfa.get_item(ship_name)
        if ship_item is None:
            raise ValueError(f"pyfa ship not found: {parsed.ship_name}")

        fit = fit_cls(ship=ship_cls(ship_item), name=parsed.fit_name)
        try:
            fit.implantLocation = 0
        except Exception:
            pass
        fit.character = character_cls.getAll5()
        fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]] = []

        for idx, spec in enumerate(parsed.module_specs, start=1):
            module_name = self._pyfa.resolve_type_name(spec.module_name)
            module_item = self._pyfa.get_item(module_name)
            if module_item is None:
                raise ValueError(f"pyfa module not found: {spec.module_name}")
            module = module_cls(module_item)
            module.owner = fit
            module_id = f"mod-{idx}"

            group_name = (module_item.group.name or "").lower()
            charge_name = self._resolve_module_charge_name(module_item, spec.charge_name)
            if self._is_weapon_like_group(group_name) and not charge_name:
                raise ValueError(f"武器缺少可解析弹药：{spec.module_name}")
            if charge_name:
                charge_item = self._pyfa.get_item(self._pyfa.resolve_type_name(charge_name))
                if charge_item is None:
                    raise ValueError(f"pyfa charge not found: {charge_name}")
                module.charge = charge_item

            default_runtime_state = "ONLINE" if default_online else ("ACTIVE" if self._is_weapon_like_group(group_name) else "ONLINE")
            runtime_state = str((state_by_module_id or {}).get(module_id, default_runtime_state) or default_runtime_state).upper()
            if spec.offline and not ignore_offline_specs:
                runtime_state = "OFFLINE"
            module.state = _pyfa_module_state_from_runtime_state(self, module, runtime_state)

            fit.modules.append(module)
            fitted_modules.append((spec, module, charge_name))

        if calculate_modified_attributes:
            _calculate_pyfa_fit_with_state_normalization(self, fit, fitted_modules)
        return fit, fitted_modules

    def _compute_pyfa_final_stats(self, fit) -> dict[str, float]:
        ship = fit.ship
        weapon_stats = self._collect_pyfa_weapon_stats(cast(list[Any], fit.modules), ship, require_volley=True)
        warp_scramble_status = float(ship.getModifiedItemAttr("warpScrambleStatus") or 0.0)
        total_dps = max(float(fit.getTotalDps().total), float(weapon_stats.get("dps", 0.0) or 0.0))
        total_volley = max(float(fit.getTotalVolley().total), float(weapon_stats.get("volley", 0.0) or 0.0))
        return {
            "dps": total_dps,
            "volley": total_volley,
            "turret_dps": weapon_stats["turret_dps"],
            "missile_dps": weapon_stats["missile_dps"],
            "turret_cycle": weapon_stats["turret_cycle"],
            "missile_cycle": weapon_stats["missile_cycle"],
            "damage_em": weapon_stats["damage_em"],
            "damage_thermal": weapon_stats["damage_thermal"],
            "damage_kinetic": weapon_stats["damage_kinetic"],
            "damage_explosive": weapon_stats["damage_explosive"],
            "turret_em_dps": weapon_stats["turret_em_dps"],
            "turret_thermal_dps": weapon_stats["turret_thermal_dps"],
            "turret_kinetic_dps": weapon_stats["turret_kinetic_dps"],
            "turret_explosive_dps": weapon_stats["turret_explosive_dps"],
            "missile_em_dps": weapon_stats["missile_em_dps"],
            "missile_thermal_dps": weapon_stats["missile_thermal_dps"],
            "missile_kinetic_dps": weapon_stats["missile_kinetic_dps"],
            "missile_explosive_dps": weapon_stats["missile_explosive_dps"],
            "missile_explosion_radius": weapon_stats["missile_explosion_radius"],
            "missile_explosion_velocity": weapon_stats["missile_explosion_velocity"],
            "missile_max_range": weapon_stats["missile_max_range"],
            "missile_damage_reduction_factor": weapon_stats["missile_damage_reduction_factor"],
            "optimal": weapon_stats["optimal"],
            "falloff": weapon_stats["falloff"],
            "tracking": weapon_stats["tracking"],
            "optimal_sig": weapon_stats["optimal_sig"],
            "max_speed": float(ship.getModifiedItemAttr("maxVelocity") or 0.0),
            "mass": float(ship.getModifiedItemAttr("mass") or 0.0),
            "agility": float(ship.getModifiedItemAttr("agility") or 0.0),
            "sig_radius": float(ship.getModifiedItemAttr("signatureRadius") or 0.0),
            "scan_resolution": float(ship.getModifiedItemAttr("scanResolution") or 0.0),
            "max_target_range": float(ship.getModifiedItemAttr("maxTargetRange") or 0.0),
            "max_locked_targets": int(fit.maxTargets),
            "scan_strength": float(fit.scanStrength or 0.0),
            "ecm_jam_chance": max(0.0, min(1.0, float(fit.jamChance or 0.0) / 100.0)),
            "sensor_strength_gravimetric": float(ship.getModifiedItemAttr("scanGravimetricStrength") or 0.0),
            "sensor_strength_ladar": float(ship.getModifiedItemAttr("scanLadarStrength") or 0.0),
            "sensor_strength_magnetometric": float(ship.getModifiedItemAttr("scanMagnetometricStrength") or 0.0),
            "sensor_strength_radar": float(ship.getModifiedItemAttr("scanRadarStrength") or 0.0),
            "warp_scramble_status": warp_scramble_status,
            "warp_stability": -warp_scramble_status,
            "energy_warfare_resistance": float(ship.getModifiedItemAttr("energyWarfareResistance") or 1.0),
            "max_cap": float(ship.getModifiedItemAttr("capacitorCapacity") or 0.0),
            "cap_recharge_time": float(ship.getModifiedItemAttr("rechargeRate") or 0.0) / 1000.0,
            "shield_hp": float(ship.getModifiedItemAttr("shieldCapacity") or 0.0),
            "armor_hp": float(ship.getModifiedItemAttr("armorHP") or 0.0),
            "structure_hp": float(ship.getModifiedItemAttr("hp") or 0.0),
            "shield_resonance_em": float(ship.getModifiedItemAttr("shieldEmDamageResonance") or 1.0),
            "shield_resonance_thermal": float(ship.getModifiedItemAttr("shieldThermalDamageResonance") or 1.0),
            "shield_resonance_kinetic": float(ship.getModifiedItemAttr("shieldKineticDamageResonance") or 1.0),
            "shield_resonance_explosive": float(ship.getModifiedItemAttr("shieldExplosiveDamageResonance") or 1.0),
            "armor_resonance_em": float(ship.getModifiedItemAttr("armorEmDamageResonance") or 1.0),
            "armor_resonance_thermal": float(ship.getModifiedItemAttr("armorThermalDamageResonance") or 1.0),
            "armor_resonance_kinetic": float(ship.getModifiedItemAttr("armorKineticDamageResonance") or 1.0),
            "armor_resonance_explosive": float(ship.getModifiedItemAttr("armorExplosiveDamageResonance") or 1.0),
            "structure_resonance_em": float(ship.getModifiedItemAttr("emDamageResonance") or 1.0),
            "structure_resonance_thermal": float(ship.getModifiedItemAttr("thermalDamageResonance") or 1.0),
            "structure_resonance_kinetic": float(ship.getModifiedItemAttr("kineticDamageResonance") or 1.0),
            "structure_resonance_explosive": float(ship.getModifiedItemAttr("explosiveDamageResonance") or 1.0),
        }

    def _extract_pyfa_ship_attribute_penalty_context(self, fit_ctx: Any) -> dict[str, dict[str, Any]]:
        modified_attrs = getattr(getattr(fit_ctx, "ship", None), "itemModifiedAttributes", None)
        if modified_attrs is None:
            return {}

        forced = getattr(modified_attrs, "_ModifiedAttributeDict__forced", {})
        pre_assigns = getattr(modified_attrs, "_ModifiedAttributeDict__preAssigns", {})
        pre_increases = getattr(modified_attrs, "_ModifiedAttributeDict__preIncreases", {})
        multipliers = getattr(modified_attrs, "_ModifiedAttributeDict__multipliers", {})
        penalized = getattr(modified_attrs, "_ModifiedAttributeDict__penalizedMultipliers", {})
        post_increases = getattr(modified_attrs, "_ModifiedAttributeDict__postIncreases", {})

        attr_map = {
            "maxVelocity": "speed",
            "signatureRadius": "sig",
            "scanResolution": "scan",
            "maxTargetRange": "range",
            "shieldCapacity": "shield_hp",
            "armorHP": "armor_hp",
            "hp": "structure_hp",
            "capacitorCapacity": "cap_max",
            "rechargeRate": "cap_recharge",
            "mass": "mass",
            "agility": "agility",
            "scanGravimetricStrength": "sensor_strength_gravimetric",
            "scanLadarStrength": "sensor_strength_ladar",
            "scanMagnetometricStrength": "sensor_strength_magnetometric",
            "scanRadarStrength": "sensor_strength_radar",
            "shieldEmDamageResonance": "shield_resonance_em",
            "shieldThermalDamageResonance": "shield_resonance_thermal",
            "shieldKineticDamageResonance": "shield_resonance_kinetic",
            "shieldExplosiveDamageResonance": "shield_resonance_explosive",
            "armorEmDamageResonance": "armor_resonance_em",
            "armorThermalDamageResonance": "armor_resonance_thermal",
            "armorKineticDamageResonance": "armor_resonance_kinetic",
            "armorExplosiveDamageResonance": "armor_resonance_explosive",
            "emDamageResonance": "structure_resonance_em",
            "thermalDamageResonance": "structure_resonance_thermal",
            "kineticDamageResonance": "structure_resonance_kinetic",
            "explosiveDamageResonance": "structure_resonance_explosive",
        }

        contexts: dict[str, dict[str, Any]] = {}
        for attr_name, formula_key in attr_map.items():
            if attr_name in forced:
                continue
            original = pre_assigns.get(attr_name, modified_attrs.getOriginal(attr_name))
            if original is None:
                continue
            try:
                pre_penalty_value = float(original or 0.0) + float(pre_increases.get(attr_name, 0.0) or 0.0)
                pre_penalty_value *= float(multipliers.get(attr_name, 1.0) or 1.0)
                post_penalty_value = float(post_increases.get(attr_name, 0.0) or 0.0)
            except Exception:
                continue

            group_values: dict[str, list[float]] = {}
            raw_groups = penalized.get(attr_name, {})
            if isinstance(raw_groups, dict):
                for group_name, values in raw_groups.items():
                    if not isinstance(values, list):
                        continue
                    parsed_values = [float(value or 1.0) for value in values]
                    if parsed_values:
                        group_values[str(group_name)] = parsed_values

            if not group_values and abs(post_penalty_value) <= 1e-9:
                continue

            contexts[formula_key] = {
                "pre": pre_penalty_value,
                "post": post_penalty_value,
                "groups": group_values,
            }
        return contexts

    def _extract_pyfa_weapon_attribute_penalty_context(self, fit_ctx: Any) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        attr_map = {
            "maxRange": "optimal",
            "falloff": "falloff",
            "trackingSpeed": "tracking",
        }

        for module in getattr(fit_ctx, "modules", []) or []:
            item = getattr(module, "item", None)
            if item is None:
                continue
            group_name = str(getattr(getattr(item, "group", None), "name", "") or "").lower()
            is_launcher = "launcher" in group_name
            is_turret = ("weapon" in group_name or "turret" in group_name) and not is_launcher
            if not (is_turret or is_launcher):
                continue

            try:
                dps_total = float(getattr(module.getDps(), "total", 0.0) or 0.0)
            except Exception:
                dps_total = 0.0
            if dps_total <= 0.0:
                continue

            modified_attrs = getattr(module, "itemModifiedAttributes", None)
            if modified_attrs is None:
                continue

            forced = getattr(modified_attrs, "_ModifiedAttributeDict__forced", {})
            pre_assigns = getattr(modified_attrs, "_ModifiedAttributeDict__preAssigns", {})
            pre_increases = getattr(modified_attrs, "_ModifiedAttributeDict__preIncreases", {})
            multipliers = getattr(modified_attrs, "_ModifiedAttributeDict__multipliers", {})
            penalized = getattr(modified_attrs, "_ModifiedAttributeDict__penalizedMultipliers", {})
            post_increases = getattr(modified_attrs, "_ModifiedAttributeDict__postIncreases", {})

            entry: dict[str, Any] = {
                "weight": dps_total,
                "kind": "gunnery" if is_turret else "launcher",
            }

            for attr_name, formula_key in attr_map.items():
                if attr_name in forced:
                    continue

                try:
                    current_value = float(module.getModifiedItemAttr(attr_name) or 0.0)
                except Exception:
                    current_value = 0.0

                original = pre_assigns.get(attr_name, modified_attrs.getOriginal(attr_name))
                context: dict[str, Any] = {
                    "current": current_value,
                }
                if original is not None:
                    try:
                        pre_penalty_value = float(original or 0.0) + float(pre_increases.get(attr_name, 0.0) or 0.0)
                        pre_penalty_value *= float(multipliers.get(attr_name, 1.0) or 1.0)
                        context["pre"] = pre_penalty_value
                        context["post"] = float(post_increases.get(attr_name, 0.0) or 0.0)
                    except Exception:
                        pass

                group_values: dict[str, list[float]] = {}
                raw_groups = penalized.get(attr_name, {})
                if isinstance(raw_groups, dict):
                    for group_name, values in raw_groups.items():
                        if not isinstance(values, list):
                            continue
                        parsed_values = [float(value or 1.0) for value in values]
                        if parsed_values:
                            group_values[str(group_name)] = parsed_values
                if group_values:
                    context["groups"] = group_values

                entry[formula_key] = context

            contexts.append(entry)

        return contexts

    @staticmethod
    def _module_state_text(value: str) -> ModuleState:
        state_name = str(value or "ONLINE").upper()
        if state_name in ModuleState.__members__:
            return ModuleState[state_name]
        return ModuleState.ONLINE

    def _build_runtime_artifacts_from_pyfa_fit(
        self,
        parsed: ParsedEftFit,
        fit_ctx: Any,
        fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]],
        state_by_module_id: dict[str, str] | None = None,
        command_booster_snapshots: list[dict[str, Any]] | None = None,
    ) -> tuple[FitRuntime, FitDescriptor, ShipProfile]:
        temporary_weapon_states: list[tuple[Any, Any]] = []
        for idx, (spec, fitted_module, _effective_charge_name) in enumerate(fitted_modules, start=1):
            module_id = f"mod-{idx}"
            if bool(spec.offline):
                continue
            requested_state = str((state_by_module_id or {}).get(module_id, "ONLINE") or "ONLINE").upper()
            if requested_state == "OFFLINE":
                continue
            group_name = str(getattr(getattr(fitted_module, "item", None), "group", None).name or "")
            if not self._is_weapon_like_group(group_name):
                continue
            active_state = _pyfa_module_state_from_runtime_state(self, fitted_module, "ACTIVE")
            if active_state is None or active_state == fitted_module.state:
                continue
            temporary_weapon_states.append((fitted_module, fitted_module.state))
            fitted_module.state = active_state

        try:
            pyfa_final = self._compute_pyfa_final_stats(fit_ctx)

            ship_item = getattr(getattr(fit_ctx, "ship", None), "item", None)
            ship_name = str(getattr(ship_item, "typeName", parsed.ship_name) or parsed.ship_name)
            role = "DPS"
            ship_group_name = str(getattr(getattr(ship_item, "group", None), "name", "") or "").lower()
            if "logistics" in ship_group_name:
                role = "LOGI"
            elif any(x in ship_group_name for x in ("electronic", "force recon", "combat recon")):
                role = "EWAR"

            modules: list[ModuleRuntime] = []
            pyfa_blueprint_modules: list[dict[str, Any]] = []
            pyfa_max_state_by_module_id: dict[str, str] = {}
            for idx, (spec, fitted_module, effective_charge_name) in enumerate(fitted_modules, start=1):
                module_id = f"mod-{idx}"
                requested_state_name = (
                    "OFFLINE"
                    if spec.offline
                    else str((state_by_module_id or {}).get(module_id, "ONLINE") or "ONLINE")
                )
                requested_state = self._module_state_text(requested_state_name)
                module = self._module_effect_pyfa(
                    parsed,
                    fit_ctx,
                    fitted_modules,
                    fitted_module,
                    idx,
                    state_by_module_id=state_by_module_id,
                    command_booster_snapshots=command_booster_snapshots,
                )
                if module is not None:
                    actual_state = (
                        ModuleState.OFFLINE
                        if spec.offline
                        else _runtime_state_from_pyfa_state_value(
                            self,
                            _pyfa_module_state_from_runtime_state(self, fitted_module, requested_state_name),
                            requested_state,
                        )
                    )
                    max_allowed_state = (
                        ModuleState.OFFLINE
                        if spec.offline
                        else _runtime_state_from_pyfa_state_value(
                            self,
                            _pyfa_module_state_from_runtime_state(self, fitted_module, "OVERHEATED"),
                            actual_state,
                        )
                    )
                    if spec.offline:
                        module.state = ModuleState.OFFLINE
                    elif state_by_module_id is not None:
                        module.state = _clamp_runtime_state_to_pyfa_max(requested_state, max_allowed_state)
                    else:
                        module.state = actual_state
                    pyfa_max_state_by_module_id[module_id] = max_allowed_state.value
                    modules.append(module)
                    pyfa_blueprint_modules.append(
                        {
                            "module_id": module.module_id,
                            "module_name": spec.module_name,
                            "charge_name": effective_charge_name,
                            "offline": bool(spec.offline),
                            "effect_names": sorted(self._module_effect_names(fitted_module)),
                        }
                    )

            turret_dps = max(0.0, float(pyfa_final.get("turret_dps", 0.0) or 0.0))
            missile_dps = max(0.0, float(pyfa_final.get("missile_dps", 0.0) or 0.0))

            profile = ShipProfile(
                dps=max(0.0, float(pyfa_final.get("dps", 0.0) or 0.0)),
                volley=max(0.0, float(pyfa_final.get("volley", 0.0) or 0.0)),
                optimal=max(1.0, float(pyfa_final.get("optimal", 0.0) or 0.0)),
                falloff=max(1.0, float(pyfa_final.get("falloff", 0.0) or 0.0)),
                tracking=max(0.0001, float(pyfa_final.get("tracking", 0.0) or 0.0)),
                optimal_sig=max(1.0, float(pyfa_final.get("optimal_sig", 40_000.0) or 40_000.0)),
                sig_radius=max(1.0, float(pyfa_final.get("sig_radius", 0.0) or 0.0)),
                scan_resolution=max(1.0, float(pyfa_final.get("scan_resolution", 0.0) or 0.0)),
                max_target_range=max(1000.0, float(pyfa_final.get("max_target_range", 0.0) or 0.0)),
                max_locked_targets=max(0, int(pyfa_final.get("max_locked_targets", 0) or 0)),
                scan_strength=max(0.0, float(pyfa_final.get("scan_strength", 0.0) or 0.0)),
                ecm_jam_chance=max(0.0, min(1.0, float(pyfa_final.get("ecm_jam_chance", 0.0) or 0.0))),
                sensor_strength_gravimetric=max(0.0, float(pyfa_final.get("sensor_strength_gravimetric", 0.0) or 0.0)),
                sensor_strength_ladar=max(0.0, float(pyfa_final.get("sensor_strength_ladar", 0.0) or 0.0)),
                sensor_strength_magnetometric=max(0.0, float(pyfa_final.get("sensor_strength_magnetometric", 0.0) or 0.0)),
                sensor_strength_radar=max(0.0, float(pyfa_final.get("sensor_strength_radar", 0.0) or 0.0)),
                warp_scramble_status=float(pyfa_final.get("warp_scramble_status", 0.0) or 0.0),
                warp_stability=float(pyfa_final.get("warp_stability", 0.0) or 0.0),
                energy_warfare_resistance=max(0.0, float(pyfa_final.get("energy_warfare_resistance", 1.0) or 1.0)),
                max_speed=max(1.0, float(pyfa_final.get("max_speed", 0.0) or 0.0)),
                max_cap=max(1.0, float(pyfa_final.get("max_cap", 0.0) or 0.0)),
                cap_recharge_time=max(1.0, float(pyfa_final.get("cap_recharge_time", 0.0) or 0.0)),
                shield_hp=max(1.0, float(pyfa_final.get("shield_hp", 0.0) or 0.0)),
                armor_hp=max(1.0, float(pyfa_final.get("armor_hp", 0.0) or 0.0)),
                structure_hp=max(1.0, float(pyfa_final.get("structure_hp", 0.0) or 0.0)),
                rep_amount=0.0,
                rep_cycle=5.0,
                weapon_system=(
                    "mixed"
                    if turret_dps > 0.0 and missile_dps > 0.0
                    else ("missile" if missile_dps > 0.0 else "turret")
                ),
                turret_dps=turret_dps,
                missile_dps=missile_dps,
                turret_cycle=max(0.0, float(pyfa_final.get("turret_cycle", 0.0) or 0.0)),
                missile_cycle=max(0.0, float(pyfa_final.get("missile_cycle", 0.0) or 0.0)),
                damage_em=max(0.0, float(pyfa_final.get("damage_em", 0.0) or 0.0)),
                damage_thermal=max(0.0, float(pyfa_final.get("damage_thermal", 0.0) or 0.0)),
                damage_kinetic=max(0.0, float(pyfa_final.get("damage_kinetic", 0.0) or 0.0)),
                damage_explosive=max(0.0, float(pyfa_final.get("damage_explosive", 0.0) or 0.0)),
                turret_em_dps=max(0.0, float(pyfa_final.get("turret_em_dps", 0.0) or 0.0)),
                turret_thermal_dps=max(0.0, float(pyfa_final.get("turret_thermal_dps", 0.0) or 0.0)),
                turret_kinetic_dps=max(0.0, float(pyfa_final.get("turret_kinetic_dps", 0.0) or 0.0)),
                turret_explosive_dps=max(0.0, float(pyfa_final.get("turret_explosive_dps", 0.0) or 0.0)),
                missile_em_dps=max(0.0, float(pyfa_final.get("missile_em_dps", 0.0) or 0.0)),
                missile_thermal_dps=max(0.0, float(pyfa_final.get("missile_thermal_dps", 0.0) or 0.0)),
                missile_kinetic_dps=max(0.0, float(pyfa_final.get("missile_kinetic_dps", 0.0) or 0.0)),
                missile_explosive_dps=max(0.0, float(pyfa_final.get("missile_explosive_dps", 0.0) or 0.0)),
                missile_explosion_radius=max(0.0, float(pyfa_final.get("missile_explosion_radius", 0.0) or 0.0)),
                missile_explosion_velocity=max(0.0, float(pyfa_final.get("missile_explosion_velocity", 0.0) or 0.0)),
                missile_max_range=max(0.0, float(pyfa_final.get("missile_max_range", 0.0) or 0.0)),
                missile_damage_reduction_factor=max(0.1, min(2.0, float(pyfa_final.get("missile_damage_reduction_factor", 0.5) or 0.5))),
                shield_resonance_em=max(0.01, min(1.0, float(pyfa_final.get("shield_resonance_em", 1.0) or 1.0))),
                shield_resonance_thermal=max(0.01, min(1.0, float(pyfa_final.get("shield_resonance_thermal", 1.0) or 1.0))),
                shield_resonance_kinetic=max(0.01, min(1.0, float(pyfa_final.get("shield_resonance_kinetic", 1.0) or 1.0))),
                shield_resonance_explosive=max(0.01, min(1.0, float(pyfa_final.get("shield_resonance_explosive", 1.0) or 1.0))),
                armor_resonance_em=max(0.01, min(1.0, float(pyfa_final.get("armor_resonance_em", 1.0) or 1.0))),
                armor_resonance_thermal=max(0.01, min(1.0, float(pyfa_final.get("armor_resonance_thermal", 1.0) or 1.0))),
                armor_resonance_kinetic=max(0.01, min(1.0, float(pyfa_final.get("armor_resonance_kinetic", 1.0) or 1.0))),
                armor_resonance_explosive=max(0.01, min(1.0, float(pyfa_final.get("armor_resonance_explosive", 1.0) or 1.0))),
                structure_resonance_em=max(0.01, min(1.0, float(pyfa_final.get("structure_resonance_em", 1.0) or 1.0))),
                structure_resonance_thermal=max(0.01, min(1.0, float(pyfa_final.get("structure_resonance_thermal", 1.0) or 1.0))),
                structure_resonance_kinetic=max(0.01, min(1.0, float(pyfa_final.get("structure_resonance_kinetic", 1.0) or 1.0))),
                structure_resonance_explosive=max(0.01, min(1.0, float(pyfa_final.get("structure_resonance_explosive", 1.0) or 1.0))),
                mass=max(0.0, float(pyfa_final.get("mass", 0.0) or 0.0)),
                agility=max(0.0, float(pyfa_final.get("agility", 0.0) or 0.0)),
            )

            hull = HullProfile(
                ship_name=ship_name,
                role=role,
                base_dps=profile.dps,
                volley=profile.volley,
                optimal=profile.optimal,
                falloff=profile.falloff,
                tracking=profile.tracking,
                sig_radius=profile.sig_radius,
                scan_resolution=profile.scan_resolution,
                max_target_range=profile.max_target_range,
                max_speed=profile.max_speed,
                cap_max=profile.max_cap,
                cap_recharge_time=profile.cap_recharge_time,
                shield_hp=profile.shield_hp,
                armor_hp=profile.armor_hp,
                structure_hp=profile.structure_hp,
                rep_amount=profile.rep_amount,
                rep_cycle=profile.rep_cycle,
                energy_warfare_resistance=profile.energy_warfare_resistance,
                mass=profile.mass,
                agility=profile.agility,
            )

            runtime = FitRuntime(
                fit_key=parsed.fit_key,
                hull=hull,
                skills=self._skills_default(),
                modules=modules,
            )
            runtime.diagnostics["runtime_controlled_module_ids"] = tuple(
                str(module.module_id)
                for module in modules
                if _module_requires_control_state(module)
            )
            runtime.diagnostics["runtime_local_stateful_module_ids"] = tuple(
                str(module.module_id)
                for module in modules
                if _module_affects_local_pyfa_profile(module)
            )
            runtime.diagnostics["pyfa_blueprint"] = {
                "ship_name": parsed.ship_name,
                "fit_name": parsed.fit_name,
                "modules": pyfa_blueprint_modules,
            }
            runtime.diagnostics["pyfa_command_boosters"] = deepcopy(command_booster_snapshots or [])
            runtime.diagnostics["pyfa_max_state_by_module_id"] = pyfa_max_state_by_module_id
            runtime.diagnostics["pyfa_ship_attribute_penalty_context"] = self._extract_pyfa_ship_attribute_penalty_context(fit_ctx)
            runtime.diagnostics["pyfa_weapon_attribute_penalty_context"] = self._extract_pyfa_weapon_attribute_penalty_context(fit_ctx)
            runtime.diagnostics["motion_params"] = {
                "mass": float(pyfa_final.get("mass", 0.0) or 0.0),
                "agility": float(pyfa_final.get("agility", 0.0) or 0.0),
            }
            fit = FitDescriptor(
                fit_key=parsed.fit_key,
                ship_name=hull.ship_name,
                role=hull.role,
                base_dps=profile.dps,
                volley=profile.volley,
                optimal_range=profile.optimal,
                falloff=profile.falloff,
                tracking=profile.tracking,
                signature_radius=profile.sig_radius,
                scan_resolution=profile.scan_resolution,
                max_target_range=profile.max_target_range,
                sensor_strength_gravimetric=profile.sensor_strength_gravimetric,
                sensor_strength_ladar=profile.sensor_strength_ladar,
                sensor_strength_magnetometric=profile.sensor_strength_magnetometric,
                sensor_strength_radar=profile.sensor_strength_radar,
                max_speed=profile.max_speed,
                max_cap=profile.max_cap,
                cap_recharge_time=profile.cap_recharge_time,
                shield_hp=profile.shield_hp,
                armor_hp=profile.armor_hp,
                structure_hp=profile.structure_hp,
                rep_amount=profile.rep_amount,
                rep_cycle=profile.rep_cycle,
                energy_warfare_resistance=profile.energy_warfare_resistance,
                mass=profile.mass,
                agility=profile.agility,
            )
            return runtime, fit, profile
        finally:
            if temporary_weapon_states:
                for fitted_module, original_state in temporary_weapon_states:
                    fitted_module.state = original_state

    def build(self, parsed: ParsedEftFit) -> tuple[FitRuntime, FitDescriptor]:
        if not self._pyfa.available:
            raise ValueError("pyfa static database unavailable")

        cached_runtime = self._runtime_cache.get(parsed.fit_key)
        cached_fit = self._fit_cache.get(parsed.fit_key)
        if cached_runtime is not None and cached_fit is not None:
            return cached_runtime, cached_fit

        fit_ctx, fitted_modules = self._build_pyfa_fit(parsed)
        runtime, fit, profile = self._build_runtime_artifacts_from_pyfa_fit(parsed, fit_ctx, fitted_modules)
        runtime_parsed = _parsed_fit_from_runtime_blueprint(runtime)
        _warm_runtime_precalculated_local_base_fit(self, runtime_parsed or parsed, runtime)
        self._runtime_cache[parsed.fit_key] = runtime
        self._fit_cache[parsed.fit_key] = fit
        self._profile_cache[parsed.fit_key] = profile
        return runtime, fit

    def build_profile(self, parsed: ParsedEftFit) -> ShipProfile:
        self.build(parsed)
        cached = self._profile_cache.get(parsed.fit_key)
        if cached is None:
            raise ValueError("missing cached fit profile")
        return replace(cached)

    @property
    def backend_status(self) -> str:
        return self._pyfa.status


class _PyfaStaticBackend:
    def __init__(self) -> None:
        self.available = False
        self.status = "Fallback: 鍐呯疆瑙勫垯鏄犲皠"
        self._get_item = None
        self._get_group = None
        self._fit_cls = None
        self._ship_cls = None
        self._module_cls = None
        self._character_cls = None
        self._fitting_module_state_active = None
        self._fitting_module_state_online = None
        self._fitting_module_state_offline = None
        self._fitting_module_state_overheated = None
        self._name_cache: dict[tuple[str, str], str] = {}
        self._resolve_cache: dict[str, str] = {}
        self._has_type_name_zh = False
        self._db_path = resolve_pyfa_source_dir() / "eve.db"
        self._init_db_meta()
        self._init_backend()

    def _init_db_meta(self) -> None:
        if not self._db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self._db_path))
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(invtypes)")
            cols = {str(r[1]).lower() for r in cur.fetchall()}
            conn.close()
            self._has_type_name_zh = "typename_zh" in cols
        except Exception:
            self._has_type_name_zh = False

    def _init_backend(self) -> None:
        pyfa_root = resolve_pyfa_source_dir()
        db_path = pyfa_root / "eve.db"
        if not self._check_db_schema(db_path):
            self.status = f"Fallback: 鏈娴嬪埌鍙敤 pyfa 闈欐€佸簱 ({db_path})"
            return

        try:
            if not hasattr(sys, "_called_from_test"):
                setattr(sys, "_called_from_test", True)
            if "wx" not in sys.modules:
                wx = types.ModuleType("wx")
                setattr(wx, "Colour", lambda *args, **kwargs: tuple(args))
                sys.modules["wx"] = wx

            src = str(pyfa_root)
            if src not in sys.path:
                sys.path.insert(0, src)

            eos_config = importlib.import_module("eos.config")

            setattr(eos_config, "gamedata_connectionstring", f"sqlite:///{db_path.as_posix()}")
            eos_db = importlib.import_module("eos.db")
            fit_mod = importlib.import_module("eos.saveddata.fit")
            ship_mod = importlib.import_module("eos.saveddata.ship")
            module_mod = importlib.import_module("eos.saveddata.module")
            char_mod = importlib.import_module("eos.saveddata.character")
            const_mod = importlib.import_module("eos.const")

            self._get_item = eos_db.getItem
            self._get_group = eos_db.getGroup
            self._fit_cls = fit_mod.Fit
            self._ship_cls = ship_mod.Ship
            self._module_cls = module_mod.Module
            self._character_cls = char_mod.Character
            self._fitting_module_state_active = const_mod.FittingModuleState.ACTIVE
            self._fitting_module_state_online = const_mod.FittingModuleState.ONLINE
            self._fitting_module_state_offline = const_mod.FittingModuleState.OFFLINE
            self._fitting_module_state_overheated = const_mod.FittingModuleState.OVERHEATED
            self.available = True
            self.status = "Pyfa: static data backend ready"
        except ModuleNotFoundError as exc:
            self.available = False
            missing = getattr(exc, "name", None) or str(exc)
            self.status = f"Fallback: pyfa鍔犺浇澶辫触 (ModuleNotFoundError: {missing})"
        except Exception as exc:
            self.available = False
            self.status = f"Fallback: pyfa鍔犺浇澶辫触 ({type(exc).__name__})"

    @staticmethod
    def _check_db_schema(db_path: Path) -> bool:
        if not db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='invtypes'")
            found = cur.fetchone() is not None
            conn.close()
            return found
        except Exception:
            return False

    def get_item(self, type_name: str):
        if not self.available or self._get_item is None:
            return None
        try:
            return self._get_item(type_name)
        except Exception:
            return None

    @property
    def fit_engine_ready(self) -> bool:
        return all(
            x is not None
            for x in (
                self._fit_cls,
                self._ship_cls,
                self._module_cls,
                self._character_cls,
                self._fitting_module_state_active,
                self._fitting_module_state_online,
                self._fitting_module_state_offline,
            )
        )

    def list_charge_options_for_module(self, module_name: str) -> list[str]:
        module = self.get_item(module_name)
        if module is None:
            return []

        if self._module_cls is not None:
            try:
                module_obj = self._module_cls(module)
                charges = module_obj.getValidCharges()
                names = sorted(
                    {
                        str(charge.typeName)
                        for charge in charges
                        if charge is not None and getattr(charge, "typeName", None)
                    }
                )
                return names
            except Exception:
                pass

        if self._get_group is None:
            return []

        module_size_attr = module.getAttribute("chargeSize", None)
        module_size = 0
        if module_size_attr is not None:
            try:
                module_size = int(float(module_size_attr))
            except Exception:
                module_size = 0

        module_capacity_attr = module.getAttribute("capacity", None)
        module_capacity: float | None = None
        if module_capacity_attr is not None:
            try:
                module_capacity = float(module_capacity_attr)
            except Exception:
                module_capacity = None

        group_ids: list[int] = []
        for i in range(0, 5):
            value = module.getAttribute(f"chargeGroup{i}", None)
            if value is None:
                continue
            try:
                gid = int(float(value))
            except Exception:
                continue
            if gid > 0 and gid not in group_ids:
                group_ids.append(gid)

        ammo_names: list[str] = []
        for gid in group_ids:
            try:
                group = self._get_group(gid)
                if group is None:
                    continue
                for item in group.items:
                    if not bool(getattr(item, "published", True)):
                        continue

                    charge_volume_attr = item.getAttribute("volume", None)
                    if module_capacity is not None and charge_volume_attr is not None:
                        try:
                            if float(charge_volume_attr) > module_capacity:
                                continue
                        except Exception:
                            pass

                    charge_size_attr = item.getAttribute("chargeSize", None)
                    charge_size = 0
                    if charge_size_attr is not None:
                        try:
                            charge_size = int(float(charge_size_attr))
                        except Exception:
                            charge_size = 0
                    if module_size > 0 and charge_size > 0 and charge_size != module_size:
                        continue
                    ammo_names.append(item.typeName)
            except Exception:
                continue
        return sorted(set(ammo_names))

    def module_reload_time_sec(self, module_name: str) -> float:
        module = self.get_item(module_name)
        if module is None:
            return 0.0
        reload_ms = module.getAttribute("reloadTime", None)
        if reload_ms is None:
            return 0.0
        return max(0.0, float(reload_ms) / 1000.0)

    def module_reload_channel(self, module_name: str) -> str:
        module = self.get_item(module_name)
        if module is None:
            return "none"
        group_name = (module.group.name or "").lower()
        if "launcher" in group_name:
            return "launcher"
        if ("weapon" in group_name) or ("turret" in group_name):
            return "turret"
        return "none"

    def is_charge_loadable_module(self, module_name: str) -> bool:
        canonical = self.resolve_type_name(module_name)
        item = self.get_item(canonical)
        if item is None:
            return False
        for i in range(0, 5):
            value = item.getAttribute(f"chargeGroup{i}", None)
            if value is None:
                continue
            try:
                if int(float(value)) > 0:
                    return True
            except Exception:
                continue
        return False

    def resolve_type_name(self, type_name: str) -> str:
        name = (type_name or "").strip()
        if not name:
            return type_name
        cached = self._resolve_cache.get(name.lower())
        if cached is not None:
            return cached

        resolved = name
        if self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                cur = conn.cursor()
                if self._has_type_name_zh:
                    cur.execute(
                        "SELECT typeName FROM invtypes "
                        "WHERE LOWER(typeName)=LOWER(?) "
                        "OR LOWER(typeName_zh)=LOWER(?) "
                        "OR LOWER(REPLACE(typeName,' ',''))=LOWER(REPLACE(?, ' ', '')) "
                        "OR LOWER(REPLACE(typeName_zh,' ',''))=LOWER(REPLACE(?, ' ', '')) "
                        "LIMIT 1",
                        (name, name, name, name),
                    )
                else:
                    cur.execute(
                        "SELECT typeName FROM invtypes "
                        "WHERE LOWER(typeName)=LOWER(?) "
                        "OR LOWER(REPLACE(typeName,' ',''))=LOWER(REPLACE(?, ' ', '')) "
                        "LIMIT 1",
                        (name, name),
                    )
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    resolved = str(row[0])
            except Exception:
                resolved = name

        self._resolve_cache[name.lower()] = resolved
        return resolved

    def localize_type_name(self, type_name: str, language: str = "en") -> str:
        canonical = self.resolve_type_name(type_name)
        name = (canonical or "").strip()
        if not name:
            return type_name
        key = (name.lower(), language.lower())
        cached = self._name_cache.get(key)
        if cached is not None:
            return cached

        resolved = name
        if self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                cur = conn.cursor()
                if language.lower().startswith("zh") and self._has_type_name_zh:
                    cur.execute("SELECT typeName_zh, typeName FROM invtypes WHERE LOWER(typeName)=LOWER(?) LIMIT 1", (name,))
                    row = cur.fetchone()
                    if row:
                        zh_name = str(row[0]).strip() if row[0] is not None else ""
                        en_name = str(row[1]).strip() if len(row) > 1 and row[1] is not None else name
                        resolved = zh_name or en_name
                else:
                    cur.execute("SELECT typeName FROM invtypes WHERE LOWER(typeName)=LOWER(?) LIMIT 1", (name,))
                    row = cur.fetchone()
                    if row and row[0]:
                        resolved = str(row[0])
                conn.close()
            except Exception:
                resolved = name

        self._name_cache[key] = resolved
        return resolved


_STATIC_BACKEND: _PyfaStaticBackend | None = None
_PYFA_RUNTIME_RESOLVED_CACHE: dict[tuple[Any, ...], tuple[FitRuntime, ShipProfile]] = {}
_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE: dict[tuple[Any, ...], tuple[Any, tuple[str | None, ...]]] = {}
_PYFA_PROJECTION_RANGE_BUCKET_M = 100.0
_PYFA_PROJECTION_SIGNATURE_ROUND_DIGITS = 6


def _get_static_backend() -> _PyfaStaticBackend:
    global _STATIC_BACKEND
    if _STATIC_BACKEND is None:
        _STATIC_BACKEND = _PyfaStaticBackend()
    return _STATIC_BACKEND


def _rollback_pyfa_saveddata_session() -> None:
    try:
        eos_db = importlib.import_module("eos.db")
    except Exception:
        return
    rollback = getattr(eos_db, "rollback", None)
    if not callable(rollback):
        return
    try:
        rollback()
    except Exception:
        pass


def get_fit_backend_status() -> str:
    return RuntimeFromEftFactory().backend_status


def _runtime_blueprint_signature(blueprint: dict[str, Any]) -> tuple[Any, ...]:
    ship_name = str(blueprint.get("ship_name", "") or "").strip()
    modules = blueprint.get("modules")
    if not ship_name or not isinstance(modules, list):
        return tuple()
    return tuple(
        [
            ship_name,
            tuple(
                sorted(
                    (
                        str(raw.get("module_id", "") or ""),
                        str(raw.get("module_name", "") or ""),
                        str(raw.get("charge_name", "") or ""),
                        bool(raw.get("offline", False)),
                    )
                    for raw in modules
                    if isinstance(raw, dict)
                )
            )
        ]
    )


def _module_state_signature(state_by_module_id: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    state_map = state_by_module_id or {}
    return tuple(sorted((str(module_id), str(state)) for module_id, state in state_map.items()))


def _repair_pyfa_fit_owner_refs(fit: Any) -> Any:
    ship = getattr(fit, "ship", None)
    if ship is not None:
        try:
            ship.owner = fit
        except Exception:
            pass

    for collection_name in (
        "modules",
        "projectedModules",
        "drones",
        "projectedDrones",
        "fighters",
        "projectedFighters",
        "implants",
        "boosters",
    ):
        collection = getattr(fit, collection_name, None)
        if collection is None:
            continue
        for item in collection:
            if item is None:
                continue
            try:
                item.owner = fit
            except Exception:
                continue
    return fit


def _copy_fitted_modules_from_template(
    parsed: ParsedEftFit,
    fit: Any,
    charge_names: tuple[str | None, ...],
) -> list[tuple[ParsedModuleSpec, Any, str | None]]:
    fit_modules = list(cast(list[Any], fit.modules))
    if len(parsed.module_specs) != len(fit_modules) or len(parsed.module_specs) != len(charge_names):
        raise ValueError("pyfa fit template/module mismatch")
    return [
        (spec, fitted_module, charge_name)
        for spec, fitted_module, charge_name in zip(parsed.module_specs, fit_modules, charge_names, strict=True)
    ]


def _parsed_neutral_fit_template_signature(parsed: ParsedEftFit) -> tuple[Any, ...]:
    return (
        str(parsed.ship_name or ""),
        tuple(
            (
                str(spec.module_name or ""),
                str(spec.charge_name or ""),
            )
            for spec in parsed.module_specs
        ),
    )


def _neutral_online_state_map(parsed: ParsedEftFit) -> dict[str, str]:
    return {
        f"mod-{idx}": "ONLINE"
        for idx, _spec in enumerate(parsed.module_specs, start=1)
    }


def _pyfa_local_base_fit_cache_key(
    parsed: ParsedEftFit,
    state_by_module_id: dict[str, str] | None,
) -> tuple[Any, ...]:
    del state_by_module_id
    return _parsed_neutral_fit_template_signature(parsed)


def _store_precalculated_local_base_fit(
    parsed: ParsedEftFit,
    state_by_module_id: dict[str, str] | None,
    fit: Any,
    fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]],
) -> None:
    charge_names = tuple(charge_name for _spec, _fitted_module, charge_name in fitted_modules)
    _PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE[_pyfa_local_base_fit_cache_key(parsed, state_by_module_id)] = (fit, charge_names)


def _get_precalculated_local_base_fit(
    parsed: ParsedEftFit,
    state_by_module_id: dict[str, str] | None,
) -> tuple[Any, tuple[str | None, ...]] | None:
    return _PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE.get(_pyfa_local_base_fit_cache_key(parsed, state_by_module_id))


def _ensure_precalculated_local_base_fit(
    factory: RuntimeFromEftFactory,
    parsed: ParsedEftFit,
    state_by_module_id: dict[str, str] | None,
) -> tuple[Any, tuple[str | None, ...]]:
    del state_by_module_id
    cached = _get_precalculated_local_base_fit(parsed, None)
    if cached is not None:
        return cached

    base_fit, fitted_modules = factory._build_pyfa_fit(
        parsed,
        state_by_module_id=_neutral_online_state_map(parsed),
        calculate_modified_attributes=True,
        default_online=True,
        ignore_offline_specs=True,
    )
    _store_precalculated_local_base_fit(parsed, None, base_fit, fitted_modules)
    cached = _get_precalculated_local_base_fit(parsed, None)
    if cached is None:
        raise ValueError("pyfa鏈湴鍩虹Fit缂撳瓨鏋勫缓澶辫触")
    return cached


def _pyfa_module_state_from_runtime_state(
    factory: RuntimeFromEftFactory,
    fitted_module: Any,
    runtime_state: str,
) -> Any:
    state_name = str(runtime_state or "ACTIVE").upper()
    offline_state = factory._pyfa._fitting_module_state_offline
    online_state = factory._pyfa._fitting_module_state_online
    active_state = factory._pyfa._fitting_module_state_active
    overheated_state = factory._pyfa._fitting_module_state_overheated or active_state
    def _apply_context_limits(candidate: Any) -> Any:
        if candidate is None:
            return None
        try:
            can_have_state = getattr(fitted_module, "canHaveState", None)
            if callable(can_have_state):
                original_state = getattr(fitted_module, "state", None)
                try:
                    fitted_module.state = candidate
                    constrained_state = can_have_state(candidate)
                finally:
                    if original_state is not None:
                        fitted_module.state = original_state
                if constrained_state is not True and constrained_state is not None:
                    return constrained_state
        except Exception:
            pass
        return candidate
    if state_name == "OFFLINE":
        requested_state = offline_state
        candidates = [offline_state]
    elif state_name == "ONLINE":
        requested_state = online_state
        candidates = [online_state, offline_state]
    elif state_name == "OVERHEATED":
        requested_state = overheated_state
        candidates = [overheated_state, active_state, online_state, offline_state]
    else:
        requested_state = active_state
        candidates = [active_state, online_state, offline_state]

    if requested_state is not None:
        try:
            if not hasattr(fitted_module, "isValidState") or fitted_module.isValidState(requested_state):
                return _apply_context_limits(requested_state)
        except Exception:
            return _apply_context_limits(requested_state)
        try:
            get_max_state = getattr(fitted_module, "getMaxState", None)
            if callable(get_max_state):
                downgraded_state = get_max_state(proposedState=requested_state)
                if downgraded_state is not None:
                    return _apply_context_limits(downgraded_state)
        except Exception:
            pass

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if not hasattr(fitted_module, "isValidState") or fitted_module.isValidState(candidate):
                return _apply_context_limits(candidate)
        except Exception:
            return _apply_context_limits(candidate)
    return offline_state if offline_state is not None else online_state


def _normalize_pyfa_local_module_states(
    factory: RuntimeFromEftFactory,
    fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]],
) -> bool:
    if not fitted_modules:
        return False
    changed = False
    online_state = factory._pyfa._fitting_module_state_online
    offline_state = factory._pyfa._fitting_module_state_offline
    for _spec, fitted_module, _charge_name in fitted_modules:
        current_state = getattr(fitted_module, "state", None)
        if current_state is None:
            continue
        try:
            can_have_state = fitted_module.canHaveState(current_state)
        except Exception:
            can_have_state = True
        if can_have_state is not True and can_have_state is not None:
            if fitted_module.state != can_have_state:
                changed = True
            fitted_module.state = can_have_state
            continue
        try:
            is_valid = not hasattr(fitted_module, "isValidState") or fitted_module.isValidState(fitted_module.state)
        except Exception:
            is_valid = True
        if is_valid:
            continue
        try:
            get_max_state = getattr(fitted_module, "getMaxState", None)
            downgraded_state = get_max_state(proposedState=fitted_module.state) if callable(get_max_state) else None
        except Exception:
            downgraded_state = None
        if downgraded_state is None:
            downgraded_state = online_state if online_state is not None else offline_state
        if downgraded_state is not None and fitted_module.state != downgraded_state:
            changed = True
            fitted_module.state = downgraded_state
    return changed


def _calculate_pyfa_fit_with_state_normalization(
    factory: RuntimeFromEftFactory,
    fit: Any,
    fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]],
) -> None:
    for _ in range(3):
        fit.calculated = False
        fit.calculateModifiedAttributes()
        if not _normalize_pyfa_local_module_states(factory, fitted_modules):
            break


def _runtime_state_from_pyfa_module_state(
    factory: RuntimeFromEftFactory,
    fitted_module: Any,
    fallback: ModuleState = ModuleState.ONLINE,
) -> ModuleState:
    current_state = getattr(fitted_module, "state", None)
    return _runtime_state_from_pyfa_state_value(factory, current_state, fallback)


def _runtime_state_from_pyfa_state_value(
    factory: RuntimeFromEftFactory,
    current_state: Any,
    fallback: ModuleState = ModuleState.ONLINE,
) -> ModuleState:
    overheated_state = factory._pyfa._fitting_module_state_overheated
    if overheated_state is not None and current_state == overheated_state:
        return ModuleState.OVERHEATED
    if current_state == factory._pyfa._fitting_module_state_active:
        return ModuleState.ACTIVE
    if current_state == factory._pyfa._fitting_module_state_online:
        return ModuleState.ONLINE
    if current_state == factory._pyfa._fitting_module_state_offline:
        return ModuleState.OFFLINE
    return fallback


def _apply_pyfa_local_state_map(
    factory: RuntimeFromEftFactory,
    fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]],
    state_by_module_id: dict[str, str] | None,
) -> None:
    state_map = state_by_module_id or {}
    for idx, (_spec, fitted_module, _charge_name) in enumerate(fitted_modules, start=1):
        module_id = f"mod-{idx}"
        if module_id not in state_map:
            continue
        fitted_module.state = _pyfa_module_state_from_runtime_state(factory, fitted_module, state_map[module_id])
    _normalize_pyfa_local_module_states(factory, fitted_modules)


def _copy_precalculated_neutral_base_fit(
    factory: RuntimeFromEftFactory,
    parsed: ParsedEftFit,
) -> tuple[Any, list[tuple[ParsedModuleSpec, Any, str | None]]]:
    base_fit, charge_names = _ensure_precalculated_local_base_fit(factory, parsed, None)
    fit_copy = _repair_pyfa_fit_owner_refs(deepcopy(base_fit))
    return fit_copy, _copy_fitted_modules_from_template(parsed, fit_copy, charge_names)


def _module_has_tag(module: ModuleRuntime, tag: str) -> bool:
    return str(tag) in tuple(str(value) for value in getattr(module, "tags", ()) or ())


def _module_requires_control_state(module: ModuleRuntime) -> bool:
    return _module_has_tag(module, "controlled") or module.can_be_active()


def _module_is_command_burst_module(module: ModuleRuntime) -> bool:
    return _module_has_tag(module, "command_burst")




def _warm_runtime_precalculated_local_base_fit(
    factory: RuntimeFromEftFactory,
    parsed: ParsedEftFit,
    runtime: FitRuntime,
) -> None:
    del runtime
    if _get_precalculated_local_base_fit(parsed, None) is not None:
        return
    base_fit, fitted_modules = factory._build_pyfa_fit(
        parsed,
        state_by_module_id=_neutral_online_state_map(parsed),
        calculate_modified_attributes=True,
        default_online=True,
        ignore_offline_specs=True,
    )
    _store_precalculated_local_base_fit(parsed, None, base_fit, fitted_modules)


def prewarm_runtime_base_cache(runtime: FitRuntime | None) -> bool:
    if runtime is None:
        return False
    parsed = _parsed_fit_from_runtime_blueprint(runtime)
    if parsed is None:
        return False
    try:
        _ensure_precalculated_local_base_fit(RuntimeFromEftFactory(), parsed, None)
    except Exception:
        return False
    return True


def prewarm_world_base_cache(world: WorldState) -> int:
    warmed = 0
    seen_keys: set[tuple[Any, ...]] = set()
    factory = RuntimeFromEftFactory()
    for ship in world.ships.values():
        runtime = getattr(ship, "runtime", None)
        if runtime is None:
            continue
        parsed = _parsed_fit_from_runtime_blueprint(runtime)
        if parsed is None:
            continue
        cache_key = _pyfa_local_base_fit_cache_key(parsed, None)
        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)
        try:
            _ensure_precalculated_local_base_fit(factory, parsed, None)
        except Exception:
            continue
        warmed += 1
    return warmed


def _runtime_module_state_map(runtime: FitRuntime) -> dict[str, str]:
    return {
        str(module.module_id): str(module.normalized_state().value)
        for module in runtime.modules
    }


def _runtime_state_rank(state: ModuleState) -> int:
    return {
        ModuleState.OFFLINE: 0,
        ModuleState.ONLINE: 1,
        ModuleState.ACTIVE: 2,
        ModuleState.OVERHEATED: 3,
    }.get(state, 0)


def _clamp_runtime_state_to_pyfa_max(
    requested_state: ModuleState,
    max_state: ModuleState,
) -> ModuleState:
    return requested_state if _runtime_state_rank(requested_state) <= _runtime_state_rank(max_state) else max_state


def _module_affects_local_pyfa_profile(module: ModuleRuntime) -> bool:
    if _module_has_tag(module, "affects_local_pyfa_profile"):
        return True
    return any(
        effect.effect_class == EffectClass.LOCAL and effect.state_required == ModuleState.ACTIVE
        for effect in module.effects
    )


def _runtime_local_profile_state_map(runtime: FitRuntime) -> dict[str, str]:
    cached_ids = runtime.diagnostics.get("runtime_local_stateful_module_ids")
    if isinstance(cached_ids, tuple):
        tracked_ids = {str(module_id) for module_id in cached_ids}
        return {
            str(module.module_id): str(module.normalized_state().value)
            for module in runtime.modules
            if str(module.module_id) in tracked_ids
        }
    return {
        str(module.module_id): str(module.normalized_state().value)
        for module in runtime.modules
        if _module_affects_local_pyfa_profile(module)
    }


def _runtime_local_profile_state_signature(runtime: FitRuntime) -> tuple[tuple[str, str], ...]:
    return _module_state_signature(_runtime_local_profile_state_map(runtime))


def _parsed_fit_from_runtime_blueprint(runtime: FitRuntime) -> ParsedEftFit | None:
    blueprint = runtime.diagnostics.get("pyfa_blueprint")
    if not isinstance(blueprint, dict):
        return None
    ship_name = str(blueprint.get("ship_name", "") or "").strip()
    fit_name = str(blueprint.get("fit_name", runtime.fit_key) or runtime.fit_key).strip()
    raw_modules = blueprint.get("modules")
    if not ship_name or not isinstance(raw_modules, list):
        return None

    module_specs: list[ParsedModuleSpec] = []
    module_names: list[str] = []
    for raw in raw_modules:
        if not isinstance(raw, dict):
            continue
        module_name = str(raw.get("module_name", "") or "").strip()
        if not module_name:
            continue
        charge_name = str(raw.get("charge_name", "") or "").strip() or None
        offline = bool(raw.get("offline", False))
        module_specs.append(ParsedModuleSpec(module_name=module_name, charge_name=charge_name, offline=offline))
        module_names.append(module_name)

    return ParsedEftFit(
        ship_name=ship_name,
        fit_name=fit_name,
        module_names=module_names,
        module_specs=module_specs,
        cargo_item_names=[],
        fit_key=runtime.fit_key,
    )


def _command_snapshot_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    blueprint_raw = snapshot.get("blueprint")
    blueprint: dict[str, Any] = blueprint_raw if isinstance(blueprint_raw, dict) else {}
    state_raw = snapshot.get("state_by_module_id")
    state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
    return (
        _runtime_blueprint_signature(blueprint),
        _module_state_signature({str(module_id): str(state) for module_id, state in state_by_module_id.items()}),
    )


def _normalized_command_booster_snapshots(
    runtime: FitRuntime,
    command_booster_snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    snapshots = command_booster_snapshots
    if snapshots is None:
        raw_snapshots = runtime.diagnostics.get("pyfa_command_boosters")
        snapshots = [snap for snap in raw_snapshots if isinstance(snap, dict)] if isinstance(raw_snapshots, list) else []
    return [snap for snap in snapshots if isinstance(snap, dict)]


def _quantize_pyfa_projection_range(distance: float) -> float:
    safe_distance = max(0.0, float(distance or 0.0))
    if _PYFA_PROJECTION_RANGE_BUCKET_M <= 0.0:
        return safe_distance
    return math.floor(safe_distance / _PYFA_PROJECTION_RANGE_BUCKET_M) * _PYFA_PROJECTION_RANGE_BUCKET_M


def _normalized_snapshot_projection_signature(snapshot: dict[str, Any]) -> tuple[str, Any]:
    projection_key_mode = str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range")
    if projection_key_mode == "exact_range":
        try:
            distance_signature: Any = round(
                _quantize_pyfa_projection_range(
                    float(snapshot.get("pyfa_projection_range", snapshot.get("projection_range", 0.0)) or 0.0)
                ),
                3,
            )
        except Exception:
            distance_signature = 0.0
        return "exact_range", distance_signature
    return "in_range", None


def _normalized_projection_effect_signature(raw: Any) -> tuple[Any, ...] | None:
    if not isinstance(raw, (tuple, list)):
        return None
    try:
        return tuple(raw)
    except Exception:
        _rollback_pyfa_saveddata_session()
        return None


def _projected_snapshot_module_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    direct_signature = _normalized_projection_effect_signature(snapshot.get("pyfa_projection_module_signature"))
    if direct_signature is not None:
        return direct_signature

    blueprint_raw = snapshot.get("blueprint")
    blueprint: dict[str, Any] = blueprint_raw if isinstance(blueprint_raw, dict) else {}
    state_raw = snapshot.get("state_by_module_id")
    state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
    command_raw = snapshot.get("command_booster_snapshots")
    command_snapshots = [snap for snap in command_raw if isinstance(snap, dict)] if isinstance(command_raw, list) else []
    return (
        "legacy_source",
        _runtime_blueprint_signature(blueprint),
        _module_state_signature({str(module_id): str(state) for module_id, state in state_by_module_id.items()}),
        tuple(sorted(_command_snapshot_signature(command_snapshot) for command_snapshot in command_snapshots)),
    )


def _projected_snapshot_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    projection_key_mode, distance_signature = _normalized_snapshot_projection_signature(snapshot)
    return (
        _projected_snapshot_module_signature(snapshot),
        projection_key_mode,
        distance_signature,
    )


def _normalized_projected_source_snapshots(
    runtime: FitRuntime,
    projected_source_snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    snapshots = projected_source_snapshots
    if snapshots is None:
        raw_snapshots = runtime.diagnostics.get("pyfa_projected_sources")
        snapshots = [snap for snap in raw_snapshots if isinstance(snap, dict)] if isinstance(raw_snapshots, list) else []
    return [snap for snap in snapshots if isinstance(snap, dict)]


def get_runtime_resolve_cache_key(
    runtime: FitRuntime,
    command_booster_snapshots: list[dict[str, Any]] | None = None,
    projected_source_snapshots: list[dict[str, Any]] | None = None,
) -> tuple[Any, ...] | None:
    blueprint = runtime.diagnostics.get("pyfa_blueprint")
    if not isinstance(blueprint, dict):
        return None

    blueprint_signature = _runtime_blueprint_signature(blueprint)
    if not blueprint_signature:
        return None

    snapshots = _normalized_command_booster_snapshots(runtime, command_booster_snapshots)
    projected_snapshots = _normalized_projected_source_snapshots(runtime, projected_source_snapshots)
    return (
        blueprint_signature,
        _runtime_local_profile_state_signature(runtime),
        tuple(sorted(_command_snapshot_signature(snapshot) for snapshot in snapshots)),
        tuple(sorted(_projected_snapshot_signature(snapshot) for snapshot in projected_snapshots)),
    )


def _copy_dynamic_runtime_state(source: FitRuntime, target: FitRuntime) -> None:
    raw_max_state_map = target.diagnostics.get("pyfa_max_state_by_module_id")
    max_state_map = raw_max_state_map if isinstance(raw_max_state_map, dict) else {}
    source_by_id = {module.module_id: module for module in source.modules}
    for module in target.modules:
        src = source_by_id.get(module.module_id)
        if src is None:
            continue
        requested_state = module.normalized_state(src.state)
        max_state_name = str(max_state_map.get(str(module.module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
        max_state = ModuleState[max_state_name] if max_state_name in ModuleState.__members__ else ModuleState.OVERHEATED
        module.state = _clamp_runtime_state_to_pyfa_max(requested_state, max_state)
        if module.charge_capacity > 0:
            module.charge_remaining = max(0.0, min(float(src.charge_remaining), float(module.charge_capacity)))


def _attach_command_fit(target_fit: Any, booster_fit: Any) -> None:
    command_fit_mod = importlib.import_module("eos.db.saveddata.fit")
    command_fit_cls = getattr(command_fit_mod, "CommandFit")
    command_link = command_fit_cls(booster_fit.ID, booster_fit, active=True)
    command_link.boostedID = target_fit.ID
    command_link.boosted_fit = target_fit
    booster_fit.boostedOnto[target_fit.ID] = command_link
    target_fit.boostedOf[booster_fit.ID] = command_link


def _attach_projected_fit(
    target_fit: Any,
    source_fit: Any,
    amount: int = 1,
    active: bool = True,
    projection_range: float | None = None,
) -> None:
    projected_fit_mod = importlib.import_module("eos.db.saveddata.fit")
    projected_fit_cls = getattr(projected_fit_mod, "ProjectedFit")
    projected_link = projected_fit_cls(source_fit.ID, source_fit, amount=amount, active=active)
    projected_link.victimID = target_fit.ID
    projected_link.victim_fit = target_fit
    projected_link.projectionRange = None if projection_range is None else float(projection_range)
    source_fit.projectedOnto[target_fit.ID] = projected_link
    target_fit.victimOf[source_fit.ID] = projected_link


def _snapshot_state_by_module_id(snapshot: dict[str, Any]) -> dict[str, str]:
    state_raw = snapshot.get("state_by_module_id")
    state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
    return {str(k): str(v) for k, v in state_by_module_id.items()}


def _snapshot_has_active_modules(state_by_module_id: dict[str, str]) -> bool:
    return any(str(state).upper() in {"ACTIVE", "OVERHEATED"} for state in state_by_module_id.values())


def _snapshot_command_booster_snapshots(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    command_raw = snapshot.get("command_booster_snapshots")
    return [snap for snap in command_raw if isinstance(snap, dict)] if isinstance(command_raw, list) else []


def _snapshot_projection_range(snapshot: dict[str, Any]) -> float:
    projection_key_mode, normalized_distance = _normalized_snapshot_projection_signature(snapshot)
    if projection_key_mode != "exact_range":
        return 0.0
    try:
        return float(normalized_distance or 0.0)
    except Exception:
        return 0.0


def _build_transient_fit_from_snapshot(
    factory: RuntimeFromEftFactory,
    snapshot: dict[str, Any],
    next_fit_id: int,
    fallback_runtime: FitRuntime,
    fit_prefix: str,
) -> tuple[Any | None, int]:
    snapshot_blueprint = snapshot.get("blueprint") if isinstance(snapshot.get("blueprint"), dict) else None
    if snapshot_blueprint is None:
        return None, next_fit_id

    snapshot_runtime = FitRuntime(
        fit_key=str(snapshot.get("fit_key", "") or f"{fit_prefix}-{next_fit_id}"),
        hull=fallback_runtime.hull,
        skills=fallback_runtime.skills,
    )
    snapshot_runtime.diagnostics["pyfa_blueprint"] = snapshot_blueprint
    snapshot_parsed = _parsed_fit_from_runtime_blueprint(snapshot_runtime)
    if snapshot_parsed is None:
        return None, next_fit_id

    state_by_module_id = _snapshot_state_by_module_id(snapshot)
    if not _snapshot_has_active_modules(state_by_module_id):
        return None, next_fit_id

    snapshot_fit, fitted_modules = _copy_precalculated_neutral_base_fit(factory, snapshot_parsed)
    _apply_pyfa_local_state_map(factory, fitted_modules, state_by_module_id)
    snapshot_fit.calculated = False
    snapshot_fit.ID = next_fit_id
    return snapshot_fit, next_fit_id + 1


def _attach_command_snapshot_fits(
    factory: RuntimeFromEftFactory,
    target_fit: Any,
    snapshots: list[dict[str, Any]],
    next_fit_id: int,
    fallback_runtime: FitRuntime,
) -> int:
    for snapshot in snapshots:
        booster_fit, next_fit_id = _build_transient_fit_from_snapshot(
            factory=factory,
            snapshot=snapshot,
            next_fit_id=next_fit_id,
            fallback_runtime=fallback_runtime,
            fit_prefix="command",
        )
        if booster_fit is None:
            continue
        _attach_command_fit(target_fit, booster_fit)
    return next_fit_id


def _attach_projected_snapshot_fits(
    factory: RuntimeFromEftFactory,
    target_fit: Any,
    snapshots: list[dict[str, Any]],
    next_fit_id: int,
    fallback_runtime: FitRuntime,
) -> int:
    for snapshot in snapshots:
        source_fit, next_fit_id = _build_transient_fit_from_snapshot(
            factory=factory,
            snapshot=snapshot,
            next_fit_id=next_fit_id,
            fallback_runtime=fallback_runtime,
            fit_prefix="projected",
        )
        if source_fit is None:
            continue

        source_command_snapshots = _snapshot_command_booster_snapshots(snapshot)
        next_fit_id = _attach_command_snapshot_fits(factory, source_fit, source_command_snapshots, next_fit_id, fallback_runtime)
        _attach_projected_fit(target_fit, source_fit, projection_range=_snapshot_projection_range(snapshot))
    return next_fit_id


def resolve_runtime_from_pyfa_runtime(
    runtime: FitRuntime,
    command_booster_snapshots: list[dict[str, Any]] | None = None,
    projected_source_snapshots: list[dict[str, Any]] | None = None,
) -> tuple[FitRuntime, ShipProfile] | None:
    backend = _get_static_backend()
    if not backend.fit_engine_ready:
        return None
    parsed = _parsed_fit_from_runtime_blueprint(runtime)
    blueprint = runtime.diagnostics.get("pyfa_blueprint")
    if parsed is None or not isinstance(blueprint, dict):
        return None

    snapshots = _normalized_command_booster_snapshots(runtime, command_booster_snapshots)
    projected_snapshots = _normalized_projected_source_snapshots(runtime, projected_source_snapshots)
    state_by_module_id = _runtime_module_state_map(runtime)
    cache_key = get_runtime_resolve_cache_key(runtime, snapshots, projected_snapshots)
    if cache_key is None:
        return None
    cached = _PYFA_RUNTIME_RESOLVED_CACHE.get(cache_key)
    if cached is not None:
        resolved_runtime = deepcopy(cached[0])
        resolved_runtime.fit_key = runtime.fit_key
        resolved_runtime.diagnostics["pyfa_blueprint"] = deepcopy(blueprint)
        resolved_runtime.diagnostics["pyfa_command_boosters"] = deepcopy(snapshots)
        resolved_runtime.diagnostics["pyfa_projected_sources"] = deepcopy(projected_snapshots)
        resolved_runtime.diagnostics["pyfa_runtime_resolve_cache"] = "hit"
        resolved_runtime.diagnostics["pyfa_projected_target_fit_cache"] = "resolved_hit"
        resolved_runtime.diagnostics["pyfa_resolve_signature"] = cache_key
        _copy_dynamic_runtime_state(runtime, resolved_runtime)
        return resolved_runtime, replace(cached[1])

    projected_target_fit_cache = "single_pass"
    try:
        factory = RuntimeFromEftFactory()
        target_fit, target_fitted_modules = _copy_precalculated_neutral_base_fit(factory, parsed)
        target_fit.ID = 1
        _apply_pyfa_local_state_map(factory, target_fitted_modules, state_by_module_id)

        next_fit_id = 2
        next_fit_id = _attach_command_snapshot_fits(factory, target_fit, snapshots, next_fit_id, runtime)
        next_fit_id = _attach_projected_snapshot_fits(factory, target_fit, projected_snapshots, next_fit_id, runtime)
        _calculate_pyfa_fit_with_state_normalization(factory, target_fit, target_fitted_modules)

        resolved_runtime, _fit, profile = factory._build_runtime_artifacts_from_pyfa_fit(
            parsed,
            target_fit,
            target_fitted_modules,
            state_by_module_id=state_by_module_id,
            command_booster_snapshots=snapshots,
        )
        resolved_runtime.diagnostics["pyfa_projected_sources"] = deepcopy(projected_snapshots)
        resolved_runtime.diagnostics["pyfa_runtime_resolve_cache"] = "miss"
        resolved_runtime.diagnostics["pyfa_projected_target_fit_cache"] = projected_target_fit_cache
        resolved_runtime.diagnostics["pyfa_resolve_signature"] = cache_key
        _copy_dynamic_runtime_state(runtime, resolved_runtime)
    except Exception:
        return None

    _PYFA_RUNTIME_RESOLVED_CACHE[cache_key] = (deepcopy(resolved_runtime), replace(profile))
    return resolved_runtime, profile


def recompute_profile_from_pyfa_runtime(
    runtime: FitRuntime,
    command_booster_snapshots: list[dict[str, Any]] | None = None,
    projected_source_snapshots: list[dict[str, Any]] | None = None,
) -> ShipProfile | None:
    resolved = resolve_runtime_from_pyfa_runtime(runtime, command_booster_snapshots, projected_source_snapshots)
    if resolved is None:
        return None
    profile = resolved[1]
    return replace(profile)


def get_common_chargeable_modules(fit_texts: list[str], usage_threshold: float = 0.05, language: str = "en") -> list[str]:
    parser = EftFitParser()
    backend = _get_static_backend()
    if not backend.available:
        return []
    total = 0
    counts: Counter[str] = Counter()
    chargeable_cache: dict[str, bool] = {}
    for text in fit_texts:
        try:
            parsed = parser.parse(text)
        except Exception:
            continue
        for spec in parsed.module_specs:
            module_name = backend.resolve_type_name(spec.module_name)
            if not module_name:
                continue
            key = module_name.lower()
            is_chargeable = chargeable_cache.get(key)
            if is_chargeable is None:
                is_chargeable = bool(backend.list_charge_options_for_module(module_name))
                chargeable_cache[key] = is_chargeable
            if not is_chargeable:
                continue
            counts[module_name] += 1
            total += 1
    total = max(1, total)
    threshold = 1 if usage_threshold <= 0 else max(1, int(total * usage_threshold + 0.9999))
    rows = [(name, count) for name, count in counts.items() if count >= threshold]
    rows.sort(key=lambda item: (-item[1], backend.localize_type_name(item[0], language).lower()))
    return [backend.localize_type_name(name, language) for name, _ in rows]


def get_charge_options_for_module(module_name: str, language: str = "en") -> list[str]:
    backend = _get_static_backend()
    canonical_module = backend.resolve_type_name(module_name)
    ammo = backend.list_charge_options_for_module(canonical_module)
    return [backend.localize_type_name(name, language) for name in ammo]


def get_type_display_name(type_name: str, language: str = "en") -> str:
    backend = _get_static_backend()
    return backend.localize_type_name(type_name, language)


def resolve_module_type_name(type_name: str) -> str:
    backend = _get_static_backend()
    return backend.resolve_type_name(type_name)


def get_module_reload_time_sec(module_name: str) -> float:
    backend = _get_static_backend()
    canonical_module = backend.resolve_type_name(module_name)
    return backend.module_reload_time_sec(canonical_module)


def get_module_reload_channel(module_name: str) -> str:
    backend = _get_static_backend()
    canonical_module = backend.resolve_type_name(module_name)
    return backend.module_reload_channel(canonical_module)


def replace_module_charge_in_fit_text(fit_text: str, module_name: str, ammo_name: str) -> str:
    backend = _get_static_backend()
    module_name = backend.resolve_type_name(module_name)
    ammo_name = backend.resolve_type_name(ammo_name)
    lines = fit_text.splitlines()
    out: list[str] = []
    target = module_name.strip().lower()
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("[") or raw.lower().startswith("dna:"):
            out.append(line)
            continue
        offline_suffix = ""
        base = raw
        if base.endswith("/offline"):
            base = base[:-8].rstrip()
            offline_suffix = " /offline"
        if base.endswith("/OFFLINE"):
            base = base[:-8].rstrip()
            offline_suffix = " /OFFLINE"
        module_name = base.split(",", 1)[0].strip()
        if module_name.lower() == target:
            out.append(f"{module_name}, {ammo_name}{offline_suffix}")
        else:
            out.append(line)
    return "\n".join(out)


def default_manual_setup() -> list[ManualShipSetup]:
    blue_ferox = """[Ferox, Rail DPS]\n250mm Railgun II, Antimatter Charge M\n250mm Railgun II, Antimatter Charge M\n250mm Railgun II, Antimatter Charge M\nMagnetic Field Stabilizer II\nMagnetic Field Stabilizer II\nTracking Enhancer II\n10MN Afterburner II\n"""
    blue_scythe = """[Scythe, Shield Logi]\n10MN Afterburner II\nSensor Booster II\n"""
    red_ferox = """[Ferox, Rail DPS]\n250mm Railgun II, Antimatter Charge M\n250mm Railgun II, Antimatter Charge M\nMagnetic Field Stabilizer II\nTracking Enhancer II\n"""
    red_blackbird = """[Blackbird, EWAR]\nStasis Webifier II\nRemote Sensor Dampener II\nTarget Painter II\nMedium Energy Neutralizer II\n"""
    rows: list[ManualShipSetup] = []
    for i in range(12):
        rows.append(
            ManualShipSetup(
                team=Team.BLUE,
                squad_id="BLUE-ALPHA",
                quality=QualityLevel.REGULAR,
                position=Vector2(-45_000 + random.uniform(-3_000, 3_000), -6_000 + random.uniform(-3_000, 3_000)),
                fit_text=blue_ferox,
            )
        )
    for i in range(4):
        rows.append(
            ManualShipSetup(
                team=Team.BLUE,
                squad_id="BLUE-LOGI",
                quality=QualityLevel.ELITE,
                position=Vector2(-48_000 + random.uniform(-2_000, 2_000), 6_000 + random.uniform(-2_000, 2_000)),
                fit_text=blue_scythe,
            )
        )
    for i in range(14):
        rows.append(
            ManualShipSetup(
                team=Team.RED,
                squad_id="RED-ALPHA",
                quality=QualityLevel.IRREGULAR,
                position=Vector2(42_000 + random.uniform(-3_000, 3_000), random.uniform(-4_000, 4_000)),
                fit_text=red_ferox,
            )
        )
    for i in range(3):
        rows.append(
            ManualShipSetup(
                team=Team.RED,
                squad_id="RED-EWAR",
                quality=QualityLevel.REGULAR,
                position=Vector2(38_000 + random.uniform(-2_000, 2_000), 6_000 + random.uniform(-2_000, 2_000)),
                fit_text=red_blackbird,
            )
        )
    return rows


def build_world_from_manual_setup(ship_setups: list[ManualShipSetup]) -> WorldState:
    def random_point_in_radius(center: Vector2, radius: float) -> Vector2:
        theta = random.uniform(0.0, 2.0 * 3.141592653589793)
        distance = radius * (random.random() ** 0.5)
        return Vector2(center.x + math.cos(theta) * distance, center.y + math.sin(theta) * distance)

    world = WorldState()
    world.beacons["gate-1"] = Beacon(
        beacon_id="gate-1",
        position=Vector2(-80_000, 0),
        radius=2000,
        interaction_range=2500,
        kind="STARGATE",
    )
    world.beacons["gate-2"] = Beacon(
        beacon_id="gate-2",
        position=Vector2(80_000, 0),
        radius=2000,
        interaction_range=2500,
        kind="STARGATE",
    )

    parser = EftFitParser()
    factory = RuntimeFromEftFactory()
    counters: dict[tuple[Team, str], int] = {}
    first_ship_per_squad: dict[str, str] = {}
    squad_centers: dict[tuple[Team, str], Vector2] = {}

    for setup in ship_setups:
        quality = QUALITY_PRESETS[setup.quality]
        parsed = parser.parse(setup.fit_text)
        runtime_template, fit = factory.build(parsed)
        runtime = deepcopy(runtime_template)
        profile = factory.build_profile(parsed)

        key = (setup.team, setup.squad_id)
        if key not in squad_centers:
            team_anchor = Vector2(-45_000.0, 0.0) if setup.team == Team.BLUE else Vector2(45_000.0, 0.0)
            squad_centers[key] = random_point_in_radius(team_anchor, 12_000.0)
        spawn_position = random_point_in_radius(squad_centers[key], 5_000.0)
        counters[key] = counters.get(key, 0) + 1
        ship_id = f"{setup.team.value}-{setup.squad_id}-{counters[key]:03d}"

        ship = ShipEntity(
            ship_id=ship_id,
            team=setup.team,
            squad_id=setup.squad_id,
            fit=fit,
            profile=profile,
            runtime=runtime,
            nav=NavigationState(position=spawn_position, velocity=Vector2(0.0, 0.0), facing_deg=0.0, max_speed=profile.max_speed),
            combat=CombatState(),
            vital=VitalState(
                shield=max(1.0, profile.shield_hp),
                armor=max(1.0, profile.armor_hp),
                structure=max(1.0, profile.structure_hp),
                shield_max=max(1.0, profile.shield_hp),
                armor_max=max(1.0, profile.armor_hp),
                structure_max=max(1.0, profile.structure_hp),
                cap=profile.max_cap,
                cap_max=profile.max_cap,
                alive=True,
            ),
            quality=QualityState(
                level=quality.level,
                reaction_delay=quality.reaction_delay,
                ignore_order_probability=quality.ignore_order_probability,
                formation_jitter=quality.formation_jitter,
            ),
        )
        world.ships[ship_id] = ship

        squad_key = f"{setup.team.value}:{setup.squad_id}"
        if squad_key not in first_ship_per_squad:
            first_ship_per_squad[squad_key] = ship_id
        if setup.is_leader and squad_key not in world.squad_leaders:
            world.squad_leaders[squad_key] = ship_id

    for squad_key, first_ship_id in first_ship_per_squad.items():
        if squad_key not in world.squad_leaders:
            world.squad_leaders[squad_key] = first_ship_id

    return world

