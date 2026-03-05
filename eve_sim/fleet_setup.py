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

from .config import resolve_pyfa_source_dir

from .fit_runtime import (
    EffectClass,
    FitRuntime,
    HullProfile,
    ModuleEffect,
    ModuleRuntime,
    ModuleState,
    SkillProfile,
)
from .math2d import Vector2
from .models import (
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
from .world import WorldState


QUALITY_PRESETS = {
    QualityLevel.ELITE: QualityState(QualityLevel.ELITE, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.REGULAR: QualityState(QualityLevel.REGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.IRREGULAR: QualityState(QualityLevel.IRREGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
}


@dataclass(slots=True)
class ParsedEftFit:
    ship_name: str
    fit_name: str
    module_names: list[str]
    module_specs: list["ParsedModuleSpec"]
    cargo_item_names: list[str]
    fit_key: str


@dataclass(slots=True)
class ParsedModuleSpec:
    module_name: str
    charge_name: str | None = None
    offline: bool = False


@dataclass(slots=True)
class ManualShipSetup:
    team: Team
    squad_id: str
    quality: QualityLevel
    position: Vector2
    fit_text: str
    is_leader: bool = False


class EftFitParser:
    _header_re = re.compile(r"^\[(?P<ship>[^,\]]+)\s*,\s*(?P<name>[^\]]+)\]$")
    _offline_suffixes = ("/offline", "/OFFLINE")

    def parse(self, fit_text: str) -> ParsedEftFit:
        lines = [line.strip() for line in fit_text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError("配装文本为空")

        m = self._header_re.match(lines[0])
        if not m:
            raise ValueError("EFT 头格式无效，需形如 [Ship, Fit Name]")

        ship_name = m.group("ship").strip()
        fit_name = m.group("name").strip()

        modules: list[str] = []
        module_specs: list[ParsedModuleSpec] = []
        cargo_item_names: list[str] = []

        for raw in lines[1:]:
            if raw.lower().startswith("dna:"):
                continue
            if raw.lower().startswith("x-"):
                continue
            line = raw
            is_quantity_line = " x" in line
            if is_quantity_line:
                qty_name = line.split(" x", 1)[0].strip()
                if qty_name:
                    cargo_item_names.append(qty_name)
                continue
            offline = False
            for suffix in self._offline_suffixes:
                if line.endswith(suffix):
                    offline = True
                    line = line[: -len(suffix)].strip()
                    break
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            charge_name: str | None = None
            if "," in line:
                mod, charge = line.split(",", 1)
                line = mod.strip()
                charge_name = charge.strip() or None
            if line.startswith("[Empty"):
                continue
            modules.append(line)
            module_specs.append(ParsedModuleSpec(module_name=line, charge_name=charge_name, offline=offline))

        fit_key = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        return ParsedEftFit(
            ship_name=ship_name,
            fit_name=fit_name,
            module_names=modules,
            module_specs=module_specs,
            cargo_item_names=cargo_item_names,
            fit_key=f"eft-{fit_key}",
        )


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

    def _module_effect_pyfa(self, fitted_module, idx: int) -> ModuleRuntime | None:
        item = getattr(fitted_module, "item", None)
        if item is None:
            return None

        group_name = (item.group.name or "").lower()
        suffix = f"-{idx}"

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

        def attr(name: str, default: float = 0.0) -> float:
            value = attr_opt(name)
            return float(default if value is None else value)

        def pct_to_mult(value: float) -> float:
            return max(0.01, 1.0 + value / 100.0)

        duration_ms = attr("duration", 0.0)
        speed_ms = attr("speed", 0.0)
        cycle_ms = duration_ms if duration_ms > 0 else speed_ms
        cycle_sec = max(0.1, cycle_ms / 1000.0) if cycle_ms > 0 else 5.0
        cap_need = max(0.0, attr("capacitorNeed", 0.0))
        reactivation_delay_sec = max(0.0, attr("moduleReactivationDelay", 0.0) / 1000.0)

        range_m = max(0.0, attr("maxRange", 0.0))
        falloff_m = max(0.0, attr("falloffEffectiveness", 0.0))
        if falloff_m <= 0.0:
            falloff_m = max(0.0, attr("falloff", 0.0))

        local_mult: dict[str, float] = {}
        local_add: dict[str, float] = {}
        projected_mult: dict[str, float] = {}
        projected_add: dict[str, float] = {}

        speed_factor = attr_opt("speedFactor")
        if speed_factor is not None:
            if speed_factor < 0 and range_m > 0:
                projected_mult["speed"] = pct_to_mult(speed_factor)
            elif speed_factor > 0:
                local_mult["speed"] = max(local_mult.get("speed", 1.0), pct_to_mult(speed_factor))

        max_velocity_bonus = attr_opt("maxVelocityBonus")
        if max_velocity_bonus is not None and max_velocity_bonus > 0:
            local_mult["speed"] = max(local_mult.get("speed", 1.0), pct_to_mult(max_velocity_bonus))

        signature_radius_bonus = attr_opt("signatureRadiusBonus")
        if signature_radius_bonus is not None:
            if signature_radius_bonus > 0 and range_m > 0:
                projected_mult["sig"] = pct_to_mult(signature_radius_bonus)
            elif signature_radius_bonus != 0:
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
            if value < 0 and range_m > 0:
                projected_mult[key] = pct_to_mult(value)
            else:
                local_mult[key] = pct_to_mult(value)

        jam_strength = max(
            attr("scanGravimetricStrengthBonus", 0.0),
            attr("scanLadarStrengthBonus", 0.0),
            attr("scanMagnetometricStrengthBonus", 0.0),
            attr("scanRadarStrengthBonus", 0.0),
        )
        if jam_strength > 0.0 and range_m > 0.0:
            suppression = max(0.0, min(0.95, jam_strength / (jam_strength + 10.0)))
            projected_mult.setdefault("scan", max(0.05, 1.0 - suppression))
            projected_mult.setdefault("range", max(0.05, 1.0 - suppression))

        if range_m > 0.0:
            shield_rep = abs(attr("shieldBonus", 0.0))
            if shield_rep > 0.0:
                projected_add["shield_rep"] = shield_rep
            armor_rep = abs(attr("armorDamageAmount", 0.0))
            if armor_rep > 0.0:
                projected_add["armor_rep"] = armor_rep
            cap_drain = abs(attr("energyNeutralizerAmount", 0.0))
            if cap_drain > 0.0:
                projected_add["cap_drain"] = cap_drain
        else:
            local_rep = abs(attr("shieldBonus", 0.0)) + abs(attr("armorDamageAmount", 0.0))
            if local_rep > 0.0:
                local_add["rep"] = local_add.get("rep", 0.0) + local_rep

        cap_capacity_bonus = attr_opt("capacitorCapacityBonus")
        if cap_capacity_bonus is not None and abs(cap_capacity_bonus) > 1e-9:
            local_mult["cap_max"] = pct_to_mult(cap_capacity_bonus)

        cap_recharge_mult = attr_opt("capacitorRechargeRateMultiplier")
        if cap_recharge_mult is not None and cap_recharge_mult > 0 and abs(cap_recharge_mult - 1.0) > 1e-6:
            local_mult["cap_recharge"] = max(0.01, cap_recharge_mult)

        if not self._is_weapon_like_group(group_name):
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

        has_projected = bool(projected_mult or projected_add)
        is_active_module = (cap_need > 0.0) or (cycle_ms > 0.0) or has_projected
        if self._is_weapon_like_group(group_name) and cycle_ms > 0.0:
            is_active_module = True

        state_required = ModuleState.ACTIVE if is_active_module else ModuleState.ONLINE
        module_state = ModuleState.ACTIVE if is_active_module else ModuleState.ONLINE
        module_id = f"mod{suffix}"

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
            )

        return ModuleRuntime(
            module_id=module_id,
            group=str(item.group.name or ""),
            state=module_state,
            effects=[effect],
        )

    @staticmethod
    def _collect_pyfa_weapon_stats(modules: list[Any], ship: Any, *, require_volley: bool) -> dict[str, float]:
        turret_dps = 0.0
        missile_dps = 0.0
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
                if require_volley:
                    _ = module.getVolley()
            except Exception:
                continue

            dps_total = float(getattr(dps_obj, "total", 0.0) or 0.0)
            if dps_total <= 0.0:
                continue

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

    def _build_pyfa_fit(self, parsed: ParsedEftFit) -> tuple[Any, list[tuple[ParsedModuleSpec, Any, str | None]]]:
        if not self._pyfa.fit_engine_ready:
            raise ValueError("pyfa Fit计算链不可用")

        fit_cls = self._pyfa._fit_cls
        ship_cls = self._pyfa._ship_cls
        module_cls = self._pyfa._module_cls
        character_cls = self._pyfa._character_cls
        active_state = self._pyfa._fitting_module_state_active
        offline_state = self._pyfa._fitting_module_state_offline
        assert fit_cls is not None
        assert ship_cls is not None
        assert module_cls is not None
        assert character_cls is not None
        assert active_state is not None
        assert offline_state is not None

        ship_name = self._pyfa.resolve_type_name(parsed.ship_name)
        ship_item = self._pyfa.get_item(ship_name)
        if ship_item is None:
            raise ValueError(f"pyfa中未找到舰船：{parsed.ship_name}")

        fit = fit_cls(ship=ship_cls(ship_item), name=parsed.fit_name)
        fit.character = character_cls.getAll5()
        fitted_modules: list[tuple[ParsedModuleSpec, Any, str | None]] = []

        for spec in parsed.module_specs:
            module_name = self._pyfa.resolve_type_name(spec.module_name)
            module_item = self._pyfa.get_item(module_name)
            if module_item is None:
                raise ValueError(f"pyfa中未找到模块：{spec.module_name}")
            module = module_cls(module_item)
            module.owner = fit
            module.state = offline_state if spec.offline else active_state

            group_name = (module_item.group.name or "").lower()
            charge_name = self._resolve_module_charge_name(module_item, spec.charge_name)
            if self._is_weapon_like_group(group_name) and not charge_name:
                raise ValueError(f"武器缺少可解析弹药：{spec.module_name}")
            if charge_name:
                charge_item = self._pyfa.get_item(self._pyfa.resolve_type_name(charge_name))
                if charge_item is None:
                    raise ValueError(f"pyfa中未找到弹药：{charge_name}")
                module.charge = charge_item

            fit.modules.append(module)
            fitted_modules.append((spec, module, charge_name))

        fit.calculateModifiedAttributes()
        return fit, fitted_modules

    def _compute_pyfa_final_stats(self, fit) -> dict[str, float]:
        ship = fit.ship
        weapon_stats = self._collect_pyfa_weapon_stats(cast(list[Any], fit.modules), ship, require_volley=True)
        return {
            "dps": float(fit.getTotalDps().total),
            "volley": float(fit.getTotalVolley().total),
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

    def build(self, parsed: ParsedEftFit) -> tuple[FitRuntime, FitDescriptor]:
        if not self._pyfa.available:
            raise ValueError("pyfa 静态数据库不可用，无法进行严格数值解析")

        cached_runtime = self._runtime_cache.get(parsed.fit_key)
        cached_fit = self._fit_cache.get(parsed.fit_key)
        if cached_runtime is not None and cached_fit is not None:
            return cached_runtime, cached_fit

        fit_ctx, fitted_modules = self._build_pyfa_fit(parsed)
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
        for idx, (spec, fitted_module, effective_charge_name) in enumerate(fitted_modules, start=1):
            module = self._module_effect_pyfa(fitted_module, idx)
            if module is not None:
                if spec.offline:
                    module.state = ModuleState.OFFLINE
                modules.append(module)
                pyfa_blueprint_modules.append(
                    {
                        "module_id": module.module_id,
                        "module_name": spec.module_name,
                        "charge_name": effective_charge_name,
                        "offline": bool(spec.offline),
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
        )

        runtime = FitRuntime(
            fit_key=parsed.fit_key,
            hull=hull,
            skills=self._skills_default(),
            modules=modules,
        )
        runtime.diagnostics["pyfa_blueprint"] = {
            "ship_name": parsed.ship_name,
            "fit_name": parsed.fit_name,
            "modules": pyfa_blueprint_modules,
        }
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
            max_speed=profile.max_speed,
            max_cap=profile.max_cap,
            cap_recharge_time=profile.cap_recharge_time,
            shield_hp=profile.shield_hp,
            armor_hp=profile.armor_hp,
            structure_hp=profile.structure_hp,
            rep_amount=profile.rep_amount,
            rep_cycle=profile.rep_cycle,
        )
        self._runtime_cache[parsed.fit_key] = runtime
        self._fit_cache[parsed.fit_key] = fit
        self._profile_cache[parsed.fit_key] = profile
        return runtime, fit

    def build_profile(self, parsed: ParsedEftFit) -> ShipProfile:
        self.build(parsed)
        cached = self._profile_cache.get(parsed.fit_key)
        if cached is None:
            raise ValueError("内部错误：缺少配装剖面缓存")
        return replace(cached)

    @property
    def backend_status(self) -> str:
        return self._pyfa.status


class _PyfaStaticBackend:
    def __init__(self) -> None:
        self.available = False
        self.status = "Fallback: 内置规则映射"
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
            self.status = f"Fallback: 未检测到可用 pyfa 静态库 ({db_path})"
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
            self.status = "Pyfa: 静态数据驱动"
        except ModuleNotFoundError as exc:
            self.available = False
            missing = getattr(exc, "name", None) or str(exc)
            self.status = f"Fallback: pyfa加载失败 (ModuleNotFoundError: {missing})"
        except Exception as exc:
            self.available = False
            self.status = f"Fallback: pyfa加载失败 ({type(exc).__name__})"

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
_PYFA_RUNTIME_PROFILE_CACHE: dict[tuple[Any, ...], ShipProfile] = {}


def _get_static_backend() -> _PyfaStaticBackend:
    global _STATIC_BACKEND
    if _STATIC_BACKEND is None:
        _STATIC_BACKEND = _PyfaStaticBackend()
    return _STATIC_BACKEND


def get_fit_backend_status() -> str:
    return RuntimeFromEftFactory().backend_status


def recompute_profile_from_pyfa_runtime(runtime: FitRuntime) -> ShipProfile | None:
    backend = _get_static_backend()
    if not backend.fit_engine_ready:
        return None
    blueprint = runtime.diagnostics.get("pyfa_blueprint")
    if not isinstance(blueprint, dict):
        return None

    ship_name = str(blueprint.get("ship_name", "") or "").strip()
    fit_name = str(blueprint.get("fit_name", "") or "").strip()
    module_specs = blueprint.get("modules")
    if not ship_name or not isinstance(module_specs, list):
        return None

    signature = tuple(sorted((m.module_id, m.state.value) for m in runtime.modules))
    blueprint_signature = tuple(
        sorted(
            (
                str(raw.get("module_id", "") or ""),
                str(raw.get("module_name", "") or ""),
                str(raw.get("charge_name", "") or ""),
                bool(raw.get("offline", False)),
            )
            for raw in module_specs
            if isinstance(raw, dict)
        )
    )
    cache_key = (runtime.fit_key, blueprint_signature, signature)
    cached = _PYFA_RUNTIME_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return replace(cached)

    fit_cls = cast(Any, backend._fit_cls)
    ship_cls = cast(Any, backend._ship_cls)
    module_cls = cast(Any, backend._module_cls)
    character_cls = cast(Any, backend._character_cls)
    state_active = cast(Any, backend._fitting_module_state_active)
    state_online = cast(Any, backend._fitting_module_state_online)
    state_offline = cast(Any, backend._fitting_module_state_offline)
    state_overheated = cast(Any, backend._fitting_module_state_overheated)
    if any(x is None for x in (fit_cls, ship_cls, module_cls, character_cls, state_active, state_online, state_offline)):
        return None

    state_by_module_id = {m.module_id: m.state.value for m in runtime.modules}
    resolved_ship = backend.resolve_type_name(ship_name)
    ship_item = backend.get_item(resolved_ship)
    if ship_item is None:
        return None

    try:
        fit = fit_cls(ship=ship_cls(ship_item), name=fit_name or runtime.fit_key)
        fit.character = character_cls.getAll5()
        for raw in module_specs:
            if not isinstance(raw, dict):
                continue
            module_name = str(raw.get("module_name", "") or "").strip()
            if not module_name:
                continue
            module_item = backend.get_item(backend.resolve_type_name(module_name))
            if module_item is None:
                continue
            module = module_cls(module_item)
            module.owner = fit

            module_id = str(raw.get("module_id", "") or "")
            original_offline = bool(raw.get("offline", False))
            runtime_state = str(state_by_module_id.get(module_id, "ONLINE") or "ONLINE").upper()
            if original_offline or runtime_state == "OFFLINE":
                module.state = state_offline
            elif runtime_state == "ACTIVE":
                module.state = state_active
            elif runtime_state == "OVERHEATED":
                module.state = state_overheated if state_overheated is not None else state_active
            else:
                module.state = state_online

            charge_name = raw.get("charge_name")
            if charge_name:
                charge_item = backend.get_item(backend.resolve_type_name(str(charge_name)))
                if charge_item is not None:
                    module.charge = charge_item
            fit.modules.append(module)

        fit.calculateModifiedAttributes()
        ship = fit.ship
        weapon_stats = RuntimeFromEftFactory._collect_pyfa_weapon_stats(cast(list[Any], fit.modules), ship, require_volley=False)

        profile = ShipProfile(
            dps=float(fit.getTotalDps().total),
            volley=float(fit.getTotalVolley().total),
            optimal=weapon_stats["optimal"],
            falloff=weapon_stats["falloff"],
            tracking=weapon_stats["tracking"],
            optimal_sig=max(1.0, weapon_stats.get("optimal_sig", 40_000.0)),
            sig_radius=max(1.0, float(ship.getModifiedItemAttr("signatureRadius") or 0.0)),
            scan_resolution=max(1.0, float(ship.getModifiedItemAttr("scanResolution") or 0.0)),
            max_target_range=max(1000.0, float(ship.getModifiedItemAttr("maxTargetRange") or 0.0)),
            max_speed=max(1.0, float(ship.getModifiedItemAttr("maxVelocity") or 0.0)),
            max_cap=max(1.0, float(ship.getModifiedItemAttr("capacitorCapacity") or 0.0)),
            cap_recharge_time=max(1.0, float(ship.getModifiedItemAttr("rechargeRate") or 0.0) / 1000.0),
            shield_hp=max(1.0, float(ship.getModifiedItemAttr("shieldCapacity") or 0.0)),
            armor_hp=max(1.0, float(ship.getModifiedItemAttr("armorHP") or 0.0)),
            structure_hp=max(1.0, float(ship.getModifiedItemAttr("hp") or 0.0)),
            rep_amount=0.0,
            rep_cycle=5.0,
            weapon_system=(
                "mixed"
                if weapon_stats["turret_dps"] > 0 and weapon_stats["missile_dps"] > 0
                else ("missile" if weapon_stats["missile_dps"] > 0 else "turret")
            ),
            turret_dps=max(0.0, weapon_stats["turret_dps"]),
            missile_dps=max(0.0, weapon_stats["missile_dps"]),
            turret_cycle=max(0.0, weapon_stats["turret_cycle"]),
            missile_cycle=max(0.0, weapon_stats["missile_cycle"]),
            damage_em=max(0.0, weapon_stats["damage_em"]),
            damage_thermal=max(0.0, weapon_stats["damage_thermal"]),
            damage_kinetic=max(0.0, weapon_stats["damage_kinetic"]),
            damage_explosive=max(0.0, weapon_stats["damage_explosive"]),
            turret_em_dps=max(0.0, weapon_stats["turret_em_dps"]),
            turret_thermal_dps=max(0.0, weapon_stats["turret_thermal_dps"]),
            turret_kinetic_dps=max(0.0, weapon_stats["turret_kinetic_dps"]),
            turret_explosive_dps=max(0.0, weapon_stats["turret_explosive_dps"]),
            missile_em_dps=max(0.0, weapon_stats["missile_em_dps"]),
            missile_thermal_dps=max(0.0, weapon_stats["missile_thermal_dps"]),
            missile_kinetic_dps=max(0.0, weapon_stats["missile_kinetic_dps"]),
            missile_explosive_dps=max(0.0, weapon_stats["missile_explosive_dps"]),
            missile_explosion_radius=max(0.0, weapon_stats["missile_explosion_radius"]),
            missile_explosion_velocity=max(0.0, weapon_stats["missile_explosion_velocity"]),
            missile_max_range=max(0.0, weapon_stats["missile_max_range"]),
            missile_damage_reduction_factor=max(0.1, min(2.0, weapon_stats["missile_damage_reduction_factor"])),
            shield_resonance_em=max(0.01, min(1.0, float(ship.getModifiedItemAttr("shieldEmDamageResonance") or 1.0))),
            shield_resonance_thermal=max(0.01, min(1.0, float(ship.getModifiedItemAttr("shieldThermalDamageResonance") or 1.0))),
            shield_resonance_kinetic=max(0.01, min(1.0, float(ship.getModifiedItemAttr("shieldKineticDamageResonance") or 1.0))),
            shield_resonance_explosive=max(0.01, min(1.0, float(ship.getModifiedItemAttr("shieldExplosiveDamageResonance") or 1.0))),
            armor_resonance_em=max(0.01, min(1.0, float(ship.getModifiedItemAttr("armorEmDamageResonance") or 1.0))),
            armor_resonance_thermal=max(0.01, min(1.0, float(ship.getModifiedItemAttr("armorThermalDamageResonance") or 1.0))),
            armor_resonance_kinetic=max(0.01, min(1.0, float(ship.getModifiedItemAttr("armorKineticDamageResonance") or 1.0))),
            armor_resonance_explosive=max(0.01, min(1.0, float(ship.getModifiedItemAttr("armorExplosiveDamageResonance") or 1.0))),
            structure_resonance_em=max(0.01, min(1.0, float(ship.getModifiedItemAttr("emDamageResonance") or 1.0))),
            structure_resonance_thermal=max(0.01, min(1.0, float(ship.getModifiedItemAttr("thermalDamageResonance") or 1.0))),
            structure_resonance_kinetic=max(0.01, min(1.0, float(ship.getModifiedItemAttr("kineticDamageResonance") or 1.0))),
            structure_resonance_explosive=max(0.01, min(1.0, float(ship.getModifiedItemAttr("explosiveDamageResonance") or 1.0))),
        )
    except Exception:
        return None

    _PYFA_RUNTIME_PROFILE_CACHE[cache_key] = replace(profile)
    return profile


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
