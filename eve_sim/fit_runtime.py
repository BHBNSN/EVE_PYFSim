from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import ShipProfile


class ModuleState(str, Enum):
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"
    ACTIVE = "ACTIVE"
    OVERHEATED = "OVERHEATED"


class EffectClass(str, Enum):
    LOCAL = "LOCAL"
    PROJECTED = "PROJECTED"


@dataclass(slots=True)
class SkillProfile:
    levels: dict[str, int] = field(default_factory=dict)

    def get(self, name: str, default: int = 0) -> int:
        return int(self.levels.get(name, default))


@dataclass(slots=True)
class HullProfile:
    ship_name: str
    role: str
    base_dps: float
    volley: float
    optimal: float
    falloff: float
    tracking: float
    sig_radius: float
    scan_resolution: float
    max_target_range: float
    max_speed: float
    cap_max: float
    cap_recharge_time: float
    shield_hp: float
    armor_hp: float
    structure_hp: float
    rep_amount: float
    rep_cycle: float
    energy_warfare_resistance: float = 1.0
    mass: float = 0.0
    agility: float = 0.0
    warp_speed_au_s: float = 0.0
    warp_capacitor_need: float = 0.0
    max_warp_distance_au: float = 0.0
    disallow_assistance: bool = False
    warp_bubble_immune: bool = False
    is_shuttle: bool = False


@dataclass(slots=True)
class ModuleEffect:
    name: str
    effect_class: EffectClass
    state_required: ModuleState = ModuleState.ACTIVE
    range_m: float = 0.0
    falloff_m: float = 0.0
    cycle_time: float = 5.0
    cap_need: float = 0.0
    reactivation_delay: float = 0.0
    local_mult: dict[str, float] = field(default_factory=dict)
    local_add: dict[str, float] = field(default_factory=dict)
    projected_mult: dict[str, float] = field(default_factory=dict)
    projected_add: dict[str, float] = field(default_factory=dict)
    projected_mult_groups: dict[str, str | None] = field(default_factory=dict)
    projected_signature: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(slots=True, weakref_slot=True)
class ModuleRuntime:
    module_id: str
    group: str
    state: ModuleState
    effects: list[ModuleEffect] = field(default_factory=list)
    charge_capacity: int = 0
    charge_rate: float = 0.0
    charge_remaining: float = 0.0
    charge_reload_time: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def is_active_for(self, required: ModuleState) -> bool:
        rank = {
            ModuleState.OFFLINE: 0,
            ModuleState.ONLINE: 1,
            ModuleState.ACTIVE: 2,
            ModuleState.OVERHEATED: 3,
        }
        return rank[self.state] >= rank[required]

    def can_be_active(self) -> bool:
        return any(effect.state_required in {ModuleState.ACTIVE, ModuleState.OVERHEATED} for effect in self.effects)

    def normalized_state(self, state: ModuleState | None = None) -> ModuleState:
        candidate = self.state if state is None else state
        if candidate in {ModuleState.ACTIVE, ModuleState.OVERHEATED} and not self.can_be_active():
            return ModuleState.ONLINE
        return candidate

    def has_tag(self, tag: str) -> bool:
        return str(tag) in self.tags


@dataclass(slots=True)
class FitRuntime:
    fit_key: str
    hull: HullProfile
    skills: SkillProfile
    modules: list[ModuleRuntime] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedImpact:
    source_ship_id: str
    target_ship_id: str
    effect: ModuleEffect
    strength: float = 1.0


class RuntimeStatEngine:
    def __init__(self) -> None:
        self._cache: dict[tuple, ShipProfile] = {}

    @staticmethod
    def _stacking_penalty_factor(index: int) -> float:
        # EVE generic stacking penalty: S(n) = 0.5 ^ (((n-1) / 2.22292081) ^ 2)
        # Here index=0 is the first strongest modifier (n=1).
        return 0.5 ** (((float(index)) / 2.22292081) ** 2)

    @staticmethod
    def _stacking_multiplier(values: list[float]) -> float:
        if not values:
            return 1.0
        val = 1.0
        bonus = [v for v in values if v > 1]
        penalty = [v for v in values if v < 1]
        bonus.sort(key=lambda x: -abs(x - 1))
        penalty.sort(key=lambda x: -abs(x - 1))
        for group in (bonus, penalty):
            for i, mult in enumerate(group):
                val *= 1 + (mult - 1) * RuntimeStatEngine._stacking_penalty_factor(i)
        return val

    @classmethod
    def _stacking_group_multiplier(cls, grouped_values: dict[str, list[float]]) -> float:
        total = 1.0
        for values in grouped_values.values():
            total *= cls._stacking_multiplier(values)
        return total

    @classmethod
    def _merged_stacking_groups(
        cls,
        base_groups: dict[str, list[float]] | None,
        projected_groups: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        merged: dict[str, list[float]] = {}
        if isinstance(base_groups, dict):
            for group_name, values in base_groups.items():
                if not isinstance(values, list):
                    continue
                merged[str(group_name)] = [float(value or 1.0) for value in values]
        for group_name, values in projected_groups.items():
            merged.setdefault(str(group_name), []).extend(float(value or 1.0) for value in values)
        return merged

    @classmethod
    def _apply_penalized_projection(
        cls,
        current_value: float,
        add_value: float,
        projected_groups: dict[str, list[float]],
        penalty_context: dict[str, Any] | None,
    ) -> float:
        if isinstance(penalty_context, dict):
            base_value = float(penalty_context.get("pre", current_value) or 0.0)
            post_value = float(penalty_context.get("post", 0.0) or 0.0)
            merged_groups = cls._merged_stacking_groups(
                penalty_context.get("groups") if isinstance(penalty_context.get("groups"), dict) else None,
                projected_groups,
            )
            return (base_value + add_value) * cls._stacking_group_multiplier(merged_groups) + post_value
        return (current_value + add_value) * cls._stacking_group_multiplier(projected_groups)

    def compute_base_profile(self, runtime: FitRuntime) -> ShipProfile:
        key = self._cache_key(runtime)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        hull = runtime.hull
        mul: dict[str, list[float]] = {
            "dps": [],
            "optimal": [],
            "falloff": [],
            "tracking": [],
            "scan": [],
            "range": [],
            "speed": [],
            "sig": [],
            "cap_max": [],
            "cap_recharge": [],
            "rep": [],
            "mass": [],
            "agility": [],
        }
        add: dict[str, float] = {k: 0.0 for k in mul}
        for module in runtime.modules:
            for effect in module.effects:
                if effect.effect_class != EffectClass.LOCAL:
                    continue
                if not module.is_active_for(effect.state_required):
                    continue
                for k, v in effect.local_mult.items():
                    if k in mul:
                        mul[k].append(v)
                for k, v in effect.local_add.items():
                    if k in add:
                        add[k] += v

        profile = ShipProfile(
            dps=(hull.base_dps + add["dps"]) * self._stacking_multiplier(mul["dps"]),
            volley=hull.volley,
            optimal=(hull.optimal + add["optimal"]) * self._stacking_multiplier(mul["optimal"]),
            falloff=(hull.falloff + add["falloff"]) * self._stacking_multiplier(mul["falloff"]),
            tracking=(hull.tracking + add["tracking"]) * self._stacking_multiplier(mul["tracking"]),
            sig_radius=max(1.0, (hull.sig_radius + add["sig"]) * self._stacking_multiplier(mul["sig"])),
            scan_resolution=max(1.0, (hull.scan_resolution + add["scan"]) * self._stacking_multiplier(mul["scan"])),
            max_target_range=max(1000.0, (hull.max_target_range + add["range"]) * self._stacking_multiplier(mul["range"])),
            max_speed=max(1.0, (hull.max_speed + add["speed"]) * self._stacking_multiplier(mul["speed"])),
            max_cap=max(1.0, (hull.cap_max + add["cap_max"]) * self._stacking_multiplier(mul["cap_max"])),
            cap_recharge_time=max(1.0, (hull.cap_recharge_time + add["cap_recharge"]) * self._stacking_multiplier(mul["cap_recharge"])),
            shield_hp=max(1.0, hull.shield_hp),
            armor_hp=max(1.0, hull.armor_hp),
            structure_hp=max(1.0, hull.structure_hp),
            rep_amount=max(0.0, (hull.rep_amount + add["rep"]) * self._stacking_multiplier(mul["rep"])),
            rep_cycle=max(0.1, hull.rep_cycle),
            energy_warfare_resistance=max(0.0, float(getattr(hull, "energy_warfare_resistance", 1.0) or 1.0)),
            mass=max(0.0, (hull.mass + add["mass"]) * self._stacking_multiplier(mul["mass"])),
            agility=max(0.0, (hull.agility + add["agility"]) * self._stacking_multiplier(mul["agility"])),
            warp_speed_au_s=max(0.0, float(getattr(hull, "warp_speed_au_s", 0.0) or 0.0)),
            warp_capacitor_need=max(0.0, float(getattr(hull, "warp_capacitor_need", 0.0) or 0.0)),
            max_warp_distance_au=max(0.0, float(getattr(hull, "max_warp_distance_au", 0.0) or 0.0)),
            disallow_assistance=bool(getattr(hull, "disallow_assistance", False)),
            warp_bubble_immune=bool(getattr(hull, "warp_bubble_immune", False)),
            is_shuttle=bool(getattr(hull, "is_shuttle", False)),
        )
        self._cache[key] = profile
        return profile

    def apply_projected_effects(
        self,
        target: ShipProfile,
        impacts: list[ProjectedImpact],
        base_penalty_context: dict[str, dict[str, Any]] | None = None,
    ) -> ShipProfile:
        mul: dict[str, dict[str, list[float]]] = {
            "speed": [],
            "sig": [],
            "tracking": [],
            "optimal": [],
            "falloff": [],
            "scan": [],
            "range": [],
            "rep": [],
            "shield_hp": [],
            "armor_hp": [],
            "structure_hp": [],
            "sensor_strength_gravimetric": [],
            "sensor_strength_ladar": [],
            "sensor_strength_magnetometric": [],
            "sensor_strength_radar": [],
            "shield_resonance_em": [],
            "shield_resonance_thermal": [],
            "shield_resonance_kinetic": [],
            "shield_resonance_explosive": [],
            "armor_resonance_em": [],
            "armor_resonance_thermal": [],
            "armor_resonance_kinetic": [],
            "armor_resonance_explosive": [],
            "structure_resonance_em": [],
            "structure_resonance_thermal": [],
            "structure_resonance_kinetic": [],
            "structure_resonance_explosive": [],
            "missile_explosion_radius": [],
            "missile_explosion_velocity": [],
            "missile_range": [],
            "dps": [],
            "cap_max": [],
            "cap_recharge": [],
            "mass": [],
            "agility": [],
        }
        mul = {key: {} for key in mul}
        add: dict[str, float] = {k: 0.0 for k in mul}
        for impact in impacts:
            eff = impact.effect
            strength = max(0.0, min(1.0, float(impact.strength)))
            mult_groups = getattr(eff, "projected_mult_groups", {}) or {}
            for k, v in eff.projected_mult.items():
                if k in mul:
                    group_name = mult_groups.get(k, "default")
                    if group_name is None:
                        group_name = f"__unstacked__:{len(mul[k])}"
                    mul[k].setdefault(str(group_name), []).append(1.0 + (v - 1.0) * strength)
            for k, v in eff.projected_add.items():
                if k in add:
                    add[k] += v * strength

        penalty_context = base_penalty_context or {}

        return ShipProfile(
            dps=max(0.0, (target.dps + add["dps"]) * self._stacking_group_multiplier(mul["dps"])),
            volley=target.volley,
            optimal=max(1.0, (target.optimal + add["optimal"]) * self._stacking_group_multiplier(mul["optimal"])),
            falloff=max(1.0, (target.falloff + add["falloff"]) * self._stacking_group_multiplier(mul["falloff"])),
            tracking=max(0.0001, (target.tracking + add["tracking"]) * self._stacking_group_multiplier(mul["tracking"])),
            sig_radius=max(1.0, self._apply_penalized_projection(target.sig_radius, add["sig"], mul["sig"], penalty_context.get("sig"))),
            scan_resolution=max(
                1.0,
                self._apply_penalized_projection(
                    target.scan_resolution,
                    add["scan"],
                    mul["scan"],
                    penalty_context.get("scan"),
                ),
            ),
            max_target_range=max(
                1000.0,
                self._apply_penalized_projection(
                    target.max_target_range,
                    add["range"],
                    mul["range"],
                    penalty_context.get("range"),
                ),
            ),
            max_speed=max(
                1.0,
                self._apply_penalized_projection(
                    target.max_speed,
                    add["speed"],
                    mul["speed"],
                    penalty_context.get("speed"),
                ),
            ),
            missile_explosion_radius=max(
                0.0,
                target.missile_explosion_radius * self._stacking_group_multiplier(mul["missile_explosion_radius"]),
            ),
            missile_explosion_velocity=max(
                0.0,
                target.missile_explosion_velocity * self._stacking_group_multiplier(mul["missile_explosion_velocity"]),
            ),
            missile_max_range=max(
                0.0,
                target.missile_max_range * self._stacking_group_multiplier(mul["missile_range"]),
            ),
            max_cap=max(
                1.0,
                self._apply_penalized_projection(
                    target.max_cap,
                    add["cap_max"],
                    mul["cap_max"],
                    penalty_context.get("cap_max"),
                ),
            ),
            cap_recharge_time=max(
                1.0,
                self._apply_penalized_projection(
                    target.cap_recharge_time,
                    add["cap_recharge"],
                    mul["cap_recharge"],
                    penalty_context.get("cap_recharge"),
                ),
            ),
            shield_hp=max(
                1.0,
                self._apply_penalized_projection(
                    target.shield_hp,
                    add["shield_hp"],
                    mul["shield_hp"],
                    penalty_context.get("shield_hp"),
                ),
            ),
            armor_hp=max(
                1.0,
                self._apply_penalized_projection(
                    target.armor_hp,
                    add["armor_hp"],
                    mul["armor_hp"],
                    penalty_context.get("armor_hp"),
                ),
            ),
            structure_hp=max(
                1.0,
                self._apply_penalized_projection(
                    target.structure_hp,
                    add["structure_hp"],
                    mul["structure_hp"],
                    penalty_context.get("structure_hp"),
                ),
            ),
            rep_amount=max(0.0, (target.rep_amount + add["rep"]) * self._stacking_group_multiplier(mul["rep"])),
            rep_cycle=target.rep_cycle,
            energy_warfare_resistance=max(0.0, float(getattr(target, "energy_warfare_resistance", 1.0) or 1.0)),
            mass=max(0.0, self._apply_penalized_projection(target.mass, add["mass"], mul["mass"], penalty_context.get("mass"))),
            agility=max(
                0.0,
                self._apply_penalized_projection(
                    target.agility,
                    add["agility"],
                    mul["agility"],
                    penalty_context.get("agility"),
                ),
            ),
            warp_speed_au_s=max(0.0, float(getattr(target, "warp_speed_au_s", 0.0) or 0.0)),
            warp_capacitor_need=max(0.0, float(getattr(target, "warp_capacitor_need", 0.0) or 0.0)),
            max_warp_distance_au=max(0.0, float(getattr(target, "max_warp_distance_au", 0.0) or 0.0)),
            disallow_assistance=bool(getattr(target, "disallow_assistance", False)),
            warp_bubble_immune=bool(getattr(target, "warp_bubble_immune", False)),
            is_shuttle=bool(getattr(target, "is_shuttle", False)),
            sensor_strength_gravimetric=max(
                0.0,
                self._apply_penalized_projection(
                    target.sensor_strength_gravimetric,
                    add["sensor_strength_gravimetric"],
                    mul["sensor_strength_gravimetric"],
                    penalty_context.get("sensor_strength_gravimetric"),
                ),
            ),
            sensor_strength_ladar=max(
                0.0,
                self._apply_penalized_projection(
                    target.sensor_strength_ladar,
                    add["sensor_strength_ladar"],
                    mul["sensor_strength_ladar"],
                    penalty_context.get("sensor_strength_ladar"),
                ),
            ),
            sensor_strength_magnetometric=max(
                0.0,
                self._apply_penalized_projection(
                    target.sensor_strength_magnetometric,
                    add["sensor_strength_magnetometric"],
                    mul["sensor_strength_magnetometric"],
                    penalty_context.get("sensor_strength_magnetometric"),
                ),
            ),
            sensor_strength_radar=max(
                0.0,
                self._apply_penalized_projection(
                    target.sensor_strength_radar,
                    add["sensor_strength_radar"],
                    mul["sensor_strength_radar"],
                    penalty_context.get("sensor_strength_radar"),
                ),
            ),
            shield_resonance_em=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.shield_resonance_em, 0.0, mul["shield_resonance_em"], penalty_context.get("shield_resonance_em"))),
            ),
            shield_resonance_thermal=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.shield_resonance_thermal, 0.0, mul["shield_resonance_thermal"], penalty_context.get("shield_resonance_thermal"))),
            ),
            shield_resonance_kinetic=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.shield_resonance_kinetic, 0.0, mul["shield_resonance_kinetic"], penalty_context.get("shield_resonance_kinetic"))),
            ),
            shield_resonance_explosive=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.shield_resonance_explosive, 0.0, mul["shield_resonance_explosive"], penalty_context.get("shield_resonance_explosive"))),
            ),
            armor_resonance_em=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.armor_resonance_em, 0.0, mul["armor_resonance_em"], penalty_context.get("armor_resonance_em"))),
            ),
            armor_resonance_thermal=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.armor_resonance_thermal, 0.0, mul["armor_resonance_thermal"], penalty_context.get("armor_resonance_thermal"))),
            ),
            armor_resonance_kinetic=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.armor_resonance_kinetic, 0.0, mul["armor_resonance_kinetic"], penalty_context.get("armor_resonance_kinetic"))),
            ),
            armor_resonance_explosive=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.armor_resonance_explosive, 0.0, mul["armor_resonance_explosive"], penalty_context.get("armor_resonance_explosive"))),
            ),
            structure_resonance_em=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.structure_resonance_em, 0.0, mul["structure_resonance_em"], penalty_context.get("structure_resonance_em"))),
            ),
            structure_resonance_thermal=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.structure_resonance_thermal, 0.0, mul["structure_resonance_thermal"], penalty_context.get("structure_resonance_thermal"))),
            ),
            structure_resonance_kinetic=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.structure_resonance_kinetic, 0.0, mul["structure_resonance_kinetic"], penalty_context.get("structure_resonance_kinetic"))),
            ),
            structure_resonance_explosive=max(
                0.01,
                min(1.0, self._apply_penalized_projection(target.structure_resonance_explosive, 0.0, mul["structure_resonance_explosive"], penalty_context.get("structure_resonance_explosive"))),
            ),
        )

    @staticmethod
    def _cache_key(runtime: FitRuntime) -> tuple:
        hull = runtime.hull
        hull_signature = (
            hull.ship_name,
            hull.role,
            hull.base_dps,
            hull.volley,
            hull.optimal,
            hull.falloff,
            hull.tracking,
            hull.sig_radius,
            hull.scan_resolution,
            hull.max_target_range,
            hull.max_speed,
            hull.cap_max,
            hull.cap_recharge_time,
            hull.shield_hp,
            hull.armor_hp,
            hull.structure_hp,
            hull.rep_amount,
            hull.rep_cycle,
            hull.energy_warfare_resistance,
            hull.mass,
            hull.agility,
            hull.disallow_assistance,
            hull.warp_bubble_immune,
            hull.is_shuttle,
        )
        modules = tuple(
            (
                m.module_id,
                m.group,
                m.state.value,
                tuple(sorted(str(tag) for tag in m.tags)),
                tuple(
                    (
                        e.name,
                        e.effect_class.value,
                        e.state_required.value,
                        e.range_m,
                        e.falloff_m,
                        e.cycle_time,
                        e.cap_need,
                        e.reactivation_delay,
                        tuple(sorted(e.local_mult.items())),
                        tuple(sorted(e.local_add.items())),
                        tuple(sorted(e.projected_mult.items())),
                        tuple(sorted(e.projected_add.items())),
                        tuple(sorted(e.projected_mult_groups.items())),
                        e.projected_signature,
                    )
                    for e in m.effects
                ),
            )
            for m in runtime.modules
        )
        skills = tuple(sorted(runtime.skills.levels.items()))
        return hull_signature, modules, skills
