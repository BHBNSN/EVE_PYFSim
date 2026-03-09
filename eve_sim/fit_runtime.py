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


@dataclass(slots=True)
class ModuleRuntime:
    module_id: str
    group: str
    state: ModuleState
    effects: list[ModuleEffect] = field(default_factory=list)
    charge_capacity: int = 0
    charge_rate: float = 0.0
    charge_remaining: float = 0.0
    charge_reload_time: float = 0.0

    def is_active_for(self, required: ModuleState) -> bool:
        rank = {
            ModuleState.OFFLINE: 0,
            ModuleState.ONLINE: 1,
            ModuleState.ACTIVE: 2,
            ModuleState.OVERHEATED: 3,
        }
        return rank[self.state] >= rank[required]


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
        )
        self._cache[key] = profile
        return profile

    def apply_projected_effects(self, target: ShipProfile, impacts: list[ProjectedImpact]) -> ShipProfile:
        mul: dict[str, list[float]] = {
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
        }
        add: dict[str, float] = {k: 0.0 for k in mul}
        for impact in impacts:
            eff = impact.effect
            strength = max(0.0, min(1.0, float(impact.strength)))
            for k, v in eff.projected_mult.items():
                if k in mul:
                    mul[k].append(1.0 + (v - 1.0) * strength)
            for k, v in eff.projected_add.items():
                if k in add:
                    add[k] += v * strength

        return ShipProfile(
            dps=max(0.0, (target.dps + add["dps"]) * self._stacking_multiplier(mul["dps"])),
            volley=target.volley,
            optimal=max(1.0, (target.optimal + add["optimal"]) * self._stacking_multiplier(mul["optimal"])),
            falloff=max(1.0, (target.falloff + add["falloff"]) * self._stacking_multiplier(mul["falloff"])),
            tracking=max(0.0001, (target.tracking + add["tracking"]) * self._stacking_multiplier(mul["tracking"])),
            sig_radius=max(1.0, (target.sig_radius + add["sig"]) * self._stacking_multiplier(mul["sig"])),
            scan_resolution=max(1.0, (target.scan_resolution + add["scan"]) * self._stacking_multiplier(mul["scan"])),
            max_target_range=max(1000.0, (target.max_target_range + add["range"]) * self._stacking_multiplier(mul["range"])),
            max_speed=max(1.0, (target.max_speed + add["speed"]) * self._stacking_multiplier(mul["speed"])),
            missile_explosion_radius=max(
                0.0,
                target.missile_explosion_radius * self._stacking_multiplier(mul["missile_explosion_radius"]),
            ),
            missile_explosion_velocity=max(
                0.0,
                target.missile_explosion_velocity * self._stacking_multiplier(mul["missile_explosion_velocity"]),
            ),
            missile_max_range=max(
                0.0,
                target.missile_max_range * self._stacking_multiplier(mul["missile_range"]),
            ),
            max_cap=max(1.0, (target.max_cap + add["cap_max"]) * self._stacking_multiplier(mul["cap_max"])),
            cap_recharge_time=max(
                1.0,
                (target.cap_recharge_time + add["cap_recharge"]) * self._stacking_multiplier(mul["cap_recharge"]),
            ),
            shield_hp=max(1.0, (target.shield_hp + add["shield_hp"]) * self._stacking_multiplier(mul["shield_hp"])),
            armor_hp=max(1.0, (target.armor_hp + add["armor_hp"]) * self._stacking_multiplier(mul["armor_hp"])),
            structure_hp=max(1.0, (target.structure_hp + add["structure_hp"]) * self._stacking_multiplier(mul["structure_hp"])),
            rep_amount=max(0.0, (target.rep_amount + add["rep"]) * self._stacking_multiplier(mul["rep"])),
            rep_cycle=target.rep_cycle,
            sensor_strength_gravimetric=max(
                0.0,
                (target.sensor_strength_gravimetric + add["sensor_strength_gravimetric"])
                * self._stacking_multiplier(mul["sensor_strength_gravimetric"]),
            ),
            sensor_strength_ladar=max(
                0.0,
                (target.sensor_strength_ladar + add["sensor_strength_ladar"])
                * self._stacking_multiplier(mul["sensor_strength_ladar"]),
            ),
            sensor_strength_magnetometric=max(
                0.0,
                (target.sensor_strength_magnetometric + add["sensor_strength_magnetometric"])
                * self._stacking_multiplier(mul["sensor_strength_magnetometric"]),
            ),
            sensor_strength_radar=max(
                0.0,
                (target.sensor_strength_radar + add["sensor_strength_radar"])
                * self._stacking_multiplier(mul["sensor_strength_radar"]),
            ),
            shield_resonance_em=max(0.01, min(1.0, target.shield_resonance_em * self._stacking_multiplier(mul["shield_resonance_em"]))),
            shield_resonance_thermal=max(0.01, min(1.0, target.shield_resonance_thermal * self._stacking_multiplier(mul["shield_resonance_thermal"]))),
            shield_resonance_kinetic=max(0.01, min(1.0, target.shield_resonance_kinetic * self._stacking_multiplier(mul["shield_resonance_kinetic"]))),
            shield_resonance_explosive=max(0.01, min(1.0, target.shield_resonance_explosive * self._stacking_multiplier(mul["shield_resonance_explosive"]))),
            armor_resonance_em=max(0.01, min(1.0, target.armor_resonance_em * self._stacking_multiplier(mul["armor_resonance_em"]))),
            armor_resonance_thermal=max(0.01, min(1.0, target.armor_resonance_thermal * self._stacking_multiplier(mul["armor_resonance_thermal"]))),
            armor_resonance_kinetic=max(0.01, min(1.0, target.armor_resonance_kinetic * self._stacking_multiplier(mul["armor_resonance_kinetic"]))),
            armor_resonance_explosive=max(0.01, min(1.0, target.armor_resonance_explosive * self._stacking_multiplier(mul["armor_resonance_explosive"]))),
            structure_resonance_em=max(0.01, min(1.0, target.structure_resonance_em * self._stacking_multiplier(mul["structure_resonance_em"]))),
            structure_resonance_thermal=max(0.01, min(1.0, target.structure_resonance_thermal * self._stacking_multiplier(mul["structure_resonance_thermal"]))),
            structure_resonance_kinetic=max(0.01, min(1.0, target.structure_resonance_kinetic * self._stacking_multiplier(mul["structure_resonance_kinetic"]))),
            structure_resonance_explosive=max(0.01, min(1.0, target.structure_resonance_explosive * self._stacking_multiplier(mul["structure_resonance_explosive"]))),
        )

    @staticmethod
    def _cache_key(runtime: FitRuntime) -> tuple:
        modules = tuple(
            (
                m.module_id,
                m.group,
                m.state.value,
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
                    )
                    for e in m.effects
                ),
            )
            for m in runtime.modules
        )
        skills = tuple(sorted(runtime.skills.levels.items()))
        return runtime.fit_key, modules, skills


def build_profile_from_runtime(runtime: FitRuntime) -> ShipProfile:
    return RuntimeStatEngine().compute_base_profile(runtime)
