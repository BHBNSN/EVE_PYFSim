from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import logging
import random
import time
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
from ..models import ShipProfile, Team
from ..pyfa_bridge import PyfaBridge
from ..sim_logging import log_sim_event
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
)

_FORMULA_PROJECTED_KEYS = frozenset(
    {
        "speed",
        "sig",
        "tracking",
        "optimal",
        "falloff",
        "scan",
        "range",
        "rep",
        "shield_hp",
        "armor_hp",
        "structure_hp",
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
        "missile_range",
        "dps",
        "cap_max",
        "cap_recharge",
        "mass",
        "agility",
    }
)

_RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS = frozenset(
    {
        "runtime_module_static_metadata",
        "runtime_module_buckets",
        "runtime_controlled_module_ids",
        "runtime_controlled_entry_lookup",
        "runtime_decision_rule_groups",
    }
)



@dataclass(frozen=True, slots=True)
class ModuleDecisionRule:
    rule_id: str
    activation_mode: str
    target_mode: str
    cap_threshold: float = 0.0


@dataclass(slots=True)
class CycleTargetSnapshot:
    distance: float
    active_effect_indices: set[int] = field(default_factory=set)
    effect_strengths: dict[int, float] = field(default_factory=dict)
    effect_damage_factors: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModuleStaticMetadata:
    active_effects: tuple[ModuleEffect, ...]
    projected_effects: tuple[tuple[int, ModuleEffect], ...]
    cycle_cost: float
    cycle_time: float
    reactivation_delay: float
    has_projected: bool
    target_side: str
    is_command_burst: bool
    is_smart_bomb: bool
    is_burst_jammer: bool
    is_area_effect: bool
    is_weapon: bool
    has_projected_rep: bool
    is_cap_warfare: bool
    is_target_ewar: bool
    is_ecm: bool
    uses_pyfa_projected_profile: bool
    is_hardener: bool
    is_cap_booster: bool
    is_propulsion: bool
    is_damage_control: bool
    affects_local_pyfa_profile: bool
    repair_layers: tuple[str, ...]
    decision_rule: ModuleDecisionRule


@dataclass(frozen=True, slots=True)
class RuntimeModuleBuckets:
    module_count: int
    controlled_entries: tuple[tuple[Any, ModuleStaticMetadata], ...]
    command_entries: tuple[tuple[Any, ModuleStaticMetadata], ...]
    runtime_projected_entries: tuple[tuple[Any, ModuleStaticMetadata], ...]
    pyfa_projected_entries: tuple[tuple[Any, ModuleStaticMetadata], ...]


def _scale_damage(dmg: DamageTuple, factor: float) -> DamageTuple:
    return dmg[0] * factor, dmg[1] * factor, dmg[2] * factor, dmg[3] * factor


def _sum_damage(dmg: DamageTuple) -> float:
    return dmg[0] + dmg[1] + dmg[2] + dmg[3]


def _layer_effective_damage(dmg: DamageTuple, resonances: DamageTuple) -> float:
    return (
        dmg[0] * resonances[0]
        + dmg[1] * resonances[1]
        + dmg[2] * resonances[2]
        + dmg[3] * resonances[3]
    )


def _apply_damage_sequence(shield: float, armor: float, structure: float, dmg: DamageTuple, target_profile) -> tuple[float, float, float]:
    remaining = dmg
    layers = [
        ("shield", shield, (target_profile.shield_resonance_em, target_profile.shield_resonance_thermal, target_profile.shield_resonance_kinetic, target_profile.shield_resonance_explosive)),
        ("armor", armor, (target_profile.armor_resonance_em, target_profile.armor_resonance_thermal, target_profile.armor_resonance_kinetic, target_profile.armor_resonance_explosive)),
        ("structure", structure, (target_profile.structure_resonance_em, target_profile.structure_resonance_thermal, target_profile.structure_resonance_kinetic, target_profile.structure_resonance_explosive)),
    ]

    new_vals = {"shield": shield, "armor": armor, "structure": structure}
    for layer_name, layer_hp, layer_res in layers:
        if layer_hp <= 0:
            continue
        eff = _layer_effective_damage(remaining, layer_res)
        if eff <= 0:
            continue
        if eff <= layer_hp:
            new_vals[layer_name] = layer_hp - eff
            return new_vals["shield"], new_vals["armor"], new_vals["structure"]

        consumed_ratio = layer_hp / eff
        consumed_ratio = max(0.0, min(1.0, consumed_ratio))
        new_vals[layer_name] = 0.0
        remaining = _scale_damage(remaining, 1.0 - consumed_ratio)

    return new_vals["shield"], new_vals["armor"], new_vals["structure"]
