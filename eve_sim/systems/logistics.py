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

from .models import *


class LogisticsSystem:
    @staticmethod
    def _apply_repair(target, amount: float) -> None:
        remaining = max(0.0, float(amount))
        if remaining <= 0.0:
            return
        missing_shield = max(0.0, float(target.vital.shield_max) - float(target.vital.shield))
        if missing_shield > 0.0:
            restored = min(remaining, missing_shield)
            target.vital.shield += restored
            remaining -= restored
        if remaining <= 0.0:
            return
        missing_armor = max(0.0, float(target.vital.armor_max) - float(target.vital.armor))
        if missing_armor > 0.0:
            target.vital.armor += min(remaining, missing_armor)

    def run(self, world: WorldState, dt: float) -> None:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        for ship in world.ships.values():
            if ship.vital.alive:
                alive_by_team[ship.team].append(ship)

        weakest_by_team: dict[Team, Any | None] = {Team.BLUE: None, Team.RED: None}
        for team, members in alive_by_team.items():
            if not members:
                weakest_by_team[team] = None
                continue
            weakest_by_team[team] = min(members, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            if ship.runtime is not None:
                # Runtime-backed fits already apply remote repair effects through CombatSystem.
                continue
            if ship.profile.rep_amount <= 0 or ship.profile.rep_cycle <= 0:
                continue

            target = weakest_by_team.get(ship.team)
            if target is None or target.ship_id == ship.ship_id:
                allies = [a for a in alive_by_team.get(ship.team, []) if a.ship_id != ship.ship_id]
                if not allies:
                    continue
                target = min(allies, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

            dist = ship.nav.position.distance_to(target.nav.position)
            if dist > ship.profile.max_target_range:
                continue

            repair = ship.profile.rep_amount * (dt / ship.profile.rep_cycle)
            self._apply_repair(target, repair)
