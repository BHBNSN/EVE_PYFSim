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



class PerceptionSystem:
    def __init__(self, sensor_range: float = 250_000.0) -> None:
        self.sensor_range = sensor_range

    def run(self, world: WorldState) -> None:
        alive = [s for s in world.ships.values() if s.vital.alive]
        if not alive:
            return
        if len(alive) <= 24:
            sensor = self.sensor_range
            for source in alive:
                source.perception = [
                    target.ship_id
                    for target in alive
                    if target.ship_id != source.ship_id
                    and source.nav.position.distance_to(target.nav.position) <= sensor
                ]
            return

        min_x = max_x = float(alive[0].nav.position.x)
        min_y = max_y = float(alive[0].nav.position.y)
        for ship in alive[1:]:
            pos = ship.nav.position
            min_x = min(min_x, float(pos.x))
            max_x = max(max_x, float(pos.x))
            min_y = min(min_y, float(pos.y))
            max_y = max(max_y, float(pos.y))
        if math.hypot(max_x - min_x, max_y - min_y) <= self.sensor_range:
            alive_ids = [ship.ship_id for ship in alive]
            for index, ship in enumerate(alive):
                ship.perception = alive_ids[:index] + alive_ids[index + 1:]
            return

        pos_array = np.array([(s.nav.position.x, s.nav.position.y) for s in alive], dtype=np.float64)
        delta = pos_array[:, None, :] - pos_array[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        for i, ship in enumerate(alive):
            mask = (dist[i] <= self.sensor_range) & (dist[i] > 0)
            ship.perception = [alive[j].ship_id for j in np.where(mask)[0].tolist()]

