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


class MovementSystem:
    def __init__(self, battlefield_radius: float) -> None:
        self.battlefield_radius = battlefield_radius
        self._large_angle_threshold_deg = 45.0

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        while angle <= -180.0:
            angle += 360.0
        while angle > 180.0:
            angle -= 360.0
        return angle

    @staticmethod
    def _align_time_for(max_speed: float) -> float:
        speed = max(150.0, float(max_speed))
        return max(2.5, min(14.0, 14_000.0 / speed))

    @staticmethod
    def _heading_vector(angle_deg: float) -> Vector2:
        facing_rad = math.radians(angle_deg)
        return Vector2(math.cos(facing_rad), math.sin(facing_rad))

    @staticmethod
    def _motion_params(ship) -> tuple[float, float]:
        profile = getattr(ship, "profile", None)
        if profile is not None:
            try:
                mass = float(getattr(profile, "mass", 0.0) or 0.0)
                agility = float(getattr(profile, "agility", 0.0) or 0.0)
            except Exception:
                mass = 0.0
                agility = 0.0
            if mass > 0.0 and agility > 0.0:
                return mass, agility

        runtime = getattr(ship, "runtime", None)
        if runtime is not None:
            diagnostics = getattr(runtime, "diagnostics", None)
            if isinstance(diagnostics, dict):
                raw = diagnostics.get("motion_params")
                if isinstance(raw, dict):
                    mass_raw = raw.get("mass")
                    agility_raw = raw.get("agility")
                    try:
                        mass = float(mass_raw) if mass_raw is not None else 0.0
                        agility = float(agility_raw) if agility_raw is not None else 0.0
                    except Exception:
                        mass = 0.0
                        agility = 0.0
                    if mass > 0.0 and agility > 0.0:
                        return mass, agility
        return 0.0, 0.0

    @classmethod
    def _motion_tau(cls, ship, speed_cap: float) -> float:
        mass, agility = cls._motion_params(ship)
        if mass > 0.0 and agility > 0.0:
            return max(0.25, (mass * agility) / 1_000_000.0)
        return max(0.25, cls._align_time_for(speed_cap))

    @staticmethod
    def _exponential_velocity_step(current_velocity: Vector2, desired_velocity: Vector2, tau: float, dt: float) -> tuple[Vector2, Vector2]:
        tau = max(1e-6, float(tau))
        decay = math.exp(-float(dt) / tau)
        new_velocity = current_velocity * decay + desired_velocity * (1.0 - decay)
        displacement = desired_velocity * float(dt) + (current_velocity - desired_velocity) * (tau * (1.0 - decay))
        return new_velocity, displacement

    @staticmethod
    def _stable_turn_radius(speed: float, speed_cap: float, tau: float) -> float:
        orbit_speed = max(0.0, min(float(speed), float(speed_cap) * 0.999999))
        if orbit_speed <= 1e-6:
            return 0.0
        turn_budget = max(0.0, float(speed_cap) ** 2 - orbit_speed ** 2)
        if turn_budget <= 1e-9:
            return float("inf")
        return max(0.0, float(tau)) * orbit_speed * orbit_speed / math.sqrt(turn_budget)

    @classmethod
    def _stable_angular_velocity(cls, speed: float, speed_cap: float, tau: float) -> float:
        radius = cls._stable_turn_radius(speed, speed_cap, tau)
        if radius == 0.0:
            return float("inf")
        if math.isinf(radius):
            return 0.0
        return max(0.0, float(speed)) / radius

    def _effective_speed_cap(self, world: WorldState, ship) -> float:
        base_cap = max(1.0, float(ship.nav.max_speed))
        squad_key = f"{ship.team.value}:{ship.squad_id}"
        leader_id = world.squad_leaders.get(squad_key)
        if leader_id != ship.ship_id:
            return base_cap
        cap = float(world.squad_leader_speed_limits.get(squad_key, 0.0) or 0.0)
        if cap <= 0.0:
            return base_cap
        return max(1.0, min(base_cap, cap))

    def _update_velocity_with_inertia(self, world: WorldState, ship, dt: float) -> Vector2:
        target = ship.nav.command_target
        speed_cap = self._effective_speed_cap(world, ship)
        desired_angle = ship.nav.facing_deg
        target_speed = 0.0
        current_velocity = ship.nav.velocity
        current_speed = current_velocity.length()

        if target is not None:
            to_target = target - ship.nav.position
            distance = to_target.length()
            if distance > max(120.0, ship.nav.radius * 1.5):
                desired_angle = to_target.angle_deg()
                target_speed = speed_cap
        elif bool(ship.nav.propulsion_command_active):
            if current_speed > 1e-6:
                # Keep burning along the existing travel vector when propulsion toggles on mid-flight.
                desired_angle = current_velocity.angle_deg()
                target_speed = speed_cap

        tau = self._motion_tau(ship, speed_cap)
        desired_velocity = self._heading_vector(desired_angle) * target_speed
        new_velocity, displacement = self._exponential_velocity_step(current_velocity, desired_velocity, tau, dt)

        if target_speed > 1e-6 and current_speed > 1e-6:
            current_heading = current_velocity.angle_deg()
            desired_turn = abs(self._wrap_angle_deg(desired_angle - current_heading))
            if desired_turn <= self._large_angle_threshold_deg:
                new_speed = new_velocity.length()
                raw_heading = new_velocity.angle_deg() if new_speed > 1e-6 else desired_angle
                angular_velocity = self._stable_angular_velocity(max(current_speed, new_speed), speed_cap, tau)
                max_turn_step_deg = 180.0 if math.isinf(angular_velocity) else math.degrees(angular_velocity * dt)
                heading_delta = self._wrap_angle_deg(raw_heading - current_heading)
                if abs(heading_delta) > max_turn_step_deg:
                    capped_heading = self._wrap_angle_deg(
                        current_heading + max(-max_turn_step_deg, min(max_turn_step_deg, heading_delta))
                    )
                    new_velocity = self._heading_vector(capped_heading) * new_speed
                    displacement = (current_velocity + new_velocity) * (0.5 * dt)

        new_speed = new_velocity.length()
        if new_speed > speed_cap:
            new_velocity = new_velocity.normalized() * speed_cap
            displacement = (current_velocity + new_velocity) * (0.5 * dt)
            new_speed = speed_cap

        ship.nav.velocity = new_velocity
        ship.nav.facing_deg = new_velocity.angle_deg() if new_speed > 1e-6 else desired_angle
        return displacement

    def run(self, world: WorldState, dt: float) -> None:
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            profile_speed = float(getattr(ship.profile, "max_speed", ship.nav.max_speed) or ship.nav.max_speed)
            if profile_speed > 0.0:
                ship.nav.max_speed = profile_speed

            displacement = self._update_velocity_with_inertia(world, ship, dt)
            next_pos = ship.nav.position + displacement
            if next_pos.length() > self.battlefield_radius:
                n = next_pos.normalized()
                next_pos = n * self.battlefield_radius
                ship.nav.velocity = Vector2(0.0, 0.0)

            for beacon in world.beacons.values():
                dist = next_pos.distance_to(beacon.position)
                if dist < beacon.radius + ship.nav.radius:
                    push_dir = (next_pos - beacon.position).normalized()
                    if push_dir.length() == 0:
                        push_dir = Vector2(1.0, 0.0)
                    next_pos = beacon.position + push_dir * (beacon.radius + ship.nav.radius)

            ship.nav.position = next_pos

