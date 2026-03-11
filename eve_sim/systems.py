from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import logging
import random
import time
from typing import Any

import numpy as np

from .fleet_setup import (
    _module_affects_local_pyfa_profile,
    _runtime_local_profile_state_signature,
    get_runtime_resolve_cache_key,
    resolve_runtime_from_pyfa_runtime,
)
from .fit_runtime import EffectClass, ModuleEffect, ModuleState, ProjectedImpact, RuntimeStatEngine
from .math2d import Vector2
from .models import ShipProfile, Team
from .pyfa_bridge import PyfaBridge
from .sim_logging import log_sim_event
from .world import WorldState


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


@dataclass(frozen=True, slots=True)
class ModuleDecisionRule:
    rule_id: str
    activation_mode: str
    target_mode: str
    cap_threshold: float = 0.0


@dataclass(slots=True)
class CycleTargetSnapshot:
    distance: float
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
    decision_rule: ModuleDecisionRule


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

        pos = np.array([(s.nav.position.x, s.nav.position.y) for s in alive], dtype=np.float64)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        for i, ship in enumerate(alive):
            mask = (dist[i] <= self.sensor_range) & (dist[i] > 0)
            ship.perception = [alive[j].ship_id for j in np.where(mask)[0].tolist()]


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
        self._module_static_metadata_by_object_id: dict[int, ModuleStaticMetadata] = {}

    def attach_logger(
        self,
        logger: logging.Logger,
        detailed_logging: bool,
        merge_window_sec: float = 1.0,
        hotspot_logging: bool = False,
    ) -> None:
        self.logger = logger
        self.event_logging_enabled = bool(detailed_logging)
        self.detailed_logging = False
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

    def _apply_runtime_projected_impacts(self, base: ShipProfile, impacts: list[ProjectedImpact]) -> ShipProfile:
        effective = self.runtime.apply_projected_effects(replace(base), impacts)
        self._copy_profile_passthrough_fields(base, effective)
        return effective

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
        cached = runtime.diagnostics.get("runtime_local_state_signature")
        if isinstance(cached, tuple):
            return cached
        signature = tuple(
            (str(module.module_id), str(module.state.value or "ONLINE").upper())
            for module in runtime.modules
            if self._module_static_metadata(module).affects_local_pyfa_profile
        )
        runtime.diagnostics["runtime_local_state_signature"] = signature
        return signature

    def _mark_pyfa_remote_inputs_dirty(self) -> None:
        self._pyfa_remote_inputs_dirty = True

    def _refresh_alive_runtime_ship_ids(self, world: WorldState) -> None:
        current_alive_runtime_ship_ids = {
            ship.ship_id
            for ship in world.ships.values()
            if ship.vital.alive and ship.runtime is not None
        }
        if current_alive_runtime_ship_ids != self._alive_runtime_ship_ids:
            self._alive_runtime_ship_ids = current_alive_runtime_ship_ids
            self._mark_pyfa_remote_inputs_dirty()

    def _cached_pyfa_remote_inputs_available(self) -> bool:
        return self._cached_command_booster_snapshots is not None and self._cached_projected_source_snapshots is not None

    def _module_static_metadata(self, module) -> ModuleStaticMetadata:
        key = id(module)
        cached = self._module_static_metadata_by_object_id.get(key)
        if cached is not None:
            return cached

        group_name = str(getattr(module, "group", "") or "").strip().lower()
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
        has_projected_damage = False
        has_projected_rep = False
        friendly_score = 0
        hostile_score = 0
        for _effect_index, effect in projected_effects:
            for key_name, value in effect.projected_add.items():
                amount = float(value or 0.0)
                if amount <= 0.0:
                    continue
                if key_name in {"shield_rep", "armor_rep"}:
                    has_projected_rep = True
                    friendly_score += 2
                elif key_name in {"cap_drain", "ecm_gravimetric", "ecm_ladar", "ecm_magnetometric", "ecm_radar"}:
                    hostile_score += 2
                elif key_name.startswith("damage_"):
                    has_projected_damage = True
                    hostile_score += 3
                elif key_name.startswith("weapon_"):
                    hostile_score += 1
            for value in effect.projected_mult.values():
                mult = float(value or 0.0)
                if mult < 1.0:
                    hostile_score += 1
                elif mult > 1.0:
                    friendly_score += 1

        target_side = "ally" if friendly_score > hostile_score else "enemy"
        is_command_burst = group_name == "command burst"
        is_smart_bomb = group_name in {"smart bomb", "structure area denial module"}
        is_burst_jammer = group_name == "burst jammer"
        is_area_effect = is_command_burst or is_smart_bomb or is_burst_jammer
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
        is_cap_warfare = any(token in group_name for token in ("energy neutral", "nosferatu"))
        is_target_ewar = any(token in group_name for token in ("target painter", "stasis web", "stasis grappler"))
        is_ecm = "ecm" in group_name
        ammo_like = int(getattr(module, "charge_capacity", 0) or 0) > 0 and float(getattr(module, "charge_rate", 0.0) or 0.0) > 0.0
        is_weapon = has_projected and has_projected_damage and (("weapon" in group_name) or ("missile launcher" in group_name) or ammo_like)
        is_offensive_ewar = (
            has_projected
            and not is_weapon
            and target_side == "enemy"
            and any(
                token in group_name
                for token in (
                    "weapon disruptor",
                    "sensor damp",
                    "energy neutral",
                    "nosferatu",
                    "ecm",
                    "warp scrambler",
                )
            )
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
                    target_mode="ally_lowest_hp",
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
            decision_rule=decision_rule,
        )
        self._module_static_metadata_by_object_id[key] = metadata
        return metadata

    def _runtime_module_metadata_list(self, runtime) -> tuple[ModuleStaticMetadata, ...]:
        cached = runtime.diagnostics.get("runtime_module_static_metadata")
        if isinstance(cached, tuple) and len(cached) == len(runtime.modules):
            return cached
        metadata_list = tuple(self._module_static_metadata(module) for module in runtime.modules)
        runtime.diagnostics["runtime_module_static_metadata"] = metadata_list
        return metadata_list

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
                    tuple(sorted((str(module_id), str(state)) for module_id, state in state_by_module_id.items())),
                )
            )
        return tuple(signature)

    @staticmethod
    def _formula_effect_signature(effect_payload: dict[str, Any]) -> tuple[Any, ...]:
        mult_raw = effect_payload.get("projected_mult")
        add_raw = effect_payload.get("projected_add")
        projected_mult: dict[str, Any] = mult_raw if isinstance(mult_raw, dict) else {}
        projected_add: dict[str, Any] = add_raw if isinstance(add_raw, dict) else {}
        return (
            str(effect_payload.get("name", "") or ""),
            tuple(sorted((str(key), round(float(value or 0.0), 6)) for key, value in projected_mult.items())),
            tuple(sorted((str(key), round(float(value or 0.0), 6)) for key, value in projected_add.items())),
        )

    @classmethod
    def _projected_snapshot_list_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        signature: list[tuple[Any, ...]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            state_raw = snapshot.get("state_by_module_id")
            state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
            command_raw = snapshot.get("command_booster_snapshots")
            command_snapshots = [snap for snap in command_raw if isinstance(snap, dict)] if isinstance(command_raw, list) else []
            formula_raw = snapshot.get("formula_effects")
            formula_effects = [raw for raw in formula_raw if isinstance(raw, dict)] if isinstance(formula_raw, list) else []
            distance_mode = str(snapshot.get("distance_mode", "pyfa_range") or "pyfa_range")
            if distance_mode == "formula" and formula_effects:
                distance_signature: Any = tuple(round(float(raw.get("strength", 1.0) or 1.0), 6) for raw in formula_effects)
            elif distance_mode == "pyfa_range":
                try:
                    distance_signature = round(float(snapshot.get("pyfa_projection_range", snapshot.get("projection_range", 0.0)) or 0.0), 3)
                except Exception:
                    distance_signature = 0.0
            else:
                distance_signature = None
            signature.append(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple(sorted((str(module_id), str(state)) for module_id, state in state_by_module_id.items())),
                    cls._command_snapshot_list_signature(command_snapshots),
                    distance_mode,
                    tuple(cls._formula_effect_signature(raw) for raw in formula_effects),
                    distance_signature,
                )
            )
        return tuple(signature)

    def _module_affects_pyfa_remote_inputs(self, module) -> bool:
        metadata = self._module_static_metadata(module)
        return metadata.is_command_burst or metadata.uses_pyfa_projected_profile

    def _runtime_has_active_pyfa_remote_inputs(self, runtime) -> bool:
        cached = runtime.diagnostics.get("runtime_has_active_pyfa_remote_inputs")
        if isinstance(cached, bool):
            return cached
        for module in runtime.modules:
            if str(module.state.value or "ONLINE").upper() not in {"ACTIVE", "OVERHEATED"}:
                continue
            if self._module_affects_pyfa_remote_inputs(module):
                runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = True
                return True
        runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = False
        return False

    @classmethod
    def _projected_snapshot_structure_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(item[:-1] for item in cls._projected_snapshot_list_signature(snapshots))

    @staticmethod
    def _projected_effect_supports_formula_profile(effect) -> bool:
        mult_keys = {str(key) for key in effect.projected_mult.keys()}
        add_keys = {str(key) for key in effect.projected_add.keys()}
        if not mult_keys and not add_keys:
            return False
        return mult_keys.issubset(_FORMULA_PROJECTED_KEYS) and add_keys.issubset(_FORMULA_PROJECTED_KEYS)

    def _projected_distance_mode(self, module) -> str:
        projected_effects = [effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED]
        if not projected_effects:
            return "pyfa_range"
        if all(max(0.0, float(getattr(effect, "falloff_m", 0.0) or 0.0)) <= 0.0 for effect in projected_effects):
            return "constant"
        if all(self._projected_effect_supports_formula_profile(effect) for effect in projected_effects):
            return "formula"
        return "pyfa_range"

    def _module_formula_effect_payloads(self, module, target_snapshot: CycleTargetSnapshot) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for effect_index, effect in enumerate(module.effects):
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            if not self._projected_effect_supports_formula_profile(effect):
                continue
            strength = max(0.0, min(1.0, float(target_snapshot.effect_strengths.get(effect_index, 0.0) or 0.0)))
            if strength <= 0.0:
                continue
            payloads.append(
                {
                    "name": str(effect.name or ""),
                    "projected_mult": dict(effect.projected_mult),
                    "projected_add": dict(effect.projected_add),
                    "strength": strength,
                }
            )
        return payloads

    def _formula_impacts_from_projected_snapshots(self, snapshots: list[dict[str, Any]]) -> list[ProjectedImpact] | None:
        impacts: list[ProjectedImpact] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                return None
            raw_effects = snapshot.get("formula_effects")
            if not isinstance(raw_effects, list) or not raw_effects:
                return None
            source_ship_id = str(snapshot.get("fit_key", "") or "")
            for raw_effect in raw_effects:
                if not isinstance(raw_effect, dict):
                    return None
                mult_raw = raw_effect.get("projected_mult")
                add_raw = raw_effect.get("projected_add")
                projected_mult: dict[str, Any] = mult_raw if isinstance(mult_raw, dict) else {}
                projected_add: dict[str, Any] = add_raw if isinstance(add_raw, dict) else {}
                strength = max(0.0, min(1.0, float(raw_effect.get("strength", 0.0) or 0.0)))
                if strength <= 0.0:
                    continue
                impacts.append(
                    ProjectedImpact(
                        source_ship_id=source_ship_id,
                        target_ship_id="",
                        effect=ModuleEffect(
                            name=str(raw_effect.get("name", "") or ""),
                            effect_class=EffectClass.PROJECTED,
                            state_required=ModuleState.ACTIVE,
                            projected_mult={str(key): float(value) for key, value in projected_mult.items()},
                            projected_add={str(key): float(value) for key, value in projected_add.items()},
                        ),
                        strength=strength,
                    )
                )
        return impacts or None

    def _remember_projection_formula_base(
        self,
        runtime,
        profile: ShipProfile,
        local_signature: tuple[tuple[str, str], ...] | None,
        booster_signature: tuple[Any, ...],
        projected_snapshots: list[dict[str, Any]],
    ) -> None:
        if projected_snapshots:
            return
        runtime.diagnostics["pyfa_projection_formula_base_profile"] = profile
        runtime.diagnostics["pyfa_projection_formula_local_signature"] = local_signature
        runtime.diagnostics["pyfa_projection_formula_command_signature"] = booster_signature

    def _recompute_profile_from_formula_base(
        self,
        runtime,
        local_signature: tuple[tuple[str, str], ...] | None,
        booster_signature: tuple[Any, ...],
        projected_snapshots: list[dict[str, Any]],
        *,
        tick: int,
        ship_id: str,
    ) -> ShipProfile | None:
        baseline = runtime.diagnostics.get("pyfa_projection_formula_base_profile")
        baseline_local_signature = runtime.diagnostics.get("pyfa_projection_formula_local_signature")
        baseline_command_signature = runtime.diagnostics.get("pyfa_projection_formula_command_signature")
        if not isinstance(baseline, ShipProfile):
            return None
        if baseline_local_signature != local_signature:
            return None
        if baseline_command_signature != booster_signature:
            return None
        impacts = self._formula_impacts_from_projected_snapshots(projected_snapshots)
        if impacts is None:
            return None
        started = time.perf_counter()
        profile = self._apply_runtime_projected_impacts(replace(baseline), impacts)
        self._log_hotspot(
            "combat.projected_formula_profile",
            started,
            tick=tick,
            ship=ship_id,
            projected_sources=len(projected_snapshots),
        )
        return profile

    @staticmethod
    def _copy_runtime_dynamic_state(source_runtime, target_runtime) -> None:
        if len(source_runtime.modules) == len(target_runtime.modules):
            for source_module, target_module in zip(source_runtime.modules, target_runtime.modules):
                target_module.module_id = source_module.module_id
                target_module.state = source_module.state
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
            module.state = source_module.state
            if module.charge_capacity > 0:
                module.charge_remaining = max(0.0, min(float(source_module.charge_remaining), float(module.charge_capacity)))

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
        for module, metadata in zip(runtime.modules, self._runtime_module_metadata_list(runtime)):
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

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            for timers in (
                ship.combat.lock_timers,
                ship.combat.module_cycle_timers,
                ship.combat.module_reactivation_timers,
                ship.combat.module_ammo_reload_timers,
                ship.combat.module_pending_ammo_reload_timers,
            ):
                for remaining in timers.values():
                    note_duration(remaining)

            for ready_at in ship.combat.fire_delay_timers.values():
                note_duration(float(ready_at) - now)

            for jam_until in ship.combat.ecm_jam_sources.values():
                note_duration(float(jam_until) - now)

            last_enemy_damage = float(getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9) or -1e9)
            note_duration(30.0 - (now - last_enemy_damage))

        for timers in world.squad_prelock_timers.values():
            for remaining in timers.values():
                note_duration(remaining)

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
            relative_speed = (source.nav.velocity - target.nav.velocity).length()
            explosion_radius = max(0.0, float(effect.projected_add.get("weapon_explosion_radius", 0.0) or 0.0))
            explosion_velocity = max(0.0, float(effect.projected_add.get("weapon_explosion_velocity", 0.0) or 0.0))
            drf = max(0.1, float(effect.projected_add.get("weapon_drf", 0.5) or 0.5))
            if explosion_radius > 0.0:
                sig_factor = target_profile.sig_radius / max(1.0, explosion_radius)
                vel_term = (sig_factor * explosion_velocity) / max(1.0, relative_speed)
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
    ) -> None:
        metadata = self._module_static_metadata(module)
        snapshot_key = self._module_cycle_snapshot_key(source.ship_id, module.module_id)
        projected_effects = metadata.projected_effects
        if not projected_effects:
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
                    target_snapshot.effect_strengths[effect_index] = max(0.0, min(1.0, strength))

            if target_snapshot.effect_strengths:
                self._module_cycle_target_snapshots[snapshot_key] = {target.ship_id: target_snapshot}
            else:
                self._module_cycle_target_snapshots.pop(snapshot_key, None)
            return

        target_snapshots: dict[str, CycleTargetSnapshot] = {}

        for _effect_index, effect in projected_effects:
            for target in self._iter_area_targets_in_range(world, source, module, effect):
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
                    target_snapshot.effect_strengths[effect_index] = max(0.0, min(1.0, strength))

        filtered = {
            target_id: snapshot
            for target_id, snapshot in target_snapshots.items()
            if snapshot.effect_strengths
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
        ship.vital.shield_max = max(1.0, float(getattr(profile, "shield_hp", ship.vital.shield_max) or ship.vital.shield_max))
        ship.vital.armor_max = max(1.0, float(getattr(profile, "armor_hp", ship.vital.armor_max) or ship.vital.armor_max))
        ship.vital.structure_max = max(1.0, float(getattr(profile, "structure_hp", ship.vital.structure_max) or ship.vital.structure_max))
        self._clamp_ship_layer_hp(ship)

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
    ) -> bool:
        if not target_id or target is None or not target.vital.alive:
            if target_id:
                ship.combat.lock_targets.discard(target_id)
                ship.combat.lock_timers.pop(target_id, None)
            return False
        now = float(world.now)
        if not self._can_target_under_ecm(ship, target_id, now):
            ship.combat.lock_targets.discard(target_id)
            ship.combat.lock_timers.pop(target_id, None)
            return False
        if target_id in ship.combat.lock_targets:
            return True
        if target_id not in ship.combat.lock_timers:
            profile_for_lock = target_profile if target_profile is not None else target.profile
            ship.combat.lock_timers[target_id] = self._cached_lock_time(ship.profile, profile_for_lock)
            if self.detailed_logging and self.logger is not None:
                self.logger.debug(
                    f"{lock_context}_start source={ship.ship_id} target={target_id} lock_time={ship.combat.lock_timers[target_id]:.2f}"
                )
        return False

    def _advance_target_locks(self, world: WorldState, dt: float) -> None:
        now = float(world.now)
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            for target_id, left in list(ship.combat.lock_timers.items()):
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive or not self._can_target_under_ecm(ship, target_id, now):
                    ship.combat.lock_timers.pop(target_id, None)
                    ship.combat.lock_targets.discard(target_id)
                    continue
                left -= dt
                if left <= 0.0:
                    ship.combat.lock_targets.add(target_id)
                    ship.combat.lock_timers.pop(target_id, None)
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(f"lock_complete attacker={ship.ship_id} target={target_id}")
                else:
                    ship.combat.lock_timers[target_id] = left
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(
                            f"lock_progress attacker={ship.ship_id} target={target_id} remaining={left:.2f}"
                        )

    @staticmethod
    def _projected_max_range(effect) -> float:
        if effect.falloff_m > 0.0:
            return max(0.0, effect.range_m) + 3.0 * max(0.0, effect.falloff_m)
        return max(0.0, effect.range_m)

    def _projected_strength(self, effect, distance: float) -> float:
        if effect.range_m > 0 or effect.falloff_m > 0:
            return self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
        return 1.0

    def _collect_projected_impacts(self, world: WorldState, dt: float) -> dict[str, list[ProjectedImpact]]:
        del dt
        impacts: dict[str, list[ProjectedImpact]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None:
                continue
            for module, metadata in zip(source.runtime.modules, self._runtime_module_metadata_list(source.runtime)):
                if metadata.uses_pyfa_projected_profile:
                    continue
                for effect_index, effect in metadata.projected_effects:
                    if not module.is_active_for(effect.state_required):
                        continue

                    target_id = source.combat.projected_targets.get(module.module_id)
                    if not target_id:
                        continue
                    target = world.ships.get(target_id)
                    if target is None or not target.vital.alive:
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
                    strength = max(0.0, float(target_snapshot.effect_strengths.get(effect_index, 0.0) or 0.0))
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
        return [candidate for candidate in candidates if candidate.vital.alive and self._module_in_projected_range(source, candidate, module)]

    def _iter_area_targets_in_range(self, world: WorldState, source, module, effect) -> list:
        targets: list = []
        metadata = self._module_static_metadata(module)
        include_self = metadata.is_command_burst
        same_team_only = metadata.is_command_burst
        max_range = self._projected_max_range(effect)

        for candidate in world.ships.values():
            if not candidate.vital.alive:
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
            if not source.vital.alive or source.runtime is None:
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue

            command_modules = [
                module
                for module in source.runtime.modules
                if self._module_static_metadata(module).is_command_burst
            ]
            if not command_modules:
                continue

            base_state_by_module_id: dict[str, str] = {}
            active_state_by_module_id: dict[str, str] = {}
            active_targets_by_module_id: dict[str, set[str]] = {}
            covered_targets: set[str] = set()

            for module in command_modules:
                state_value = str(module.state.value or "ONLINE").upper()
                base_state_by_module_id[module.module_id] = "ONLINE" if state_value in {"ACTIVE", "OVERHEATED"} else state_value
                if state_value not in {"ACTIVE", "OVERHEATED"}:
                    continue

                target_ids = {
                    target_id
                    for target_id in self._module_cycle_snapshots_for(source.ship_id, module.module_id)
                    if (target := world.ships.get(target_id)) is not None and target.vital.alive and target.team == source.team and target.runtime is not None
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
            if not source.vital.alive or source.runtime is None:
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue

            source_command_snapshots = command_boosters_by_ship.get(source.ship_id, [])
            base_state_by_module_id: dict[str, str] = {}
            active_projected_modules: list[tuple[Any, str]] = []

            for module, metadata in zip(source.runtime.modules, self._runtime_module_metadata_list(source.runtime)):
                state_value = str(module.state.value or "ONLINE").upper()
                projected_state = state_value

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
                if target is None or not target.vital.alive or target.runtime is None:
                    continue
                target_snapshot = self._module_cycle_snapshot_for_target(source.ship_id, active_projected_module.module_id, target_id)
                if target_snapshot is None:
                    continue
                distance_mode = self._projected_distance_mode(active_projected_module)

                state_by_module_id = dict(base_state_by_module_id)
                state_by_module_id[active_projected_module.module_id] = active_state
                snapshots_by_ship.setdefault(target_id, []).append(
                    {
                        "fit_key": f"{source.runtime.fit_key}:{active_projected_module.module_id}",
                        "blueprint": blueprint,
                        "state_by_module_id": state_by_module_id,
                        "command_booster_snapshots": source_command_snapshots,
                        "distance_mode": distance_mode,
                        "formula_effects": self._module_formula_effect_payloads(active_projected_module, target_snapshot),
                        "pyfa_projection_range": target_snapshot.distance,
                        "projection_range": target_snapshot.distance,
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
            local_signature = self._local_runtime_state_signature_from_metadata(ship.runtime)
            booster_signature = self._command_snapshot_list_signature(booster_snapshots)
            projected_signature = self._projected_snapshot_list_signature(projected_snapshots)
            projected_structure_signature = self._projected_snapshot_structure_signature(projected_snapshots)
            cached_signature = ship.runtime.diagnostics.get("pyfa_resolve_signature")
            cached_base_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
            cached_local_signature = ship.runtime.diagnostics.get("pyfa_local_state_signature")
            cached_command_signature = ship.runtime.diagnostics.get("pyfa_command_booster_signature")
            cached_projected_signature = ship.runtime.diagnostics.get("pyfa_projected_sources_signature")
            cached_projected_structure_signature = ship.runtime.diagnostics.get("pyfa_projected_sources_structure_signature")
            if (
                local_signature is not None
                and cached_local_signature == local_signature
                and cached_command_signature == booster_signature
                and cached_projected_signature == projected_signature
                and isinstance(cached_base_profile, ShipProfile)
            ):
                ship.runtime.diagnostics["pyfa_command_boosters"] = booster_snapshots
                ship.runtime.diagnostics["pyfa_projected_sources"] = projected_snapshots
                ship.runtime.diagnostics["pyfa_command_booster_signature"] = booster_signature
                ship.runtime.diagnostics["pyfa_projected_sources_signature"] = projected_signature
                ship.runtime.diagnostics["pyfa_projected_sources_structure_signature"] = projected_structure_signature
                ship.profile = cached_base_profile
                self._remember_projection_formula_base(ship.runtime, ship.profile, local_signature, booster_signature, projected_snapshots)
                continue

            formula_profile = None
            if projected_snapshots:
                formula_profile = self._recompute_profile_from_formula_base(
                    ship.runtime,
                    local_signature,
                    booster_signature,
                    projected_snapshots,
                    tick=int(world.tick),
                    ship_id=ship.ship_id,
                )
            if formula_profile is not None:
                ship.runtime.diagnostics.pop("pyfa_resolve_signature", None)
                if local_signature is not None:
                    ship.runtime.diagnostics["pyfa_local_state_signature"] = local_signature
                ship.runtime.diagnostics["pyfa_command_boosters"] = booster_snapshots
                ship.runtime.diagnostics["pyfa_projected_sources"] = projected_snapshots
                ship.runtime.diagnostics["pyfa_command_booster_signature"] = booster_signature
                ship.runtime.diagnostics["pyfa_projected_sources_signature"] = projected_signature
                ship.runtime.diagnostics["pyfa_projected_sources_structure_signature"] = projected_structure_signature
                ship.runtime.diagnostics["pyfa_base_profile"] = formula_profile
                ship.profile = formula_profile
                continue

            cache_key = get_runtime_resolve_cache_key(ship.runtime, booster_snapshots, projected_snapshots)
            if cache_key is not None and cached_signature == cache_key and isinstance(cached_base_profile, ShipProfile):
                if local_signature is not None:
                    ship.runtime.diagnostics["pyfa_local_state_signature"] = local_signature
                ship.runtime.diagnostics["pyfa_command_boosters"] = booster_snapshots
                ship.runtime.diagnostics["pyfa_projected_sources"] = projected_snapshots
                ship.runtime.diagnostics["pyfa_command_booster_signature"] = booster_signature
                ship.runtime.diagnostics["pyfa_projected_sources_signature"] = projected_signature
                ship.runtime.diagnostics["pyfa_projected_sources_structure_signature"] = projected_structure_signature
                ship.profile = cached_base_profile
                self._remember_projection_formula_base(ship.runtime, ship.profile, local_signature, booster_signature, projected_snapshots)
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
                continue

            resolved_runtime, resolved_profile = resolved
            resolved_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

            for index, pending in enumerate(pending_group):
                source_runtime = pending["runtime"]
                ship = pending["ship"]
                target_runtime = resolved_runtime if index == 0 else deepcopy(resolved_runtime)
                target_runtime.fit_key = source_runtime.fit_key
                minimum_potential_cycle_time = self._runtime_minimum_potential_cycle_time(source_runtime)

                blueprint = source_runtime.diagnostics.get("pyfa_blueprint")
                if isinstance(blueprint, dict):
                    target_runtime.diagnostics["pyfa_blueprint"] = deepcopy(blueprint)

                if pending["cache_key"] is not None:
                    target_runtime.diagnostics["pyfa_resolve_signature"] = pending["cache_key"]
                else:
                    target_runtime.diagnostics.pop("pyfa_resolve_signature", None)

                if pending["local_signature"] is not None:
                    target_runtime.diagnostics["pyfa_local_state_signature"] = pending["local_signature"]
                else:
                    target_runtime.diagnostics.pop("pyfa_local_state_signature", None)
                target_runtime.diagnostics["runtime_local_state_signature"] = pending["local_signature"]
                target_runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = self._runtime_has_active_pyfa_remote_inputs(source_runtime)
                target_runtime.diagnostics["runtime_minimum_potential_cycle_signature"] = self._runtime_offline_module_signature(source_runtime)
                target_runtime.diagnostics["runtime_minimum_potential_cycle_time"] = minimum_potential_cycle_time

                target_runtime.diagnostics["pyfa_command_boosters"] = pending["booster_snapshots"]
                target_runtime.diagnostics["pyfa_projected_sources"] = pending["projected_snapshots"]
                target_runtime.diagnostics["pyfa_command_booster_signature"] = pending["booster_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_signature"] = pending["projected_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_structure_signature"] = self._projected_snapshot_structure_signature(pending["projected_snapshots"])
                target_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

                self._copy_runtime_dynamic_state(source_runtime, target_runtime)
                self._remember_projection_formula_base(
                    target_runtime,
                    resolved_profile,
                    pending["local_signature"],
                    pending["booster_signature"],
                    pending["projected_snapshots"],
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
        if not self._can_target_under_ecm(source, target_id, float(world.now)):
            return False

        if rule.target_mode == "weapon_focus_prefocus":
            focus_queue = world.squad_focus_queues.get(self._focus_key(source.team, source.squad_id), [])
            if not focus_queue:
                return False
            allowed_ids: set[str] = {str(focus_queue[0])}
            if len(focus_queue) > 1:
                allowed_ids.add(str(focus_queue[1]))
            return target_id in allowed_ids and target_id in enemy_ids

        if rule.target_mode == "ally_lowest_hp":
            if target_id == source.ship_id:
                return False
            return self._is_lowest_hp_ally_target(source, module, allies_pool, target_id)

        if rule.target_mode in {"enemy_random", "enemy_nearest"}:
            return target_id in enemy_ids

        side = self._module_target_side(module)
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
            last_hit_at = float(getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9) or -1e9)
            return (float(world.now) - last_hit_at) <= 30.0
        if rule.activation_mode == "enemy_in_area":
            return self._module_has_area_enemies_in_range(world, ship, module)
        if rule.activation_mode == "weapon_focus_only":
            if not target_id:
                return False
            return self._weapon_fire_delay_ready(ship, target_id, float(world.now))
        return True

    @staticmethod
    def _module_target_side(module) -> str:
        friendly_score = 0
        hostile_score = 0
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            for key, value in effect.projected_add.items():
                amount = float(value or 0.0)
                if amount <= 0.0:
                    continue
                if key in {"shield_rep", "armor_rep"}:
                    friendly_score += 2
                elif key in {"cap_drain", "ecm_gravimetric", "ecm_ladar", "ecm_magnetometric", "ecm_radar"}:
                    hostile_score += 2
                elif key.startswith("damage_"):
                    hostile_score += 3
                elif key.startswith("weapon_"):
                    hostile_score += 1
            for value in effect.projected_mult.values():
                mult = float(value or 0.0)
                if mult < 1.0:
                    hostile_score += 1
                elif mult > 1.0:
                    friendly_score += 1
        if friendly_score > hostile_score:
            return "ally"
        return "enemy"

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
        if rule.target_mode == "ally_lowest_hp":
            return self._select_ally_lowest_hp_in_range(source, module, allies_pool, existing_target_id)
        if rule.target_mode == "enemy_random":
            return self._select_enemy_random_in_range(source, module, enemies_pool, existing_target_id)
        if rule.target_mode == "enemy_nearest":
            return self._select_enemy_nearest_in_range(source, module, enemies_pool, existing_target_id)

        side = self._module_target_side(module)
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
                ship.combat.lock_timers.pop(target_id, None)
        for target_id in list(ship.combat.fire_delay_timers.keys()):
            if target_id not in active_sources:
                ship.combat.fire_delay_timers.pop(target_id, None)
        for module_id, target_id in list(ship.combat.projected_targets.items()):
            if target_id not in active_sources:
                ship.combat.projected_targets.pop(module_id, None)
        if ship.combat.current_target and ship.combat.current_target not in active_sources:
            ship.combat.current_target = None

    def _update_ecm_restrictions(self, world: WorldState) -> None:
        now = float(world.now)
        for ship in world.ships.values():
            if not ship.vital.alive:
                ship.combat.ecm_jam_sources.clear()
                continue
            self._enforce_ecm_restrictions(ship, now)

    @staticmethod
    def _break_all_locks(ship) -> None:
        ship.combat.lock_targets.clear()
        ship.combat.lock_timers.clear()
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

    def _update_module_states(self, world: WorldState, dt: float) -> bool:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        alive_ids_by_team: dict[Team, set[str]] = {Team.BLUE: set(), Team.RED: set()}
        for candidate in world.ships.values():
            if candidate.vital.alive:
                alive_by_team[candidate.team].append(candidate)
                alive_ids_by_team[candidate.team].add(candidate.ship_id)

        changed_focus_keys = self._changed_focus_queues(world)
        pyfa_remote_inputs_dirty = False

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            runtime = ship.runtime
            allies_pool = alive_by_team.get(ship.team, [])
            enemies_alive = alive_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, [])
            ally_ids = alive_ids_by_team.get(ship.team, set())
            enemy_ids = alive_ids_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, set())
            force_target_reselect = self._focus_key(ship.team, ship.squad_id) in changed_focus_keys
            local_signature_dirty = False
            active_pyfa_remote_inputs_dirty = False
            synced_weapon_fire_delay_pairs: set[tuple[str | None, str | None]] = set()
            module_metadata_list = self._runtime_module_metadata_list(runtime)

            for module, metadata in zip(runtime.modules, module_metadata_list):
                if module.state == module.state.ACTIVE:
                    active_timer = ship.combat.module_cycle_timers.get(module.module_id)
                    if active_timer is not None:
                        timer_left = active_timer - dt
                        if timer_left > 0:
                            ship.combat.module_cycle_timers[module.module_id] = timer_left
                            continue

                if module.state == module.state.OFFLINE:
                    if self._module_affects_pyfa_remote_inputs(module) and (
                        module.module_id in ship.combat.projected_targets
                        or module.module_id in ship.combat.module_cycle_timers
                        or bool(self._module_cycle_snapshots_for(ship.ship_id, module.module_id))
                    ):
                        pyfa_remote_inputs_dirty = True
                    self._clear_module_cycle_snapshots(ship.ship_id, module.module_id)
                    ship.combat.module_reactivation_timers.pop(module.module_id, None)
                    ship.combat.module_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                    continue

                previous_state = module.state
                previous_projected_target = ship.combat.projected_targets.get(module.module_id)
                active_timer = ship.combat.module_cycle_timers.get(module.module_id) if module.state == module.state.ACTIVE else None

                active_effects = metadata.active_effects
                if not active_effects:
                    if previous_state == module.state.ACTIVE:
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                    self._clear_module_cycle_snapshots(ship.ship_id, module.module_id)
                    module.state = module.state.ONLINE
                    ship.combat.module_cycle_timers.pop(module.module_id, None)
                    ship.combat.module_reactivation_timers.pop(module.module_id, None)
                    ship.combat.module_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.projected_targets.pop(module.module_id, None)
                    continue

                cycle_cost = metadata.cycle_cost
                cycle_time = metadata.cycle_time
                reactivation_delay = metadata.reactivation_delay
                cycle_just_completed = False
                ammo_reload_started_this_tick = False

                if module.state == module.state.ACTIVE and cycle_time > 0:
                    if active_timer is not None:
                        timer_left = active_timer - dt
                        if timer_left > 0:
                            ship.combat.module_cycle_timers[module.module_id] = timer_left
                            continue
                        ship.combat.module_cycle_timers.pop(module.module_id, None)
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                        self._clear_module_cycle_snapshots(ship.ship_id, module.module_id)
                        cycle_just_completed = True
                        if reactivation_delay > 0.0:
                            ship.combat.module_reactivation_timers[module.module_id] = max(
                                ship.combat.module_reactivation_timers.get(module.module_id, 0.0),
                                reactivation_delay,
                            )
                        pending_ammo_reload = max(
                            0.0,
                            float(ship.combat.module_pending_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
                        )

                        if module.charge_capacity > 0 and module.charge_rate > 0.0:
                            module.charge_remaining = max(0.0, float(module.charge_remaining) - float(module.charge_rate))
                            if module.charge_remaining <= 0.0:
                                module.charge_remaining = 0.0
                                if pending_ammo_reload <= 0.0:
                                    auto_reload_time = max(0.0, float(module.charge_reload_time))
                                    if auto_reload_time > 0.0:
                                        ship.combat.module_ammo_reload_timers[module.module_id] = max(
                                            auto_reload_time,
                                            float(ship.combat.module_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
                                        )
                                        ammo_reload_started_this_tick = True
                                    else:
                                        module.charge_remaining = float(module.charge_capacity)

                        if pending_ammo_reload > 0.0:
                            ship.combat.module_ammo_reload_timers[module.module_id] = pending_ammo_reload
                            ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                            ammo_reload_started_this_tick = True

                decision_rule = metadata.decision_rule
                desired_active = False
                projected_target_id: str | None = None
                has_projected = metadata.has_projected
                cycle_started = False

                if has_projected:
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
                                now=float(world.now),
                            )
                            synced_weapon_fire_delay_pairs.add(delay_pair)

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
                    float(ship.combat.module_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
                )
                if ammo_reload_left > 0.0:
                    if not ammo_reload_started_this_tick:
                        ammo_reload_left = max(0.0, ammo_reload_left - dt)
                    if ammo_reload_left > 0.0:
                        ship.combat.module_ammo_reload_timers[module.module_id] = ammo_reload_left
                        desired_active = False
                    else:
                        ship.combat.module_ammo_reload_timers.pop(module.module_id, None)
                        if module.charge_capacity > 0:
                            module.charge_remaining = float(module.charge_capacity)

                pending_ammo_reload_left = max(
                    0.0,
                    float(ship.combat.module_pending_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
                )
                active_ammo_reload_left = max(
                    0.0,
                    float(ship.combat.module_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
                )
                current_cycle_left = max(
                    0.0,
                    float(ship.combat.module_cycle_timers.get(module.module_id, 0.0) or 0.0),
                )
                if active_ammo_reload_left <= 0.0 and pending_ammo_reload_left > 0.0 and current_cycle_left <= 0.0:
                    ship.combat.module_ammo_reload_timers[module.module_id] = pending_ammo_reload_left
                    ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                    desired_active = False

                if module.charge_capacity > 0 and module.charge_rate > 0.0 and module.charge_remaining <= 0.0:
                    if module.module_id not in ship.combat.module_ammo_reload_timers:
                        auto_reload_time = max(0.0, float(module.charge_reload_time))
                        if auto_reload_time > 0.0:
                            ship.combat.module_ammo_reload_timers[module.module_id] = auto_reload_time
                        else:
                            module.charge_remaining = float(module.charge_capacity)
                    desired_active = False

                cooldown_left = ship.combat.module_reactivation_timers.get(module.module_id)
                if cooldown_left is not None:
                    if not cycle_just_completed:
                        cooldown_left -= dt
                    if cooldown_left > 0.0:
                        ship.combat.module_reactivation_timers[module.module_id] = cooldown_left
                        desired_active = False
                    else:
                        ship.combat.module_reactivation_timers.pop(module.module_id, None)

                activation_target_id: str | None = (
                    projected_target_id
                    if has_projected and not metadata.is_area_effect
                    else None
                )

                if desired_active and activation_target_id is not None:
                    activation_target = world.ships.get(activation_target_id)
                    if not self._ensure_target_lock(
                        world,
                        ship,
                        activation_target_id,
                        activation_target,
                        lock_context="module_lock",
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
                            ship.combat.module_cycle_timers[module.module_id] = cycle_time
                            cycle_started = True
                    else:
                        ship.combat.module_cycle_timers.pop(module.module_id, None)
                else:
                    self._clear_module_cycle_snapshots(ship.ship_id, module.module_id)
                    ship.combat.module_cycle_timers.pop(module.module_id, None)

                module.state = module.state.ACTIVE if desired_active else module.state.ONLINE
                if projected_target_id is not None:
                    ship.combat.projected_targets[module.module_id] = projected_target_id
                elif module.module_id in ship.combat.projected_targets:
                    ship.combat.projected_targets.pop(module.module_id, None)

                # ECM is resolved once at cycle start so first activation round shows immediate result.
                if cycle_started:
                    if metadata.is_burst_jammer:
                        self._resolve_area_ecm_cycle(world, ship, module)
                    elif metadata.is_ecm and projected_target_id is not None:
                        self._resolve_ecm_cycle(world, ship, module, projected_target_id)

                if module.state == module.state.ACTIVE and (
                    cycle_started
                    or previous_state != module.state.ACTIVE
                    or previous_projected_target != projected_target_id
                ):
                    self._capture_module_cycle_snapshots(world, ship, module, projected_target_id)

                if cycle_started and self._uses_cycle_start_projected_application(metadata):
                    self._mark_projected_cycle_started(ship.ship_id, module.module_id)

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
                            "module": module.module_id,
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
                            "module": module.module_id,
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

            if local_signature_dirty:
                runtime.diagnostics["runtime_local_state_signature"] = tuple(
                    (str(module.module_id), str(module.state.value or "ONLINE").upper())
                    for module in runtime.modules
                    if self._module_static_metadata(module).affects_local_pyfa_profile
                )
            if active_pyfa_remote_inputs_dirty:
                runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = any(
                    str(module.state.value or "ONLINE").upper() in {"ACTIVE", "OVERHEATED"}
                    and self._module_affects_pyfa_remote_inputs(module)
                    for module in runtime.modules
                )

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

        shield_rep = float(effect.projected_add.get("shield_rep", 0.0) or 0.0)
        if shield_rep > 0.0:
            amount = shield_rep * strength
            before = target.vital.shield
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + amount)
            shield_repaired = max(0.0, target.vital.shield - before)

        armor_rep = float(effect.projected_add.get("armor_rep", 0.0) or 0.0)
        if armor_rep > 0.0:
            amount = armor_rep * strength
            before = target.vital.armor
            target.vital.armor = min(target.vital.armor_max, target.vital.armor + amount)
            armor_repaired = max(0.0, target.vital.armor - before)

        cap_drain = float(effect.projected_add.get("cap_drain", 0.0) or 0.0)
        if cap_drain > 0.0:
            amount = cap_drain * strength
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
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        damage_factor = strength if damage_factor_override is None else max(0.0, float(damage_factor_override))

        dealt_damage = _scale_damage(base_damage, damage_factor)
        total_damage = _sum_damage(dealt_damage)
        if total_damage <= 0.0:
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
        applied = (shield_before + armor_before + structure_before) - (
            target.vital.shield + target.vital.armor + target.vital.structure
        )
        if applied > 0.0:
            target.combat.last_damaged_at = world.now
        if target.vital.structure <= 0:
            target.vital.alive = False
            target.nav.velocity = Vector2(0.0, 0.0)

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
        return self._apply_runtime_projected_impacts(ship.profile, applied)

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
            if not ship.vital.alive:
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
                if target is None or (not target.vital.alive) or target.team == own_team:
                    continue
                seen.add(target_id)
                cleaned.append(target_id)
            world.squad_focus_queues[focus_key] = cleaned

            pre_targets = cleaned[1:] if len(cleaned) > 1 else []
            valid_pre = set(pre_targets)

            prelocked = world.squad_prelocked_targets.setdefault(focus_key, set())
            timers = world.squad_prelock_timers.setdefault(focus_key, {})
            for target_id in list(prelocked):
                if target_id not in valid_pre:
                    prelocked.discard(target_id)
            for target_id in list(timers.keys()):
                if target_id not in valid_pre:
                    timers.pop(target_id, None)

            if not pre_targets:
                continue

            leader = members[0]
            attacker_profile = effective_profiles.get(leader.ship_id) or leader.profile
            for target_id in pre_targets:
                if target_id in prelocked:
                    continue
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive:
                    continue
                target_profile = effective_profiles.get(target_id) or target.profile
                left = timers.get(target_id)
                if left is None:
                    timers[target_id] = self._cached_lock_time(attacker_profile, target_profile)
                    continue
                left -= dt
                if left <= 0:
                    prelocked.add(target_id)
                    timers.pop(target_id, None)
                else:
                    timers[target_id] = left

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
        if self.event_logging_enabled:
            self._advance_merge_window(world.now)
        else:
            self._merged_event_buckets.clear()
            self._merge_window_start_time = None
            self._merge_window_end_time = None

        started = time.perf_counter()
        self._update_ecm_restrictions(world)
        self._log_hotspot("combat.update_ecm_restrictions", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._advance_target_locks(world, dt)
        self._log_hotspot("combat.advance_target_locks", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        if self._update_module_states(world, dt):
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
            if not source.vital.alive or source.runtime is None:
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
                if not cycle_target_snapshots:
                    continue

                for effect_index, effect in metadata.projected_effects:

                    targets: list[tuple[Any, CycleTargetSnapshot, float]] = []
                    if metadata.is_smart_bomb:
                        for target_id, target_snapshot in cycle_target_snapshots.items():
                            target = world.ships.get(target_id)
                            if target is None or not target.vital.alive:
                                continue
                            strength = max(0.0, float(target_snapshot.effect_strengths.get(effect_index, 0.0) or 0.0))
                            if strength > 0.0:
                                targets.append((target, target_snapshot, strength))
                    else:
                        tgt_id = source.combat.projected_targets.get(module.module_id)
                        if not tgt_id:
                            continue
                        target = world.ships.get(tgt_id)
                        if target is None or not target.vital.alive:
                            continue
                        target_snapshot = cycle_target_snapshots.get(tgt_id)
                        if target_snapshot is None:
                            continue
                        strength = max(0.0, float(target_snapshot.effect_strengths.get(effect_index, 0.0) or 0.0))
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
            if ship.nav.velocity.length() > ship.nav.max_speed:
                ship.nav.velocity = ship.nav.velocity.normalized() * ship.nav.max_speed

            ship.vital.cap_max = ship_profile.max_cap
            if ship.vital.cap > ship.vital.cap_max:
                ship.vital.cap = ship.vital.cap_max

            ship.vital.cap = self._resolve_cap_recharge(
                cap_now=ship.vital.cap,
                cap_max=ship.vital.cap_max,
                recharge_time=ship_profile.cap_recharge_time,
                dt=dt,
            )

            current_target_id = ship.combat.current_target
            if current_target_id:
                current_target = world.ships.get(current_target_id)
                if (
                    current_target is None
                    or not current_target.vital.alive
                    or current_target.team == ship.team
                ):
                    ship.combat.current_target = None

            if not ship.combat.current_target:
                queue = list(world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), []))
                for candidate_id in queue:
                    candidate = world.ships.get(candidate_id)
                    if candidate is None or not candidate.vital.alive or candidate.team == ship.team:
                        continue
                    ship.combat.current_target = candidate_id
                    break

        if self.event_logging_enabled:
            self._advance_merge_window(world.now)


class LogisticsSystem:
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
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + repair)
