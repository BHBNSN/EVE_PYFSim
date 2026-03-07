from __future__ import annotations

from dataclasses import dataclass, replace
import math
import logging
import random
from typing import Any

import numpy as np

from .fleet_setup import recompute_profile_from_pyfa_runtime
from .fit_runtime import EffectClass, ProjectedImpact, RuntimeStatEngine
from .math2d import Vector2
from .models import ShipProfile, Team
from .pyfa_bridge import PyfaBridge
from .sim_logging import log_sim_event
from .world import WorldState


DamageTuple = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ModuleDecisionRule:
    rule_id: str
    activation_mode: str
    target_mode: str
    cap_threshold: float = 0.0


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
        pos = np.array([(s.nav.position.x, s.nav.position.y) for s in alive], dtype=np.float64)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        for i, ship in enumerate(alive):
            mask = (dist[i] <= self.sensor_range) & (dist[i] > 0)
            ship.perception = [alive[j].ship_id for j in np.where(mask)[0].tolist()]


class MovementSystem:
    def __init__(self, battlefield_radius: float) -> None:
        self.battlefield_radius = battlefield_radius

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
    def _motion_tau(ship, speed_cap: float) -> float:
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
                        return max(0.25, (mass * agility) / 1_000_000.0)
        return max(0.25, MovementSystem._align_time_for(speed_cap))

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

    def _update_velocity_with_inertia(self, world: WorldState, ship, dt: float) -> None:
        target = ship.nav.command_target
        speed_cap = self._effective_speed_cap(world, ship)
        desired_angle = ship.nav.facing_deg
        throttle = 0.0

        if target is not None:
            to_target = target - ship.nav.position
            distance = to_target.length()
            if distance > max(120.0, ship.nav.radius * 1.5):
                throttle = 1.0
                desired_angle = to_target.angle_deg()

        align_time = self._align_time_for(speed_cap)
        max_turn_rate = 220.0 / max(0.5, align_time)
        max_turn_step = max_turn_rate * dt
        angle_delta = self._wrap_angle_deg(desired_angle - ship.nav.facing_deg)
        angle_step = max(-max_turn_step, min(max_turn_step, angle_delta))
        ship.nav.facing_deg = self._wrap_angle_deg(ship.nav.facing_deg + angle_step)

        tau = self._motion_tau(ship, speed_cap)
        thrust_acc = (speed_cap / max(0.1, tau)) * throttle
        facing_rad = math.radians(ship.nav.facing_deg)
        thrust_vec = Vector2(math.cos(facing_rad) * thrust_acc, math.sin(facing_rad) * thrust_acc)
        drag_vec = ship.nav.velocity * (1.0 / max(0.1, tau))
        acceleration = thrust_vec - drag_vec
        ship.nav.velocity = ship.nav.velocity + acceleration * dt

    def run(self, world: WorldState, dt: float) -> None:
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            self._update_velocity_with_inertia(world, ship, dt)
            next_pos = ship.nav.position + ship.nav.velocity * dt
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
        self.event_logging_enabled: bool = False
        self.event_merge_window_sec: float = 1.0
        self._diag_logged_ships: set[str] = set()
        self._lock_time_cache: dict[tuple[float, float], float] = {}
        self._projected_cycle_totals: dict[tuple[str, str, str], dict[str, float]] = {}
        self._merged_event_buckets: dict[tuple, dict[str, Any]] = {}
        self._merge_window_start_time: float | None = None
        self._merge_window_end_time: float | None = None

    def attach_logger(self, logger: logging.Logger, detailed_logging: bool, merge_window_sec: float = 1.0) -> None:
        self.logger = logger
        self.event_logging_enabled = bool(detailed_logging)
        self.detailed_logging = False
        try:
            self.event_merge_window_sec = max(0.1, float(merge_window_sec))
        except Exception:
            self.event_merge_window_sec = 1.0
        self._merge_window_start_time = None
        self._merge_window_end_time = None
        self._merged_event_buckets.clear()

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

    def _resolve_cap_recharge(self, cap_now: float, cap_max: float, recharge_time: float, dt: float) -> float:
        if cap_max <= 0 or recharge_time <= 0:
            return cap_now
        cap = max(0.0, min(cap_max, cap_now))
        tau = recharge_time / 5.0
        if tau <= 0:
            return cap
        inner = 1.0 + (math.sqrt(max(cap / cap_max, 0.0)) - 1.0) * math.exp(-dt / tau)
        return max(0.0, min(cap_max, (inner * inner) * cap_max))

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
            for module in source.runtime.modules:
                for effect in module.effects:
                    if effect.effect_class != EffectClass.PROJECTED:
                        continue
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

                    distance = source.nav.position.distance_to(target.nav.position)
                    max_range = self._projected_max_range(effect)
                    if max_range > 0 and distance > max_range:
                        continue

                    strength = self._projected_strength(effect, distance)
                    if strength <= 0:
                        continue
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(
                            f"projected_formula source={source.ship_id} target={target_id} module={module.module_id} dist={distance:.1f} range={effect.range_m:.1f} falloff={effect.falloff_m:.1f} strength={strength:.4f}"
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
    def _module_has_projected(module) -> bool:
        return any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects)

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
    def _module_prefers_propulsion_command(module) -> bool:
        for effect in module.effects:
            if effect.effect_class != EffectClass.LOCAL:
                continue
            if str(effect.state_required.value).upper() != "ACTIVE":
                continue
            speed_mult = float(effect.local_mult.get("speed", 1.0) or 1.0)
            if speed_mult > 1.0:
                return True
        return False

    @staticmethod
    def _module_group_name(module) -> str:
        return str(getattr(module, "group", "") or "").lower()

    @staticmethod
    def _module_group_has_any(module, tokens: tuple[str, ...]) -> bool:
        group_name = CombatSystem._module_group_name(module)
        return any(token in group_name for token in tokens)

    @staticmethod
    def _module_group_has_equal(module, tokens: tuple[str, ...]) -> bool:
        group_name = CombatSystem._module_group_name(module)
        return any(token == group_name for token in tokens)

    @staticmethod
    def _module_has_projected_damage(module) -> bool:
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            for key in ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive"):
                if float(effect.projected_add.get(key, 0.0) or 0.0) > 0.0:
                    return True
        return False

    @staticmethod
    def _module_has_projected_rep(module) -> bool:
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            if float(effect.projected_add.get("shield_rep", 0.0) or 0.0) > 0.0:
                return True
            if float(effect.projected_add.get("armor_rep", 0.0) or 0.0) > 0.0:
                return True
        return False

    @staticmethod
    def _module_is_weapon_module(module) -> bool:
        if not CombatSystem._module_has_projected(module):
            return False
        if not CombatSystem._module_has_projected_damage(module):
            return False
        group_name = CombatSystem._module_group_name(module)
        looks_like_weapon_group = (
            ("weapon" in group_name)
            or ("missile launcher" in group_name)
        )
        has_ammo_like = int(getattr(module, "charge_capacity", 0) or 0) > 0 and float(getattr(module, "charge_rate", 0.0) or 0.0) > 0.0
        return looks_like_weapon_group or has_ammo_like

    @staticmethod
    def _module_is_offensive_ewar_module(module) -> bool:
        if not CombatSystem._module_has_projected(module):
            return False
        if CombatSystem._module_is_weapon_module(module):
            return False
        if CombatSystem._module_target_side(module) != "enemy":
            return False
        return CombatSystem._module_group_has_any(
            module,
            (
                "weapon disruptor",
                "sensor damp",
                "energy neutral",
                "nosferatu",
                "ecm",
                "warp scrambler",
            ),
        )

    @staticmethod
    def _module_is_target_ewar_module(module) -> bool:
        return CombatSystem._module_group_has_any(
            module,
            (
                "target painter",
                "stasis web",
                "stasis grappler",
            ),
        )

    @staticmethod
    def _module_is_hardener_module(module) -> bool:
        return CombatSystem._module_group_has_any(
            module,
            (
                "shield hardener",
                "armor hardener",
                "energized",
                "armor resistance shift hardener",
            ),
        )

    @staticmethod
    def _module_is_cap_booster_module(module) -> bool:
        return CombatSystem._module_group_has_any(
            module,
            (
                "capacitor booster",
            ),
        )

    @staticmethod
    def _module_is_propulsion_module(module) -> bool:
        return CombatSystem._module_group_has_any(
            module,
            (
                "propulsion module",
            ),
        )

    @staticmethod
    def _module_is_damage_control_module(module) -> bool:
        return CombatSystem._module_group_has_equal(
            module,
            (
                "damage control",
            ),
        )

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
            focus_queue = list(world.squad_focus_queues.get(self._focus_key(source.team, source.squad_id), []))
            if not focus_queue:
                return False
            allowed_ids: set[str] = {str(focus_queue[0])}
            if len(focus_queue) > 1:
                allowed_ids.add(str(focus_queue[1]))
            return target_id in allowed_ids and self._ship_id_in_pool(target_id, enemies_pool)

        if rule.target_mode == "ally_lowest_hp":
            if target_id == source.ship_id:
                return False
            return self._is_lowest_hp_ally_target(source, module, allies_pool, target_id)

        if rule.target_mode in {"enemy_random", "enemy_nearest"}:
            return self._ship_id_in_pool(target_id, enemies_pool)

        side = self._module_target_side(module)
        if side == "ally":
            if target_id == source.ship_id:
                return False
            return self._ship_id_in_pool(target_id, allies_pool)
        return self._ship_id_in_pool(target_id, enemies_pool)

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

    def _select_weapon_focus_target(self, world: WorldState, source, module, enemies_pool: list, existing_target_id: str | None) -> str | None:
        focus_queue = list(world.squad_focus_queues.get(self._focus_key(source.team, source.squad_id), []))
        if not focus_queue:
            return None

        queue_targets: list[str] = []
        if len(focus_queue) >= 1:
            queue_targets.append(str(focus_queue[0]))
        if len(focus_queue) >= 2:
            queue_targets.append(str(focus_queue[1]))
        if not queue_targets:
            return None

        ranged_ids = {
            enemy.ship_id
            for enemy in self._candidates_in_projected_range(source, module, enemies_pool)
        }

        valid_focus_id = queue_targets[0] if queue_targets[0] in ranged_ids else None
        valid_prefocus_id = None
        if len(queue_targets) > 1 and queue_targets[1] in ranged_ids:
            valid_prefocus_id = queue_targets[1]

        valid_ids = {candidate_id for candidate_id in (valid_focus_id, valid_prefocus_id) if candidate_id}
        if not valid_ids:
            return None
        if existing_target_id in valid_ids:
            return existing_target_id

        if valid_focus_id and valid_prefocus_id:
            use_prefocus = random.random() < self._prefocus_fire_probability(source)
            return valid_prefocus_id if use_prefocus else valid_focus_id
        return valid_focus_id or valid_prefocus_id

    def _module_decision_rule(self, module) -> ModuleDecisionRule:
        # Central policy router: extend here when adding new active module behaviors.
        if self._module_is_weapon_module(module):
            return ModuleDecisionRule(
                rule_id="weapon_focus_only",
                activation_mode="weapon_focus_only",
                target_mode="weapon_focus_prefocus",
            )

        if self._module_has_projected(module):
            if self._module_has_projected_rep(module):
                return ModuleDecisionRule(
                    rule_id="projected_remote_repair",
                    activation_mode="always",
                    target_mode="ally_lowest_hp",
                )
            if self._module_is_offensive_ewar_module(module):
                return ModuleDecisionRule(
                    rule_id="projected_offensive_ewar",
                    activation_mode="cap_min",
                    target_mode="enemy_random",
                    cap_threshold=0.15,
                )
            if self._module_is_target_ewar_module(module):
                return ModuleDecisionRule(
                    rule_id="weapon_focus_only",
                    activation_mode="weapon_focus_only",
                    target_mode="weapon_focus_prefocus",
                )
            side = self._module_target_side(module)
            if side == "ally":
                return ModuleDecisionRule(
                    rule_id="projected_support_generic",
                    activation_mode="always",
                    target_mode="ally_lowest_hp",
                )
            return ModuleDecisionRule(
                rule_id="projected_hostile_generic",
                activation_mode="never",
                target_mode="none",
            )

        if self._module_is_propulsion_module(module):
            return ModuleDecisionRule(
                rule_id="local_propulsion",
                activation_mode="propulsion_command",
                target_mode="none",
            )

        if self._module_is_damage_control_module(module):
            return ModuleDecisionRule(
                rule_id="local_damage_control",
                activation_mode="recent_enemy_weapon_damage",
                target_mode="none",
            )

        if self._module_is_hardener_module(module):
            return ModuleDecisionRule(
                rule_id="local_hardener",
                activation_mode="cap_or_low_hp",
                target_mode="none",
                cap_threshold=0.10,
            )

        if self._module_is_cap_booster_module(module):
            return ModuleDecisionRule(
                rule_id="local_cap_booster",
                activation_mode="cap_max",
                target_mode="none",
                cap_threshold=0.85,
            )

        return ModuleDecisionRule(
            rule_id="local_active_default",
            activation_mode="never",
            target_mode="none",
        )

    def _should_activate_module(self, world: WorldState, ship, rule: ModuleDecisionRule, target_id: str | None) -> bool:
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
            return self._select_weapon_focus_target(world, source, module, enemies_pool, existing_target_id)
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

    def _update_module_states(self, world: WorldState, dt: float) -> None:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        for candidate in world.ships.values():
            if candidate.vital.alive:
                alive_by_team[candidate.team].append(candidate)

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            allies_pool = alive_by_team.get(ship.team, [])
            enemies_alive = alive_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, [])

            for module in ship.runtime.modules:
                if module.state == module.state.OFFLINE:
                    ship.combat.module_reactivation_timers.pop(module.module_id, None)
                    ship.combat.module_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                    continue

                previous_state = module.state
                previous_projected_target = ship.combat.projected_targets.get(module.module_id)

                active_effects = [
                    effect
                    for effect in module.effects
                    if str(effect.state_required.value).upper() == "ACTIVE"
                ]
                if not active_effects:
                    if previous_state == module.state.ACTIVE:
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                    module.state = module.state.ONLINE
                    ship.combat.module_cycle_timers.pop(module.module_id, None)
                    ship.combat.module_reactivation_timers.pop(module.module_id, None)
                    ship.combat.module_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.module_pending_ammo_reload_timers.pop(module.module_id, None)
                    ship.combat.projected_targets.pop(module.module_id, None)
                    continue

                cycle_cost = sum(max(0.0, effect.cap_need) for effect in active_effects)
                cycle_time = min(
                    (max(0.1, effect.cycle_time) for effect in active_effects if effect.cycle_time > 0),
                    default=0.0,
                )
                reactivation_delay = max(
                    (max(0.0, float(getattr(effect, "reactivation_delay", 0.0) or 0.0)) for effect in active_effects),
                    default=0.0,
                )
                cycle_just_completed = False
                ammo_reload_started_this_tick = False

                if module.state == module.state.ACTIVE and cycle_time > 0:
                    active_timer = ship.combat.module_cycle_timers.get(module.module_id)
                    if active_timer is not None:
                        timer_left = active_timer - dt
                        if timer_left > 0:
                            ship.combat.module_cycle_timers[module.module_id] = timer_left
                            continue
                        ship.combat.module_cycle_timers.pop(module.module_id, None)
                        self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
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

                decision_rule = self._module_decision_rule(module)
                desired_active = False
                projected_target_id: str | None = None
                has_projected = self._module_has_projected(module)
                cycle_started = False

                if has_projected:
                    if self._can_reuse_projected_target(
                        world,
                        ship,
                        module,
                        decision_rule,
                        previous_projected_target,
                        allies_pool,
                        enemies_alive,
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
                        self._sync_weapon_fire_delay(
                            ship,
                            previous_target_id=previous_projected_target,
                            new_target_id=projected_target_id,
                            now=float(world.now),
                        )

                desired_active = self._should_activate_module(
                    world,
                    ship,
                    decision_rule,
                    projected_target_id,
                )
                if has_projected and projected_target_id is None:
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

                activation_target_id: str | None = projected_target_id if has_projected else None

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

                if has_projected and projected_target_id is None:
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
                    ship.combat.module_cycle_timers.pop(module.module_id, None)

                module.state = module.state.ACTIVE if desired_active else module.state.ONLINE
                if projected_target_id is not None:
                    ship.combat.projected_targets[module.module_id] = projected_target_id
                elif module.module_id in ship.combat.projected_targets:
                    ship.combat.projected_targets.pop(module.module_id, None)

                # ECM is resolved once at cycle start so first activation round shows immediate result.
                if cycle_started and projected_target_id is not None:
                    self._resolve_ecm_cycle(world, ship, module, projected_target_id)

                if previous_projected_target and (
                    module.state != module.state.ACTIVE or previous_projected_target != projected_target_id
                ):
                    self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)

                if previous_state != module.state:
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

    def _apply_projected_cycle_effects(
        self,
        world: WorldState,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        dt: float,
        strength: float,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        if target is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        strength = max(0.0, min(1.0, strength))
        cycle_scale = dt / max(0.1, effect.cycle_time)

        shield_repaired = 0.0
        armor_repaired = 0.0
        cap_drained = 0.0

        shield_rep = float(effect.projected_add.get("shield_rep", 0.0) or 0.0)
        if shield_rep > 0.0:
            amount = shield_rep * strength * cycle_scale
            before = target.vital.shield
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + amount)
            shield_repaired = max(0.0, target.vital.shield - before)

        armor_rep = float(effect.projected_add.get("armor_rep", 0.0) or 0.0)
        if armor_rep > 0.0:
            amount = armor_rep * strength * cycle_scale
            before = target.vital.armor
            target.vital.armor = min(target.vital.armor_max, target.vital.armor + amount)
            armor_repaired = max(0.0, target.vital.armor - before)

        cap_drain = float(effect.projected_add.get("cap_drain", 0.0) or 0.0)
        if cap_drain > 0.0:
            amount = cap_drain * strength * cycle_scale
            before_cap = target.vital.cap
            target.vital.cap = max(0.0, target.vital.cap - amount)
            cap_drained = max(0.0, before_cap - target.vital.cap)

        base_damage = (
            max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)) * cycle_scale,
            max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)) * cycle_scale,
            max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)) * cycle_scale,
            max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)) * cycle_scale,
        )
        if _sum_damage(base_damage) <= 0.0:
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        damage_factor = strength
        if float(effect.projected_add.get("weapon_is_turret", 0.0) or 0.0) > 0.5:
            relative_velocity = source.nav.velocity - target.nav.velocity
            radial = (target.nav.position - source.nav.position).normalized()
            tangential = Vector2(-radial.y, radial.x)
            transversal = abs(relative_velocity.x * tangential.x + relative_velocity.y * tangential.y)
            chance = self.pyfa.turret_chance_to_hit(
                tracking=max(0.0, float(effect.projected_add.get("weapon_tracking", 0.0) or 0.0)),
                optimal_sig=max(1.0, float(effect.projected_add.get("weapon_optimal_sig", 40_000.0) or 40_000.0)),
                distance=source.nav.position.distance_to(target.nav.position),
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

        pyfa_runtime_profile = recompute_profile_from_pyfa_runtime(ship.runtime)
        if pyfa_runtime_profile is not None:
            base = replace(pyfa_runtime_profile)
            applied = impacts.get(ship.ship_id)
            if not applied:
                return base
            effective = self.runtime.apply_projected_effects(base, applied)
            for attr in (
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
                "sensor_strength_gravimetric",
                "sensor_strength_ladar",
                "sensor_strength_magnetometric",
                "sensor_strength_radar",
                "shield_hp",
                "armor_hp",
                "structure_hp",
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
            ):
                setattr(effective, attr, getattr(base, attr, getattr(effective, attr, 0.0)))
            return effective

        preserved = ship.profile
        base = replace(self.runtime.compute_base_profile(ship.runtime))
        for attr in (
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
            "missile_explosion_radius",
            "missile_explosion_velocity",
            "missile_max_range",
            "missile_damage_reduction_factor",
            "sensor_strength_gravimetric",
            "sensor_strength_ladar",
            "sensor_strength_magnetometric",
            "sensor_strength_radar",
            "shield_hp",
            "armor_hp",
            "structure_hp",
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
        ):
            setattr(base, attr, getattr(preserved, attr, getattr(base, attr, 0.0)))
        applied = impacts.get(ship.ship_id)
        if not applied:
            return base
        effective = self.runtime.apply_projected_effects(base, applied)
        for attr in (
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
            "sensor_strength_gravimetric",
            "sensor_strength_ladar",
            "sensor_strength_magnetometric",
            "sensor_strength_radar",
            "shield_hp",
            "armor_hp",
            "structure_hp",
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
        ):
            setattr(effective, attr, getattr(base, attr, getattr(effective, attr, 0.0)))
        return effective

    @staticmethod
    def _focus_key(team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

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
        if self.event_logging_enabled:
            self._advance_merge_window(world.now)
        else:
            self._merged_event_buckets.clear()
            self._merge_window_start_time = None
            self._merge_window_end_time = None
        self._update_ecm_restrictions(world)
        self._advance_target_locks(world, dt)
        self._update_module_states(world, dt)
        projected = self._collect_projected_impacts(world, dt)

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

        self._update_squad_prelocks(world, dt, effective_profiles)

        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None:
                continue
            for module in source.runtime.modules:
                if module.state != module.state.ACTIVE:
                    continue
                tgt_id = source.combat.projected_targets.get(module.module_id)
                if not tgt_id:
                    continue
                target = world.ships.get(tgt_id)
                if target is None or not target.vital.alive:
                    continue
                distance = source.nav.position.distance_to(target.nav.position)
                target_profile = effective_profiles.get(target.ship_id, target.profile)
                for effect in module.effects:
                    if effect.effect_class != EffectClass.PROJECTED:
                        continue
                    max_range = self._projected_max_range(effect)
                    if max_range > 0 and distance > max_range:
                        continue
                    strength = self._projected_strength(effect, distance)
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
                        dt=dt,
                        strength=strength,
                    )
                    hp_after = target.vital.shield + target.vital.armor + target.vital.structure
                    applied_damage = max(0.0, hp_before - hp_after)
                    if (
                        applied_damage > 0.0
                        and source.team != target.team
                        and self._module_is_weapon_module(module)
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
