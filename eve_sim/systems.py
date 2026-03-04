from __future__ import annotations

from dataclasses import replace
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


def _scale_damage(dmg: DamageTuple, factor: float) -> DamageTuple:
    return dmg[0] * factor, dmg[1] * factor, dmg[2] * factor, dmg[3] * factor


def _sum_damage(dmg: DamageTuple) -> float:
    return dmg[0] + dmg[1] + dmg[2] + dmg[3]


def _fmt_damage(dmg: DamageTuple) -> str:
    return f"EM={dmg[0]:.3f},TH={dmg[1]:.3f},KI={dmg[2]:.3f},EX={dmg[3]:.3f}"


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


def _apply_damage_sequence_with_trace(
    shield: float,
    armor: float,
    structure: float,
    dmg: DamageTuple,
    target_profile,
) -> tuple[float, float, float, list[dict[str, float | str]]]:
    remaining = dmg
    layers = [
        ("shield", shield, (target_profile.shield_resonance_em, target_profile.shield_resonance_thermal, target_profile.shield_resonance_kinetic, target_profile.shield_resonance_explosive)),
        ("armor", armor, (target_profile.armor_resonance_em, target_profile.armor_resonance_thermal, target_profile.armor_resonance_kinetic, target_profile.armor_resonance_explosive)),
        ("structure", structure, (target_profile.structure_resonance_em, target_profile.structure_resonance_thermal, target_profile.structure_resonance_kinetic, target_profile.structure_resonance_explosive)),
    ]
    new_vals = {"shield": shield, "armor": armor, "structure": structure}
    trace: list[dict[str, float | str]] = []
    for layer_name, layer_hp, layer_res in layers:
        if layer_hp <= 0:
            continue
        eff = _layer_effective_damage(remaining, layer_res)
        if eff <= 0:
            continue
        if eff <= layer_hp:
            new_vals[layer_name] = layer_hp - eff
            trace.append({
                "layer": layer_name,
                "before": layer_hp,
                "effective_applied": eff,
                "after": new_vals[layer_name],
                "overflow_ratio": 0.0,
                "res_em": layer_res[0],
                "res_th": layer_res[1],
                "res_ki": layer_res[2],
                "res_ex": layer_res[3],
                "raw_em": remaining[0],
                "raw_th": remaining[1],
                "raw_ki": remaining[2],
                "raw_ex": remaining[3],
            })
            return new_vals["shield"], new_vals["armor"], new_vals["structure"], trace

        consumed_ratio = max(0.0, min(1.0, layer_hp / eff))
        trace.append({
            "layer": layer_name,
            "before": layer_hp,
            "effective_applied": layer_hp,
            "after": 0.0,
            "overflow_ratio": 1.0 - consumed_ratio,
            "res_em": layer_res[0],
            "res_th": layer_res[1],
            "res_ki": layer_res[2],
            "res_ex": layer_res[3],
            "raw_em": remaining[0],
            "raw_th": remaining[1],
            "raw_ki": remaining[2],
            "raw_ex": remaining[3],
        })
        new_vals[layer_name] = 0.0
        remaining = _scale_damage(remaining, 1.0 - consumed_ratio)

    return new_vals["shield"], new_vals["armor"], new_vals["structure"], trace


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
    ) -> None:
        key = (source_ship_id, module_id, target_ship_id)
        entry = self._projected_cycle_totals.setdefault(
            key,
            {"shield_repaired": 0.0, "armor_repaired": 0.0, "cap_drained": 0.0},
        )
        entry["shield_repaired"] += max(0.0, float(shield_repaired))
        entry["armor_repaired"] += max(0.0, float(armor_repaired))
        entry["cap_drained"] += max(0.0, float(cap_drained))

    def _flush_projected_cycle_total(self, world: WorldState, source_ship_id: str, module, target_ship_id: str | None) -> None:
        if not target_ship_id:
            return
        key = (source_ship_id, module.module_id, target_ship_id)
        totals = self._projected_cycle_totals.pop(key, None)
        if not totals:
            return
        if totals["shield_repaired"] <= 0.0 and totals["armor_repaired"] <= 0.0 and totals["cap_drained"] <= 0.0:
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

    @staticmethod
    def _sample_fire_delay(ship) -> float:
        level = ship.quality.level.value
        r = random.random()
        if level == "ELITE":
            if r < 0.70:
                return random.uniform(0.5, 1.2)
            if r < 0.95:
                return random.uniform(1.2, 2.0)
            return random.uniform(2.0, 3.0)
        if level == "REGULAR":
            if r < 0.35:
                return random.uniform(0.5, 1.2)
            if r < 0.80:
                return random.uniform(1.2, 2.2)
            return random.uniform(2.2, 3.0)
        if r < 0.15:
            return random.uniform(0.5, 1.2)
        if r < 0.55:
            return random.uniform(1.2, 2.2)
        return random.uniform(2.2, 3.0)

    def _collect_projected_impacts(self, world: WorldState, dt: float) -> dict[str, list[ProjectedImpact]]:
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

                    if target_id not in source.combat.lock_targets:
                        if target_id == source.combat.current_target:
                            continue
                        left = source.combat.lock_timers.get(target_id)
                        if left is None:
                            source.combat.lock_timers[target_id] = self._cached_lock_time(source.profile, target.profile)
                            if self.detailed_logging and self.logger is not None:
                                self.logger.debug(
                                    f"projected_lock_start source={source.ship_id} target={target_id} lock_time={source.combat.lock_timers[target_id]:.2f}"
                                )
                            continue
                        left -= dt
                        if left <= 0:
                            source.combat.lock_targets.add(target_id)
                            source.combat.lock_timers.pop(target_id, None)
                            if self.detailed_logging and self.logger is not None:
                                self.logger.debug(f"projected_lock_complete source={source.ship_id} target={target_id}")
                        else:
                            source.combat.lock_timers[target_id] = left
                            continue

                    distance = source.nav.position.distance_to(target.nav.position)
                    max_range = effect.range_m + max(0.0, effect.falloff_m)
                    if max_range > 0 and distance > max_range:
                        continue

                    if effect.range_m > 0 or effect.falloff_m > 0:
                        strength = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
                    else:
                        strength = 1.0
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
            max_range = effect.range_m + max(0.0, effect.falloff_m)
            if max_range <= 0 or distance <= max_range:
                return True
        return not has_projected

    @staticmethod
    def _is_weapon_module(group_name: str) -> bool:
        return ("weapon" in group_name) or ("turret" in group_name) or ("launcher" in group_name)

    @staticmethod
    def _is_propulsion_module(group_name: str) -> bool:
        return (
            "propulsion" in group_name
            or "afterburner" in group_name
            or "microwarpdrive" in group_name
        )

    @staticmethod
    def _is_remote_repair_module(group_name: str) -> bool:
        return (
            "shield transporter" in group_name
            or "remote shield booster" in group_name
            or "remote armor repair" in group_name
        )

    @staticmethod
    def _is_web_module(group_name: str) -> bool:
        return ("stasis web" in group_name) or ("stasis grappler" in group_name)

    @staticmethod
    def _is_ewar_module(group_name: str) -> bool:
        return (
            "sensor dampener" in group_name
            or "tracking disruptor" in group_name
            or "weapon disruptor" in group_name
            or "ecm" in group_name
            or "target painter" in group_name
            or "energy neutralizer" in group_name
        )

    @staticmethod
    def _is_adc_module(group_name: str) -> bool:
        return "assault damage control" in group_name

    @staticmethod
    def _is_resist_module(group_name: str) -> bool:
        return (
            "hardener" in group_name
            or "resistance" in group_name
            or "energized armor" in group_name
            or "adaptive invulnerability" in group_name
            or "multispectrum" in group_name
            or ("damage control" in group_name)
        )

    def _update_module_states(self, world: WorldState, dt: float) -> None:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        for candidate in world.ships.values():
            if candidate.vital.alive:
                alive_by_team[candidate.team].append(candidate)

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            target = world.ships.get(ship.combat.current_target) if ship.combat.current_target else None
            if target is not None and not target.vital.alive:
                target = None
            cap_ratio = ship.vital.cap / max(1.0, ship.vital.cap_max)
            recently_damaged = (world.now - ship.combat.last_damaged_at) <= 60.0

            allies_pool = alive_by_team.get(ship.team, [])
            enemies_alive = alive_by_team.get(Team.RED if ship.team == Team.BLUE else Team.BLUE, [])

            for module in ship.runtime.modules:
                if module.state == module.state.OFFLINE:
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
                    ship.combat.projected_targets.pop(module.module_id, None)
                    continue

                cycle_cost = sum(max(0.0, effect.cap_need) for effect in active_effects)
                cycle_time = min(
                    (max(0.1, effect.cycle_time) for effect in active_effects if effect.cycle_time > 0),
                    default=0.0,
                )

                if module.state == module.state.ACTIVE and cycle_time > 0:
                    timer_left = ship.combat.module_cycle_timers.get(module.module_id, cycle_time) - dt
                    if timer_left > 0:
                        ship.combat.module_cycle_timers[module.module_id] = timer_left
                        continue
                    self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)

                desired_active = False
                projected_target_id: str | None = None
                group_name = module.group.lower()
                has_projected = self._module_has_projected(module)
                cycle_started = False

                if self._is_propulsion_module(group_name):
                    desired_active = bool(ship.nav.propulsion_command_active)

                elif self._is_weapon_module(group_name):
                    if target is not None and self._module_in_projected_range(ship, target, module):
                        desired_active = True
                        if has_projected:
                            projected_target_id = target.ship_id

                elif self._is_remote_repair_module(group_name):
                    candidates = [
                        ally for ally in allies_pool
                        if ally.ship_id != ship.ship_id
                        and self._hp_ratio(ally) < 0.999
                        and self._module_in_projected_range(ship, ally, module)
                    ]
                    if candidates:
                        lowest = min(candidates, key=self._hp_ratio)
                        projected_target_id = lowest.ship_id
                        desired_active = True

                elif self._is_web_module(group_name):
                    if target is not None and self._module_in_projected_range(ship, target, module):
                        projected_target_id = target.ship_id
                        desired_active = True

                elif self._is_ewar_module(group_name):
                    if cap_ratio >= 0.20:
                        current_target_id = ship.combat.projected_targets.get(module.module_id)
                        if current_target_id:
                            current_target = world.ships.get(current_target_id)
                            if (
                                current_target is not None
                                and current_target.vital.alive
                                and self._module_in_projected_range(ship, current_target, module)
                            ):
                                projected_target_id = current_target_id
                                desired_active = True
                        candidates = [
                            enemy for enemy in enemies_alive
                            if self._module_in_projected_range(ship, enemy, module)
                        ]
                        if not desired_active and candidates:
                            chosen = random.choice(candidates)
                            projected_target_id = chosen.ship_id
                            desired_active = True

                elif self._is_adc_module(group_name):
                    desired_active = recently_damaged

                elif self._is_resist_module(group_name):
                    desired_active = (cap_ratio > 0.15) or recently_damaged

                else:
                    if has_projected:
                        if target is not None and self._module_in_projected_range(ship, target, module):
                            projected_target_id = target.ship_id
                            desired_active = True
                    else:
                        desired_active = module.state == module.state.ACTIVE

                if has_projected and projected_target_id is None:
                    desired_active = False

                if desired_active:
                    if cycle_cost > 0 and cycle_time > 0:
                        if cycle_cost > max(0.0, ship.vital.cap):
                            desired_active = False
                        else:
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

    @staticmethod
    def _apply_projected_support_effects(source, target, effect, dt: float, strength: float) -> tuple[float, float, float]:
        if target is None:
            return 0.0, 0.0, 0.0
        strength = max(0.0, min(1.0, strength))
        shield_repaired = 0.0
        armor_repaired = 0.0
        cap_drained = 0.0
        shield_rep = float(effect.projected_add.get("shield_rep", 0.0) or 0.0)
        armor_rep = float(effect.projected_add.get("armor_rep", 0.0) or 0.0)
        if shield_rep > 0:
            amount = shield_rep * strength * (dt / max(0.1, effect.cycle_time))
            before = target.vital.shield
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + amount)
            shield_repaired = max(0.0, target.vital.shield - before)
        if armor_rep > 0:
            amount = armor_rep * strength * (dt / max(0.1, effect.cycle_time))
            before = target.vital.armor
            target.vital.armor = min(target.vital.armor_max, target.vital.armor + amount)
            armor_repaired = max(0.0, target.vital.armor - before)
        cap_drain = float(effect.projected_add.get("cap_drain", 0.0) or 0.0)
        if cap_drain > 0:
            amount = cap_drain * strength * (dt / max(0.1, effect.cycle_time))
            before_cap = target.vital.cap
            target.vital.cap = max(0.0, target.vital.cap - amount)
            cap_drained = max(0.0, before_cap - target.vital.cap)
        return shield_repaired, armor_repaired, cap_drained

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
                "missile_explosion_radius",
                "missile_explosion_velocity",
                "missile_max_range",
                "missile_damage_reduction_factor",
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
            "missile_explosion_radius",
            "missile_explosion_velocity",
            "missile_max_range",
            "missile_damage_reduction_factor",
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
    def _weapon_activity_ratio(ship) -> tuple[float, float]:
        if ship.runtime is None:
            return 1.0, 1.0
        turret_total = 0
        turret_active = 0
        missile_total = 0
        missile_active = 0
        for module in ship.runtime.modules:
            group = (module.group or "").lower()
            is_launcher = "launcher" in group
            is_weapon = ("weapon" in group) or ("turret" in group) or is_launcher
            if not is_weapon:
                continue
            if is_launcher:
                missile_total += 1
                if module.state == module.state.ACTIVE:
                    missile_active += 1
            else:
                turret_total += 1
                if module.state == module.state.ACTIVE:
                    turret_active += 1

        turret_ratio = 1.0 if turret_total <= 0 else (turret_active / turret_total)
        missile_ratio = 1.0 if missile_total <= 0 else (missile_active / missile_total)
        return max(0.0, min(1.0, turret_ratio)), max(0.0, min(1.0, missile_ratio))

    @staticmethod
    def _focus_key(team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    @staticmethod
    def _turret_damage(ship_profile, chance: float) -> DamageTuple:
        cycle = max(0.0, ship_profile.turret_cycle)
        scale = cycle
        return (
            ship_profile.turret_em_dps * scale,
            ship_profile.turret_thermal_dps * scale,
            ship_profile.turret_kinetic_dps * scale,
            ship_profile.turret_explosive_dps * scale,
        )

    @staticmethod
    def _missile_factor(ship_profile, target_profile, relative_speed: float) -> float:
        if ship_profile.missile_explosion_radius <= 0:
            return 1.0
        sig_factor = target_profile.sig_radius / max(1.0, ship_profile.missile_explosion_radius)
        vel_term = (sig_factor * ship_profile.missile_explosion_velocity) / max(1.0, relative_speed)
        drf = max(0.1, ship_profile.missile_damage_reduction_factor)
        vel_factor = vel_term ** drf
        return max(0.0, min(1.0, min(sig_factor, vel_factor, 1.0)))

    @staticmethod
    def _missile_damage(ship_profile, target_profile, distance: float, relative_speed: float) -> DamageTuple:
        if ship_profile.missile_dps <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        if ship_profile.missile_max_range > 0 and distance > ship_profile.missile_max_range:
            return (0.0, 0.0, 0.0, 0.0)
        app = CombatSystem._missile_factor(ship_profile, target_profile, relative_speed)
        cycle = max(0.0, ship_profile.missile_cycle)
        scale = cycle * app
        return (
            ship_profile.missile_em_dps * scale,
            ship_profile.missile_thermal_dps * scale,
            ship_profile.missile_kinetic_dps * scale,
            ship_profile.missile_explosive_dps * scale,
        )

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
                for effect in module.effects:
                    if effect.effect_class != EffectClass.PROJECTED:
                        continue
                    max_range = effect.range_m + max(0.0, effect.falloff_m)
                    if max_range > 0 and distance > max_range:
                        continue
                    if effect.range_m > 0 or effect.falloff_m > 0:
                        strength = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
                    else:
                        strength = 1.0
                    shield_repaired, armor_repaired, cap_drained = self._apply_projected_support_effects(source, target, effect, dt, strength)
                    if shield_repaired > 0.0 or armor_repaired > 0.0 or cap_drained > 0.0:
                        self._add_projected_cycle_total(
                            source_ship_id=source.ship_id,
                            module_id=module.module_id,
                            target_ship_id=target.ship_id,
                            shield_repaired=shield_repaired,
                            armor_repaired=armor_repaired,
                            cap_drained=cap_drained,
                        )

        if self.detailed_logging and self.logger is not None:
            total_impacts = sum(len(v) for v in projected.values())
            self.logger.debug(f"combat_tick dt={dt:.4f} projected_impacts={total_impacts}")

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            for pending_target in list(ship.combat.fire_delay_timers.keys()):
                left = float(ship.combat.fire_delay_timers.get(pending_target, 0.0)) - dt
                ship.combat.fire_delay_timers[pending_target] = max(0.0, left)

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

            target_id = ship.combat.current_target
            if not target_id:
                continue
            target = world.ships.get(target_id)
            if target is None or not target.vital.alive:
                queue = list(world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), []))
                fallback_candidates: list[str] = []
                if ship.combat.last_attack_target:
                    fallback_candidates.append(ship.combat.last_attack_target)
                fallback_candidates.extend(queue)
                chosen_fallback: str | None = None
                for candidate_id in fallback_candidates:
                    candidate = world.ships.get(candidate_id)
                    if candidate is None or not candidate.vital.alive or candidate.team == ship.team:
                        continue
                    chosen_fallback = candidate_id
                    break
                if not chosen_fallback:
                    continue
                ship.combat.current_target = chosen_fallback
                target_id = chosen_fallback
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive:
                    continue

            target_profile = effective_profiles.get(target.ship_id, target.profile)
            turret_active_ratio, missile_active_ratio = self._weapon_activity_ratio(ship)

            distance = ship.nav.position.distance_to(target.nav.position)
            if distance > ship_profile.max_target_range:
                if self.detailed_logging and self.logger is not None:
                    self.logger.debug(
                        f"skip_out_of_lock_range attacker={ship.ship_id} target={target.ship_id} dist={distance:.1f} max_lock={ship_profile.max_target_range:.1f}"
                    )
                continue

            if target_id not in ship.combat.lock_targets:
                left = ship.combat.lock_timers.get(target_id)
                if left is None:
                    ship.combat.lock_timers[target_id] = self._cached_lock_time(ship_profile, target_profile)
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(
                            f"lock_start attacker={ship.ship_id} target={target.ship_id} lock_time={ship.combat.lock_timers[target_id]:.2f}"
                        )
                else:
                    left -= dt
                    if left <= 0:
                        ship.combat.lock_targets.add(target_id)
                        ship.combat.lock_timers.pop(target_id, None)
                        if target_id != ship.combat.last_attack_target:
                            ship.combat.fire_delay_timers[target_id] = self._sample_fire_delay(ship)
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(f"lock_complete attacker={ship.ship_id} target={target.ship_id}")
                    else:
                        ship.combat.lock_timers[target_id] = left
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(
                                f"lock_progress attacker={ship.ship_id} target={target.ship_id} remaining={left:.2f}"
                            )
                continue

            if target_id != ship.combat.last_attack_target:
                pending_delay = ship.combat.fire_delay_timers.get(target_id)
                if pending_delay is None:
                    ship.combat.fire_delay_timers[target_id] = self._sample_fire_delay(ship)
                    continue
                if pending_delay > 0:
                    continue

            relative_velocity = ship.nav.velocity - target.nav.velocity
            relative_speed = relative_velocity.length()
            radial = (target.nav.position - ship.nav.position).normalized()
            tangential = Vector2(-radial.y, radial.x)
            transversal = abs(relative_velocity.x * tangential.x + relative_velocity.y * tangential.y)
            angular_velocity = 0.0 if distance <= 0 else abs(transversal / distance)
            ctc_distance = ship.nav.radius + distance + target.nav.radius
            if ctc_distance > 0:
                angular_velocity = abs(transversal / ctc_distance)
            if self.detailed_logging and self.logger is not None:
                self.logger.debug(
                    f"kinematics attacker={ship.ship_id} target={target.ship_id}"
                    f" attacker_v=({ship.nav.velocity.x:.2f},{ship.nav.velocity.y:.2f})|{ship.nav.velocity.length():.2f}"
                    f" target_v=({target.nav.velocity.x:.2f},{target.nav.velocity.y:.2f})|{target.nav.velocity.length():.2f}"
                    f" rel_v=({relative_velocity.x:.2f},{relative_velocity.y:.2f})|{relative_speed:.2f}"
                    f" dist={distance:.2f} transv={transversal:.2f} angular={angular_velocity:.6f}"
                )
            turret_damage = (0.0, 0.0, 0.0, 0.0)
            range_factor = 0.0
            tracking_factor = 0.0
            expected_mult = 0.0
            turret_fired = False
            if ship_profile.turret_dps > 0 and turret_active_ratio > 0:
                if ship.combat.turret_reload_timer <= 0:
                    ship.combat.turret_reload_timer = max(0.0, ship_profile.turret_cycle)
                ship.combat.turret_reload_timer = max(0.0, ship.combat.turret_reload_timer - dt)
                chance = self.pyfa.turret_chance_to_hit(
                    tracking=ship_profile.tracking,
                    optimal_sig=max(1.0, ship_profile.optimal_sig),
                    distance=distance,
                    optimal=ship_profile.optimal,
                    falloff=ship_profile.falloff,
                    transversal_speed=transversal,
                    target_sig=target_profile.sig_radius,
                    attacker_radius=ship.nav.radius,
                    target_radius=target.nav.radius,
                )
                range_factor = self.pyfa.turret_range_factor(ship_profile.optimal, ship_profile.falloff, distance)
                tracking_factor = 0.0 if range_factor <= 0 else max(0.0, min(1.0, chance / max(1e-9, range_factor)))
                expected_mult = self.pyfa.turret_damage_multiplier(chance)
                if ship.combat.turret_reload_timer <= 0 and ship_profile.turret_cycle > 0:
                    turret_fired = True
                    turret_damage = _scale_damage(self._turret_damage(ship_profile, chance), expected_mult * turret_active_ratio)
                    ship.combat.turret_reload_timer += ship_profile.turret_cycle
                if self.detailed_logging and self.logger is not None:
                    tracking_term = 0.0
                    if ship_profile.tracking > 0 and target_profile.sig_radius > 0:
                        tracking_term = (angular_velocity * max(1.0, ship_profile.optimal_sig)) / (
                            ship_profile.tracking * target_profile.sig_radius
                        )
                    range_excess = max(0.0, distance - ship_profile.optimal)
                    range_term = range_excess / max(1e-6, ship_profile.falloff)
                    self.logger.debug(
                        f"turret_calc attacker={ship.ship_id} target={target.ship_id} dist={distance:.1f} transv={transversal:.1f} chance={chance:.4f} cycle={ship_profile.turret_cycle:.3f} fired={turret_fired} turret_dmg={_sum_damage(turret_damage):.3f}"
                    )
                    self.logger.debug(
                        f"turret_formula attacker={ship.ship_id} target={target.ship_id} "
                        f"chance~0.5^((tracking_term^2)+(range_term^2)); "
                        f"tracking_term={tracking_term:.6f}; range_term={range_term:.6f}; "
                        f"tracking={ship_profile.tracking:.6f}; optimal={ship_profile.optimal:.1f}; falloff={ship_profile.falloff:.1f}; target_sig={target_profile.sig_radius:.2f}; "
                        f"range_factor={range_factor:.6f}; tracking_factor={tracking_factor:.6f}; expected_mult={expected_mult:.6f}"
                    )

            missile_damage = (0.0, 0.0, 0.0, 0.0)
            missile_fired = False
            if ship_profile.missile_dps > 0 and missile_active_ratio > 0:
                if ship.combat.missile_reload_timer <= 0:
                    ship.combat.missile_reload_timer = max(0.0, ship_profile.missile_cycle)
                ship.combat.missile_reload_timer = max(0.0, ship.combat.missile_reload_timer - dt)
                if ship.combat.missile_reload_timer <= 0 and ship_profile.missile_cycle > 0:
                    missile_fired = True
                    missile_damage = _scale_damage(
                        self._missile_damage(ship_profile, target_profile, distance, relative_speed),
                        missile_active_ratio,
                    )
                    ship.combat.missile_reload_timer += ship_profile.missile_cycle
            if self.detailed_logging and self.logger is not None and ship_profile.missile_dps > 0:
                sig_factor = target_profile.sig_radius / max(1.0, ship_profile.missile_explosion_radius)
                vel_term = (sig_factor * ship_profile.missile_explosion_velocity) / max(1.0, relative_speed)
                vel_factor = vel_term ** max(0.1, ship_profile.missile_damage_reduction_factor)
                app_factor = self._missile_factor(ship_profile, target_profile, relative_speed)
                self.logger.debug(
                    f"missile_calc attacker={ship.ship_id} target={target.ship_id} dist={distance:.1f} rel_speed={relative_speed:.1f} cycle={ship_profile.missile_cycle:.3f} fired={missile_fired} missile_dmg={_sum_damage(missile_damage):.3f}"
                )
                self.logger.debug(
                    f"missile_formula attacker={ship.ship_id} target={target.ship_id} "
                    f"app=min(1,sig_factor,vel_factor); sig_factor={sig_factor:.6f}; vel_term={vel_term:.6f}; vel_factor={vel_factor:.6f}; app={app_factor:.6f}; "
                    f"explosion_radius={ship_profile.missile_explosion_radius:.2f}; explosion_velocity={ship_profile.missile_explosion_velocity:.2f}; drf={ship_profile.missile_damage_reduction_factor:.3f}"
                )
            dmg = (
                turret_damage[0] + missile_damage[0],
                turret_damage[1] + missile_damage[1],
                turret_damage[2] + missile_damage[2],
                turret_damage[3] + missile_damage[3],
            )
            if turret_fired or missile_fired:
                ship.combat.last_attack_target = target_id
                ship.combat.fire_delay_timers.pop(target_id, None)
            total_damage = _sum_damage(dmg)
            if total_damage <= 0:
                continue

            self._queue_merged_event(
                "active_module_cycle_effect",
                merge_fields={
                    "team": ship.team.value,
                    "squad": ship.squad_id,
                    "ship_type": ship.fit.ship_name,
                    "module": "weapon",
                    "weapon_system": ship_profile.weapon_system,
                    "target_type": target.fit.ship_name,
                    "turret_fired": turret_fired,
                    "missile_fired": missile_fired,
                },
                sum_fields={
                    "em": dmg[0],
                    "thermal": dmg[1],
                    "kinetic": dmg[2],
                    "explosive": dmg[3],
                    "total_damage": total_damage,
                },
            )

            if self.detailed_logging and self.logger is not None:
                best_turret_dph = ship_profile.turret_dps * max(0.0, ship_profile.turret_cycle) * self.pyfa.turret_damage_multiplier(1.0)
                best_missile_dph = ship_profile.missile_dps * max(0.0, ship_profile.missile_cycle)
                best_dph = best_turret_dph + best_missile_dph
                self.logger.debug(
                    f"dph_reference attacker={ship.ship_id} target={target.ship_id} "
                    f"best_range_turret={ship_profile.optimal:.1f} best_range_missile={ship_profile.missile_max_range:.1f} "
                    f"best_dph={best_dph:.4f} (turret={best_turret_dph:.4f}, missile={best_missile_dph:.4f}) "
                    f"cycle=(turret={ship_profile.turret_cycle:.3f},missile={ship_profile.missile_cycle:.3f})"
                )
                self.logger.debug(
                    f"dph_raw attacker={ship.ship_id} target={target.ship_id} dt={dt:.4f} fired=(turret={turret_fired},missile={missile_fired})"
                    f" turret=({_fmt_damage(turret_damage)}) missile=({_fmt_damage(missile_damage)}) total=({_fmt_damage(dmg)}) sum={_sum_damage(dmg):.4f}"
                )

            shield_before = target.vital.shield
            armor_before = target.vital.armor
            structure_before = target.vital.structure
            if self.detailed_logging and self.logger is not None:
                target.vital.shield, target.vital.armor, target.vital.structure, trace = _apply_damage_sequence_with_trace(
                    target.vital.shield,
                    target.vital.armor,
                    target.vital.structure,
                    dmg,
                    target_profile,
                )
                self.logger.debug(
                    f"damage_apply attacker={ship.ship_id} target={target.ship_id} raw={_sum_damage(dmg):.3f}"
                    f" hp_before=({shield_before:.1f},{armor_before:.1f},{structure_before:.1f})"
                    f" hp_after=({target.vital.shield:.1f},{target.vital.armor:.1f},{target.vital.structure:.1f})"
                    f" layers={trace}"
                )
                applied = (shield_before + armor_before + structure_before) - (
                    target.vital.shield + target.vital.armor + target.vital.structure
                )
                if applied > 0:
                    target.combat.last_damaged_at = world.now
                for item in trace:
                    self.logger.debug(
                        f"res_formula attacker={ship.ship_id} target={target.ship_id} layer={item['layer']} "
                        f"effective=(EM*ResEM + TH*ResTH + KI*ResKI + EX*ResEX) "
                        f"=({item['raw_em']:.4f}*{item['res_em']:.4f} + {item['raw_th']:.4f}*{item['res_th']:.4f} + "
                        f"{item['raw_ki']:.4f}*{item['res_ki']:.4f} + {item['raw_ex']:.4f}*{item['res_ex']:.4f}) "
                        f"=> applied={item['effective_applied']:.4f}, before={item['before']:.4f}, after={item['after']:.4f}, overflow_ratio={item['overflow_ratio']:.4f}"
                    )
            else:
                target.vital.shield, target.vital.armor, target.vital.structure = _apply_damage_sequence(
                    target.vital.shield,
                    target.vital.armor,
                    target.vital.structure,
                    dmg,
                    target_profile,
                )
                applied = (shield_before + armor_before + structure_before) - (
                    target.vital.shield + target.vital.armor + target.vital.structure
                )
                if applied > 0:
                    target.combat.last_damaged_at = world.now
            if target.vital.structure <= 0:
                target.vital.alive = False
                target.nav.velocity = Vector2(0.0, 0.0)
                if self.detailed_logging and self.logger is not None:
                    self.logger.debug(f"destroyed attacker={ship.ship_id} target={target.ship_id}")

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
