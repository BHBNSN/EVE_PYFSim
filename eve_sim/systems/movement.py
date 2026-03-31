from __future__ import annotations

import math

from ..math2d import Vector2
from ..models import WarpInterdictionSnapshot
from ..world import WorldState


class MovementSystem:
    AU_METERS = 149_597_870_700.0
    MIN_WARP_DISTANCE_M = 150_000.0
    WARP_BUBBLE_CATCH_WINDOW_M = 500_000.0
    WARP_ALIGNMENT_CONE_DEG = 5.0

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
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    @staticmethod
    def _ship_has_warp_request(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") in {"align", "warp"}

    @staticmethod
    def _ship_is_scrammed(ship) -> bool:
        profile = getattr(ship, "profile", None)
        if profile is None:
            return False
        return float(getattr(profile, "warp_scramble_status", 0.0) or 0.0) > 0.0

    @staticmethod
    def _ship_is_shuttle(ship) -> bool:
        profile = getattr(ship, "profile", None)
        if profile is None:
            return False
        return bool(getattr(profile, "is_shuttle", False))

    @staticmethod
    def _warp_bubble_immune_from_profile(ship) -> bool:
        profile = getattr(ship, "profile", None)
        if profile is None:
            return False
        return bool(getattr(profile, "warp_bubble_immune", False))

    @staticmethod
    def _field_allows_nullifier(interdiction_kind: str) -> bool:
        return str(interdiction_kind or "") == "probe"

    @classmethod
    def _bubble_field_affects_warp(
        cls,
        ship,
        *,
        interdiction_kind: str,
        bubble_immune_snapshot: bool,
    ) -> bool:
        if cls._ship_is_shuttle(ship):
            return False
        if bubble_immune_snapshot and cls._field_allows_nullifier(interdiction_kind):
            return False
        return True

    @staticmethod
    def _dot(a: Vector2, b: Vector2) -> float:
        return float(a.x) * float(b.x) + float(a.y) * float(b.y)

    @classmethod
    def _ray_circle_intersections(
        cls,
        origin: Vector2,
        direction: Vector2,
        center: Vector2,
        radius_m: float,
    ) -> tuple[float, float] | None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return None
        delta = center - origin
        projection = cls._dot(delta, direction)
        closest_sq = cls._dot(delta, delta) - (projection * projection)
        radius_sq = radius * radius
        if closest_sq > radius_sq:
            return None
        offset = math.sqrt(max(0.0, radius_sq - closest_sq))
        return projection - offset, projection + offset

    @classmethod
    def _capture_warp_interdiction_snapshots(cls, world: WorldState, ship) -> tuple[WarpInterdictionSnapshot, ...]:
        snapshots: list[WarpInterdictionSnapshot] = []
        now = float(world.now)
        for field in world.bubble_fields.values():
            if not field.alive or not field.blocks_warp:
                continue
            if field.anchor_ship_id is None and float(field.expires_at) <= now:
                continue
            snapshots.append(
                WarpInterdictionSnapshot(
                    field_id=str(field.field_id),
                    kind=str(field.kind),
                    interdiction_kind=str(field.interdiction_kind),
                    position=Vector2(field.position.x, field.position.y),
                    radius_m=max(0.0, float(field.radius_m or 0.0)),
                    blocks_warp=bool(field.blocks_warp),
                )
            )
        return tuple(snapshots)

    @classmethod
    def _ship_inside_current_warp_disruption(cls, world: WorldState, ship, bubble_immune_snapshot: bool) -> bool:
        now = float(world.now)
        for field in world.bubble_fields.values():
            if not field.alive or not field.blocks_warp:
                continue
            if field.anchor_ship_id is None and float(field.expires_at) <= now:
                continue
            if not cls._bubble_field_affects_warp(
                ship,
                interdiction_kind=str(field.interdiction_kind),
                bubble_immune_snapshot=bubble_immune_snapshot,
            ):
                continue
            if ship.nav.position.distance_to(field.position) <= max(0.0, float(field.radius_m or 0.0)):
                return True
        return False

    @classmethod
    def _warp_interception_distance(cls, ship, origin: Vector2, direction: Vector2, warp_distance_m: float) -> float | None:
        if warp_distance_m <= 0.0:
            return None
        snapshots = getattr(getattr(ship.nav, "warp", None), "interdiction_snapshots", ()) or ()
        bubble_immune_snapshot = bool(getattr(getattr(ship.nav, "warp", None), "bubble_immune_snapshot", False))
        best_entry_distance: float | None = None
        window_start = max(0.0, float(warp_distance_m) - cls.WARP_BUBBLE_CATCH_WINDOW_M)
        window_end = float(warp_distance_m) + cls.WARP_BUBBLE_CATCH_WINDOW_M
        for snapshot in snapshots:
            if not snapshot.blocks_warp:
                continue
            if not cls._bubble_field_affects_warp(
                ship,
                interdiction_kind=str(snapshot.interdiction_kind),
                bubble_immune_snapshot=bubble_immune_snapshot,
            ):
                continue
            intersections = cls._ray_circle_intersections(
                origin,
                direction,
                snapshot.position,
                float(snapshot.radius_m or 0.0),
            )
            if intersections is None:
                continue
            entry_distance, exit_distance = intersections
            if exit_distance < 0.0:
                continue
            entry_distance = max(0.0, entry_distance)
            if entry_distance > window_end or exit_distance < window_start:
                continue
            if best_entry_distance is None or entry_distance < best_entry_distance:
                best_entry_distance = entry_distance
        return best_entry_distance

    @classmethod
    def _bubble_speed_multiplier(cls, world: WorldState, ship) -> float:
        multiplier = 1.0
        now = float(world.now)
        for field in world.bubble_fields.values():
            if not field.alive:
                continue
            if field.anchor_ship_id is None and float(field.expires_at) <= now:
                continue
            speed_factor = max(0.01, float(field.speed_factor_mult or 1.0))
            if abs(speed_factor - 1.0) <= 1e-9:
                continue
            if ship.nav.position.distance_to(field.position) > max(0.0, float(field.radius_m or 0.0)):
                continue
            multiplier *= speed_factor
        return max(0.01, multiplier)

    @classmethod
    def _warp_time_seconds(cls, max_warp_speed_au_s: float, max_subwarp_speed_m_s: float, warp_distance_m: float) -> float:
        distance = max(0.0, float(warp_distance_m or 0.0))
        if distance <= 0.0:
            return 0.0
        max_warp_speed = max(1e-6, float(max_warp_speed_au_s or 0.0))
        max_subwarp_speed = max(0.0, float(max_subwarp_speed_m_s or 0.0))
        k_accel = max_warp_speed
        k_decel = min(max_warp_speed / 3.0, 2.0)
        warp_dropout_speed = min(max_subwarp_speed / 2.0, 100.0)
        warp_dropout_speed = max(1e-6, warp_dropout_speed)
        max_ms_warp_speed = max_warp_speed * cls.AU_METERS

        accel_dist = cls.AU_METERS
        decel_dist = max_ms_warp_speed / max(1e-6, k_decel)
        minimum_dist = accel_dist + decel_dist
        cruise_time = 0.0
        if minimum_dist > distance:
            max_ms_warp_speed = distance * k_accel * k_decel / max(1e-6, (k_accel + k_decel))
        else:
            cruise_time = (distance - minimum_dist) / max(1e-6, max_ms_warp_speed)

        accel_time = math.log(max_ms_warp_speed / max(1e-6, k_accel)) / max(1e-6, k_accel)
        decel_time = math.log(max_ms_warp_speed / warp_dropout_speed) / max(1e-6, k_decel)
        return max(0.0, cruise_time + accel_time + decel_time)

    @classmethod
    def _warp_distance_for_available_cap(cls, ship, requested_distance_m: float) -> tuple[float, float]:
        profile = getattr(ship, "profile", None)
        if profile is None:
            return 0.0, 0.0
        requested = max(0.0, float(requested_distance_m or 0.0))
        warp_capacitor_need = max(0.0, float(getattr(profile, "warp_capacitor_need", 0.0) or 0.0))
        mass = max(0.0, float(getattr(profile, "mass", 0.0) or 0.0))
        if requested <= 0.0:
            return 0.0, 0.0
        if warp_capacitor_need <= 0.0 or mass <= 0.0:
            return requested, 0.0

        available_cap = max(0.0, float(getattr(ship.vital, "cap", 0.0) or 0.0))
        cap_per_au = mass * warp_capacitor_need
        if available_cap <= 0.0 or cap_per_au <= 0.0:
            return 0.0, 0.0
        max_distance_au = available_cap / cap_per_au
        actual_distance = min(requested, max_distance_au * cls.AU_METERS)
        cap_cost = cap_per_au * (actual_distance / cls.AU_METERS)
        return actual_distance, min(available_cap, max(0.0, cap_cost))

    @staticmethod
    def _cancel_warp(ship) -> None:
        ship.nav.warp.phase = "idle"
        ship.nav.warp.target_position = None
        ship.nav.warp.target_ship_id = None
        ship.nav.warp.target_beacon_id = None
        ship.nav.warp.align_elapsed = 0.0
        ship.nav.warp.destination = None
        ship.nav.warp.origin = None
        ship.nav.warp.warp_distance_m = 0.0
        ship.nav.warp.warp_duration = 0.0
        ship.nav.warp.warp_elapsed = 0.0
        ship.nav.warp.capacitor_cost = 0.0
        ship.nav.warp.bubble_immune_snapshot = False
        ship.nav.warp.interdiction_snapshots_captured = False
        ship.nav.warp.interdiction_snapshots = tuple()
        ship.nav.command_target = None

    def _resolve_warp_target(self, world: WorldState, ship) -> tuple[Vector2 | None, float]:
        warp = ship.nav.warp
        if warp.target_ship_id:
            target_ship = world.ships.get(str(warp.target_ship_id))
            if target_ship is None or not target_ship.vital.alive:
                return None, 0.0
            landing_offset = max(0.0, float(getattr(target_ship.nav, "radius", 0.0) or 0.0)) + max(
                0.0, float(getattr(ship.nav, "radius", 0.0) or 0.0)
            )
            return Vector2(target_ship.nav.position.x, target_ship.nav.position.y), landing_offset
        if warp.target_beacon_id:
            beacon = world.beacons.get(str(warp.target_beacon_id))
            if beacon is None:
                return None, 0.0
            landing_offset = max(0.0, float(getattr(beacon, "radius", 0.0) or 0.0)) + max(
                0.0, float(getattr(ship.nav, "radius", 0.0) or 0.0)
            )
            return Vector2(beacon.position.x, beacon.position.y), landing_offset
        if warp.target_position is not None:
            return Vector2(warp.target_position.x, warp.target_position.y), 0.0
        return None, 0.0

    def _alignment_ready_for_warp(self, world: WorldState, ship, target_position: Vector2) -> bool:
        direction = target_position - ship.nav.position
        if direction.length() <= 1e-6:
            return True
        speed_cap = max(1.0, self._effective_speed_cap(world, ship))
        speed = ship.nav.velocity.length()
        if speed < (0.75 * speed_cap):
            return False
        move_angle = ship.nav.velocity.angle_deg() if speed > 1e-6 else float(ship.nav.facing_deg or 0.0)
        target_angle = direction.angle_deg()
        angle_error = abs(self._wrap_angle_deg(target_angle - move_angle))
        return angle_error <= self.WARP_ALIGNMENT_CONE_DEG

    def _start_warp(self, ship, target_position: Vector2, landing_offset: float) -> bool:
        to_target = target_position - ship.nav.position
        distance = to_target.length()
        if distance <= 1e-6:
            self._cancel_warp(ship)
            return False
        requested_distance = max(0.0, distance - max(0.0, float(landing_offset or 0.0)))
        actual_distance, cap_cost = self._warp_distance_for_available_cap(ship, requested_distance)
        if actual_distance <= 1e-6:
            self._cancel_warp(ship)
            return False

        direction = to_target.normalized()
        destination_distance = actual_distance
        interception_distance = self._warp_interception_distance(
            ship,
            Vector2(ship.nav.position.x, ship.nav.position.y),
            direction,
            actual_distance,
        )
        if interception_distance is not None:
            destination_distance = max(0.0, interception_distance)
        if actual_distance > 1e-6 and destination_distance < actual_distance:
            cap_cost *= destination_distance / actual_distance
        destination = ship.nav.position + direction * destination_distance
        if destination.length() > self.battlefield_radius:
            destination = destination.normalized() * self.battlefield_radius
            destination_distance = ship.nav.position.distance_to(destination)
            cap_cost = min(cap_cost, max(0.0, float(ship.vital.cap or 0.0)))

        warp_speed_au_s = max(0.1, float(getattr(ship.profile, "warp_speed_au_s", 0.0) or 0.0))
        subwarp_speed = max(1.0, float(getattr(ship.profile, "max_speed", ship.nav.max_speed) or ship.nav.max_speed))
        duration = max(0.05, self._warp_time_seconds(warp_speed_au_s, subwarp_speed, destination_distance))
        ship.vital.cap = max(0.0, float(ship.vital.cap) - cap_cost)
        ship.nav.warp.phase = "warp"
        ship.nav.warp.origin = Vector2(ship.nav.position.x, ship.nav.position.y)
        ship.nav.warp.destination = destination
        ship.nav.warp.warp_distance_m = destination_distance
        ship.nav.warp.warp_duration = duration
        ship.nav.warp.warp_elapsed = 0.0
        ship.nav.warp.capacitor_cost = cap_cost
        ship.nav.command_target = None
        average_speed = destination_distance / max(1e-6, duration)
        ship.nav.velocity = direction * average_speed
        ship.nav.facing_deg = direction.angle_deg()
        return True

    def _prepare_warp_alignment(self, world: WorldState, ship) -> None:
        if str(ship.nav.warp.phase or "idle") != "align":
            return
        if self._ship_is_scrammed(ship):
            self._cancel_warp(ship)
            return
        target_position, landing_offset = self._resolve_warp_target(world, ship)
        if target_position is None:
            self._cancel_warp(ship)
            return
        ship.nav.command_target = Vector2(target_position.x, target_position.y)
        ship.nav.warp.target_position = Vector2(target_position.x, target_position.y)
        if not bool(getattr(ship.nav.warp, "interdiction_snapshots_captured", False)):
            ship.nav.warp.bubble_immune_snapshot = self._warp_bubble_immune_from_profile(ship)
            ship.nav.warp.interdiction_snapshots = self._capture_warp_interdiction_snapshots(world, ship)
            ship.nav.warp.interdiction_snapshots_captured = True
        if max(0.0, ship.nav.position.distance_to(target_position) - landing_offset) < self.MIN_WARP_DISTANCE_M:
            self._cancel_warp(ship)

    def _finalize_warp_alignment(self, world: WorldState, ship, dt: float) -> None:
        if str(ship.nav.warp.phase or "idle") != "align":
            return
        if self._ship_is_scrammed(ship):
            self._cancel_warp(ship)
            return
        target_position, landing_offset = self._resolve_warp_target(world, ship)
        if target_position is None:
            self._cancel_warp(ship)
            return
        remaining_distance = max(0.0, ship.nav.position.distance_to(target_position) - landing_offset)
        if remaining_distance < self.MIN_WARP_DISTANCE_M:
            self._cancel_warp(ship)
            return
        ship.nav.warp.target_position = Vector2(target_position.x, target_position.y)
        ship.nav.warp.align_elapsed = max(0.0, float(ship.nav.warp.align_elapsed or 0.0)) + max(0.0, float(dt))
        if self._ship_inside_current_warp_disruption(world, ship, bool(ship.nav.warp.bubble_immune_snapshot)):
            return
        if self._alignment_ready_for_warp(world, ship, target_position) or ship.nav.warp.align_elapsed >= float(ship.nav.warp.align_timeout):
            self._start_warp(ship, target_position, landing_offset)

    def _advance_in_warp(self, ship, dt: float) -> None:
        if str(ship.nav.warp.phase or "idle") != "warp":
            return
        origin = ship.nav.warp.origin
        destination = ship.nav.warp.destination
        duration = max(0.0, float(ship.nav.warp.warp_duration or 0.0))
        if origin is None or destination is None or duration <= 1e-6:
            ship.nav.position = destination if destination is not None else ship.nav.position
            ship.nav.velocity = Vector2(0.0, 0.0)
            self._cancel_warp(ship)
            return

        ship.nav.warp.warp_elapsed = min(duration, float(ship.nav.warp.warp_elapsed or 0.0) + max(0.0, float(dt)))
        progress = max(0.0, min(1.0, ship.nav.warp.warp_elapsed / duration))
        travel = destination - origin
        ship.nav.position = origin + travel * progress
        direction = travel.normalized()
        if progress >= 1.0:
            ship.nav.position = Vector2(destination.x, destination.y)
            ship.nav.velocity = Vector2(0.0, 0.0)
            ship.nav.facing_deg = direction.angle_deg() if direction.length() > 0.0 else ship.nav.facing_deg
            self._cancel_warp(ship)
            return

        average_speed = max(0.0, float(ship.nav.warp.warp_distance_m or 0.0) / duration)
        ship.nav.velocity = direction * average_speed
        if direction.length() > 0.0:
            ship.nav.facing_deg = direction.angle_deg()

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
            return max(1.0, base_cap * self._bubble_speed_multiplier(world, ship))
        cap = float(world.squad_leader_speed_limits.get(squad_key, 0.0) or 0.0)
        if cap <= 0.0:
            return max(1.0, base_cap * self._bubble_speed_multiplier(world, ship))
        return max(1.0, min(base_cap, cap) * self._bubble_speed_multiplier(world, ship))

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

            if self._ship_in_warp(ship):
                self._advance_in_warp(ship, dt)
                continue

            self._prepare_warp_alignment(world, ship)

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
            self._finalize_warp_alignment(world, ship, dt)

