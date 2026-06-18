from __future__ import annotations

import math
import random

from ..math2d import Vector2
from ..models import WarpInterdictionSnapshot
from ..world import WorldState


class MovementSystem:
    AU_METERS = 149_597_870_700.0
    DEFAULT_SYSTEM_RADIUS_M = 30.0 * AU_METERS
    MIN_WARP_DISTANCE_M = 150_000.0
    WARP_BUBBLE_CATCH_WINDOW_M = 500_000.0
    WARP_ALIGNMENT_CONE_DEG = 5.0
    STARGATE_USE_DISTANCE_M = 2_500.0
    STARGATE_JUMP_OFFSET_MIN_M = 10_000.0
    STARGATE_JUMP_OFFSET_MAX_M = 15_000.0
    STARGATE_GATE_CLOAK_SEC = 60.0

    def __init__(self) -> None:
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
    def _random_point_in_radius(center: Vector2, radius: float) -> Vector2:
        theta = random.uniform(0.0, math.tau)
        distance = max(0.0, float(radius)) * math.sqrt(random.random())
        return Vector2(center.x + math.cos(theta) * distance, center.y + math.sin(theta) * distance)

    @staticmethod
    def _random_point_in_annulus(center: Vector2, min_radius: float, max_radius: float) -> Vector2:
        minimum = max(0.0, float(min_radius))
        maximum = max(minimum, float(max_radius))
        theta = random.uniform(0.0, math.tau)
        distance = math.sqrt(random.uniform(minimum * minimum, maximum * maximum))
        return Vector2(center.x + math.cos(theta) * distance, center.y + math.sin(theta) * distance)

    @staticmethod
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    @staticmethod
    def _ship_is_gate_cloaked(ship, now: float | None = None) -> bool:
        cloak = getattr(ship.nav, "cloak", None)
        if cloak is None or not bool(getattr(cloak, "active", False)):
            return False
        if now is not None and float(getattr(cloak, "expires_at", 0.0) or 0.0) <= float(now):
            cloak.active = False
            cloak.expires_at = 0.0
            cloak.source = ""
            return False
        return True

    @staticmethod
    def _entity_system_id(entity) -> str:
        nav = getattr(entity, "nav", None)
        if nav is not None:
            return str(getattr(nav, "system_id", "") or "")
        return str(getattr(entity, "system_id", "") or "")

    @staticmethod
    def _clear_gate_cloak(ship) -> None:
        cloak = getattr(ship.nav, "cloak", None)
        if cloak is None:
            return
        cloak.active = False
        cloak.expires_at = 0.0
        cloak.source = ""

    @staticmethod
    def _clear_gate_transit(ship) -> None:
        gate = getattr(ship.nav, "gate", None)
        if gate is None:
            return
        gate.target_structure_id = None

    @staticmethod
    def _clear_navigation_command(ship) -> None:
        ship.nav.command_target = None
        ship.nav.command_mode = "move"
        ship.nav.command_target_ship_id = None
        ship.nav.command_target_structure_id = None
        ship.nav.command_range_m = 0.0
        ship.nav.command_orbit_clockwise = True

    @staticmethod
    def _edge_distance_to_structure(position: Vector2, structure) -> float:
        radius = max(0.0, float(getattr(structure, "radius", 0.0) or 0.0))
        return max(0.0, position.distance_to(structure.position) - radius)

    @staticmethod
    def _system_definition(world: WorldState, system_id: str):
        map_definition = getattr(world, "map_definition", None)
        if map_definition is None:
            return None
        try:
            return map_definition.system_by_id(system_id)
        except Exception:
            return None

    def _system_center_and_radius(self, world: WorldState, ship) -> tuple[Vector2, float]:
        system = self._system_definition(world, str(getattr(ship.nav, "system_id", "") or ""))
        if system is None:
            return Vector2(0.0, 0.0), float(self.DEFAULT_SYSTEM_RADIUS_M)
        return Vector2(0.0, 0.0), max(1_000.0, float(system.radius_m or 1_000.0))

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
        ship_system_id = cls._entity_system_id(ship)
        for field in world.bubble_fields.values():
            if str(getattr(field, "system_id", "") or "") != ship_system_id:
                continue
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
        ship_system_id = cls._entity_system_id(ship)
        for field in world.bubble_fields.values():
            if str(getattr(field, "system_id", "") or "") != ship_system_id:
                continue
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
        ship_system_id = cls._entity_system_id(ship)
        for field in world.bubble_fields.values():
            if str(getattr(field, "system_id", "") or "") != ship_system_id:
                continue
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
        ship.nav.command_mode = "move"
        ship.nav.command_target_ship_id = None
        ship.nav.command_target_structure_id = None
        ship.nav.command_range_m = 0.0
        ship.nav.command_orbit_clockwise = True

    def _resolve_warp_target(self, world: WorldState, ship) -> tuple[Vector2 | None, float]:
        warp = ship.nav.warp
        source_system_id = str(getattr(ship.nav, "system_id", "") or "")
        if warp.target_ship_id:
            target_ship = world.combat_entity(str(warp.target_ship_id))
            if target_ship is None or not target_ship.vital.alive:
                return None, 0.0
            if str(getattr(target_ship.nav, "system_id", "") or "") != source_system_id:
                return None, 0.0
            landing_offset = max(0.0, float(getattr(target_ship.nav, "radius", 0.0) or 0.0)) + max(
                0.0, float(getattr(ship.nav, "radius", 0.0) or 0.0)
            )
            return Vector2(target_ship.nav.position.x, target_ship.nav.position.y), landing_offset
        if warp.target_beacon_id:
            beacon = world.structures.get(str(warp.target_beacon_id))
            if beacon is None:
                return None, 0.0
            if str(getattr(beacon, "system_id", "") or "") != source_system_id:
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

    def _start_warp(self, world: WorldState, ship, target_position: Vector2, landing_offset: float) -> bool:
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
        system_center, system_radius = self._system_center_and_radius(world, ship)
        relative_destination = destination - system_center
        if relative_destination.length() > system_radius:
            destination = system_center + relative_destination.normalized() * system_radius
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
        ship.nav.command_mode = "move"
        ship.nav.command_target_ship_id = None
        ship.nav.command_target_structure_id = None
        ship.nav.command_range_m = 0.0
        ship.nav.command_orbit_clockwise = True
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
            self._start_warp(world, ship, target_position, landing_offset)

    def _advance_in_warp(self, world: WorldState, ship, dt: float) -> None:
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

    def _activate_stargate_jump(self, world: WorldState, ship, source_gate, destination_gate) -> None:
        del source_gate
        destination_position = self._random_point_in_annulus(
            Vector2(destination_gate.position.x, destination_gate.position.y),
            self.STARGATE_JUMP_OFFSET_MIN_M,
            self.STARGATE_JUMP_OFFSET_MAX_M,
        )
        ship.nav.system_id = str(getattr(destination_gate, "system_id", "") or ship.nav.system_id)
        ship.nav.position = destination_position
        ship.nav.velocity = Vector2(0.0, 0.0)
        ship.nav.command_target = None
        self._cancel_warp(ship)
        self._clear_gate_transit(ship)
        cloak = getattr(ship.nav, "cloak", None)
        if cloak is not None:
            cloak.active = True
            cloak.expires_at = float(world.now) + self.STARGATE_GATE_CLOAK_SEC
            cloak.source = "stargate"

        leader_key = f"{ship.team.value}:{ship.squad_id}"
        leader_id = str(world.squad_leaders.get(leader_key, "") or "")
        if leader_id and leader_id != ship.ship_id:
            ship.nav.follow_hold_active = True
            ship.nav.follow_hold_leader_id = leader_id
        else:
            ship.nav.follow_hold_active = False
            ship.nav.follow_hold_leader_id = None

    def _prepare_gate_transit(self, world: WorldState, ship) -> None:
        target_structure_id = str(getattr(getattr(ship.nav, "gate", None), "target_structure_id", "") or "").strip()
        if not target_structure_id:
            return
        structure = world.structures.get(target_structure_id)
        if structure is None or str(getattr(structure, "kind", "") or "").upper() != "STARGATE":
            self._clear_gate_transit(ship)
            self._clear_navigation_command(ship)
            return
        if str(getattr(structure, "system_id", "") or "") != str(getattr(ship.nav, "system_id", "") or ""):
            self._clear_gate_transit(ship)
            self._clear_navigation_command(ship)
            return
        linked_id = str(getattr(structure, "linked_structure_id", "") or "").strip()
        if not linked_id or world.structures.get(linked_id) is None:
            self._clear_gate_transit(ship)
            self._clear_navigation_command(ship)
            return

        activation_range = max(
            float(getattr(structure, "interaction_range", 0.0) or 0.0),
            float(getattr(getattr(ship.nav, "gate", None), "activation_range_m", self.STARGATE_USE_DISTANCE_M) or self.STARGATE_USE_DISTANCE_M),
        )
        ship.nav.gate.activation_range_m = activation_range
        if self._edge_distance_to_structure(ship.nav.position, structure) <= activation_range:
            self._activate_stargate_jump(world, ship, structure, world.structures[linked_id])
            return
        ship.nav.command_mode = "approach"
        ship.nav.command_target_ship_id = None
        ship.nav.command_target_structure_id = str(target_structure_id)
        ship.nav.command_range_m = activation_range
        ship.nav.command_target = Vector2(structure.position.x, structure.position.y)

    def _update_gate_cloak(self, world: WorldState, ship) -> None:
        if not self._ship_is_gate_cloaked(ship, float(world.now)):
            return
        warp_phase = str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle")
        gate_target_id = str(getattr(getattr(ship.nav, "gate", None), "target_structure_id", "") or "").strip()
        if (
            warp_phase != "idle"
            or ship.nav.command_target is not None
            or gate_target_id
            or ship.nav.velocity.length() > 5.0
        ):
            self._clear_gate_cloak(ship)

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

    @classmethod
    def _stable_orbit_speed(cls, radius_m: float, speed_cap: float, tau: float) -> float:
        radius = max(0.0, float(radius_m))
        cap = max(0.0, float(speed_cap))
        if radius <= 1e-6 or cap <= 1e-6:
            return 0.0
        lo = 0.0
        hi = cap * 0.999
        for _ in range(40):
            mid = (lo + hi) * 0.5
            stable_radius = cls._stable_turn_radius(mid, cap, tau)
            if stable_radius <= radius:
                lo = mid
            else:
                hi = mid
        return max(0.0, min(cap, lo))

    @staticmethod
    def _rotate_90(v: Vector2, clockwise: bool) -> Vector2:
        if clockwise:
            return Vector2(v.y, -v.x)
        return Vector2(-v.y, v.x)

    @staticmethod
    def _clamped_vector(v: Vector2, max_length: float) -> Vector2:
        length = v.length()
        cap = max(0.0, float(max_length))
        if length <= cap or length <= 1e-9:
            return v
        return v.normalized() * cap

    @staticmethod
    def _command_target_info(world: WorldState, ship) -> tuple[Vector2 | None, float, Vector2]:
        target_ship_id = str(getattr(ship.nav, "command_target_ship_id", "") or "").strip()
        if target_ship_id:
            target_ship = world.combat_entity(target_ship_id)
            if target_ship is None or not target_ship.vital.alive:
                return None, 0.0, Vector2(0.0, 0.0)
            if str(getattr(target_ship.nav, "system_id", "") or "") != str(getattr(ship.nav, "system_id", "") or ""):
                return None, 0.0, Vector2(0.0, 0.0)
            return (
                Vector2(target_ship.nav.position.x, target_ship.nav.position.y),
                max(0.0, float(getattr(target_ship.nav, "radius", 0.0) or 0.0)),
                Vector2(target_ship.nav.velocity.x, target_ship.nav.velocity.y),
            )
        target_structure_id = str(getattr(ship.nav, "command_target_structure_id", "") or "").strip()
        if target_structure_id:
            structure = world.structures.get(target_structure_id)
            if structure is None:
                return None, 0.0, Vector2(0.0, 0.0)
            if str(getattr(structure, "system_id", "") or "") != str(getattr(ship.nav, "system_id", "") or ""):
                return None, 0.0, Vector2(0.0, 0.0)
            return (
                Vector2(structure.position.x, structure.position.y),
                max(0.0, float(getattr(structure, "radius", 0.0) or 0.0)),
                Vector2(0.0, 0.0),
            )
        target = getattr(ship.nav, "command_target", None)
        if target is None:
            return None, 0.0, Vector2(0.0, 0.0)
        return Vector2(target.x, target.y), 0.0, Vector2(0.0, 0.0)

    def _relative_command_velocity(self, target_velocity: Vector2, relative_velocity: Vector2, speed_cap: float) -> Vector2:
        if target_velocity.length() <= 1e-6:
            return self._clamped_vector(relative_velocity, speed_cap)
        return self._clamped_vector(target_velocity + relative_velocity, speed_cap)

    def _desired_navigation_velocity(self, world: WorldState, ship, speed_cap: float, tau: float) -> tuple[Vector2, float]:
        current_velocity = ship.nav.velocity
        current_speed = current_velocity.length()
        target_position, target_radius, target_velocity = self._command_target_info(world, ship)
        if target_position is None:
            self._clear_navigation_command(ship)
            return Vector2(0.0, 0.0), float(ship.nav.facing_deg or 0.0)
        ship.nav.command_target = target_position

        to_target = target_position - ship.nav.position
        center_distance = to_target.length()
        if center_distance <= 1e-6:
            radial = self._heading_vector(float(ship.nav.facing_deg or 0.0))
        else:
            radial = to_target.normalized()
        desired_angle = radial.angle_deg()
        mode = str(getattr(ship.nav, "command_mode", "move") or "move").strip().lower()
        own_radius = max(0.0, float(getattr(ship.nav, "radius", 0.0) or 0.0))
        edge_distance = max(0.0, center_distance - target_radius - own_radius)
        requested_range = max(0.0, float(getattr(ship.nav, "command_range_m", 0.0) or 0.0))

        if mode == "move":
            arrive_radius = max(120.0, own_radius * 1.5)
            if center_distance <= arrive_radius:
                return Vector2(0.0, 0.0), desired_angle
            return radial * speed_cap, desired_angle

        if mode == "approach":
            stop_range = max(0.0, requested_range)
            if edge_distance <= max(120.0, stop_range):
                desired = self._relative_command_velocity(target_velocity, Vector2(0.0, 0.0), speed_cap)
                return desired, desired.angle_deg() if desired.length() > 1e-6 else desired_angle
            desired = self._relative_command_velocity(target_velocity, radial * speed_cap, speed_cap)
            return desired, desired.angle_deg() if desired.length() > 1e-6 else desired_angle

        if mode == "keep_range":
            error = edge_distance - requested_range
            tolerance = max(100.0, min(1_000.0, requested_range * 0.05))
            if abs(error) <= tolerance:
                desired = self._relative_command_velocity(target_velocity, Vector2(0.0, 0.0), speed_cap)
                return desired, desired.angle_deg() if desired.length() > 1e-6 else current_velocity.angle_deg() if current_speed > 1e-6 else desired_angle
            direction = radial if error > 0.0 else radial * -1.0
            ramp = max(500.0, requested_range * 0.25)
            speed = speed_cap * max(0.15, min(1.0, abs(error) / ramp))
            desired = self._relative_command_velocity(target_velocity, direction * speed, speed_cap)
            return desired, desired.angle_deg() if desired.length() > 1e-6 else direction.angle_deg()

        if mode == "orbit":
            orbit_radius = max(target_radius + own_radius + requested_range, own_radius + target_radius + 100.0)
            if center_distance <= 1e-6:
                return Vector2(0.0, 0.0), desired_angle
            tangent = self._rotate_90(radial, bool(getattr(ship.nav, "command_orbit_clockwise", True)))
            orbit_speed = self._stable_orbit_speed(orbit_radius, speed_cap, tau)
            relative_current_velocity = current_velocity - target_velocity
            relative_current_speed = relative_current_velocity.length()
            radial_error = center_distance - orbit_radius
            entry_margin = max(500.0, min(20_000.0, orbit_radius * 0.18 + relative_current_speed * tau * 0.35))
            if radial_error > entry_margin:
                desired = self._relative_command_velocity(target_velocity, radial * speed_cap, speed_cap)
                return desired, desired.angle_deg() if desired.length() > 1e-6 else desired_angle
            if radial_error < -entry_margin:
                outward = radial * -speed_cap
                desired = self._relative_command_velocity(target_velocity, outward, speed_cap)
                return desired, desired.angle_deg() if desired.length() > 1e-6 else outward.angle_deg()

            tangent_factor = 1.0 if entry_margin <= 1e-6 else max(0.0, min(1.0, 1.0 - abs(radial_error) / entry_margin))
            correction_time = max(0.5, min(8.0, tau * 0.75))
            radial_speed = max(-speed_cap * 0.65, min(speed_cap * 0.65, radial_error / correction_time))
            relative_desired = tangent * (orbit_speed * tangent_factor) + radial * radial_speed
            desired = self._relative_command_velocity(target_velocity, relative_desired, speed_cap)
            if desired.length() <= 1e-6:
                return Vector2(0.0, 0.0), tangent.angle_deg()
            return desired, desired.angle_deg()

        return radial * speed_cap, desired_angle

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
        speed_cap = self._effective_speed_cap(world, ship)
        desired_angle = ship.nav.facing_deg
        current_velocity = ship.nav.velocity
        current_speed = current_velocity.length()
        target_speed = 0.0
        desired_velocity = Vector2(0.0, 0.0)

        if ship.nav.command_target is not None:
            tau = self._motion_tau(ship, speed_cap)
            desired_velocity, desired_angle = self._desired_navigation_velocity(world, ship, speed_cap, tau)
            target_speed = desired_velocity.length()
        elif bool(ship.nav.propulsion_command_active):
            if current_speed > 1e-6:
                # Keep burning along the existing travel vector when propulsion toggles on mid-flight.
                desired_angle = current_velocity.angle_deg()
                target_speed = speed_cap
                desired_velocity = self._heading_vector(desired_angle) * target_speed

        tau = self._motion_tau(ship, speed_cap)
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

            self._update_gate_cloak(world, ship)

            if self._ship_in_warp(ship):
                self._advance_in_warp(world, ship, dt)
                continue

            self._prepare_gate_transit(world, ship)
            if self._ship_in_warp(ship):
                self._advance_in_warp(world, ship, dt)
                continue

            self._prepare_warp_alignment(world, ship)

            displacement = self._update_velocity_with_inertia(world, ship, dt)
            next_pos = ship.nav.position + displacement
            system_center, system_radius = self._system_center_and_radius(world, ship)
            relative_next = next_pos - system_center
            if relative_next.length() > system_radius:
                n = relative_next.normalized()
                next_pos = system_center + n * system_radius
                ship.nav.velocity = Vector2(0.0, 0.0)

            for beacon in world.structures.values():
                if str(getattr(beacon, "system_id", "") or "") != str(getattr(ship.nav, "system_id", "") or ""):
                    continue
                dist = next_pos.distance_to(beacon.position)
                if dist < beacon.radius + ship.nav.radius:
                    push_dir = (next_pos - beacon.position).normalized()
                    if push_dir.length() == 0:
                        push_dir = Vector2(1.0, 0.0)
                    next_pos = beacon.position + push_dir * (beacon.radius + ship.nav.radius)

            ship.nav.position = next_pos
            self._prepare_gate_transit(world, ship)
            if self._ship_in_warp(ship):
                self._advance_in_warp(world, ship, 0.0)
                continue
            self._finalize_warp_alignment(world, ship, dt)

