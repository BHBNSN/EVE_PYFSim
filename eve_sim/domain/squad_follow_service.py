from __future__ import annotations

from collections import deque

from ..math2d import Vector2
from ..models import ShipEntity, SquadLeaderLocation
from ..squad_identity import squad_key
from ..world import WorldState

FOLLOW_LEADER_SYSTEM = "FOLLOW_LEADER_SYSTEM"
WARP_TO_LEADER = "WARP_TO_LEADER"
FORMATION_FOLLOW = "FORMATION_FOLLOW"
WARP_FOLLOW_TRIGGER_DISTANCE_M = 170_000.0
WARP_FOLLOW_RESET_DISTANCE_M = 150_000.0

FOLLOW_TRANSIT_STATES = frozenset({FOLLOW_LEADER_SYSTEM, WARP_TO_LEADER})


class SquadFollowService:
    """Apply the global-leader follow state machine for one ship decision."""

    @staticmethod
    def _clear_navigation_command(ship: ShipEntity) -> None:
        ship.nav.command_target = None
        ship.nav.command_mode = "move"
        ship.nav.command_target_ship_id = None
        ship.nav.command_target_structure_id = None
        ship.nav.command_range_m = 0.0
        ship.nav.command_orbit_clockwise = True

    @classmethod
    def _cancel_warp(cls, ship: ShipEntity) -> None:
        warp = ship.nav.warp
        warp.phase = "idle"
        warp.target_position = None
        warp.target_ship_id = None
        warp.target_beacon_id = None
        warp.align_elapsed = 0.0
        warp.origin = None
        warp.destination = None
        warp.warp_distance_m = 0.0
        warp.warp_duration = 0.0
        warp.warp_elapsed = 0.0
        warp.capacitor_cost = 0.0
        warp.bubble_immune_snapshot = False
        warp.interdiction_snapshots_captured = False
        warp.interdiction_snapshots = tuple()
        cls._clear_navigation_command(ship)

    @classmethod
    def _begin_warp(
        cls,
        ship: ShipEntity,
        target_position: Vector2,
        *,
        target_ship_id: str | None = None,
        target_beacon_id: str | None = None,
    ) -> None:
        cls._cancel_warp(ship)
        warp = ship.nav.warp
        warp.phase = "align"
        warp.target_position = Vector2(target_position.x, target_position.y)
        warp.target_ship_id = target_ship_id
        warp.target_beacon_id = target_beacon_id

    @staticmethod
    def _clear_combat_for_follow(ship: ShipEntity) -> None:
        ship.order_queue = [
            order
            for order in ship.order_queue
            if order.kind not in {"ATTACK", "MOVE", "WARP", "USE_STARGATE"}
        ]
        ship.combat.current_target = None
        ship.combat.last_attack_target = None
        ship.combat.projected_targets.clear()

    @staticmethod
    def _next_gate_toward(world: WorldState, source_system: str, target_system: str):
        if not source_system or not target_system or source_system == target_system:
            return None
        edges: dict[str, list[tuple[str, str]]] = {}
        for gate_id, gate in world.structures.items():
            if str(getattr(gate, "kind", "") or "").upper() != "STARGATE":
                continue
            linked_id = str(getattr(gate, "linked_structure_id", "") or "").strip()
            linked = world.structures.get(linked_id)
            if linked is None:
                continue
            gate_system = str(getattr(gate, "system_id", "") or "")
            linked_system = str(getattr(linked, "system_id", "") or "")
            if gate_system and linked_system and gate_system != linked_system:
                edges.setdefault(gate_system, []).append((linked_system, str(gate_id)))
        for values in edges.values():
            values.sort(key=lambda item: (item[0], item[1]))

        pending = deque([(source_system, None)])
        visited = {source_system}
        while pending:
            current_system, first_gate_id = pending.popleft()
            for next_system, gate_id in edges.get(current_system, []):
                if next_system in visited:
                    continue
                next_first_gate = first_gate_id or gate_id
                if next_system == target_system:
                    return world.structures.get(next_first_gate)
                visited.add(next_system)
                pending.append((next_system, next_first_gate))
        return None

    @staticmethod
    def _leader_location(world: WorldState, ship: ShipEntity) -> SquadLeaderLocation | None:
        return world.squad_leader_locations.get(squad_key(ship.team, ship.squad_id))

    @classmethod
    def _configure_cross_system_follow(
        cls,
        world: WorldState,
        ship: ShipEntity,
        location: SquadLeaderLocation,
    ) -> None:
        cls._clear_combat_for_follow(ship)
        ship.nav.squad_follow_state = FOLLOW_LEADER_SYSTEM
        ship.nav.squad_follow_leader_id = location.leader_id
        ship.nav.squad_follow_leader_location_version = int(location.location_version)

        source_system = str(ship.nav.system_id or "")
        gate = cls._next_gate_toward(world, source_system, location.system_id)
        if gate is None:
            cls._cancel_warp(ship)
            ship.nav.gate.target_structure_id = None
            return

        gate_id = str(gate.structure_id)
        warp_phase = str(ship.nav.warp.phase or "idle")
        if warp_phase != "idle" and str(ship.nav.warp.target_beacon_id or "") != gate_id:
            cls._cancel_warp(ship)
            warp_phase = "idle"
        ship.nav.gate.target_structure_id = gate_id
        distance = ship.nav.position.distance_to(gate.position)
        if (
            distance > WARP_FOLLOW_TRIGGER_DISTANCE_M
            and warp_phase == "idle"
            and float(getattr(ship.profile, "warp_scramble_status", 0.0) or 0.0) <= 0.0
        ):
            cls._begin_warp(ship, gate.position, target_beacon_id=gate_id)
            ship.nav.gate.target_structure_id = gate_id
        elif distance <= WARP_FOLLOW_TRIGGER_DISTANCE_M and warp_phase == "align":
            cls._cancel_warp(ship)
            ship.nav.gate.target_structure_id = gate_id

    @classmethod
    def apply(
        cls,
        world: WorldState,
        ship: ShipEntity,
        leader: ShipEntity | None,
    ) -> bool:
        """Return true when transit following owns this ship's current decision."""
        location = cls._leader_location(world, ship)
        if location is None:
            ship.nav.squad_follow_state = FORMATION_FOLLOW
            ship.nav.squad_follow_leader_id = None
            return False

        version_changed = (
            ship.nav.squad_follow_leader_id != location.leader_id
            or int(ship.nav.squad_follow_leader_location_version) != int(location.location_version)
        )
        if version_changed and ship.nav.squad_follow_leader_id is not None:
            cls._cancel_warp(ship)
            ship.nav.gate.target_structure_id = None

        ship.nav.squad_follow_leader_id = location.leader_id
        ship.nav.squad_follow_leader_location_version = int(location.location_version)
        if ship.ship_id == location.leader_id:
            ship.nav.squad_follow_state = FORMATION_FOLLOW
            ship.nav.squad_follow_warp_ready = True
            return False

        if str(ship.nav.system_id or "") != location.system_id:
            cls._configure_cross_system_follow(world, ship, location)
            return True

        if leader is None or not leader.vital.alive or str(leader.nav.system_id or "") != str(ship.nav.system_id or ""):
            cls._configure_cross_system_follow(world, ship, location)
            return True

        ship.nav.gate.target_structure_id = None
        distance = ship.nav.position.distance_to(leader.nav.position)
        warp_phase = str(ship.nav.warp.phase or "idle")
        if warp_phase != "idle" and str(ship.nav.warp.target_ship_id or "") != leader.ship_id:
            cls._clear_combat_for_follow(ship)
            ship.nav.squad_follow_state = WARP_TO_LEADER
            return True

        if distance < WARP_FOLLOW_RESET_DISTANCE_M:
            if warp_phase == "align":
                cls._cancel_warp(ship)
            ship.nav.squad_follow_state = FORMATION_FOLLOW
            ship.nav.squad_follow_warp_ready = True
            return False

        if distance > WARP_FOLLOW_TRIGGER_DISTANCE_M:
            cls._clear_combat_for_follow(ship)
            ship.nav.squad_follow_state = WARP_TO_LEADER
            if (
                warp_phase == "idle"
                and bool(ship.nav.squad_follow_warp_ready)
                and float(getattr(ship.profile, "warp_scramble_status", 0.0) or 0.0) <= 0.0
            ):
                cls._begin_warp(ship, leader.nav.position, target_ship_id=leader.ship_id)
                ship.nav.squad_follow_warp_ready = False
            return True

        if ship.nav.squad_follow_state == WARP_TO_LEADER:
            cls._clear_combat_for_follow(ship)
            return True
        ship.nav.squad_follow_state = FORMATION_FOLLOW
        return False
