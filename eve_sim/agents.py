from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random

from .math2d import Vector2
from .models import FleetIntent, Order, ShipEntity, SquadLeaderLocation, Team
from .world import WorldState


FOLLOW_LEADER_SYSTEM = "FOLLOW_LEADER_SYSTEM"
WARP_TO_LEADER = "WARP_TO_LEADER"
FORMATION_FOLLOW = "FORMATION_FOLLOW"
WARP_FOLLOW_TRIGGER_DISTANCE_M = 170_000.0
WARP_FOLLOW_RESET_DISTANCE_M = 150_000.0


def _squad_key(team: Team, squad_id: str) -> str:
    return f"{team.value}:{squad_id}"


def _valid_squad_leader_candidate(ship: ShipEntity | None, team: Team, squad_id: str) -> bool:
    return bool(ship is not None and ship.vital.alive and ship.team == team and ship.squad_id == squad_id)


def _replacement_leader(
    candidates: list[ShipEntity],
    previous_ship: ShipEntity | None,
    previous_location: SquadLeaderLocation | None,
) -> ShipEntity | None:
    if not candidates:
        return None
    previous_system = str(
        previous_location.system_id
        if previous_location is not None
        else getattr(getattr(previous_ship, "nav", None), "system_id", "") or ""
    )
    previous_group = str(getattr(previous_ship, "ship_group_id", "") or "")

    def priority(ship: ShipEntity) -> tuple[int, int, str]:
        same_system = bool(previous_system) and str(ship.nav.system_id or "") == previous_system
        same_group = bool(previous_group) and str(ship.ship_group_id or "") == previous_group
        if same_system and same_group:
            locality = 0
        elif same_system:
            locality = 1
        elif same_group:
            locality = 2
        else:
            locality = 3
        return locality, int(getattr(ship, "command_priority", 0) or 0), str(ship.ship_id)

    return min(candidates, key=priority)


def refresh_global_squad_leaders(world: WorldState) -> None:
    """Elect one deterministic leader per squad and refresh its read-only location snapshot."""
    members_by_key: dict[str, list[ShipEntity]] = {}
    for ship in world.ships.values():
        if ship.vital.alive:
            members_by_key.setdefault(_squad_key(ship.team, ship.squad_id), []).append(ship)

    active_keys = set(members_by_key)
    active_keys.update(str(key) for key in world.squad_leaders)
    for squad_key in sorted(active_keys):
        candidates = members_by_key.get(squad_key, [])
        mapped_id = str(world.squad_leaders.get(squad_key, "") or "")
        mapped_ship = world.ships.get(mapped_id)
        previous_location = world.squad_leader_locations.get(squad_key)
        if mapped_ship is not None and mapped_ship.vital.alive and any(ship.ship_id == mapped_ship.ship_id for ship in candidates):
            leader = mapped_ship
        else:
            leader = _replacement_leader(candidates, mapped_ship, previous_location)

        if leader is None:
            world.squad_leaders.pop(squad_key, None)
            world.squad_leader_locations.pop(squad_key, None)
            world.squad_leader_location_versions.pop(squad_key, None)
            world.squad_focus_queues.pop(squad_key, None)
            world.squad_focus_updated_at.pop(squad_key, None)
            world.squad_leader_speed_limits.pop(squad_key, None)
            continue

        leader_id = str(leader.ship_id)
        leader_system = str(leader.nav.system_id or "")
        old_id = previous_location.leader_id if previous_location is not None else mapped_id
        old_system = previous_location.system_id if previous_location is not None else leader_system
        version = int(world.squad_leader_location_versions.get(squad_key, 0) or 0)
        if previous_location is not None and (old_id != leader_id or old_system != leader_system):
            version += 1
        if previous_location is not None and old_system != leader_system:
            world.squad_focus_queues.pop(squad_key, None)
            world.squad_focus_updated_at.pop(squad_key, None)

        world.squad_leaders[squad_key] = leader_id
        world.squad_leader_location_versions[squad_key] = version
        world.squad_leader_locations[squad_key] = SquadLeaderLocation(
            leader_id=leader_id,
            system_id=leader_system,
            location_version=version,
        )


@dataclass(slots=True)
class BaseAgent:
    agent_id: str
    state: str = "IDLE"
    perception_buffer: list[str] = field(default_factory=list)
    orders_queue: list[Order] = field(default_factory=list)

    def sense(self, world: WorldState) -> None:
        del world

    def think(self, world: WorldState) -> None:
        del world

    def act(self, world: WorldState, delta_time: float) -> None:
        del world
        del delta_time


@dataclass(slots=True)
class ShipAgent(BaseAgent):
    ship_id: str = ""
    current_order: Order | None = None

    def _ship(self, world: WorldState) -> ShipEntity:
        return world.ships[self.ship_id]

    @staticmethod
    def _focus_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

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
        ship.order_queue = [order for order in ship.order_queue if order.kind not in {"ATTACK", "MOVE", "WARP", "USE_STARGATE"}]
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
        return world.squad_leader_locations.get(_squad_key(ship.team, ship.squad_id))

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
    def _apply_squad_follow_state(
        cls,
        world: WorldState,
        ship: ShipEntity,
        leader: ShipEntity | None,
    ) -> bool:
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

    def sense(self, world: WorldState) -> None:
        ship = self._ship(world)
        self.perception_buffer = ship.perception.copy()

    @staticmethod
    def _find_squad_leader(world: WorldState, ship: ShipEntity) -> ShipEntity | None:
        leader_id = str(world.squad_leaders.get(_squad_key(ship.team, ship.squad_id), "") or "")
        leader = world.ships.get(leader_id)
        return leader if _valid_squad_leader_candidate(leader, ship.team, ship.squad_id) else None

    def think(self, world: WorldState) -> None:
        ship = self._ship(world)
        if not ship.vital.alive:
            return

        leader = self._find_squad_leader(world, ship)
        is_leader = leader is not None and leader.ship_id == ship.ship_id
        if self._apply_squad_follow_state(world, ship, leader):
            self.current_order = None
            return

        if self.current_order and self.current_order.kind == "ATTACK":
            target_id = str(self.current_order.payload.get("target_id", "") or "")
            queue = world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), [])
            target = world.combat_entity(target_id)
            if not target_id or target_id not in queue or target is None or not target.vital.alive:
                self.current_order = None
                ship.combat.current_target = None
                ship.combat.last_attack_target = None

        if ship.order_queue:
            oldest = ship.order_queue[0]
            immediate = bool(oldest.payload.get("immediate", False))
            custom_delay = float(oldest.payload.get("delay_sec", 0.0) or 0.0)
            execute_delay = max(0.0, custom_delay)
            if immediate or world.now - oldest.issue_time >= execute_delay:
                self.current_order = oldest
                ship.order_queue.pop(0)

        if self.current_order and self.current_order.kind == "PROPULSION":
            ship.nav.propulsion_command_active = bool(self.current_order.payload.get("active", False))

        if self.current_order and self.current_order.kind == "WARP":
            payload = self.current_order.payload
            ship.nav.gate.target_structure_id = None
            ship.nav.warp.phase = "align"
            ship.nav.warp.target_position = Vector2(
                float(payload.get("x", ship.nav.position.x)),
                float(payload.get("y", ship.nav.position.y)),
            )
            ship.nav.warp.target_ship_id = str(payload.get("target_ship_id", "") or "") or None
            ship.nav.warp.target_beacon_id = str(payload.get("target_beacon_id", "") or "") or None
            ship.nav.warp.align_elapsed = 0.0
            ship.nav.warp.origin = None
            ship.nav.warp.destination = None
            ship.nav.warp.warp_distance_m = 0.0
            ship.nav.warp.warp_duration = 0.0
            ship.nav.warp.warp_elapsed = 0.0
            ship.nav.warp.capacitor_cost = 0.0
            ship.nav.warp.bubble_immune_snapshot = False
            ship.nav.warp.interdiction_snapshots_captured = False
            ship.nav.warp.interdiction_snapshots = tuple()
            self._clear_navigation_command(ship)
            self.current_order = None

        if self.current_order and self.current_order.kind == "USE_STARGATE":
            target_structure_id = str(self.current_order.payload.get("target_structure_id", "") or "").strip()
            ship.nav.gate.target_structure_id = target_structure_id or None
            self._clear_navigation_command(ship)
            self.current_order = None

        if str(getattr(ship.nav.warp, "phase", "idle") or "idle") != "idle":
            return

        if self.current_order and self.current_order.kind == "MOVE" and is_leader:
            payload = self.current_order.payload
            move_target = Vector2(float(payload["x"]), float(payload["y"]))
            mode = str(payload.get("mode", "move") or "move").strip().lower()
            ship.nav.command_mode = mode if mode in {"move", "approach", "keep_range", "orbit"} else "move"
            ship.nav.command_target_ship_id = str(payload.get("target_ship_id", "") or "").strip() or None
            ship.nav.command_target_structure_id = str(payload.get("target_structure_id", "") or "").strip() or None
            ship.nav.command_range_m = max(0.0, float(payload.get("range_m", 0.0) or 0.0))
            ship.nav.command_orbit_clockwise = bool(payload.get("orbit_clockwise", True))
            arrive_radius = max(120.0, ship.nav.radius * 1.5)
            if ship.nav.command_mode == "move" and ship.nav.position.distance_to(move_target) <= arrive_radius and ship.nav.velocity.length() <= 50.0:
                ship.nav.command_target = None
                ship.nav.command_target_ship_id = None
                ship.nav.command_target_structure_id = None
                ship.nav.command_range_m = 0.0
                self.current_order = None
            else:
                ship.nav.command_target = move_target

        if not is_leader and leader is not None:
            to_leader = leader.nav.position - ship.nav.position
            dist = to_leader.length()
            if dist > max(600.0, ship.nav.radius * 6.0):
                ship.nav.command_target = Vector2(leader.nav.position.x, leader.nav.position.y)
                ship.nav.command_mode = "move"
                ship.nav.command_target_ship_id = None
                ship.nav.command_target_structure_id = None
            else:
                ship.nav.command_target = None

        if self.current_order and self.current_order.kind == "ATTACK":
            ship.combat.current_target = self.current_order.payload.get("target_id")


@dataclass(slots=True)
class CommanderAgent(BaseAgent):
    team: Team = Team.BLUE
    squad_ids: list[str] = field(default_factory=list)

    @staticmethod
    def _prefocus_switch_probability(ship: ShipEntity) -> float:
        level = ship.quality.level.value
        if level == "ELITE":
            return 0.01
        if level == "REGULAR":
            return 0.05
        return 0.10

    @staticmethod
    def _focus_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    @staticmethod
    def _intent_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    @staticmethod
    def _alive_members(world: WorldState, squad_id: str, team: Team) -> list[ShipEntity]:
        squad_key = _squad_key(team, squad_id)
        location = world.squad_leader_locations.get(squad_key)
        leader_system = location.system_id if location is not None else None
        members = [
            s
            for s in world.ships.values()
            if s.squad_id == squad_id
            and s.team == team
            and s.vital.alive
            and (leader_system is None or str(s.nav.system_id or "") == leader_system)
        ]
        members.sort(key=lambda s: s.ship_id)
        return members

    @staticmethod
    def _sanitize_queue(world: WorldState, members: list[ShipEntity], queue: list[str]) -> list[str]:
        if not members:
            return []
        own_team = members[0].team
        system_id = str(members[0].nav.system_id or "")
        seen: set[str] = set()
        out: list[str] = []
        for target_id in queue:
            if target_id in seen:
                continue
            target = world.combat_entity(target_id)
            if (
                target is None
                or not target.vital.alive
                or str(getattr(getattr(target.nav, "warp", None), "phase", "idle") or "idle") != "idle"
                or target.team == own_team
                or str(getattr(target.nav, "system_id", "") or "") != system_id
            ):
                continue
            seen.add(target_id)
            out.append(target_id)
        return out

    @staticmethod
    def _issue_attack(members: list[ShipEntity], target_id: str, now: float) -> None:
        for ship in members:
            ship.order_queue = [o for o in ship.order_queue if o.kind != "ATTACK"]
            ship.order_queue.append(
                Order(
                    kind="ATTACK",
                    payload={"target_id": target_id, "immediate": True},
                    issue_time=now,
                )
            )

    @staticmethod
    def _issue_attack_per_ship(members: list[ShipEntity], target_by_ship: dict[str, str], now: float) -> None:
        for ship in members:
            target_id = target_by_ship.get(ship.ship_id)
            if not target_id:
                continue
            ship.order_queue = [o for o in ship.order_queue if o.kind != "ATTACK"]
            ship.order_queue.append(
                Order(
                    kind="ATTACK",
                    payload={"target_id": target_id, "immediate": True},
                    issue_time=now,
                )
            )

    @staticmethod
    def _clear_attack(members: list[ShipEntity]) -> None:
        for ship in members:
            ship.order_queue = [o for o in ship.order_queue if o.kind != "ATTACK"]
            ship.combat.current_target = None
            ship.combat.last_attack_target = None
            ship.combat.fire_delay_timers.clear()
            ship.combat.prelocked_targets.clear()
            ship.combat.prelock_timers.clear()

    @staticmethod
    def _sample_propulsion_delay(ship: ShipEntity) -> float:
        level = ship.quality.level.value
        r = random.random()
        if level == "ELITE":
            if r < 0.70:
                return random.uniform(0.5, 1.5)
            if r < 0.95:
                return random.uniform(1.5, 3.0)
            return random.uniform(3.0, 5.0)
        if level == "REGULAR":
            if r < 0.35:
                return random.uniform(0.5, 1.5)
            if r < 0.80:
                return random.uniform(1.5, 3.5)
            return random.uniform(3.5, 5.0)
        if r < 0.15:
            return random.uniform(0.5, 1.5)
        if r < 0.55:
            return random.uniform(1.5, 3.5)
        return random.uniform(3.5, 5.0)

    def think(self, world: WorldState) -> None:
        refresh_global_squad_leaders(world)
        for squad in list(self.squad_ids):
            intent_key = self._intent_key(self.team, squad)
            intent = world.intents.pop(intent_key, None)
            if intent is None:
                intent = world.intents.pop(squad, None)
            if intent is None:
                continue
            self._dispatch_intent(world, intent)
        for squad in list(self.squad_ids):
            self._update_squad_focus_state(world, squad)

    def _update_squad_focus_state(self, world: WorldState, squad_id: str) -> None:
        focus_key = self._focus_key(self.team, squad_id)
        members = self._alive_members(world, squad_id, self.team)
        if not members:
            world.squad_focus_queues.pop(focus_key, None)
            world.squad_focus_updated_at.pop(focus_key, None)
            return

        queue = self._sanitize_queue(world, members, list(world.squad_focus_queues.get(focus_key, [])))

        if not queue:
            self._clear_attack(members)
            world.squad_focus_queues[focus_key] = []
            return

        fallback_target = queue[0]
        next_target = queue[1] if len(queue) > 1 else None

        updates: dict[str, str] = {}
        for ship in members:
            ship_prelocked = ship.combat.prelocked_targets
            ship_prelock_timers = ship.combat.prelock_timers
            next_available = bool(next_target) and (
                (next_target in ship_prelocked) or (next_target in ship_prelock_timers)
            )
            current_target = ship.combat.current_target
            current_valid = False
            if current_target:
                current_entity = world.combat_entity(current_target)
                current_valid = (
                    current_entity is not None
                    and current_entity.vital.alive
                    and current_entity.team != ship.team
                    and current_target in queue
                )

            if current_valid:
                continue

            desired = fallback_target
            if next_available and next_target is not None:
                if random.random() < self._prefocus_switch_probability(ship):
                    desired = next_target
            updates[ship.ship_id] = desired

        if updates:
            self._issue_attack_per_ship(members, updates, world.now)
            for ship in members:
                assigned = updates.get(ship.ship_id)
                if not assigned:
                    continue
                ship_prelocked = ship.combat.prelocked_targets
                ship_prelock_timers = ship.combat.prelock_timers
                if assigned in ship_prelocked:
                    ship.combat.lock_targets.add(assigned)
                    ship.combat.lock_started_at.setdefault(assigned, float(world.now))
                    ship.combat.lock_timers.pop(assigned, None)
                    ship.combat.lock_deadlines.pop(assigned, None)
                else:
                    remaining = ship_prelock_timers.get(assigned)
                    if remaining is not None and remaining > 0:
                        remaining_float = float(remaining)
                        ship.combat.lock_targets.discard(assigned)
                        ship.combat.lock_started_at.setdefault(assigned, float(world.now))
                        ship.combat.lock_timers[assigned] = remaining_float
                        ship.combat.lock_deadlines[assigned] = float(world.now) + remaining_float

        world.squad_focus_queues[focus_key] = queue

    def _dispatch_intent(self, world: WorldState, intent: FleetIntent) -> None:
        members = self._alive_members(world, intent.squad_id, self.team)
        leader_id = str(world.squad_leaders.get(_squad_key(self.team, intent.squad_id), "") or "")
        leader = world.ships.get(leader_id)
        leader_id = leader.ship_id if leader is not None else None

        if intent.focus_target:
            focus_key = self._focus_key(self.team, intent.squad_id)
            queue = list(world.squad_focus_queues.get(focus_key, []))
            previous_focus = queue[0] if queue else None
            if previous_focus and previous_focus != intent.focus_target:
                # On focus switch, drop previous primary focus instead of demoting it to prefocus slot.
                queue = [
                    intent.focus_target,
                    *[
                        tid
                        for tid in queue
                        if tid != intent.focus_target and tid != previous_focus
                    ],
                ]
            else:
                queue = [intent.focus_target] + [tid for tid in queue if tid != intent.focus_target]
            world.squad_focus_queues[focus_key] = queue
            world.squad_focus_updated_at[focus_key] = float(world.now)
            self._issue_attack(members, intent.focus_target, world.now)

        for ship in world.ships.values():
            if ship.team != self.team or ship.squad_id != intent.squad_id or not ship.vital.alive:
                continue
            if intent.target_position is not None:
                ship.order_queue = [o for o in ship.order_queue if o.kind != "MOVE"]
                if ship.ship_id == leader_id:
                    ship.order_queue.append(
                        Order(
                            kind="MOVE",
                            payload={
                                "x": intent.target_position.x,
                                "y": intent.target_position.y,
                                "mode": str(intent.movement_mode or "move"),
                                "target_ship_id": str(intent.target_ship_id or ""),
                                "target_structure_id": str(intent.target_structure_id or ""),
                                "range_m": max(0.0, float(intent.target_range_m or 0.0)),
                                "orbit_clockwise": True,
                                "immediate": True,
                            },
                            issue_time=world.now,
                        )
                    )
            if intent.propulsion_active is not None:
                prop_key = self._focus_key(self.team, intent.squad_id)
                world.squad_propulsion_commands[prop_key] = bool(intent.propulsion_active)
                ship.order_queue = [o for o in ship.order_queue if o.kind != "PROPULSION"]
                ship.order_queue.append(
                    Order(
                        kind="PROPULSION",
                        payload={
                            "active": bool(intent.propulsion_active),
                            "delay_sec": self._sample_propulsion_delay(ship),
                            "immediate": False,
                        },
                        issue_time=world.now,
                    )
                )
