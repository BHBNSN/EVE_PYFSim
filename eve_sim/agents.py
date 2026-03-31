from __future__ import annotations

from dataclasses import dataclass, field
import random

from .math2d import Vector2
from .models import FleetIntent, Order, ShipEntity, Team
from .world import WorldState


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

    def sense(self, world: WorldState) -> None:
        ship = self._ship(world)
        self.perception_buffer = ship.perception.copy()

    @staticmethod
    def _find_squad_leader(world: WorldState, ship: ShipEntity) -> ShipEntity | None:
        squad_key = f"{ship.team.value}:{ship.squad_id}"
        mapped_id = world.squad_leaders.get(squad_key)
        if mapped_id:
            mapped_ship = world.ships.get(mapped_id)
            if (
                mapped_ship is not None
                and mapped_ship.vital.alive
                and str(getattr(getattr(mapped_ship.nav, "warp", None), "phase", "idle") or "idle") == "idle"
                and mapped_ship.team == ship.team
                and mapped_ship.squad_id == ship.squad_id
            ):
                return mapped_ship

        members = [
            s
            for s in world.ships.values()
            if s.team == ship.team
            and s.squad_id == ship.squad_id
            and s.vital.alive
            and str(getattr(getattr(s.nav, "warp", None), "phase", "idle") or "idle") == "idle"
        ]
        if not members:
            return None
        leader = random.choice(members)
        world.squad_leaders[squad_key] = leader.ship_id
        return leader

    def think(self, world: WorldState) -> None:
        ship = self._ship(world)
        if not ship.vital.alive:
            return

        if self.current_order and self.current_order.kind == "ATTACK":
            target_id = str(self.current_order.payload.get("target_id", "") or "")
            queue = world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), [])
            target = world.ships.get(target_id)
            if not target_id or target_id not in queue or target is None or not target.vital.alive:
                self.current_order = None
                ship.combat.current_target = None
                ship.combat.last_attack_target = None

        leader = self._find_squad_leader(world, ship)
        is_leader = leader is not None and leader.ship_id == ship.ship_id

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
            ship.nav.command_target = None
            self.current_order = None

        if str(getattr(ship.nav.warp, "phase", "idle") or "idle") != "idle":
            return

        if self.current_order and self.current_order.kind == "MOVE" and is_leader:
            move_target = Vector2(self.current_order.payload["x"], self.current_order.payload["y"])
            arrive_radius = max(120.0, ship.nav.radius * 1.5)
            if ship.nav.position.distance_to(move_target) <= arrive_radius and ship.nav.velocity.length() <= 50.0:
                ship.nav.command_target = None
                self.current_order = None
            else:
                ship.nav.command_target = move_target

        if not is_leader and leader is not None:
            to_leader = leader.nav.position - ship.nav.position
            dist = to_leader.length()
            if dist > max(600.0, ship.nav.radius * 6.0):
                ship.nav.command_target = leader.nav.position
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
        members = [
            s
            for s in world.ships.values()
            if s.squad_id == squad_id
            and s.team == team
            and s.vital.alive
            and str(getattr(getattr(s.nav, "warp", None), "phase", "idle") or "idle") == "idle"
        ]
        members.sort(key=lambda s: s.ship_id)
        return members

    @staticmethod
    def _current_target_of(members: list[ShipEntity], world: WorldState) -> str | None:
        for ship in members:
            target_id = ship.combat.current_target
            if not target_id:
                continue
            target = world.ships.get(target_id)
            if target is not None and target.vital.alive and target.team != ship.team:
                return target_id
        return None

    @staticmethod
    def _sanitize_queue(world: WorldState, members: list[ShipEntity], queue: list[str]) -> list[str]:
        if not members:
            return []
        own_team = members[0].team
        seen: set[str] = set()
        out: list[str] = []
        for target_id in queue:
            if target_id in seen:
                continue
            target = world.ships.get(target_id)
            if (
                target is None
                or not target.vital.alive
                or str(getattr(getattr(target.nav, "warp", None), "phase", "idle") or "idle") != "idle"
                or target.team == own_team
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
            world.squad_prelocked_targets.pop(focus_key, None)
            world.squad_prelock_timers.pop(focus_key, None)
            return

        queue = self._sanitize_queue(world, members, list(world.squad_focus_queues.get(focus_key, [])))

        if not queue:
            self._clear_attack(members)
            world.squad_focus_queues[focus_key] = []
            return

        prelocked_by_ship = world.squad_prelocked_targets.get(focus_key, {})
        prelock_timers_by_ship = world.squad_prelock_timers.get(focus_key, {})
        fallback_target = queue[0]
        next_target = queue[1] if len(queue) > 1 else None

        updates: dict[str, str] = {}
        for ship in members:
            ship_prelocked = prelocked_by_ship.get(ship.ship_id, set())
            ship_prelock_timers = prelock_timers_by_ship.get(ship.ship_id, {})
            next_available = bool(next_target) and (
                (next_target in ship_prelocked) or (next_target in ship_prelock_timers)
            )
            current_target = ship.combat.current_target
            current_valid = False
            if current_target:
                current_entity = world.ships.get(current_target)
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
                ship_prelocked = prelocked_by_ship.get(ship.ship_id, set())
                ship_prelock_timers = prelock_timers_by_ship.get(ship.ship_id, {})
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
        leader_id = random.choice(members).ship_id if members else None
        if members:
            squad_key = f"{members[0].team.value}:{intent.squad_id}"
            mapped = world.squad_leaders.get(squad_key)
            if mapped:
                mapped_ship = world.ships.get(mapped)
                if mapped_ship is not None and mapped_ship.vital.alive and mapped_ship.squad_id == intent.squad_id:
                    leader_id = mapped_ship.ship_id
            if leader_id is not None:
                world.squad_leaders[squad_key] = leader_id

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
                            payload={"x": intent.target_position.x, "y": intent.target_position.y, "immediate": True},
                            issue_time=world.now,
                        )
                    )
            if intent.propulsion_active is not None:
                prop_key = self._focus_key(self.team, intent.squad_id)
                world.squad_propulsion_commands[prop_key] = bool(intent.propulsion_active)
                world.squad_propulsion_commands.pop(intent.squad_id, None)
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
