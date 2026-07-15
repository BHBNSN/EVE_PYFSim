from __future__ import annotations

from dataclasses import replace

from ..math2d import Vector2
from ..models import FleetIntent, Order, Team
from ..squad_identity import squad_key
from ..world import WorldState


class SquadCommandError(ValueError):
    pass


class SquadNavigationService:
    @staticmethod
    def _members(world: WorldState, team: Team, squad_id: str):
        members = [
            ship
            for ship in world.ships.values()
            if ship.team == team and ship.squad_id == squad_id and ship.vital.alive
        ]
        if not members:
            raise SquadCommandError("squad has no alive ships")
        return members

    @staticmethod
    def _intent(world: WorldState, team: Team, squad_id: str) -> FleetIntent:
        existing = world.intents.get(squad_key(team, squad_id))
        return existing if existing is not None else FleetIntent(squad_id=squad_id)

    def issue_move(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        target: Vector2,
        *,
        mode: str = "move",
        target_ship_id: str | None = None,
        target_structure_id: str | None = None,
        range_m: float = 0.0,
    ) -> None:
        self._members(world, team, squad_id)
        intent = self._intent(world, team, squad_id)
        world.intents[squad_key(team, squad_id)] = replace(
            intent,
            target_position=Vector2(float(target.x), float(target.y)),
            movement_mode=str(mode or "move"),
            target_ship_id=target_ship_id,
            target_structure_id=target_structure_id,
            target_range_m=max(0.0, float(range_m)),
        )

    def set_propulsion(self, world: WorldState, team: Team, squad_id: str, active: bool) -> None:
        self._members(world, team, squad_id)
        key = squad_key(team, squad_id)
        world.squad_propulsion_commands[key] = bool(active)
        world.intents[key] = replace(self._intent(world, team, squad_id), propulsion_active=bool(active))

    def clear_navigation(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        *,
        require_ship_members: bool = True,
    ) -> None:
        if require_ship_members:
            self._members(world, team, squad_id)
        intent = self._intent(world, team, squad_id)
        world.intents[squad_key(team, squad_id)] = replace(
            intent,
            target_position=None,
            movement_mode="move",
            target_ship_id=None,
            target_structure_id=None,
            target_range_m=0.0,
        )

    def issue_warp(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        target: Vector2,
        *,
        target_ship_id: str | None = None,
        target_beacon_id: str | None = None,
    ) -> None:
        members = self._members(world, team, squad_id)
        for ship in members:
            ship.order_queue = [order for order in ship.order_queue if order.kind not in {"WARP", "MOVE", "ATTACK"}]
            if ship.nav.position.distance_to(target) < 150_000.0:
                continue
            ship.order_queue.append(
                Order(
                    kind="WARP",
                    payload={
                        "x": float(target.x),
                        "y": float(target.y),
                        "target_ship_id": target_ship_id or "",
                        "target_beacon_id": target_beacon_id or "",
                        "immediate": True,
                    },
                    issue_time=float(world.now),
                )
            )

    def use_gate(self, world: WorldState, team: Team, squad_id: str, structure_id: str) -> None:
        if not structure_id or structure_id not in world.structures:
            raise SquadCommandError("stargate does not exist")
        for ship in self._members(world, team, squad_id):
            ship.order_queue = [order for order in ship.order_queue if order.kind != "USE_STARGATE"]
            ship.order_queue.append(
                Order(
                    kind="USE_STARGATE",
                    payload={"target_structure_id": structure_id, "immediate": True},
                    issue_time=float(world.now),
                )
            )


class SquadTargetService:
    @staticmethod
    def _members(world: WorldState, team: Team, squad_id: str):
        return [ship for ship in world.ships.values() if ship.team == team and ship.squad_id == squad_id and ship.vital.alive]

    def issue_focus(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> None:
        target = world.combat_entity(target_id)
        if target is None or not target.vital.alive or target.team == team:
            raise SquadCommandError("focus target is invalid")
        members = self._members(world, team, squad_id)
        if not members:
            raise SquadCommandError("squad has no alive ships")
        key = squad_key(team, squad_id)
        queue = list(world.squad_focus_queues.get(key, []))
        previous = queue[0] if queue else None
        queue = [target_id] + [item for item in queue if item != target_id and item != previous]
        world.squad_focus_queues[key] = queue
        world.squad_focus_updated_at[key] = float(world.now)

    def prefocus(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> None:
        target = world.combat_entity(target_id)
        if target is None or not target.vital.alive or target.team == team:
            raise SquadCommandError("prefocus target is invalid")
        key = squad_key(team, squad_id)
        queue = list(world.squad_focus_queues.get(key, []))
        if target_id not in queue:
            queue.append(target_id)
        world.squad_focus_queues[key] = queue
        if queue:
            world.squad_focus_updated_at.setdefault(key, float(world.now))

    def cancel_prefocus(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> None:
        key = squad_key(team, squad_id)
        queue = list(world.squad_focus_queues.get(key, []))
        if queue:
            head, *tail = queue
            world.squad_focus_queues[key] = tail if head == target_id else [head, *[item for item in tail if item != target_id]]
            world.squad_focus_updated_at[key] = float(world.now)
        for ship in self._members(world, team, squad_id):
            ship.combat.prelocked_targets.discard(target_id)
            ship.combat.prelock_timers.pop(target_id, None)
            ship.combat.lock_targets.discard(target_id)
            ship.combat.lock_started_at.pop(target_id, None)
            ship.combat.lock_timers.pop(target_id, None)
            ship.combat.lock_deadlines.pop(target_id, None)
            ship.combat.fire_delay_timers.pop(target_id, None)

    def clear_focus(self, world: WorldState, team: Team, squad_id: str) -> None:
        key = squad_key(team, squad_id)
        world.squad_focus_queues.pop(key, None)
        world.squad_focus_updated_at.pop(key, None)
        for ship in self._members(world, team, squad_id):
            ship.order_queue = [order for order in ship.order_queue if order.kind != "ATTACK"]
            ship.combat.current_target = None
            ship.combat.last_attack_target = None
            ship.combat.lock_targets.clear()
            ship.combat.lock_started_at.clear()
            ship.combat.lock_timers.clear()
            ship.combat.lock_deadlines.clear()
            ship.combat.fire_delay_timers.clear()
            ship.combat.prelocked_targets.clear()
            ship.combat.prelock_timers.clear()

    def set_speed_limit(self, world: WorldState, team: Team, squad_id: str, limit: float) -> None:
        key = squad_key(team, squad_id)
        if limit <= 0.0:
            world.squad_leader_speed_limits.pop(key, None)
        else:
            world.squad_leader_speed_limits[key] = float(limit)
