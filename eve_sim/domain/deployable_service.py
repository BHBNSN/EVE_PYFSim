from __future__ import annotations

from typing import Protocol

from ..math2d import Vector2
from ..models import Team
from ..world import WorldState


class DeployableCommandPort(Protocol):
    """Simulation-side operations required by deployable application commands."""

    def command_fighter_squad_move(self, world: WorldState, team: Team, squad_id: str, target: Vector2) -> int: ...
    def command_fighter_squad_navigation(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        *,
        target_kind: str,
        target_id: str,
        movement_mode: str,
        range_m: float = 0.0,
    ) -> int: ...
    def clear_fighter_squad_navigation(self, world: WorldState, team: Team, squad_id: str) -> int: ...
    def set_squad_fighter_target(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> bool: ...
    def clear_fighter_squad_target(self, world: WorldState, team: Team, squad_id: str) -> int: ...
    def launch_squad_drones(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int: ...
    def launch_squad_fighters(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int: ...
    def recall_squad_deployables(self, world: WorldState, team: Team, squad_id: str) -> int: ...
    def set_squad_drone_target(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> bool: ...
    def activate_fighter_ability(self, world: WorldState, team: Team, squad_id: str, ability_id: str) -> int: ...


class DeployableCommandService:
    """Domain-facing deployable command boundary backed by a simulation port."""

    def __init__(self, port: DeployableCommandPort) -> None:
        self._port = port

    @staticmethod
    def has_fighters(world: WorldState, team: Team, squad_id: str) -> bool:
        return any(
            fighter.team == team and fighter.squad_id == squad_id and fighter.vital.alive
            for fighter in world.fighters.values()
        )

    def move_fighters(self, world: WorldState, team: Team, squad_id: str, target: Vector2) -> int:
        return self._port.command_fighter_squad_move(world, team, squad_id, target)

    def navigate_fighters(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        *,
        target_kind: str,
        target_id: str,
        movement_mode: str,
        range_m: float = 0.0,
    ) -> int:
        return self._port.command_fighter_squad_navigation(
            world,
            team,
            squad_id,
            target_kind=target_kind,
            target_id=target_id,
            movement_mode=movement_mode,
            range_m=range_m,
        )

    def clear_fighter_navigation(self, world: WorldState, team: Team, squad_id: str) -> int:
        return self._port.clear_fighter_squad_navigation(world, team, squad_id)

    def set_fighter_target(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> bool:
        return self._port.set_squad_fighter_target(world, team, squad_id, target_id)

    def clear_fighter_target(self, world: WorldState, team: Team, squad_id: str) -> int:
        return self._port.clear_fighter_squad_target(world, team, squad_id)

    def launch_drones(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int:
        return self._port.launch_squad_drones(world, team, squad_id, type_name)

    def launch_fighters(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int:
        return self._port.launch_squad_fighters(world, team, squad_id, type_name)

    def recall(self, world: WorldState, team: Team, squad_id: str) -> int:
        return self._port.recall_squad_deployables(world, team, squad_id)

    def set_drone_target(self, world: WorldState, team: Team, squad_id: str, target_id: str) -> bool:
        return self._port.set_squad_drone_target(world, team, squad_id, target_id)

    def activate_fighter_ability(self, world: WorldState, team: Team, squad_id: str, ability_id: str) -> int:
        return self._port.activate_fighter_ability(world, team, squad_id, ability_id)
