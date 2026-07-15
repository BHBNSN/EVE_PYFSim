from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b
import math
import random

from ..math2d import Vector2
from ..maps import MapDefinition, instantiate_structures
from ..models import Team
from ..world import WorldState
from .squad_commands import SquadNavigationService


class ScenarioService:
    """Scenario and roster state transitions independent of presentation concerns."""

    @staticmethod
    def assign_ships_to_squad(
        world: WorldState,
        team: Team,
        ship_ids: Iterable[str],
        squad_id: str,
    ) -> tuple[str, ...]:
        target_squad = str(squad_id or "").strip()
        normalized_ids = tuple(dict.fromkeys(str(ship_id or "").strip() for ship_id in ship_ids if str(ship_id or "").strip()))
        if not target_squad:
            raise ValueError("squad_id is required")
        if not normalized_ids:
            raise ValueError("at least one ship is required")
        ships = []
        for ship_id in normalized_ids:
            ship = world.ships.get(ship_id)
            if ship is None:
                raise ValueError(f"ship does not exist: {ship_id}")
            if ship.team != team:
                raise ValueError(f"ship is not controlled by {team.value}: {ship_id}")
            ships.append(ship)
        for ship in ships:
            ship.squad_id = target_squad
        return normalized_ids

    @staticmethod
    def set_ship_deployment(
        world: WorldState,
        ship_id: str,
        deployed: bool,
        *,
        system_id: str | None = None,
        position: Vector2 | None = None,
    ) -> None:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError("ship does not exist")
        if system_id is not None:
            ship.nav.system_id = str(system_id)
        if position is not None:
            ship.nav.position = Vector2(position.x, position.y)
            ship.nav.velocity = Vector2(0.0, 0.0)
        ship.vital.alive = bool(deployed)
        ship.deployed = bool(deployed)
        if not deployed:
            ship.order_queue.clear()

    @staticmethod
    def initialize_team_deployment(world: WorldState, team: Team) -> tuple[str, ...]:
        ships = sorted(
            (ship for ship in world.ships.values() if ship.team == team),
            key=lambda ship: ship.ship_id,
        )
        for ship in ships:
            ship.deployed = False
            ship.vital.alive = False
            ship.nav.velocity = Vector2(0.0, 0.0)
            ship.order_queue.clear()
        return tuple(ship.ship_id for ship in ships)

    @staticmethod
    def install_map_definition(world: WorldState, map_definition: MapDefinition) -> str:
        map_id = str(getattr(map_definition, "map_id", "") or "").strip()
        if not map_id:
            raise ValueError("map definition requires map_id")
        world.map_id = map_id
        world.map_name = str(getattr(map_definition, "name", "") or "")
        world.map_definition = map_definition
        world.structures = instantiate_structures(map_definition)
        return map_id

    @staticmethod
    def _spawn_position(world: WorldState, ship_id: str, center: Vector2, radius_m: float) -> Vector2:
        material = f"{world.tick}:{ship_id}:{center.x:.6f}:{center.y:.6f}".encode("utf-8")
        seed = int.from_bytes(blake2b(material, digest_size=8).digest(), "big", signed=False)
        rng = random.Random(seed)
        angle = rng.uniform(0.0, math.tau)
        distance = max(0.0, float(radius_m)) * math.sqrt(rng.random())
        return Vector2(
            center.x + math.cos(angle) * distance,
            center.y + math.sin(angle) * distance,
        )

    @classmethod
    def induce_ships(
        cls,
        world: WorldState,
        team: Team,
        ship_ids: Iterable[str],
        *,
        center: Vector2,
        system_id: str,
        radius_m: float = 5_000.0,
    ) -> tuple[str, ...]:
        normalized_ids = tuple(dict.fromkeys(str(ship_id or "").strip() for ship_id in ship_ids if str(ship_id or "").strip()))
        if not normalized_ids:
            raise ValueError("at least one ship is required")
        ships = []
        for ship_id in normalized_ids:
            ship = world.ships.get(ship_id)
            if ship is None:
                raise ValueError(f"ship does not exist: {ship_id}")
            if ship.team != team:
                raise ValueError(f"ship is not controlled by {team.value}: {ship_id}")
            ships.append(ship)

        affected_squads: set[str] = set()
        for ship in ships:
            affected_squads.add(ship.squad_id)
            ship.nav.position = cls._spawn_position(world, ship.ship_id, center, radius_m)
            ship.nav.system_id = str(system_id)
            ship.nav.velocity = Vector2(0.0, 0.0)
            ship.order_queue.clear()
            ship.deployed = True
            ship.vital.alive = True
            ship.vital.shield = ship.vital.shield_max
            ship.vital.armor = ship.vital.armor_max
            ship.vital.structure = ship.vital.structure_max
            ship.vital.cap = ship.vital.cap_max
            ship.combat.current_target = None
            ship.combat.last_attack_target = None
            ship.combat.lock_targets.clear()
            ship.combat.lock_started_at.clear()
            ship.combat.lock_timers.clear()
            ship.combat.lock_deadlines.clear()
            ship.combat.fire_delay_timers.clear()

        navigation = SquadNavigationService()
        for squad_id in sorted(affected_squads):
            navigation.clear_navigation(world, team, squad_id)
        return normalized_ids

    @classmethod
    def induce_undeployed_ships(
        cls,
        world: WorldState,
        team: Team,
        *,
        center: Vector2,
        system_id: str,
        squad_id: str | None = None,
        radius_m: float = 5_000.0,
    ) -> tuple[str, ...]:
        ship_ids = tuple(
            ship.ship_id
            for ship in sorted(world.ships.values(), key=lambda item: item.ship_id)
            if ship.team == team
            and not ship.deployed
            and (squad_id is None or ship.squad_id == squad_id)
        )
        if not ship_ids:
            raise ValueError("no undeployed ships match the command")
        return cls.induce_ships(
            world,
            team,
            ship_ids,
            center=center,
            system_id=system_id,
            radius_m=radius_m,
        )
