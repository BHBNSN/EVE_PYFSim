from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import BubbleField, DroneEntity, FighterEntity, FleetIntent, ProjectileBlast, ProjectileEntity, ShipEntity, StructureEntity

if TYPE_CHECKING:
    from .maps import MapDefinition


@dataclass(slots=True)
class WorldState:
    now: float = 0.0
    tick: int = 0
    map_id: str = ""
    map_name: str = ""
    map_definition: "MapDefinition | None" = None
    ships: dict[str, ShipEntity] = field(default_factory=dict)
    structures: dict[str, StructureEntity] = field(default_factory=dict)
    intents: dict[str, FleetIntent] = field(default_factory=dict)
    squad_leaders: dict[str, str] = field(default_factory=dict)
    squad_propulsion_commands: dict[str, bool] = field(default_factory=dict)
    squad_leader_speed_limits: dict[str, float] = field(default_factory=dict)
    squad_focus_queues: dict[str, list[str]] = field(default_factory=dict)
    squad_focus_updated_at: dict[str, float] = field(default_factory=dict)
    squad_prelocked_targets: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    squad_prelock_timers: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    drones: dict[str, DroneEntity] = field(default_factory=dict)
    fighters: dict[str, FighterEntity] = field(default_factory=dict)
    projectiles: dict[str, ProjectileEntity] = field(default_factory=dict)
    projectile_blasts: dict[str, ProjectileBlast] = field(default_factory=dict)
    bubble_fields: dict[str, BubbleField] = field(default_factory=dict)

    @property
    def beacons(self) -> dict[str, StructureEntity]:
        return self.structures

    def combat_entity(self, entity_id: str | None):
        key = str(entity_id or "").strip()
        if not key:
            return None
        return self.ships.get(key) or self.drones.get(key) or self.fighters.get(key)

    def iter_combat_entities(self):
        yield from self.ships.values()
        yield from self.drones.values()
        yield from self.fighters.values()
