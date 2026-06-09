from __future__ import annotations

from dataclasses import dataclass, field

from ..math2d import Vector2


@dataclass(slots=True)
class MapSpawnAnchorDefinition:
    anchor_id: str
    system_id: str
    position: Vector2
    radius_m: float
    team: str = "ALL"
    squad_id: str = ""
    label: str = ""


@dataclass(slots=True)
class MapBuildingDefinition:
    building_id: str
    system_id: str
    position: Vector2
    radius_m: float
    kind: str
    name: str = ""
    interaction_range_m: float = 0.0
    icon_key: str = ""
    linked_building_id: str | None = None


@dataclass(slots=True)
class MapSystemDefinition:
    system_id: str
    name: str
    radius_m: float
    # Kept only for backward-compatible JSON loading; simulation uses per-system local coordinates.
    origin: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    buildings: list[MapBuildingDefinition] = field(default_factory=list)
    spawn_anchors: list[MapSpawnAnchorDefinition] = field(default_factory=list)


@dataclass(slots=True)
class MapDefinition:
    map_id: str
    name: str
    description: str = ""
    version: int = 1
    systems: list[MapSystemDefinition] = field(default_factory=list)

    def system_by_id(self, system_id: str) -> MapSystemDefinition | None:
        target = str(system_id or "")
        for system in self.systems:
            if system.system_id == target:
                return system
        return None

    def all_buildings(self) -> list[MapBuildingDefinition]:
        buildings: list[MapBuildingDefinition] = []
        for system in self.systems:
            buildings.extend(system.buildings)
        return buildings

    def building_by_id(self, building_id: str) -> MapBuildingDefinition | None:
        target = str(building_id or "")
        for building in self.all_buildings():
            if building.building_id == target:
                return building
        return None

    def all_spawn_anchors(self) -> list[MapSpawnAnchorDefinition]:
        anchors: list[MapSpawnAnchorDefinition] = []
        for system in self.systems:
            anchors.extend(system.spawn_anchors)
        return anchors

    def extent_radius_m(self) -> float:
        radius = 0.0
        for system in self.systems:
            radius = max(radius, max(0.0, float(system.radius_m or 0.0)))
        return max(radius, 1_000.0)


@dataclass(frozen=True, slots=True)
class MapCatalogEntry:
    map_id: str
    name: str
    path: str
    description: str = ""

__all__ = [
    "MapBuildingDefinition",
    "MapCatalogEntry",
    "MapDefinition",
    "MapSpawnAnchorDefinition",
    "MapSystemDefinition",
]
