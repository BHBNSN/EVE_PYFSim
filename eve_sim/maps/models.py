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


def is_stargate_building(building: MapBuildingDefinition) -> bool:
    return str(getattr(building, "kind", "") or "").upper() == "STARGATE"


def normalize_stargate_links(map_definition: MapDefinition) -> MapDefinition:
    building_system_by_id: dict[str, str] = {}
    gates_by_id: dict[str, MapBuildingDefinition] = {}
    ordered_gate_ids: list[str] = []
    for system in map_definition.systems:
        system_id = str(system.system_id or "")
        for building in system.buildings:
            building.system_id = system_id
            building_id = str(building.building_id or "")
            if not building_id:
                continue
            building_system_by_id[building_id] = system_id
            if is_stargate_building(building):
                gates_by_id[building_id] = building
                ordered_gate_ids.append(building_id)
            else:
                building.linked_building_id = None

    requested_links: list[tuple[str, str]] = []
    for source_id in ordered_gate_ids:
        source = gates_by_id[source_id]
        target_id = str(source.linked_building_id or "").strip()
        if not target_id:
            source.linked_building_id = None
            continue
        if target_id == source_id or target_id not in gates_by_id:
            source.linked_building_id = None
            continue
        if building_system_by_id.get(target_id, "") == building_system_by_id.get(source_id, ""):
            source.linked_building_id = None
            continue
        requested_links.append((source_id, target_id))

    for gate in gates_by_id.values():
        gate.linked_building_id = None

    for source_id, target_id in requested_links:
        source = gates_by_id.get(source_id)
        target = gates_by_id.get(target_id)
        if source is None or target is None:
            continue
        if building_system_by_id.get(source_id, "") == building_system_by_id.get(target_id, ""):
            continue
        for gate in gates_by_id.values():
            if gate.linked_building_id in {source_id, target_id}:
                gate.linked_building_id = None
        source.linked_building_id = target_id
        target.linked_building_id = source_id
    return map_definition


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
    "is_stargate_building",
    "normalize_stargate_links",
]
