from __future__ import annotations

import json
from pathlib import Path

from ..math2d import Vector2
from ..models import StructureEntity
from .models import (
    MapBuildingDefinition,
    MapCatalogEntry,
    MapDefinition,
    MapSpawnAnchorDefinition,
    MapSystemDefinition,
)


DEFAULT_MAP_ID = "dual_system_crossroads"


def map_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "res" / "map"


def _vector_from_payload(raw: object) -> Vector2:
    if isinstance(raw, dict):
        try:
            return Vector2(float(raw.get("x", 0.0)), float(raw.get("y", 0.0)))
        except Exception:
            return Vector2(0.0, 0.0)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return Vector2(float(raw[0]), float(raw[1]))
        except Exception:
            return Vector2(0.0, 0.0)
    return Vector2(0.0, 0.0)


def _vector_payload(value: Vector2) -> dict[str, float]:
    return {"x": float(value.x), "y": float(value.y)}


def deserialize_map_definition(payload: dict[str, object]) -> MapDefinition:
    raw_systems = payload.get("systems")
    systems: list[MapSystemDefinition] = []
    for raw_system in raw_systems if isinstance(raw_systems, list) else []:
        if not isinstance(raw_system, dict):
            continue
        system_id = str(raw_system.get("system_id", "") or "").strip()
        if not system_id:
            continue
        raw_buildings = raw_system.get("buildings")
        buildings: list[MapBuildingDefinition] = []
        for raw_building in raw_buildings if isinstance(raw_buildings, list) else []:
            if not isinstance(raw_building, dict):
                continue
            building_id = str(raw_building.get("building_id", "") or "").strip()
            if not building_id:
                continue
            buildings.append(
                MapBuildingDefinition(
                    building_id=building_id,
                    system_id=system_id,
                    position=_vector_from_payload(raw_building.get("position")),
                    radius_m=max(0.0, float(raw_building.get("radius_m", 0.0) or 0.0)),
                    kind=str(raw_building.get("kind", "STRUCTURE") or "STRUCTURE"),
                    name=str(raw_building.get("name", "") or ""),
                    interaction_range_m=max(
                        0.0,
                        float(raw_building.get("interaction_range_m", raw_building.get("interaction_range", 0.0)) or 0.0),
                    ),
                    icon_key=str(raw_building.get("icon_key", "") or ""),
                    linked_building_id=str(raw_building.get("linked_building_id", "") or "") or None,
                )
            )
        raw_anchors = raw_system.get("spawn_anchors")
        spawn_anchors: list[MapSpawnAnchorDefinition] = []
        for raw_anchor in raw_anchors if isinstance(raw_anchors, list) else []:
            if not isinstance(raw_anchor, dict):
                continue
            anchor_id = str(raw_anchor.get("anchor_id", "") or "").strip()
            if not anchor_id:
                continue
            spawn_anchors.append(
                MapSpawnAnchorDefinition(
                    anchor_id=anchor_id,
                    system_id=system_id,
                    position=_vector_from_payload(raw_anchor.get("position")),
                    radius_m=max(0.0, float(raw_anchor.get("radius_m", 0.0) or 0.0)),
                    team=str(raw_anchor.get("team", "ALL") or "ALL").upper(),
                    squad_id=str(raw_anchor.get("squad_id", "") or ""),
                    label=str(raw_anchor.get("label", "") or ""),
                )
            )
        systems.append(
            MapSystemDefinition(
                system_id=system_id,
                name=str(raw_system.get("name", system_id) or system_id),
                origin=_vector_from_payload(raw_system.get("origin")),
                radius_m=max(1_000.0, float(raw_system.get("radius_m", 1_000.0) or 1_000.0)),
                buildings=buildings,
                spawn_anchors=spawn_anchors,
            )
        )
    map_id = str(payload.get("map_id", DEFAULT_MAP_ID) or DEFAULT_MAP_ID).strip() or DEFAULT_MAP_ID
    return MapDefinition(
        map_id=map_id,
        name=str(payload.get("name", map_id) or map_id),
        description=str(payload.get("description", "") or ""),
        version=max(1, int(float(payload.get("version", 1) or 1))),
        systems=systems,
    )


def serialize_map_definition(map_definition: MapDefinition) -> dict[str, object]:
    return {
        "map_id": map_definition.map_id,
        "name": map_definition.name,
        "description": map_definition.description,
        "version": int(map_definition.version),
        "systems": [
            {
                "system_id": system.system_id,
                "name": system.name,
                "radius_m": float(system.radius_m),
                "buildings": [
                    {
                        "building_id": building.building_id,
                        "name": building.name,
                        "kind": building.kind,
                        "position": _vector_payload(building.position),
                        "radius_m": float(building.radius_m),
                        "interaction_range_m": float(building.interaction_range_m),
                        "icon_key": building.icon_key,
                        "linked_building_id": building.linked_building_id,
                    }
                    for building in system.buildings
                ],
                "spawn_anchors": [
                    {
                        "anchor_id": anchor.anchor_id,
                        "team": anchor.team,
                        "squad_id": anchor.squad_id,
                        "label": anchor.label,
                        "position": _vector_payload(anchor.position),
                        "radius_m": float(anchor.radius_m),
                    }
                    for anchor in system.spawn_anchors
                ],
            }
            for system in map_definition.systems
        ],
    }


def list_map_catalog() -> list[MapCatalogEntry]:
    entries: list[MapCatalogEntry] = []
    directory = map_directory()
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            loaded = deserialize_map_definition(payload)
        except Exception:
            continue
        entries.append(
            MapCatalogEntry(
                map_id=loaded.map_id,
                name=loaded.name,
                description=loaded.description,
                path=str(path),
            )
        )
    return entries


def load_map_definition(map_id: str | None = None) -> MapDefinition:
    target_map_id = str(map_id or DEFAULT_MAP_ID).strip() or DEFAULT_MAP_ID
    directory = map_directory()
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("map_id", "") or "") != target_map_id:
            continue
        return deserialize_map_definition(payload)
    fallback = directory / f"{DEFAULT_MAP_ID}.json"
    if fallback.exists():
        payload = json.loads(fallback.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return deserialize_map_definition(payload)
    raise FileNotFoundError(f"Map not found: {target_map_id}")


def save_map_definition(map_definition: MapDefinition, path: str | Path | None = None) -> Path:
    directory = map_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = Path(path) if path is not None else directory / f"{map_definition.map_id}.json"
    target.write_text(
        json.dumps(serialize_map_definition(map_definition), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def instantiate_structures(map_definition: MapDefinition) -> dict[str, StructureEntity]:
    structures: dict[str, StructureEntity] = {}
    for system in map_definition.systems:
        for building in system.buildings:
            structures[building.building_id] = StructureEntity(
                structure_id=building.building_id,
                position=Vector2(float(building.position.x), float(building.position.y)),
                radius=max(0.0, float(building.radius_m or 0.0)),
                interaction_range=max(0.0, float(building.interaction_range_m or 0.0)),
                kind=str(building.kind or "STRUCTURE"),
                system_id=system.system_id,
                display_name=str(building.name or building.building_id),
                icon_key=str(building.icon_key or ""),
                linked_structure_id=str(building.linked_building_id or "") or None,
            )
    return structures


__all__ = [
    "DEFAULT_MAP_ID",
    "deserialize_map_definition",
    "instantiate_structures",
    "list_map_catalog",
    "load_map_definition",
    "map_directory",
    "save_map_definition",
    "serialize_map_definition",
]
