from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .models import ScenarioDefinition, ScenarioFleet, ScenarioShip
from .validators import validate_scenario


LIBRARY_DIR = Path(__file__).with_name("library")


def _ship_from_dict(team: str, data: dict[str, Any]) -> ScenarioShip:
    raw_position = data.get("position")
    position = {
        str(key): float(value)
        for key, value in (raw_position.items() if isinstance(raw_position, dict) else ())
    }
    return ScenarioShip(
        ship_id=str(data.get("ship_id", "") or ""),
        team=team,
        squad_id=str(data.get("squad_id", "") or ""),
        fit=str(data.get("fit", "") or ""),
        count=int(data.get("count", 1) or 1),
        role=str(data.get("role", "") or ""),
        position=position,
    )


def scenario_from_dict(data: dict[str, Any]) -> ScenarioDefinition:
    errors = validate_scenario(data)
    if errors:
        raise ValueError("; ".join(errors))
    fleets: list[ScenarioFleet] = []
    for fleet in data.get("fleets", []):
        team = str(fleet.get("team", "") or "")
        ships = tuple(_ship_from_dict(team, ship) for ship in fleet.get("ships", []))
        fleets.append(ScenarioFleet(team=team, ships=ships))
    return ScenarioDefinition(
        scenario_id=str(data.get("scenario_id", "") or ""),
        name=str(data.get("name", "") or ""),
        duration_s=float(data.get("duration_s", 0.0) or 0.0),
        seed=int(data.get("seed", 0) or 0),
        fleets=tuple(fleets),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def load_scenario(path: str | Path) -> ScenarioDefinition:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario root must be an object")
    return scenario_from_dict(data)


def load_scenario_library(path: str | Path = LIBRARY_DIR) -> dict[str, ScenarioDefinition]:
    root = Path(path)
    scenarios: dict[str, ScenarioDefinition] = {}
    for item in sorted(root.glob("*.yaml")):
        scenario = load_scenario(item)
        scenarios[scenario.scenario_id] = scenario
    return scenarios


__all__ = [
    "LIBRARY_DIR",
    "load_scenario",
    "load_scenario_library",
    "scenario_from_dict",
]
