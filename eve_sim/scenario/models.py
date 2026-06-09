from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ScenarioShip:
    ship_id: str
    team: str
    squad_id: str
    fit: str
    count: int = 1
    role: str = ""
    position: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioFleet:
    team: str
    ships: tuple[ScenarioShip, ...]


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    duration_s: float
    seed: int
    fleets: tuple[ScenarioFleet, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
