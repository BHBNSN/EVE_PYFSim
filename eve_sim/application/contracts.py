from __future__ import annotations

from dataclasses import dataclass

from ..math2d import Vector2


@dataclass(frozen=True, slots=True)
class ShipSetupSpec:
    """Immutable scenario-transfer value shared by commands and queries."""

    ship_id: str
    squad_id: str
    ship_group_id: str
    fit_text: str
    position: Vector2
    velocity: Vector2
    facing_deg: float
    system_id: str
    deployed: bool
    alive: bool
    shield: float
    armor: float
    structure: float
    cap: float
    quality_level: str = "REGULAR"
    quality_reaction_delay: float = 0.0
    quality_ignore_order_probability: float = 0.0
    quality_formation_jitter: float = 0.0
