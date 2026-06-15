from __future__ import annotations

from dataclasses import dataclass

from ..math2d import Vector2
from ..models import QualityLevel, QualityState, Team


QUALITY_PRESETS = {
    QualityLevel.ELITE: QualityState(
        QualityLevel.ELITE,
        reaction_delay=0.0,
        ignore_order_probability=0.0,
        formation_jitter=0.0,
    ),
    QualityLevel.REGULAR: QualityState(
        QualityLevel.REGULAR,
        reaction_delay=0.0,
        ignore_order_probability=0.0,
        formation_jitter=0.0,
    ),
    QualityLevel.IRREGULAR: QualityState(
        QualityLevel.IRREGULAR,
        reaction_delay=0.0,
        ignore_order_probability=0.0,
        formation_jitter=0.0,
    ),
}


@dataclass(slots=True)
class ParsedCargoSpec:
    item_name: str
    quantity: int = 1


@dataclass(slots=True)
class ParsedMutationSpec:
    base_item_name: str
    mutaplasmid_name: str
    attributes: dict[str, float]


@dataclass(slots=True)
class ParsedEftFit:
    ship_name: str
    fit_name: str
    module_names: list[str]
    module_specs: list["ParsedModuleSpec"]
    cargo_item_names: list[str]
    fit_key: str
    cargo_specs: list[ParsedCargoSpec] | None = None
    implant_names: list[str] | None = None
    booster_names: list[str] | None = None
    mutation_specs: dict[int, ParsedMutationSpec] | None = None


@dataclass(slots=True)
class ParsedModuleSpec:
    module_name: str
    charge_name: str | None = None
    offline: bool = False
    mutation_ref: int | None = None


@dataclass(slots=True)
class ManualShipSetup:
    team: Team
    squad_id: str
    quality: QualityLevel
    position: Vector2
    fit_text: str
    is_leader: bool = False
    ship_group_id: str = ""


__all__ = [
    "ManualShipSetup",
    "ParsedCargoSpec",
    "ParsedEftFit",
    "ParsedModuleSpec",
    "ParsedMutationSpec",
    "QUALITY_PRESETS",
]
