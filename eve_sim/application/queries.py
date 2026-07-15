from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MatchStatusView:
    match_id: str
    status: str
    tick: int
    now: float


@dataclass(frozen=True, slots=True)
class SquadView:
    team: str
    squad_id: str
    leader_id: str | None
    member_ids: tuple[str, ...]
    focus_queue: tuple[str, ...]
    propulsion_active: bool
    speed_limit: float | None


@dataclass(frozen=True, slots=True)
class ShipView:
    ship_id: str
    team: str
    squad_id: str
    alive: bool
    system_id: str
    position: tuple[float, float]
    current_target: str | None
    deployed: bool
    fit_text: str
    locked_module_charges: Mapping[str, str]
    module_manual_modes: Mapping[str, str]
    module_target_modes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class OverviewQuery:
    team: str | None = None
    system_id: str | None = None
    alive_only: bool = False


@dataclass(frozen=True, slots=True)
class OverviewView:
    ships: tuple[ShipView, ...]


@dataclass(frozen=True, slots=True)
class SimulationDiagnosticsView:
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModuleTargetRulesView:
    choices: tuple[str, ...]
    default_mode: str
