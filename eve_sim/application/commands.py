from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from ..math2d import Vector2
from ..maps import MapDefinition
from ..models import Team
from .contracts import ShipSetupSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class MatchCommand:
    command_id: str = field(default_factory=lambda: uuid4().hex)
    issued_at_tick: int | None = None


@dataclass(frozen=True, slots=True)
class PauseMatch(MatchCommand):
    pass


@dataclass(frozen=True, slots=True)
class ResumeMatch(MatchCommand):
    pass


@dataclass(frozen=True, slots=True)
class AdvanceTicks(MatchCommand):
    count: int = 1


@dataclass(frozen=True, slots=True)
class IssueSquadMove(MatchCommand):
    team: Team
    squad_id: str
    target: Vector2


@dataclass(frozen=True, slots=True)
class IssueSquadApproach(MatchCommand):
    team: Team
    squad_id: str
    target_id: str
    range_m: float = 0.0


@dataclass(frozen=True, slots=True)
class IssueSquadNavigate(MatchCommand):
    team: Team
    squad_id: str
    target_kind: str
    target_id: str
    mode: str
    range_m: float = 0.0


@dataclass(frozen=True, slots=True)
class ClearSquadNavigation(MatchCommand):
    team: Team
    squad_id: str


@dataclass(frozen=True, slots=True)
class IssueSquadWarp(MatchCommand):
    team: Team
    squad_id: str
    target: Vector2
    target_ship_id: str | None = None
    target_beacon_id: str | None = None


@dataclass(frozen=True, slots=True)
class IssueSquadUseGate(MatchCommand):
    team: Team
    squad_id: str
    structure_id: str


@dataclass(frozen=True, slots=True)
class IssueSquadFocus(MatchCommand):
    team: Team
    squad_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class PrefocusSquadTarget(MatchCommand):
    team: Team
    squad_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class CancelSquadFocus(MatchCommand):
    team: Team
    squad_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class ClearSquadFocus(MatchCommand):
    team: Team
    squad_id: str


@dataclass(frozen=True, slots=True)
class SetSquadPropulsion(MatchCommand):
    team: Team
    squad_id: str
    active: bool


@dataclass(frozen=True, slots=True)
class SetSquadSpeedLimit(MatchCommand):
    team: Team
    squad_id: str
    limit: float


@dataclass(frozen=True, slots=True)
class LaunchSquadDrones(MatchCommand):
    team: Team
    squad_id: str
    type_name: str


@dataclass(frozen=True, slots=True)
class RecallSquadDrones(MatchCommand):
    team: Team
    squad_id: str


@dataclass(frozen=True, slots=True)
class LaunchSquadFighters(MatchCommand):
    team: Team
    squad_id: str
    type_name: str


@dataclass(frozen=True, slots=True)
class RecallSquadFighters(MatchCommand):
    team: Team
    squad_id: str


@dataclass(frozen=True, slots=True)
class SetSquadDroneTarget(MatchCommand):
    team: Team
    squad_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class SetSquadFighterTarget(MatchCommand):
    team: Team
    squad_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class ActivateSquadFighterAbility(MatchCommand):
    team: Team
    squad_id: str
    ability_id: str


@dataclass(frozen=True, slots=True)
class AssignShipsToSquad(MatchCommand):
    team: Team
    ship_ids: tuple[str, ...]
    squad_id: str


@dataclass(frozen=True, slots=True)
class InduceShips(MatchCommand):
    team: Team
    ship_ids: tuple[str, ...]
    center: Vector2
    system_id: str
    radius_m: float = 5_000.0


@dataclass(frozen=True, slots=True)
class InduceUndeployedShips(MatchCommand):
    team: Team
    center: Vector2
    system_id: str
    squad_id: str | None = None
    radius_m: float = 5_000.0


@dataclass(frozen=True, slots=True)
class InitializeTeamDeployment(MatchCommand):
    team: Team


@dataclass(frozen=True, slots=True)
class InstallMapDefinition(MatchCommand):
    map_definition: MapDefinition


@dataclass(frozen=True, slots=True)
class SetShipModuleManualMode(MatchCommand):
    team: Team
    ship_id: str
    module_id: str
    mode: str


@dataclass(frozen=True, slots=True)
class SetShipModuleTargetMode(MatchCommand):
    team: Team
    ship_id: str
    module_id: str
    mode: str


@dataclass(frozen=True, slots=True)
class SyncSquadModuleControls(MatchCommand):
    team: Team
    ship_id: str
    module_id: str
    manual_mode: str
    target_mode: str


@dataclass(frozen=True, slots=True)
class SetShipModuleChargeLock(MatchCommand):
    team: Team
    ship_id: str
    module_id: str
    charge_name: str


@dataclass(frozen=True, slots=True)
class ClearShipModuleChargeLock(MatchCommand):
    team: Team
    ship_id: str
    module_id: str


@dataclass(frozen=True, slots=True)
class SetFleetModuleCharge(MatchCommand):
    team: Team
    module_name: str
    charge_name: str


@dataclass(frozen=True, slots=True)
class SyncScenarioShips(MatchCommand):
    team: Team
    ships: tuple[ShipSetupSpec, ...]


@dataclass(frozen=True, slots=True)
class SetShipDeployment(MatchCommand):
    ship_id: str
    deployed: bool
    system_id: str | None = None
    position: Vector2 | None = None


@dataclass(frozen=True, slots=True)
class ReplaceScenario(MatchCommand):
    scenario_id: str
