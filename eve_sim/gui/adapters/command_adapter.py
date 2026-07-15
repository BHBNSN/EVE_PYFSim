from __future__ import annotations

from collections.abc import Callable

from ...application.command_bus import CommandResult
from ...application.commands import (
    ActivateSquadFighterAbility,
    AssignShipsToSquad,
    CancelSquadFocus,
    ClearSquadFocus,
    ClearShipModuleChargeLock,
    IssueSquadApproach,
    IssueSquadFocus,
    IssueSquadMove,
    IssueSquadNavigate,
    IssueSquadUseGate,
    IssueSquadWarp,
    InduceUndeployedShips,
    InitializeTeamDeployment,
    InstallMapDefinition,
    LaunchSquadDrones,
    LaunchSquadFighters,
    MatchCommand,
    PrefocusSquadTarget,
    RecallSquadDrones,
    SetSquadDroneTarget,
    SetSquadFighterTarget,
    SetSquadPropulsion,
    SetSquadSpeedLimit,
    SetShipModuleManualMode,
    SetShipModuleChargeLock,
    SetFleetModuleCharge,
    SetShipModuleTargetMode,
    SyncSquadModuleControls,
    SyncScenarioShips,
)
from ...application.contracts import ShipSetupSpec
from ...application.match_application import MatchApplication
from ...math2d import Vector2
from ...maps import MapDefinition
from ...models import Team


class GuiCommandAdapter:
    """Qt-free translation of GUI gestures into typed application commands."""

    def __init__(
        self,
        application: MatchApplication,
        executor: Callable[[MatchCommand], CommandResult] | None = None,
    ) -> None:
        self._application = application
        self._executor = executor or application.execute

    def _execute(self, command: MatchCommand) -> CommandResult:
        return self._executor(command)

    def move(self, team: Team, squad_id: str, target: Vector2) -> CommandResult:
        return self._execute(IssueSquadMove(team=team, squad_id=squad_id, target=target))

    def approach(self, team: Team, squad_id: str, target_id: str, range_m: float = 0.0) -> CommandResult:
        return self._execute(IssueSquadApproach(team=team, squad_id=squad_id, target_id=target_id, range_m=range_m))

    def navigate(self, team: Team, squad_id: str, target_kind: str, target_id: str, mode: str, range_m: float = 0.0) -> CommandResult:
        return self._execute(
            IssueSquadNavigate(team=team, squad_id=squad_id, target_kind=target_kind, target_id=target_id, mode=mode, range_m=range_m)
        )

    def warp(self, team: Team, squad_id: str, target: Vector2, *, target_ship_id: str | None = None, target_beacon_id: str | None = None) -> CommandResult:
        return self._execute(
            IssueSquadWarp(team=team, squad_id=squad_id, target=target, target_ship_id=target_ship_id, target_beacon_id=target_beacon_id)
        )

    def use_gate(self, team: Team, squad_id: str, structure_id: str) -> CommandResult:
        return self._execute(IssueSquadUseGate(team=team, squad_id=squad_id, structure_id=structure_id))

    def focus(self, team: Team, squad_id: str, target_id: str) -> CommandResult:
        return self._execute(IssueSquadFocus(team=team, squad_id=squad_id, target_id=target_id))

    def prefocus(self, team: Team, squad_id: str, target_id: str) -> CommandResult:
        return self._execute(PrefocusSquadTarget(team=team, squad_id=squad_id, target_id=target_id))

    def cancel_prefocus(self, team: Team, squad_id: str, target_id: str) -> CommandResult:
        return self._execute(CancelSquadFocus(team=team, squad_id=squad_id, target_id=target_id))

    def clear_focus(self, team: Team, squad_id: str) -> CommandResult:
        return self._execute(ClearSquadFocus(team=team, squad_id=squad_id))

    def propulsion(self, team: Team, squad_id: str, active: bool) -> CommandResult:
        return self._execute(SetSquadPropulsion(team=team, squad_id=squad_id, active=active))

    def speed_limit(self, team: Team, squad_id: str, limit: float) -> CommandResult:
        return self._execute(SetSquadSpeedLimit(team=team, squad_id=squad_id, limit=limit))

    def launch_drones(self, team: Team, squad_id: str, type_name: str) -> CommandResult:
        return self._execute(LaunchSquadDrones(team=team, squad_id=squad_id, type_name=type_name))

    def launch_fighters(self, team: Team, squad_id: str, type_name: str) -> CommandResult:
        return self._execute(LaunchSquadFighters(team=team, squad_id=squad_id, type_name=type_name))

    def recall_deployables(self, team: Team, squad_id: str) -> CommandResult:
        return self._execute(RecallSquadDrones(team=team, squad_id=squad_id))

    def drone_target(self, team: Team, squad_id: str, target_id: str) -> CommandResult:
        return self._execute(SetSquadDroneTarget(team=team, squad_id=squad_id, target_id=target_id))

    def fighter_target(self, team: Team, squad_id: str, target_id: str) -> CommandResult:
        return self._execute(SetSquadFighterTarget(team=team, squad_id=squad_id, target_id=target_id))

    def fighter_ability(self, team: Team, squad_id: str, ability_id: str) -> CommandResult:
        return self._execute(ActivateSquadFighterAbility(team=team, squad_id=squad_id, ability_id=ability_id))

    def assign_ships(self, team: Team, ship_ids: tuple[str, ...], squad_id: str) -> CommandResult:
        return self._execute(AssignShipsToSquad(team=team, ship_ids=ship_ids, squad_id=squad_id))

    def induce_undeployed_ships(
        self,
        team: Team,
        center: Vector2,
        system_id: str,
        squad_id: str | None = None,
        radius_m: float = 5_000.0,
    ) -> CommandResult:
        return self._execute(
            InduceUndeployedShips(
                team=team,
                center=center,
                system_id=system_id,
                squad_id=squad_id,
                radius_m=radius_m,
            )
        )

    def initialize_local_team_deployment(self, team: Team) -> CommandResult:
        """Prepare the client-owned scenario copy; this is never a LAN command."""
        return self._application.execute(InitializeTeamDeployment(team=team))

    def install_local_map_definition(self, map_definition: MapDefinition) -> CommandResult:
        """Install authoritative/replicated map data in the local session."""
        return self._application.execute(InstallMapDefinition(map_definition=map_definition))

    def set_ship_module_manual_mode(self, team: Team, ship_id: str, module_id: str, mode: str) -> CommandResult:
        return self._execute(
            SetShipModuleManualMode(team=team, ship_id=ship_id, module_id=module_id, mode=mode)
        )

    def set_ship_module_target_mode(self, team: Team, ship_id: str, module_id: str, mode: str) -> CommandResult:
        return self._execute(
            SetShipModuleTargetMode(team=team, ship_id=ship_id, module_id=module_id, mode=mode)
        )

    def sync_squad_module_controls(
        self,
        team: Team,
        ship_id: str,
        module_id: str,
        manual_mode: str,
        target_mode: str,
    ) -> CommandResult:
        return self._execute(
            SyncSquadModuleControls(
                team=team,
                ship_id=ship_id,
                module_id=module_id,
                manual_mode=manual_mode,
                target_mode=target_mode,
            )
        )

    def set_ship_module_charge_lock(
        self,
        team: Team,
        ship_id: str,
        module_id: str,
        charge_name: str,
    ) -> CommandResult:
        return self._execute(
            SetShipModuleChargeLock(
                team=team,
                ship_id=ship_id,
                module_id=module_id,
                charge_name=charge_name,
            )
        )

    def clear_ship_module_charge_lock(self, team: Team, ship_id: str, module_id: str) -> CommandResult:
        return self._execute(
            ClearShipModuleChargeLock(team=team, ship_id=ship_id, module_id=module_id)
        )

    def set_fleet_module_charge(
        self,
        team: Team,
        module_name: str,
        charge_name: str,
    ) -> CommandResult:
        return self._execute(
            SetFleetModuleCharge(
                team=team,
                module_name=module_name,
                charge_name=charge_name,
            )
        )

    def sync_scenario_ships(self, team: Team, ships: tuple[ShipSetupSpec, ...]) -> CommandResult:
        return self._execute(SyncScenarioShips(team=team, ships=ships))
