from __future__ import annotations

from typing import cast

from ..domain.events import ApplicationEvent
from ..domain.scenario_service import ScenarioService
from ..domain.squad_commands import SquadNavigationService, SquadTargetService
from ..math2d import Vector2
from .commands import (
    ActivateSquadFighterAbility,
    AssignShipsToSquad,
    CancelSquadFocus,
    ClearSquadNavigation,
    ClearSquadFocus,
    ClearShipModuleChargeLock,
    IssueSquadApproach,
    IssueSquadFocus,
    IssueSquadMove,
    IssueSquadNavigate,
    IssueSquadUseGate,
    IssueSquadWarp,
    InduceShips,
    InduceUndeployedShips,
    InitializeTeamDeployment,
    InstallMapDefinition,
    LaunchSquadDrones,
    LaunchSquadFighters,
    MatchCommand,
    PrefocusSquadTarget,
    RecallSquadDrones,
    RecallSquadFighters,
    ReplaceScenario,
    SetShipDeployment,
    SetShipModuleChargeLock,
    SetFleetModuleCharge,
    SetShipModuleManualMode,
    SetShipModuleTargetMode,
    SetSquadDroneTarget,
    SetSquadFighterTarget,
    SetSquadPropulsion,
    SetSquadSpeedLimit,
    SyncSquadModuleControls,
    SyncScenarioShips,
)
from .errors import UnsupportedScenarioError
from .session import MatchSession


class DefaultCommandHandlers:
    def __init__(self) -> None:
        self.navigation = SquadNavigationService()
        self.scenarios = ScenarioService()
        self.targets = SquadTargetService()

    @staticmethod
    def _event(kind: str, command: MatchCommand, **payload: object) -> tuple[ApplicationEvent, ...]:
        return (ApplicationEvent(kind, {"command_id": command.command_id, **payload}),)

    def move(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadMove, raw)
        if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
            session.deployable_commands.move_fighters(session.world, command.team, command.squad_id, command.target)
        else:
            self.navigation.issue_move(session.world, command.team, command.squad_id, command.target)
        return self._event("squad_navigation_changed", command, team=command.team.value, squad_id=command.squad_id, mode="move")

    def approach(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadApproach, raw)
        target = session.world.combat_entity(command.target_id)
        if target is None or not target.vital.alive:
            raise ValueError("approach target is invalid")
        target_position = Vector2(target.nav.position.x, target.nav.position.y)
        if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
            session.deployable_commands.navigate_fighters(
                session.world, command.team, command.squad_id, target_kind="ship", target_id=command.target_id,
                movement_mode="approach", range_m=command.range_m,
            )
        else:
            self.navigation.issue_move(
                session.world, command.team, command.squad_id, target_position,
                mode="approach", target_ship_id=command.target_id, range_m=command.range_m,
            )
        return self._event("squad_navigation_changed", command, team=command.team.value, squad_id=command.squad_id, mode="approach")

    def navigate(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadNavigate, raw)
        kind = command.target_kind.strip().lower()
        if kind == "ship":
            target = session.world.combat_entity(command.target_id)
            if target is None or not target.vital.alive:
                raise ValueError("navigation target is invalid")
            target_position = Vector2(target.nav.position.x, target.nav.position.y)
            if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
                session.deployable_commands.navigate_fighters(
                    session.world, command.team, command.squad_id, target_kind="ship", target_id=command.target_id,
                    movement_mode=command.mode, range_m=command.range_m,
                )
            else:
                self.navigation.issue_move(
                    session.world, command.team, command.squad_id, target_position, mode=command.mode,
                    target_ship_id=command.target_id, range_m=command.range_m,
                )
        elif kind == "structure":
            structure = session.world.structures.get(command.target_id)
            if structure is None:
                raise ValueError("navigation structure is invalid")
            target_position = Vector2(structure.position.x, structure.position.y)
            if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
                session.deployable_commands.navigate_fighters(
                    session.world, command.team, command.squad_id, target_kind="structure", target_id=command.target_id,
                    movement_mode=command.mode, range_m=command.range_m,
                )
            else:
                self.navigation.issue_move(
                    session.world, command.team, command.squad_id, target_position, mode=command.mode,
                    target_structure_id=command.target_id, range_m=command.range_m,
                )
        else:
            raise ValueError("navigation target kind is invalid")
        return self._event("squad_navigation_changed", command, team=command.team.value, squad_id=command.squad_id, mode=command.mode)

    def warp(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadWarp, raw)
        self.navigation.clear_navigation(session.world, command.team, command.squad_id)
        self.navigation.set_propulsion(session.world, command.team, command.squad_id, False)
        self.navigation.issue_warp(
            session.world, command.team, command.squad_id, command.target,
            target_ship_id=command.target_ship_id, target_beacon_id=command.target_beacon_id,
        )
        return self._event("squad_warp_issued", command, team=command.team.value, squad_id=command.squad_id)

    def clear_navigation(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(ClearSquadNavigation, raw)
        has_ships = any(
            ship.team == command.team and ship.squad_id == command.squad_id and ship.vital.alive
            for ship in session.world.ships.values()
        )
        has_fighters = session.deployable_commands.has_fighters(session.world, command.team, command.squad_id)
        if not has_ships and not has_fighters:
            raise ValueError("squad has no alive members")
        self.navigation.clear_navigation(
            session.world,
            command.team,
            command.squad_id,
            require_ship_members=has_ships,
        )
        if has_fighters:
            session.deployable_commands.clear_fighter_navigation(session.world, command.team, command.squad_id)
        return self._event("squad_navigation_cleared", command, team=command.team.value, squad_id=command.squad_id)

    def use_gate(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadUseGate, raw)
        structure = session.world.structures.get(command.structure_id)
        if structure is None or str(getattr(structure, "kind", "") or "").upper() != "STARGATE":
            raise ValueError("stargate does not exist")
        self.navigation.clear_navigation(session.world, command.team, command.squad_id)
        self.navigation.set_propulsion(session.world, command.team, command.squad_id, False)
        self.navigation.use_gate(session.world, command.team, command.squad_id, command.structure_id)
        return self._event("squad_gate_issued", command, team=command.team.value, squad_id=command.squad_id)

    def focus(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(IssueSquadFocus, raw)
        self.targets.issue_focus(session.world, command.team, command.squad_id, command.target_id)
        if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
            session.deployable_commands.set_fighter_target(session.world, command.team, command.squad_id, command.target_id)
        return self._event("squad_focus_changed", command, team=command.team.value, squad_id=command.squad_id, target_id=command.target_id)

    def prefocus(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(PrefocusSquadTarget, raw)
        self.targets.prefocus(session.world, command.team, command.squad_id, command.target_id)
        return self._event("squad_prefocus_changed", command, team=command.team.value, squad_id=command.squad_id, target_id=command.target_id)

    def cancel_focus(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(CancelSquadFocus, raw)
        self.targets.cancel_prefocus(session.world, command.team, command.squad_id, command.target_id)
        return self._event("squad_prefocus_cancelled", command, team=command.team.value, squad_id=command.squad_id, target_id=command.target_id)

    def clear_focus(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(ClearSquadFocus, raw)
        self.targets.clear_focus(session.world, command.team, command.squad_id)
        if session.deployable_commands.has_fighters(session.world, command.team, command.squad_id):
            session.deployable_commands.clear_fighter_target(session.world, command.team, command.squad_id)
        return self._event("squad_focus_cleared", command, team=command.team.value, squad_id=command.squad_id)

    def propulsion(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetSquadPropulsion, raw)
        self.navigation.set_propulsion(session.world, command.team, command.squad_id, command.active)
        return self._event("squad_propulsion_changed", command, team=command.team.value, squad_id=command.squad_id, active=command.active)

    def speed_limit(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetSquadSpeedLimit, raw)
        self.targets.set_speed_limit(session.world, command.team, command.squad_id, command.limit)
        return self._event("squad_speed_limit_changed", command, team=command.team.value, squad_id=command.squad_id, limit=command.limit)

    def launch_drones(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(LaunchSquadDrones, raw)
        session.deployable_commands.launch_drones(session.world, command.team, command.squad_id, command.type_name)
        return self._event("squad_drones_launched", command, team=command.team.value, squad_id=command.squad_id, type_name=command.type_name)

    def launch_fighters(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(LaunchSquadFighters, raw)
        session.deployable_commands.launch_fighters(session.world, command.team, command.squad_id, command.type_name)
        return self._event("squad_fighters_launched", command, team=command.team.value, squad_id=command.squad_id, type_name=command.type_name)

    def recall_deployables(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(RecallSquadDrones | RecallSquadFighters, raw)
        session.deployable_commands.recall(session.world, command.team, command.squad_id)
        return self._event("squad_deployables_recalled", command, team=command.team.value, squad_id=command.squad_id)

    def drone_target(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetSquadDroneTarget, raw)
        session.deployable_commands.set_drone_target(session.world, command.team, command.squad_id, command.target_id)
        return self._event("squad_drone_target_changed", command, team=command.team.value, squad_id=command.squad_id, target_id=command.target_id)

    def fighter_target(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetSquadFighterTarget, raw)
        session.deployable_commands.set_fighter_target(session.world, command.team, command.squad_id, command.target_id)
        return self._event("squad_fighter_target_changed", command, team=command.team.value, squad_id=command.squad_id, target_id=command.target_id)

    def fighter_ability(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(ActivateSquadFighterAbility, raw)
        session.deployable_commands.activate_fighter_ability(session.world, command.team, command.squad_id, command.ability_id)
        return self._event("squad_fighter_ability_activated", command, team=command.team.value, squad_id=command.squad_id, ability_id=command.ability_id)

    def deployment(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetShipDeployment, raw)
        self.scenarios.set_ship_deployment(
            session.world,
            command.ship_id,
            command.deployed,
            system_id=command.system_id,
            position=command.position,
        )
        return self._event("ship_deployment_changed", command, ship_id=command.ship_id, deployed=command.deployed)

    def assign_squad(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(AssignShipsToSquad, raw)
        ship_ids = self.scenarios.assign_ships_to_squad(
            session.world,
            command.team,
            command.ship_ids,
            command.squad_id,
        )
        return self._event(
            "ships_assigned_to_squad",
            command,
            team=command.team.value,
            squad_id=command.squad_id,
            ship_ids=ship_ids,
        )

    def induce_ships(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(InduceShips, raw)
        ship_ids = self.scenarios.induce_ships(
            session.world,
            command.team,
            command.ship_ids,
            center=command.center,
            system_id=command.system_id,
            radius_m=command.radius_m,
        )
        return self._event(
            "ships_induced",
            command,
            team=command.team.value,
            ship_ids=ship_ids,
            system_id=command.system_id,
        )

    def induce_undeployed_ships(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(InduceUndeployedShips, raw)
        ship_ids = self.scenarios.induce_undeployed_ships(
            session.world,
            command.team,
            center=command.center,
            system_id=command.system_id,
            squad_id=command.squad_id,
            radius_m=command.radius_m,
        )
        return self._event(
            "ships_induced",
            command,
            team=command.team.value,
            ship_ids=ship_ids,
            squad_id=command.squad_id,
            system_id=command.system_id,
        )

    def initialize_team_deployment(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(InitializeTeamDeployment, raw)
        ship_ids = self.scenarios.initialize_team_deployment(session.world, command.team)
        return self._event(
            "team_deployment_initialized",
            command,
            team=command.team.value,
            ship_ids=ship_ids,
        )

    def install_map_definition(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(InstallMapDefinition, raw)
        map_id = self.scenarios.install_map_definition(session.world, command.map_definition)
        return self._event("map_definition_installed", command, map_id=map_id)

    def set_ship_module_manual_mode(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetShipModuleManualMode, raw)
        mode = session.module_commands.set_manual_mode(
            session.world, command.team, command.ship_id, command.module_id, command.mode
        )
        return self._event(
            "ship_module_manual_mode_changed",
            command,
            team=command.team.value,
            ship_id=command.ship_id,
            module_id=command.module_id,
            mode=mode,
        )

    def set_ship_module_target_mode(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetShipModuleTargetMode, raw)
        mode = session.module_commands.set_target_mode(
            session.world, command.team, command.ship_id, command.module_id, command.mode
        )
        return self._event(
            "ship_module_target_mode_changed",
            command,
            team=command.team.value,
            ship_id=command.ship_id,
            module_id=command.module_id,
            mode=mode,
        )

    def sync_squad_module_controls(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SyncSquadModuleControls, raw)
        ship_ids = session.module_commands.sync_squad_controls(
            session.world,
            command.team,
            command.ship_id,
            command.module_id,
            command.manual_mode,
            command.target_mode,
        )
        return self._event(
            "squad_module_controls_changed",
            command,
            team=command.team.value,
            source_ship_id=command.ship_id,
            module_id=command.module_id,
            ship_ids=ship_ids,
        )

    def set_ship_module_charge_lock(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetShipModuleChargeLock, raw)
        module_name, reload_time = session.ship_fit_commands.set_charge_lock(
            session.world,
            command.team,
            command.ship_id,
            command.module_id,
            command.charge_name,
        )
        return self._event(
            "ship_module_charge_lock_changed",
            command,
            team=command.team.value,
            ship_id=command.ship_id,
            module_id=command.module_id,
            charge_name=command.charge_name,
            module_name=module_name,
            reload_time=reload_time,
        )

    def clear_ship_module_charge_lock(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(ClearShipModuleChargeLock, raw)
        session.ship_fit_commands.clear_charge_lock(
            session.world, command.team, command.ship_id, command.module_id
        )
        return self._event(
            "ship_module_charge_lock_cleared",
            command,
            team=command.team.value,
            ship_id=command.ship_id,
            module_id=command.module_id,
        )

    def set_fleet_module_charge(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SetFleetModuleCharge, raw)
        ship_ids, reload_time = session.ship_fit_commands.set_fleet_charge(
            session.world,
            command.team,
            command.module_name,
            command.charge_name,
        )
        return self._event(
            "fleet_module_charge_changed",
            command,
            team=command.team.value,
            ship_ids=ship_ids,
            module_name=command.module_name,
            charge_name=command.charge_name,
            reload_time=reload_time,
        )

    def sync_scenario_ships(self, session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(SyncScenarioShips, raw)
        if not command.ships:
            raise ValueError("at least one ship setup is required")
        rows = {
            item.ship_id: {
                "team": command.team.value,
                "squad_id": item.squad_id,
                "ship_group_id": item.ship_group_id,
                "fit_text": item.fit_text,
                "position": {"x": item.position.x, "y": item.position.y},
                "velocity": {"x": item.velocity.x, "y": item.velocity.y},
                "facing_deg": item.facing_deg,
                "system_id": item.system_id,
                "deployed": item.deployed,
                "alive": item.alive,
                "shield": item.shield,
                "armor": item.armor,
                "structure": item.structure,
                "cap": item.cap,
                "quality_level": item.quality_level,
                "quality_reaction_delay": item.quality_reaction_delay,
                "quality_ignore_order_probability": item.quality_ignore_order_probability,
                "quality_formation_jitter": item.quality_formation_jitter,
            }
            for item in command.ships
        }
        result = session.scenario_snapshot_loader.apply_delta(session.world, {"ships": rows})
        for ship_id in result.added_ship_ids:
            session.engine.register_ship(ship_id)
        return self._event(
            "scenario_ships_synchronized",
            command,
            team=command.team.value,
            ship_ids=tuple(rows),
        )

    def replace_scenario(self, _session: MatchSession, raw: MatchCommand) -> tuple[ApplicationEvent, ...]:
        command = cast(ReplaceScenario, raw)
        raise UnsupportedScenarioError(f"scenario replacement is not registered: {command.scenario_id}")
