from __future__ import annotations

from collections.abc import Mapping

from .application.command_bus import CommandResult
from .application.commands import (
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
    LaunchSquadDrones,
    LaunchSquadFighters,
    MatchCommand,
    PrefocusSquadTarget,
    RecallSquadDrones,
    SetShipModuleManualMode,
    SetShipModuleChargeLock,
    SetFleetModuleCharge,
    SetShipModuleTargetMode,
    SetSquadDroneTarget,
    SetSquadFighterTarget,
    SetSquadPropulsion,
    SetSquadSpeedLimit,
    SyncSquadModuleControls,
    SyncScenarioShips,
)
from .application.contracts import ShipSetupSpec
from .application.errors import CommandValidationError
from .lan_commands import (
    CMD_INDUCE_FLEET_AT,
    CMD_INDUCE_SQUAD_AT,
    CMD_SET_FLEET_MODULE_CHARGE,
    CMD_ASSIGN_SHIPS_TO_SQUAD,
    CMD_CLEAR_MODULE_CHARGE_LOCK,
    CMD_SET_MODULE_CHARGE_LOCK,
    CMD_SET_MODULE_MANUAL_MODE,
    CMD_SET_MODULE_TARGET_MODE,
    CMD_SYNC_MODULE_CONTROLS,
    CMD_SYNC_SETUP,
    CMD_SQUAD_APPROACH,
    CMD_SQUAD_ATTACK,
    CMD_SQUAD_CANCEL_PREFOCUS,
    CMD_SQUAD_CLEAR_FOCUS,
    CMD_SQUAD_DRONE_ATTACK,
    CMD_SQUAD_FIGHTER_ABILITY,
    CMD_SQUAD_FIGHTER_ATTACK,
    CMD_SQUAD_LEADER_SPEED_LIMIT,
    CMD_SQUAD_LAUNCH_DRONES,
    CMD_SQUAD_LAUNCH_FIGHTERS,
    CMD_SQUAD_MOVE,
    CMD_SQUAD_NAVIGATE,
    CMD_SQUAD_PREFOCUS,
    CMD_SQUAD_PROPULSION,
    CMD_SQUAD_RECALL_DEPLOYABLES,
    CMD_SQUAD_USE_GATE,
    CMD_SQUAD_WARP,
)
from .math2d import Vector2
from .models import Team


class LanCommandAdapter:
    """Translate transport dictionaries into typed application commands."""

    def decode(self, payload: Mapping[str, object], *, team: Team) -> MatchCommand:
        kind = str(payload.get("kind", "") or "").upper()
        squad_id = str(payload.get("squad_id", "") or "").strip()
        command_id = str(payload.get("command_id", "") or "").strip()
        if not command_id:
            raise CommandValidationError("command_id is required")
        common = {"command_id": command_id}
        if kind in {CMD_SET_MODULE_MANUAL_MODE, CMD_SET_MODULE_TARGET_MODE, CMD_SYNC_MODULE_CONTROLS}:
            ship_id = str(payload.get("ship_id", "") or "").strip()
            module_id = str(payload.get("module_id", "") or "").strip()
            if not ship_id or not module_id:
                raise CommandValidationError("ship_id and module_id are required")
            if kind == CMD_SET_MODULE_MANUAL_MODE:
                return SetShipModuleManualMode(
                    team=team,
                    ship_id=ship_id,
                    module_id=module_id,
                    mode=str(payload.get("mode", "") or "auto"),
                    **common,
                )
            if kind == CMD_SET_MODULE_TARGET_MODE:
                return SetShipModuleTargetMode(
                    team=team,
                    ship_id=ship_id,
                    module_id=module_id,
                    mode=str(payload.get("mode", "") or "auto"),
                    **common,
                )
            return SyncSquadModuleControls(
                team=team,
                ship_id=ship_id,
                module_id=module_id,
                manual_mode=str(payload.get("mode", "") or "auto"),
                target_mode=str(payload.get("target_mode", "") or "auto"),
                **common,
            )
        if kind in {CMD_INDUCE_FLEET_AT, CMD_INDUCE_SQUAD_AT}:
            system_id = str(payload.get("system_id", "") or "").strip()
            if not system_id:
                raise CommandValidationError("system_id is required")
            if kind == CMD_INDUCE_SQUAD_AT and not squad_id:
                raise CommandValidationError("squad_id is required")
            return InduceUndeployedShips(
                team=team,
                center=Vector2(float(payload.get("x", 0.0)), float(payload.get("y", 0.0))),
                system_id=system_id,
                squad_id=squad_id if kind == CMD_INDUCE_SQUAD_AT else None,
                radius_m=float(payload.get("radius_m", 5_000.0) or 5_000.0),
                **common,
            )
        if kind in {CMD_SET_MODULE_CHARGE_LOCK, CMD_CLEAR_MODULE_CHARGE_LOCK}:
            ship_id = str(payload.get("ship_id", "") or "").strip()
            module_id = str(payload.get("module_id", "") or "").strip()
            if not ship_id or not module_id:
                raise CommandValidationError("ship_id and module_id are required")
            if kind == CMD_CLEAR_MODULE_CHARGE_LOCK:
                return ClearShipModuleChargeLock(team=team, ship_id=ship_id, module_id=module_id, **common)
            return SetShipModuleChargeLock(
                team=team,
                ship_id=ship_id,
                module_id=module_id,
                charge_name=str(payload.get("charge_name", "") or ""),
                **common,
            )
        if kind == CMD_SET_FLEET_MODULE_CHARGE:
            module_name = str(payload.get("module_name", "") or "").strip()
            if not module_name:
                raise CommandValidationError("module_name is required")
            return SetFleetModuleCharge(
                team=team,
                module_name=module_name,
                charge_name=str(payload.get("charge_name", "") or ""),
                **common,
            )
        if kind == CMD_ASSIGN_SHIPS_TO_SQUAD:
            raw_ship_ids = payload.get("ship_ids")
            ship_ids = tuple(
                str(ship_id).strip()
                for ship_id in raw_ship_ids
                if str(ship_id).strip()
            ) if isinstance(raw_ship_ids, list) else ()
            if not squad_id or not ship_ids:
                raise CommandValidationError("squad_id and ship_ids are required")
            return AssignShipsToSquad(
                team=team,
                ship_ids=ship_ids,
                squad_id=squad_id,
                **common,
            )
        if kind == CMD_SYNC_SETUP:
            raw_ships = payload.get("ships")
            if not isinstance(raw_ships, list):
                raise CommandValidationError("ships are required")
            ships: list[ShipSetupSpec] = []
            for item in raw_ships:
                if not isinstance(item, Mapping):
                    continue
                ship_id = str(item.get("ship_id", "") or "").strip()
                fit_text = str(item.get("fit_text", "") or "")
                if not ship_id or not fit_text.strip():
                    continue
                position = item.get("position") if isinstance(item.get("position"), Mapping) else {}
                velocity = item.get("velocity") if isinstance(item.get("velocity"), Mapping) else {}
                ships.append(
                    ShipSetupSpec(
                        ship_id=ship_id,
                        squad_id=str(item.get("squad_id", "") or ""),
                        ship_group_id=str(item.get("ship_group_id", "") or ""),
                        fit_text=fit_text,
                        position=Vector2(float(position.get("x", 0.0)), float(position.get("y", 0.0))),
                        velocity=Vector2(float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0))),
                        facing_deg=float(item.get("facing_deg", 0.0) or 0.0),
                        system_id=str(item.get("system_id", "") or ""),
                        deployed=bool(item.get("deployed", False)),
                        alive=bool(item.get("alive", False)),
                        shield=float(item.get("shield", 0.0) or 0.0),
                        armor=float(item.get("armor", 0.0) or 0.0),
                        structure=float(item.get("structure", 0.0) or 0.0),
                        cap=float(item.get("cap", 0.0) or 0.0),
                        quality_level=str(item.get("quality_level", "REGULAR") or "REGULAR"),
                        quality_reaction_delay=float(item.get("quality_reaction_delay", 0.0) or 0.0),
                        quality_ignore_order_probability=float(item.get("quality_ignore_order_probability", 0.0) or 0.0),
                        quality_formation_jitter=float(item.get("quality_formation_jitter", 0.0) or 0.0),
                    )
                )
            if not ships:
                raise CommandValidationError("at least one valid ship setup is required")
            return SyncScenarioShips(team=team, ships=tuple(ships), **common)
        if not squad_id:
            raise CommandValidationError("squad_id is required")
        if kind == CMD_SQUAD_MOVE:
            return IssueSquadMove(
                team=team,
                squad_id=squad_id,
                target=Vector2(float(payload.get("x", 0.0)), float(payload.get("y", 0.0))),
                **common,
            )
        if kind == CMD_SQUAD_APPROACH:
            return IssueSquadApproach(
                team=team,
                squad_id=squad_id,
                target_id=str(payload.get("target_id", "") or ""),
                range_m=float(payload.get("range_m", 0.0) or 0.0),
                **common,
            )
        if kind == CMD_SQUAD_NAVIGATE:
            return IssueSquadNavigate(
                team=team,
                squad_id=squad_id,
                target_kind=str(payload.get("target_kind", "") or ""),
                target_id=str(payload.get("target_id", "") or ""),
                mode=str(payload.get("mode", "") or ""),
                range_m=float(payload.get("range_m", 0.0) or 0.0),
                **common,
            )
        if kind == CMD_SQUAD_WARP:
            return IssueSquadWarp(
                team=team,
                squad_id=squad_id,
                target=Vector2(float(payload.get("x", 0.0)), float(payload.get("y", 0.0))),
                target_ship_id=str(payload.get("target_ship_id", "") or "") or None,
                target_beacon_id=str(payload.get("target_beacon_id", "") or "") or None,
                **common,
            )
        if kind == CMD_SQUAD_USE_GATE:
            return IssueSquadUseGate(
                team=team,
                squad_id=squad_id,
                structure_id=str(payload.get("target_structure_id", "") or ""),
                **common,
            )
        if kind == CMD_SQUAD_ATTACK:
            return IssueSquadFocus(team=team, squad_id=squad_id, target_id=str(payload.get("target_id", "") or ""), **common)
        if kind == CMD_SQUAD_PREFOCUS:
            return PrefocusSquadTarget(team=team, squad_id=squad_id, target_id=str(payload.get("target_id", "") or ""), **common)
        if kind == CMD_SQUAD_CANCEL_PREFOCUS:
            return CancelSquadFocus(team=team, squad_id=squad_id, target_id=str(payload.get("target_id", "") or ""), **common)
        if kind == CMD_SQUAD_CLEAR_FOCUS:
            return ClearSquadFocus(team=team, squad_id=squad_id, **common)
        if kind == CMD_SQUAD_PROPULSION:
            return SetSquadPropulsion(team=team, squad_id=squad_id, active=bool(payload.get("active", False)), **common)
        if kind == CMD_SQUAD_LEADER_SPEED_LIMIT:
            return SetSquadSpeedLimit(team=team, squad_id=squad_id, limit=float(payload.get("limit", 0.0) or 0.0), **common)
        if kind == CMD_SQUAD_LAUNCH_DRONES:
            return LaunchSquadDrones(team=team, squad_id=squad_id, type_name=str(payload.get("type_name", "") or ""), **common)
        if kind == CMD_SQUAD_LAUNCH_FIGHTERS:
            return LaunchSquadFighters(team=team, squad_id=squad_id, type_name=str(payload.get("type_name", "") or ""), **common)
        if kind == CMD_SQUAD_RECALL_DEPLOYABLES:
            return RecallSquadDrones(team=team, squad_id=squad_id, **common)
        if kind == CMD_SQUAD_DRONE_ATTACK:
            return SetSquadDroneTarget(team=team, squad_id=squad_id, target_id=str(payload.get("target_id", "") or ""), **common)
        if kind == CMD_SQUAD_FIGHTER_ATTACK:
            return SetSquadFighterTarget(team=team, squad_id=squad_id, target_id=str(payload.get("target_id", "") or ""), **common)
        if kind == CMD_SQUAD_FIGHTER_ABILITY:
            return ActivateSquadFighterAbility(
                team=team,
                squad_id=squad_id,
                ability_id=str(payload.get("ability_id", "") or ""),
                **common,
            )
        raise CommandValidationError(f"unsupported LAN command: {kind}")

    def encode(self, command: MatchCommand) -> dict[str, object]:
        payload: dict[str, object] = {"command_id": command.command_id}
        if isinstance(command, IssueSquadMove):
            payload.update(kind=CMD_SQUAD_MOVE, squad_id=command.squad_id, x=command.target.x, y=command.target.y)
        elif isinstance(command, IssueSquadApproach):
            payload.update(
                kind=CMD_SQUAD_APPROACH,
                squad_id=command.squad_id,
                target_id=command.target_id,
                range_m=command.range_m,
            )
        elif isinstance(command, IssueSquadNavigate):
            payload.update(
                kind=CMD_SQUAD_NAVIGATE,
                squad_id=command.squad_id,
                target_kind=command.target_kind,
                target_id=command.target_id,
                mode=command.mode,
                range_m=command.range_m,
            )
        elif isinstance(command, IssueSquadWarp):
            payload.update(kind=CMD_SQUAD_WARP, squad_id=command.squad_id, x=command.target.x, y=command.target.y)
            if command.target_ship_id:
                payload["target_ship_id"] = command.target_ship_id
            if command.target_beacon_id:
                payload["target_beacon_id"] = command.target_beacon_id
        elif isinstance(command, IssueSquadUseGate):
            payload.update(kind=CMD_SQUAD_USE_GATE, squad_id=command.squad_id, target_structure_id=command.structure_id)
        elif isinstance(command, IssueSquadFocus):
            payload.update(kind=CMD_SQUAD_ATTACK, squad_id=command.squad_id, target_id=command.target_id)
        elif isinstance(command, PrefocusSquadTarget):
            payload.update(kind=CMD_SQUAD_PREFOCUS, squad_id=command.squad_id, target_id=command.target_id)
        elif isinstance(command, CancelSquadFocus):
            payload.update(kind=CMD_SQUAD_CANCEL_PREFOCUS, squad_id=command.squad_id, target_id=command.target_id)
        elif isinstance(command, ClearSquadFocus):
            payload.update(kind=CMD_SQUAD_CLEAR_FOCUS, squad_id=command.squad_id)
        elif isinstance(command, SetSquadPropulsion):
            payload.update(kind=CMD_SQUAD_PROPULSION, squad_id=command.squad_id, active=command.active)
        elif isinstance(command, SetSquadSpeedLimit):
            payload.update(kind=CMD_SQUAD_LEADER_SPEED_LIMIT, squad_id=command.squad_id, limit=command.limit)
        elif isinstance(command, LaunchSquadDrones):
            payload.update(kind=CMD_SQUAD_LAUNCH_DRONES, squad_id=command.squad_id, type_name=command.type_name)
        elif isinstance(command, LaunchSquadFighters):
            payload.update(kind=CMD_SQUAD_LAUNCH_FIGHTERS, squad_id=command.squad_id, type_name=command.type_name)
        elif isinstance(command, RecallSquadDrones):
            payload.update(kind=CMD_SQUAD_RECALL_DEPLOYABLES, squad_id=command.squad_id)
        elif isinstance(command, SetSquadDroneTarget):
            payload.update(kind=CMD_SQUAD_DRONE_ATTACK, squad_id=command.squad_id, target_id=command.target_id)
        elif isinstance(command, SetSquadFighterTarget):
            payload.update(kind=CMD_SQUAD_FIGHTER_ATTACK, squad_id=command.squad_id, target_id=command.target_id)
        elif isinstance(command, ActivateSquadFighterAbility):
            payload.update(kind=CMD_SQUAD_FIGHTER_ABILITY, squad_id=command.squad_id, ability_id=command.ability_id)
        elif isinstance(command, SetShipModuleManualMode):
            payload.update(kind=CMD_SET_MODULE_MANUAL_MODE, ship_id=command.ship_id, module_id=command.module_id, mode=command.mode)
        elif isinstance(command, SetShipModuleTargetMode):
            payload.update(kind=CMD_SET_MODULE_TARGET_MODE, ship_id=command.ship_id, module_id=command.module_id, mode=command.mode)
        elif isinstance(command, SyncSquadModuleControls):
            payload.update(
                kind=CMD_SYNC_MODULE_CONTROLS,
                ship_id=command.ship_id,
                module_id=command.module_id,
                mode=command.manual_mode,
                target_mode=command.target_mode,
            )
        elif isinstance(command, SetShipModuleChargeLock):
            payload.update(
                kind=CMD_SET_MODULE_CHARGE_LOCK,
                ship_id=command.ship_id,
                module_id=command.module_id,
                charge_name=command.charge_name,
            )
        elif isinstance(command, ClearShipModuleChargeLock):
            payload.update(kind=CMD_CLEAR_MODULE_CHARGE_LOCK, ship_id=command.ship_id, module_id=command.module_id)
        elif isinstance(command, SetFleetModuleCharge):
            payload.update(
                kind=CMD_SET_FLEET_MODULE_CHARGE,
                module_name=command.module_name,
                charge_name=command.charge_name,
            )
        elif isinstance(command, AssignShipsToSquad):
            payload.update(
                kind=CMD_ASSIGN_SHIPS_TO_SQUAD,
                squad_id=command.squad_id,
                ship_ids=list(command.ship_ids),
            )
        elif isinstance(command, InduceUndeployedShips):
            payload.update(
                kind=CMD_INDUCE_SQUAD_AT if command.squad_id else CMD_INDUCE_FLEET_AT,
                system_id=command.system_id,
                x=command.center.x,
                y=command.center.y,
                radius_m=command.radius_m,
            )
            if command.squad_id:
                payload["squad_id"] = command.squad_id
        elif isinstance(command, SyncScenarioShips):
            payload.update(
                kind=CMD_SYNC_SETUP,
                ships=[
                    {
                        "ship_id": item.ship_id,
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
                ],
            )
        else:
            raise CommandValidationError(f"command cannot be encoded for LAN: {type(command).__name__}")
        return payload

    @staticmethod
    def encode_result(result: CommandResult) -> dict[str, object]:
        return {
            "command_id": result.command_id,
            "accepted": result.accepted,
            "applied_tick": result.applied_tick,
            "error_code": result.error_code,
            "message": result.message,
        }


class LanCommandGateway:
    """Send typed commands over LAN while preserving the CommandResult interface."""

    def __init__(self, session, adapter: LanCommandAdapter | None = None) -> None:
        self._session = session
        self._adapter = adapter or LanCommandAdapter()

    def execute(self, command: MatchCommand) -> CommandResult:
        self._session.send_command(self._adapter.encode(command))
        return CommandResult(command.command_id, True, None, message="sent over LAN")
