from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping

from ..domain.events import ApplicationEvent, TickResult
from .command_bus import CommandBus, CommandResult
from .command_handlers import DefaultCommandHandlers
from .commands import AdvanceTicks, MatchCommand, PauseMatch, ResumeMatch
from .query_service import QueryService
from .session import MatchSession, MatchStatus

if TYPE_CHECKING:
    from ..serialization.snapshot_builder import SnapshotOptions


class MatchApplication:
    """Single application entry point for UI, LAN, tests and future headless hosts."""

    def __init__(self, session: MatchSession, command_bus: CommandBus | None = None, query_service: QueryService | None = None) -> None:
        self.session = session
        self.command_bus = command_bus or self._default_command_bus()
        self.query_service = query_service or QueryService(session)

    @classmethod
    def from_engine(
        cls,
        engine,
        *,
        match_id: str | None = None,
        fit_parser=None,
        fit_factory=None,
    ) -> "MatchApplication":
        return cls(
            MatchSession.from_engine(
                engine,
                match_id=match_id,
                fit_parser=fit_parser,
                fit_factory=fit_factory,
            )
        )

    @staticmethod
    def _default_command_bus() -> CommandBus:
        from . import commands

        handlers = DefaultCommandHandlers()
        bus = CommandBus()
        registrations = {
            commands.IssueSquadMove: handlers.move,
            commands.IssueSquadApproach: handlers.approach,
            commands.IssueSquadNavigate: handlers.navigate,
            commands.ClearSquadNavigation: handlers.clear_navigation,
            commands.IssueSquadWarp: handlers.warp,
            commands.IssueSquadUseGate: handlers.use_gate,
            commands.IssueSquadFocus: handlers.focus,
            commands.PrefocusSquadTarget: handlers.prefocus,
            commands.CancelSquadFocus: handlers.cancel_focus,
            commands.ClearSquadFocus: handlers.clear_focus,
            commands.SetSquadPropulsion: handlers.propulsion,
            commands.SetSquadSpeedLimit: handlers.speed_limit,
            commands.LaunchSquadDrones: handlers.launch_drones,
            commands.LaunchSquadFighters: handlers.launch_fighters,
            commands.RecallSquadDrones: handlers.recall_deployables,
            commands.RecallSquadFighters: handlers.recall_deployables,
            commands.SetSquadDroneTarget: handlers.drone_target,
            commands.SetSquadFighterTarget: handlers.fighter_target,
            commands.ActivateSquadFighterAbility: handlers.fighter_ability,
            commands.AssignShipsToSquad: handlers.assign_squad,
            commands.InduceShips: handlers.induce_ships,
            commands.InduceUndeployedShips: handlers.induce_undeployed_ships,
            commands.InitializeTeamDeployment: handlers.initialize_team_deployment,
            commands.InstallMapDefinition: handlers.install_map_definition,
            commands.SetShipModuleManualMode: handlers.set_ship_module_manual_mode,
            commands.SetShipModuleTargetMode: handlers.set_ship_module_target_mode,
            commands.SyncSquadModuleControls: handlers.sync_squad_module_controls,
            commands.SetShipModuleChargeLock: handlers.set_ship_module_charge_lock,
            commands.ClearShipModuleChargeLock: handlers.clear_ship_module_charge_lock,
            commands.SetFleetModuleCharge: handlers.set_fleet_module_charge,
            commands.SyncScenarioShips: handlers.sync_scenario_ships,
            commands.SetShipDeployment: handlers.deployment,
            commands.ReplaceScenario: handlers.replace_scenario,
        }
        for command_type, handler in registrations.items():
            bus.register(command_type, handler)
        return bus

    def execute(self, command: MatchCommand) -> CommandResult:
        if self.session.status is MatchStatus.CLOSED:
            return CommandResult(command.command_id, False, None, "closed", "match is closed")
        if isinstance(command, (PauseMatch, ResumeMatch, AdvanceTicks)):
            if not self.session.command_queue.claim(command.command_id):
                return CommandResult(
                    command.command_id,
                    False,
                    None,
                    "duplicate_command",
                    "command_id was already received",
                )
        if isinstance(command, PauseMatch):
            self.session.status = MatchStatus.PAUSED
            event = ApplicationEvent("match_paused", {"command_id": command.command_id})
            self.session.event_outbox.publish(event)
            result = CommandResult(command.command_id, True, int(self.session.world.tick), emitted_events=(event,))
            self.session.command_results[command.command_id] = result
            return result
        if isinstance(command, ResumeMatch):
            self.session.status = MatchStatus.RUNNING
            event = ApplicationEvent("match_resumed", {"command_id": command.command_id})
            self.session.event_outbox.publish(event)
            result = CommandResult(command.command_id, True, int(self.session.world.tick), emitted_events=(event,))
            self.session.command_results[command.command_id] = result
            return result
        if isinstance(command, AdvanceTicks):
            results = self.step(command.count)
            result = CommandResult(command.command_id, True, results[-1].tick if results else int(self.session.world.tick))
            self.session.command_results[command.command_id] = result
            return result
        sequence = self.session.command_queue.enqueue(command)
        if sequence is None:
            return CommandResult(command.command_id, False, None, "duplicate_command", "command_id was already received")
        return CommandResult(command.command_id, True, None, message=f"queued at sequence {sequence}")

    def _apply_queued_commands(self) -> tuple[ApplicationEvent, ...]:
        events: list[ApplicationEvent] = []
        for queued in self.session.command_queue.drain():
            result = self.command_bus.dispatch(self.session, queued.command)
            self.session.command_results[queued.command.command_id] = result
            if result.accepted:
                events.extend(result.emitted_events)
        if events:
            self.session.event_outbox.publish(*events)
        return tuple(events)

    def prepare(self) -> tuple[ApplicationEvent, ...]:
        """Commit queued scenario setup before the first simulation tick."""
        if self.session.status is not MatchStatus.CREATED:
            raise RuntimeError("match setup can only be committed before the first tick")
        return self._apply_queued_commands()

    def step(self, count: int = 1) -> list[TickResult]:
        if self.session.status is MatchStatus.CLOSED:
            raise RuntimeError("match is closed")
        if self.session.status is MatchStatus.PAUSED:
            return []
        self.session.status = MatchStatus.RUNNING
        results: list[TickResult] = []
        for _ in range(max(0, int(count))):
            application_events = self._apply_queued_commands()
            try:
                tick_result = self.session.engine.step()
            except Exception:
                self.session.status = MatchStatus.FAILED
                raise
            if application_events:
                tick_result = TickResult(
                    tick=tick_result.tick,
                    now=tick_result.now,
                    events=(*application_events, *tick_result.events),
                    transfers=tick_result.transfers,
                    diagnostics=tick_result.diagnostics,
                    terminal_state=tick_result.terminal_state,
                )
            results.append(tick_result)
        return results

    def snapshot(self, options: "SnapshotOptions | None" = None):
        from ..serialization.snapshot_builder import SnapshotBuilder

        diagnostics = self.session.engine.runtime_diagnostics()
        metadata = {
            "system_execution_mode": diagnostics["execution_mode"],
            "effective_system_execution_mode": diagnostics["effective_execution_mode"],
            "parallel_disabled_reason": diagnostics["parallel_disabled_reason"],
            "parallel_disabled_at_tick": diagnostics["parallel_disabled_at_tick"],
            "parallel_failure_count": diagnostics["parallel_failure_count"],
            "engine_config": asdict(self.session.config),
        }
        return SnapshotBuilder().build(
            self.session.world,
            options,
            simulation_metadata=metadata,
        )

    def apply_replica_snapshot(self, snapshot: Mapping[str, Any]):
        """Apply an authoritative snapshot to a non-authoritative match replica."""
        result = self.session.scenario_snapshot_loader.apply_delta(self.session.world, snapshot)
        for ship_id in result.added_ship_ids:
            self.session.engine.register_ship(ship_id)
        for ship_id in result.removed_ship_ids:
            self.session.engine.unregister_ship(ship_id)
        return result

    def create_presentation_world(self):
        """Create a detached world projection for rich presentation rendering."""
        return self.session.scenario_snapshot_loader.load_world(self.snapshot())

    def refresh_presentation_world(self, world) -> None:
        """Refresh a detached presentation projection from authoritative state."""
        self.session.scenario_snapshot_loader.apply_replica(world, self.snapshot())

    def install_replica_map(self, map_definition) -> bool:
        """Install changed map metadata on a non-authoritative replica."""
        world = self.session.world
        if world.map_id == str(map_definition.map_id or "") and world.structures:
            return False
        from .commands import InstallMapDefinition

        self.execute(InstallMapDefinition(map_definition=map_definition))
        self.prepare()
        return True

    def next_tick_delay_ms(self) -> int:
        return int(self.session.engine.next_tick_delay_ms())

    def update_tidi_after_step(self, elapsed_ms: float) -> None:
        self.session.engine.update_tidi_after_step(float(elapsed_ms))

    def tidi_factor(self) -> float:
        return self.session.engine.current_tidi_factor()

    def log_user_action(
        self,
        action: str,
        *,
        network_mode: str,
        controlled_team: str,
        **fields: object,
    ) -> None:
        if not bool(self.session.config.detailed_logging):
            return
        from ..sim_logging import get_sim_logger, log_sim_event

        payload: dict[str, object] = {
            "action": str(action),
            "network_mode": str(network_mode),
            "controlled_team": str(controlled_team),
            "tick": int(self.session.world.tick),
        }
        payload.update(fields)
        log_sim_event(get_sim_logger(self.session.config), "user_operation", **payload)

    def attach_combat_event_sink(self, sink) -> None:
        self.session.engine.subscribe_combat_events(sink)

    def flush_pending_events(self) -> None:
        self.session.engine.flush_pending_combat_events()

    def apply_replica_config(self, payload: dict[str, object]) -> None:
        """Apply host timing/logging settings to a non-authoritative replica engine."""
        engine = self.session.engine
        config = self.session.config
        try:
            tick_rate = max(1, int(float(payload.get("tick_rate", config.tick_rate))))
        except (TypeError, ValueError):
            tick_rate = max(1, int(config.tick_rate))
        try:
            substeps = max(1, int(float(payload.get("physics_substeps", config.physics_substeps))))
        except (TypeError, ValueError):
            substeps = max(1, int(config.physics_substeps))
        try:
            merge_window = max(0.1, float(payload.get("log_merge_window_sec", config.log_merge_window_sec)))
        except (TypeError, ValueError):
            merge_window = max(0.1, float(config.log_merge_window_sec))
        try:
            tidi_min_factor = max(0.01, min(1.0, float(payload.get("tidi_min_factor", config.tidi_min_factor))))
        except (TypeError, ValueError):
            tidi_min_factor = max(0.01, min(1.0, float(getattr(config, "tidi_min_factor", 0.1))))

        config.tick_rate = tick_rate
        config.physics_substeps = substeps
        config.lockstep = bool(payload.get("lockstep", config.lockstep))
        config.tidi_min_factor = tidi_min_factor
        config.detailed_logging = bool(payload.get("detailed_logging", config.detailed_logging))
        config.hotspot_logging = bool(payload.get("hotspot_logging", config.hotspot_logging))
        config.detail_log_file = str(payload.get("detail_log_file", config.detail_log_file))
        config.hotspot_log_file = str(payload.get("hotspot_log_file", config.hotspot_log_file))
        config.log_merge_window_sec = merge_window

        engine.refresh_runtime_from_config()

    def apply_replica_tidi_factor(self, value: object) -> float:
        """Update non-authoritative timing state without exposing the engine to adapters."""
        try:
            factor = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            factor = 1.0
        self.session.engine.set_replica_tidi_factor(factor)
        return factor

    def close(self) -> None:
        if self.session.status is MatchStatus.CLOSED:
            return
        self.session.engine.close()
        self.session.status = MatchStatus.CLOSED
