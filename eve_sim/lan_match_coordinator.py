from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from .application import MatchApplication
from .application.commands import MatchCommand, SyncScenarioShips
from .application.errors import CommandValidationError
from .lan_command_adapter import LanCommandAdapter, LanCommandGateway
from .lan_session import ClientLanSession, HostLanSession
from .lan_snapshot_adapter import LanPublishResult, LanSnapshotPublisher
from .maps import deserialize_map_definition
from .models import Team


@dataclass(frozen=True, slots=True)
class ClientPollResult:
    received_snapshot: bool
    removed_ship_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HostTickGate:
    should_step: bool
    status_message: str = ""


class LanMatchCoordinator:
    """Own LAN transport state, protocol translation and host/client readiness."""

    def __init__(
        self,
        application: MatchApplication,
        *,
        mode: str,
        server: HostLanSession | None = None,
        client: ClientLanSession | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._application = application
        self.mode = str(mode)
        self._server = server
        self._client = client
        self._clock = clock
        self._codec = LanCommandAdapter()
        self._gateway = LanCommandGateway(client, self._codec) if client is not None else None
        self._publisher = LanSnapshotPublisher(server, application) if server is not None else None
        self._setup_synced = False
        self._countdown_started_at: float | None = None
        self._match_started = self.mode != "host"
        self.remote_tidi_factor = 1.0
        self.debug_enabled = False

    def _debug(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[LAN][{self.mode}] {message}")

    def execute_remote(self, command: MatchCommand):
        if self._gateway is None:
            raise RuntimeError("LAN client command gateway is unavailable")
        return self._gateway.execute(command)

    def poll_client(self, team: Team) -> ClientPollResult:
        client = self._client
        if client is None or not client.connected:
            self._setup_synced = False
            return ClientPollResult(False)

        if not self._setup_synced:
            command = SyncScenarioShips(
                team=team,
                ships=self._application.query_service.scenario_ship_specs(team),
            )
            self.execute_remote(command)
            self._setup_synced = True

        packet = client.consume_latest_state()
        if packet is None:
            return ClientPollResult(False)
        result = self.apply_remote_packet(packet)
        return ClientPollResult(True, tuple(result.removed_ship_ids))

    def apply_remote_packet(self, packet: dict[str, Any]):
        self._debug("recv-snapshot")
        lan = packet.get("lan") if isinstance(packet.get("lan"), dict) else None
        if isinstance(lan, dict):
            config = lan.get("engine_config")
            if isinstance(config, dict):
                self._application.apply_replica_config(config)
            self.remote_tidi_factor = self._application.apply_replica_tidi_factor(
                lan.get("tidi_factor", self.remote_tidi_factor)
            )
            raw_map = lan.get("map")
            if isinstance(raw_map, dict):
                try:
                    map_definition = deserialize_map_definition(raw_map)
                except Exception:
                    map_definition = None
                if map_definition is not None:
                    self._application.install_replica_map(map_definition)

        snapshot = packet.get("snapshot") if isinstance(packet.get("snapshot"), dict) else packet
        if not isinstance(snapshot, dict):
            snapshot = {}
        return self._application.apply_replica_snapshot(snapshot)

    def receive_host_commands(self) -> None:
        if self._server is None:
            return
        for payload in self._server.poll_commands():
            kind = str(payload.get("kind", "") or "").upper()
            self._debug(f"recv-cmd kind={kind} payload={payload}")
            self._application.log_user_action(
                "remote_command",
                network_mode=self.mode,
                controlled_team=Team.RED.value,
                kind=kind,
                squad=str(payload.get("squad_id", "") or ""),
                target=str(payload.get("target_id", "") or ""),
            )
            try:
                command = self._codec.decode(payload, team=Team.RED)
            except CommandValidationError:
                continue
            self._application.execute(command)

    def prepare_host_tick(self, *, has_remote_fleet: bool) -> HostTickGate:
        self.receive_host_commands()
        server = self._server
        if server is None:
            return HostTickGate(True)
        if not server.client_connected:
            self._countdown_started_at = None
            self._match_started = False
            self.publish(countdown_left=10.0, started=False)
            return HostTickGate(False, "waiting for red client...")
        if not has_remote_fleet:
            self._countdown_started_at = None
            self._match_started = False
            self.publish(countdown_left=10.0, started=False)
            return HostTickGate(False, "waiting for red fleet sync...")
        if self._match_started:
            return HostTickGate(True)

        now = self._clock()
        if self._countdown_started_at is None:
            self._countdown_started_at = now
        left = 10.0 - (now - self._countdown_started_at)
        if left > 0.0:
            self.publish(countdown_left=left, started=False)
            return HostTickGate(False, f"match starts in {left:.1f}s")
        self._match_started = True
        return HostTickGate(True)

    def publish(self, *, countdown_left: float | None = None, started: bool = True) -> LanPublishResult:
        if self._publisher is None:
            return LanPublishResult(False, False)
        result = self._publisher.publish(
            countdown_left=countdown_left,
            started=started,
            tidi_factor=self._application.tidi_factor(),
        )
        if result.published:
            self._debug(
                f"send-snapshot ships={result.changed_ship_count} full={result.full_sync} "
                f"removed={result.removed_ship_count} countdown={countdown_left} started={started}"
            )
        return result

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
        if self._client is not None:
            self._client.close()


__all__ = ["ClientPollResult", "HostTickGate", "LanMatchCoordinator"]
