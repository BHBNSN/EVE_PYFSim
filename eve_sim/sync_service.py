from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .lan_session import ClientLanSession, HostLanSession


class HostSessionProtocol(Protocol):
    def client_connected(self) -> bool: ...
    def poll_commands(self) -> list[dict[str, Any]]: ...
    def send_state(self, snapshot: dict[str, Any]) -> None: ...
    def stop(self) -> None: ...


class ClientSessionProtocol(Protocol):
    def connected(self) -> bool: ...
    def consume_latest_state(self) -> dict[str, Any] | None: ...
    def send_command(self, command: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class HostSyncService:
    session: HostSessionProtocol

    @classmethod
    def create(cls, host: str, port: int) -> "HostSyncService":
        session = HostLanSession(host, port)
        session.start()
        return cls(session=session)

    def connected(self) -> bool:
        return bool(self.session.client_connected())

    def poll_commands(self) -> list[dict[str, Any]]:
        return self.session.poll_commands()

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.session.send_state(snapshot)

    def close(self) -> None:
        self.session.stop()


@dataclass(slots=True)
class ClientSyncService:
    session: ClientSessionProtocol

    @classmethod
    def create(cls, host: str, port: int, timeout_sec: float = 6.0) -> "ClientSyncService":
        session = ClientLanSession(host, port)
        session.connect(timeout_sec=timeout_sec)
        return cls(session=session)

    def connected(self) -> bool:
        return bool(self.session.connected())

    def consume_snapshot(self) -> dict[str, Any] | None:
        return self.session.consume_latest_state()

    def send_command(self, command: dict[str, Any]) -> None:
        self.session.send_command(dict(command))

    def close(self) -> None:
        self.session.close()


__all__ = [
    "ClientSyncService",
    "HostSyncService",
]
