from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from ..config import EngineConfig
from ..domain.deployable_service import DeployableCommandService
from ..domain.events import ApplicationEvent
from ..domain.module_service import ShipModuleService
from ..domain.ship_fit_service import ShipFitService
from ..serialization.runtime_ship_factory import RuntimeReplicaShipFactory
from ..serialization.snapshot_loader import SnapshotLoader
from ..world import WorldState
from .commands import MatchCommand

if TYPE_CHECKING:
    from ..simulation_engine import SimulationEngine


class MatchStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


@dataclass(slots=True)
class QueuedCommand:
    sequence: int
    command: MatchCommand


@dataclass(slots=True)
class CommandQueue:
    _next_sequence: int = 0
    _items: list[QueuedCommand] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)

    def claim(self, command_id: str) -> bool:
        if command_id in self._seen_ids:
            return False
        self._seen_ids.add(command_id)
        return True

    def enqueue(self, command: MatchCommand) -> int | None:
        if not self.claim(command.command_id):
            return None
        self._next_sequence += 1
        self._items.append(QueuedCommand(self._next_sequence, command))
        return self._next_sequence

    def drain(self) -> list[QueuedCommand]:
        items = sorted(self._items, key=lambda item: item.sequence)
        self._items = []
        return items


@dataclass(slots=True)
class EventOutbox:
    _events: list[ApplicationEvent] = field(default_factory=list)

    def publish(self, *events: ApplicationEvent) -> None:
        self._events.extend(events)

    def drain(self) -> tuple[ApplicationEvent, ...]:
        events = tuple(self._events)
        self._events = []
        return events


@dataclass(slots=True)
class MatchSession:
    match_id: str
    world: WorldState
    engine: "SimulationEngine"
    config: EngineConfig
    deployable_commands: DeployableCommandService
    module_commands: ShipModuleService
    ship_fit_commands: ShipFitService
    scenario_snapshot_loader: SnapshotLoader
    command_queue: CommandQueue = field(default_factory=CommandQueue)
    event_outbox: EventOutbox = field(default_factory=EventOutbox)
    status: MatchStatus = MatchStatus.CREATED
    command_results: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_engine(
        cls,
        engine: "SimulationEngine",
        *,
        match_id: str | None = None,
        fit_parser=None,
        fit_factory=None,
    ) -> "MatchSession":
        if fit_parser is None or fit_factory is None:
            from ..fleet_setup import EftFitParser, RuntimeFromEftFactory

            fit_parser = fit_parser or EftFitParser()
            fit_factory = fit_factory or RuntimeFromEftFactory()
        from ..fleet_setup.charge_catalog import FitChargeCatalog
        ports = engine.command_ports()
        return cls(
            match_id=match_id or uuid4().hex,
            world=engine.world,
            engine=engine,
            config=engine.config,
            deployable_commands=DeployableCommandService(ports.deployables),
            module_commands=ShipModuleService(ports.module_metadata),
            ship_fit_commands=ShipFitService(
                fit_parser,
                fit_factory,
                ports.fit_runtime,
                FitChargeCatalog(),
            ),
            scenario_snapshot_loader=SnapshotLoader(RuntimeReplicaShipFactory(fit_parser, fit_factory)),
        )
