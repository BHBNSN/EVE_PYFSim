from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...application import MatchApplication
from ...world import WorldState


class WorldViewSource(Protocol):
    """Minimal read source consumed by presentation widgets."""

    @property
    def world(self) -> WorldState: ...


class ApplicationRuntimeView:
    """Read adapter that keeps presentation code away from SimulationEngine internals."""

    __slots__ = ("_application", "_world")

    def __init__(self, application: MatchApplication) -> None:
        self._application = application
        self._world = application.create_presentation_world()

    @property
    def world(self) -> WorldState:
        return self._world

    def refresh(self) -> None:
        self._application.refresh_presentation_world(self._world)

    @property
    def tick_rate(self) -> int:
        values = self._application.query_service.diagnostics().values
        tick_rate = values.get("tick_rate", 1)
        return max(1, int(tick_rate))

    @property
    def simulation_dt(self) -> float:
        return 1.0 / float(self.tick_rate)

    @property
    def tidi_factor(self) -> float:
        return self._application.tidi_factor()

    def diagnostics(self) -> dict[str, object]:
        return dict(self._application.query_service.diagnostics().values)


@dataclass(slots=True)
class ReplayRuntimeView:
    """Detached read source for replay rendering; it owns no simulation kernel."""

    world: WorldState
    tick_rate: int = 1
    tidi_factor: float = 1.0

    @property
    def simulation_dt(self) -> float:
        return 1.0 / float(max(1, int(self.tick_rate)))


__all__ = ["ApplicationRuntimeView", "ReplayRuntimeView", "WorldViewSource"]
