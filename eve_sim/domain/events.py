from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DomainEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApplicationEvent(DomainEvent):
    """A committed application-level event suitable for adapters to consume."""


@dataclass(frozen=True, slots=True)
class TickDiagnostics:
    execution_mode: str
    effective_execution_mode: str
    step_ms: float
    tidi_factor: float
    parallel_disabled_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TickResult:
    tick: int
    now: float
    events: tuple[DomainEvent, ...] = ()
    transfers: tuple[Any, ...] = ()
    diagnostics: TickDiagnostics | None = None
    terminal_state: str | None = None
