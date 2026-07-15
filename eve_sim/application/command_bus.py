from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..domain.events import ApplicationEvent
from .commands import MatchCommand
from .errors import CommandValidationError

if TYPE_CHECKING:
    from .session import MatchSession


CommandHandler = Callable[["MatchSession", MatchCommand], tuple[ApplicationEvent, ...]]


@dataclass(frozen=True, slots=True)
class CommandResult:
    command_id: str
    accepted: bool
    applied_tick: int | None
    error_code: str | None = None
    message: str = ""
    emitted_events: tuple[ApplicationEvent, ...] = ()


class CommandBus:
    def __init__(self) -> None:
        self._handlers: dict[type[MatchCommand], CommandHandler] = {}

    def register(self, command_type: type[MatchCommand], handler: CommandHandler) -> None:
        self._handlers[command_type] = handler

    def dispatch(self, session: "MatchSession", command: MatchCommand) -> CommandResult:
        handler = next((self._handlers[typ] for typ in type(command).__mro__ if typ in self._handlers), None)
        if handler is None:
            return CommandResult(command.command_id, False, None, "unsupported_command", type(command).__name__)
        try:
            events = handler(session, command)
        except CommandValidationError as exc:
            return CommandResult(command.command_id, False, None, "validation_error", str(exc))
        except ValueError as exc:
            return CommandResult(command.command_id, False, None, "validation_error", str(exc))
        return CommandResult(command.command_id, True, int(session.world.tick), emitted_events=events)
