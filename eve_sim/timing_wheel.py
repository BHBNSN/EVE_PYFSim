from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

class EventType(Enum):
    CYCLE_END = 1
    RELOAD_END = 2
    REACTIVATION_END = 3
    EVALUATE_AI = 4
    GENERIC = 99

@dataclass(order=True)
class WheelEvent:
    trigger_time: float
    # Use counter to prevent comparing ship_id/module_id if trigger_time is equal
    _seq: int 
    event_type: EventType = field(compare=False)
    ship_id: str = field(compare=False)
    module_id: str | None = field(compare=False)
    payload: Any = field(default=None, compare=False)

class TimingWheel:
    def __init__(self) -> None:
        self._heap: list[WheelEvent] = []
        self._seq_counter_val = 0

    def schedule(self, trigger_time: float, event_type: EventType, ship_id: str, module_id: str | None = None, payload: Any = None) -> None:
        self._seq_counter_val += 1
        event = WheelEvent(
            trigger_time=trigger_time,
            _seq=self._seq_counter_val,
            event_type=event_type,
            ship_id=ship_id,
            module_id=module_id,
            payload=payload,
        )
        heapq.heappush(self._heap, event)

    def pop_due_events(self, current_time: float) -> list[WheelEvent]:
        due = []
        while self._heap and self._heap[0].trigger_time <= current_time:
            due.append(heapq.heappop(self._heap))
        return due
        
    def clear(self) -> None:
        self._heap.clear()
        
    def __len__(self) -> int:
        return len(self._heap)
