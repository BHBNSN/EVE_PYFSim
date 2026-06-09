from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator
import json

from .delta import apply_frame
from .schema import CombatEvent, ReplayFrame, ReplaySnapshot


class ReplayPlayer:
    def __init__(
        self,
        events: Iterable[CombatEvent] | None = None,
        *,
        snapshots: Iterable[ReplaySnapshot] | None = None,
        frames: Iterable[ReplayFrame] | None = None,
        metadata: dict[str, Any] | None = None,
        scenario_id: str = "",
        rng_seed: int = 0,
    ) -> None:
        self.scenario_id = str(scenario_id)
        self.rng_seed = int(rng_seed)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.events = sorted(list(events), key=lambda event: (event.tick, event.at, event.rng_counter))
        snapshot_frames = [ReplayFrame.from_snapshot(snapshot) for snapshot in (snapshots or ())]
        self.frames = sorted(list(frames or snapshot_frames), key=lambda frame: (frame.tick, frame.at))
        self._last_resolved_index: int | None = None
        self._last_resolved_snapshot: dict[str, Any] | None = None
        self._keyframe_indices = [index for index, frame in enumerate(self.frames) if frame.kind == "keyframe"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayPlayer":
        frames = [ReplayFrame.from_dict(item) for item in data.get("frames", [])]
        snapshots = [] if frames else [ReplaySnapshot.from_dict(item) for item in data.get("snapshots", [])]
        return cls(
            (CombatEvent.from_dict(item) for item in data.get("events", [])),
            snapshots=snapshots,
            frames=frames,
            metadata=dict(data.get("metadata", {}) or {}),
            scenario_id=str(data.get("scenario_id", "")),
            rng_seed=int(data.get("rng_seed", 0)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayPlayer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @property
    def snapshot_count(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        event_duration = max((float(event.at) for event in self.events), default=0.0)
        frame_duration = max((float(frame.at) for frame in self.frames), default=0.0)
        return max(event_duration, frame_duration)

    def iter_events(self, *, kind: str | None = None) -> Iterator[CombatEvent]:
        for event in self.events:
            if kind is not None and event.kind != kind:
                continue
            yield event

    def events_between(self, start_tick: int, end_tick: int) -> list[CombatEvent]:
        return [
            event
            for event in self.events
            if int(start_tick) <= int(event.tick) <= int(end_tick)
        ]

    def events_until_tick(self, tick: int) -> list[CombatEvent]:
        return [event for event in self.events if int(event.tick) <= int(tick)]

    def snapshot_at_index(self, index: int) -> ReplaySnapshot:
        if not self.frames:
            raise IndexError("Replay has no frames")
        clamped = max(0, min(int(index), len(self.frames) - 1))
        frame = self.frames[clamped]
        if (
            self._last_resolved_index is not None
            and self._last_resolved_snapshot is not None
            and clamped == self._last_resolved_index + 1
        ):
            snapshot = apply_frame(self._last_resolved_snapshot, frame)
            self._last_resolved_index = clamped
            self._last_resolved_snapshot = snapshot
            return ReplaySnapshot(tick=int(frame.tick), at=float(frame.at), snapshot=snapshot)

        start_index = self._nearest_keyframe_index(clamped)
        current: dict[str, Any] | None = None
        for frame_index in range(start_index, clamped + 1):
            current = apply_frame(current, self.frames[frame_index])
        if current is None:
            raise IndexError("Replay frame could not be resolved")
        self._last_resolved_index = clamped
        self._last_resolved_snapshot = current
        return ReplaySnapshot(tick=int(frame.tick), at=float(frame.at), snapshot=current)

    def index_for_tick(self, tick: int) -> int:
        if not self.frames:
            return 0
        resolved_tick = int(tick)
        best_index = 0
        for index, frame in enumerate(self.frames):
            if int(frame.tick) > resolved_tick:
                break
            best_index = index
        return best_index

    def _nearest_keyframe_index(self, index: int) -> int:
        if not self._keyframe_indices:
            return 0
        best = 0
        for keyframe_index in self._keyframe_indices:
            if keyframe_index > index:
                break
            best = keyframe_index
        return best
