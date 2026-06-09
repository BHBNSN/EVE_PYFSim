from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

from .compact_json import compact_replay_data, expand_replay_document
from .delta import frame_has_changes, make_delta_frame, make_keyframe, replay_frames_to_snapshots
from .schema import SCHEMA_VERSION, CombatEvent, ReplayFrame, ReplaySnapshot


class ReplayRecorder:
    def __init__(
        self,
        scenario_id: str = "",
        *,
        rng_seed: int = 0,
        events: Iterable[CombatEvent] | None = None,
        frames: Iterable[ReplayFrame] | None = None,
        metadata: dict[str, Any] | None = None,
        keyframe_interval_s: float = 30.0,
    ) -> None:
        self.scenario_id = str(scenario_id)
        self.rng_seed = int(rng_seed)
        self.events: list[CombatEvent] = list(events or ())
        self.frames: list[ReplayFrame] = list(frames or ())
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.keyframe_interval_s = max(1.0, float(keyframe_interval_s))
        self._last_full_snapshot: dict[str, Any] | None = None
        self._last_keyframe_at: float | None = None
        self._snapshots_cache: list[ReplaySnapshot] | None = None
        for frame in self.frames:
            if frame.kind == "keyframe":
                self._last_keyframe_at = float(frame.at)

    def record(self, event: CombatEvent) -> None:
        self.events.append(event)

    def __call__(self, event: CombatEvent) -> None:
        self.record(event)

    def record_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        tick: int | None = None,
        at: float | None = None,
        force_keyframe: bool = False,
        force_frame: bool = False,
    ) -> None:
        resolved_tick = int(snapshot.get("tick", 0) if tick is None else tick)
        resolved_at = float(snapshot.get("now", snapshot.get("at", 0.0)) if at is None else at)
        keyframe_due = (
            self._last_full_snapshot is None
            or self._last_keyframe_at is None
            or (resolved_at - self._last_keyframe_at) >= self.keyframe_interval_s
            or bool(force_keyframe)
        )
        if keyframe_due:
            frame = make_keyframe(snapshot, tick=resolved_tick, at=resolved_at)
            self._last_keyframe_at = resolved_at
        else:
            frame = make_delta_frame(self._last_full_snapshot, snapshot, tick=resolved_tick, at=resolved_at)
            same_timeline = (
                int(self._last_full_snapshot.get("tick", -1)) == resolved_tick
                and float(self._last_full_snapshot.get("now", -1.0)) == resolved_at
            )
            if not frame_has_changes(frame) and (not force_frame or same_timeline):
                return
        self.frames.append(frame)
        self._last_full_snapshot = make_keyframe(snapshot, tick=resolved_tick, at=resolved_at).world
        self._snapshots_cache = None

    def clear(self) -> None:
        self.events.clear()
        self.frames.clear()
        self._last_full_snapshot = None
        self._last_keyframe_at = None
        self._snapshots_cache = None

    @property
    def snapshots(self) -> list[ReplaySnapshot]:
        if self._snapshots_cache is None:
            self._snapshots_cache = replay_frames_to_snapshots(self.frames)
        return list(self._snapshots_cache)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        event_duration = max((float(event.at) for event in self.events), default=0.0)
        frame_duration = max((float(frame.at) for frame in self.frames), default=0.0)
        return max(event_duration, frame_duration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "rng_seed": self.rng_seed,
            "metadata": dict(self.metadata),
            "events": [event.to_dict() for event in self.events],
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayRecorder":
        data = expand_replay_document(data)
        events = [CombatEvent.from_dict(item) for item in data.get("events", [])]
        frames = [ReplayFrame.from_dict(item) for item in data.get("frames", [])]
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            rng_seed=int(data.get("rng_seed", 0)),
            events=events,
            frames=frames,
            metadata=dict(data.get("metadata", {}) or {}),
            keyframe_interval_s=float((data.get("metadata", {}) or {}).get("keyframe_interval_s", 30.0)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(compact_replay_data(self.to_dict()), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReplayRecorder":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
