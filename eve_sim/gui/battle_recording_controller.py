from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from ..application import MatchApplication
from ..models import Team
from ..replay import ReplayRecorder


class BattleRecordingController:
    """Own replay recording state without coupling it to Qt window lifecycle code."""

    def __init__(
        self,
        application: MatchApplication,
        *,
        network_mode: str,
        controlled_team: Team,
        keyframe_interval_s: float = 300.0,
    ) -> None:
        self._application = application
        self._network_mode = str(network_mode)
        self._controlled_team = controlled_team
        self._last_snapshot_tick: int | None = None
        self.active = True
        self.recorder = ReplayRecorder(
            self._new_scenario_id(),
            keyframe_interval_s=keyframe_interval_s,
        )
        self.recorder.metadata.update(self._metadata(self._application.snapshot()))

    def _new_scenario_id(self) -> str:
        return f"{self._network_mode}-{time.strftime('%Y%m%d-%H%M%S')}"

    def _metadata(self, snapshot: dict[str, Any]) -> dict[str, object]:
        simulation_metadata = snapshot.get("simulation_metadata")
        simulation_metadata = simulation_metadata if isinstance(simulation_metadata, dict) else {}
        engine_config = simulation_metadata.get("engine_config")
        engine_config = dict(engine_config) if isinstance(engine_config, dict) else {}
        map_payload = snapshot.get("map")
        map_payload = dict(map_payload) if isinstance(map_payload, dict) else None
        metadata: dict[str, object] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "network_mode": self._network_mode,
            "controlled_team": self._controlled_team.value,
            "engine_config": engine_config,
            "replay_schema": 3,
            "keyframe_interval_s": float(self.recorder.keyframe_interval_s),
            "map_id": str((map_payload or {}).get("map_id", "") or ""),
            "map_name": str((map_payload or {}).get("name", "") or ""),
        }
        if map_payload is not None:
            metadata["map"] = map_payload
        return metadata

    def attach(self) -> None:
        self._application.attach_combat_event_sink(self.recorder.record)

    def record_snapshot(self, *, force: bool = False) -> None:
        if not self.active:
            return
        try:
            snapshot = self._application.snapshot()
        except Exception:
            return
        self._record_snapshot_payload(snapshot, force=force)

    def _record_snapshot_payload(self, snapshot: dict[str, Any], *, force: bool) -> None:
        tick = int(snapshot.get("tick", 0))
        if not force and self._last_snapshot_tick == tick:
            return
        self.recorder.record_snapshot(
            snapshot,
            tick=tick,
            at=float(snapshot.get("now", 0.0)),
            force_frame=True,
        )
        self._last_snapshot_tick = tick

    def default_path(self) -> Path:
        replay_dir = Path("logs") / "replays"
        name = f"{self.recorder.scenario_id or self._new_scenario_id()}.replay.json"
        return (replay_dir / name).resolve()

    def save(self, path: str | Path) -> Path:
        self._application.flush_pending_events()
        snapshot = self._application.snapshot()
        self._record_snapshot_payload(snapshot, force=True)
        self.recorder.metadata.update(self._metadata(snapshot))
        self.recorder.metadata.update(
            {
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "duration_s": float(snapshot.get("now", 0.0)),
                "final_tick": int(snapshot.get("tick", 0)),
            }
        )
        target = Path(path)
        self.recorder.save(target)
        return target

    def stop(self) -> None:
        self.active = False


__all__ = ["BattleRecordingController"]
