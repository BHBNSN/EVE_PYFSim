from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any, Protocol


class SnapshotSource(Protocol):
    def snapshot(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class LanPublishResult:
    published: bool
    full_sync: bool
    changed_ship_count: int = 0
    removed_ship_count: int = 0


def _stable_signature(value: Any) -> Any:
    """Create a deterministic, mildly quantized transport-change signature."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _stable_signature(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_stable_signature(item) for item in value)
    if isinstance(value, float):
        return round(value, 2)
    return value


class LanSnapshotPublisher:
    """Publish authoritative snapshots without leaking transport state into Qt code."""

    def __init__(
        self,
        session: Any,
        source: SnapshotSource,
        *,
        full_sync_interval_sec: float = 30.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session = session
        self._source = source
        self._full_sync_interval_sec = max(0.0, float(full_sync_interval_sec))
        self._clock = clock
        self._last_full_sync_at = 0.0
        self._last_ship_signatures: dict[str, Any] = {}

    def reset(self) -> None:
        self._last_full_sync_at = 0.0
        self._last_ship_signatures.clear()

    def publish(
        self,
        *,
        countdown_left: float | None = None,
        started: bool = True,
        tidi_factor: float = 1.0,
    ) -> LanPublishResult:
        if not bool(getattr(self._session, "client_connected", False)):
            self.reset()
            return LanPublishResult(False, False)

        now = self._clock()
        countdown_active = countdown_left is not None and float(countdown_left) > 0.0
        full_sync = (
            countdown_active
            or not self._last_ship_signatures
            or (now - self._last_full_sync_at) >= self._full_sync_interval_sec
        )

        base = self._source.snapshot()
        raw_ships = base.get("ships")
        ships = raw_ships if isinstance(raw_ships, Mapping) else {}
        next_signatures: dict[str, Any] = {}
        changed_ships: dict[str, dict[str, Any]] = {}
        for raw_ship_id, raw in ships.items():
            if not isinstance(raw, Mapping):
                continue
            ship_id = str(raw_ship_id)
            row = dict(raw)
            signature = _stable_signature(row)
            next_signatures[ship_id] = signature
            if full_sync or self._last_ship_signatures.get(ship_id) != signature:
                changed_ships[ship_id] = row

        removed_ship_ids = sorted(set(self._last_ship_signatures) - set(next_signatures))
        metadata = base.get("simulation_metadata")
        metadata_map = metadata if isinstance(metadata, Mapping) else {}
        engine_config = metadata_map.get("engine_config")
        engine_config_payload = dict(engine_config) if isinstance(engine_config, Mapping) else {}
        map_payload = base.get("map") if full_sync else None

        packet = {
            "snapshot": {
                "tick": base.get("tick", 0),
                "now": base.get("now", 0.0),
                "ships": changed_ships,
                "drones": base.get("drones", {}),
                "fighters": base.get("fighters", {}),
                "projectiles": base.get("projectiles", {}),
                "projectile_blasts": base.get("projectile_blasts", {}),
                "bubble_fields": base.get("bubble_fields", {}),
                "removed_ship_ids": removed_ship_ids,
                "squad_leaders": base.get("squad_leaders", {}),
                "squad_leader_location_versions": base.get("squad_leader_location_versions", {}),
                "squad_propulsion_commands": base.get("squad_propulsion_commands", {}),
                "squad_leader_speed_limits": base.get("squad_leader_speed_limits", {}),
                "squad_focus_queues": base.get("squad_focus_queues", {}),
                "squad_focus_updated_at": base.get("squad_focus_updated_at", {}),
                "partial": not full_sync,
            },
            "lan": {
                "started": bool(started),
                "countdown_left": float(max(0.0, countdown_left or 0.0)),
                "tidi_factor": max(0.0, min(1.0, float(tidi_factor))),
                "engine_config": engine_config_payload,
                "map": map_payload,
            },
        }
        self._session.send_state(packet)
        if full_sync:
            self._last_full_sync_at = now
        self._last_ship_signatures = next_signatures
        return LanPublishResult(
            True,
            full_sync,
            changed_ship_count=len(changed_ships),
            removed_ship_count=len(removed_ship_ids),
        )
