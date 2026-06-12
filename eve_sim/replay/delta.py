from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .schema import ReplayFrame, ReplaySnapshot


COLLECTION_FIELDS = ("ships", "drones", "fighters", "projectiles", "projectile_blasts", "bubble_fields")
REPLACE_FIELDS = ("intents", "squad_focus_queues", "squad_focus_updated_at")
TIMELINE_FIELDS = {"tick", "now", "at"}
FLOAT_EPSILON = 1e-3


def normalize_snapshot(snapshot: Mapping[str, Any], *, tick: int, at: float) -> dict[str, Any]:
    normalized = deepcopy(dict(snapshot))
    normalized["tick"] = int(tick)
    normalized["now"] = float(at)
    return normalized


def make_keyframe(snapshot: Mapping[str, Any], *, tick: int, at: float) -> ReplayFrame:
    return ReplayFrame(tick=int(tick), at=float(at), kind="keyframe", world=normalize_snapshot(snapshot, tick=tick, at=at))


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _object_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for key, value in current.items():
        if not _values_equal(previous.get(key), value):
            patch[str(key)] = deepcopy(value)
    return patch


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= FLOAT_EPSILON
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_values_equal(left.get(key), right.get(key)) for key in left.keys())
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    return left == right


def _position_plus_velocity(position: Any, velocity: Any, dt: float) -> Any:
    if not isinstance(position, Mapping) or not isinstance(velocity, Mapping):
        return deepcopy(position)
    return {
        "x": _float(position.get("x")) + (_float(velocity.get("x")) * dt),
        "y": _float(position.get("y")) + (_float(velocity.get("y")) * dt),
    }


def _advance_cycle_timers(raw: Any, dt: float) -> Any:
    if not isinstance(raw, Mapping):
        return deepcopy(raw)
    advanced: dict[str, float] = {}
    for module_id, remaining in raw.items():
        advanced[str(module_id)] = max(0.0, _float(remaining) - dt)
    return advanced


def _advance_collection_item(field: str, item: Mapping[str, Any], dt: float) -> dict[str, Any]:
    advanced = deepcopy(dict(item))
    if dt <= 0.0:
        return advanced
    if field in {"ships", "drones", "fighters", "projectiles"} and "position" in advanced:
        advanced["position"] = _position_plus_velocity(advanced.get("position"), advanced.get("velocity"), dt)
    if field == "ships" and "module_cycle_timers" in advanced:
        advanced["module_cycle_timers"] = _advance_cycle_timers(advanced.get("module_cycle_timers"), dt)
    if field == "drones":
        for timer_key in ("cycle_timer", "ewar_cycle_timer"):
            if timer_key in advanced:
                advanced[timer_key] = max(0.0, _float(advanced.get(timer_key)) - dt)
    if field == "fighters":
        if "ability_cycle_timers" in advanced:
            advanced["ability_cycle_timers"] = _advance_cycle_timers(advanced.get("ability_cycle_timers"), dt)
        if "ability_reload_timers" in advanced:
            advanced["ability_reload_timers"] = _advance_cycle_timers(advanced.get("ability_reload_timers"), dt)
        for timer_key in ("mwd_active_timer", "mwd_cooldown_timer"):
            if timer_key in advanced:
                advanced[timer_key] = max(0.0, _float(advanced.get(timer_key)) - dt)
    if field == "projectiles":
        if "age" in advanced:
            advanced["age"] = max(0.0, _float(advanced.get("age")) + dt)
        if "flight_time" in advanced:
            advanced["flight_time"] = max(0.0, _float(advanced.get("flight_time")) + dt)
        if "distance_traveled" in advanced:
            speed = _float(advanced.get("speed"), 0.0)
            if speed <= 0.0 and isinstance(advanced.get("velocity"), Mapping):
                vx = _float(advanced["velocity"].get("x"))
                vy = _float(advanced["velocity"].get("y"))
                speed = ((vx * vx) + (vy * vy)) ** 0.5
            advanced["distance_traveled"] = max(0.0, _float(advanced.get("distance_traveled")) + (speed * dt))
    return advanced


def _timeline_delta(previous: Mapping[str, Any], at: float) -> float:
    return max(0.0, float(at) - _float(previous.get("now", previous.get("at", at)), float(at)))


def _advance_snapshot(snapshot: dict[str, Any], dt: float) -> None:
    if dt <= 0.0:
        return
    for field in COLLECTION_FIELDS:
        collection = snapshot.get(field)
        if not isinstance(collection, dict):
            continue
        for item_id, item in list(collection.items()):
            if isinstance(item, Mapping):
                collection[str(item_id)] = _advance_collection_item(field, item, dt)


def make_delta_frame(previous: Mapping[str, Any], current: Mapping[str, Any], *, tick: int, at: float) -> ReplayFrame:
    normalized_current = normalize_snapshot(current, tick=tick, at=at)
    patch: dict[str, Any] = {}
    removed: dict[str, list[str]] = {}
    dt = _timeline_delta(previous, at)

    for field in COLLECTION_FIELDS:
        previous_items = _mapping_or_empty(previous.get(field))
        current_items = _mapping_or_empty(normalized_current.get(field))
        updates: dict[str, Any] = {}
        for item_id, current_item in current_items.items():
            sid = str(item_id)
            if not isinstance(current_item, Mapping):
                if previous_items.get(item_id) != current_item:
                    updates[sid] = deepcopy(current_item)
                continue
            previous_item = previous_items.get(item_id)
            if not isinstance(previous_item, Mapping):
                updates[sid] = deepcopy(dict(current_item))
                continue
            predicted_previous_item = _advance_collection_item(field, previous_item, dt)
            item_patch = _object_delta(predicted_previous_item, current_item)
            if item_patch:
                updates[sid] = item_patch
        removed_ids = sorted(str(item_id) for item_id in previous_items.keys() if item_id not in current_items)
        if updates:
            patch[field] = updates
        if removed_ids:
            removed[field] = removed_ids

    for field in REPLACE_FIELDS:
        if previous.get(field) != normalized_current.get(field):
            patch[field] = deepcopy(normalized_current.get(field, {}))

    skipped = set(COLLECTION_FIELDS) | set(REPLACE_FIELDS) | TIMELINE_FIELDS
    for key, value in normalized_current.items():
        if key in skipped:
            continue
        if previous.get(key) != value:
            patch[str(key)] = deepcopy(value)

    return ReplayFrame(tick=int(tick), at=float(at), kind="delta", patch=patch, removed=removed)


def frame_has_changes(frame: ReplayFrame) -> bool:
    return frame.kind == "keyframe" or bool(frame.patch) or bool(frame.removed)


def apply_frame(previous: Mapping[str, Any] | None, frame: ReplayFrame) -> dict[str, Any]:
    if frame.kind == "keyframe":
        return normalize_snapshot(frame.world, tick=frame.tick, at=frame.at)

    snapshot = deepcopy(dict(previous or {}))
    dt = _timeline_delta(snapshot, frame.at)
    _advance_snapshot(snapshot, dt)
    snapshot["tick"] = int(frame.tick)
    snapshot["now"] = float(frame.at)

    for field in COLLECTION_FIELDS:
        collection = snapshot.get(field)
        if not isinstance(collection, dict):
            collection = {}
            snapshot[field] = collection
        for item_id in frame.removed.get(field, []):
            collection.pop(str(item_id), None)
        updates = frame.patch.get(field)
        if not isinstance(updates, Mapping):
            continue
        for item_id, item_patch in updates.items():
            sid = str(item_id)
            if isinstance(item_patch, Mapping):
                existing = collection.get(sid)
                if isinstance(existing, Mapping):
                    merged = dict(existing)
                    for key, value in item_patch.items():
                        merged[str(key)] = deepcopy(value)
                    collection[sid] = merged
                else:
                    collection[sid] = deepcopy(dict(item_patch))
            else:
                collection[sid] = deepcopy(item_patch)

    skipped = set(COLLECTION_FIELDS)
    for key, value in frame.patch.items():
        if key in skipped:
            continue
        snapshot[str(key)] = deepcopy(value)

    return snapshot


def replay_frames_to_snapshots(frames: list[ReplayFrame]) -> list[ReplaySnapshot]:
    snapshots: list[ReplaySnapshot] = []
    current: dict[str, Any] | None = None
    for frame in frames:
        current = apply_frame(current, frame)
        snapshots.append(ReplaySnapshot(tick=int(frame.tick), at=float(frame.at), snapshot=deepcopy(current)))
    return snapshots
