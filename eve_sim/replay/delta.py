from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .schema import ReplayFrame, ReplaySnapshot


COLLECTION_FIELDS = ("ships", "projectiles", "projectile_blasts", "bubble_fields")
REPLACE_FIELDS = ("intents", "squad_focus_queues")
TIMELINE_FIELDS = {"tick", "now", "at"}


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
        if previous.get(key) != value:
            patch[str(key)] = deepcopy(value)
    return patch


def make_delta_frame(previous: Mapping[str, Any], current: Mapping[str, Any], *, tick: int, at: float) -> ReplayFrame:
    normalized_current = normalize_snapshot(current, tick=tick, at=at)
    patch: dict[str, Any] = {}
    removed: dict[str, list[str]] = {}

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
            item_patch = _object_delta(previous_item, current_item)
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
