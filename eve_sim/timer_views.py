from __future__ import annotations

from typing import Mapping, MutableMapping


def deadline_remaining(deadline: float | None, now: float) -> float | None:
    if deadline is None:
        return None
    try:
        return max(0.0, float(deadline) - float(now))
    except Exception:
        return None


def sync_deadline_view(
    deadline_map: Mapping[str, float],
    view_map: MutableMapping[str, float],
    now: float,
) -> None:
    for key, deadline in list(deadline_map.items()):
        remaining = deadline_remaining(deadline, now)
        if remaining is None:
            continue
        view_map[str(key)] = remaining


def normalize_remaining_view(
    raw_view: Mapping[object, object],
    *,
    epsilon: float = 0.0,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, raw_value in raw_view.items():
        key = str(raw_key or "")
        if not key:
            continue
        try:
            remaining = max(0.0, float(raw_value))
        except Exception:
            continue
        if remaining <= float(epsilon):
            normalized[key] = 0.0
            continue
        normalized[key] = remaining
    return normalized


def deadline_map_from_remaining_view(
    raw_view: Mapping[object, object],
    now: float,
    *,
    epsilon: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float]]:
    normalized = normalize_remaining_view(raw_view, epsilon=epsilon)
    deadlines: dict[str, float] = {}
    now_value = float(now)
    for key, remaining in normalized.items():
        if remaining <= float(epsilon):
            continue
        deadlines[key] = now_value + remaining
    return normalized, deadlines

