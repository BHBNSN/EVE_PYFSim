from __future__ import annotations

from typing import Any, Callable, Mapping


def normalize_projection_effect_signature(raw: Any) -> tuple[Any, ...] | None:
    if not isinstance(raw, (tuple, list)):
        return None
    try:
        return tuple(raw)
    except Exception:
        return None


def quantize_projection_range(distance: float, bucket_m: float = 100.0) -> float:
    safe_distance = max(0.0, float(distance or 0.0))
    if bucket_m <= 0.0:
        return safe_distance
    return float(int(safe_distance // bucket_m) * bucket_m)


def normalized_snapshot_projection_signature(
    snapshot: Mapping[str, Any],
    *,
    bucket_m: float = 100.0,
) -> tuple[str, Any]:
    projection_key_mode = str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range")
    if projection_key_mode == "exact_range":
        try:
            distance_signature: Any = round(
                quantize_projection_range(
                    float(snapshot.get("pyfa_projection_range", snapshot.get("projection_range", 0.0)) or 0.0),
                    bucket_m=bucket_m,
                ),
                3,
            )
        except Exception:
            distance_signature = 0.0
        return "exact_range", distance_signature
    return "in_range", None


def projected_snapshot_module_signature(
    snapshot: Mapping[str, Any],
    *,
    legacy_builder: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> tuple[Any, ...]:
    direct_signature = normalize_projection_effect_signature(snapshot.get("pyfa_projection_module_signature"))
    if direct_signature is not None:
        return direct_signature
    return legacy_builder(snapshot)


def projected_snapshot_list_signature(
    snapshots: list[dict[str, Any]],
    *,
    module_signature_builder: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    bucket_m: float = 100.0,
) -> tuple[Any, ...]:
    signature: list[tuple[Any, ...]] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        projection_key_mode, distance_signature = normalized_snapshot_projection_signature(snapshot, bucket_m=bucket_m)
        signature.append(
            (
                module_signature_builder(snapshot),
                projection_key_mode,
                distance_signature,
            )
        )
    return tuple(signature)
