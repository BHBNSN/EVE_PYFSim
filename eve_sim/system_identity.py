from __future__ import annotations

import hashlib
from urllib.parse import quote


def normalize_system_namespace(system_id: str) -> str:
    """Return a stable, delimiter-safe namespace for a simulation system."""
    normalized = str(system_id or "").strip()
    if not normalized:
        raise ValueError("system_id must not be empty")
    return quote(normalized, safe="-._~")


def stable_system_seed(match_seed: int, system_id: str) -> int:
    namespace = normalize_system_namespace(system_id)
    payload = f"{int(match_seed)}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


__all__ = ["normalize_system_namespace", "stable_system_seed"]
