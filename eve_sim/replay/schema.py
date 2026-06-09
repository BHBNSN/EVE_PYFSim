from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


SCHEMA_VERSION = 4


@dataclass(slots=True)
class CombatEvent:
    tick: int
    at: float
    kind: str
    source_id: str
    target_id: str | None
    module_id: str | None
    rng_seed: int
    rng_counter: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "at": float(self.at),
            "kind": str(self.kind),
            "source_id": str(self.source_id),
            "target_id": None if self.target_id is None else str(self.target_id),
            "module_id": None if self.module_id is None else str(self.module_id),
            "rng_seed": int(self.rng_seed),
            "rng_counter": int(self.rng_counter),
            "payload": deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombatEvent":
        return cls(
            tick=int(data.get("tick", 0)),
            at=float(data.get("at", 0.0)),
            kind=str(data.get("kind", "")),
            source_id=str(data.get("source_id", "")),
            target_id=None if data.get("target_id") is None else str(data.get("target_id")),
            module_id=None if data.get("module_id") is None else str(data.get("module_id")),
            rng_seed=int(data.get("rng_seed", 0)),
            rng_counter=int(data.get("rng_counter", 0)),
            payload=deepcopy(data.get("payload", {}) or {}),
        )


CombatEventSink = Callable[[CombatEvent], None]


@dataclass(slots=True)
class ReplaySnapshot:
    tick: int
    at: float
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "at": float(self.at),
            "snapshot": deepcopy(self.snapshot),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplaySnapshot":
        return cls(
            tick=int(data.get("tick", 0)),
            at=float(data.get("at", 0.0)),
            snapshot=deepcopy(data.get("snapshot", {}) or {}),
        )


@dataclass(slots=True)
class ReplayFrame:
    tick: int
    at: float
    kind: str
    world: dict[str, Any] = field(default_factory=dict)
    patch: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tick": int(self.tick),
            "at": float(self.at),
            "kind": str(self.kind),
        }
        if self.kind == "keyframe":
            result["world"] = deepcopy(self.world)
        else:
            result["patch"] = deepcopy(self.patch)
            if self.removed:
                result["removed"] = deepcopy(self.removed)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayFrame":
        kind = str(data.get("kind") or ("keyframe" if "world" in data else "delta"))
        return cls(
            tick=int(data.get("tick", 0)),
            at=float(data.get("at", 0.0)),
            kind="keyframe" if kind == "keyframe" else "delta",
            world=deepcopy(data.get("world", {}) or {}),
            patch=deepcopy(data.get("patch", {}) or {}),
            removed={
                str(key): [str(item) for item in value]
                for key, value in (data.get("removed", {}) or {}).items()
                if isinstance(value, list)
            },
        )
