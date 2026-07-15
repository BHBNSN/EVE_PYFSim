from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SnapshotOptions:
    include_runtime: bool = True
    include_diagnostics: bool = True
    include_modules: bool = True
    include_transient_entities: bool = True
