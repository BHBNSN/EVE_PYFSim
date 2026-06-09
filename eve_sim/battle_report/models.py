from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BattleReport:
    scenario_id: str
    duration_s: float
    total_damage_by_team: dict[str, float] = field(default_factory=dict)
    rep_applied_by_team: dict[str, float] = field(default_factory=dict)
    jam_uptime_by_target: dict[str, float] = field(default_factory=dict)
    burst_coverage_by_effect: dict[str, float] = field(default_factory=dict)
    ship_deaths: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "duration_s": float(self.duration_s),
            "total_damage_by_team": dict(self.total_damage_by_team),
            "rep_applied_by_team": dict(self.rep_applied_by_team),
            "jam_uptime_by_target": dict(self.jam_uptime_by_target),
            "burst_coverage_by_effect": dict(self.burst_coverage_by_effect),
            "ship_deaths": deepcopy(self.ship_deaths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BattleReport":
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            total_damage_by_team={str(k): float(v) for k, v in (data.get("total_damage_by_team", {}) or {}).items()},
            rep_applied_by_team={str(k): float(v) for k, v in (data.get("rep_applied_by_team", {}) or {}).items()},
            jam_uptime_by_target={str(k): float(v) for k, v in (data.get("jam_uptime_by_target", {}) or {}).items()},
            burst_coverage_by_effect={str(k): float(v) for k, v in (data.get("burst_coverage_by_effect", {}) or {}).items()},
            ship_deaths=deepcopy(data.get("ship_deaths", []) or []),
        )
