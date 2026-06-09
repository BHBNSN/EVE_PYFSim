from __future__ import annotations

from typing import Any, Iterable

from ..replay.schema import CombatEvent
from .models import BattleReport


_DAMAGE_SPLIT_FIELDS = ("em", "thermal", "kinetic", "explosive")
_DAMAGE_NAMED_FIELDS = ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive")
_REPAIR_FIELDS = ("shield_repaired", "armor_repaired", "structure_repaired", "hull_repaired", "rep_applied")


def _payload_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _add_total(target: dict[str, float], key: str, amount: float) -> None:
    if amount <= 0.0:
        return
    target[key] = float(target.get(key, 0.0)) + float(amount)


def _team_key(event: CombatEvent) -> str:
    team = event.payload.get("team") or event.payload.get("source_team")
    if team is not None and str(team):
        return str(team)
    if event.source_id:
        return str(event.source_id)
    return "unknown"


def _target_key(event: CombatEvent) -> str:
    target = event.target_id or event.payload.get("target") or event.payload.get("target_id")
    if target is not None and str(target):
        return str(target)
    return "unknown"


def _event_damage(event: CombatEvent) -> float:
    if event.kind == "ship_death":
        return 0.0
    payload = event.payload
    total_damage = _payload_float(payload, "total_damage")
    if total_damage > 0.0:
        return total_damage
    damage = _payload_float(payload, "damage") + _payload_float(payload, "applied_damage")
    if damage > 0.0:
        return damage
    split_total = sum(_payload_float(payload, key) for key in _DAMAGE_SPLIT_FIELDS)
    if split_total > 0.0:
        return split_total
    return sum(_payload_float(payload, key) for key in _DAMAGE_NAMED_FIELDS)


def _event_repair(event: CombatEvent) -> float:
    return sum(_payload_float(event.payload, key) for key in _REPAIR_FIELDS)


class BattleReportService:
    def build(
        self,
        scenario_id: str,
        events: Iterable[CombatEvent],
        *,
        duration_s: float | None = None,
    ) -> BattleReport:
        ordered_events = sorted(list(events), key=lambda event: (event.tick, event.at, event.rng_counter))
        if duration_s is None:
            duration_s = max((float(event.at) for event in ordered_events), default=0.0)

        report = BattleReport(scenario_id=str(scenario_id), duration_s=max(0.0, float(duration_s)))
        for event in ordered_events:
            damage = _event_damage(event)
            if damage > 0.0:
                _add_total(report.total_damage_by_team, _team_key(event), damage)

            repaired = _event_repair(event)
            if repaired > 0.0:
                _add_total(report.rep_applied_by_team, _team_key(event), repaired)

            if event.kind == "ecm_jam_applied":
                duration = _payload_float(event.payload, "duration_s")
                if duration > 0.0:
                    _add_total(report.jam_uptime_by_target, _target_key(event), duration)

            if event.kind in {"active_module_cycle", "command_burst"}:
                group = str(event.payload.get("group", "") or "")
                effects = str(event.payload.get("effects", "") or "")
                if "burst" in group.lower() or "burst" in effects.lower() or event.kind == "command_burst":
                    effect_key = effects or group or event.kind
                    coverage = _payload_float(event.payload, "duration_s") or _payload_float(event.payload, "cycle_time") or _payload_float(event.payload, "count", 1.0)
                    _add_total(report.burst_coverage_by_effect, effect_key, coverage)

            if event.kind == "ship_death" or bool(event.payload.get("destroyed", False)):
                report.ship_deaths.append(
                    {
                        "tick": int(event.tick),
                        "at": float(event.at),
                        "ship_id": _target_key(event),
                        "source_id": str(event.source_id),
                        "module_id": event.module_id,
                        "team": str(event.payload.get("target_team", "") or ""),
                    }
                )

        return report
