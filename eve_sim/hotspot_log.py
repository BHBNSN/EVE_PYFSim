from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import shlex


@dataclass(slots=True)
class HotspotRecord:
    timestamp: str
    name: str
    duration_ms: float
    tick: int | None
    line_number: int
    fields: dict[str, str]
    raw_line: str


@dataclass(slots=True)
class HotspotSummary:
    name: str
    calls: int
    total_ms: float
    avg_ms: float
    p95_ms: float
    max_ms: float
    last_tick: int | None


@dataclass(slots=True)
class TickAggregate:
    tick: int
    total_ms: float
    max_ms: float
    calls: int


def parse_event_fields(message: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        tokens = shlex.split(message)
    except Exception:
        return fields
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        fields[key] = value
    return fields


def parse_hotspot_line(line: str, *, line_number: int = 0) -> HotspotRecord | None:
    text = line.rstrip("\n")
    parts = text.split(" | ", 2)
    if len(parts) != 3:
        return None
    timestamp, _level, message = parts
    fields = parse_event_fields(message)
    if fields.get("event") != "hotspot":
        return None
    name = str(fields.get("name", "")).strip()
    if not name:
        return None
    try:
        duration_ms = float(fields.get("duration_ms", ""))
    except Exception:
        return None
    tick: int | None = None
    tick_raw = fields.get("tick")
    if tick_raw is not None:
        try:
            tick = int(float(tick_raw))
        except Exception:
            tick = None
    return HotspotRecord(
        timestamp=timestamp.strip(),
        name=name,
        duration_ms=duration_ms,
        tick=tick,
        line_number=line_number,
        fields=fields,
        raw_line=text,
    )


def load_hotspot_records(path: str | Path) -> list[HotspotRecord]:
    log_path = Path(path)
    records: list[HotspotRecord] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = parse_hotspot_line(line, line_number=line_number)
            if record is not None:
                records.append(record)
    return records


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    bounded = max(0.0, min(1.0, float(ratio)))
    index = (len(ordered) - 1) * bounded
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_records(records: list[HotspotRecord]) -> list[HotspotSummary]:
    grouped: dict[str, list[HotspotRecord]] = defaultdict(list)
    for record in records:
        grouped[record.name].append(record)

    summaries: list[HotspotSummary] = []
    for name, grouped_records in grouped.items():
        durations = [record.duration_ms for record in grouped_records]
        last_tick = max((record.tick for record in grouped_records if record.tick is not None), default=None)
        total_ms = sum(durations)
        calls = len(grouped_records)
        summaries.append(
            HotspotSummary(
                name=name,
                calls=calls,
                total_ms=total_ms,
                avg_ms=(total_ms / calls) if calls else 0.0,
                p95_ms=_percentile(durations, 0.95),
                max_ms=max(durations) if durations else 0.0,
                last_tick=last_tick,
            )
        )
    summaries.sort(key=lambda item: (-item.total_ms, -item.max_ms, item.name))
    return summaries


def aggregate_duration_by_tick(records: list[HotspotRecord]) -> list[TickAggregate]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record.tick is None:
            continue
        grouped[record.tick].append(record.duration_ms)

    rows: list[TickAggregate] = []
    for tick in sorted(grouped.keys()):
        durations = grouped[tick]
        rows.append(
            TickAggregate(
                tick=tick,
                total_ms=sum(durations),
                max_ms=max(durations),
                calls=len(durations),
            )
        )
    return rows


def format_record_context(record: HotspotRecord) -> str:
    preferred_keys = (
        "ship_ids",
        "fit_key",
        "batch_size",
        "resolve_cache",
        "ships",
        "targets",
        "agents",
        "commanders",
        "command_sources",
        "projected_sources",
        "slice_index",
        "slice_dt",
        "dt",
    )
    parts: list[str] = []
    for key in preferred_keys:
        value = record.fields.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    if parts:
        return " ".join(parts)
    fallback_parts: list[str] = []
    for key in sorted(record.fields.keys()):
        if key in {"event", "name", "duration_ms", "tick"}:
            continue
        value = record.fields[key]
        if not value:
            continue
        fallback_parts.append(f"{key}={value}")
        if len(fallback_parts) >= 4:
            break
    return " ".join(fallback_parts)