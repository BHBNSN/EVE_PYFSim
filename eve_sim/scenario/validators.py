from __future__ import annotations

from typing import Any


def validate_scenario(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    scenario_id = str(data.get("scenario_id", "") or "")
    if not scenario_id:
        errors.append("scenario_id is required")
    if not str(data.get("name", "") or ""):
        errors.append("name is required")
    try:
        duration = float(data.get("duration_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0.0:
        errors.append("duration_s must be positive")
    fleets = data.get("fleets")
    if not isinstance(fleets, list) or not fleets:
        errors.append("fleets must be a non-empty list")
        return errors
    for fleet_index, fleet in enumerate(fleets):
        if not isinstance(fleet, dict):
            errors.append(f"fleets[{fleet_index}] must be an object")
            continue
        team = str(fleet.get("team", "") or "")
        if team not in {"BLUE", "RED"}:
            errors.append(f"fleets[{fleet_index}].team must be BLUE or RED")
        ships = fleet.get("ships")
        if not isinstance(ships, list) or not ships:
            errors.append(f"fleets[{fleet_index}].ships must be a non-empty list")
            continue
        for ship_index, ship in enumerate(ships):
            if not isinstance(ship, dict):
                errors.append(f"fleets[{fleet_index}].ships[{ship_index}] must be an object")
                continue
            for field in ("ship_id", "squad_id", "fit"):
                if not str(ship.get(field, "") or ""):
                    errors.append(f"fleets[{fleet_index}].ships[{ship_index}].{field} is required")
            try:
                count = int(ship.get("count", 1) or 1)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                errors.append(f"fleets[{fleet_index}].ships[{ship_index}].count must be positive")
    return errors
