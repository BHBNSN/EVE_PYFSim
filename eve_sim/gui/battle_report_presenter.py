from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def format_battle_report(report: dict[str, Any], translate: Callable[[str], str]) -> str:
    """Format a battle report independently of Qt widgets and dialogs."""
    lines = [
        translate("Battle Report"),
        f"{translate('Scenario')}: {report.get('scenario_id', '')}",
        f"{translate('Duration')}: {float(report.get('duration_s', 0.0) or 0.0):.2f}s",
        "",
        f"{translate('Total Damage By Team')}:",
    ]
    for team, value in sorted((report.get("total_damage_by_team", {}) or {}).items()):
        lines.append(f"  {team}: {float(value):.1f}")
    lines.extend(["", f"{translate('Rep Applied By Team')}:"])
    for team, value in sorted((report.get("rep_applied_by_team", {}) or {}).items()):
        lines.append(f"  {team}: {float(value):.1f}")
    lines.extend(["", f"{translate('Jam Uptime By Target')}:"])
    for target, value in sorted((report.get("jam_uptime_by_target", {}) or {}).items()):
        lines.append(f"  {target}: {float(value):.1f}s")
    lines.extend(["", f"{translate('Burst Coverage By Effect')}:"])
    for effect, value in sorted((report.get("burst_coverage_by_effect", {}) or {}).items()):
        lines.append(f"  {effect}: {float(value):.1f}s")
    lines.extend(["", f"{translate('Ship Deaths')}:"])
    deaths = report.get("ship_deaths", []) or []
    if not deaths:
        lines.append(f"  {translate('none')}")
    for death in deaths:
        lines.append(
            f"  t={float(death.get('at', 0.0) or 0.0):.2f}s tick={int(death.get('tick', 0) or 0)} "
            f"ship={death.get('ship_id', '')} source={death.get('source_id', '')}"
        )
    lines.extend(["", f"{translate('Raw JSON')}:", json.dumps(report, ensure_ascii=False, indent=2)])
    return "\n".join(lines)


__all__ = ["format_battle_report"]
