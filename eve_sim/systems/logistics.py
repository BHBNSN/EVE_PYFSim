from __future__ import annotations

from typing import Any

from ..models import Team
from ..world import WorldState


class LogisticsSystem:
    @staticmethod
    def _apply_repair(target, amount: float) -> None:
        remaining = max(0.0, float(amount))
        if remaining <= 0.0:
            return
        missing_shield = max(0.0, float(target.vital.shield_max) - float(target.vital.shield))
        if missing_shield > 0.0:
            restored = min(remaining, missing_shield)
            target.vital.shield += restored
            remaining -= restored
        if remaining <= 0.0:
            return
        missing_armor = max(0.0, float(target.vital.armor_max) - float(target.vital.armor))
        if missing_armor > 0.0:
            target.vital.armor += min(remaining, missing_armor)

    def run(self, world: WorldState, dt: float) -> None:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        for ship in world.ships.values():
            if ship.vital.alive:
                alive_by_team[ship.team].append(ship)

        weakest_by_team: dict[Team, Any | None] = {Team.BLUE: None, Team.RED: None}
        for team, members in alive_by_team.items():
            if not members:
                weakest_by_team[team] = None
                continue
            weakest_by_team[team] = min(members, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            if ship.runtime is not None:
                # Runtime-backed fits already apply remote repair effects through CombatSystem.
                continue
            if ship.profile.rep_amount <= 0 or ship.profile.rep_cycle <= 0:
                continue

            target = weakest_by_team.get(ship.team)
            if target is None or target.ship_id == ship.ship_id:
                allies = [a for a in alive_by_team.get(ship.team, []) if a.ship_id != ship.ship_id]
                if not allies:
                    continue
                target = min(allies, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

            dist = ship.nav.position.distance_to(target.nav.position)
            if dist > ship.profile.max_target_range:
                continue

            repair = ship.profile.rep_amount * (dt / ship.profile.rep_cycle)
            self._apply_repair(target, repair)
