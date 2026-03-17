from __future__ import annotations

from dataclasses import dataclass, field

from .models import Beacon, FleetIntent, ShipEntity, Team


@dataclass(slots=True)
class WorldState:
    now: float = 0.0
    tick: int = 0
    ships: dict[str, ShipEntity] = field(default_factory=dict)
    beacons: dict[str, Beacon] = field(default_factory=dict)
    intents: dict[str, FleetIntent] = field(default_factory=dict)
    squad_leaders: dict[str, str] = field(default_factory=dict)
    squad_propulsion_commands: dict[str, bool] = field(default_factory=dict)
    squad_leader_speed_limits: dict[str, float] = field(default_factory=dict)
    squad_focus_queues: dict[str, list[str]] = field(default_factory=dict)
    squad_prelocked_targets: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    squad_prelock_timers: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def by_team(self, team: Team) -> list[ShipEntity]:
        return [s for s in self.ships.values() if s.team == team and s.vital.alive]

    def enemies_of(self, team: Team) -> list[ShipEntity]:
        return [s for s in self.ships.values() if s.team != team and s.vital.alive]
