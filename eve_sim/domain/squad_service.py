from __future__ import annotations

from dataclasses import dataclass

from ..models import ShipEntity, SquadLeaderLocation
from ..squad_identity import squad_key
from ..world import WorldState
from .events import DomainEvent


@dataclass(frozen=True, slots=True)
class LeadershipChangeSet:
    events: tuple[DomainEvent, ...] = ()


class SquadLeadershipService:
    """The sole authority for deterministic, global squad-leader election."""

    @staticmethod
    def _replacement_leader(
        candidates: list[ShipEntity],
        previous_ship: ShipEntity | None,
        previous_location: SquadLeaderLocation | None,
    ) -> ShipEntity | None:
        if not candidates:
            return None
        previous_system = str(
            previous_location.system_id
            if previous_location is not None
            else getattr(getattr(previous_ship, "nav", None), "system_id", "") or ""
        )
        previous_group = str(getattr(previous_ship, "ship_group_id", "") or "")

        def priority(ship: ShipEntity) -> tuple[int, int, str]:
            same_system = bool(previous_system) and str(ship.nav.system_id or "") == previous_system
            same_group = bool(previous_group) and str(ship.ship_group_id or "") == previous_group
            locality = 0 if same_system and same_group else 1 if same_system else 2 if same_group else 3
            return locality, int(getattr(ship, "command_priority", 0) or 0), str(ship.ship_id)

        return min(candidates, key=priority)

    def refresh(self, world: WorldState) -> LeadershipChangeSet:
        members_by_key: dict[str, list[ShipEntity]] = {}
        for ship in world.ships.values():
            if ship.vital.alive:
                members_by_key.setdefault(squad_key(ship.team, ship.squad_id), []).append(ship)

        events: list[DomainEvent] = []
        active_keys = set(members_by_key)
        for scoped_state in (
            world.squad_leaders,
            world.squad_leader_locations,
            world.squad_leader_location_versions,
            world.squad_propulsion_commands,
            world.squad_leader_speed_limits,
            world.squad_focus_queues,
            world.squad_focus_updated_at,
            world.intents,
        ):
            active_keys.update(str(key) for key in scoped_state)
        for key in sorted(active_keys):
            candidates = members_by_key.get(key, [])
            mapped_id = str(world.squad_leaders.get(key, "") or "")
            mapped_ship = world.ships.get(mapped_id)
            previous_location = world.squad_leader_locations.get(key)
            if mapped_ship is not None and mapped_ship.vital.alive and any(ship.ship_id == mapped_ship.ship_id for ship in candidates):
                leader = mapped_ship
            else:
                leader = self._replacement_leader(candidates, mapped_ship, previous_location)

            if leader is None:
                if mapped_id:
                    events.append(DomainEvent("squad_leader_cleared", {"squad_key": key, "leader_id": mapped_id}))
                world.squad_leaders.pop(key, None)
                world.squad_leader_locations.pop(key, None)
                world.squad_leader_location_versions.pop(key, None)
                world.squad_propulsion_commands.pop(key, None)
                world.squad_focus_queues.pop(key, None)
                world.squad_focus_updated_at.pop(key, None)
                world.squad_leader_speed_limits.pop(key, None)
                world.intents.pop(key, None)
                continue

            leader_id = str(leader.ship_id)
            leader_system = str(leader.nav.system_id or "")
            old_id = previous_location.leader_id if previous_location is not None else mapped_id
            old_system = previous_location.system_id if previous_location is not None else leader_system
            version = int(world.squad_leader_location_versions.get(key, 0) or 0)
            changed = previous_location is not None and (old_id != leader_id or old_system != leader_system)
            if changed:
                version += 1
            if previous_location is not None and old_system != leader_system:
                world.squad_focus_queues.pop(key, None)
                world.squad_focus_updated_at.pop(key, None)

            world.squad_leaders[key] = leader_id
            world.squad_leader_location_versions[key] = version
            world.squad_leader_locations[key] = SquadLeaderLocation(
                leader_id=leader_id,
                system_id=leader_system,
                location_version=version,
            )
            if old_id != leader_id or old_system != leader_system:
                events.append(
                    DomainEvent(
                        "squad_leader_changed",
                        {
                            "squad_key": key,
                            "leader_id": leader_id,
                            "system_id": leader_system,
                            "location_version": version,
                        },
                    )
                )
        return LeadershipChangeSet(tuple(events))
