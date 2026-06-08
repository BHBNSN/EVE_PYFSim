from __future__ import annotations

from ..spatial_index import SpatialHash2D
from ..world import WorldState



class PerceptionSystem:
    def __init__(self, sensor_range: float = 250_000.0) -> None:
        self.sensor_range = sensor_range
        self._spatial_index = SpatialHash2D(cell_size=sensor_range)

    @staticmethod
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    @staticmethod
    def _ship_is_gate_cloaked(ship, now: float | None = None) -> bool:
        cloak = getattr(getattr(ship, "nav", None), "cloak", None)
        if cloak is None or not bool(getattr(cloak, "active", False)):
            return False
        if now is not None and float(getattr(cloak, "expires_at", 0.0) or 0.0) <= float(now):
            cloak.active = False
            cloak.expires_at = 0.0
            cloak.source = ""
            return False
        return True

    @staticmethod
    def _ship_system_id(ship) -> str:
        nav = getattr(ship, "nav", None)
        if nav is None:
            return ""
        system_id = getattr(nav, "system_id", "")
        return system_id if isinstance(system_id, str) else str(system_id or "")

    @staticmethod
    def _distance_sq(a, b) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)
        return (dx * dx) + (dy * dy)

    @classmethod
    def _bbox_diagonal_sq(cls, ships: list) -> float:
        if not ships:
            return 0.0
        min_x = max_x = float(ships[0].nav.position.x)
        min_y = max_y = float(ships[0].nav.position.y)
        for ship in ships[1:]:
            pos = ship.nav.position
            pos_x = float(pos.x)
            pos_y = float(pos.y)
            if pos_x < min_x:
                min_x = pos_x
            if pos_x > max_x:
                max_x = pos_x
            if pos_y < min_y:
                min_y = pos_y
            if pos_y > max_y:
                max_y = pos_y
        dx = max_x - min_x
        dy = max_y - min_y
        return (dx * dx) + (dy * dy)

    def run(self, world: WorldState) -> None:
        now = float(world.now)
        alive = [
            s
            for s in world.ships.values()
            if s.vital.alive and not self._ship_in_warp(s) and not self._ship_is_gate_cloaked(s, now)
        ]
        alive_ids = {ship.ship_id for ship in alive}
        for ship in world.ships.values():
            if ship.ship_id not in alive_ids:
                ship.perception = []
                ship.perception_allies = []
                ship.perception_enemies = []
                ship.perception_split_ready = True
        if not alive:
            self._spatial_index.clear()
            return
        sensor = float(self.sensor_range)
        sensor_sq = sensor * sensor
        alive_by_system: dict[str, list] = {}
        for ship in alive:
            alive_by_system.setdefault(self._ship_system_id(ship), []).append(ship)

        for system_alive in alive_by_system.values():
            if len(system_alive) <= 24:
                for source in system_alive:
                    allies: list[str] = []
                    enemies: list[str] = []
                    for target in system_alive:
                        if target.ship_id == source.ship_id:
                            continue
                        if self._distance_sq(source.nav.position, target.nav.position) > sensor_sq:
                            continue
                        if target.team == source.team:
                            allies.append(target.ship_id)
                        else:
                            enemies.append(target.ship_id)
                    source.perception_allies = allies
                    source.perception_enemies = enemies
                    source.perception = allies + enemies
                    source.perception_split_ready = True
                continue

            team_members: dict[object, list] = {}
            for ship in system_alive:
                team_members.setdefault(ship.team, []).append(ship)
            team_member_ids: dict[object, tuple[str, ...]] = {
                team: tuple(member.ship_id for member in members)
                for team, members in team_members.items()
            }
            enemy_ids_by_team: dict[object, tuple[str, ...]] = {
                team: tuple(
                    member.ship_id
                    for other_team, members in team_members.items()
                    if other_team != team
                    for member in members
                )
                for team in team_members
            }

            if self._bbox_diagonal_sq(system_alive) <= sensor_sq:
                for ship in system_alive:
                    allies = [target_id for target_id in team_member_ids.get(ship.team, ()) if target_id != ship.ship_id]
                    enemies = list(enemy_ids_by_team.get(ship.team, ()))
                    ship.perception_allies = allies
                    ship.perception_enemies = enemies
                    ship.perception = allies + enemies
                    ship.perception_split_ready = True
                continue

            compact_team_visibility: dict[object, bool] = {
                team: self._bbox_diagonal_sq(members) <= sensor_sq
                for team, members in team_members.items()
            }

            self._spatial_index.rebuild({ship.ship_id: ship.nav.position for ship in system_alive})
            for source in system_alive:
                if compact_team_visibility.get(source.team, False):
                    allies = [target_id for target_id in team_member_ids.get(source.team, ()) if target_id != source.ship_id]
                    visible = list(allies)
                    seen = set(allies)
                else:
                    allies = []
                    visible = []
                    seen = set()
                enemies: list[str] = []
                for target_id in self._spatial_index.query_radius(source.nav.position, sensor):
                    if target_id == source.ship_id or target_id in seen:
                        continue
                    target = world.ships.get(target_id)
                    if target is None or not target.vital.alive:
                        continue
                    if self._distance_sq(source.nav.position, target.nav.position) > sensor_sq:
                        continue
                    if target.team == source.team:
                        allies.append(target_id)
                    else:
                        enemies.append(target_id)
                    visible.append(target_id)
                    seen.add(target_id)
                source.perception_allies = allies
                source.perception_enemies = enemies
                source.perception = visible
                source.perception_split_ready = True

