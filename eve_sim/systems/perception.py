from __future__ import annotations

import math

import numpy as np

from ..world import WorldState



class PerceptionSystem:
    def __init__(self, sensor_range: float = 250_000.0) -> None:
        self.sensor_range = sensor_range

    @staticmethod
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    def run(self, world: WorldState) -> None:
        alive = [s for s in world.ships.values() if s.vital.alive and not self._ship_in_warp(s)]
        alive_ids = {ship.ship_id for ship in alive}
        for ship in world.ships.values():
            if ship.ship_id not in alive_ids:
                ship.perception = []
        if not alive:
            return
        if len(alive) <= 24:
            sensor = self.sensor_range
            for source in alive:
                source.perception = [
                    target.ship_id
                    for target in alive
                    if target.ship_id != source.ship_id
                    and source.nav.position.distance_to(target.nav.position) <= sensor
                ]
            return

        min_x = max_x = float(alive[0].nav.position.x)
        min_y = max_y = float(alive[0].nav.position.y)
        for ship in alive[1:]:
            pos = ship.nav.position
            min_x = min(min_x, float(pos.x))
            max_x = max(max_x, float(pos.x))
            min_y = min(min_y, float(pos.y))
            max_y = max(max_y, float(pos.y))
        if math.hypot(max_x - min_x, max_y - min_y) <= self.sensor_range:
            alive_ids = [ship.ship_id for ship in alive]
            for index, ship in enumerate(alive):
                ship.perception = alive_ids[:index] + alive_ids[index + 1:]
            return

        pos_array = np.array([(s.nav.position.x, s.nav.position.y) for s in alive], dtype=np.float64)
        delta = pos_array[:, None, :] - pos_array[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        for i, ship in enumerate(alive):
            mask = (dist[i] <= self.sensor_range) & (dist[i] > 0)
            ship.perception = [alive[j].ship_id for j in np.where(mask)[0].tolist()]

