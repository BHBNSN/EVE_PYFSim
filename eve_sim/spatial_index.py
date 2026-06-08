from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

from .math2d import Vector2


@dataclass(slots=True)
class SpatialHash2D:
    cell_size: float
    _cells: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    _positions: dict[str, Vector2] = field(default_factory=dict)

    def clear(self) -> None:
        self._cells.clear()
        self._positions.clear()

    def _cell_key(self, position: Vector2) -> tuple[int, int]:
        size = max(1.0, float(self.cell_size))
        return (math.floor(float(position.x) / size), math.floor(float(position.y) / size))

    def rebuild(self, positions: dict[str, Vector2]) -> None:
        buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        for item_id, position in positions.items():
            buckets[self._cell_key(position)].append(item_id)
        self._cells = dict(buckets)
        self._positions = dict(positions)

    def query_radius(self, center: Vector2, radius: float) -> list[str]:
        if not self._positions:
            return []
        size = max(1.0, float(self.cell_size))
        reach = max(0, int(math.ceil(max(0.0, float(radius)) / size)))
        cell_x, cell_y = self._cell_key(center)
        found: list[str] = []
        for offset_x in range(-reach, reach + 1):
            for offset_y in range(-reach, reach + 1):
                found.extend(self._cells.get((cell_x + offset_x, cell_y + offset_y), ()))
        return found


__all__ = ["SpatialHash2D"]
