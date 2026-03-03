from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class Vector2:
    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, value: float) -> "Vector2":
        return Vector2(self.x * value, self.y * value)

    __rmul__ = __mul__

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vector2":
        n = self.length()
        if n == 0:
            return Vector2(0.0, 0.0)
        return Vector2(self.x / n, self.y / n)

    def distance_to(self, other: "Vector2") -> float:
        return (self - other).length()

    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y, self.x))


ZERO = Vector2(0.0, 0.0)
