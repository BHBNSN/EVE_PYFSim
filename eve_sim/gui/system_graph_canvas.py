from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QCoreApplication
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SystemGraphCanvas(QWidget):
    _MARGIN_PX = 42.0
    _NODE_RADIUS_PX = 14.0
    _BG_COLOR = QColor(10, 14, 20)
    _EDGE_COLOR = QColor(115, 136, 170, 160)
    _NODE_FILL = QColor(42, 84, 122)
    _NODE_BORDER = QColor(132, 198, 255)
    _NODE_CURRENT = QColor(255, 230, 90)
    _NODE_TEXT = QColor(222, 230, 242)

    def __init__(
        self,
        map_definition_getter: Callable[[], object | None],
        current_system_getter: Callable[[], str],
        jump_to_system: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.map_definition_getter = map_definition_getter
        self.current_system_getter = current_system_getter
        self.jump_to_system = jump_to_system
        self._layout_cache_signature: tuple | None = None
        self._layout_cache_positions: dict[str, QPointF] = {}
        self.setMinimumSize(300, 260)

    def sync_layout(self) -> None:
        self._layout_cache_signature = None
        self.update()

    def _map_definition(self):
        return self.map_definition_getter()

    @staticmethod
    def _system_label(system) -> str:
        return str(getattr(system, "name", "") or getattr(system, "system_id", "") or "")

    def system_edges(self) -> tuple[tuple[str, str], ...]:
        map_definition = self._map_definition()
        if map_definition is None:
            return tuple()
        building_system_by_id: dict[str, str] = {}
        for system in getattr(map_definition, "systems", []):
            system_id = str(getattr(system, "system_id", "") or "")
            for building in getattr(system, "buildings", []):
                building_system_by_id[str(getattr(building, "building_id", "") or "")] = system_id

        edges: set[tuple[str, str]] = set()
        for system in getattr(map_definition, "systems", []):
            source_system_id = str(getattr(system, "system_id", "") or "")
            if not source_system_id:
                continue
            for building in getattr(system, "buildings", []):
                if str(getattr(building, "kind", "") or "").upper() != "STARGATE":
                    continue
                linked_id = str(getattr(building, "linked_building_id", "") or "").strip()
                target_system_id = building_system_by_id.get(linked_id, "")
                if not target_system_id or target_system_id == source_system_id:
                    continue
                edges.add(tuple(sorted((source_system_id, target_system_id))))
        return tuple(sorted(edges))

    def _layout_signature(self) -> tuple:
        map_definition = self._map_definition()
        systems = tuple(
            (str(getattr(system, "system_id", "") or ""), self._system_label(system))
            for system in (getattr(map_definition, "systems", []) if map_definition is not None else [])
        )
        return systems, self.system_edges(), int(self.width()), int(self.height())

    @staticmethod
    def _initial_positions(system_ids: list[str]) -> dict[str, tuple[float, float]]:
        count = len(system_ids)
        if count <= 0:
            return {}
        if count == 1:
            return {system_ids[0]: (0.0, 0.0)}
        positions: dict[str, tuple[float, float]] = {}
        for index, system_id in enumerate(system_ids):
            angle = (-math.pi / 2.0) + (math.tau * index / count)
            positions[system_id] = (math.cos(angle), math.sin(angle))
        return positions

    @classmethod
    def _force_layout(cls, system_ids: list[str], edges: tuple[tuple[str, str], ...]) -> dict[str, tuple[float, float]]:
        count = len(system_ids)
        positions = cls._initial_positions(system_ids)
        if count <= 2:
            return positions

        adjacency = {system_id: set() for system_id in system_ids}
        for a, b in edges:
            if a in adjacency and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)

        area = 4.0
        ideal = math.sqrt(area / max(1, count))
        temperature = 0.35
        for _iteration in range(90):
            displacement = {system_id: [0.0, 0.0] for system_id in system_ids}
            for i, a in enumerate(system_ids):
                ax, ay = positions[a]
                for b in system_ids[i + 1 :]:
                    bx, by = positions[b]
                    dx = ax - bx
                    dy = ay - by
                    dist = max(0.01, math.hypot(dx, dy))
                    force = (ideal * ideal) / dist
                    nx = dx / dist
                    ny = dy / dist
                    displacement[a][0] += nx * force
                    displacement[a][1] += ny * force
                    displacement[b][0] -= nx * force
                    displacement[b][1] -= ny * force

            for a, b in edges:
                if a not in positions or b not in positions:
                    continue
                ax, ay = positions[a]
                bx, by = positions[b]
                dx = ax - bx
                dy = ay - by
                dist = max(0.01, math.hypot(dx, dy))
                force = (dist * dist) / ideal
                nx = dx / dist
                ny = dy / dist
                displacement[a][0] -= nx * force
                displacement[a][1] -= ny * force
                displacement[b][0] += nx * force
                displacement[b][1] += ny * force

            for system_id in system_ids:
                x, y = positions[system_id]
                dx, dy = displacement[system_id]
                dx -= x * 0.04
                dy -= y * 0.04
                delta_len = max(0.01, math.hypot(dx, dy))
                step = min(delta_len, temperature)
                x += (dx / delta_len) * step
                y += (dy / delta_len) * step
                positions[system_id] = (max(-2.0, min(2.0, x)), max(-2.0, min(2.0, y)))
            temperature *= 0.965
        return positions

    def _screen_positions(self) -> dict[str, QPointF]:
        signature = self._layout_signature()
        if signature == self._layout_cache_signature:
            return dict(self._layout_cache_positions)

        map_definition = self._map_definition()
        systems = list(getattr(map_definition, "systems", []) if map_definition is not None else [])
        systems.sort(key=lambda system: (self._system_label(system), str(getattr(system, "system_id", "") or "")))
        system_ids = [str(getattr(system, "system_id", "") or "") for system in systems if str(getattr(system, "system_id", "") or "")]
        raw_positions = self._force_layout(system_ids, self.system_edges())
        if not raw_positions:
            self._layout_cache_signature = signature
            self._layout_cache_positions = {}
            return {}

        min_x = min(x for x, _y in raw_positions.values())
        max_x = max(x for x, _y in raw_positions.values())
        min_y = min(y for _x, y in raw_positions.values())
        max_y = max(y for _x, y in raw_positions.values())
        span_x = max(0.1, max_x - min_x)
        span_y = max(0.1, max_y - min_y)
        usable_w = max(1.0, float(self.width()) - (self._MARGIN_PX * 2.0))
        usable_h = max(1.0, float(self.height()) - (self._MARGIN_PX * 2.0))
        positions: dict[str, QPointF] = {}
        for system_id, (x, y) in raw_positions.items():
            sx = self._MARGIN_PX + ((x - min_x) / span_x) * usable_w
            sy = self._MARGIN_PX + ((y - min_y) / span_y) * usable_h
            positions[system_id] = QPointF(sx, sy)

        self._layout_cache_signature = signature
        self._layout_cache_positions = dict(positions)
        return positions

    def _system_at(self, point: QPointF) -> str | None:
        radius = self._NODE_RADIUS_PX + 8.0
        for system_id, pos in self._screen_positions().items():
            if math.hypot(point.x() - pos.x(), point.y() - pos.y()) <= radius:
                return system_id
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        system_id = self._system_at(event.position())
        if system_id and self.jump_to_system is not None:
            self.jump_to_system(system_id)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), self._BG_COLOR)
            painter.setPen(QPen(QColor(180, 190, 210, 120), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

            map_definition = self._map_definition()
            if map_definition is None:
                painter.setPen(QPen(self._NODE_TEXT, 1))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, QCoreApplication.translate("eve_sim", "No map loaded"))
                return
            systems = list(getattr(map_definition, "systems", []))
            if not systems:
                painter.setPen(QPen(self._NODE_TEXT, 1))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, QCoreApplication.translate("eve_sim", "No systems"))
                return

            positions = self._screen_positions()
            painter.setPen(QPen(self._EDGE_COLOR, 2))
            for a, b in self.system_edges():
                pa = positions.get(a)
                pb = positions.get(b)
                if pa is None or pb is None:
                    continue
                painter.drawLine(pa, pb)

            current_system_id = str(self.current_system_getter() or "")
            system_by_id = {str(getattr(system, "system_id", "") or ""): system for system in systems}
            for system_id, pos in positions.items():
                selected = system_id == current_system_id
                radius = self._NODE_RADIUS_PX + (4.0 if selected else 0.0)
                fill = QColor(self._NODE_FILL)
                fill.setAlpha(230)
                painter.setPen(QPen(self._NODE_CURRENT if selected else self._NODE_BORDER, 3 if selected else 2))
                painter.setBrush(fill)
                painter.drawEllipse(QRectF(pos.x() - radius, pos.y() - radius, radius * 2.0, radius * 2.0))

                label = self._system_label(system_by_id.get(system_id))
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label)
                text_rect = QRectF(pos.x() - text_width / 2.0 - 4.0, pos.y() + radius + 4.0, text_width + 8.0, metrics.height() + 2.0)
                painter.setPen(QPen(self._NODE_TEXT, 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
        finally:
            if painter.isActive():
                painter.end()


__all__ = ["SystemGraphCanvas"]
