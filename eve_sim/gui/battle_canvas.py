from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Callable

from PySide6.QtCore import QPoint, QRectF, Qt, QCoreApplication
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QWidget

from ..config import UiConfig
from ..fleet_setup import get_ship_icon_key, get_type_display_name
from ..fit_runtime import EffectClass, ModuleRuntime, ModuleState
from ..math2d import Vector2
from ..models import ShipEntity, Team
from ..squad_identity import squad_key
from .adapters.runtime_view import WorldViewSource
from .models import AreaCycleOverlay
class BattleCanvas(QWidget):
    _SHIP_ICON_SIZE_PX = 25
    _DEPLOYABLE_ICON_SIZE_PX = 16
    _STRUCTURE_ICON_SIZE_PX = 18
    _SCREEN_DRAW_MARGIN_PX = 256.0
    _SHIP_ICON_SOURCE_CACHE: dict[str, QImage] | None = None
    _SHIP_ICON_PIXMAP_CACHE: dict[tuple[str, int, int, int, int, int], QPixmap] = {}
    _DEPLOYABLE_ICON_SOURCE_CACHE: dict[str, QImage] | None = None
    _DEPLOYABLE_ICON_PIXMAP_CACHE: dict[tuple[str, int, int, int, int, int], QPixmap] = {}
    _MIN_ZOOM = 0.3
    _MAX_ZOOM = 100.0
    _ZOOM_WORLD_SCALE_BASE = 0.003
    _TEAM_BLUE_COLOR = QColor(80, 180, 255)
    _TEAM_RED_COLOR = QColor(255, 92, 92)
    _SELECTED_SQUAD_COLOR = QColor(186, 102, 255)
    _GATE_CLOAK_COLOR = QColor(160, 242, 176)
    _DESTROYED_SHIP_COLOR = QColor(130, 130, 130)
    _SELECTION_HIGHLIGHT_COLOR = QColor(255, 230, 90)
    _STRUCTURE_COLOR = QColor(188, 198, 212)
    _STARGATE_COLOR = QColor(255, 194, 92)

    def __init__(
        self,
        runtime_view: WorldViewSource,
        ui_cfg: UiConfig,
        on_issue_move: Callable[[str, Vector2], None],
        on_issue_approach: Callable[[str, str], None],
        on_issue_warp_ship: Callable[[str, str], None],
        on_issue_warp_beacon: Callable[[str, str], None],
        on_issue_focus: Callable[[str], None],
        on_issue_prefocus: Callable[[str], None],
        on_cancel_prefocus: Callable[[str], None],
        on_show_ship_context_menu: Callable[[str, QPoint], None],
        on_induce_squad_spawn: Callable[[str, Vector2], None],
        on_induce_fleet_spawn: Callable[[Vector2], None],
        controlled_squads_getter: Callable[[], list[str]],
        ship_visible_getter: Callable[[str], bool],
        squad_guidance_target_getter: Callable[[str], Vector2 | None],
        on_show_status: Callable[[str], None],
        language_getter: Callable[[], str],
        controlled_team_getter: Callable[[], Team],
        on_select_squad: Callable[[str], None],
        on_select_enemy: Callable[[str], None],
        on_show_structure_context_menu: Callable[[str, QPoint], None] | None = None,
        squad_drone_types_getter: Callable[[str], list[str]] | None = None,
        squad_fighter_types_getter: Callable[[str], list[str]] | None = None,
        on_launch_squad_drones: Callable[[str, str], None] | None = None,
        on_launch_squad_fighters: Callable[[str, str], None] | None = None,
        on_recall_squad_deployables: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.runtime_view = runtime_view
        self.ui_cfg = ui_cfg
        self.on_issue_move = on_issue_move
        self.on_issue_approach = on_issue_approach
        self.on_issue_warp_ship = on_issue_warp_ship
        self.on_issue_warp_beacon = on_issue_warp_beacon
        self.on_issue_focus = on_issue_focus
        self.on_issue_prefocus = on_issue_prefocus
        self.on_cancel_prefocus = on_cancel_prefocus
        self.on_show_ship_context_menu = on_show_ship_context_menu
        self.on_show_structure_context_menu = on_show_structure_context_menu or (lambda *_args: None)
        self.on_induce_squad_spawn = on_induce_squad_spawn
        self.on_induce_fleet_spawn = on_induce_fleet_spawn
        self.controlled_squads_getter = controlled_squads_getter
        self.ship_visible_getter = ship_visible_getter
        self.squad_guidance_target_getter = squad_guidance_target_getter
        self.on_show_status = on_show_status
        self.language_getter = language_getter
        self.controlled_team_getter = controlled_team_getter
        self.on_select_squad = on_select_squad
        self.on_select_enemy = on_select_enemy
        self.squad_drone_types_getter = squad_drone_types_getter or (lambda _squad: [])
        self.squad_fighter_types_getter = squad_fighter_types_getter or (lambda _squad: [])
        self.on_launch_squad_drones = on_launch_squad_drones or (lambda _squad, _type_name: None)
        self.on_launch_squad_fighters = on_launch_squad_fighters or (lambda _squad, _type_name: None)
        self.on_recall_squad_deployables = on_recall_squad_deployables or (lambda _squad: None)
        self.setMinimumSize(ui_cfg.width, ui_cfg.height)

        self.zoom = self._clamp_zoom(ui_cfg.world_to_screen_scale)
        self.pan_world = Vector2(0.0, 0.0)
        self.selected_squad = "BLUE-ALPHA"
        self.selected_enemy_target: str | None = None
        self.selected_ship_id: str | None = None
        self.selected_structure_id: str | None = None
        self.current_view_system_id: str = ""
        self.highlighted_roster_ship_ids: set[str] = set()

        self.pan_active = False
        self.pan_start: QPoint | None = None
        self.pan_start_world = Vector2(0.0, 0.0)
        self._bg_cache: QPixmap | None = None
        self._bg_cache_w = 0
        self._bg_cache_h = 0
        self._area_cycle_overlays: dict[tuple[str, str], AreaCycleOverlay] = {}
        now = time.perf_counter()
        self._authoritative_frame_wall_time = now
        self._render_frame_wall_time = now
        self._render_frame_dt_s = 0.0
        self._render_frame_index = 0
        self._visual_ship_positions: dict[str, Vector2] = {}
        self._visual_ship_frame_ids: dict[str, int] = {}
        self._visual_ship_system_ids: dict[str, str] = {}
        self._rendering_visual_frame = False

    def _ui_text(self, text: str) -> str:
        if not str(self.language_getter() or "").lower().startswith("zh"):
            return QCoreApplication.translate("eve_sim", text)
        translations = {
            "Launch Squad Drones": "释放小队无人机",
            "Launch Squad Fighters": "释放小队舰载机",
            "Recall Squad Drones/Fighters": "收回小队无人机/舰载机",
        }
        return translations.get(text, QCoreApplication.translate("eve_sim", text))

    def _display_type_name(self, type_name: str) -> str:
        return get_type_display_name(str(type_name or ""), language=self.language_getter())

    @classmethod
    def _ship_icon_dir(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "res" / "icon"

    @classmethod
    def _ensure_ship_icon_sources(cls) -> None:
        if cls._SHIP_ICON_SOURCE_CACHE is not None:
            return
        sources: dict[str, QImage] = {}
        icon_dir = cls._ship_icon_dir()
        if icon_dir.exists():
            for path in sorted(icon_dir.glob("*_64.png")):
                image = QImage(str(path))
                if image.isNull():
                    continue
                stem = path.stem
                icon_key = stem[:-3] if stem.endswith("_64") else stem
                sources[str(icon_key)] = image
        cls._SHIP_ICON_SOURCE_CACHE = sources

    @classmethod
    def _ship_icon_pixmap(cls, icon_key: str, color: QColor, size_px: int) -> QPixmap | None:
        cls._ensure_ship_icon_sources()
        source_cache = cls._SHIP_ICON_SOURCE_CACHE or {}
        source = source_cache.get(str(icon_key))
        if source is None:
            return None
        cache_key = (
            str(icon_key),
            int(size_px),
            int(color.red()),
            int(color.green()),
            int(color.blue()),
            int(color.alpha()),
        )
        cached = cls._SHIP_ICON_PIXMAP_CACHE.get(cache_key)
        if cached is not None:
            return cached

        scaled = source.scaled(
            int(size_px),
            int(size_px),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tinted = QImage(scaled.size(), QImage.Format.Format_ARGB32_Premultiplied)
        tinted.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted)
        try:
            painter.drawImage(0, 0, scaled)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(tinted.rect(), color)
        finally:
            if painter.isActive():
                painter.end()

        pixmap = QPixmap.fromImage(tinted)
        cls._SHIP_ICON_PIXMAP_CACHE[cache_key] = pixmap
        return pixmap

    @classmethod
    def _ship_icon_for_name(cls, ship_name: str, color: QColor, *, size_px: int | None = None) -> QPixmap | None:
        icon_key = get_ship_icon_key(ship_name)
        if not icon_key:
            return None
        return cls._ship_icon_pixmap(icon_key, color, size_px or cls._SHIP_ICON_SIZE_PX)

    @classmethod
    def _ensure_deployable_icon_sources(cls) -> None:
        if cls._DEPLOYABLE_ICON_SOURCE_CACHE is not None:
            return
        sources: dict[str, QImage] = {}
        icon_dir = cls._ship_icon_dir()
        if icon_dir.exists():
            for path in sorted(icon_dir.glob("*_16.png")):
                image = QImage(str(path))
                if image.isNull():
                    continue
                stem = path.stem
                icon_key = stem[:-3] if stem.endswith("_16") else stem
                sources[str(icon_key)] = image
        cls._DEPLOYABLE_ICON_SOURCE_CACHE = sources

    @classmethod
    def _deployable_icon_pixmap(cls, icon_key: str, color: QColor, size_px: int | None = None) -> QPixmap | None:
        cls._ensure_deployable_icon_sources()
        source = (cls._DEPLOYABLE_ICON_SOURCE_CACHE or {}).get(str(icon_key))
        if source is None:
            return None
        size = int(size_px or cls._DEPLOYABLE_ICON_SIZE_PX)
        cache_key = (
            str(icon_key),
            size,
            int(color.red()),
            int(color.green()),
            int(color.blue()),
            int(color.alpha()),
        )
        cached = cls._DEPLOYABLE_ICON_PIXMAP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        scaled = source.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tinted = QImage(scaled.size(), QImage.Format.Format_ARGB32_Premultiplied)
        tinted.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted)
        try:
            painter.drawImage(0, 0, scaled)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(tinted.rect(), color)
        finally:
            if painter.isActive():
                painter.end()
        pixmap = QPixmap.fromImage(tinted)
        cls._DEPLOYABLE_ICON_PIXMAP_CACHE[cache_key] = pixmap
        return pixmap

    @staticmethod
    def _deployable_icon_key(entity) -> str:
        definition = getattr(entity, "definition", None)
        role = str(getattr(getattr(entity, "fit", None), "role", "") or "").upper()
        group = str(getattr(definition, "group_name", "") or "").lower()
        type_name = str(getattr(definition, "type_name", "") or "").lower()
        if role.startswith("FIGHTER"):
            slot = str(getattr(definition, "slot_kind", "") or "").lower()
            if slot == "heavy":
                return "fightersquadh"
            if slot == "support":
                return "fightersquadm"
            return "fightersquad"
        if bool(getattr(definition, "is_sentry", False)):
            return "dronesentry"
        if "logistic" in group or "logistic" in type_name:
            return "dronelogistics"
        ewar = getattr(definition, "ewar", None)
        if "electronic" in group or "ewar" in group or bool(getattr(ewar, "has_effect", False)):
            return "droneew"
        if "fighter" in group or "fighter" in type_name:
            return "dronefighter"
        if "heavy" in group or "heavy" in type_name:
            return "droneheavyattack"
        if "medium" in group or "medium" in type_name:
            return "dronemediumscout"
        if "light" in group or "light" in type_name:
            return "dronelightscout"
        if "mining" in group or "mining" in type_name:
            return "dronemining"
        return "droneattack"

    @classmethod
    def _deployable_draw_color(cls, entity, controlled_team: Team, selected_squad: str) -> QColor:
        if not entity.vital.alive:
            return QColor(cls._DESTROYED_SHIP_COLOR)
        if entity.team == controlled_team and entity.squad_id == selected_squad:
            return QColor(cls._SELECTED_SQUAD_COLOR)
        if entity.team == Team.BLUE:
            return QColor(cls._TEAM_BLUE_COLOR)
        return QColor(cls._TEAM_RED_COLOR)

    @classmethod
    def _ship_draw_color(cls, ship: ShipEntity, controlled_team: Team, selected_squad: str) -> QColor:
        if not ship.vital.alive:
            return QColor(cls._DESTROYED_SHIP_COLOR)
        cloak = getattr(getattr(ship, "nav", None), "cloak", None)
        if cloak is not None and bool(getattr(cloak, "active", False)):
            return QColor(cls._GATE_CLOAK_COLOR)
        if ship.team == controlled_team and ship.squad_id == selected_squad:
            return QColor(cls._SELECTED_SQUAD_COLOR)
        if ship.team == Team.BLUE:
            return QColor(cls._TEAM_BLUE_COLOR)
        return QColor(cls._TEAM_RED_COLOR)

    def _ship_selection_highlight_level(self, ship: ShipEntity) -> int:
        if not ship.vital.alive:
            return 0
        layers = 0
        if ship.ship_id in self.highlighted_roster_ship_ids:
            layers += 1
        if self.selected_ship_id and ship.ship_id == self.selected_ship_id:
            layers += 1
        if self.selected_enemy_target and ship.ship_id == self.selected_enemy_target:
            layers += 2
        return layers

    def _ship_selection_highlight_size_px(self, ship: ShipEntity, base_size_px: int) -> int | None:
        level = self._ship_selection_highlight_level(ship)
        if level <= 0:
            return None
        return int(base_size_px + 2 + (level * 2))

    def _ship_draw_priority(self, ship: ShipEntity) -> int:
        return self._ship_selection_highlight_level(ship)

    def _ensure_bg_cache(self) -> None:
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return
        if self._bg_cache is not None and self._bg_cache_w == width and self._bg_cache_h == height:
            return
        bg = QPixmap(width, height)
        painter = QPainter(bg)
        try:
            painter.fillRect(0, 0, width, height, QColor(15, 18, 24))
            pen_grid = QPen(QColor(40, 44, 52), 1)
            painter.setPen(pen_grid)
            for i in range(0, width, 50):
                painter.drawLine(i, 0, i, height)
            for j in range(0, height, 50):
                painter.drawLine(0, j, width, j)
        finally:
            if painter.isActive():
                painter.end()
        self._bg_cache = bg
        self._bg_cache_w = width
        self._bg_cache_h = height

    def resizeEvent(self, event) -> None:
        self._bg_cache = None
        self._bg_cache_w = 0
        self._bg_cache_h = 0
        super().resizeEvent(event)

    def _matches_current_view_system(self, system_id: str | None) -> bool:
        current = str(self.current_view_system_id or "").strip()
        target = str(system_id or "").strip()
        return not current or (bool(target) and current == target)

    def _current_system_definition(self):
        map_definition = getattr(self.runtime_view.world, "map_definition", None)
        current = str(self.current_view_system_id or "").strip()
        if map_definition is None or not current:
            return None
        try:
            return map_definition.system_by_id(current)
        except Exception:
            return None

    def _current_system_center(self) -> Vector2:
        return Vector2(0.0, 0.0)

    def note_authoritative_frame(self) -> None:
        self._authoritative_frame_wall_time = time.perf_counter()
        self._render_frame_index += 1

    def _begin_render_frame(self) -> None:
        now = time.perf_counter()
        self._render_frame_dt_s = max(0.0, min(0.1, now - float(self._render_frame_wall_time)))
        self._render_frame_wall_time = now
        self._render_frame_index += 1

    def _render_lookahead_s(self) -> float:
        try:
            elapsed_wall = max(0.0, time.perf_counter() - float(self._authoritative_frame_wall_time))
        except Exception:
            return 0.0
        try:
            tidi_factor = max(0.0, min(1.0, float(getattr(self.runtime_view, "tidi_factor", 1.0) or 1.0)))
        except Exception:
            tidi_factor = 1.0
        try:
            max_dt = max(0.0, float(getattr(self.runtime_view, "simulation_dt", 1.0)))
        except Exception:
            max_dt = 1.0
        return max(0.0, min(max_dt, elapsed_wall * tidi_factor))

    def _project_position(self, position: Vector2, velocity: Vector2 | None) -> Vector2:
        dt = self._render_lookahead_s()
        if dt <= 1e-6 or velocity is None:
            return position
        return Vector2(float(position.x) + float(velocity.x) * dt, float(position.y) + float(velocity.y) * dt)

    def _ship_render_position(self, ship: ShipEntity) -> Vector2:
        target = self._project_position(ship.nav.position, ship.nav.velocity)
        if not bool(getattr(self, "_rendering_visual_frame", False)):
            return target
        ship_id = str(ship.ship_id)
        if self._visual_ship_frame_ids.get(ship_id) == self._render_frame_index:
            return self._visual_ship_positions.get(ship_id, target)

        system_id = str(getattr(ship.nav, "system_id", "") or "")
        previous = self._visual_ship_positions.get(ship_id)
        previous_system = self._visual_ship_system_ids.get(ship_id)
        if previous is None or previous_system != system_id:
            visual = target
        else:
            correction = previous.distance_to(target)
            speed = max(0.0, float(ship.nav.velocity.length()))
            snap_distance = max(20_000.0, speed * 8.0)
            if correction >= snap_distance:
                visual = target
            else:
                alpha = 1.0 - math.exp(-max(0.0, self._render_frame_dt_s) / 0.12)
                visual = previous * (1.0 - alpha) + target * alpha

        self._visual_ship_positions[ship_id] = visual
        self._visual_ship_frame_ids[ship_id] = self._render_frame_index
        self._visual_ship_system_ids[ship_id] = system_id
        return visual

    def _projectile_render_position(self, projectile) -> Vector2:
        return self._project_position(projectile.position, getattr(projectile, "velocity", None))

    def _prune_visual_ship_positions(self) -> None:
        live_ship_ids = {str(ship_id) for ship_id in self.runtime_view.world.ships.keys()}
        stale = [ship_id for ship_id in self._visual_ship_positions.keys() if ship_id not in live_ship_ids]
        for ship_id in stale:
            self._visual_ship_positions.pop(ship_id, None)
            self._visual_ship_frame_ids.pop(ship_id, None)
            self._visual_ship_system_ids.pop(ship_id, None)

    def world_scale(self) -> float:
        return max(1e-12, float(self.zoom) * float(self._ZOOM_WORLD_SCALE_BASE))

    def _pick_ship_at(self, p: QPoint, max_px_distance: float = 14.0):
        chosen = None
        chosen_dist = max_px_distance
        for ship in self.runtime_view.world.ships.values():
            if not self.ship_visible_getter(ship.ship_id):
                continue
            if not ship.vital.alive:
                continue
            if not self._matches_current_view_system(getattr(ship.nav, "system_id", "")):
                continue
            sx, sy = self._to_screen(self._ship_render_position(ship))
            dx = sx - p.x()
            dy = sy - p.y()
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= chosen_dist:
                chosen = ship
                chosen_dist = dist
        return chosen

    def _pick_deployable_at(self, p: QPoint, max_px_distance: float = 12.0):
        chosen = None
        chosen_dist = max_px_distance
        for entity in list(self.runtime_view.world.drones.values()) + list(self.runtime_view.world.fighters.values()):
            if not entity.vital.alive:
                continue
            if not self._matches_current_view_system(getattr(entity.nav, "system_id", "")):
                continue
            sx, sy = self._to_screen(self._project_position(entity.nav.position, entity.nav.velocity))
            dx = sx - p.x()
            dy = sy - p.y()
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= chosen_dist:
                chosen = entity
                chosen_dist = dist
        return chosen

    def _pick_beacon_at(self, p: QPoint, max_px_distance: float = 14.0):
        chosen = None
        chosen_dist = max_px_distance
        for beacon in self.runtime_view.world.structures.values():
            if not self._matches_current_view_system(getattr(beacon, "system_id", "")):
                continue
            sx, sy = self._to_screen(beacon.position)
            dx = sx - p.x()
            dy = sy - p.y()
            dist = (dx * dx + dy * dy) ** 0.5
            pick_radius = max(max_px_distance, float(beacon.radius) * self.world_scale())
            if dist <= pick_radius and dist <= chosen_dist:
                chosen = beacon
                chosen_dist = dist
        return chosen

    def _to_screen(self, p: Vector2) -> tuple[int, int]:
        x, y = self._to_screen_float(p)
        return int(x), int(y)

    def _to_screen_float(self, p: Vector2) -> tuple[float, float]:
        cx = self.width() // 2
        cy = self.height() // 2
        scale = self.world_scale()
        x = float(cx) + (float(p.x) - float(self.pan_world.x)) * scale
        y = float(cy) + (float(p.y) - float(self.pan_world.y)) * scale
        return x, y

    def _screen_circle_rect(self, world_center: Vector2, radius_px: float) -> QRectF | None:
        x, y = self._to_screen_float(world_center)
        r = max(0.0, float(radius_px))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            return None
        margin = float(self._SCREEN_DRAW_MARGIN_PX)
        width = float(self.width())
        height = float(self.height())
        if x + r < -margin or x - r > width + margin or y + r < -margin or y - r > height + margin:
            return None
        return QRectF(x - r, y - r, r * 2.0, r * 2.0)

    def _screen_point_if_visible(self, world_point: Vector2, radius_px: float = 0.0) -> tuple[int, int] | None:
        x, y = self._to_screen_float(world_point)
        r = max(0.0, float(radius_px))
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        margin = float(self._SCREEN_DRAW_MARGIN_PX) + r
        width = float(self.width())
        height = float(self.height())
        if x < -margin or x > width + margin or y < -margin or y > height + margin:
            return None
        return int(round(x)), int(round(y))

    def _to_world(self, p: QPoint) -> Vector2:
        cx = self.width() // 2
        cy = self.height() // 2
        scale = self.world_scale()
        wx = (p.x() - cx) / scale + self.pan_world.x
        wy = (p.y() - cy) / scale + self.pan_world.y
        return Vector2(wx, wy)

    def center_on_world(self, position: Vector2) -> None:
        self.pan_world = Vector2(float(position.x), float(position.y))
        self.update()

    def focus_ship(self, ship_id: str) -> None:
        ship = self.runtime_view.world.ships.get(str(ship_id))
        if ship is None:
            return
        self.current_view_system_id = str(getattr(ship.nav, "system_id", "") or self.current_view_system_id)
        self.selected_structure_id = None
        self.selected_ship_id = str(ship.ship_id)
        self.center_on_world(ship.nav.position)

    def focus_structure(self, structure_id: str) -> None:
        structure = self.runtime_view.world.structures.get(str(structure_id))
        if structure is None:
            return
        self.current_view_system_id = str(getattr(structure, "system_id", "") or self.current_view_system_id)
        self.selected_ship_id = None
        self.selected_structure_id = str(structure.structure_id)
        self.center_on_world(structure.position)

    def focus_system(self, system_id: str, padding_ratio: float = 0.82) -> None:
        map_definition = getattr(self.runtime_view.world, "map_definition", None)
        if map_definition is None:
            return
        system = map_definition.system_by_id(str(system_id))
        if system is None:
            return
        self.current_view_system_id = str(system.system_id)
        available_w = max(200.0, float(self.width() or self.minimumWidth() or 1))
        available_h = max(200.0, float(self.height() or self.minimumHeight() or 1))
        diameter = max(1.0, float(system.radius_m) * 2.0)
        target_scale = min((available_w * padding_ratio) / diameter, (available_h * padding_ratio) / diameter)
        target_zoom = target_scale / float(self._ZOOM_WORLD_SCALE_BASE)
        self.zoom = self._clamp_zoom(target_zoom)
        self.selected_ship_id = None
        self.selected_structure_id = None
        self.center_on_world(self._current_system_center())

    def fit_to_map(self, padding_ratio: float = 0.9) -> None:
        current = str(self.current_view_system_id or "").strip()
        if current:
            self.focus_system(current, padding_ratio=padding_ratio)
            return
        map_definition = getattr(self.runtime_view.world, "map_definition", None)
        if map_definition is not None and getattr(map_definition, "systems", None):
            first_system_id = str(getattr(map_definition.systems[0], "system_id", "") or "")
            if first_system_id:
                self.focus_system(first_system_id, padding_ratio=padding_ratio)
                return
        min_x, min_y, max_x, max_y = self._map_bounds()
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        self.pan_world = Vector2((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        available_w = max(200.0, float(self.width() or self.minimumWidth() or 1))
        available_h = max(200.0, float(self.height() or self.minimumHeight() or 1))
        target_scale = min((available_w * padding_ratio) / span_x, (available_h * padding_ratio) / span_y)
        self.zoom = self._clamp_zoom(target_scale / float(self._ZOOM_WORLD_SCALE_BASE))
        self.update()

    @classmethod
    def _clamp_zoom(cls, zoom: float) -> float:
        return max(cls._MIN_ZOOM, min(cls._MAX_ZOOM, float(zoom)))

    def _set_zoom_anchored(self, target_zoom: float, anchor: QPoint | None = None) -> None:
        next_zoom = self._clamp_zoom(target_zoom)
        if math.isclose(next_zoom, self.zoom, rel_tol=0.0, abs_tol=1e-12):
            return
        if anchor is None:
            anchor = QPoint(self.width() // 2, self.height() // 2)
        focus_world = self._to_world(anchor)
        self.zoom = next_zoom
        cx = self.width() // 2
        cy = self.height() // 2
        scale = self.world_scale()
        self.pan_world = Vector2(
            focus_world.x - (anchor.x() - cx) / scale,
            focus_world.y - (anchor.y() - cy) / scale,
        )

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self._set_zoom_anchored(self.zoom * 1.15, event.position().toPoint())
        elif delta < 0:
            self._set_zoom_anchored(self.zoom / 1.15, event.position().toPoint())
        self.update()

    def _map_bounds(self) -> tuple[float, float, float, float]:
        system = self._current_system_definition()
        if system is not None:
            radius = max(1_000.0, float(system.radius_m or 1_000.0))
            return -radius, -radius, radius, radius
        if not self.runtime_view.world.ships:
            return -100_000.0, -100_000.0, 100_000.0, 100_000.0
        xs = [
            float(ship.nav.position.x)
            for ship in self.runtime_view.world.ships.values()
            if self._matches_current_view_system(getattr(ship.nav, "system_id", ""))
        ]
        ys = [
            float(ship.nav.position.y)
            for ship in self.runtime_view.world.ships.values()
            if self._matches_current_view_system(getattr(ship.nav, "system_id", ""))
        ]
        if not xs or not ys:
            return -100_000.0, -100_000.0, 100_000.0, 100_000.0
        pad = 50_000.0
        return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad

    @classmethod
    def _structure_style(cls, structure) -> tuple[QColor, QColor]:
        if str(getattr(structure, "kind", "") or "").upper() == "STARGATE":
            return QColor(cls._STARGATE_COLOR), QColor(255, 206, 120, 40)
        return QColor(cls._STRUCTURE_COLOR), QColor(188, 198, 212, 28)

    def _draw_structure_icon(self, painter: QPainter, x: int, y: int, structure) -> None:
        border, _fill = self._structure_style(structure)
        if str(getattr(structure, "structure_id", "")) == str(self.selected_structure_id or ""):
            painter.setPen(QPen(self._SELECTION_HIGHLIGHT_COLOR, 3))
            radius_px = max(10, (self._STRUCTURE_ICON_SIZE_PX // 2) + 5)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)
        size = self._STRUCTURE_ICON_SIZE_PX
        half = size // 2
        painter.setPen(QPen(border, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        kind = str(getattr(structure, "kind", "") or "").upper()
        if kind == "STARGATE":
            painter.drawLine(x, y - half, x + half, y)
            painter.drawLine(x + half, y, x, y + half)
            painter.drawLine(x, y + half, x - half, y)
            painter.drawLine(x - half, y, x, y - half)
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            return
        painter.drawRect(x - half, y - half, size, size)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            clicked = self._pick_ship_at(event.position().toPoint())
            clicked_deployable = None if clicked is not None else self._pick_deployable_at(event.position().toPoint())
            clicked_beacon = None if clicked is not None or clicked_deployable is not None else self._pick_beacon_at(event.position().toPoint())
            if clicked is not None:
                self.selected_ship_id = clicked.ship_id
                self.selected_structure_id = None
                controlled_team = self.controlled_team_getter()
                if clicked.team == controlled_team:
                    self.selected_squad = clicked.squad_id
                    self.on_select_squad(clicked.squad_id)
                else:
                    self.selected_enemy_target = clicked.ship_id
                    self.on_select_enemy(clicked.ship_id)
            elif clicked_deployable is not None:
                self.selected_ship_id = clicked_deployable.ship_id
                self.selected_structure_id = None
                controlled_team = self.controlled_team_getter()
                if clicked_deployable.team == controlled_team:
                    self.selected_squad = clicked_deployable.squad_id
                    self.on_select_squad(clicked_deployable.squad_id)
                else:
                    self.selected_enemy_target = clicked_deployable.ship_id
                    self.on_select_enemy(clicked_deployable.ship_id)
            elif clicked_beacon is not None:
                self.selected_ship_id = None
                self.selected_structure_id = str(clicked_beacon.structure_id)
            self.update()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_active = True
            self.pan_start = event.position().toPoint()
            self.pan_start_world = Vector2(self.pan_world.x, self.pan_world.y)

    def mouseMoveEvent(self, event) -> None:
        if self.pan_active and self.pan_start is not None:
            now = event.position().toPoint()
            dx = now.x() - self.pan_start.x()
            dy = now.y() - self.pan_start.y()
            scale = self.world_scale()
            self.pan_world = Vector2(
                self.pan_start_world.x - dx / scale,
                self.pan_start_world.y - dy / scale,
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            world_target = self._to_world(event.position().toPoint())
            clicked = self._pick_ship_at(event.position().toPoint())
            clicked_deployable = None if clicked is not None else self._pick_deployable_at(event.position().toPoint())
            clicked_beacon = None if clicked is not None or clicked_deployable is not None else self._pick_beacon_at(event.position().toPoint())
            if clicked is not None and clicked.vital.alive:
                self.selected_ship_id = clicked.ship_id
                self.selected_structure_id = None
                self.on_show_ship_context_menu(clicked.ship_id, event.globalPosition().toPoint())
                self.update()
                return
            elif clicked_deployable is not None and clicked_deployable.vital.alive:
                self.selected_ship_id = clicked_deployable.ship_id
                self.selected_structure_id = None
                self.on_show_ship_context_menu(clicked_deployable.ship_id, event.globalPosition().toPoint())
                self.update()
                return
            elif clicked_beacon is not None:
                self.selected_ship_id = None
                self.selected_structure_id = str(clicked_beacon.beacon_id)
                self.on_show_structure_context_menu(clicked_beacon.beacon_id, event.globalPosition().toPoint())
                self.update()
                return

            menu = QMenu(self)
            squad = str(self.selected_squad or "")
            drone_menu = menu.addMenu(self._ui_text("Launch Squad Drones"))
            for type_name in self.squad_drone_types_getter(squad):
                action = QAction(self._display_type_name(type_name), self)
                action.triggered.connect(lambda _checked=False, name=type_name, sid=squad: self.on_launch_squad_drones(sid, name))
                drone_menu.addAction(action)
            if not drone_menu.actions():
                drone_menu.setEnabled(False)

            fighter_menu = menu.addMenu(self._ui_text("Launch Squad Fighters"))
            for type_name in self.squad_fighter_types_getter(squad):
                action = QAction(self._display_type_name(type_name), self)
                action.triggered.connect(lambda _checked=False, name=type_name, sid=squad: self.on_launch_squad_fighters(sid, name))
                fighter_menu.addAction(action)
            if not fighter_menu.actions():
                fighter_menu.setEnabled(False)

            action_recall = QAction(self._ui_text("Recall Squad Drones/Fighters"), self)
            action_recall.triggered.connect(lambda _checked=False, sid=squad: self.on_recall_squad_deployables(sid))
            action_recall.setEnabled(bool(squad))
            menu.addAction(action_recall)
            menu.addSeparator()
            squad_menu = menu.addMenu(QCoreApplication.translate("eve_sim", 'Induce Squad Here'))
            squads = self.controlled_squads_getter()
            for squad_id in squads:
                action = QAction(squad_id, self)
                action.triggered.connect(
                    lambda _checked=False, sid=squad_id, t=Vector2(world_target.x, world_target.y): self.on_induce_squad_spawn(sid, t)
                )
                squad_menu.addAction(action)
            if not squads:
                squad_menu.setEnabled(False)

            action_induce_fleet = QAction(QCoreApplication.translate("eve_sim", 'Induce Fleet Here'), self)
            action_induce_fleet.triggered.connect(lambda: self.on_induce_fleet_spawn(Vector2(world_target.x, world_target.y)))
            menu.addAction(action_induce_fleet)

            menu.exec(event.globalPosition().toPoint())
            self.update()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_active = False

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        clicked = self._pick_ship_at(event.position().toPoint())
        clicked_deployable = None if clicked is not None else self._pick_deployable_at(event.position().toPoint())
        clicked_beacon = None if clicked is not None or clicked_deployable is not None else self._pick_beacon_at(event.position().toPoint())
        if clicked is not None and clicked.vital.alive and self.selected_squad:
            controlled_team = self.controlled_team_getter()
            if clicked.team != controlled_team:
                self.selected_enemy_target = clicked.ship_id
                self.on_select_enemy(clicked.ship_id)
            self.on_issue_approach(self.selected_squad, clicked.ship_id)
        elif clicked_deployable is not None and clicked_deployable.vital.alive and self.selected_squad:
            controlled_team = self.controlled_team_getter()
            if clicked_deployable.team != controlled_team:
                self.selected_enemy_target = clicked_deployable.ship_id
                self.on_select_enemy(clicked_deployable.ship_id)
            self.on_issue_approach(self.selected_squad, clicked_deployable.ship_id)
        elif clicked_beacon is not None:
            self.focus_structure(str(clicked_beacon.structure_id))
        elif self.selected_squad:
            world_target = self._to_world(event.position().toPoint())
            self.on_issue_move(self.selected_squad, world_target)
        self.update()

    def _selected_squad_leader_ship(self):
        controlled_team = self.controlled_team_getter()
        leader_key = squad_key(controlled_team, self.selected_squad)
        leader_id = self.runtime_view.world.squad_leaders.get(leader_key)
        leader_ship = self.runtime_view.world.ships.get(leader_id) if leader_id else None
        if (
            leader_ship is None
            or not leader_ship.vital.alive
            or not self.ship_visible_getter(leader_ship.ship_id)
            or not self._matches_current_view_system(getattr(leader_ship.nav, "system_id", ""))
        ):
            members = [
                s
                for s in self.runtime_view.world.ships.values()
                if (
                    s.team == controlled_team
                    and s.squad_id == self.selected_squad
                    and s.vital.alive
                    and self.ship_visible_getter(s.ship_id)
                    and self._matches_current_view_system(getattr(s.nav, "system_id", ""))
                )
            ]
            leader_ship = members[0] if members else None
        return leader_ship

    @staticmethod
    def _module_area_style(module: ModuleRuntime) -> tuple[QColor, QColor] | None:
        tags = {str(value) for value in getattr(module, "tags", ()) or ()}
        if "command_burst" in tags:
            return QColor(88, 214, 141, 13), QColor(88, 214, 141, 13)
        if "smart_bomb" in tags:
            return QColor(255, 145, 77, 13), QColor(255, 165, 96, 13)
        if "bubble" in tags:
            return QColor(77, 171, 247, 13), QColor(116, 192, 252, 13)
        return None

    @staticmethod
    def _module_area_radius(module: ModuleRuntime) -> float:
        radius_m = 0.0
        for effect in module.effects:
            if effect.effect_class == EffectClass.PROJECTED:
                radius_m = max(radius_m, max(0.0, float(effect.range_m or 0.0)))
            radius_m = max(radius_m, max(0.0, float(effect.local_add.get("bubble_radius_m", 0.0) or 0.0)))
        return radius_m

    @staticmethod
    def _module_area_expand_duration(module: ModuleRuntime) -> float:
        tags = {str(value) for value in getattr(module, "tags", ()) or ()}
        if "command_burst" in tags:
            return 0.35
        return 0.0

    def _sync_area_cycle_overlays(self) -> None:
        now = float(self.runtime_view.world.now)
        cycle_restart_margin = 0.2
        for key, overlay in list(self._area_cycle_overlays.items()):
            ship = self.runtime_view.world.ships.get(overlay.ship_id)
            if ship is None or ship.runtime is None or not ship.vital.alive or not self.ship_visible_getter(overlay.ship_id):
                self._area_cycle_overlays.pop(key, None)
                continue
            module = next((m for m in ship.runtime.modules if m.module_id == overlay.module_id), None)
            if module is None or self._module_area_style(module) is None or module.state != ModuleState.ACTIVE:
                self._area_cycle_overlays.pop(key, None)
                continue
            overlay.center = Vector2(ship.nav.position.x, ship.nav.position.y)
            overlay.radius_m = self._module_area_radius(module)
            cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(overlay.module_id, 0.0) or 0.0))
            if cycle_left <= 0.0 or now >= overlay.expires_at:
                self._area_cycle_overlays.pop(key, None)

        for ship in self.runtime_view.world.ships.values():
            if not ship.vital.alive or ship.runtime is None or not self.ship_visible_getter(ship.ship_id):
                continue
            for module in ship.runtime.modules:
                style = self._module_area_style(module)
                if style is None:
                    continue
                cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module.module_id, 0.0) or 0.0))
                if module.state != ModuleState.ACTIVE or cycle_left <= 0.0:
                    continue
                radius_m = self._module_area_radius(module)
                if radius_m <= 0.0:
                    continue
                key = (ship.ship_id, module.module_id)
                existing_overlay = self._area_cycle_overlays.get(key)
                remaining = max(0.0, existing_overlay.expires_at - now) if existing_overlay is not None else 0.0
                if existing_overlay is not None and cycle_left <= remaining + cycle_restart_margin:
                    continue
                self._area_cycle_overlays[key] = AreaCycleOverlay(
                    ship_id=ship.ship_id,
                    module_id=module.module_id,
                    center=Vector2(ship.nav.position.x, ship.nav.position.y),
                    radius_m=radius_m,
                    fill_color=style[0],
                    border_color=style[1],
                    started_at=now,
                    expires_at=now + cycle_left,
                    expand_duration_sec=self._module_area_expand_duration(module),
                )

    def _iter_active_area_overlays(self) -> list[AreaCycleOverlay]:
        self._sync_area_cycle_overlays()
        overlays: list[AreaCycleOverlay] = []
        for overlay in self._area_cycle_overlays.values():
            ship = self.runtime_view.world.ships.get(overlay.ship_id)
            if ship is None or not self._matches_current_view_system(getattr(ship.nav, "system_id", "")):
                continue
            overlays.append(overlay)
        return overlays

    def _iter_active_projectile_blasts(self):
        now = float(self.runtime_view.world.now)
        for blast in self.runtime_view.world.projectile_blasts.values():
            if float(blast.expires_at) <= now:
                continue
            if str(blast.kind) != "bomb":
                continue
            if not self._matches_current_view_system(getattr(blast, "system_id", "")):
                continue
            yield blast

    def _iter_active_bubble_fields(self):
        now = float(self.runtime_view.world.now)
        for field in self.runtime_view.world.bubble_fields.values():
            if not bool(getattr(field, "alive", True)):
                continue
            if float(getattr(field, "expires_at", now + 1.0)) <= now and getattr(field, "anchor_ship_id", None) is None:
                continue
            field_system_id = str(getattr(field, "system_id", "") or "")
            if not field_system_id:
                anchor_ship_id = str(getattr(field, "anchor_ship_id", "") or getattr(field, "source_ship_id", "") or "")
                anchor_ship = self.runtime_view.world.ships.get(anchor_ship_id) if anchor_ship_id else None
                field_system_id = str(getattr(getattr(anchor_ship, "nav", None), "system_id", "") or "")
            if not self._matches_current_view_system(field_system_id):
                continue
            yield field

    @staticmethod
    def _bubble_field_style(kind: str) -> tuple[QColor, QColor]:
        if str(kind) == "webification_probe":
            return QColor(88, 196, 255, 20), QColor(88, 196, 255, 120)
        if str(kind) == "hic_warp_field":
            return QColor(255, 118, 86, 18), QColor(255, 118, 86, 110)
        return QColor(255, 78, 78, 18), QColor(255, 78, 78, 110)

    @staticmethod
    def _projectile_colors(kind: str) -> tuple[QColor, QColor]:
        if str(kind) == "bomb":
            return QColor(150, 100, 55), QColor(160, 110, 70, 28)
        return QColor(184, 96, 255), QColor(184, 96, 255, 24)

    def paintEvent(self, event) -> None:
        del event
        self._begin_render_frame()
        self._rendering_visual_frame = True
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._ensure_bg_cache()
            if self._bg_cache is not None:
                painter.drawPixmap(0, 0, self._bg_cache)
            else:
                painter.fillRect(self.rect(), QColor(15, 18, 24))
            scale = self.world_scale()

            for beacon in self.runtime_view.world.structures.values():
                if not self._matches_current_view_system(getattr(beacon, "system_id", "")):
                    continue
                r = max(2.0, float(beacon.radius) * scale)
                rect = self._screen_circle_rect(beacon.position, r)
                if rect is None:
                    continue
                border, fill = self._structure_style(beacon)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(border, 1))
                painter.drawEllipse(rect)
                point = self._screen_point_if_visible(beacon.position, self._STRUCTURE_ICON_SIZE_PX)
                if point is not None:
                    self._draw_structure_icon(painter, point[0], point[1], beacon)

            for overlay in self._iter_active_area_overlays():
                expand_duration = max(0.0, float(getattr(overlay, "expand_duration_sec", 0.0) or 0.0))
                if expand_duration > 0.0:
                    progress = max(0.0, min(1.0, (float(self.runtime_view.world.now) - float(overlay.started_at)) / expand_duration))
                else:
                    progress = 1.0
                radius_px = max(1.0, float(overlay.radius_m) * scale * progress)
                center = overlay.center
                ship = self.runtime_view.world.ships.get(str(getattr(overlay, "ship_id", "") or ""))
                if ship is not None:
                    center = self._ship_render_position(ship)
                rect = self._screen_circle_rect(center, radius_px)
                if rect is None:
                    continue
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(overlay.fill_color)
                painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(overlay.border_color, 1))
                painter.drawEllipse(rect)

            for blast in self._iter_active_projectile_blasts():
                radius_px = max(1.0, float(blast.radius_m) * scale)
                rect = self._screen_circle_rect(blast.position, radius_px)
                if rect is None:
                    continue
                fill = QColor(160, 110, 70, 18)
                border = QColor(176, 122, 76, 120)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(border, 1))
                painter.drawEllipse(rect)

            for field in self._iter_active_bubble_fields():
                radius_px = max(1.0, float(field.radius_m) * scale)
                rect = self._screen_circle_rect(field.position, radius_px)
                if rect is None:
                    continue
                point = self._screen_point_if_visible(field.position, 4.0)
                fill, border = self._bubble_field_style(str(field.kind))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(border, 1))
                painter.drawEllipse(rect)
                if point is not None:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(border)
                    painter.drawEllipse(QRectF(point[0] - 2.0, point[1] - 2.0, 4.0, 4.0))

            for projectile in self.runtime_view.world.projectiles.values():
                projectile_system_id = str(getattr(projectile, "system_id", "") or "")
                if not projectile_system_id:
                    source_ship = self.runtime_view.world.ships.get(str(getattr(projectile, "source_ship_id", "") or ""))
                    projectile_system_id = str(getattr(getattr(source_ship, "nav", None), "system_id", "") or "")
                if not self._matches_current_view_system(projectile_system_id):
                    continue
                projectile_color, _trail_color = self._projectile_colors(projectile.kind)
                radius_px = 3.0 if str(projectile.kind) == "bomb" else 2.0
                rect = self._screen_circle_rect(self._projectile_render_position(projectile), radius_px)
                if rect is None:
                    continue
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(projectile_color)
                painter.drawEllipse(rect)

            controlled_team = self.controlled_team_getter()
            for drone in self.runtime_view.world.drones.values():
                if not drone.vital.alive or not self._matches_current_view_system(getattr(drone.nav, "system_id", "")):
                    continue
                point = self._screen_point_if_visible(self._project_position(drone.nav.position, drone.nav.velocity), 8.0)
                if point is None:
                    continue
                x, y = point
                color = self._deployable_draw_color(drone, controlled_team, self.selected_squad)
                icon_key = self._deployable_icon_key(drone)
                icon = self._deployable_icon_pixmap(icon_key, color)
                if str(drone.ship_id) == str(self.selected_ship_id or ""):
                    highlight_icon = self._deployable_icon_pixmap(
                        icon_key,
                        self._SELECTION_HIGHLIGHT_COLOR,
                        self._DEPLOYABLE_ICON_SIZE_PX + 6,
                    )
                    if highlight_icon is not None:
                        painter.drawPixmap(
                            x - (highlight_icon.width() // 2),
                            y - (highlight_icon.height() // 2),
                            highlight_icon,
                        )
                    else:
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(self._SELECTION_HIGHLIGHT_COLOR)
                        painter.drawEllipse(x - 6, y - 6, 12, 12)
                if icon is not None:
                    old_opacity = painter.opacity()
                    if not getattr(drone, "connected", True):
                        painter.setOpacity(0.42)
                    painter.drawPixmap(x - (icon.width() // 2), y - (icon.height() // 2), icon)
                    painter.setOpacity(old_opacity)
                else:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(color)
                    painter.drawEllipse(x - 4, y - 4, 8, 8)

            for fighter in self.runtime_view.world.fighters.values():
                if not fighter.vital.alive or not self._matches_current_view_system(getattr(fighter.nav, "system_id", "")):
                    continue
                point = self._screen_point_if_visible(self._project_position(fighter.nav.position, fighter.nav.velocity), 10.0)
                if point is None:
                    continue
                x, y = point
                color = self._deployable_draw_color(fighter, controlled_team, self.selected_squad)
                icon_key = self._deployable_icon_key(fighter)
                icon = self._deployable_icon_pixmap(icon_key, color)
                if str(fighter.ship_id) == str(self.selected_ship_id or ""):
                    highlight_icon = self._deployable_icon_pixmap(
                        icon_key,
                        self._SELECTION_HIGHLIGHT_COLOR,
                        self._DEPLOYABLE_ICON_SIZE_PX + 8,
                    )
                    if highlight_icon is not None:
                        painter.drawPixmap(
                            x - (highlight_icon.width() // 2),
                            y - (highlight_icon.height() // 2),
                            highlight_icon,
                        )
                    else:
                        painter.setPen(QPen(self._SELECTION_HIGHLIGHT_COLOR, 3))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawLine(x, y - 10, x + 10, y)
                        painter.drawLine(x + 10, y, x, y + 10)
                        painter.drawLine(x, y + 10, x - 10, y)
                        painter.drawLine(x - 10, y, x, y - 10)
                if icon is not None:
                    old_opacity = painter.opacity()
                    if not getattr(fighter, "connected", True):
                        painter.setOpacity(0.42)
                    painter.drawPixmap(x - (icon.width() // 2), y - (icon.height() // 2), icon)
                    painter.setOpacity(old_opacity)
                else:
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawLine(x, y - 7, x + 7, y)
                    painter.drawLine(x + 7, y, x, y + 7)
                    painter.drawLine(x, y + 7, x - 7, y)
                    painter.drawLine(x - 7, y, x, y - 7)

            leader_ship = self._selected_squad_leader_ship()
            if leader_ship is not None:
                leader_point = self._screen_point_if_visible(self._ship_render_position(leader_ship), 1.0)
                if leader_point is not None:
                    cx, cy = leader_point
                else:
                    cx = cy = None
                ring_km = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200)
                if cx is not None and cy is not None:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    ring_pen = QPen(QColor(200, 200, 200, 120), 1)
                    painter.setPen(ring_pen)
                    label_pen = QPen(QColor(200, 200, 200, 140), 1)
                    for km in ring_km:
                        radius_px = max(1.0, float(km) * 1000.0 * scale)
                        rect = QRectF(cx - radius_px, cy - radius_px, radius_px * 2.0, radius_px * 2.0)
                        painter.drawEllipse(rect)
                        text = str(km)
                        metrics = painter.fontMetrics()
                        text_w = metrics.horizontalAdvance(text)
                        text_h = metrics.height()
                        pad = 4
                        painter.setPen(label_pen)
                        painter.drawText(
                            int(cx - (text_w // 2) - pad),
                            int(cy - radius_px - text_h),
                            text_w + pad * 2,
                            text_h,
                            Qt.AlignmentFlag.AlignCenter,
                            text,
                        )
                        painter.drawText(
                            int(cx - (text_w // 2) - pad),
                            int(cy + radius_px),
                            text_w + pad * 2,
                            text_h,
                            Qt.AlignmentFlag.AlignCenter,
                            text,
                        )
                        painter.drawText(
                            int(cx - radius_px - text_w - pad * 2),
                            int(cy - (text_h // 2)),
                            text_w + pad * 2,
                            text_h,
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                            text,
                        )
                        painter.drawText(
                            int(cx + radius_px + pad),
                            int(cy - (text_h // 2)),
                            text_w + pad * 2,
                            text_h,
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            text,
                        )
                        painter.setPen(ring_pen)

            controlled_team = self.controlled_team_getter()
            visible_ships = [
                ship
                for ship in self.runtime_view.world.ships.values()
                if self.ship_visible_getter(ship.ship_id)
                and self._matches_current_view_system(getattr(ship.nav, "system_id", ""))
            ]
            visible_ships.sort(key=self._ship_draw_priority)
            for ship in visible_ships:
                color = self._ship_draw_color(ship, controlled_team, self.selected_squad)
                point = self._screen_point_if_visible(self._ship_render_position(ship), self._SHIP_ICON_SIZE_PX)
                if point is None:
                    continue
                x, y = point
                ship_icon = self._ship_icon_for_name(ship.fit.ship_name, color)
                highlight_size_px = self._ship_selection_highlight_size_px(ship, self._SHIP_ICON_SIZE_PX)
                if ship_icon is not None:
                    if highlight_size_px is not None:
                        highlight_icon = self._ship_icon_for_name(
                            ship.fit.ship_name,
                            self._SELECTION_HIGHLIGHT_COLOR,
                            size_px=highlight_size_px,
                        )
                        if highlight_icon is not None:
                            painter.drawPixmap(
                                x - (highlight_icon.width() // 2),
                                y - (highlight_icon.height() // 2),
                                highlight_icon,
                            )
                    icon_w = ship_icon.width()
                    icon_h = ship_icon.height()
                    painter.drawPixmap(x - (icon_w // 2), y - (icon_h // 2), ship_icon)
                else:
                    highlight_radius = 0
                    if highlight_size_px is not None:
                        highlight_radius = max(1, int(round(highlight_size_px / 2.0)))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(self._SELECTION_HIGHLIGHT_COLOR)
                        painter.drawEllipse(
                            x - highlight_radius,
                            y - highlight_radius,
                            highlight_radius * 2,
                            highlight_radius * 2,
                        )
                    base_radius = 5
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(color)
                    painter.drawEllipse(x - base_radius, y - base_radius, base_radius * 2, base_radius * 2)

                hp_ratio = (ship.vital.shield + ship.vital.armor + ship.vital.structure) / (
                    ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
                )
                w = 16
                h = 3
                painter.setBrush(QColor(48, 48, 48))
                painter.drawRect(x - w // 2, y - 12, w, h)
                painter.setBrush(QColor(64, 220, 120))
                painter.drawRect(x - w // 2, y - 12, max(1, int(w * hp_ratio)), h)

            guidance_target = self.squad_guidance_target_getter(self.selected_squad)
            if leader_ship is not None and guidance_target is not None:
                start_point = self._screen_point_if_visible(self._ship_render_position(leader_ship), 1.0)
                end_point = self._screen_point_if_visible(guidance_target, 1.0)
                if start_point is not None and end_point is not None:
                    painter.setPen(QPen(QColor(120, 210, 255), 2, Qt.PenStyle.DashLine))
                    painter.drawLine(start_point[0], start_point[1], end_point[0], end_point[1])

            info = QCoreApplication.translate("eve_sim", 'Zoom: {zoom:.2f}  Pan: ({x:.0f}, {y:.0f})').format(zoom=self.zoom, x=self.pan_world.x, y=self.pan_world.y)
            painter.setPen(QPen(QColor(220, 220, 220), 1))
            painter.drawText(12, 20, info)

            controlled_team = self.controlled_team_getter()
            focus_queue = list(self.runtime_view.world.squad_focus_queues.get(squad_key(controlled_team, self.selected_squad), []))
            current_focus = focus_queue[0] if focus_queue else QCoreApplication.translate("eve_sim", 'None')
            prefocus_list = ", ".join(focus_queue[1:]) if len(focus_queue) > 1 else QCoreApplication.translate("eve_sim", 'None')
            right_x = max(12, self.width() - 520)
            painter.drawText(right_x, 20, QCoreApplication.translate("eve_sim", '{squad} Current Focus: {target}').format(squad=self.selected_squad, target=current_focus))
            painter.drawText(right_x, 40, QCoreApplication.translate("eve_sim", 'Pre-focus Queue: {targets}').format(targets=prefocus_list))
        finally:
            self._rendering_visual_frame = False
            self._prune_visual_ship_positions()
            if painter.isActive():
                painter.end()




