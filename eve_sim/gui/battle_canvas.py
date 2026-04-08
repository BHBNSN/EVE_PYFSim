from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Literal, cast

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QSortFilterProxyModel, QTimer, Qt, QLocale, QCoreApplication
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..agents import CommanderAgent
from ..config import EngineConfig, UiConfig
from ..fleet_setup import (
    ManualShipSetup,
    ParsedModuleSpec,
    build_world_from_manual_setup,
    EftFitParser,
    RuntimeFromEftFactory,
    recompute_profile_from_pyfa_runtime,
    get_charge_options_for_module,
    get_fit_backend_status,
    get_common_chargeable_modules,
    get_ship_icon_key,
    get_module_reload_time_sec,
    resolve_module_type_name,
    get_type_display_name,
)
from ..fit_runtime import EffectClass, ModuleRuntime, ModuleState, RuntimeStatEngine

from ..lan_session import ClientLanSession, HostLanSession
from ..lan_commands import (
    CMD_INDUCE_FLEET_AT,
    CMD_INDUCE_SQUAD_AT,
    CMD_SQUAD_APPROACH,
    CMD_SQUAD_ATTACK,
    CMD_SQUAD_CANCEL_PREFOCUS,
    CMD_SQUAD_CLEAR_FOCUS,
    CMD_SQUAD_LEADER_SPEED_LIMIT,
    CMD_SQUAD_MOVE,
    CMD_SQUAD_PREFOCUS,
    CMD_SQUAD_PROPULSION,
    CMD_SYNC_SETUP,
    SQUAD_FOCUS_COMMANDS,
)
from ..math2d import Vector2
from ..models import (
    CombatState,
    FitDescriptor,
    FleetIntent,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipEntity,
    ShipProfile,
    Team,
    VitalState,
)
from ..pyfa_bridge import PyfaBridge
from ..sim_logging import get_sim_logger, log_sim_event
from ..simulation_engine import SimulationEngine
from ..systems import CombatSystem



from .models import AreaCycleOverlay
class BattleCanvas(QWidget):
    _SHIP_ICON_SIZE_PX = 25
    _SHIP_ICON_SOURCE_CACHE: dict[str, QImage] | None = None
    _SHIP_ICON_PIXMAP_CACHE: dict[tuple[str, int, int, int, int, int], QPixmap] = {}
    _MIN_ZOOM = 0.00005
    _MAX_ZOOM = 0.1
    _TEAM_BLUE_COLOR = QColor(80, 180, 255)
    _TEAM_RED_COLOR = QColor(255, 92, 92)
    _SELECTED_SQUAD_COLOR = QColor(186, 102, 255)
    _DESTROYED_SHIP_COLOR = QColor(130, 130, 130)
    _SELECTION_HIGHLIGHT_COLOR = QColor(255, 230, 90)

    def __init__(
        self,
        engine: SimulationEngine,
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.ui_cfg = ui_cfg
        self.on_issue_move = on_issue_move
        self.on_issue_approach = on_issue_approach
        self.on_issue_warp_ship = on_issue_warp_ship
        self.on_issue_warp_beacon = on_issue_warp_beacon
        self.on_issue_focus = on_issue_focus
        self.on_issue_prefocus = on_issue_prefocus
        self.on_cancel_prefocus = on_cancel_prefocus
        self.on_show_ship_context_menu = on_show_ship_context_menu
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
        self.setMinimumSize(ui_cfg.width, ui_cfg.height)

        self.zoom = ui_cfg.world_to_screen_scale
        self.pan_world = Vector2(0.0, 0.0)
        self.selected_squad = "BLUE-ALPHA"
        self.selected_enemy_target: str | None = None
        self.selected_ship_id: str | None = None
        self.highlighted_roster_ship_ids: set[str] = set()

        self.pan_active = False
        self.pan_start: QPoint | None = None
        self.pan_start_world = Vector2(0.0, 0.0)
        self._bg_cache: QPixmap | None = None
        self._bg_cache_w = 0
        self._bg_cache_h = 0
        self._area_cycle_overlays: dict[tuple[str, str], AreaCycleOverlay] = {}

    @staticmethod
    def _focus_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

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
        painter.drawImage(0, 0, scaled)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), color)
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
    def _ship_draw_color(cls, ship: ShipEntity, controlled_team: Team, selected_squad: str) -> QColor:
        if not ship.vital.alive:
            return QColor(cls._DESTROYED_SHIP_COLOR)
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
        painter.fillRect(0, 0, width, height, QColor(15, 18, 24))
        pen_grid = QPen(QColor(40, 44, 52), 1)
        painter.setPen(pen_grid)
        for i in range(0, width, 50):
            painter.drawLine(i, 0, i, height)
        for j in range(0, height, 50):
            painter.drawLine(0, j, width, j)
        painter.end()
        self._bg_cache = bg
        self._bg_cache_w = width
        self._bg_cache_h = height

    def resizeEvent(self, event) -> None:
        self._bg_cache = None
        self._bg_cache_w = 0
        self._bg_cache_h = 0
        super().resizeEvent(event)

    def _pick_ship_at(self, p: QPoint, max_px_distance: float = 14.0):
        chosen = None
        chosen_dist = max_px_distance
        for ship in self.engine.world.ships.values():
            if not self.ship_visible_getter(ship.ship_id):
                continue
            if not ship.vital.alive:
                continue
            sx, sy = self._to_screen(ship.nav.position)
            dx = sx - p.x()
            dy = sy - p.y()
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= chosen_dist:
                chosen = ship
                chosen_dist = dist
        return chosen

    def _pick_beacon_at(self, p: QPoint, max_px_distance: float = 14.0):
        chosen = None
        chosen_dist = max_px_distance
        for beacon in self.engine.world.beacons.values():
            sx, sy = self._to_screen(beacon.position)
            dx = sx - p.x()
            dy = sy - p.y()
            dist = (dx * dx + dy * dy) ** 0.5
            pick_radius = max(max_px_distance, float(beacon.radius) * self.zoom)
            if dist <= pick_radius and dist <= chosen_dist:
                chosen = beacon
                chosen_dist = dist
        return chosen

    def _to_screen(self, p: Vector2) -> tuple[int, int]:
        cx = self.width() // 2
        cy = self.height() // 2
        x = int(cx + (p.x - self.pan_world.x) * self.zoom)
        y = int(cy + (p.y - self.pan_world.y) * self.zoom)
        return x, y

    def _to_world(self, p: QPoint) -> Vector2:
        cx = self.width() // 2
        cy = self.height() // 2
        wx = (p.x() - cx) / self.zoom + self.pan_world.x
        wy = (p.y() - cy) / self.zoom + self.pan_world.y
        return Vector2(wx, wy)

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
        self.pan_world = Vector2(
            focus_world.x - (anchor.x() - cx) / self.zoom,
            focus_world.y - (anchor.y() - cy) / self.zoom,
        )

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self._set_zoom_anchored(self.zoom * 1.15, event.position().toPoint())
        elif delta < 0:
            self._set_zoom_anchored(self.zoom / 1.15, event.position().toPoint())
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            clicked = self._pick_ship_at(event.position().toPoint())
            if clicked is not None:
                self.selected_ship_id = clicked.ship_id
                controlled_team = self.controlled_team_getter()
                if clicked.team == controlled_team:
                    self.selected_squad = clicked.squad_id
                    self.on_select_squad(clicked.squad_id)
                else:
                    self.selected_enemy_target = clicked.ship_id
                    self.on_select_enemy(clicked.ship_id)
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
            self.pan_world = Vector2(
                self.pan_start_world.x - dx / self.zoom,
                self.pan_start_world.y - dy / self.zoom,
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            world_target = self._to_world(event.position().toPoint())
            clicked = self._pick_ship_at(event.position().toPoint())
            clicked_beacon = None if clicked is not None else self._pick_beacon_at(event.position().toPoint())
            lang = self.language_getter()
            menu = QMenu(self)
            if clicked is not None and clicked.vital.alive:
                self.selected_ship_id = clicked.ship_id
                self.on_show_ship_context_menu(clicked.ship_id, event.globalPosition().toPoint())
                self.update()
                return
            elif clicked_beacon is not None:
                action_warp_beacon = QAction(
                    QCoreApplication.translate("eve_sim", '{squad} Warp To {beacon}').format(squad=self.selected_squad, beacon=clicked_beacon.beacon_id),
                    self,
                )
                action_warp_beacon.triggered.connect(lambda: self.on_issue_warp_beacon(self.selected_squad, clicked_beacon.beacon_id))
                menu.addAction(action_warp_beacon)
                menu.addSeparator()

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
        if clicked is not None and clicked.vital.alive and self.selected_squad:
            controlled_team = self.controlled_team_getter()
            if clicked.team != controlled_team:
                self.selected_enemy_target = clicked.ship_id
                self.on_select_enemy(clicked.ship_id)
            self.on_issue_approach(self.selected_squad, clicked.ship_id)
        elif self.selected_squad:
            world_target = self._to_world(event.position().toPoint())
            self.on_issue_move(self.selected_squad, world_target)
        self.update()

    def _selected_squad_leader_ship(self):
        controlled_team = self.controlled_team_getter()
        leader_key = self._focus_key(controlled_team, self.selected_squad)
        leader_id = self.engine.world.squad_leaders.get(leader_key)
        leader_ship = self.engine.world.ships.get(leader_id) if leader_id else None
        if leader_ship is None or not leader_ship.vital.alive or not self.ship_visible_getter(leader_ship.ship_id):
            members = [
                s
                for s in self.engine.world.ships.values()
                if s.team == controlled_team and s.squad_id == self.selected_squad and s.vital.alive and self.ship_visible_getter(s.ship_id)
            ]
            leader_ship = members[0] if members else None
        return leader_ship

    @staticmethod
    def _module_area_style(module: ModuleRuntime) -> tuple[QColor, QColor] | None:
        group = str(getattr(module, "group", "") or "").strip().lower()
        if group == "command burst":
            return QColor(88, 214, 141, 13), QColor(88, 214, 141, 13)
        if group in {"smart bomb", "structure area denial module"}:
            return QColor(255, 145, 77, 13), QColor(255, 165, 96, 13)
        return None

    @staticmethod
    def _module_area_radius(module: ModuleRuntime) -> float:
        radius_m = 0.0
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            radius_m = max(radius_m, max(0.0, float(effect.range_m or 0.0)))
        return radius_m

    @staticmethod
    def _module_area_expand_duration(module: ModuleRuntime) -> float:
        group = str(getattr(module, "group", "") or "").strip().lower()
        if group == "command burst":
            return 0.35
        return 0.0

    def _sync_area_cycle_overlays(self) -> None:
        now = float(self.engine.world.now)
        cycle_restart_margin = 0.2
        for key, overlay in list(self._area_cycle_overlays.items()):
            ship = self.engine.world.ships.get(overlay.ship_id)
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

        for ship in self.engine.world.ships.values():
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
        return list(self._area_cycle_overlays.values())

    def _iter_active_projectile_blasts(self):
        now = float(self.engine.world.now)
        for blast in self.engine.world.projectile_blasts.values():
            if float(blast.expires_at) <= now:
                continue
            if str(blast.kind) != "bomb":
                continue
            yield blast

    def _iter_active_bubble_fields(self):
        now = float(self.engine.world.now)
        for field in self.engine.world.bubble_fields.values():
            if not bool(getattr(field, "alive", True)):
                continue
            if float(getattr(field, "expires_at", now + 1.0)) <= now and getattr(field, "anchor_ship_id", None) is None:
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._ensure_bg_cache()
        if self._bg_cache is not None:
            painter.drawPixmap(0, 0, self._bg_cache)
        else:
            painter.fillRect(self.rect(), QColor(15, 18, 24))

        for beacon in self.engine.world.beacons.values():
            x, y = self._to_screen(beacon.position)
            r = max(3, int(beacon.radius * self.zoom))
            painter.setPen(QPen(QColor(255, 182, 74), 2))
            painter.drawEllipse(x - r, y - r, r * 2, r * 2)

        for overlay in self._iter_active_area_overlays():
            x, y = self._to_screen(overlay.center)
            expand_duration = max(0.0, float(getattr(overlay, "expand_duration_sec", 0.0) or 0.0))
            if expand_duration > 0.0:
                progress = max(0.0, min(1.0, (float(self.engine.world.now) - float(overlay.started_at)) / expand_duration))
            else:
                progress = 1.0
            radius_px = max(1, int(overlay.radius_m * self.zoom * progress))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(overlay.fill_color)
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(overlay.border_color, 1))
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)

        for blast in self._iter_active_projectile_blasts():
            x, y = self._to_screen(blast.position)
            radius_px = max(1, int(float(blast.radius_m) * self.zoom))
            fill = QColor(160, 110, 70, 18)
            border = QColor(176, 122, 76, 120)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(border, 1))
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)

        for field in self._iter_active_bubble_fields():
            x, y = self._to_screen(field.position)
            radius_px = max(1, int(float(field.radius_m) * self.zoom))
            fill, border = self._bubble_field_style(str(field.kind))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(border, 1))
            painter.drawEllipse(x - radius_px, y - radius_px, radius_px * 2, radius_px * 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(border)
            painter.drawEllipse(x - 2, y - 2, 4, 4)

        for projectile in self.engine.world.projectiles.values():
            px, py = self._to_screen(projectile.position)
            projectile_color, _trail_color = self._projectile_colors(projectile.kind)
            radius_px = 3 if str(projectile.kind) == "bomb" else 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(projectile_color)
            painter.drawEllipse(px - radius_px, py - radius_px, radius_px * 2, radius_px * 2)

        leader_ship = self._selected_squad_leader_ship()
        if leader_ship is not None:
            cx, cy = self._to_screen(leader_ship.nav.position)
            ring_km = (5, 10, 20, 30, 40, 50, 75, 100, 150, 200)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_pen = QPen(QColor(200, 200, 200, 120), 1)
            painter.setPen(ring_pen)
            label_pen = QPen(QColor(200, 200, 200, 140), 1)
            for km in ring_km:
                radius_px = max(1, int(km * 1000.0 * self.zoom))
                painter.drawEllipse(cx - radius_px, cy - radius_px, radius_px * 2, radius_px * 2)
                text = str(km)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(text)
                text_h = metrics.height()
                pad = 4
                painter.setPen(label_pen)
                painter.drawText(
                    cx - (text_w // 2) - pad,
                    cy - radius_px - text_h,
                    text_w + pad * 2,
                    text_h,
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )
                painter.drawText(
                    cx - (text_w // 2) - pad,
                    cy + radius_px,
                    text_w + pad * 2,
                    text_h,
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )
                painter.drawText(
                    cx - radius_px - text_w - pad * 2,
                    cy - (text_h // 2),
                    text_w + pad * 2,
                    text_h,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    text,
                )
                painter.drawText(
                    cx + radius_px + pad,
                    cy - (text_h // 2),
                    text_w + pad * 2,
                    text_h,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    text,
                )
                painter.setPen(ring_pen)

        controlled_team = self.controlled_team_getter()
        visible_ships = [
            ship
            for ship in self.engine.world.ships.values()
            if self.ship_visible_getter(ship.ship_id)
        ]
        visible_ships.sort(key=self._ship_draw_priority)
        for ship in visible_ships:
            color = self._ship_draw_color(ship, controlled_team, self.selected_squad)
            x, y = self._to_screen(ship.nav.position)
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
            start_x, start_y = self._to_screen(leader_ship.nav.position)
            end_x, end_y = self._to_screen(guidance_target)
            painter.setPen(QPen(QColor(120, 210, 255), 2, Qt.PenStyle.DashLine))
            painter.drawLine(start_x, start_y, end_x, end_y)

        lang = self.language_getter()
        info = QCoreApplication.translate("eve_sim", 'Zoom: {zoom:.5f}  Pan: ({x:.0f}, {y:.0f})').format(zoom=self.zoom, x=self.pan_world.x, y=self.pan_world.y)
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.drawText(12, 20, info)
        painter.drawText(12, 40, QCoreApplication.translate("eve_sim", 'Left click: select friendly squad/enemy target | Double-click space: move | Double-click ship: continuous approach | Right-click menu: induce deploy/focus | Middle drag: pan | Wheel: zoom'))

        controlled_team = self.controlled_team_getter()
        focus_queue = list(self.engine.world.squad_focus_queues.get(self._focus_key(controlled_team, self.selected_squad), []))
        current_focus = focus_queue[0] if focus_queue else QCoreApplication.translate("eve_sim", 'None')
        prefocus_list = ", ".join(focus_queue[1:]) if len(focus_queue) > 1 else QCoreApplication.translate("eve_sim", 'None')
        right_x = max(12, self.width() - 520)
        painter.drawText(right_x, 20, QCoreApplication.translate("eve_sim", '{squad} Current Focus: {target}').format(squad=self.selected_squad, target=current_focus))
        painter.drawText(right_x, 40, QCoreApplication.translate("eve_sim", 'Pre-focus Queue: {targets}').format(targets=prefocus_list))

        painter.end()




