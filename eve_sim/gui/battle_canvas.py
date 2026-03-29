from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Literal, cast

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QSortFilterProxyModel, QTimer, Qt, QLocale
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
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
    get_module_reload_time_sec,
    resolve_module_type_name,
    get_type_display_name,
)
from ..fit_runtime import EffectClass, ModuleRuntime, ModuleState, RuntimeStatEngine
from ..i18n import tr
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


def _localize_fit_error(lang: str, error: Exception | str) -> str:
    msg = str(error).strip()

    def tail(text: str) -> str:
        return text.split("：", 1)[1].strip() if "：" in text else ""

    if msg.startswith("pyfa Fit计算链不可用"):
        return tr(lang, "err_pyfa_chain_unavailable")
    if msg.startswith("pyfa Fit计算链初始化不完整"):
        return tr(lang, "err_pyfa_chain_incomplete")
    if msg.startswith("pyfa中未找到舰船"):
        return tr(lang, "err_pyfa_ship_not_found", name=tail(msg))
    if msg.startswith("pyfa中未找到模块"):
        return tr(lang, "err_pyfa_module_not_found", name=tail(msg))
    if msg.startswith("pyfa中未找到弹药"):
        return tr(lang, "err_pyfa_charge_not_found", name=tail(msg))
    if msg.startswith("武器缺少可解析弹药"):
        return tr(lang, "err_weapon_no_ammo", name=tail(msg))
    if msg.startswith("弹药与武器口径/类型不匹配"):
        return tr(lang, "err_ammo_mismatch", detail=tail(msg))
    return msg



from .models import *
class BattleCanvas(QWidget):
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

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(0.02, self.zoom * 1.15)
        elif delta < 0:
            self.zoom = max(0.0001, self.zoom / 1.15)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            clicked = self._pick_ship_at(event.position().toPoint())
            if clicked is not None:
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
                action_status = QAction(tr(lang, "menu_show_status", ship=clicked.ship_id), self)
                action_status.triggered.connect(lambda: self.on_show_status(clicked.ship_id))
                menu.addAction(action_status)
                action_warp = QAction(tr(lang, "menu_warp_ship", squad=self.selected_squad, ship=clicked.ship_id), self)
                action_warp.triggered.connect(lambda: self.on_issue_warp_ship(self.selected_squad, clicked.ship_id))
                menu.addAction(action_warp)
                controlled_team = self.controlled_team_getter()
                if clicked.team != controlled_team:
                    self.selected_enemy_target = clicked.ship_id
                    self.on_select_enemy(clicked.ship_id)
                    action_focus = QAction(tr(lang, "menu_focus", squad=self.selected_squad, ship=clicked.ship_id), self)
                    action_focus.triggered.connect(lambda: self.on_issue_focus(clicked.ship_id))
                    menu.addAction(action_focus)
                    action_prefocus = QAction(tr(lang, "menu_prefocus", squad=self.selected_squad, ship=clicked.ship_id), self)
                    action_prefocus.triggered.connect(lambda: self.on_issue_prefocus(clicked.ship_id))
                    menu.addAction(action_prefocus)
                    focus_key = self._focus_key(controlled_team, self.selected_squad)
                    queue = self.engine.world.squad_focus_queues.get(focus_key, [])
                    in_prequeue = clicked.ship_id in queue
                    prelocked_by_ship = self.engine.world.squad_prelocked_targets.get(focus_key, {})
                    prelock_timers_by_ship = self.engine.world.squad_prelock_timers.get(focus_key, {})
                    prelocked = any(clicked.ship_id in targets for targets in prelocked_by_ship.values())
                    prelocking = any(clicked.ship_id in timers for timers in prelock_timers_by_ship.values())
                    if in_prequeue or prelocked or prelocking:
                        action_cancel_prefocus = QAction(
                            tr(lang, "menu_cancel_prefocus", squad=self.selected_squad, ship=clicked.ship_id),
                            self,
                        )
                        action_cancel_prefocus.triggered.connect(lambda: self.on_cancel_prefocus(clicked.ship_id))
                        menu.addAction(action_cancel_prefocus)
            elif clicked_beacon is not None:
                action_warp_beacon = QAction(
                    tr(lang, "menu_warp_beacon", squad=self.selected_squad, beacon=clicked_beacon.beacon_id),
                    self,
                )
                action_warp_beacon.triggered.connect(lambda: self.on_issue_warp_beacon(self.selected_squad, clicked_beacon.beacon_id))
                menu.addAction(action_warp_beacon)
                menu.addSeparator()

            menu.addSeparator()
            squad_menu = menu.addMenu(tr(lang, "menu_induce_squad_here"))
            squads = self.controlled_squads_getter()
            for squad_id in squads:
                action = QAction(squad_id, self)
                action.triggered.connect(
                    lambda _checked=False, sid=squad_id, t=Vector2(world_target.x, world_target.y): self.on_induce_squad_spawn(sid, t)
                )
                squad_menu.addAction(action)
            if not squads:
                squad_menu.setEnabled(False)

            action_induce_fleet = QAction(tr(lang, "menu_induce_fleet_here"), self)
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
                    expires_at=now + cycle_left,
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
            radius_px = max(1, int(overlay.radius_m * self.zoom))
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

        for ship in self.engine.world.ships.values():
            if not self.ship_visible_getter(ship.ship_id):
                continue
            color = QColor(80, 180, 255) if ship.team == Team.BLUE else QColor(255, 92, 92)
            if not ship.vital.alive:
                color = QColor(130, 130, 130)
            x, y = self._to_screen(ship.nav.position)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(x - 4, y - 4, 8, 8)

            controlled_team = self.controlled_team_getter()
            if ship.team == controlled_team and ship.squad_id == self.selected_squad and ship.vital.alive:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(188, 238, 255), 2))
                painter.drawEllipse(x - 8, y - 8, 16, 16)
                painter.setPen(Qt.PenStyle.NoPen)

            if self.selected_enemy_target and ship.ship_id == self.selected_enemy_target and ship.vital.alive:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 230, 90), 2))
                painter.drawEllipse(x - 10, y - 10, 20, 20)
                painter.setPen(Qt.PenStyle.NoPen)

            hp_ratio = (ship.vital.shield + ship.vital.armor + ship.vital.structure) / (
                ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
            )
            w = 16
            h = 3
            painter.setBrush(QColor(48, 48, 48))
            painter.drawRect(x - w // 2, y - 12, w, h)
            painter.setBrush(QColor(64, 220, 120))
            painter.drawRect(x - w // 2, y - 12, max(1, int(w * hp_ratio)), h)

        controlled_team = self.controlled_team_getter()
        guidance_target = self.squad_guidance_target_getter(self.selected_squad)
        if leader_ship is not None and guidance_target is not None:
            start_x, start_y = self._to_screen(leader_ship.nav.position)
            end_x, end_y = self._to_screen(guidance_target)
            painter.setPen(QPen(QColor(120, 210, 255), 2, Qt.PenStyle.DashLine))
            painter.drawLine(start_x, start_y, end_x, end_y)

        lang = self.language_getter()
        info = tr(lang, "canvas_zoom_pan", zoom=self.zoom, x=self.pan_world.x, y=self.pan_world.y)
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.drawText(12, 20, info)
        painter.drawText(12, 40, tr(lang, "canvas_help"))

        controlled_team = self.controlled_team_getter()
        focus_queue = list(self.engine.world.squad_focus_queues.get(self._focus_key(controlled_team, self.selected_squad), []))
        current_focus = focus_queue[0] if focus_queue else tr(lang, "focus_none")
        prefocus_list = ", ".join(focus_queue[1:]) if len(focus_queue) > 1 else tr(lang, "focus_none")
        right_x = max(12, self.width() - 520)
        painter.drawText(right_x, 20, tr(lang, "canvas_focus_current", squad=self.selected_squad, target=current_focus))
        painter.drawText(right_x, 40, tr(lang, "canvas_focus_queue", targets=prefocus_list))

        painter.end()



