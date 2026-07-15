from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from PySide6.QtCore import QModelIndex, QPoint, QTimer, Qt, QCoreApplication, QItemSelectionModel
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..application import MatchApplication
from ..battle_report import BattleReportService
from ..config import UiConfig
from ..fleet_setup import (
    EftFitParser,
    RuntimeFromEftFactory,
    get_charge_option_values_for_module,
    get_common_chargeable_modules,
    module_supports_unloaded_charge,
    get_type_display_name,
)
from ..i18n import install_language, normalize_language
from ..lan_match_coordinator import LanMatchCoordinator
from ..lan_session import ClientLanSession, HostLanSession
from ..math2d import Vector2
from ..module_control import normalize_module_manual_mode, normalize_module_target_mode
from ..models import FighterAbilityProfile, Team
from ..replay import ReplayPlayer, ReplayRecorder
from ..squad_identity import squad_key
from ..user_errors import display_user_error
from .battle_canvas import BattleCanvas
from .battle_recording_controller import BattleRecordingController
from .battle_report_presenter import format_battle_report
from .adapters import ApplicationRuntimeView, GuiCommandAdapter
from .dialogs import OverviewOptionsDialog, ShipStatusDialog
from .replay_dialog import ReplayPlaybackDialog
from .system_graph_window import SystemGraphWindow
from .models import PreferencesStore, UiPreferences, UiState
from .table_models import BlueRosterTableModel, OverviewFilterProxyModel, OverviewTableModel

AU_METERS = 149_597_870_700.0
NAV_RANGE_OPTIONS_M = (1_000.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0, 50_000.0, 70_000.0, 100_000.0)

class MainWindow(QMainWindow):
    """
    涓荤獥鍙ｇ被 (Main Window)
    
    璇ョ被璐熻矗缁勭粐鏁翠釜搴旂敤鐨勫浘褰㈢晫闈?(GUI)锛岃繛鎺ュ悇涓瓙妯″潡 (濡傛瑙堥潰鏉裤€佹垬鏂楃敾甯冪瓑)锛?
    骞剁鐞嗗叾鐢熷懡鍛ㄦ湡銆?
    """
    def __init__(
        self,
        application: MatchApplication,
        ui_cfg: UiConfig,
        network_mode: str = "local",
        controlled_team: Team = Team.BLUE,
        lan_server: HostLanSession | None = None,
        lan_client: ClientLanSession | None = None,
        initial_language: str | None = None,
    ) -> None:
        super().__init__()
        self.application = application
        self.runtime_view = ApplicationRuntimeView(application)
        self._parser = EftFitParser()
        self._factory = RuntimeFromEftFactory()
        self.network = LanMatchCoordinator(
            application,
            mode=network_mode,
            server=lan_server,
            client=lan_client,
        )
        remote_executor = (
            self.network.execute_remote
            if network_mode == "client" and lan_client is not None
            else None
        )
        self.command_adapter = GuiCommandAdapter(self.application, remote_executor)
        self.ui_cfg = ui_cfg
        self.network_mode = network_mode
        self.controlled_team = controlled_team
        self._charge_module_ammo_selection: dict[str, str] = {}
        self._squad_guidance_targets: dict[str, Vector2] = {}
        self.store = PreferencesStore()
        self.prefs = self.store.load()
        if initial_language is not None:
            self.prefs.language = normalize_language(initial_language, "en_US")
            self.store.save(self.prefs)
        install_language(self.current_language())
        if self.prefs.filter_team in ("BLUE", "RED"):
            if self.prefs.filter_team == self.controlled_team.value:
                self.prefs.filter_team = "FRIENDLY"
            else:
                self.prefs.filter_team = "ENEMY"
        if self.network_mode == "client":
            if self.prefs.filter_team == "ALL":
                self.prefs.filter_team = "ENEMY"

        self._initialize_deployment_state()
        self.runtime_view.refresh()

        self.ui_state = UiState(selected_squad=self.prefs.selected_squad, selected_enemy_target=None)
        self.setWindowTitle(QCoreApplication.translate("eve_sim", 'EVE SIM - Continuous Space Wargame'))
        self.resize(ui_cfg.width + 560, ui_cfg.height)
        try:
            configured_tick_rate = self.runtime_view.tick_rate
        except Exception:
            configured_tick_rate = 1
        refresh_interval_ticks = 3 if self.network_mode == "client" else max(1, int(round(configured_tick_rate / 10.0)))
        self._ui_refresh_interval_ticks = refresh_interval_ticks
        self._overview_refresh_interval_ticks = refresh_interval_ticks
        self._ui_tick_counter = 0
        self._last_overview_rows: list[dict] = []
        self._ship_type_display_cache: dict[tuple[str, str], str] = {}
        self._status_dialogs: dict[str, ShipStatusDialog] = {}
        self._step_ms_ema: float = 0.0
        self._client_poll_interval_ms: int = 50
        self._last_roster_refresh_tick: int | None = None
        self._view_system_id: str = ""
        self.recording = BattleRecordingController(
            self.application,
            network_mode=self.network_mode,
            controlled_team=self.controlled_team,
        )
        self.recording.attach()

        self._create_menu()

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter)

        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        self.canvas = BattleCanvas(
            self.runtime_view,
            ui_cfg,
            self.issue_move_to,
            self.issue_approach_target,
            self.issue_warp_to_ship,
            self.issue_warp_to_beacon,
            self.issue_focus_target,
            self.issue_prefocus_target,
            self.cancel_prefocus_target,
            self.show_ship_context_menu,
            self.induce_spawn_squad_at,
            self.induce_spawn_fleet_at,
            self._inducible_controlled_squad_ids,
            self._is_ship_visible,
            self._guidance_target_for_squad,
            self.show_ship_status,
            self.current_language,
            lambda: self.controlled_team,
            self.on_canvas_select_squad,
            self.on_canvas_select_enemy,
            on_show_structure_context_menu=self.show_structure_context_menu,
            squad_drone_types_getter=self.available_squad_drone_types,
            squad_fighter_types_getter=self.available_squad_fighter_types,
            on_launch_squad_drones=self.launch_squad_drones,
            on_launch_squad_fighters=self.launch_squad_fighters,
            on_recall_squad_deployables=self.recall_squad_deployables,
        )
        self.canvas.current_view_system_id = self._view_system_id
        if self.prefs.zoom is not None:
            self.canvas.zoom = self.canvas._clamp_zoom(float(self.prefs.zoom))
        splitter.addWidget(self.canvas)
        splitter.setSizes([560, ui_cfg.width])
        if self.prefs.zoom is None and self.runtime_view.world.map_definition is not None:
            QTimer.singleShot(0, self.canvas.fit_to_map)
        self.system_graph_window = SystemGraphWindow(
            runtime_view=self.runtime_view,
            current_system_getter=lambda: str(self.canvas.current_view_system_id or self._current_view_system_id()),
            jump_to_system=lambda system_id: self._set_view_system(system_id, center=True),
        )
        self.system_graph_window.show()

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.on_tick)
        self.tick_timer.start(self._tick_timer_interval_ms())

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.on_render_frame)
        self.render_timer.start(16)

        self._sync_blue_squads()
        self.refresh_blue_roster()
        self.request_overview_refresh(force=True)
        self.recording.record_snapshot(force=True)

    def show_ship_status(self, ship_id: str) -> None:
        dialog = self._status_dialogs.get(ship_id)
        if dialog is None:
            dialog = ShipStatusDialog(
                self.runtime_view,
                ship_id,
                self.current_language,
                self.get_ship_fit_text,
                self._get_ship_locked_module_charge,
                self._set_ship_module_charge_lock,
                self._clear_ship_module_charge_lock,
                self._get_ship_module_target_mode,
                self._set_ship_module_target_mode,
                self._get_ship_module_manual_mode,
                self._set_ship_module_manual_mode,
                self._sync_ship_module_controls_to_matching_squad_fit,
                self._module_target_rules,
                self,
            )
            self._status_dialogs[ship_id] = dialog
            dialog.finished.connect(lambda _r, sid=ship_id: self._status_dialogs.pop(sid, None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _center_canvas_on_world(self, system_id: str, position: Vector2) -> None:
        normalized_system_id = str(system_id or "").strip()
        if normalized_system_id:
            self._set_view_system(normalized_system_id, center=False)
        self.canvas.center_on_world(position)

    def _log_user_action(self, action: str, **fields) -> None:
        self.application.log_user_action(
            action,
            network_mode=self.network_mode,
            controlled_team=self.controlled_team.value,
            **fields,
        )

    def _initialize_deployment_state(self) -> None:
        self.command_adapter.initialize_local_team_deployment(self.controlled_team)
        self.application.prepare()

    @staticmethod
    def _ship_is_gate_cloaked(ship, now: float | None = None) -> bool:
        cloak = getattr(getattr(ship, "nav", None), "cloak", None)
        if cloak is None or not bool(getattr(cloak, "active", False)):
            return False
        if now is not None and float(getattr(cloak, "expires_at", 0.0) or 0.0) <= float(now):
            return False
        return True

    def _ship_is_visible_gate_cloak_leader(self, ship) -> bool:
        if ship.team != self.controlled_team:
            return False
        leader_key = squad_key(ship.team, ship.squad_id)
        leader_id = str(self.runtime_view.world.squad_leaders.get(leader_key, "") or "")
        if leader_id:
            return leader_id == ship.ship_id
        squad_members = [
            candidate
            for candidate in self.runtime_view.world.ships.values()
            if candidate.team == ship.team and candidate.squad_id == ship.squad_id and candidate.vital.alive
        ]
        if not squad_members:
            return False
        squad_members.sort(key=lambda candidate: candidate.ship_id)
        return squad_members[0].ship_id == ship.ship_id

    def _is_ship_visible(self, ship_id: str) -> bool:
        ship = self.runtime_view.world.ships.get(str(ship_id))
        if ship is None or not ship.deployed:
            return False
        if not self._ship_is_gate_cloaked(ship, float(self.runtime_view.world.now)):
            return True
        return self._ship_is_visible_gate_cloak_leader(ship)

    def _guidance_target_for_squad(self, squad_id: str) -> Vector2 | None:
        scoped_key = squad_key(self.controlled_team, squad_id)
        leader = self._current_command_leader(squad_id)
        if leader is not None:
            target_ship_id = str(getattr(leader.nav, "command_target_ship_id", "") or "").strip()
            if target_ship_id:
                target_ship = self.runtime_view.world.ships.get(target_ship_id)
                if target_ship is not None and target_ship.vital.alive:
                    if hasattr(self, "canvas") and hasattr(self.canvas, "_ship_render_position"):
                        return self.canvas._ship_render_position(target_ship)
                    return Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
            target_structure_id = str(getattr(leader.nav, "command_target_structure_id", "") or "").strip()
            if target_structure_id:
                structure = self.runtime_view.world.structures.get(target_structure_id)
                if structure is not None:
                    return Vector2(structure.position.x, structure.position.y)
            command_target = getattr(leader.nav, "command_target", None)
            if command_target is not None:
                return Vector2(command_target.x, command_target.y)
        if self._team_has_fighter_squad(self.controlled_team, squad_id):
            fighters = [
                fighter
                for fighter in self.runtime_view.world.fighters.values()
                if fighter.team == self.controlled_team and fighter.squad_id == squad_id and fighter.vital.alive
            ]
            fighters.sort(key=lambda fighter: fighter.ship_id)
            for fighter in fighters:
                target_ship_id = str(getattr(fighter.nav, "command_target_ship_id", "") or "").strip()
                if target_ship_id:
                    target = self.runtime_view.world.combat_entity(target_ship_id)
                    if target is not None and target.vital.alive:
                        return Vector2(target.nav.position.x, target.nav.position.y)
                target_structure_id = str(getattr(fighter.nav, "command_target_structure_id", "") or "").strip()
                if target_structure_id:
                    structure = self.runtime_view.world.structures.get(target_structure_id)
                    if structure is not None:
                        return Vector2(structure.position.x, structure.position.y)
                command_target = getattr(fighter.nav, "command_target", None)
                if command_target is not None:
                    return Vector2(command_target.x, command_target.y)
        target = self._squad_guidance_targets.get(scoped_key)
        return target

    def _inducible_controlled_squad_ids(self) -> list[str]:
        squads = {
            s.squad_id
            for s in self.runtime_view.world.ships.values()
            if s.team == self.controlled_team and not s.deployed
        }
        return sorted(squads)

    def on_render_frame(self) -> None:
        self.canvas.update()

    def get_ship_fit_text(self, ship_id: str) -> str | None:
        view = self.application.query_service.ship_view(ship_id)
        return view.fit_text if view is not None else None

    def _get_ship_locked_module_charge(self, ship_id: str, module_id: str) -> str | None:
        view = self.application.query_service.ship_view(ship_id)
        return view.locked_module_charges.get(module_id) if view is not None else None

    def _charge_selection_entries(self, module_name: str, *, language: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        if module_supports_unloaded_charge(module_name):
            entries.append(("", QCoreApplication.translate("eve_sim", "None")))
        for charge_name in get_charge_option_values_for_module(module_name):
            entries.append((charge_name, get_type_display_name(charge_name, language=language)))
        return entries

    def _get_ship_module_manual_mode(self, ship_id: str, module_id: str) -> str:
        view = self.application.query_service.ship_view(ship_id)
        if view is None:
            return "auto"
        return normalize_module_manual_mode(view.module_manual_modes.get(module_id))

    def _get_ship_module_target_mode(self, ship_id: str, module_id: str) -> str:
        view = self.application.query_service.ship_view(ship_id)
        if view is None:
            return "auto"
        return normalize_module_target_mode(view.module_target_modes.get(module_id))

    def _module_target_rules(self, ship_id: str, module_id: str) -> tuple[tuple[str, ...], str]:
        view = self.application.query_service.module_target_rules(ship_id, module_id)
        return view.choices, view.default_mode

    def _set_ship_module_manual_mode(
        self,
        ship_id: str,
        module_id: str,
        mode: str,
        *,
        ignore_team_permission: bool = False,
    ) -> tuple[bool, str]:
        normalized_mode = normalize_module_manual_mode(mode)
        view = self.application.query_service.ship_view(ship_id)
        team = Team(view.team) if ignore_team_permission and view is not None else self.controlled_team
        result = self.command_adapter.set_ship_module_manual_mode(team, ship_id, module_id, normalized_mode)
        return result.accepted, result.message or normalized_mode


    def _install_map_definition(self, map_definition) -> None:
        self.command_adapter.install_local_map_definition(map_definition)
        self.application.prepare()
        self.runtime_view.refresh()
        self._refresh_system_view_controls()
        if hasattr(self, "system_graph_window"):
            self.system_graph_window.sync_from_world()
        if hasattr(self, "canvas") and self.prefs.zoom is None:
            QTimer.singleShot(0, self.canvas.fit_to_map)

    def _selected_anchor_system_id(self) -> str:
        members = [
            s
            for s in self.runtime_view.world.ships.values()
            if s.team == self.controlled_team and s.squad_id == self.ui_state.selected_squad and s.vital.alive
        ]
        if not members:
            return ""
        return str(getattr(members[0].nav, "system_id", "") or "")

    def _refresh_system_view_controls(self) -> None:
        if not hasattr(self, "system_view_combo"):
            return
        map_definition = getattr(self.runtime_view.world, "map_definition", None)
        current_system_id = str(self._view_system_id or self.system_view_combo.currentData() or "").strip()
        self.system_view_combo.blockSignals(True)
        self.system_view_combo.clear()
        if map_definition is not None:
            for system in getattr(map_definition, "systems", []):
                label = str(getattr(system, "name", "") or getattr(system, "system_id", "") or "")
                self.system_view_combo.addItem(label, str(system.system_id))
        if self.system_view_combo.count() > 0:
            idx = self.system_view_combo.findData(current_system_id)
            self.system_view_combo.setCurrentIndex(0 if idx < 0 else idx)
            self._view_system_id = str(self.system_view_combo.currentData() or "").strip()
            if hasattr(self, "canvas"):
                self.canvas.current_view_system_id = self._view_system_id
        self.system_view_combo.setEnabled(self.system_view_combo.count() > 0)
        self.btn_view_system.setEnabled(self.system_view_combo.count() > 0)
        self.btn_fit_map.setEnabled(self.runtime_view.world.map_definition is not None)
        self.system_view_combo.blockSignals(False)

    def _current_view_system_id(self) -> str:
        if self._view_system_id:
            return str(self._view_system_id)
        if hasattr(self, "system_view_combo"):
            value = str(self.system_view_combo.currentData() or "").strip()
            if value:
                return value
        map_definition = getattr(self.runtime_view.world, "map_definition", None)
        if map_definition is None or not getattr(map_definition, "systems", None):
            return ""
        anchor_system = self._selected_anchor_system_id()
        if anchor_system:
            return anchor_system
        return str(getattr(map_definition.systems[0], "system_id", "") or "")

    def _set_view_system(self, system_id: str, *, center: bool = True) -> None:
        normalized = str(system_id or "").strip()
        if not normalized:
            return
        self._view_system_id = normalized
        if hasattr(self, "canvas"):
            self.canvas.current_view_system_id = normalized
        if hasattr(self, "system_view_combo"):
            idx = self.system_view_combo.findData(normalized)
            if idx >= 0 and self.system_view_combo.currentIndex() != idx:
                self.system_view_combo.blockSignals(True)
                self.system_view_combo.setCurrentIndex(idx)
                self.system_view_combo.blockSignals(False)
        if center:
            self.canvas.focus_system(normalized)
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _on_system_view_changed(self, _index: int) -> None:
        system_id = str(self.system_view_combo.currentData() or "").strip()
        if not system_id:
            return
        self._set_view_system(system_id, center=True)

    def _focus_selected_system(self) -> None:
        system_id = str(self.system_view_combo.currentData() or "").strip()
        if not system_id:
            return
        self._set_view_system(system_id, center=True)

    def _fit_view_to_map(self) -> None:
        self.canvas.fit_to_map()

    def _set_ship_module_target_mode(
        self,
        ship_id: str,
        module_id: str,
        mode: str,
        *,
        ignore_team_permission: bool = False,
    ) -> tuple[bool, str]:
        normalized_mode = normalize_module_target_mode(mode)
        view = self.application.query_service.ship_view(ship_id)
        team = Team(view.team) if ignore_team_permission and view is not None else self.controlled_team
        result = self.command_adapter.set_ship_module_target_mode(team, ship_id, module_id, normalized_mode)
        return result.accepted, result.message or normalized_mode


    def _sync_ship_module_controls_to_matching_squad_fit(
        self,
        ship_id: str,
        module_id: str,
        mode: str,
        target_mode: str,
        *,
        ignore_team_permission: bool = False,
    ) -> tuple[bool, str]:
        normalized_mode = normalize_module_manual_mode(mode)
        normalized_target_mode = normalize_module_target_mode(target_mode)
        view = self.application.query_service.ship_view(ship_id)
        team = Team(view.team) if ignore_team_permission and view is not None else self.controlled_team
        result = self.command_adapter.sync_squad_module_controls(
            team,
            ship_id,
            module_id,
            normalized_mode,
            normalized_target_mode,
        )
        return result.accepted, result.message or QCoreApplication.translate("eve_sim", "Sync queued")


    def _set_ship_module_charge_lock(
        self,
        ship_id: str,
        module_id: str,
        ammo_name: str,
        *,
        ignore_team_permission: bool = False,
    ) -> tuple[bool, str]:
        view = self.application.query_service.ship_view(ship_id)
        if view is None:
            return False, QCoreApplication.translate("eve_sim", "Ship not found")
        team = Team(view.team)
        if not ignore_team_permission and not self._is_ammo_configurable_team(team):
            return False, QCoreApplication.translate("eve_sim", "Cannot modify this ship's charge in current mode")
        result = self.command_adapter.set_ship_module_charge_lock(
            team,
            ship_id,
            module_id,
            str(ammo_name or "").strip(),
        )
        if not result.accepted:
            return False, result.message or QCoreApplication.translate("eve_sim", "Charge command was rejected")
        return True, result.message or QCoreApplication.translate("eve_sim", "Charge change queued")


    def _clear_ship_module_charge_lock(
        self,
        ship_id: str,
        module_id: str,
        *,
        ignore_team_permission: bool = False,
    ) -> tuple[bool, str]:
        view = self.application.query_service.ship_view(ship_id)
        if view is None:
            return False, QCoreApplication.translate("eve_sim", 'Ship not found')
        team = Team(view.team)
        if not ignore_team_permission and not self._is_ammo_configurable_team(team):
            return False, QCoreApplication.translate("eve_sim", "Cannot modify this ship's charge in current mode")
        result = self.command_adapter.clear_ship_module_charge_lock(team, ship_id, module_id)
        if not result.accepted:
            return False, result.message
        return True, result.message or QCoreApplication.translate("eve_sim", 'Charge lock clear queued')

    def current_language(self) -> str:
        return normalize_language(self.prefs.language, "en_US")

    def _ui_text(self, text: str) -> str:
        if not self.current_language().lower().startswith("zh"):
            return QCoreApplication.translate("eve_sim", text)
        translations = {
            "Fighter Abilities": "舰载机技能",
            "{squad} Drone Attack {ship}": "{squad} 无人机攻击 {ship}",
            "{squad} Fighter Attack {ship}": "{squad} 舰载机攻击 {ship}",
            "Battle Report": "战报",
            "Scenario": "场景",
            "Duration": "时长",
            "Total Damage By Team": "各队总伤害",
            "Rep Applied By Team": "各队维修量",
            "Jam Uptime By Target": "目标被干扰时长",
            "Burst Coverage By Effect": "会战加成覆盖",
            "Ship Deaths": "舰船损失",
            "Raw JSON": "原始 JSON",
            "none": "无",
        }
        return translations.get(text, QCoreApplication.translate("eve_sim", text))

    def _display_type_name(self, type_name: str) -> str:
        return get_type_display_name(str(type_name or ""), language=self.current_language())

    def _display_ship_type(self, ship_name: str, *, language: str) -> str:
        cache_key = (str(language or ""), str(ship_name or ""))
        cached = self._ship_type_display_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = get_type_display_name(ship_name, language=language)
        self._ship_type_display_cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _display_structure_type(kind: str) -> str:
        normalized = str(kind or "").strip().upper()
        if normalized == "STARGATE":
            return QCoreApplication.translate("eve_sim", "Stargate")
        return QCoreApplication.translate("eve_sim", "Structure")

    def _display_structure_name(self, structure) -> str:
        kind = str(getattr(structure, "kind", "") or "").strip().upper()
        if kind == "STARGATE":
            system_id = str(getattr(structure, "system_id", "") or "").strip()
            map_definition = getattr(self.runtime_view.world, "map_definition", None)
            if map_definition is not None and system_id:
                system = map_definition.system_by_id(system_id)
                if system is not None:
                    system_name = str(getattr(system, "name", "") or system_id).strip()
                    if system_name:
                        return QCoreApplication.translate("eve_sim", "{system} Stargate").format(system=system_name)
            return QCoreApplication.translate("eve_sim", "Stargate")
        display_name = str(getattr(structure, "display_name", "") or getattr(structure, "structure_id", "") or "").strip()
        return display_name or self._display_structure_type(kind)

    def end_battle_and_save_recording(self) -> None:
        default_path = self.recording.default_path()
        path, _filter = QFileDialog.getSaveFileName(
            self,
            QCoreApplication.translate("eve_sim", "Save Battle Recording"),
            str(default_path),
            QCoreApplication.translate("eve_sim", "Replay Files (*.replay.json);;JSON Files (*.json);;All Files (*)"),
        )
        if not path:
            return
        try:
            saved_path = self.recording.save(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                QCoreApplication.translate("eve_sim", "Save Failed"),
                str(exc),
            )
            return
        self.recording.stop()
        self.tick_timer.stop()
        self.act_end_battle.setEnabled(False)
        self.status.setText(
            QCoreApplication.translate("eve_sim", "Battle ended. Recording saved: {path}").format(path=str(saved_path))
        )
        QMessageBox.information(
            self,
            QCoreApplication.translate("eve_sim", "Battle Recording Saved"),
            QCoreApplication.translate("eve_sim", "Saved recording to:\n{path}").format(path=str(saved_path)),
        )

    def open_replay_recording(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate("eve_sim", "Open Battle Recording"),
            str((Path("logs") / "replays").resolve()),
            QCoreApplication.translate("eve_sim", "Replay Files (*.replay.json *.json);;All Files (*)"),
        )
        if not path:
            return
        try:
            player = ReplayPlayer.from_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                QCoreApplication.translate("eve_sim", "Open Failed"),
                str(exc),
            )
            return
        if player.snapshot_count <= 0:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("eve_sim", "Replay Unavailable"),
                QCoreApplication.translate("eve_sim", "This recording has no world snapshots to replay."),
            )
            return
        dialog = ReplayPlaybackDialog(player, self.ui_cfg, self.current_language, self)
        dialog.resize(min(self.ui_cfg.width + 120, 1400), min(self.ui_cfg.height + 140, 920))
        dialog.exec()

    def open_battle_report_from_recording(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate("eve_sim", "Open Battle Recording"),
            str((Path("logs") / "replays").resolve()),
            QCoreApplication.translate("eve_sim", "Replay Files (*.replay.json *.json);;All Files (*)"),
        )
        if not path:
            return
        try:
            recorder = ReplayRecorder.load(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                QCoreApplication.translate("eve_sim", "Open Failed"),
                str(exc),
            )
            return
        duration = recorder.metadata.get("duration_s")
        if duration is None and recorder.frame_count:
            duration = recorder.duration_s
        report = BattleReportService().build(
            recorder.scenario_id,
            recorder.events,
            duration_s=float(duration) if duration is not None else None,
        )
        self._show_battle_report_dialog(report.to_dict())

    def _show_battle_report_dialog(self, report: dict[str, Any]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui_text("Battle Report"))
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(format_battle_report(report, self._ui_text))
        layout.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(760, 620)
        dialog.exec()

    def _create_menu(self) -> None:
        self.menu_overview = self.menuBar().addMenu(QCoreApplication.translate("eve_sim", 'Overview'))
        self.act_overview_filter = QAction(QCoreApplication.translate("eve_sim", 'Filters...'), self)
        self.act_overview_filter.triggered.connect(self.open_overview_options)
        self.menu_overview.addAction(self.act_overview_filter)

        self.act_overview_reset = QAction(QCoreApplication.translate("eve_sim", 'Reset Filters'), self)
        self.act_overview_reset.triggered.connect(self.reset_overview_options)
        self.menu_overview.addAction(self.act_overview_reset)

        self.menu_battle = self.menuBar().addMenu(QCoreApplication.translate("eve_sim", "Battle"))
        self.act_end_battle = QAction(QCoreApplication.translate("eve_sim", "End Battle and Save Recording"), self)
        self.act_end_battle.triggered.connect(self.end_battle_and_save_recording)
        self.menu_battle.addAction(self.act_end_battle)

        self.act_open_replay = QAction(QCoreApplication.translate("eve_sim", "Open Recording Replay"), self)
        self.act_open_replay.triggered.connect(self.open_replay_recording)
        self.menu_battle.addAction(self.act_open_replay)

        self.act_open_report = QAction(QCoreApplication.translate("eve_sim", "Battle Report from Recording"), self)
        self.act_open_report.triggered.connect(self.open_battle_report_from_recording)
        self.menu_battle.addAction(self.act_open_report)

    def _build_left_panel(self) -> QWidget:
        side = QWidget(self)
        side_layout = QVBoxLayout(side)
        side.setMinimumWidth(520)

        header = QHBoxLayout()
        self.lbl_selected_squad = QLabel(QCoreApplication.translate("eve_sim", 'Selected Squad'))
        header.addWidget(self.lbl_selected_squad)
        self.squad_combo = QComboBox()
        self.squad_combo.setEditable(False)
        self.squad_combo.currentTextChanged.connect(self.on_selected_squad_changed)
        header.addWidget(self.squad_combo, 1)
        self.lbl_controlled_side = QLabel("")
        header.addWidget(self.lbl_controlled_side)
        self.lbl_controlled_team_value = QLabel("")
        header.addWidget(self.lbl_controlled_team_value)
        self.btn_switch_controlled_team = QPushButton("")
        self.btn_switch_controlled_team.clicked.connect(self.toggle_local_controlled_team)
        header.addWidget(self.btn_switch_controlled_team)
        side_layout.addLayout(header)

        system_row = QHBoxLayout()
        self.lbl_view_system = QLabel(QCoreApplication.translate("eve_sim", "View System"))
        system_row.addWidget(self.lbl_view_system)
        self.system_view_combo = QComboBox(self)
        self.system_view_combo.setMinimumWidth(180)
        self.system_view_combo.currentIndexChanged.connect(self._on_system_view_changed)
        system_row.addWidget(self.system_view_combo, 1)
        self.btn_view_system = QPushButton(QCoreApplication.translate("eve_sim", "Center"))
        self.btn_view_system.clicked.connect(self._focus_selected_system)
        system_row.addWidget(self.btn_view_system)
        self.btn_fit_map = QPushButton(QCoreApplication.translate("eve_sim", "Fit Map"))
        self.btn_fit_map.clicked.connect(self._fit_view_to_map)
        system_row.addWidget(self.btn_fit_map)
        side_layout.addLayout(system_row)

        leader_limit_row = QHBoxLayout()
        self.lbl_leader_speed_limit = QLabel(QCoreApplication.translate("eve_sim", 'Leader Max Speed (0=Unlimited):'))
        leader_limit_row.addWidget(self.lbl_leader_speed_limit)
        self.spin_leader_speed_limit = QDoubleSpinBox(self)
        self.spin_leader_speed_limit.setDecimals(1)
        self.spin_leader_speed_limit.setRange(0.0, 1_000_000.0)
        self.spin_leader_speed_limit.setSingleStep(50.0)
        self.spin_leader_speed_limit.setValue(0.0)
        leader_limit_row.addWidget(self.spin_leader_speed_limit, 1)
        side_layout.addLayout(leader_limit_row)

        buttons_top2 = QHBoxLayout()
        self.btn_propulsion_toggle = QPushButton(QCoreApplication.translate("eve_sim", 'Click to Enable Prop'))
        buttons_top2.addWidget(self.btn_propulsion_toggle)
        self.btn_clear_focus = QPushButton(QCoreApplication.translate("eve_sim", 'Clear Focus Targets'))
        buttons_top2.addWidget(self.btn_clear_focus)
        side_layout.addLayout(buttons_top2)

        self.fighter_ability_row = QWidget(self)
        fighter_ability_layout = QHBoxLayout(self.fighter_ability_row)
        fighter_ability_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_fighter_abilities = QLabel(self._ui_text("Fighter Abilities"))
        fighter_ability_layout.addWidget(self.lbl_fighter_abilities)
        self.fighter_ability_buttons_layout = QHBoxLayout()
        fighter_ability_layout.addLayout(self.fighter_ability_buttons_layout, 1)
        side_layout.addWidget(self.fighter_ability_row)
        self.fighter_ability_row.setVisible(False)

        ammo_layout = QVBoxLayout()
        ammo_row1 = QHBoxLayout()
        self.lbl_freq_charge_module = QLabel(QCoreApplication.translate("eve_sim", 'Common Charge-Loadable Modules (all, sorted by count):'))
        ammo_row1.addWidget(self.lbl_freq_charge_module)
        self.charge_module_combo = QComboBox()
        self.charge_module_combo.setMinimumWidth(260)
        ammo_row1.addWidget(self.charge_module_combo, 1)
        ammo_layout.addLayout(ammo_row1)

        ammo_row2 = QHBoxLayout()
        self.lbl_ammo = QLabel(QCoreApplication.translate("eve_sim", 'Ammo:'))
        ammo_row2.addWidget(self.lbl_ammo)
        self.ammo_combo = QComboBox()
        self.ammo_combo.setMinimumWidth(260)
        ammo_row2.addWidget(self.ammo_combo, 1)
        self.apply_ammo_btn = QPushButton(QCoreApplication.translate("eve_sim", 'Apply to Fleet'))
        ammo_row2.addWidget(self.apply_ammo_btn)
        ammo_layout.addLayout(ammo_row2)
        side_layout.addLayout(ammo_layout)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_overview_tab(), QCoreApplication.translate("eve_sim", 'Overview'))
        self.tabs.addTab(self._build_fleet_tab(), QCoreApplication.translate("eve_sim", 'Fleet'))
        side_layout.addWidget(self.tabs, 1)

        self.status = QLabel(f"{QCoreApplication.translate("eve_sim", 'Tick')}: 0")
        side_layout.addWidget(self.status)

        self.btn_propulsion_toggle.clicked.connect(self.toggle_selected_squad_propulsion)
        self.btn_clear_focus.clicked.connect(self.clear_focus_targets)
        self.spin_leader_speed_limit.valueChanged.connect(self.on_selected_squad_leader_speed_limit_changed)
        self.charge_module_combo.currentTextChanged.connect(self._on_charge_module_changed)
        self.apply_ammo_btn.clicked.connect(self._apply_selected_ammo)
        self._refresh_common_charge_modules()
        self._refresh_selected_squad_leader_speed_limit()
        self._refresh_propulsion_button_text()
        self._refresh_fighter_ability_controls()
        self._refresh_controlled_team_widgets()
        self._refresh_system_view_controls()
        return side

    def _get_squad_propulsion_state(self, squad_id: str) -> bool:
        return self.application.query_service.squad_view(self.controlled_team, squad_id).propulsion_active

    def _team_display_text(self, team: Team) -> str:
        return QCoreApplication.translate("eve_sim", "BLUE") if team == Team.BLUE else QCoreApplication.translate("eve_sim", "RED")

    def _refresh_controlled_team_widgets(self) -> None:
        if not hasattr(self, "lbl_controlled_side"):
            return
        is_local_mode = self.network_mode == "local"
        self.lbl_controlled_side.setVisible(is_local_mode)
        self.lbl_controlled_team_value.setVisible(is_local_mode)
        self.btn_switch_controlled_team.setVisible(is_local_mode)
        if not is_local_mode:
            return
        self.lbl_controlled_side.setText(QCoreApplication.translate("eve_sim", 'Controlled Side'))
        self.lbl_controlled_team_value.setText(self._team_display_text(self.controlled_team))
        next_team = Team.RED if self.controlled_team == Team.BLUE else Team.BLUE
        self.btn_switch_controlled_team.setText(
            QCoreApplication.translate("eve_sim", 'Switch to {team}').format(team=self._team_display_text(next_team))
        )

    def _clear_selected_enemy_if_not_enemy(self) -> None:
        target_id = str(self.ui_state.selected_enemy_target or "").strip()
        if not target_id:
            return
        target = self.runtime_view.world.ships.get(target_id)
        if target is not None and target.team != self.controlled_team and target.deployed:
            return
        self.ui_state.selected_enemy_target = None
        self.canvas.selected_enemy_target = None

    def toggle_local_controlled_team(self) -> None:
        if self.network_mode != "local":
            return
        self.controlled_team = Team.RED if self.controlled_team == Team.BLUE else Team.BLUE
        self._log_user_action("switch_controlled_team", team=self.controlled_team.value)
        self._refresh_controlled_team_widgets()
        self._clear_selected_enemy_if_not_enemy()
        self._sync_blue_squads()
        self.refresh_blue_roster()
        self._refresh_common_charge_modules()
        self.overview.clearSelection()
        self.blue_roster.clearSelection()
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _current_command_leader(self, squad_id: str):
        key = squad_key(self.controlled_team, squad_id)
        leader_id = self.runtime_view.world.squad_leaders.get(key)
        if leader_id:
            leader = self.runtime_view.world.ships.get(str(leader_id))
            if leader is not None and leader.vital.alive and leader.team == self.controlled_team and leader.squad_id == squad_id:
                return leader
        for ship in self.runtime_view.world.ships.values():
            if ship.team == self.controlled_team and ship.squad_id == squad_id and ship.vital.alive:
                return ship
        return None

    def _range_to_target_from_squad(self, squad_id: str, *, target_ship_id: str | None = None, target_structure_id: str | None = None) -> float:
        leader = self._current_command_leader(squad_id)
        source_position: Vector2 | None = None
        source_radius = 0.0
        if leader is None:
            fighters = [
                fighter
                for fighter in self.runtime_view.world.fighters.values()
                if fighter.team == self.controlled_team and fighter.squad_id == squad_id and fighter.vital.alive
            ]
            if not fighters:
                return 0.0
            source_position = Vector2(
                sum(fighter.nav.position.x for fighter in fighters) / len(fighters),
                sum(fighter.nav.position.y for fighter in fighters) / len(fighters),
            )
            source_radius = max(max(0.0, float(getattr(fighter.nav, "radius", 0.0) or 0.0)) for fighter in fighters)
        else:
            source_position = Vector2(leader.nav.position.x, leader.nav.position.y)
            source_radius = max(0.0, float(getattr(leader.nav, "radius", 0.0) or 0.0))
        target_position: Vector2 | None = None
        target_radius = 0.0
        if target_ship_id:
            target_ship = self.runtime_view.world.combat_entity(str(target_ship_id))
            if target_ship is not None:
                target_position = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
                target_radius = max(0.0, float(getattr(target_ship.nav, "radius", 0.0) or 0.0))
        elif target_structure_id:
            structure = self.runtime_view.world.structures.get(str(target_structure_id))
            if structure is not None:
                target_position = Vector2(structure.position.x, structure.position.y)
                target_radius = max(0.0, float(getattr(structure, "radius", 0.0) or 0.0))
        if target_position is None:
            return 0.0
        return max(0.0, source_position.distance_to(target_position) - target_radius - source_radius)

    def _refresh_propulsion_button_text(self) -> None:
        active = self._get_squad_propulsion_state(self.ui_state.selected_squad)
        self.btn_propulsion_toggle.setText(QCoreApplication.translate("eve_sim", 'Click to Disable Prop') if active else QCoreApplication.translate("eve_sim", 'Click to Enable Prop'))

    @staticmethod
    def _clear_layout_widgets(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fighter_ability_label(self, ability: FighterAbilityProfile) -> str:
        kind = str(getattr(ability, "kind", "") or "").strip().lower()
        if self.current_language().lower().startswith("zh"):
            kind_labels = {
                "mwd": "加速",
                "heavy_attack": "强力攻击",
                "ewar": "电子战",
                "support": "支援",
                "normal_attack": "普通攻击",
            }
            if kind in kind_labels:
                return kind_labels[kind]
        name = str(getattr(ability, "name", "") or "").strip()
        if name:
            return self._display_type_name(name)
        return kind.replace("_", " ").title() or str(getattr(ability, "ability_id", "") or "Ability")

    def _fighter_ability_button_state(self, squad: str, ability: FighterAbilityProfile) -> tuple[str, str, bool]:
        label = self._fighter_ability_label(ability)
        zh = self.current_language().lower().startswith("zh")

        def word(en: str, cn: str) -> str:
            return cn if zh else en

        fighters = [
            fighter
            for fighter in self.runtime_view.world.fighters.values()
            if fighter.team == self.controlled_team
            and fighter.squad_id == squad
            and fighter.vital.alive
            and fighter.connected
            and any(item.ability_id == ability.ability_id for item in fighter.definition.abilities)
        ]
        tooltip = str(getattr(ability, "effect_name", "") or getattr(ability, "name", "") or getattr(ability, "kind", "") or "")
        if not fighters:
            return label, tooltip, False

        ability_id = str(ability.ability_id)
        kind = str(getattr(ability, "kind", "") or "").strip().lower()
        if kind == "mwd":
            ready = [
                fighter
                for fighter in fighters
                if ability_id not in fighter.pending_manual_abilities
                and max(0.0, float(fighter.mwd_active_timer or 0.0)) <= 0.0
                and max(0.0, float(fighter.mwd_cooldown_timer or 0.0)) <= 0.0
            ]
            if ready:
                return f"{label} ({word('Ready', '就绪')} {len(ready)}/{len(fighters)})", tooltip, True
            active_left = max((max(0.0, float(fighter.mwd_active_timer or 0.0)) for fighter in fighters), default=0.0)
            if active_left > 0.0:
                return f"{label} ({word('Active', '持续')} {active_left:.0f}s)", tooltip, False
            cooldown_left = min(
                (max(0.0, float(fighter.mwd_cooldown_timer or 0.0)) for fighter in fighters if max(0.0, float(fighter.mwd_cooldown_timer or 0.0)) > 0.0),
                default=0.0,
            )
            if cooldown_left > 0.0:
                return f"{label} (CD {cooldown_left:.0f}s)", tooltip, False
            return label, tooltip, False

        ready_count = 0
        pending_count = 0
        no_target_count = 0
        locking_count = 0
        range_count = 0
        no_ammo_count = 0
        reload_left: list[float] = []
        cycle_left: list[float] = []
        ammo_total = 0
        ammo_capacity_total = 0
        for fighter in fighters:
            fighter_ability = next((item for item in fighter.definition.abilities if item.ability_id == ability_id), ability)
            capacity = max(0, int(getattr(fighter_ability, "ammo_capacity", 0) or 0))
            if capacity > 0:
                ammo_capacity_total += capacity
                ammo_total += max(0, int(fighter.ability_ammo_remaining.get(ability_id, capacity)))
            if ability_id in fighter.pending_manual_abilities:
                pending_count += 1
                continue
            target_id = str(getattr(fighter, "target_id", "") or "").strip()
            target = self.runtime_view.world.combat_entity(target_id) if target_id else None
            if target is None or not target.vital.alive or target.team == fighter.team:
                no_target_count += 1
                continue
            if target.ship_id not in fighter.combat.lock_targets:
                locking_count += 1
                continue
            distance = fighter.nav.position.distance_to(target.nav.position)
            max_range = max(
                float(getattr(fighter_ability, "optimal_range_m", 0.0) or 0.0)
                + max(0.0, float(getattr(fighter_ability, "falloff_m", 0.0) or 0.0)) * 3.0,
                float(getattr(getattr(fighter_ability, "ewar", None), "optimal_range_m", 0.0) or 0.0)
                + max(0.0, float(getattr(getattr(fighter_ability, "ewar", None), "falloff_m", 0.0) or 0.0)) * 3.0,
            )
            if max_range > 0.0 and distance > max_range:
                range_count += 1
                continue
            reload_timer = max(0.0, float(fighter.ability_reload_timers.get(ability_id, 0.0) or 0.0))
            if reload_timer > 0.0:
                reload_left.append(reload_timer)
                continue
            cycle_timer = max(0.0, float(fighter.ability_cycle_timers.get(ability_id, 0.0) or 0.0))
            if cycle_timer > 0.0:
                cycle_left.append(cycle_timer)
                continue
            if capacity > 0 and int(fighter.ability_ammo_remaining.get(ability_id, capacity)) <= 0:
                no_ammo_count += 1
                continue
            ready_count += 1

        suffixes: list[str] = []
        if ammo_capacity_total > 0:
            suffixes.append(f"x{ammo_total}")
        if ready_count > 0:
            suffixes.append(f"{word('Ready', '就绪')} {ready_count}/{len(fighters)}")
            return f"{label} ({' | '.join(suffixes)})", tooltip, True
        if pending_count:
            suffixes.append(word("Pending", "待释放"))
        elif no_target_count == len(fighters):
            suffixes.append(word("No Target", "无目标"))
        elif locking_count:
            suffixes.append(word("Locking", "锁定中"))
        elif range_count:
            suffixes.append(word("Out of Range", "射程外"))
        elif reload_left:
            suffixes.append(f"{word('Reload', '装填')} {min(reload_left):.0f}s")
        elif cycle_left:
            suffixes.append(f"CD {min(cycle_left):.0f}s")
        elif no_ammo_count:
            suffixes.append(word("No Ammo", "弹药 0"))
        return f"{label} ({' | '.join(suffixes)})" if suffixes else label, tooltip, False

    def _refresh_fighter_ability_controls(self) -> None:
        if not hasattr(self, "fighter_ability_row") or not hasattr(self, "fighter_ability_buttons_layout"):
            return
        self._clear_layout_widgets(self.fighter_ability_buttons_layout)
        squad = self.ui_state.selected_squad
        abilities = self._selected_fighter_squad_abilities(squad) if self._is_fighter_squad(squad) else []
        self.fighter_ability_row.setVisible(bool(abilities))
        if not abilities:
            return
        for ability in abilities:
            text, tooltip, enabled = self._fighter_ability_button_state(squad, ability)
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, ability_id=ability.ability_id: self.activate_selected_fighter_ability(ability_id))
            button.setEnabled(enabled)
            self.fighter_ability_buttons_layout.addWidget(button)
        self.fighter_ability_buttons_layout.addStretch(1)

    def _is_ammo_configurable_team(self, team: Team) -> bool:
        return team == self.controlled_team

    def toggle_selected_squad_propulsion(self) -> None:
        squad = self.ui_state.selected_squad
        new_state = not self._get_squad_propulsion_state(squad)
        self._log_user_action("toggle_propulsion", squad=squad, enabled=new_state)

        self.command_adapter.propulsion(self.controlled_team, squad, new_state)
        self._refresh_propulsion_button_text()

    def _refresh_common_charge_modules(self) -> None:
        fit_texts = list(self.application.query_service.team_fit_texts(self.controlled_team))
        current = self.charge_module_combo.currentText()
        try:
            charge_modules = get_common_chargeable_modules(fit_texts, usage_threshold=0.0, language=self.current_language())
        except Exception as exc:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("eve_sim", "Ammo Configuration"),
                QCoreApplication.translate("eve_sim", "Failed to resolve chargeable modules: {error}").format(error=display_user_error(exc)),
            )
            charge_modules = []
        self.charge_module_combo.blockSignals(True)
        self.charge_module_combo.clear()
        self.charge_module_combo.addItems(charge_modules)
        if current and current in charge_modules:
            self.charge_module_combo.setCurrentText(current)
        self.charge_module_combo.blockSignals(False)

        for module_name in charge_modules:
            try:
                ammo_entries = self._charge_selection_entries(module_name, language=self.current_language())
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    QCoreApplication.translate("eve_sim", "Ammo Configuration"),
                    QCoreApplication.translate("eve_sim", "Failed to resolve ammo options: {error}").format(error=display_user_error(exc)),
                )
                return
            if not ammo_entries:
                continue
            selected = self._charge_module_ammo_selection.get(module_name)
            valid_values = {value for value, _label in ammo_entries}
            if selected is None or selected not in valid_values:
                default_entry = ammo_entries[0]
                self._charge_module_ammo_selection[module_name] = default_entry[0]

        self._on_charge_module_changed(self.charge_module_combo.currentText())

    def _on_charge_module_changed(self, module_name: str) -> None:
        self.ammo_combo.blockSignals(True)
        self.ammo_combo.clear()
        if not module_name:
            self.ammo_combo.blockSignals(False)
            return
        try:
            ammo_entries = self._charge_selection_entries(module_name, language=self.current_language())
        except Exception as exc:
            self.ammo_combo.blockSignals(False)
            QMessageBox.warning(
                self,
                QCoreApplication.translate("eve_sim", "Ammo Configuration"),
                QCoreApplication.translate("eve_sim", "Failed to resolve ammo options: {error}").format(error=display_user_error(exc)),
            )
            return
        for value, label in ammo_entries:
            self.ammo_combo.addItem(label, value)
        if not ammo_entries:
            self.ammo_combo.blockSignals(False)
            return
        selected = self._charge_module_ammo_selection.get(module_name)
        valid_values = {value for value, _label in ammo_entries}
        if selected is None or selected not in valid_values:
            selected = ammo_entries[0][0]
            self._charge_module_ammo_selection[module_name] = selected
        idx = self.ammo_combo.findData(selected)
        self.ammo_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.ammo_combo.blockSignals(False)

    def _apply_selected_ammo(self) -> None:
        lang = self.current_language()
        module_name = self.charge_module_combo.currentText().strip()
        ammo_name = str(self.ammo_combo.currentData() or "").strip()
        if not module_name:
            return

        self._charge_module_ammo_selection[module_name] = ammo_name
        result = self.command_adapter.set_fleet_module_charge(
            self.controlled_team,
            module_name,
            ammo_name,
        )
        if not result.accepted:
            QMessageBox.warning(
                self,
                QCoreApplication.translate("eve_sim", "Ammo Configuration"),
                result.message or QCoreApplication.translate("eve_sim", "Charge command was rejected"),
            )
            return

        self.request_overview_refresh(force=True)
        self.canvas.update()
        QMessageBox.information(
            self,
            QCoreApplication.translate("eve_sim", "Ammo Configuration"),
            QCoreApplication.translate(
                "eve_sim",
                "Queued {module} charge change to {ammo} for the controlled fleet.",
            ).format(
                module=module_name,
                ammo=(
                    get_type_display_name(ammo_name, language=lang)
                    if ammo_name
                    else QCoreApplication.translate("eve_sim", "None")
                ),
            ),
        )
        self._log_user_action(
            "apply_ammo",
            module=module_name,
            ammo=ammo_name,
        )


    def _build_overview_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self.overview = QTableView(self)
        self.overview_model = OverviewTableModel(
            selected_squad_getter=lambda: self.ui_state.selected_squad,
            selected_target_getter=lambda: self.ui_state.selected_enemy_target,
            language_getter=self.current_language,
            controlled_team_getter=lambda: self.controlled_team,
        )
        self.overview_proxy = OverviewFilterProxyModel(lambda: self.prefs, lambda: self.controlled_team, self)
        self.overview_proxy.setSourceModel(self.overview_model)
        self.overview.setModel(self.overview_proxy)
        self.overview.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.overview.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.overview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.overview.customContextMenuRequested.connect(self.show_overview_menu)
        self.overview.selectionModel().selectionChanged.connect(self._on_overview_selection_changed)
        self.overview.doubleClicked.connect(self._on_overview_double_clicked)
        self.overview.setAlternatingRowColors(True)
        self.overview.setWordWrap(False)
        self.overview.setSortingEnabled(True)
        self.overview.verticalHeader().setVisible(False)
        self.overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.overview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in range(2, 4):
            self.overview.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.overview, 1)
        return page

    def _build_fleet_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self.lbl_fleet_tip = QLabel(QCoreApplication.translate("eve_sim", 'Multi-select ships to assign squad; edit name to create squad'))
        layout.addWidget(self.lbl_fleet_tip)

        self.blue_roster = QTableView(self)
        self.blue_roster_model = BlueRosterTableModel(self.current_language)
        self.blue_roster.setModel(self.blue_roster_model)
        self.blue_roster.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.blue_roster.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.blue_roster.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.blue_roster.customContextMenuRequested.connect(self.show_blue_roster_menu)
        self.blue_roster.selectionModel().selectionChanged.connect(self._on_blue_roster_selection_changed)
        self.blue_roster.setWordWrap(False)
        self.blue_roster.verticalHeader().setVisible(False)
        self.blue_roster.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.blue_roster.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in (2, 3, 4, 5):
            self.blue_roster.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.blue_roster, 1)

        controls = QHBoxLayout()
        self.assign_squad_edit = QLineEdit(self.ui_state.selected_squad)
        self.btn_assign = QPushButton(QCoreApplication.translate("eve_sim", 'Assign Selected Ships'))
        self.lbl_target_squad = QLabel(QCoreApplication.translate("eve_sim", 'Target Squad'))
        controls.addWidget(self.lbl_target_squad)
        controls.addWidget(self.assign_squad_edit, 1)
        controls.addWidget(self.btn_assign)
        layout.addLayout(controls)

        self.btn_assign.clicked.connect(self.assign_blue_ships)
        return page

    def open_overview_options(self) -> None:
        dlg = OverviewOptionsDialog(self.prefs, self.current_language(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_prefs = dlg.to_preferences(self.prefs)
            new_prefs.selected_squad = self.ui_state.selected_squad
            new_prefs.zoom = self.canvas.zoom
            self.prefs = new_prefs
            self.store.save(self.prefs)
            self.overview_proxy.apply_preferences()
            self.request_overview_refresh(force=True)

    def reset_overview_options(self) -> None:
        selected = self.ui_state.selected_squad
        zoom = self.canvas.zoom
        self.prefs = UiPreferences(
            selected_squad=selected,
            selected_map_id=self.prefs.selected_map_id,
            zoom=zoom,
            language=self.prefs.language,
            engine_tick_rate=self.prefs.engine_tick_rate,
            engine_physics_substeps=self.prefs.engine_physics_substeps,
            engine_lockstep=self.prefs.engine_lockstep,
            engine_detailed_logging=self.prefs.engine_detailed_logging,
            engine_hotspot_logging=self.prefs.engine_hotspot_logging,
            engine_detail_log_file=self.prefs.engine_detail_log_file,
            engine_hotspot_log_file=self.prefs.engine_hotspot_log_file,
            engine_log_merge_window_sec=self.prefs.engine_log_merge_window_sec,
        )
        self.store.save(self.prefs)
        self.overview_proxy.apply_preferences()
        self.request_overview_refresh(force=True)

    def _controlled_ship_squad_ids(self) -> list[str]:
        return sorted(
            {
                s.squad_id
                for s in self.runtime_view.world.ships.values()
                if s.team == self.controlled_team and s.deployed
            }
        )

    def _controlled_fighter_squad_ids(self) -> list[str]:
        return sorted(
            {
                fighter.squad_id
                for fighter in self.runtime_view.world.fighters.values()
                if fighter.team == self.controlled_team and fighter.vital.alive and str(fighter.squad_id or "").strip()
            },
            key=str.casefold,
        )

    def _controlled_squad_ids(self) -> list[str]:
        squads = set(self._controlled_ship_squad_ids())
        squads.update(self._controlled_fighter_squad_ids())
        return sorted(squads, key=str.casefold)

    def _is_fighter_squad(self, squad_id: str | None = None) -> bool:
        squad = str(squad_id if squad_id is not None else self.ui_state.selected_squad or "").strip()
        if not squad:
            return False
        return self._team_has_fighter_squad(self.controlled_team, squad)

    def _team_has_fighter_squad(self, team: Team, squad_id: str) -> bool:
        squad = str(squad_id or "").strip()
        if not squad:
            return False
        return any(
            fighter.team == team
            and fighter.squad_id == squad
            and fighter.vital.alive
            for fighter in self.runtime_view.world.fighters.values()
        )

    def _selected_fighter_squad_abilities(self, squad_id: str | None = None) -> list[FighterAbilityProfile]:
        squad = str(squad_id if squad_id is not None else self.ui_state.selected_squad or "").strip()
        abilities: dict[str, FighterAbilityProfile] = {}
        for fighter in self.runtime_view.world.fighters.values():
            if fighter.team != self.controlled_team or fighter.squad_id != squad or not fighter.vital.alive:
                continue
            for ability in fighter.definition.abilities:
                if ability.kind == "normal_attack":
                    continue
                abilities.setdefault(ability.ability_id, ability)
        return sorted(abilities.values(), key=lambda item: (item.kind, item.ability_id))

    def _has_squad_fighter_control(self, squad_id: str) -> bool:
        squad = str(squad_id or "").strip()
        if not squad:
            return False
        if self._is_fighter_squad(squad):
            return True
        if self.available_squad_fighter_types(squad):
            return True
        return any(
            fighter.team == self.controlled_team
            and (fighter.owner_squad_id == squad or fighter.squad_id == squad)
            and fighter.vital.alive
            for fighter in self.runtime_view.world.fighters.values()
        )

    def _sync_blue_squads(self) -> None:
        squads = self._controlled_squad_ids()
        current = self.squad_combo.currentText().strip() or self.ui_state.selected_squad

        self.squad_combo.blockSignals(True)
        self.squad_combo.clear()
        self.squad_combo.addItems(squads)
        if current and current in squads:
            self.squad_combo.setCurrentText(current)
        elif squads:
            self.squad_combo.setCurrentText(squads[0])
        self.squad_combo.blockSignals(False)

        if squads:
            self.on_selected_squad_changed(self.squad_combo.currentText())
            return

        self.ui_state.selected_squad = ""
        self.canvas.selected_squad = ""
        self.assign_squad_edit.setText("")
        self.prefs.selected_squad = ""
        self.store.save(self.prefs)
        self._refresh_selected_squad_leader_speed_limit()
        self._refresh_propulsion_button_text()
        self._refresh_fighter_ability_controls()
        self.overview_model.notify_visual_state_changed()

    def on_selected_squad_changed(self, squad_id: str) -> None:
        squad = squad_id.strip()
        if not squad:
            return
        self.ui_state.selected_squad = squad
        self.canvas.selected_squad = squad
        self.assign_squad_edit.setText(squad)
        self.prefs.selected_squad = squad
        self.store.save(self.prefs)
        self._refresh_selected_squad_leader_speed_limit()
        self._refresh_propulsion_button_text()
        self._refresh_fighter_ability_controls()
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _refresh_selected_squad_leader_speed_limit(self) -> None:
        squad = self.application.query_service.squad_view(self.controlled_team, self.ui_state.selected_squad)
        value = float(squad.speed_limit or 0.0)
        self.spin_leader_speed_limit.blockSignals(True)
        self.spin_leader_speed_limit.setValue(value)
        self.spin_leader_speed_limit.blockSignals(False)

    def on_selected_squad_leader_speed_limit_changed(self, value: float) -> None:
        self._log_user_action("leader_speed_limit", squad=self.ui_state.selected_squad, limit=value)
        self.command_adapter.speed_limit(self.controlled_team, self.ui_state.selected_squad, float(value))

    def on_canvas_select_squad(self, squad_id: str) -> None:
        self.squad_combo.setCurrentText(squad_id)

    def on_canvas_select_enemy(self, ship_id: str) -> None:
        self._set_highlighted_overview_object({"entity_kind": "ship", "id": ship_id})
        self.ui_state.selected_enemy_target = ship_id
        self.canvas.selected_enemy_target = ship_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _selected_anchor(self) -> Vector2:
        members = [
            s
            for s in self.runtime_view.world.ships.values()
            if s.team == self.controlled_team and s.squad_id == self.ui_state.selected_squad and s.vital.alive
        ]
        if not members:
            return Vector2(0.0, 0.0)
        return Vector2(
            sum(m.nav.position.x for m in members) / len(members),
            sum(m.nav.position.y for m in members) / len(members),
        )

    def _overview_anchor_for_system(self, system_id: str) -> Vector2:
        members = [
            s
            for s in self.runtime_view.world.ships.values()
            if (
                s.team == self.controlled_team
                and s.squad_id == self.ui_state.selected_squad
                and s.vital.alive
                and str(getattr(s.nav, "system_id", "") or "") == str(system_id or "")
            )
        ]
        if not members:
            return Vector2(0.0, 0.0)
        return Vector2(
            sum(m.nav.position.x for m in members) / len(members),
            sum(m.nav.position.y for m in members) / len(members),
        )

    def _current_command_squad(self) -> str:
        squad_id = str(self.ui_state.selected_squad or "").strip()
        if squad_id:
            return squad_id
        if hasattr(self, "squad_combo"):
            squad_id = str(self.squad_combo.currentText() or "").strip()
            if squad_id:
                return squad_id
        for ship in self.runtime_view.world.ships.values():
            if ship.team == self.controlled_team and ship.deployed:
                return str(ship.squad_id or "").strip()
        for ship in self.runtime_view.world.ships.values():
            if ship.team == self.controlled_team:
                return str(ship.squad_id or "").strip()
        return ""

    def _apply_induce_spawn(self, team: Team, center: Vector2, squad_id: str | None = None) -> None:
        target_system_id = self._current_view_system_id()
        affected_squads = (
            {squad_id}
            if squad_id is not None
            else {
                ship.squad_id
                for ship in self.runtime_view.world.ships.values()
                if ship.team == team and not ship.deployed
            }
        )
        for squad in affected_squads:
            scoped_key = squad_key(team, squad)
            self._squad_guidance_targets.pop(scoped_key, None)
        self.command_adapter.induce_undeployed_ships(
            team,
            center,
            target_system_id,
            squad_id,
        )

    def induce_spawn_squad_at(self, squad_id: str, target: Vector2) -> None:
        squad = squad_id.strip()
        if not squad:
            return
        self._log_user_action("induce_squad", squad=squad, x=target.x, y=target.y)
        self._apply_induce_spawn(self.controlled_team, target, squad)

    def induce_spawn_fleet_at(self, target: Vector2) -> None:
        self._log_user_action("induce_fleet", x=target.x, y=target.y)
        self._apply_induce_spawn(self.controlled_team, target, None)

    def issue_approach_target(self, squad_id: str, target_id: str) -> None:
        squad = squad_id.strip()
        target = target_id.strip()
        if not squad or not target:
            return
        self._log_user_action("squad_approach", squad=squad, target=target)
        self.command_adapter.approach(self.controlled_team, squad, target)

    def issue_navigation_to_target(
        self,
        squad_id: str,
        *,
        target_kind: str,
        target_id: str,
        movement_mode: str,
        range_m: float = 0.0,
    ) -> None:
        squad = squad_id.strip()
        target_key = target_id.strip()
        mode = str(movement_mode or "").strip().lower()
        kind = str(target_kind or "").strip().lower()
        if not squad or not target_key or mode not in {"approach", "keep_range", "orbit"} or kind not in {"ship", "structure"}:
            return
        self._log_user_action("squad_navigation", squad=squad, target=target_key, target_kind=kind, mode=mode, range_m=range_m)
        self.command_adapter.navigate(
            self.controlled_team,
            squad,
            kind,
            target_key,
            mode,
            max(0.0, float(range_m or 0.0)),
        )

    def issue_approach_structure(self, squad_id: str, structure_id: str) -> None:
        self.issue_navigation_to_target(
            squad_id,
            target_kind="structure",
            target_id=structure_id,
            movement_mode="approach",
            range_m=0.0,
        )

    def _distance_action_label(self, range_m: float, *, current: bool = False) -> str:
        km = max(0.0, float(range_m)) / 1000.0
        if current:
            return QCoreApplication.translate("eve_sim", "Current Distance ({km:.1f} km)").format(km=km)
        if abs(km - round(km)) <= 1e-6:
            return QCoreApplication.translate("eve_sim", "{km:g} km").format(km=km)
        return QCoreApplication.translate("eve_sim", "{km:.1f} km").format(km=km)

    def _add_navigation_submenus(self, menu: QMenu, *, target_kind: str, target_id: str, target_label: str) -> None:
        squad = self._current_command_squad()
        action_approach = QAction(
            QCoreApplication.translate("eve_sim", "{squad} Approach {target}").format(squad=squad, target=target_label),
            self,
        )
        if target_kind == "ship":
            action_approach.triggered.connect(lambda: self.issue_approach_target(self._current_command_squad(), target_id))
        else:
            action_approach.triggered.connect(lambda: self.issue_approach_structure(self._current_command_squad(), target_id))
        menu.addAction(action_approach)

        current_range = self._range_to_target_from_squad(
            squad,
            target_ship_id=target_id if target_kind == "ship" else None,
            target_structure_id=target_id if target_kind == "structure" else None,
        )
        for mode, title in (
            ("orbit", QCoreApplication.translate("eve_sim", "{squad} Orbit {target}").format(squad=squad, target=target_label)),
            ("keep_range", QCoreApplication.translate("eve_sim", "{squad} Keep Range {target}").format(squad=squad, target=target_label)),
        ):
            submenu = QMenu(title, self)
            current_action = QAction(self._distance_action_label(current_range, current=True), self)
            current_action.triggered.connect(
                lambda _checked=False, m=mode, r=current_range: self.issue_navigation_to_target(
                    self._current_command_squad(),
                    target_kind=target_kind,
                    target_id=target_id,
                    movement_mode=m,
                    range_m=r,
                )
            )
            submenu.addAction(current_action)
            submenu.addSeparator()
            for range_m in NAV_RANGE_OPTIONS_M:
                action = QAction(self._distance_action_label(range_m), self)
                action.triggered.connect(
                    lambda _checked=False, m=mode, r=range_m: self.issue_navigation_to_target(
                        self._current_command_squad(),
                        target_kind=target_kind,
                        target_id=target_id,
                        movement_mode=m,
                        range_m=r,
                    )
                )
                submenu.addAction(action)
            menu.addMenu(submenu)

    def issue_warp_to_ship(self, squad_id: str, target_id: str) -> None:
        squad = squad_id.strip()
        target = target_id.strip()
        if not squad or not target:
            return
        self._log_user_action("squad_warp_ship", squad=squad, target=target)
        target_ship = self.runtime_view.world.combat_entity(target)
        if target_ship is None or not target_ship.vital.alive:
            return
        target_position = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
        self.command_adapter.warp(self.controlled_team, squad, target_position, target_ship_id=target)
        self._squad_guidance_targets[squad_key(self.controlled_team, squad)] = target_position

    def issue_warp_to_beacon(self, squad_id: str, beacon_id: str) -> None:
        squad = squad_id.strip()
        beacon_key = beacon_id.strip()
        if not squad or not beacon_key:
            return
        self._log_user_action("squad_warp_beacon", squad=squad, beacon=beacon_key)
        beacon = self.runtime_view.world.structures.get(beacon_key)
        if beacon is None:
            return
        target_position = Vector2(beacon.position.x, beacon.position.y)
        self.command_adapter.warp(self.controlled_team, squad, target_position, target_beacon_id=beacon_key)
        self._squad_guidance_targets[squad_key(self.controlled_team, squad)] = target_position

    def issue_use_gate(self, squad_id: str, structure_id: str) -> None:
        squad = squad_id.strip()
        structure_key = structure_id.strip()
        if not squad or not structure_key:
            return
        structure = self.runtime_view.world.structures.get(structure_key)
        if structure is None or str(getattr(structure, "kind", "") or "").upper() != "STARGATE":
            return
        self._log_user_action("squad_use_gate", squad=squad, structure=structure_key)
        self.command_adapter.use_gate(self.controlled_team, squad, structure_key)
        self._squad_guidance_targets[squad_key(self.controlled_team, squad)] = Vector2(structure.position.x, structure.position.y)

    def issue_move_to(self, squad_id: str, target: Vector2) -> None:
        self._log_user_action("squad_move", squad=squad_id, x=target.x, y=target.y)
        scoped_key = squad_key(self.controlled_team, squad_id)
        self._squad_guidance_targets[scoped_key] = Vector2(target.x, target.y)
        self.command_adapter.move(self.controlled_team, squad_id, target)

    def _squad_deployable_types(self, squad_id: str, bay_attr: str) -> list[str]:
        squad = str(squad_id or "").strip()
        names: set[str] = set()
        for ship in self.runtime_view.world.ships.values():
            if ship.team != self.controlled_team or ship.squad_id != squad or not ship.vital.alive:
                continue
            for entry in getattr(ship, bay_attr, []) or []:
                if int(getattr(entry, "quantity", 0) or 0) <= 0:
                    continue
                name = str(getattr(entry, "type_name", "") or "").strip()
                if name:
                    names.add(name)
        return sorted(names, key=str.casefold)

    def available_squad_drone_types(self, squad_id: str | None = None) -> list[str]:
        return self._squad_deployable_types(squad_id or self._current_command_squad(), "drone_bay")

    def available_squad_fighter_types(self, squad_id: str | None = None) -> list[str]:
        return self._squad_deployable_types(squad_id or self._current_command_squad(), "fighter_bay")

    def launch_squad_drones(self, squad_id: str, type_name: str) -> None:
        squad = str(squad_id or "").strip()
        name = str(type_name or "").strip()
        if not squad or not name:
            return
        self._log_user_action("squad_launch_drones", squad=squad, type_name=name)
        self.command_adapter.launch_drones(self.controlled_team, squad, name)

    def launch_squad_fighters(self, squad_id: str, type_name: str) -> None:
        squad = str(squad_id or "").strip()
        name = str(type_name or "").strip()
        if not squad or not name:
            return
        self._log_user_action("squad_launch_fighters", squad=squad, type_name=name)
        self.command_adapter.launch_fighters(self.controlled_team, squad, name)

    def recall_squad_deployables(self, squad_id: str) -> None:
        squad = str(squad_id or "").strip()
        if not squad:
            return
        self._log_user_action("squad_recall_deployables", squad=squad)
        self.command_adapter.recall_deployables(self.controlled_team, squad)

    def issue_drone_attack_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        target = str(target_id or "").strip()
        if not squad or not target:
            return
        target_entity = self.runtime_view.world.combat_entity(target)
        if target_entity is None or not target_entity.vital.alive or target_entity.team == self.controlled_team:
            return
        self._log_user_action("squad_drone_attack", squad=squad, target=target)
        self.command_adapter.drone_target(self.controlled_team, squad, target)

    def issue_fighter_attack_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        target = str(target_id or "").strip()
        if not squad or not target or self._is_fighter_squad(squad):
            return
        target_entity = self.runtime_view.world.combat_entity(target)
        if target_entity is None or not target_entity.vital.alive or target_entity.team == self.controlled_team:
            return
        self._log_user_action("squad_fighter_attack", squad=squad, target=target)
        self.command_adapter.fighter_target(self.controlled_team, squad, target)

    def activate_selected_fighter_ability(self, ability_id: str) -> None:
        squad = self.ui_state.selected_squad
        ability_key = str(ability_id or "").strip()
        if not squad or not ability_key or not self._is_fighter_squad(squad):
            return
        self._log_user_action("fighter_ability", squad=squad, ability=ability_key)
        self.command_adapter.fighter_ability(self.controlled_team, squad, ability_key)

    def issue_focus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        self._log_user_action("squad_focus", squad=squad, target=target_id)

        self.command_adapter.focus(self.controlled_team, squad, target_id)
        self.ui_state.selected_enemy_target = target_id
        self.canvas.selected_enemy_target = target_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def issue_prefocus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        self._log_user_action("squad_prefocus", squad=squad, target=target_id)

        self.command_adapter.prefocus(self.controlled_team, squad, target_id)
        self.ui_state.selected_enemy_target = target_id
        self.canvas.selected_enemy_target = target_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def cancel_prefocus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        self._log_user_action("squad_cancel_prefocus", squad=squad, target=target_id)

        self.command_adapter.cancel_prefocus(self.controlled_team, squad, target_id)

    def clear_focus_targets(self) -> None:
        squad = self.ui_state.selected_squad
        self._log_user_action("squad_clear_focus", squad=squad)

        self.command_adapter.clear_focus(self.controlled_team, squad)
        self.ui_state.selected_enemy_target = None
        self.canvas.selected_enemy_target = None
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def _iter_overview_rows(self) -> list[dict]:
        lang = self.current_language()
        current_system_id = self._current_view_system_id()
        anchor = self._overview_anchor_for_system(current_system_id)
        rows: list[dict] = []
        for ship in self.runtime_view.world.ships.values():
            if not self._is_ship_visible(ship.ship_id):
                continue
            ship_system_id = str(getattr(ship.nav, "system_id", "") or "")
            if current_system_id and ship_system_id and ship_system_id != current_system_id:
                continue
            hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
            hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
            hp_pct = round(100.0 * hp_cur / hp_max, 1) if hp_max > 0 else 0.0
            dist_m = ship.nav.position.distance_to(anchor)
            if dist_m >= (0.1 * AU_METERS):
                dist_display = f"{dist_m / AU_METERS:.2f} AU"
            else:
                dist_display = f"{dist_m / 1000.0:.1f} km"
            rows.append(
                {
                    "entity_kind": "ship",
                    "id": ship.ship_id,
                    "display_name": ship.ship_id,
                    "ship_name": ship.fit.ship_name,
                    "ship_name_display": self._display_ship_type(ship.fit.ship_name, language=lang),
                    "ship_type": ship.fit.ship_name,
                    "ship_type_display": self._display_ship_type(ship.fit.ship_name, language=lang),
                    "team": ship.team.value,
                    "team_display": ship.team.value,
                    "squad": ship.squad_id,
                    "role": ship.fit.role,
                    "alive": ship.vital.alive,
                    "system_id": ship_system_id,
                    "dist": dist_m,
                    "dist_display": dist_display,
                    "hp": hp_pct,
                    "dps": round(ship.profile.dps, 1),
                }
            )
        for structure in self.runtime_view.world.structures.values():
            structure_system_id = str(getattr(structure, "system_id", "") or "")
            if current_system_id and structure_system_id and structure_system_id != current_system_id:
                continue
            display_name = str(getattr(structure, "display_name", "") or getattr(structure, "structure_id", "") or "")
            dist_m = structure.position.distance_to(anchor)
            if dist_m >= (0.1 * AU_METERS):
                dist_display = f"{dist_m / AU_METERS:.2f} AU"
            else:
                dist_display = f"{dist_m / 1000.0:.1f} km"
            rows.append(
                {
                    "entity_kind": "structure",
                    "id": str(structure.structure_id),
                    "display_name": display_name,
                    "ship_name": display_name,
                    "ship_name_display": display_name,
                    "ship_type": str(getattr(structure, "kind", "") or "STRUCTURE"),
                    "ship_type_display": self._display_structure_type(str(getattr(structure, "kind", "") or "STRUCTURE")),
                    "team": "STRUCTURE",
                    "team_display": QCoreApplication.translate("eve_sim", "Structure"),
                    "squad": "",
                    "role": "STRUCTURE",
                    "alive": True,
                    "system_id": structure_system_id,
                    "dist": dist_m,
                    "dist_display": dist_display,
                    "hp": 100.0,
                    "dps": 0.0,
                }
            )
        return rows

    def request_overview_refresh(self, force: bool = False) -> None:
        rows = self._iter_overview_rows()
        if not force and rows == self._last_overview_rows:
            return
        self._last_overview_rows = rows
        self.refresh_overview(rows)

    def refresh_overview(self, rows: list[dict]) -> None:
        self.overview_model.set_rows(rows)
        self.overview_proxy.apply_preferences()
        self._restore_overview_selection()

    def _set_highlighted_overview_object(self, row_data: dict | None) -> None:
        entity_kind = str((row_data or {}).get("entity_kind", "ship") or "ship")
        entity_id = str((row_data or {}).get("id", "") or "").strip() or None
        self.canvas.selected_ship_id = entity_id if entity_kind == "ship" else None
        self.canvas.selected_structure_id = entity_id if entity_kind == "structure" else None
        self.canvas.update()

    def _selected_overview_key(self) -> tuple[str, str] | None:
        if self.canvas.selected_ship_id:
            return ("ship", str(self.canvas.selected_ship_id))
        if self.canvas.selected_structure_id:
            return ("structure", str(self.canvas.selected_structure_id))
        return None

    def _set_blue_roster_highlighted_ships(self, ship_ids: set[str]) -> None:
        self.canvas.highlighted_roster_ship_ids = {str(ship_id).strip() for ship_id in ship_ids if str(ship_id).strip()}
        self.canvas.update()

    def _selected_blue_roster_ship_ids(self) -> set[str]:
        selection_model = self.blue_roster.selectionModel()
        if selection_model is None:
            return set()
        ship_ids: set[str] = set()
        for index in selection_model.selectedRows():
            row_data = self.blue_roster_model.get_row(index.row())
            if row_data:
                ship_ids.add(str(row_data.get("ship_id", "")).strip())
        return {ship_id for ship_id in ship_ids if ship_id}

    def _restore_blue_roster_selection(self, selected_ship_ids: set[str]) -> None:
        selection_model = self.blue_roster.selectionModel()
        if selection_model is None:
            self._set_blue_roster_highlighted_ships(set())
            return
        valid_ids = {ship_id for ship_id in selected_ship_ids if ship_id}
        selection_model.clearSelection()
        if not valid_ids:
            self._set_blue_roster_highlighted_ships(set())
            return
        restored_ids: set[str] = set()
        for row in range(self.blue_roster_model.rowCount()):
            row_data = self.blue_roster_model.get_row(row)
            ship_id = str(row_data.get("ship_id", "")).strip() if row_data else ""
            if ship_id and ship_id in valid_ids:
                model_index = self.blue_roster_model.index(row, 0)
                selection_model.select(
                    model_index,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )
                restored_ids.add(ship_id)
        self._set_blue_roster_highlighted_ships(restored_ids)

    def _restore_overview_selection(self) -> None:
        selected_key = self._selected_overview_key()
        if selected_key is None:
            return
        current_index = self.overview.currentIndex()
        if current_index.isValid():
            row_data = self.overview_proxy.get_row(current_index.row())
            if row_data and (
                str(row_data.get("entity_kind", "ship")),
                str(row_data.get("id", "")),
            ) == selected_key:
                return
        for row in range(self.overview_proxy.rowCount()):
            row_data = self.overview_proxy.get_row(row)
            if row_data and (
                str(row_data.get("entity_kind", "ship")),
                str(row_data.get("id", "")),
            ) == selected_key:
                self.overview.selectRow(row)
                return

    def _on_overview_selection_changed(self, *_args) -> None:
        indexes = self.overview.selectionModel().selectedRows() if self.overview.selectionModel() is not None else []
        if not indexes:
            self._set_highlighted_overview_object(None)
            return
        row_data = self.overview_proxy.get_row(indexes[0].row())
        self._set_highlighted_overview_object(row_data)

    def _on_overview_double_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        row_data = self.overview_proxy.get_row(index.row())
        if not row_data:
            return
        system_id = str(row_data.get("system_id", "") or "").strip()
        if system_id:
            self._set_view_system(system_id, center=False)
        entity_kind = str(row_data.get("entity_kind", "ship") or "ship")
        entity_id = str(row_data.get("id", "") or "").strip()
        if not entity_id:
            return
        if entity_kind == "structure":
            self.canvas.focus_structure(entity_id)
        else:
            self.canvas.focus_ship(entity_id)

    def _on_blue_roster_selection_changed(self, *_args) -> None:
        self._set_blue_roster_highlighted_ships(self._selected_blue_roster_ship_ids())

    def _build_ship_context_menu(self, ship_id: str) -> QMenu | None:
        ship = self.runtime_view.world.combat_entity(str(ship_id))
        if ship is None or not ship.vital.alive:
            return None
        target_id = str(ship.ship_id)
        controlled_team = self.controlled_team
        enemy_team = Team.RED.value if controlled_team == Team.BLUE else Team.BLUE.value
        is_real_ship = target_id in self.runtime_view.world.ships
        target_label = (
            target_id
            if is_real_ship
            else self._display_type_name(str(getattr(getattr(ship, "fit", None), "ship_name", target_id) or target_id))
        )

        self._set_highlighted_overview_object({"entity_kind": "ship", "id": target_id} if is_real_ship else None)
        if ship.team != controlled_team:
            self.ui_state.selected_enemy_target = target_id
            self.canvas.selected_enemy_target = target_id
            self.overview_model.notify_visual_state_changed()
            self.request_overview_refresh(force=True)

        menu = QMenu(self)
        if is_real_ship:
            action_status = QAction(QCoreApplication.translate("eve_sim", 'View {ship} Status').format(ship=target_id), self)
            action_status.triggered.connect(lambda: self.show_ship_status(target_id))
            menu.addAction(action_status)
        else:
            action_status = QAction(QCoreApplication.translate("eve_sim", '{ship}').format(ship=target_label), self)
            action_status.setEnabled(False)
            menu.addAction(action_status)

        action_warp = QAction(
            QCoreApplication.translate("eve_sim", '{squad} Warp To {ship}').format(
                squad=self._current_command_squad(),
                ship=target_label,
            ),
            self,
        )
        action_warp.triggered.connect(lambda: self.issue_warp_to_ship(self._current_command_squad(), target_id))
        menu.addAction(action_warp)
        self._add_navigation_submenus(menu, target_kind="ship", target_id=target_id, target_label=target_label)

        if ship.team.value == enemy_team:
            action_focus = QAction(
                QCoreApplication.translate("eve_sim", '{squad} Focus {ship}').format(
                    squad=self.ui_state.selected_squad,
                    ship=target_label,
                ),
                self,
            )
            action_focus.triggered.connect(lambda: self.issue_focus_target(target_id))
            menu.addAction(action_focus)

            action_prefocus = QAction(
                QCoreApplication.translate("eve_sim", '{squad} Pre-focus {ship}').format(
                    squad=self.ui_state.selected_squad,
                    ship=target_label,
                ),
                self,
            )
            action_prefocus.triggered.connect(lambda: self.issue_prefocus_target(target_id))
            menu.addAction(action_prefocus)

            has_squad_drones = bool(self.available_squad_drone_types(self.ui_state.selected_squad)) or any(
                drone.team == controlled_team and drone.squad_id == self.ui_state.selected_squad
                for drone in self.runtime_view.world.drones.values()
            )
            if has_squad_drones:
                action_drone_attack = QAction(
                    self._ui_text('{squad} Drone Attack {ship}').format(
                        squad=self.ui_state.selected_squad,
                        ship=target_label,
                    ),
                    self,
                )
                action_drone_attack.triggered.connect(lambda: self.issue_drone_attack_target(target_id))
                menu.addAction(action_drone_attack)

            if not self._is_fighter_squad(self.ui_state.selected_squad) and self._has_squad_fighter_control(self.ui_state.selected_squad):
                action_fighter_attack = QAction(
                    self._ui_text('{squad} Fighter Attack {ship}').format(
                        squad=self.ui_state.selected_squad,
                        ship=target_label,
                    ),
                    self,
                )
                action_fighter_attack.triggered.connect(lambda: self.issue_fighter_attack_target(target_id))
                menu.addAction(action_fighter_attack)

            queue = self.application.query_service.squad_view(
                controlled_team,
                self.ui_state.selected_squad,
            ).focus_queue
            in_prequeue = target_id in queue
            squad_members = [
                ship
                for ship in self.runtime_view.world.ships.values()
                if ship.team == controlled_team and ship.squad_id == self.ui_state.selected_squad
            ]
            prelocked = any(target_id in ship.combat.prelocked_targets for ship in squad_members)
            prelocking = any(target_id in ship.combat.prelock_timers for ship in squad_members)
            if in_prequeue or prelocked or prelocking:
                action_cancel_prefocus = QAction(
                    QCoreApplication.translate("eve_sim", '{squad} Cancel Pre-lock {ship}').format(
                        squad=self.ui_state.selected_squad,
                        ship=target_id,
                    ),
                    self,
                )
                action_cancel_prefocus.triggered.connect(lambda: self.cancel_prefocus_target(target_id))
                menu.addAction(action_cancel_prefocus)

        return menu

    def show_ship_context_menu(self, ship_id: str, global_pos: QPoint) -> None:
        menu = self._build_ship_context_menu(ship_id)
        if menu is None:
            return
        menu.exec(global_pos)

    def _build_structure_context_menu(self, structure_id: str) -> QMenu | None:
        structure = self.runtime_view.world.structures.get(str(structure_id))
        if structure is None:
            return None
        structure_key = str(structure.structure_id)
        structure_name = self._display_structure_name(structure)
        self._set_highlighted_overview_object({"entity_kind": "structure", "id": structure_key})

        menu = QMenu(self)
        action_center = QAction(
            QCoreApplication.translate("eve_sim", "Center View on {structure}").format(structure=structure_name),
            self,
        )
        action_center.triggered.connect(lambda: self.canvas.focus_structure(structure_key))
        menu.addAction(action_center)

        action_warp = QAction(
            QCoreApplication.translate("eve_sim", "{squad} Warp To {structure}").format(
                squad=self._current_command_squad(),
                structure=structure_name,
            ),
            self,
        )
        action_warp.triggered.connect(lambda: self.issue_warp_to_beacon(self._current_command_squad(), structure_key))
        menu.addAction(action_warp)
        self._add_navigation_submenus(menu, target_kind="structure", target_id=structure_key, target_label=structure_name)
        if str(getattr(structure, "kind", "") or "").upper() == "STARGATE":
            linked_id = str(getattr(structure, "linked_structure_id", "") or "").strip()
            if linked_id:
                action_use_gate = QAction(
                    QCoreApplication.translate("eve_sim", "{squad} Take Gate {structure}").format(
                        squad=self._current_command_squad(),
                        structure=structure_name,
                    ),
                    self,
                )
                action_use_gate.triggered.connect(lambda: self.issue_use_gate(self._current_command_squad(), structure_key))
                menu.addAction(action_use_gate)
        return menu

    def show_structure_context_menu(self, structure_id: str, global_pos: QPoint) -> None:
        menu = self._build_structure_context_menu(structure_id)
        if menu is None:
            return
        menu.exec(global_pos)

    def show_overview_menu(self, pos: QPoint) -> None:
        index = self.overview.indexAt(pos)
        if not index.isValid():
            return
        row_data = self.overview_proxy.get_row(index.row())
        if not row_data:
            return
        if str(row_data.get("entity_kind", "ship")) == "structure":
            self.show_structure_context_menu(str(row_data["id"]), self.overview.mapToGlobal(pos))
            return
        self.show_ship_context_menu(str(row_data["id"]), self.overview.mapToGlobal(pos))

    def show_blue_roster_menu(self, pos: QPoint) -> None:
        index = self.blue_roster.indexAt(pos)
        if not index.isValid():
            return
        row_data = self.blue_roster_model.get_row(index.row())
        if not row_data:
            return
        self.show_ship_context_menu(str(row_data["ship_id"]), self.blue_roster.mapToGlobal(pos))

    def refresh_blue_roster(self) -> None:
        selected_ship_ids = self._selected_blue_roster_ship_ids()
        ships = sorted(
            [s for s in self.runtime_view.world.ships.values() if s.team == self.controlled_team and s.deployed],
            key=lambda s: s.ship_id,
        )
        lang = self.current_language()
        rows: list[dict] = []
        for ship in ships:
            hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
            hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
            hp_pct = 100.0 * hp_cur / hp_max if hp_max > 0 else 0.0
            rows.append(
                {
                    "ship_id": ship.ship_id,
                    "ship_name": ship.fit.ship_name,
                    "ship_name_display": self._display_ship_type(ship.fit.ship_name, language=lang),
                    "squad": ship.squad_id,
                    "role": ship.fit.role,
                    "alive": ship.vital.alive,
                    "hp": hp_pct,
                }
            )
        self.blue_roster_model.set_rows(rows)
        self._restore_blue_roster_selection(selected_ship_ids)

    def assign_blue_ships(self) -> None:
        target_squad = self.assign_squad_edit.text().strip()
        if not target_squad:
            return

        selected_rows = sorted({idx.row() for idx in self.blue_roster.selectionModel().selectedRows()})
        if not selected_rows:
            return

        ship_ids: list[str] = []
        for row in selected_rows:
            row_data = self.blue_roster_model.get_row(row)
            if not row_data:
                continue
            ship_ids.append(str(row_data["ship_id"]))

        self._log_user_action("assign_squad", target_squad=target_squad, ship_count=len(ship_ids))
        self.command_adapter.assign_ships(self.controlled_team, tuple(ship_ids), target_squad)

    def _consume_removed_ship_ids(self, ship_ids: tuple[str, ...]) -> None:
        for ship_id in ship_ids:
            dialog = self._status_dialogs.pop(ship_id, None)
            if dialog is not None:
                dialog.close()

    @staticmethod
    def _format_tidi_percent(tidi_factor: float) -> str:
        try:
            factor = float(tidi_factor)
        except Exception:
            factor = 1.0
        factor = max(0.0, min(1.0, factor))
        return f"{factor * 100.0:.0f}%"

    def _effective_tidi_factor(self) -> float:
        if self.network_mode == "client":
            value = self.network.remote_tidi_factor
        else:
            value = self.application.tidi_factor()
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 1.0

    def _tick_timer_interval_ms(self) -> int:
        if self.network_mode == "client":
            return max(1, int(getattr(self, "_client_poll_interval_ms", 50)))
        return self.application.next_tick_delay_ms()

    def _reschedule_tick_timer(self) -> None:
        timer = getattr(self, "tick_timer", None)
        if timer is not None:
            timer.setInterval(self._tick_timer_interval_ms())

    def _status_tidi_text(self) -> str:
        return f"TiDi: {self._format_tidi_percent(self._effective_tidi_factor())}"

    def _refresh_sim_status(self) -> None:
        if not hasattr(self, "status") or not hasattr(self, "canvas"):
            return
        alive_blue = 0
        alive_red = 0
        total_ships = 0
        for ship in self.runtime_view.world.ships.values():
            total_ships += 1
            if not ship.vital.alive:
                continue
            if ship.team == Team.BLUE:
                alive_blue += 1
            elif ship.team == Team.RED:
                alive_red += 1
        tick = self.runtime_view.world.tick
        diagnostics = self.runtime_view.diagnostics()
        mode = str(diagnostics.get("execution_mode", "global_serial") or "global_serial")
        effective_mode = str(diagnostics.get("effective_execution_mode", mode) or mode)
        disabled_reason = str(diagnostics.get("parallel_disabled_reason", "") or "")
        execution_text = f"Mode: {mode} / effective: {effective_mode}"
        if disabled_reason:
            execution_text += f" ({disabled_reason})"
        self.status.setText(
            f"{QCoreApplication.translate("eve_sim", 'Tick')}: {tick} | {QCoreApplication.translate("eve_sim", 'Ships')}: {total_ships} | "
            f"{QCoreApplication.translate("eve_sim", 'BLUE')}: {alive_blue} | {QCoreApplication.translate("eve_sim", 'RED')}: {alive_red} | "
            f"{self._status_tidi_text()} | {QCoreApplication.translate("eve_sim", 'Zoom')}: {self.canvas.zoom:.2f} | "
            f"{QCoreApplication.translate("eve_sim", 'Step ms')}: {self._step_ms_ema:.2f} | {execution_text}"
        )

    def _set_waiting_status(self, message: str) -> None:
        self.status.setText(f"{QCoreApplication.translate("eve_sim", 'Tick')}: {message} | {self._status_tidi_text()}")

    def _refresh_roster_for_current_tick(self) -> None:
        try:
            tick = int(self.runtime_view.world.tick)
        except Exception:
            return
        if tick <= 0 or (tick % 10) != 0:
            return
        if getattr(self, "_last_roster_refresh_tick", None) == tick:
            return
        self.refresh_blue_roster()
        self._last_roster_refresh_tick = tick

    def on_tick(self) -> None:
        if self.network_mode == "client":
            poll_result = self.network.poll_client(self.controlled_team)
            if poll_result.received_snapshot:
                self.runtime_view.refresh()
                self._consume_removed_ship_ids(poll_result.removed_ship_ids)
                self._reschedule_tick_timer()
                if hasattr(self, "canvas") and hasattr(self.canvas, "note_authoritative_frame"):
                    self.canvas.note_authoritative_frame()
            self._ui_tick_counter += 1
            if (self._ui_tick_counter % self._ui_refresh_interval_ticks) == 0:
                self._sync_blue_squads()
            if (self._ui_tick_counter % self._overview_refresh_interval_ticks) == 0:
                self.request_overview_refresh(force=True)
            if hasattr(self, "_refresh_roster_for_current_tick"):
                self._refresh_roster_for_current_tick()
            elif self.runtime_view.world.tick % 10 == 0:
                self.refresh_blue_roster()
            if poll_result.received_snapshot:
                self.recording.record_snapshot()
            if hasattr(self, "_refresh_sim_status") and (self._ui_tick_counter % self._ui_refresh_interval_ticks) == 0:
                self._refresh_sim_status()
                self._refresh_fighter_ability_controls()
            return

        if self.network_mode == "host":
            has_remote_red = any(s.team == Team.RED for s in self.runtime_view.world.ships.values())
            gate = self.network.prepare_host_tick(has_remote_fleet=has_remote_red)
            if not gate.should_step:
                self._set_waiting_status(gate.status_message)
                return

        t0 = time.perf_counter()
        tick_results = self.application.step()
        self.runtime_view.refresh()
        if any(
            event.kind in {"ships_assigned_to_squad", "ships_induced"}
            for result in tick_results
            for event in result.events
        ):
            self._sync_blue_squads()
            self.refresh_blue_roster()
            self.request_overview_refresh(force=True)
        if hasattr(self, "canvas") and hasattr(self.canvas, "note_authoritative_frame"):
            self.canvas.note_authoritative_frame()
        step_ms = (time.perf_counter() - t0) * 1000.0
        if self._step_ms_ema <= 0:
            self._step_ms_ema = step_ms
        else:
            self._step_ms_ema = self._step_ms_ema * 0.85 + step_ms * 0.15
        self.application.update_tidi_after_step(step_ms)
        if hasattr(self, "_reschedule_tick_timer"):
            self._reschedule_tick_timer()

        if self.network_mode == "host":
            self.network.publish(countdown_left=0.0, started=True)

        self.recording.record_snapshot()

        self._ui_tick_counter += 1
        refresh_ui = (self._ui_tick_counter % self._ui_refresh_interval_ticks) == 0
        refresh_overview = (self._ui_tick_counter % self._overview_refresh_interval_ticks) == 0

        if refresh_ui:
            self._refresh_propulsion_button_text()
            self._refresh_fighter_ability_controls()
        if hasattr(self, "_refresh_sim_status"):
            self._refresh_sim_status()

        if refresh_overview:
            self.request_overview_refresh()

        if hasattr(self, "_refresh_roster_for_current_tick"):
            self._refresh_roster_for_current_tick()
        elif self.runtime_view.world.tick % 10 == 0:
            self.refresh_blue_roster()

        if self.ui_state.selected_enemy_target:
            target = self.runtime_view.world.combat_entity(self.ui_state.selected_enemy_target)
            if target is None or not target.vital.alive:
                self.ui_state.selected_enemy_target = None
                self.canvas.selected_enemy_target = None
                self.overview_model.notify_visual_state_changed()
                self.request_overview_refresh(force=True)
        if self.canvas.selected_ship_id:
            selected_ship = self.runtime_view.world.combat_entity(self.canvas.selected_ship_id)
            if selected_ship is None or not selected_ship.deployed:
                self._set_highlighted_overview_object(None)
        if self.canvas.selected_structure_id:
            selected_structure = self.runtime_view.world.structures.get(self.canvas.selected_structure_id)
            if selected_structure is None:
                self._set_highlighted_overview_object(None)
        if self.canvas.highlighted_roster_ship_ids:
            valid_highlighted_ids = {
                ship_id
                for ship_id in self.canvas.highlighted_roster_ship_ids
                if ship_id in self.runtime_view.world.ships and self.runtime_view.world.ships[ship_id].deployed
            }
            if valid_highlighted_ids != self.canvas.highlighted_roster_ship_ids:
                self._set_blue_roster_highlighted_ships(valid_highlighted_ids)

    def closeEvent(self, event) -> None:
        self.prefs.selected_squad = self.ui_state.selected_squad
        self.prefs.zoom = self.canvas.zoom
        self.store.save(self.prefs)
        self.application.flush_pending_events()
        if hasattr(self, "tick_timer"):
            self.tick_timer.stop()
        if hasattr(self, "render_timer"):
            self.render_timer.stop()
        if hasattr(self, "system_graph_window"):
            self.system_graph_window.close()
            self.system_graph_window.deleteLater()
        self.network.close()
        self.application.close()
        super().closeEvent(event)




