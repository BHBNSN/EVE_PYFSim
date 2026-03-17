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
from .table_models import *
from .dialogs import *
from .fleet_setup_dialog import *
from .battle_canvas import *
class MainWindow(QMainWindow):
    """
    主窗口类 (Main Window)
    
    该类负责组织整个应用的图形界面 (GUI)，连接各个子模块 (如概览面板、战斗画布等)，
    并管理其生命周期。
    """
    def __init__(
        self,
        engine: SimulationEngine,
        ui_cfg: UiConfig,
        blue_commander: CommanderAgent,
        red_commander: CommanderAgent,
        manual_setup: list[ManualShipSetup],
        network_mode: str = "local",
        controlled_team: Team = Team.BLUE,
        lan_server: HostLanSession | None = None,
        lan_client: ClientLanSession | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.ui_cfg = ui_cfg
        self.blue_commander = blue_commander
        self.red_commander = red_commander
        self.manual_setup = manual_setup
        self.network_mode = network_mode
        self.controlled_team = controlled_team
        self.lan_server = lan_server
        self.lan_client = lan_client
        self._match_started = network_mode != "host"
        self._countdown_started_at: float | None = None
        self._parser = EftFitParser()
        self._factory = RuntimeFromEftFactory()
        self._ship_fit_texts: dict[str, str] = {}
        self._charge_module_ammo_selection: dict[str, str] = {}
        self._ship_locked_module_charges: dict[str, dict[str, str]] = {}
        self._squad_approach_targets: dict[str, str] = {}
        self._squad_guidance_targets: dict[str, Vector2] = {}
        self._undeployed_ship_ids: set[str] = set()
        world_ship_ids = list(self.engine.world.ships.keys())
        for idx, ship_id in enumerate(world_ship_ids):
            if idx < len(self.manual_setup):
                self._ship_fit_texts[ship_id] = self.manual_setup[idx].fit_text
        self.store = PreferencesStore()
        self.prefs = self.store.load()
        if self.prefs.filter_team in ("BLUE", "RED"):
            if self.prefs.filter_team == self.controlled_team.value:
                self.prefs.filter_team = "FRIENDLY"
            else:
                self.prefs.filter_team = "ENEMY"
        if self.network_mode == "client":
            if self.prefs.filter_team == "ALL":
                self.prefs.filter_team = "ENEMY"

        self._initialize_deployment_state()

        self.ui_state = UiState(selected_squad=self.prefs.selected_squad, selected_enemy_target=None)
        self.setWindowTitle(tr(self.current_language(), "app_title"))
        self.resize(ui_cfg.width + 560, ui_cfg.height)
        self._ui_refresh_interval_ticks = 3
        self._overview_refresh_interval_ticks = 3
        self._ui_tick_counter = 0
        self._last_overview_rows: list[dict] = []
        self._status_dialogs: dict[str, ShipStatusDialog] = {}
        self._step_ms_ema: float = 0.0
        self._pending_tick_ops: list[Callable[[], None]] = []
        self._setup_synced = False
        self._last_network_send_at: float = 0.0
        self._last_full_sync_at: float = 0.0
        self._network_send_interval_sec: float = 1.0 / 20.0
        self._network_full_sync_interval_sec: float = 1.0
        self._last_sent_ship_signatures: dict[str, tuple] = {}
        self._last_sent_fit_texts: dict[str, str] = {}
        self._lan_debug_enabled = False

        self._create_menu()

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter)

        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        self.canvas = BattleCanvas(
            engine,
            ui_cfg,
            self.issue_move_to,
            self.issue_approach_target,
            self.issue_focus_target,
            self.issue_prefocus_target,
            self.cancel_prefocus_target,
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
        )
        if self.prefs.zoom is not None:
            self.canvas.zoom = float(self.prefs.zoom)
        splitter.addWidget(self.canvas)
        splitter.setSizes([560, ui_cfg.width])

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.on_tick)
        self.tick_timer.start(int(1000 / self.engine.config.tick_rate))

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.on_render_frame)
        self.render_timer.start(16)

        self._sync_blue_squads()
        self.refresh_blue_roster()
        self.request_overview_refresh(force=True)

    def show_ship_status(self, ship_id: str) -> None:
        dialog = self._status_dialogs.get(ship_id)
        if dialog is None:
            dialog = ShipStatusDialog(
                self.engine,
                ship_id,
                self.current_language,
                self.get_ship_fit_text,
                self._get_ship_locked_module_charge,
                self._set_ship_module_charge_lock,
                self._clear_ship_module_charge_lock,
                self,
            )
            self._status_dialogs[ship_id] = dialog
            dialog.finished.connect(lambda _r, sid=ship_id: self._status_dialogs.pop(sid, None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _enqueue_tick_op(self, op: Callable[[], None]) -> None:
        self._pending_tick_ops.append(op)

    def _lan_debug(self, message: str) -> None:
        if not self._lan_debug_enabled:
            return
        print(f"[LAN][{self.network_mode}] {message}")

    def _log_user_action(self, action: str, **fields) -> None:
        if not bool(self.engine.config.detailed_logging):
            return
        payload: dict[str, object] = {
            "action": action,
            "mode": self.network_mode,
            "team": self.controlled_team.value,
            "tick": int(self.engine.world.tick),
        }
        payload.update(fields)
        log_sim_event(getattr(self.engine, "_logger", None), "user_operation", **payload)

    def _flush_tick_ops(self) -> None:
        if not self._pending_tick_ops:
            return
        ops = self._pending_tick_ops
        self._pending_tick_ops = []
        for op in ops:
            try:
                op()
            except Exception:
                continue

    def _initialize_deployment_state(self) -> None:
        target_team = self.controlled_team
        for ship in self.engine.world.ships.values():
            if ship.team != target_team:
                continue
            self._undeployed_ship_ids.add(ship.ship_id)
            ship.vital.alive = False
            ship.nav.velocity = Vector2(0.0, 0.0)
            ship.order_queue.clear()

    def _is_ship_visible(self, ship_id: str) -> bool:
        return ship_id not in self._undeployed_ship_ids

    @staticmethod
    def _parse_team_squad_key(scoped_key: str, default_team: Team) -> tuple[Team, str]:
        text = str(scoped_key or "")
        if ":" in text:
            head, tail = text.split(":", 1)
            if tail and head in (Team.BLUE.value, Team.RED.value):
                return Team(head), tail
        return default_team, text

    def _guidance_target_for_squad(self, squad_id: str) -> Vector2 | None:
        scoped_key = self._focus_key(self.controlled_team, squad_id)
        target = self._squad_guidance_targets.get(scoped_key)
        if target is not None:
            return target
        return self._squad_guidance_targets.get(squad_id)

    def _inducible_controlled_squad_ids(self) -> list[str]:
        squads = {
            s.squad_id
            for s in self.engine.world.ships.values()
            if s.team == self.controlled_team and s.ship_id in self._undeployed_ship_ids
        }
        return sorted(squads)

    def on_render_frame(self) -> None:
        self.canvas.update()

    def get_ship_fit_text(self, ship_id: str) -> str | None:
        return self._ship_fit_texts.get(ship_id)

    @staticmethod
    def _module_index_from_id(module_id: str) -> int | None:
        parts = str(module_id).rsplit("-", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return None
        idx = int(parts[1])
        return idx if idx > 0 else None

    def _get_ship_locked_module_charge(self, ship_id: str, module_id: str) -> str | None:
        return self._ship_locked_module_charges.get(ship_id, {}).get(module_id)

    def _prune_ship_locked_module_charges(self, ship_id: str, runtime_module_ids: set[str]) -> None:
        locked = self._ship_locked_module_charges.get(ship_id)
        if not locked:
            return
        for module_id in list(locked.keys()):
            if module_id not in runtime_module_ids:
                locked.pop(module_id, None)
        if not locked:
            self._ship_locked_module_charges.pop(ship_id, None)

    def _sync_manual_setup_fit_text(self, ship_id: str, fit_text: str) -> None:
        ship_ids = list(self.engine.world.ships.keys())
        try:
            idx = ship_ids.index(ship_id)
        except ValueError:
            return
        if 0 <= idx < len(self.manual_setup):
            self.manual_setup[idx].fit_text = fit_text

    def _rewrite_fit_text_with_lock_rules(
        self,
        ship_id: str,
        fit_text: str,
        *,
        target_module_name: str = "",
        target_ammo_name: str = "",
        force_module_id: str = "",
        force_ammo_name: str = "",
    ) -> str:
        locked_map = self._ship_locked_module_charges.get(ship_id, {})
        canonical_target_module = resolve_module_type_name(target_module_name).strip().lower() if target_module_name else ""
        canonical_target_ammo = resolve_module_type_name(target_ammo_name).strip() if target_ammo_name else ""
        canonical_force_ammo = resolve_module_type_name(force_ammo_name).strip() if force_ammo_name else ""

        lines = fit_text.splitlines()
        out: list[str] = []
        module_idx = 0
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("[") or raw.lower().startswith("dna:") or raw.lower().startswith("x-"):
                out.append(line)
                continue

            if " x" in raw:
                out.append(line)
                continue

            offline_suffix = ""
            base = raw
            if base.endswith("/offline"):
                base = base[:-8].rstrip()
                offline_suffix = " /offline"
            if base.endswith("/OFFLINE"):
                base = base[:-8].rstrip()
                offline_suffix = " /OFFLINE"

            if not base or base.startswith("[Empty"):
                out.append(line)
                continue

            module_idx += 1
            module_id = f"mod-{module_idx}"
            module_name = base.split(",", 1)[0].strip()
            canonical_module_name = resolve_module_type_name(module_name).strip().lower()

            if force_module_id and module_id == force_module_id and canonical_force_ammo:
                out.append(f"{module_name}, {canonical_force_ammo}{offline_suffix}")
                continue

            locked_ammo = str(locked_map.get(module_id) or "").strip()
            if locked_ammo:
                out.append(f"{module_name}, {locked_ammo}{offline_suffix}")
                continue

            if canonical_target_module and canonical_target_ammo and canonical_module_name == canonical_target_module:
                out.append(f"{module_name}, {canonical_target_ammo}{offline_suffix}")
            else:
                out.append(line)

        return "\n".join(out)

    def _rebuild_ship_from_fit_text(self, ship_id: str, fit_text: str, lang: str) -> tuple[bool, str, object | None]:
        ship = self.engine.world.ships.get(ship_id)
        if ship is None:
            return False, tr(lang, "ship_missing"), None
        try:
            parsed = self._parser.parse(fit_text)
            runtime_template, fit = self._factory.build(parsed)
            runtime = deepcopy(runtime_template)
            profile = self._factory.build_profile(parsed)
        except Exception as exc:
            return False, _localize_fit_error(lang, exc), None

        ship.runtime = runtime
        ship.fit = fit
        ship.profile = profile

        runtime_module_ids = {m.module_id for m in runtime.modules}
        for timer_map in (
            ship.combat.module_cycle_timers,
            ship.combat.module_reactivation_timers,
            ship.combat.module_ammo_reload_timers,
            ship.combat.module_pending_ammo_reload_timers,
        ):
            for module_id in list(timer_map.keys()):
                if module_id not in runtime_module_ids:
                    timer_map.pop(module_id, None)

        self._prune_ship_locked_module_charges(ship_id, runtime_module_ids)
        self._ship_fit_texts[ship_id] = fit_text
        self._sync_manual_setup_fit_text(ship_id, fit_text)
        return True, "", parsed

    def _changed_module_ids_between_parsed(self, old_parsed, new_parsed) -> list[str]:
        old_specs = list(getattr(old_parsed, "module_specs", []) or [])
        new_specs = list(getattr(new_parsed, "module_specs", []) or [])
        changed: list[str] = []
        count = min(len(old_specs), len(new_specs))
        for idx in range(count):
            old_charge = resolve_module_type_name(str(old_specs[idx].charge_name or "")).strip().lower()
            new_charge = resolve_module_type_name(str(new_specs[idx].charge_name or "")).strip().lower()
            if old_charge != new_charge:
                changed.append(f"mod-{idx + 1}")
        return changed

    def _set_ship_module_charge_lock(self, ship_id: str, module_id: str, ammo_name: str) -> tuple[bool, str]:
        lang = self.current_language()
        ship = self.engine.world.ships.get(ship_id)
        if ship is None:
            return False, tr(lang, "ship_missing")
        if not self._is_ammo_configurable_team(ship.team):
            return False, tr(lang, "status_lock_team_denied")

        old_text = self.get_ship_fit_text(ship_id) or ""
        if not old_text.strip():
            return False, tr(lang, "status_lock_module_missing")

        try:
            parsed_old = self._parser.parse(old_text)
        except Exception as exc:
            return False, _localize_fit_error(lang, exc)

        module_idx = self._module_index_from_id(module_id)
        if module_idx is None or module_idx > len(parsed_old.module_specs):
            return False, tr(lang, "status_lock_module_missing")

        spec = parsed_old.module_specs[module_idx - 1]
        ammo_options = get_charge_options_for_module(spec.module_name, language=lang)
        if not ammo_options:
            return False, tr(lang, "status_lock_module_not_chargeable")

        canonical_ammo = resolve_module_type_name(ammo_name).strip()
        if not canonical_ammo:
            return False, tr(lang, "status_lock_ammo_invalid")
        option_keys = {resolve_module_type_name(opt).strip().lower() for opt in ammo_options}
        if canonical_ammo.strip().lower() not in option_keys:
            return False, tr(lang, "status_lock_ammo_invalid")

        locked_map = self._ship_locked_module_charges.setdefault(ship_id, {})
        locked_map[module_id] = canonical_ammo
        new_text = self._rewrite_fit_text_with_lock_rules(
            ship_id,
            old_text,
            force_module_id=module_id,
            force_ammo_name=canonical_ammo,
        )

        ok, message, parsed_new = self._rebuild_ship_from_fit_text(ship_id, new_text, lang)
        if not ok:
            locked_map.pop(module_id, None)
            if not locked_map:
                self._ship_locked_module_charges.pop(ship_id, None)
            return False, message

        changed_module_ids = self._changed_module_ids_between_parsed(parsed_old, parsed_new)
        reload_sec = max(0.0, get_module_reload_time_sec(spec.module_name))
        if reload_sec > 0.0 and module_id in changed_module_ids:
            cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module_id, 0.0) or 0.0))
            active_reload_left = max(0.0, float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0))
            if cycle_left > 0.0 or active_reload_left > 0.0:
                ship.combat.module_pending_ammo_reload_timers[module_id] = reload_sec
            else:
                ship.combat.module_ammo_reload_timers[module_id] = reload_sec
                ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)

        module_name = get_type_display_name(spec.module_name, language=lang)
        ammo_display = get_type_display_name(canonical_ammo, language=lang)
        return True, tr(lang, "status_lock_set_done", module=module_name, ammo=ammo_display)

    def _clear_ship_module_charge_lock(self, ship_id: str, module_id: str) -> tuple[bool, str]:
        lang = self.current_language()
        module_label = module_id
        fit_text = self.get_ship_fit_text(ship_id) or ""
        try:
            parsed = self._parser.parse(fit_text) if fit_text else None
            module_idx = self._module_index_from_id(module_id)
            if parsed is not None and module_idx is not None and module_idx <= len(parsed.module_specs):
                module_label = get_type_display_name(parsed.module_specs[module_idx - 1].module_name, language=lang)
        except Exception:
            pass

        locked_map = self._ship_locked_module_charges.get(ship_id)
        if not locked_map or module_id not in locked_map:
            return True, tr(lang, "status_lock_clear_done", module=module_label)

        locked_map.pop(module_id, None)
        if not locked_map:
            self._ship_locked_module_charges.pop(ship_id, None)

        return True, tr(lang, "status_lock_clear_done", module=module_label)

    def current_language(self) -> str:
        lang = (self.prefs.language or "zh_CN").strip()
        return lang if lang in ("zh_CN", "en_US") else "zh_CN"

    def _create_menu(self) -> None:
        lang = self.current_language()
        self.menu_overview = self.menuBar().addMenu(tr(lang, "tab_overview"))
        self.act_overview_filter = QAction(tr(lang, "menu_overview_filter"), self)
        self.act_overview_filter.triggered.connect(self.open_overview_options)
        self.menu_overview.addAction(self.act_overview_filter)

        self.act_overview_reset = QAction(tr(lang, "menu_overview_reset"), self)
        self.act_overview_reset.triggered.connect(self.reset_overview_options)
        self.menu_overview.addAction(self.act_overview_reset)

    def _build_left_panel(self) -> QWidget:
        side = QWidget(self)
        side_layout = QVBoxLayout(side)
        side.setMinimumWidth(520)

        header = QHBoxLayout()
        self.lbl_selected_squad = QLabel(tr(self.current_language(), "selected_squad"))
        header.addWidget(self.lbl_selected_squad)
        self.squad_combo = QComboBox()
        self.squad_combo.setEditable(False)
        self.squad_combo.currentTextChanged.connect(self.on_selected_squad_changed)
        header.addWidget(self.squad_combo, 1)
        self.lbl_language = QLabel(tr(self.current_language(), "lang_label"))
        header.addWidget(self.lbl_language)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("中文", "zh_CN")
        self.lang_combo.addItem("English", "en_US")
        self.lang_combo.setCurrentIndex(0 if self.current_language() == "zh_CN" else 1)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        header.addWidget(self.lang_combo)
        side_layout.addLayout(header)

        leader_limit_row = QHBoxLayout()
        self.lbl_leader_speed_limit = QLabel(tr(self.current_language(), "leader_speed_limit"))
        leader_limit_row.addWidget(self.lbl_leader_speed_limit)
        self.spin_leader_speed_limit = QDoubleSpinBox(self)
        self.spin_leader_speed_limit.setDecimals(1)
        self.spin_leader_speed_limit.setRange(0.0, 1_000_000.0)
        self.spin_leader_speed_limit.setSingleStep(50.0)
        self.spin_leader_speed_limit.setValue(0.0)
        leader_limit_row.addWidget(self.spin_leader_speed_limit, 1)
        side_layout.addLayout(leader_limit_row)

        buttons_top2 = QHBoxLayout()
        self.btn_propulsion_toggle = QPushButton(tr(self.current_language(), "btn_prop_on"))
        buttons_top2.addWidget(self.btn_propulsion_toggle)
        self.btn_clear_focus = QPushButton(tr(self.current_language(), "btn_clear_focus"))
        buttons_top2.addWidget(self.btn_clear_focus)
        side_layout.addLayout(buttons_top2)

        ammo_layout = QVBoxLayout()
        ammo_row1 = QHBoxLayout()
        self.lbl_freq_charge_module = QLabel(tr(self.current_language(), "freq_charge_module"))
        ammo_row1.addWidget(self.lbl_freq_charge_module)
        self.charge_module_combo = QComboBox()
        self.charge_module_combo.setMinimumWidth(260)
        ammo_row1.addWidget(self.charge_module_combo, 1)
        ammo_layout.addLayout(ammo_row1)

        ammo_row2 = QHBoxLayout()
        self.lbl_ammo = QLabel(tr(self.current_language(), "ammo"))
        ammo_row2.addWidget(self.lbl_ammo)
        self.ammo_combo = QComboBox()
        self.ammo_combo.setMinimumWidth(260)
        ammo_row2.addWidget(self.ammo_combo, 1)
        self.apply_ammo_btn = QPushButton(tr(self.current_language(), "apply_all"))
        ammo_row2.addWidget(self.apply_ammo_btn)
        ammo_layout.addLayout(ammo_row2)
        side_layout.addLayout(ammo_layout)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_overview_tab(), tr(self.current_language(), "tab_overview"))
        self.tabs.addTab(self._build_fleet_tab(), tr(self.current_language(), "tab_fleet"))
        side_layout.addWidget(self.tabs, 1)

        self.status = QLabel(f"{tr(self.current_language(), 'status_prefix')}: 0")
        side_layout.addWidget(self.status)

        self.btn_propulsion_toggle.clicked.connect(self.toggle_selected_squad_propulsion)
        self.btn_clear_focus.clicked.connect(self.clear_focus_targets)
        self.spin_leader_speed_limit.valueChanged.connect(self.on_selected_squad_leader_speed_limit_changed)
        self.charge_module_combo.currentTextChanged.connect(self._on_charge_module_changed)
        self.apply_ammo_btn.clicked.connect(self._apply_selected_ammo)
        self._refresh_common_charge_modules()
        self._refresh_selected_squad_leader_speed_limit()
        self._refresh_propulsion_button_text()
        return side

    def _get_squad_propulsion_state(self, squad_id: str) -> bool:
        return self._get_team_propulsion_state(self.controlled_team, squad_id)

    def _get_team_propulsion_state(self, team: Team, squad_id: str) -> bool:
        key = self._focus_key(team, squad_id)
        if key in self.engine.world.squad_propulsion_commands:
            return bool(self.engine.world.squad_propulsion_commands.get(key, False))
        return bool(self.engine.world.squad_propulsion_commands.get(squad_id, False))

    def _set_team_propulsion_state(self, team: Team, squad_id: str, active: bool) -> None:
        key = self._focus_key(team, squad_id)
        self.engine.world.squad_propulsion_commands[key] = bool(active)
        self.engine.world.squad_propulsion_commands.pop(squad_id, None)

    def _get_team_intent(self, team: Team, squad_id: str) -> FleetIntent | None:
        key = self._focus_key(team, squad_id)
        intent = self.engine.world.intents.get(key)
        if intent is not None:
            return intent
        return self.engine.world.intents.get(squad_id)

    def _set_team_intent(self, team: Team, squad_id: str, intent: FleetIntent) -> None:
        key = self._focus_key(team, squad_id)
        self.engine.world.intents[key] = intent
        self.engine.world.intents.pop(squad_id, None)

    def on_language_changed(self, _index: int) -> None:
        lang = str(self.lang_combo.currentData() or "zh_CN")
        self.prefs.language = lang
        self.store.save(self.prefs)
        self.retranslate_ui()
        self._refresh_common_charge_modules()
        self.request_overview_refresh(force=True)

    def retranslate_ui(self) -> None:
        lang = self.current_language()
        self.setWindowTitle(tr(lang, "app_title"))
        self.menu_overview.setTitle(tr(lang, "tab_overview"))
        self.act_overview_filter.setText(tr(lang, "menu_overview_filter"))
        self.act_overview_reset.setText(tr(lang, "menu_overview_reset"))
        self.lbl_selected_squad.setText(tr(lang, "selected_squad"))
        self.lbl_language.setText(tr(lang, "lang_label"))
        self.lbl_leader_speed_limit.setText(tr(lang, "leader_speed_limit"))
        self.btn_clear_focus.setText(tr(lang, "btn_clear_focus"))
        self.lbl_freq_charge_module.setText(tr(lang, "freq_charge_module"))
        self.lbl_ammo.setText(tr(lang, "ammo"))
        self.apply_ammo_btn.setText(tr(lang, "apply_all"))
        self.tabs.setTabText(0, tr(lang, "tab_overview"))
        self.tabs.setTabText(1, tr(lang, "tab_fleet"))
        self.lbl_overview_hint.setText(tr(lang, "overview_hint"))
        self.lbl_fleet_tip.setText(tr(lang, "fleet_tip"))
        self.lbl_target_squad.setText(tr(lang, "target_squad"))
        self.btn_assign.setText(tr(lang, "assign_btn"))
        self.overview_model.notify_headers_changed()
        self.blue_roster_model.notify_headers_changed()
        self._refresh_propulsion_button_text()

    def _refresh_propulsion_button_text(self) -> None:
        active = self._get_squad_propulsion_state(self.ui_state.selected_squad)
        lang = self.current_language()
        self.btn_propulsion_toggle.setText(tr(lang, "btn_prop_off") if active else tr(lang, "btn_prop_on"))

    def _is_ammo_configurable_team(self, team: Team) -> bool:
        if self.network_mode == "local":
            return team == self.controlled_team
        return True

    def toggle_selected_squad_propulsion(self) -> None:
        squad = self.ui_state.selected_squad
        new_state = not self._get_squad_propulsion_state(squad)
        self._log_user_action("toggle_propulsion", squad=squad, enabled=new_state)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_PROPULSION, "squad_id": squad, "active": new_state})
            old = self._get_team_intent(self.controlled_team, squad)
            self._set_team_propulsion_state(self.controlled_team, squad, new_state)
            self._set_team_intent(self.controlled_team, squad, FleetIntent(
                squad_id=squad,
                target_position=old.target_position if old else None,
                focus_target=old.focus_target if old else None,
                propulsion_active=new_state,
            ))
            self._refresh_propulsion_button_text()
            return

        old = self._get_team_intent(self.controlled_team, squad)
        self._set_team_propulsion_state(self.controlled_team, squad, new_state)
        self._set_team_intent(self.controlled_team, squad, FleetIntent(
            squad_id=squad,
            target_position=old.target_position if old else None,
            focus_target=old.focus_target if old else None,
            propulsion_active=new_state,
        ))
        self._refresh_propulsion_button_text()

    def _refresh_common_charge_modules(self) -> None:
        fit_texts = [r.fit_text for r in self.manual_setup if self._is_ammo_configurable_team(r.team)]
        current = self.charge_module_combo.currentText()
        charge_modules = get_common_chargeable_modules(fit_texts, usage_threshold=0.0, language=self.current_language())
        self.charge_module_combo.blockSignals(True)
        self.charge_module_combo.clear()
        self.charge_module_combo.addItems(charge_modules)
        if current and current in charge_modules:
            self.charge_module_combo.setCurrentText(current)
        self.charge_module_combo.blockSignals(False)

        for module_name in charge_modules:
            ammo_list = get_charge_options_for_module(module_name, language=self.current_language())
            if not ammo_list:
                continue
            selected = self._charge_module_ammo_selection.get(module_name)
            if not selected or selected not in ammo_list:
                self._charge_module_ammo_selection[module_name] = ammo_list[0]
            try:
                self._factory.set_charge_module_ammo_override(
                    module_name,
                    self._charge_module_ammo_selection[module_name],
                )
            except Exception:
                continue

        self._on_charge_module_changed(self.charge_module_combo.currentText())

    def _on_charge_module_changed(self, module_name: str) -> None:
        self.ammo_combo.clear()
        if not module_name:
            return
        ammo = get_charge_options_for_module(module_name, language=self.current_language())
        self.ammo_combo.addItems(ammo)
        if not ammo:
            return
        selected = self._charge_module_ammo_selection.get(module_name)
        if not selected or selected not in ammo:
            selected = ammo[0]
            self._charge_module_ammo_selection[module_name] = selected
            try:
                self._factory.set_charge_module_ammo_override(module_name, selected)
            except Exception:
                pass
        self.ammo_combo.setCurrentText(selected)

    def _apply_selected_ammo(self) -> None:
        lang = self.current_language()
        module_name = self.charge_module_combo.currentText().strip()
        ammo_name = self.ammo_combo.currentText().strip()
        if not module_name or not ammo_name:
            return

        self._charge_module_ammo_selection[module_name] = ammo_name
        self._factory.set_charge_module_ammo_override(module_name, ammo_name)

        reload_sec = max(0.0, get_module_reload_time_sec(module_name))
        updated_ships = 0
        ship_ids = list(self.engine.world.ships.keys())
        for idx, ship_id in enumerate(ship_ids):
            ship = self.engine.world.ships.get(ship_id)
            if ship is None:
                continue
            if not self._is_ammo_configurable_team(ship.team):
                continue

            old_text = self._ship_fit_texts.get(ship_id)
            if old_text is None and idx < len(self.manual_setup):
                old_text = self.manual_setup[idx].fit_text
            if old_text is None:
                continue

            old_text = str(old_text)
            try:
                parsed_old = self._parser.parse(old_text)
            except Exception:
                parsed_old = None

            new_text = self._rewrite_fit_text_with_lock_rules(
                ship_id,
                old_text,
                target_module_name=module_name,
                target_ammo_name=ammo_name,
            )
            if new_text == old_text:
                continue

            ok, message, parsed_new = self._rebuild_ship_from_fit_text(ship_id, new_text, lang)
            if not ok:
                QMessageBox.warning(
                    self,
                    tr(lang, "ammo_title"),
                    tr(lang, "ammo_rebuild_failed", ship=ship_id, error=message),
                )
                continue

            if reload_sec > 0.0:
                changed_module_ids: list[str] = []
                if parsed_old is not None and parsed_new is not None:
                    changed_module_ids = self._changed_module_ids_between_parsed(parsed_old, parsed_new)

                for module_id in changed_module_ids:
                    cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module_id, 0.0) or 0.0))
                    active_reload_left = max(0.0, float(ship.combat.module_ammo_reload_timers.get(module_id, 0.0) or 0.0))
                    if cycle_left > 0.0 or active_reload_left > 0.0:
                        ship.combat.module_pending_ammo_reload_timers[module_id] = reload_sec
                    else:
                        ship.combat.module_ammo_reload_timers[module_id] = reload_sec
                        ship.combat.module_pending_ammo_reload_timers.pop(module_id, None)
            updated_ships += 1

        self.request_overview_refresh(force=True)
        self.canvas.update()
        if updated_ships <= 0:
            QMessageBox.information(self, tr(lang, "ammo_title"), tr(lang, "ammo_no_replace"))
            return
        QMessageBox.information(
            self,
            tr(lang, "ammo_title"),
            tr(lang, "ammo_switch_done", module=module_name, ammo=ammo_name, count=updated_ships, reload=reload_sec),
        )
        self._log_user_action(
            "apply_ammo",
            module=module_name,
            ammo=ammo_name,
            updated_ships=updated_ships,
            reload_sec=reload_sec,
        )

    def _build_overview_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self.lbl_overview_hint = QLabel(tr(self.current_language(), "overview_hint"))
        layout.addWidget(self.lbl_overview_hint)

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
        self.overview.setAlternatingRowColors(True)
        self.overview.setWordWrap(False)
        self.overview.setSortingEnabled(True)
        self.overview.verticalHeader().setVisible(False)
        self.overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 4):
            self.overview.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.overview, 1)
        return page

    def _build_fleet_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self.lbl_fleet_tip = QLabel(tr(self.current_language(), "fleet_tip"))
        layout.addWidget(self.lbl_fleet_tip)

        self.blue_roster = QTableView(self)
        self.blue_roster_model = BlueRosterTableModel(self.current_language)
        self.blue_roster.setModel(self.blue_roster_model)
        self.blue_roster.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.blue_roster.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.blue_roster.setWordWrap(False)
        self.blue_roster.verticalHeader().setVisible(False)
        self.blue_roster.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            self.blue_roster.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.blue_roster, 1)

        controls = QHBoxLayout()
        self.assign_squad_edit = QLineEdit(self.ui_state.selected_squad)
        self.btn_assign = QPushButton(tr(self.current_language(), "assign_btn"))
        self.lbl_target_squad = QLabel(tr(self.current_language(), "target_squad"))
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
            zoom=zoom,
            language=self.prefs.language,
            engine_tick_rate=self.prefs.engine_tick_rate,
            engine_physics_substeps=self.prefs.engine_physics_substeps,
            engine_lockstep=self.prefs.engine_lockstep,
            engine_battlefield_radius=self.prefs.engine_battlefield_radius,
            engine_detailed_logging=self.prefs.engine_detailed_logging,
            engine_hotspot_logging=self.prefs.engine_hotspot_logging,
            engine_detail_log_file=self.prefs.engine_detail_log_file,
            engine_hotspot_log_file=self.prefs.engine_hotspot_log_file,
            engine_log_merge_window_sec=self.prefs.engine_log_merge_window_sec,
        )
        self.store.save(self.prefs)
        self.overview_proxy.apply_preferences()
        self.request_overview_refresh(force=True)

    def _controlled_squad_ids(self) -> list[str]:
        return sorted(
            {
                s.squad_id
                for s in self.engine.world.ships.values()
                if s.team == self.controlled_team and s.ship_id not in self._undeployed_ship_ids
            }
        )

    def _sync_blue_squads(self) -> None:
        squads = self._controlled_squad_ids()
        if self.controlled_team == Team.BLUE:
            self.blue_commander.squad_ids = squads
        else:
            self.red_commander.squad_ids = squads
        current = self.squad_combo.currentText().strip() or self.ui_state.selected_squad

        self.squad_combo.blockSignals(True)
        self.squad_combo.clear()
        self.squad_combo.addItems(squads)
        if current and current in squads:
            self.squad_combo.setCurrentText(current)
        elif squads:
            self.squad_combo.setCurrentText(squads[0])
        self.squad_combo.blockSignals(False)

        self.on_selected_squad_changed(self.squad_combo.currentText())

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
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _selected_squad_leader_speed_limit_key(self) -> str:
        return f"{self.controlled_team.value}:{self.ui_state.selected_squad}"

    def _refresh_selected_squad_leader_speed_limit(self) -> None:
        key = self._selected_squad_leader_speed_limit_key()
        value = float(self.engine.world.squad_leader_speed_limits.get(key, 0.0) or 0.0)
        self.spin_leader_speed_limit.blockSignals(True)
        self.spin_leader_speed_limit.setValue(value)
        self.spin_leader_speed_limit.blockSignals(False)

    def on_selected_squad_leader_speed_limit_changed(self, value: float) -> None:
        key = self._selected_squad_leader_speed_limit_key()
        if value <= 0.0:
            self.engine.world.squad_leader_speed_limits.pop(key, None)
        else:
            self.engine.world.squad_leader_speed_limits[key] = float(value)
        self._log_user_action("leader_speed_limit", squad=self.ui_state.selected_squad, limit=value)
        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command(
                {
                    "kind": CMD_SQUAD_LEADER_SPEED_LIMIT,
                    "squad_id": self.ui_state.selected_squad,
                    "limit": float(value),
                }
            )

    def on_canvas_select_squad(self, squad_id: str) -> None:
        self.squad_combo.setCurrentText(squad_id)

    def on_canvas_select_enemy(self, ship_id: str) -> None:
        self.ui_state.selected_enemy_target = ship_id
        self.canvas.selected_enemy_target = ship_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)
        self.canvas.update()

    def _selected_anchor(self) -> Vector2:
        members = [
            s
            for s in self.engine.world.ships.values()
            if s.team == self.controlled_team and s.squad_id == self.ui_state.selected_squad and s.vital.alive
        ]
        if not members:
            return Vector2(0.0, 0.0)
        return Vector2(
            sum(m.nav.position.x for m in members) / len(members),
            sum(m.nav.position.y for m in members) / len(members),
        )

    @staticmethod
    def _focus_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    @staticmethod
    def _random_point_in_radius(center: Vector2, radius: float) -> Vector2:
        angle = random.uniform(0.0, math.tau)
        distance = radius * math.sqrt(random.random())
        return Vector2(center.x + math.cos(angle) * distance, center.y + math.sin(angle) * distance)

    def _apply_induce_spawn(self, team: Team, center: Vector2, squad_id: str | None = None) -> None:
        affected_squads: set[str] = set()
        for ship in self.engine.world.ships.values():
            if ship.team != team:
                continue
            if squad_id is not None and ship.squad_id != squad_id:
                continue
            if ship.ship_id not in self._undeployed_ship_ids:
                continue
            affected_squads.add(ship.squad_id)
            ship.nav.position = self._random_point_in_radius(center, 5_000.0)
            ship.nav.velocity = Vector2(0.0, 0.0)
            ship.order_queue.clear()
            ship.vital.alive = True
            ship.vital.shield = ship.vital.shield_max
            ship.vital.armor = ship.vital.armor_max
            ship.vital.structure = ship.vital.structure_max
            ship.vital.cap = ship.vital.cap_max
            ship.combat.current_target = None
            ship.combat.last_attack_target = None
            ship.combat.lock_targets.clear()
            ship.combat.lock_timers.clear()
            ship.combat.fire_delay_timers.clear()
            self._undeployed_ship_ids.discard(ship.ship_id)
        for squad in affected_squads:
            scoped_key = self._focus_key(team, squad)
            self._squad_approach_targets.pop(scoped_key, None)
            self._squad_guidance_targets.pop(scoped_key, None)
            # Backward compatibility for stale unscoped cache keys.
            self._squad_approach_targets.pop(squad, None)
            self._squad_guidance_targets.pop(squad, None)
            old = self._get_team_intent(team, squad)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=None,
                focus_target=old.focus_target if old else None,
                propulsion_active=old.propulsion_active if old else None,
            ))
        if affected_squads:
            self._sync_blue_squads()

    def induce_spawn_squad_at(self, squad_id: str, target: Vector2) -> None:
        squad = squad_id.strip()
        if not squad:
            return
        self._log_user_action("induce_squad", squad=squad, x=target.x, y=target.y)
        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_INDUCE_SQUAD_AT, "squad_id": squad, "x": target.x, "y": target.y})
            return

        def apply() -> None:
            self._apply_induce_spawn(self.controlled_team, target, squad)
            self.request_overview_refresh(force=True)

        self._enqueue_tick_op(apply)

    def induce_spawn_fleet_at(self, target: Vector2) -> None:
        self._log_user_action("induce_fleet", x=target.x, y=target.y)
        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_INDUCE_FLEET_AT, "x": target.x, "y": target.y})
            return

        def apply() -> None:
            self._apply_induce_spawn(self.controlled_team, target, None)
            self.request_overview_refresh(force=True)

        self._enqueue_tick_op(apply)

    def issue_approach_target(self, squad_id: str, target_id: str) -> None:
        squad = squad_id.strip()
        target = target_id.strip()
        if not squad or not target:
            return
        self._log_user_action("squad_approach", squad=squad, target=target)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_APPROACH, "squad_id": squad, "target_id": target})
            scoped_key = self._focus_key(self.controlled_team, squad)
            self._squad_approach_targets[scoped_key] = target
            target_ship = self.engine.world.ships.get(target)
            if target_ship is not None and target_ship.vital.alive:
                self._squad_guidance_targets[scoped_key] = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
                old = self._get_team_intent(self.controlled_team, squad)
                prop_state = self._get_team_propulsion_state(self.controlled_team, squad)
                self._set_team_intent(self.controlled_team, squad, FleetIntent(
                    squad_id=squad,
                    target_position=Vector2(target_ship.nav.position.x, target_ship.nav.position.y),
                    focus_target=old.focus_target if old else None,
                    propulsion_active=prop_state,
                ))
            return

        def apply() -> None:
            target_ship = self.engine.world.ships.get(target)
            if target_ship is None or not target_ship.vital.alive:
                return
            scoped_key = self._focus_key(self.controlled_team, squad)
            self._squad_approach_targets[scoped_key] = target
            self._squad_guidance_targets[scoped_key] = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
            old = self._get_team_intent(self.controlled_team, squad)
            prop_state = self._get_team_propulsion_state(self.controlled_team, squad)
            self._set_team_intent(self.controlled_team, squad, FleetIntent(
                squad_id=squad,
                target_position=Vector2(target_ship.nav.position.x, target_ship.nav.position.y),
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))

        self._enqueue_tick_op(apply)

    def _update_approach_targets(self) -> None:
        if not self._squad_approach_targets:
            return
        stale: list[str] = []
        for scoped_key, target_id in list(self._squad_approach_targets.items()):
            team, squad = self._parse_team_squad_key(scoped_key, self.controlled_team)
            target_ship = self.engine.world.ships.get(target_id)
            if target_ship is None or not target_ship.vital.alive:
                stale.append(scoped_key)
                continue
            scoped_write_key = self._focus_key(team, squad)
            self._squad_guidance_targets[scoped_write_key] = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
            old = self._get_team_intent(team, squad)
            prop_state = self._get_team_propulsion_state(team, squad)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=Vector2(target_ship.nav.position.x, target_ship.nav.position.y),
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))
        for squad in stale:
            self._squad_approach_targets.pop(squad, None)
            self._squad_guidance_targets.pop(squad, None)

    def issue_move_to(self, squad_id: str, target: Vector2) -> None:
        self._log_user_action("squad_move", squad=squad_id, x=target.x, y=target.y)
        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_MOVE, "squad_id": squad_id, "x": target.x, "y": target.y})
            scoped_key = self._focus_key(self.controlled_team, squad_id)
            self._squad_approach_targets.pop(scoped_key, None)
            self._squad_guidance_targets[scoped_key] = Vector2(target.x, target.y)
            old = self._get_team_intent(self.controlled_team, squad_id)
            prop_state = self._get_team_propulsion_state(self.controlled_team, squad_id)
            self._set_team_intent(self.controlled_team, squad_id, FleetIntent(
                squad_id=squad_id,
                target_position=target,
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))
            return

        def apply() -> None:
            scoped_key = self._focus_key(self.controlled_team, squad_id)
            self._squad_approach_targets.pop(scoped_key, None)
            self._squad_guidance_targets[scoped_key] = Vector2(target.x, target.y)
            members = [
                s
                for s in self.engine.world.ships.values()
                if s.team == self.controlled_team and s.squad_id == squad_id and s.vital.alive
            ]
            for ship in members:
                ship.order_queue = [o for o in ship.order_queue if o.kind != "MOVE"]

            old = self._get_team_intent(self.controlled_team, squad_id)
            prop_state = self._get_team_propulsion_state(self.controlled_team, squad_id)
            self._set_team_intent(self.controlled_team, squad_id, FleetIntent(
                squad_id=squad_id,
                target_position=target,
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))

        self._enqueue_tick_op(apply)

    def issue_focus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        focus_key = self._focus_key(self.controlled_team, squad)
        self._log_user_action("squad_focus", squad=squad, target=target_id)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_ATTACK, "squad_id": squad, "target_id": target_id})
            queue = list(self.engine.world.squad_focus_queues.get(focus_key, []))
            queue = [target_id] + [tid for tid in queue if tid != target_id]
            self.engine.world.squad_focus_queues[focus_key] = queue
            self.ui_state.selected_enemy_target = target_id
            self.canvas.selected_enemy_target = target_id
            self.overview_model.notify_visual_state_changed()
            self.request_overview_refresh(force=True)
            return

        def apply() -> None:
            old = self._get_team_intent(self.controlled_team, squad)
            prop_state = self._get_team_propulsion_state(self.controlled_team, squad)
            self._set_team_intent(self.controlled_team, squad, FleetIntent(
                squad_id=squad,
                target_position=old.target_position if old else None,
                focus_target=target_id,
                propulsion_active=prop_state,
            ))

        self._enqueue_tick_op(apply)
        self.ui_state.selected_enemy_target = target_id
        self.canvas.selected_enemy_target = target_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def issue_prefocus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        focus_key = self._focus_key(self.controlled_team, squad)
        self._log_user_action("squad_prefocus", squad=squad, target=target_id)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_PREFOCUS, "squad_id": squad, "target_id": target_id})
            queue = list(self.engine.world.squad_focus_queues.get(focus_key, []))
            if target_id not in queue:
                queue.append(target_id)
            self.engine.world.squad_focus_queues[focus_key] = queue
            self.ui_state.selected_enemy_target = target_id
            self.canvas.selected_enemy_target = target_id
            self.overview_model.notify_visual_state_changed()
            self.request_overview_refresh(force=True)
            return

        def apply() -> None:
            members = [
                s
                for s in self.engine.world.ships.values()
                if s.team == self.controlled_team and s.squad_id == squad and s.vital.alive
            ]
            queue = list(self.engine.world.squad_focus_queues.get(focus_key, []))
            if not queue:
                for ship in members:
                    current_id = ship.combat.current_target
                    if not current_id:
                        continue
                    target = self.engine.world.ships.get(current_id)
                    if target is not None and target.vital.alive and target.team != ship.team:
                        queue.append(current_id)
                        break
            if target_id not in queue:
                queue.append(target_id)
            self.engine.world.squad_focus_queues[focus_key] = queue

        self._enqueue_tick_op(apply)
        self.ui_state.selected_enemy_target = target_id
        self.canvas.selected_enemy_target = target_id
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def cancel_prefocus_target(self, target_id: str) -> None:
        squad = self.ui_state.selected_squad
        focus_key = self._focus_key(self.controlled_team, squad)
        self._log_user_action("squad_cancel_prefocus", squad=squad, target=target_id)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_CANCEL_PREFOCUS, "squad_id": squad, "target_id": target_id})
            queue = [tid for tid in self.engine.world.squad_focus_queues.get(focus_key, []) if tid != target_id]
            self.engine.world.squad_focus_queues[focus_key] = queue
            self.overview_model.notify_visual_state_changed()
            self.request_overview_refresh(force=True)
            return

        def apply() -> None:
            queue = list(self.engine.world.squad_focus_queues.get(focus_key, []))
            if queue:
                head = queue[0]
                tail = [tid for tid in queue[1:] if tid != target_id]
                if head == target_id:
                    queue = tail
                else:
                    queue = [head] + tail
                self.engine.world.squad_focus_queues[focus_key] = queue

            self._discard_squad_prelock_target(focus_key, target_id)

            for ship in self.engine.world.ships.values():
                if ship.team != self.controlled_team or ship.squad_id != squad:
                    continue
                ship.combat.lock_targets.discard(target_id)
                ship.combat.lock_timers.pop(target_id, None)
                ship.combat.fire_delay_timers.pop(target_id, None)

        self._enqueue_tick_op(apply)

    def clear_focus_targets(self) -> None:
        squad = self.ui_state.selected_squad
        focus_key = self._focus_key(self.controlled_team, squad)
        self._log_user_action("squad_clear_focus", squad=squad)

        if self.network_mode == "client" and self.lan_client is not None:
            self.lan_client.send_command({"kind": CMD_SQUAD_CLEAR_FOCUS, "squad_id": squad})
            self.engine.world.squad_focus_queues.pop(focus_key, None)
            self.ui_state.selected_enemy_target = None
            self.canvas.selected_enemy_target = None
            self.overview_model.notify_visual_state_changed()
            self.request_overview_refresh(force=True)
            return

        def apply() -> None:
            self.engine.world.squad_focus_queues.pop(focus_key, None)
            self.engine.world.squad_prelocked_targets.pop(focus_key, None)
            self.engine.world.squad_prelock_timers.pop(focus_key, None)
            old = self._get_team_intent(self.controlled_team, squad)
            prop_state = self._get_team_propulsion_state(self.controlled_team, squad)
            self._set_team_intent(self.controlled_team, squad, FleetIntent(
                squad_id=squad,
                target_position=old.target_position if old else None,
                focus_target=None,
                propulsion_active=prop_state,
            ))
            for ship in self.engine.world.ships.values():
                if ship.team != self.controlled_team or ship.squad_id != squad:
                    continue
                ship.order_queue = [o for o in ship.order_queue if o.kind != "ATTACK"]
                ship.combat.current_target = None
                ship.combat.last_attack_target = None
                ship.combat.lock_targets.clear()
                ship.combat.lock_timers.clear()
                ship.combat.fire_delay_timers.clear()

        self._enqueue_tick_op(apply)
        self.ui_state.selected_enemy_target = None
        self.canvas.selected_enemy_target = None
        self.overview_model.notify_visual_state_changed()
        self.request_overview_refresh(force=True)

    def _iter_overview_rows(self) -> list[dict]:
        anchor = self._selected_anchor()
        rows: list[dict] = []
        for ship in self.engine.world.ships.values():
            if ship.ship_id in self._undeployed_ship_ids:
                continue
            hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
            hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
            hp_pct = round(100.0 * hp_cur / hp_max, 1) if hp_max > 0 else 0.0
            dist_km = round(ship.nav.position.distance_to(anchor) / 1000.0, 1)
            rows.append(
                {
                    "id": ship.ship_id,
                    "ship_type": ship.fit.ship_name,
                    "ship_type_display": get_type_display_name(ship.fit.ship_name, language=self.current_language()),
                    "team": ship.team.value,
                    "squad": ship.squad_id,
                    "role": ship.fit.role,
                    "alive": ship.vital.alive,
                    "dist": dist_km,
                    "hp": hp_pct,
                    "dps": round(ship.profile.dps, 1),
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

    def show_overview_menu(self, pos: QPoint) -> None:
        index = self.overview.indexAt(pos)
        if not index.isValid():
            return
        row_data = self.overview_proxy.get_row(index.row())
        if not row_data:
            return
        target_id = row_data["id"]
        target_team = row_data["team"]
        is_alive = bool(row_data["alive"])
        if not is_alive:
            return

        lang = self.current_language()
        menu = QMenu(self)
        action_status = QAction(tr(lang, "menu_show_status", ship=target_id), self)
        action_status.triggered.connect(lambda: self.show_ship_status(target_id))
        menu.addAction(action_status)
        enemy_team = Team.RED.value if self.controlled_team == Team.BLUE else Team.BLUE.value
        if target_team == enemy_team:
            action_focus = QAction(tr(lang, "menu_focus", squad=self.ui_state.selected_squad, ship=target_id), self)
            action_focus.triggered.connect(lambda: self.issue_focus_target(target_id))
            menu.addAction(action_focus)
        menu.exec(self.overview.mapToGlobal(pos))

    def refresh_blue_roster(self) -> None:
        ships = sorted(
            [s for s in self.engine.world.ships.values() if s.team == self.controlled_team and s.ship_id not in self._undeployed_ship_ids],
            key=lambda s: s.ship_id,
        )
        rows: list[dict] = []
        for ship in ships:
            hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
            hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
            hp_pct = 100.0 * hp_cur / hp_max if hp_max > 0 else 0.0
            rows.append(
                {
                    "ship_id": ship.ship_id,
                    "squad": ship.squad_id,
                    "role": ship.fit.role,
                    "alive": ship.vital.alive,
                    "hp": hp_pct,
                }
            )
        self.blue_roster_model.set_rows(rows)

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

        def apply() -> None:
            for ship_id in ship_ids:
                ship = self.engine.world.ships.get(ship_id)
                if ship and ship.team == self.controlled_team:
                    ship.squad_id = target_squad
            self._sync_blue_squads()
            self.refresh_blue_roster()
            self.request_overview_refresh(force=True)

        self._enqueue_tick_op(apply)

    def _apply_remote_command(self, cmd: dict) -> None:
        kind = str(cmd.get("kind", "")).upper()
        self._lan_debug(f"recv-cmd kind={kind} payload={cmd}")
        self._log_user_action(
            "remote_command",
            kind=kind,
            squad=str(cmd.get("squad_id", "") or ""),
            target=str(cmd.get("target_id", "") or ""),
        )
        team = Team.RED
        if kind == CMD_SQUAD_LEADER_SPEED_LIMIT:
            squad = str(cmd.get("squad_id", "")).strip()
            if not squad:
                return
            limit = float(cmd.get("limit", 0.0) or 0.0)
            key = f"{team.value}:{squad}"
            if limit <= 0.0:
                self.engine.world.squad_leader_speed_limits.pop(key, None)
            else:
                self.engine.world.squad_leader_speed_limits[key] = limit
            return
        if kind == CMD_SYNC_SETUP:
            ships = cmd.get("ships")
            if not isinstance(ships, list):
                return
            for item in ships:
                if not isinstance(item, dict):
                    continue
                ship_id = str(item.get("ship_id", "")).strip()
                if not ship_id:
                    continue
                ship = self._ensure_remote_ship(ship_id, item)
                ship.team = Team.BLUE if str(item.get("team", "RED")).upper() == "BLUE" else Team.RED
                ship.squad_id = str(item.get("squad_id", ship.squad_id))
                raw_pos = item.get("position")
                raw_vel = item.get("velocity")
                pos = raw_pos if isinstance(raw_pos, dict) else {}
                vel = raw_vel if isinstance(raw_vel, dict) else {}
                ship.nav.position = Vector2(float(pos.get("x", ship.nav.position.x)), float(pos.get("y", ship.nav.position.y)))
                ship.nav.velocity = Vector2(float(vel.get("x", ship.nav.velocity.x)), float(vel.get("y", ship.nav.velocity.y)))
                ship.nav.facing_deg = float(item.get("facing_deg", ship.nav.facing_deg))
                ship.vital.shield = float(item.get("shield", ship.vital.shield))
                ship.vital.armor = float(item.get("armor", ship.vital.armor))
                ship.vital.structure = float(item.get("structure", ship.vital.structure))
                ship.vital.cap = float(item.get("cap", ship.vital.cap))
                ship.vital.alive = bool(item.get("alive", ship.vital.alive))
                deployed = bool(item.get("deployed", ship.ship_id not in self._undeployed_ship_ids))
                if deployed:
                    self._undeployed_ship_ids.discard(ship.ship_id)
                else:
                    self._undeployed_ship_ids.add(ship.ship_id)
                    ship.vital.alive = False
            return

        squad = str(cmd.get("squad_id", "")).strip()
        if kind == CMD_INDUCE_FLEET_AT:
            try:
                center = Vector2(float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0)))
            except Exception:
                return
            self._apply_induce_spawn(team, center, None)
            return
        if kind == CMD_INDUCE_SQUAD_AT:
            if not squad:
                return
            try:
                center = Vector2(float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0)))
            except Exception:
                return
            self._apply_induce_spawn(team, center, squad)
            return
        if not squad:
            return
        if kind == CMD_SQUAD_MOVE:
            try:
                target = Vector2(float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0)))
            except Exception:
                return
            scoped_key = self._focus_key(team, squad)
            self._squad_approach_targets.pop(scoped_key, None)
            self._squad_guidance_targets[scoped_key] = Vector2(target.x, target.y)
            old = self._get_team_intent(team, squad)
            prop_state = self._get_team_propulsion_state(team, squad)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=target,
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))
        elif kind == CMD_SQUAD_APPROACH:
            target_id = str(cmd.get("target_id", "")).strip()
            target_ship = self.engine.world.ships.get(target_id)
            if not target_id or target_ship is None or not target_ship.vital.alive:
                return
            scoped_key = self._focus_key(team, squad)
            self._squad_approach_targets[scoped_key] = target_id
            self._squad_guidance_targets[scoped_key] = Vector2(target_ship.nav.position.x, target_ship.nav.position.y)
            old = self._get_team_intent(team, squad)
            prop_state = self._get_team_propulsion_state(team, squad)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=Vector2(target_ship.nav.position.x, target_ship.nav.position.y),
                focus_target=old.focus_target if old else None,
                propulsion_active=prop_state,
            ))
        elif kind == CMD_SQUAD_ATTACK:
            target_id = str(cmd.get("target_id", "")).strip()
            if not target_id:
                return
            old = self._get_team_intent(team, squad)
            prop_state = self._get_team_propulsion_state(team, squad)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=old.target_position if old else None,
                focus_target=target_id,
                propulsion_active=prop_state,
            ))
        elif kind == CMD_SQUAD_PROPULSION:
            old = self._get_team_intent(team, squad)
            new_state = bool(cmd.get("active", False))
            self._set_team_propulsion_state(team, squad, new_state)
            self._set_team_intent(team, squad, FleetIntent(
                squad_id=squad,
                target_position=old.target_position if old else None,
                focus_target=old.focus_target if old else None,
                propulsion_active=new_state,
            ))
        elif kind in SQUAD_FOCUS_COMMANDS:
            focus_key = self._focus_key(team, squad)
            target_id = str(cmd.get("target_id", "")).strip()
            if kind == CMD_SQUAD_PREFOCUS and target_id:
                queue = list(self.engine.world.squad_focus_queues.get(focus_key, []))
                if target_id not in queue:
                    queue.append(target_id)
                self.engine.world.squad_focus_queues[focus_key] = queue
            elif kind == CMD_SQUAD_CANCEL_PREFOCUS and target_id:
                queue = [tid for tid in self.engine.world.squad_focus_queues.get(focus_key, []) if tid != target_id]
                self.engine.world.squad_focus_queues[focus_key] = queue
                self._discard_squad_prelock_target(focus_key, target_id)
                for ship in self.engine.world.ships.values():
                    if ship.team != team or ship.squad_id != squad:
                        continue
                    ship.combat.lock_targets.discard(target_id)
                    ship.combat.lock_timers.pop(target_id, None)
                    ship.combat.fire_delay_timers.pop(target_id, None)
            elif kind == CMD_SQUAD_CLEAR_FOCUS:
                self.engine.world.squad_focus_queues.pop(focus_key, None)
                self.engine.world.squad_prelocked_targets.pop(focus_key, None)
                self.engine.world.squad_prelock_timers.pop(focus_key, None)
                old = self._get_team_intent(team, squad)
                prop_state = self._get_team_propulsion_state(team, squad)
                self._set_team_intent(team, squad, FleetIntent(
                    squad_id=squad,
                    target_position=old.target_position if old else None,
                    focus_target=None,
                    propulsion_active=prop_state,
                ))
                for ship in self.engine.world.ships.values():
                    if ship.team != team or ship.squad_id != squad:
                        continue
                    ship.order_queue = [o for o in ship.order_queue if o.kind != "ATTACK"]
                    ship.combat.current_target = None
                    ship.combat.last_attack_target = None
                    ship.combat.lock_targets.clear()
                    ship.combat.lock_timers.clear()
                    ship.combat.fire_delay_timers.clear()

    def _discard_squad_prelock_target(self, focus_key: str, target_id: str) -> None:
        prelocked_by_ship = self.engine.world.squad_prelocked_targets.get(focus_key)
        if isinstance(prelocked_by_ship, dict):
            for ship_id in list(prelocked_by_ship.keys()):
                targets = prelocked_by_ship.get(ship_id)
                if not isinstance(targets, set):
                    prelocked_by_ship.pop(ship_id, None)
                    continue
                targets.discard(target_id)
                if not targets:
                    prelocked_by_ship.pop(ship_id, None)
            if not prelocked_by_ship:
                self.engine.world.squad_prelocked_targets.pop(focus_key, None)

        timers_by_ship = self.engine.world.squad_prelock_timers.get(focus_key)
        if isinstance(timers_by_ship, dict):
            for ship_id in list(timers_by_ship.keys()):
                ship_timers = timers_by_ship.get(ship_id)
                if not isinstance(ship_timers, dict):
                    timers_by_ship.pop(ship_id, None)
                    continue
                ship_timers.pop(target_id, None)
                if not ship_timers:
                    timers_by_ship.pop(ship_id, None)
            if not timers_by_ship:
                self.engine.world.squad_prelock_timers.pop(focus_key, None)

    def _build_remote_ship_artifacts(
        self,
        ship_id: str,
        data: dict,
        *,
        existing: ShipEntity | None = None,
    ) -> tuple[FitDescriptor, ShipProfile, FitRuntime | None]:
        fit_text = str(data.get("fit_text", "") or "").strip()
        if fit_text:
            self._ship_fit_texts[ship_id] = fit_text
            try:
                parsed = self._parser.parse(fit_text)
                runtime_template, parsed_fit = self._factory.build(parsed)
                runtime = deepcopy(runtime_template)
                profile = self._factory.build_profile(parsed)
                return parsed_fit, profile, runtime
            except Exception:
                pass

        existing_fit = existing.fit if existing is not None else None
        existing_profile = existing.profile if existing is not None else None
        fit = FitDescriptor(
            fit_key=str(data.get("fit_key", getattr(existing_fit, "fit_key", f"remote-{ship_id}"))),
            ship_name=str(data.get("ship_name", getattr(existing_fit, "ship_name", ship_id))),
            role=str(data.get("fit_role", getattr(existing_fit, "role", "REMOTE"))),
            base_dps=float(data.get("base_dps", getattr(existing_fit, "base_dps", 0.0))),
            volley=float(data.get("fit_volley", getattr(existing_fit, "volley", 0.0))),
            optimal_range=float(data.get("fit_optimal_range", getattr(existing_fit, "optimal_range", 0.0))),
            falloff=float(data.get("fit_falloff", getattr(existing_fit, "falloff", 0.0))),
            tracking=float(data.get("fit_tracking", getattr(existing_fit, "tracking", 0.0))),
            max_speed=float(data.get("profile_max_speed", getattr(existing_fit, "max_speed", 1800.0))),
            max_cap=float(data.get("profile_max_cap", getattr(existing_fit, "max_cap", 4000.0))),
            cap_recharge_time=float(data.get("profile_cap_recharge_time", getattr(existing_fit, "cap_recharge_time", 450.0))),
            shield_hp=float(data.get("profile_shield_hp", getattr(existing_fit, "shield_hp", 5000.0))),
            armor_hp=float(data.get("profile_armor_hp", getattr(existing_fit, "armor_hp", 4000.0))),
            structure_hp=float(data.get("profile_structure_hp", getattr(existing_fit, "structure_hp", 4000.0))),
            rep_amount=float(data.get("profile_rep_amount", getattr(existing_fit, "rep_amount", 0.0))),
            rep_cycle=float(data.get("profile_rep_cycle", getattr(existing_fit, "rep_cycle", 5.0))),
        )
        profile = ShipProfile(
            dps=float(data.get("profile_dps", getattr(existing_profile, "dps", 0.0))),
            volley=float(data.get("profile_volley", getattr(existing_profile, "volley", 0.0))),
            optimal=float(data.get("profile_optimal", getattr(existing_profile, "optimal", 0.0))),
            falloff=float(data.get("profile_falloff", getattr(existing_profile, "falloff", 0.0))),
            tracking=float(data.get("profile_tracking", getattr(existing_profile, "tracking", 0.0))),
            sig_radius=float(data.get("profile_sig_radius", getattr(existing_profile, "sig_radius", 120.0))),
            scan_resolution=float(data.get("profile_scan_resolution", getattr(existing_profile, "scan_resolution", 300.0))),
            max_target_range=float(data.get("profile_max_target_range", getattr(existing_profile, "max_target_range", 120000.0))),
            max_locked_targets=int(data.get("profile_max_locked_targets", getattr(existing_profile, "max_locked_targets", 0)) or 0),
            scan_strength=float(data.get("profile_scan_strength", getattr(existing_profile, "scan_strength", 0.0))),
            ecm_jam_chance=float(data.get("profile_ecm_jam_chance", getattr(existing_profile, "ecm_jam_chance", 0.0))),
            warp_scramble_status=float(data.get("profile_warp_scramble_status", getattr(existing_profile, "warp_scramble_status", 0.0))),
            warp_stability=float(data.get("profile_warp_stability", getattr(existing_profile, "warp_stability", 0.0))),
            max_speed=float(data.get("profile_max_speed", getattr(existing_profile, "max_speed", 1800.0))),
            max_cap=float(data.get("profile_max_cap", getattr(existing_profile, "max_cap", 4000.0))),
            cap_recharge_time=float(data.get("profile_cap_recharge_time", getattr(existing_profile, "cap_recharge_time", 450.0))),
            shield_hp=float(data.get("profile_shield_hp", getattr(existing_profile, "shield_hp", 5000.0))),
            armor_hp=float(data.get("profile_armor_hp", getattr(existing_profile, "armor_hp", 4000.0))),
            structure_hp=float(data.get("profile_structure_hp", getattr(existing_profile, "structure_hp", 4000.0))),
            rep_amount=float(data.get("profile_rep_amount", getattr(existing_profile, "rep_amount", 0.0))),
            rep_cycle=float(data.get("profile_rep_cycle", getattr(existing_profile, "rep_cycle", 5.0))),
        )
        runtime = existing.runtime if existing is not None else None
        return fit, profile, runtime

    def _ensure_remote_ship(self, ship_id: str, data: dict) -> ShipEntity:
        existing = self.engine.world.ships.get(ship_id)
        fit_text_existing = str(data.get("fit_text", "") or "").strip()
        if existing is not None:
            if fit_text_existing:
                previous_fit_text = self._ship_fit_texts.get(ship_id, "")
                self._ship_fit_texts[ship_id] = fit_text_existing
                if fit_text_existing != previous_fit_text or existing.runtime is None:
                    fit, profile, runtime = self._build_remote_ship_artifacts(ship_id, data, existing=existing)
                    existing.fit = fit
                    existing.profile = profile
                    existing.runtime = runtime
                    existing.nav.max_speed = profile.max_speed
            return existing
        team_text = str(data.get("team", "BLUE"))
        team = Team.BLUE if team_text == "BLUE" else Team.RED
        fit, profile, runtime = self._build_remote_ship_artifacts(ship_id, data)
        ship = ShipEntity(
            ship_id=ship_id,
            team=team,
            squad_id=str(data.get("squad_id", "")),
            fit=fit,
            profile=profile,
            nav=NavigationState(
                position=Vector2(float(data.get("x", 0.0)), float(data.get("y", 0.0))),
                velocity=Vector2(0.0, 0.0),
                facing_deg=float(data.get("facing_deg", 0.0)),
                max_speed=float(data.get("profile_max_speed", 1800.0)),
            ),
            combat=CombatState(),
            vital=VitalState(
                shield=float(data.get("shield", max(1.0, profile.shield_hp))),
                armor=float(data.get("armor", max(1.0, profile.armor_hp))),
                structure=float(data.get("structure", max(1.0, profile.structure_hp))),
                shield_max=float(data.get("vital_shield_max", max(1.0, profile.shield_hp))),
                armor_max=float(data.get("vital_armor_max", max(1.0, profile.armor_hp))),
                structure_max=float(data.get("vital_structure_max", max(1.0, profile.structure_hp))),
                cap=float(data.get("cap", profile.max_cap)),
                cap_max=float(data.get("vital_cap_max", profile.max_cap)),
                alive=bool(data.get("alive", True)),
            ),
            quality=QualityState(
                level=QualityLevel(str(data.get("quality_level", "REGULAR"))),
                reaction_delay=float(data.get("quality_reaction_delay", 0.0)),
                ignore_order_probability=float(data.get("quality_ignore_order_probability", 0.0)),
                formation_jitter=float(data.get("quality_formation_jitter", 0.0)),
            ),
            runtime=runtime,
        )
        self.engine.world.ships[ship_id] = ship
        self.engine.register_ship(ship_id)
        if self.network_mode == "host" and team != self.controlled_team:
            self._undeployed_ship_ids.add(ship_id)
            ship.vital.alive = False
        commander = self.blue_commander if team == Team.BLUE else self.red_commander
        if ship.squad_id and ship.squad_id not in commander.squad_ids:
            commander.squad_ids.append(ship.squad_id)
        return ship

    def _build_setup_sync_payload(self) -> list[dict]:
        payload: list[dict] = []
        for ship in self.engine.world.ships.values():
            if ship.team != self.controlled_team:
                continue
            payload.append(
                {
                    "ship_id": ship.ship_id,
                    "team": ship.team.value,
                    "squad_id": ship.squad_id,
                    "ship_name": ship.fit.ship_name,
                    "fit_text": self._ship_fit_texts.get(ship.ship_id, ""),
                    "fit_role": ship.fit.role,
                    "base_dps": ship.fit.base_dps,
                    "fit_volley": ship.fit.volley,
                    "fit_optimal_range": ship.fit.optimal_range,
                    "fit_falloff": ship.fit.falloff,
                    "fit_tracking": ship.fit.tracking,
                    "profile_dps": ship.profile.dps,
                    "profile_volley": ship.profile.volley,
                    "profile_optimal": ship.profile.optimal,
                    "profile_falloff": ship.profile.falloff,
                    "profile_tracking": ship.profile.tracking,
                    "profile_sig_radius": ship.profile.sig_radius,
                    "profile_scan_resolution": ship.profile.scan_resolution,
                    "profile_max_target_range": ship.profile.max_target_range,
                    "profile_max_locked_targets": ship.profile.max_locked_targets,
                    "profile_scan_strength": ship.profile.scan_strength,
                    "profile_ecm_jam_chance": ship.profile.ecm_jam_chance,
                    "profile_warp_scramble_status": ship.profile.warp_scramble_status,
                    "profile_warp_stability": ship.profile.warp_stability,
                    "profile_max_speed": ship.profile.max_speed,
                    "profile_max_cap": ship.profile.max_cap,
                    "profile_cap_recharge_time": ship.profile.cap_recharge_time,
                    "profile_shield_hp": ship.profile.shield_hp,
                    "profile_armor_hp": ship.profile.armor_hp,
                    "profile_structure_hp": ship.profile.structure_hp,
                    "profile_rep_amount": ship.profile.rep_amount,
                    "profile_rep_cycle": ship.profile.rep_cycle,
                    "position": {"x": ship.nav.position.x, "y": ship.nav.position.y},
                    "velocity": {"x": ship.nav.velocity.x, "y": ship.nav.velocity.y},
                    "facing_deg": ship.nav.facing_deg,
                    "x": ship.nav.position.x,
                    "y": ship.nav.position.y,
                    "shield": ship.vital.shield,
                    "armor": ship.vital.armor,
                    "structure": ship.vital.structure,
                    "vital_shield_max": ship.vital.shield_max,
                    "vital_armor_max": ship.vital.armor_max,
                    "vital_structure_max": ship.vital.structure_max,
                    "cap": ship.vital.cap,
                    "vital_cap_max": ship.vital.cap_max,
                    "alive": ship.vital.alive,
                    "quality_level": ship.quality.level.value,
                    "quality_reaction_delay": ship.quality.reaction_delay,
                    "quality_ignore_order_probability": ship.quality.ignore_order_probability,
                    "quality_formation_jitter": ship.quality.formation_jitter,
                    "deployed": ship.ship_id not in self._undeployed_ship_ids,
                }
            )
        return payload

    def _apply_remote_snapshot(self, packet: dict) -> None:
        self._lan_debug("recv-snapshot")
        lan = packet.get("lan") if isinstance(packet.get("lan"), dict) else None
        if isinstance(lan, dict):
            cfg_payload = lan.get("engine_config")
            if isinstance(cfg_payload, dict):
                self._apply_host_engine_config(cfg_payload)
        snapshot = packet.get("snapshot") if isinstance(packet.get("snapshot"), dict) else packet
        if isinstance(snapshot, dict):
            try:
                self.engine.world.tick = int(snapshot.get("tick", self.engine.world.tick))
                self.engine.world.now = float(snapshot.get("now", self.engine.world.now))
            except Exception:
                pass
            removed = snapshot.get("removed_ship_ids")
            if isinstance(removed, list):
                for ship_id in removed:
                    sid = str(ship_id)
                    self.engine.world.ships.pop(sid, None)
                    self.engine.ship_agents.pop(sid, None)
                    self._ship_fit_texts.pop(sid, None)
                    self._undeployed_ship_ids.discard(sid)
                    dialog = self._status_dialogs.pop(sid, None)
                    if dialog is not None:
                        dialog.close()
            queues = snapshot.get("squad_focus_queues")
            if isinstance(queues, dict):
                rebuilt: dict[str, list[str]] = {}
                for key, value in queues.items():
                    if isinstance(value, list):
                        rebuilt[str(key)] = [str(v) for v in value if str(v)]
                self.engine.world.squad_focus_queues = rebuilt
        ships = snapshot.get("ships") if isinstance(snapshot, dict) else None
        if not isinstance(ships, dict):
            return
        for ship_id, raw in ships.items():
            if not isinstance(raw, dict):
                continue
            ship = self._ensure_remote_ship(str(ship_id), raw)
            ship.squad_id = str(raw.get("squad_id", ship.squad_id))
            ship.team = Team.BLUE if str(raw.get("team", ship.team.value)) == "BLUE" else Team.RED
            pos = raw.get("position", {})
            vel = raw.get("velocity", {})
            ship.nav.position = Vector2(float(pos.get("x", ship.nav.position.x)), float(pos.get("y", ship.nav.position.y)))
            ship.nav.velocity = Vector2(float(vel.get("x", ship.nav.velocity.x)), float(vel.get("y", ship.nav.velocity.y)))
            ship.nav.facing_deg = float(raw.get("facing_deg", ship.nav.facing_deg))
            ship.vital.shield = float(raw.get("shield", ship.vital.shield))
            ship.vital.armor = float(raw.get("armor", ship.vital.armor))
            ship.vital.structure = float(raw.get("structure", ship.vital.structure))
            ship.vital.shield_max = float(raw.get("shield_max", ship.vital.shield_max))
            ship.vital.armor_max = float(raw.get("armor_max", ship.vital.armor_max))
            ship.vital.structure_max = float(raw.get("structure_max", ship.vital.structure_max))
            ship.vital.alive = bool(raw.get("alive", ship.vital.alive))
            ship.vital.cap = float(raw.get("cap", ship.vital.cap))
            ship.vital.cap_max = float(raw.get("cap_max", ship.vital.cap_max))
            deployed = bool(raw.get("deployed", ship.ship_id not in self._undeployed_ship_ids))
            if deployed:
                self._undeployed_ship_ids.discard(ship.ship_id)
            else:
                self._undeployed_ship_ids.add(ship.ship_id)
                ship.vital.alive = False
            ship.combat.current_target = raw.get("target")
            fit_text = str(raw.get("fit_text", "") or "").strip()
            if fit_text:
                self._ship_fit_texts[str(ship_id)] = fit_text
            projected_targets = raw.get("projected_targets")
            if isinstance(projected_targets, dict):
                ship.combat.projected_targets = {
                    str(module_id): str(target_id)
                    for module_id, target_id in projected_targets.items()
                    if str(module_id)
                }
            else:
                ship.combat.projected_targets.clear()

            module_cycle_timers = raw.get("module_cycle_timers")
            if isinstance(module_cycle_timers, dict):
                parsed_cycle_timers: dict[str, float] = {}
                for module_id, timer_left in module_cycle_timers.items():
                    mid = str(module_id)
                    if not mid:
                        continue
                    try:
                        parsed_cycle_timers[mid] = max(0.0, float(timer_left))
                    except Exception:
                        continue
                ship.combat.module_cycle_timers = parsed_cycle_timers
            else:
                ship.combat.module_cycle_timers.clear()

            ecm_sources = raw.get("ecm_jam_sources")
            if isinstance(ecm_sources, dict):
                parsed_sources: dict[str, float] = {}
                for source_id, jam_until in ecm_sources.items():
                    sid = str(source_id)
                    if not sid:
                        continue
                    try:
                        parsed_sources[sid] = float(jam_until)
                    except Exception:
                        continue
                ship.combat.ecm_jam_sources = parsed_sources
            else:
                ship.combat.ecm_jam_sources.clear()

            ecm_attempt_target = str(raw.get("ecm_last_attempt_target", "") or "").strip()
            ship.combat.ecm_last_attempt_target = ecm_attempt_target or None
            ecm_attempt_module = str(raw.get("ecm_last_attempt_module", "") or "").strip()
            ship.combat.ecm_last_attempt_module = ecm_attempt_module or None
            raw_success = raw.get("ecm_last_attempt_success")
            ship.combat.ecm_last_attempt_success = raw_success if isinstance(raw_success, bool) else None
            try:
                ship.combat.ecm_last_attempt_chance = max(0.0, min(1.0, float(raw.get("ecm_last_attempt_chance", 0.0) or 0.0)))
            except Exception:
                ship.combat.ecm_last_attempt_chance = 0.0
            try:
                raw_last_attempt_at = raw.get("ecm_last_attempt_at", -1e9)
                ship.combat.ecm_last_attempt_at = float(raw_last_attempt_at if raw_last_attempt_at is not None else -1e9)
            except Exception:
                ship.combat.ecm_last_attempt_at = -1e9

            ecm_target_by_module = raw.get("ecm_last_attempt_target_by_module")
            if isinstance(ecm_target_by_module, dict):
                ship.combat.ecm_last_attempt_target_by_module = {
                    str(module_id): str(target_id)
                    for module_id, target_id in ecm_target_by_module.items()
                    if str(module_id)
                }
            else:
                ship.combat.ecm_last_attempt_target_by_module.clear()

            ecm_success_by_module = raw.get("ecm_last_attempt_success_by_module")
            if isinstance(ecm_success_by_module, dict):
                parsed_success: dict[str, bool] = {}
                for module_id, success in ecm_success_by_module.items():
                    mid = str(module_id)
                    if not mid or not isinstance(success, bool):
                        continue
                    parsed_success[mid] = success
                ship.combat.ecm_last_attempt_success_by_module = parsed_success
            else:
                ship.combat.ecm_last_attempt_success_by_module.clear()

            ecm_at_by_module = raw.get("ecm_last_attempt_at_by_module")
            if isinstance(ecm_at_by_module, dict):
                parsed_at: dict[str, float] = {}
                for module_id, ts in ecm_at_by_module.items():
                    mid = str(module_id)
                    if not mid:
                        continue
                    try:
                        parsed_at[mid] = float(ts)
                    except Exception:
                        continue
                ship.combat.ecm_last_attempt_at_by_module = parsed_at
            else:
                ship.combat.ecm_last_attempt_at_by_module.clear()

            module_states = raw.get("module_states")
            if isinstance(module_states, dict) and ship.runtime is not None:
                state_map = {str(mid): str(state) for mid, state in module_states.items()}
                for module in ship.runtime.modules:
                    state_name = state_map.get(module.module_id)
                    if not state_name:
                        continue
                    if state_name in ModuleState.__members__:
                        module.state = module.normalized_state(ModuleState[state_name])

    @staticmethod
    def _ship_signature(raw: dict) -> tuple:
        raw_pos = raw.get("position")
        raw_vel = raw.get("velocity")
        raw_projected = raw.get("projected_targets")
        raw_cycle_timers = raw.get("module_cycle_timers")
        raw_ecm_sources = raw.get("ecm_jam_sources")
        raw_ecm_target_by_module = raw.get("ecm_last_attempt_target_by_module")
        raw_ecm_success_by_module = raw.get("ecm_last_attempt_success_by_module")
        raw_ecm_at_by_module = raw.get("ecm_last_attempt_at_by_module")
        raw_module_states = raw.get("module_states")
        pos: dict = raw_pos if isinstance(raw_pos, dict) else {}
        vel: dict = raw_vel if isinstance(raw_vel, dict) else {}
        projected: dict = raw_projected if isinstance(raw_projected, dict) else {}
        cycle_timers: dict = raw_cycle_timers if isinstance(raw_cycle_timers, dict) else {}
        ecm_sources: dict = raw_ecm_sources if isinstance(raw_ecm_sources, dict) else {}
        ecm_target_by_module: dict = raw_ecm_target_by_module if isinstance(raw_ecm_target_by_module, dict) else {}
        ecm_success_by_module: dict = raw_ecm_success_by_module if isinstance(raw_ecm_success_by_module, dict) else {}
        ecm_at_by_module: dict = raw_ecm_at_by_module if isinstance(raw_ecm_at_by_module, dict) else {}
        module_states: dict = raw_module_states if isinstance(raw_module_states, dict) else {}
        cycle_timers_sig: list[tuple[str, float]] = []
        for module_id, timer_left in cycle_timers.items():
            mid = str(module_id)
            if not mid:
                continue
            try:
                cycle_timers_sig.append((mid, round(float(timer_left), 2)))
            except Exception:
                continue
        ecm_sources_sig: list[tuple[str, float]] = []
        for source_id, jam_until in ecm_sources.items():
            sid = str(source_id)
            if not sid:
                continue
            try:
                ecm_sources_sig.append((sid, round(float(jam_until), 2)))
            except Exception:
                continue
        ecm_at_by_module_sig: list[tuple[str, float]] = []
        for module_id, ts in ecm_at_by_module.items():
            mid = str(module_id)
            if not mid:
                continue
            try:
                ecm_at_by_module_sig.append((mid, round(float(ts), 2)))
            except Exception:
                continue
        return (
            str(raw.get("team", "")),
            str(raw.get("squad_id", "")),
            bool(raw.get("deployed", True)),
            bool(raw.get("alive", False)),
            round(float(pos.get("x", 0.0)), 1),
            round(float(pos.get("y", 0.0)), 1),
            round(float(vel.get("x", 0.0)), 1),
            round(float(vel.get("y", 0.0)), 1),
            round(float(raw.get("facing_deg", 0.0)), 1),
            round(float(raw.get("shield", 0.0)), 1),
            round(float(raw.get("armor", 0.0)), 1),
            round(float(raw.get("structure", 0.0)), 1),
            round(float(raw.get("cap", 0.0)), 1),
            round(float(raw.get("shield_max", 0.0)), 1),
            round(float(raw.get("armor_max", 0.0)), 1),
            round(float(raw.get("structure_max", 0.0)), 1),
            round(float(raw.get("cap_max", 0.0)), 1),
            str(raw.get("target", "") or ""),
            tuple(sorted((str(k), str(v)) for k, v in projected.items())),
            tuple(sorted(cycle_timers_sig)),
            tuple(sorted(ecm_sources_sig)),
            str(raw.get("ecm_last_attempt_target", "") or ""),
            str(raw.get("ecm_last_attempt_module", "") or ""),
            raw.get("ecm_last_attempt_success", None),
            round(float(raw.get("ecm_last_attempt_chance", 0.0) or 0.0), 3),
            round(
                float(
                    raw.get("ecm_last_attempt_at", -1e9)
                    if raw.get("ecm_last_attempt_at", -1e9) is not None
                    else -1e9
                ),
                2,
            ),
            tuple(sorted((str(k), str(v)) for k, v in ecm_target_by_module.items())),
            tuple(sorted((str(k), bool(v)) for k, v in ecm_success_by_module.items() if isinstance(v, bool))),
            tuple(sorted(ecm_at_by_module_sig)),
            tuple(sorted((str(k), str(v)) for k, v in module_states.items())),
        )

    def _engine_config_payload(self) -> dict[str, object]:
        cfg = self.engine.config
        return {
            "tick_rate": int(cfg.tick_rate),
            "physics_substeps": int(cfg.physics_substeps),
            "lockstep": bool(cfg.lockstep),
            "battlefield_radius": float(cfg.battlefield_radius),
            "detailed_logging": bool(cfg.detailed_logging),
            "hotspot_logging": bool(cfg.hotspot_logging),
            "detail_log_file": str(cfg.detail_log_file),
            "hotspot_log_file": str(cfg.hotspot_log_file),
            "log_merge_window_sec": float(cfg.log_merge_window_sec),
        }

    def _apply_host_engine_config(self, payload: dict) -> None:
        if self.network_mode != "client":
            return
        try:
            tick_rate = max(1, int(float(payload.get("tick_rate", self.engine.config.tick_rate))))
        except Exception:
            tick_rate = max(1, int(self.engine.config.tick_rate))
        try:
            substeps = max(1, int(float(payload.get("physics_substeps", self.engine.config.physics_substeps))))
        except Exception:
            substeps = max(1, int(self.engine.config.physics_substeps))
        try:
            radius = max(1_000.0, float(payload.get("battlefield_radius", self.engine.config.battlefield_radius)))
        except Exception:
            radius = max(1_000.0, float(self.engine.config.battlefield_radius))
        try:
            merge_window = max(0.1, float(payload.get("log_merge_window_sec", self.engine.config.log_merge_window_sec)))
        except Exception:
            merge_window = max(0.1, float(self.engine.config.log_merge_window_sec))

        old_tick_rate = int(self.engine.config.tick_rate)
        self.engine.config.tick_rate = tick_rate
        self.engine.config.physics_substeps = substeps
        self.engine.config.lockstep = bool(payload.get("lockstep", self.engine.config.lockstep))
        self.engine.config.battlefield_radius = radius
        self.engine.config.detailed_logging = bool(payload.get("detailed_logging", self.engine.config.detailed_logging))
        self.engine.config.hotspot_logging = bool(payload.get("hotspot_logging", self.engine.config.hotspot_logging))
        self.engine.config.detail_log_file = str(payload.get("detail_log_file", self.engine.config.detail_log_file))
        self.engine.config.hotspot_log_file = str(payload.get("hotspot_log_file", self.engine.config.hotspot_log_file))
        self.engine.config.log_merge_window_sec = merge_window

        new_logger = get_sim_logger(self.engine.config)
        self.engine._logger = new_logger
        self.engine.combat.attach_logger(
            new_logger,
            self.engine.config.detailed_logging,
            self.engine.config.log_merge_window_sec,
            self.engine.config.hotspot_logging,
        )

        if hasattr(self.engine, "_dt"):
            self.engine._dt = 1.0 / float(tick_rate)
        if hasattr(self.engine, "movement") and hasattr(self.engine.movement, "battlefield_radius"):
            self.engine.movement.battlefield_radius = radius

        if old_tick_rate != tick_rate:
            self.tick_timer.setInterval(max(1, int(1000 / tick_rate)))

    def _send_host_state(self, countdown_left: float | None = None, started: bool = True) -> None:
        if self.lan_server is None:
            return
        if not self.lan_server.client_connected:
            self._last_network_send_at = 0.0
            self._last_full_sync_at = 0.0
            self._last_sent_ship_signatures.clear()
            self._last_sent_fit_texts.clear()
            return
        now = time.perf_counter()
        countdown_active = (countdown_left is not None) and (float(countdown_left) > 0.0)
        if not countdown_active and (now - self._last_network_send_at) < self._network_send_interval_sec:
            return

        full_sync = (
            countdown_active
            or (now - self._last_full_sync_at) >= self._network_full_sync_interval_sec
            or not self._last_sent_ship_signatures
        )

        base = self.engine.snapshot()
        raw_ships = base.get("ships")
        ships_raw: dict = raw_ships if isinstance(raw_ships, dict) else {}
        next_signatures: dict[str, tuple] = {}
        ships_out: dict[str, dict] = {}
        for ship_id, raw in ships_raw.items():
            if not isinstance(raw, dict):
                continue
            sid = str(ship_id)
            row = dict(raw)
            row["deployed"] = sid not in self._undeployed_ship_ids
            signature = self._ship_signature(row)
            next_signatures[sid] = signature
            fit_text = self._ship_fit_texts.get(sid, "")
            fit_changed = self._last_sent_fit_texts.get(sid, "") != fit_text
            changed = full_sync or (self._last_sent_ship_signatures.get(sid) != signature) or fit_changed
            if not changed:
                continue
            if full_sync or fit_changed:
                row["fit_text"] = fit_text
            ships_out[sid] = row

        removed_ship_ids = sorted(set(self._last_sent_ship_signatures.keys()) - set(next_signatures.keys()))
        packet = {
            "snapshot": {
                "tick": base.get("tick", self.engine.world.tick),
                "now": base.get("now", self.engine.world.now),
                "ships": ships_out,
                "removed_ship_ids": removed_ship_ids,
                "squad_focus_queues": base.get("squad_focus_queues", {}),
                "partial": not full_sync,
            },
            "lan": {
                "started": bool(started),
                "countdown_left": float(max(0.0, countdown_left or 0.0)),
                "engine_config": self._engine_config_payload(),
            },
        }
        self._lan_debug(
            f"send-snapshot ships={len(ships_out)} full={full_sync} removed={len(removed_ship_ids)} countdown={countdown_left} started={started}"
        )
        self.lan_server.send_state(packet)
        self._last_network_send_at = now
        if full_sync:
            self._last_full_sync_at = now
        self._last_sent_ship_signatures = next_signatures
        self._last_sent_fit_texts = {sid: self._ship_fit_texts.get(sid, "") for sid in next_signatures.keys()}

    def on_tick(self) -> None:
        self._flush_tick_ops()
        if self.network_mode == "client":
            if self.lan_client is None or not self.lan_client.connected:
                self._setup_synced = False
            if not self._setup_synced and self.lan_client is not None and self.lan_client.connected:
                self.lan_client.send_command({"kind": CMD_SYNC_SETUP, "ships": self._build_setup_sync_payload()})
                self._setup_synced = True
            if self.lan_client is not None:
                packet = self.lan_client.consume_latest_state()
                if packet is not None:
                    self._apply_remote_snapshot(packet)
            self._update_approach_targets()
            self._ui_tick_counter += 1
            if (self._ui_tick_counter % self._ui_refresh_interval_ticks) == 0:
                self._sync_blue_squads()
            if (self._ui_tick_counter % self._overview_refresh_interval_ticks) == 0:
                self.request_overview_refresh(force=True)
            if self.engine.world.tick % 10 == 0:
                self.refresh_blue_roster()
            return

        if self.network_mode == "host" and self.lan_server is not None:
            for cmd in self.lan_server.poll_commands():
                self._apply_remote_command(cmd)
            has_remote_red = any(s.team == Team.RED for s in self.engine.world.ships.values())
            if not self.lan_server.client_connected:
                self._countdown_started_at = None
                self._match_started = False
                self.status.setText(f"{tr(self.current_language(), 'status_prefix')}: waiting for red client...")
                self._send_host_state(countdown_left=10.0, started=False)
                return
            if not has_remote_red:
                self._countdown_started_at = None
                self._match_started = False
                self.status.setText(f"{tr(self.current_language(), 'status_prefix')}: waiting for red fleet sync...")
                self._send_host_state(countdown_left=10.0, started=False)
                return
            if not self._match_started:
                now = time.perf_counter()
                if self._countdown_started_at is None:
                    self._countdown_started_at = now
                left = 10.0 - (now - self._countdown_started_at)
                if left > 0:
                    self.status.setText(f"{tr(self.current_language(), 'status_prefix')}: match starts in {left:.1f}s")
                    self._send_host_state(countdown_left=left, started=False)
                    return
                self._match_started = True

            self._update_approach_targets()

        t0 = time.perf_counter()
        self.engine.step()
        step_ms = (time.perf_counter() - t0) * 1000.0
        if self._step_ms_ema <= 0:
            self._step_ms_ema = step_ms
        else:
            self._step_ms_ema = self._step_ms_ema * 0.85 + step_ms * 0.15

        if self.network_mode == "host":
            self._send_host_state(countdown_left=0.0, started=True)

        self._ui_tick_counter += 1
        refresh_ui = (self._ui_tick_counter % self._ui_refresh_interval_ticks) == 0
        refresh_overview = (self._ui_tick_counter % self._overview_refresh_interval_ticks) == 0

        if refresh_ui:
            self._refresh_propulsion_button_text()
            lang = self.current_language()
            alive_blue = sum(1 for s in self.engine.world.ships.values() if s.team == Team.BLUE and s.vital.alive)
            alive_red = sum(1 for s in self.engine.world.ships.values() if s.team == Team.RED and s.vital.alive)
            tick = self.engine.world.tick
            total_ships = len(self.engine.world.ships)
            self.status.setText(
                f"{tr(lang, 'status_prefix')}: {tick} | {tr(lang, 'status_ships')}: {total_ships} | "
                f"{tr(lang, 'status_blue')}: {alive_blue} | {tr(lang, 'status_red')}: {alive_red} | "
                f"{tr(lang, 'status_zoom')}: {self.canvas.zoom:.5f} | {tr(lang, 'status_step_ms')}: {self._step_ms_ema:.2f}"
            )

        if refresh_overview:
            self.request_overview_refresh()

        if self.engine.world.tick % 10 == 0:
            self.refresh_blue_roster()

        if self.ui_state.selected_enemy_target:
            target = self.engine.world.ships.get(self.ui_state.selected_enemy_target)
            if target is None or not target.vital.alive:
                self.ui_state.selected_enemy_target = None
                self.canvas.selected_enemy_target = None
                self.overview_model.notify_visual_state_changed()
                self.request_overview_refresh(force=True)

    def closeEvent(self, event) -> None:
        self.prefs.selected_squad = self.ui_state.selected_squad
        self.prefs.zoom = self.canvas.zoom
        self.store.save(self.prefs)
        if hasattr(self.engine, "combat") and hasattr(self.engine.combat, "flush_pending_events"):
            self.engine.combat.flush_pending_events()
        if self.lan_server is not None:
            self.lan_server.stop()
        if self.lan_client is not None:
            self.lan_client.close()
        super().closeEvent(event)



