from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Literal, cast

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QSortFilterProxyModel, QTimer, Qt, QLocale, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QAbstractItemView,
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
    QTableWidget,
    QTableWidgetItem,
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


class _PopupAwareComboBox(QComboBox):
    popup_visibility_changed = Signal(bool)

    def showPopup(self) -> None:
        self.popup_visibility_changed.emit(True)
        super().showPopup()

    def hidePopup(self) -> None:
        super().hidePopup()
        self.popup_visibility_changed.emit(False)


class FleetLibraryDialog(QDialog):
    def __init__(self, templates: dict[str, list[dict]], lang: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lang = lang
        self._templates: dict[str, list[dict]] = json.loads(json.dumps(templates, ensure_ascii=False))
        self._current_name: str = ""
        self._parser = EftFitParser()
        self._factory = RuntimeFromEftFactory()

        self.setWindowTitle(tr(lang, "fleet_lib_title"))
        self.resize(980, 620)

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.lbl_fleet = QLabel(tr(lang, "fleet_lib_name"))
        self.fleet_combo = QComboBox(self)
        self.btn_new = QPushButton(tr(lang, "fleet_lib_new"))
        self.btn_delete = QPushButton(tr(lang, "fleet_lib_delete"))
        self.btn_save = QPushButton(tr(lang, "fleet_lib_save"))
        top.addWidget(self.lbl_fleet)
        top.addWidget(self.fleet_combo, 1)
        top.addWidget(self.btn_new)
        top.addWidget(self.btn_delete)
        top.addWidget(self.btn_save)
        layout.addLayout(top)

        body = QHBoxLayout()
        self.table_model = FleetSetupTableModel([], lambda: self._lang)
        self.table = QTableView(self)
        self.table.setModel(self.table_model)
        self.table.setItemDelegate(SetupRowDelegate(lambda: self._lang, self.table))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setColumnHidden(0, True)
        body.addWidget(self.table, 3)

        right = QVBoxLayout()
        self.btn_add_row = QPushButton(tr(lang, "fleet_lib_add_row"))
        self.btn_remove_row = QPushButton(tr(lang, "fleet_lib_remove_row"))
        right.addWidget(self.btn_add_row)
        right.addWidget(self.btn_remove_row)
        right.addStretch(1)
        body.addLayout(right, 1)
        layout.addLayout(body, 2)

        self.fit_editor = QPlainTextEdit(self)
        self.fit_editor.setPlaceholderText(tr(lang, "setup_fit_placeholder"))
        layout.addWidget(self.fit_editor, 2)

        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        actions.button(QDialogButtonBox.StandardButton.Ok).setText(tr(lang, "fleet_lib_confirm"))
        actions.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(lang, "setup_cancel"))
        layout.addWidget(actions)

        self.fleet_combo.currentTextChanged.connect(self._on_fleet_changed)
        self.btn_new.clicked.connect(self._new_fleet)
        self.btn_delete.clicked.connect(self._delete_fleet)
        self.btn_save.clicked.connect(self._save_current)
        self.btn_add_row.clicked.connect(self._add_row)
        self.btn_remove_row.clicked.connect(self._remove_row)
        self.table.clicked.connect(self._on_table_clicked)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self.fit_editor.textChanged.connect(self._on_fit_changed)
        actions.accepted.connect(self._on_accept)
        actions.rejected.connect(self.reject)

        self._refresh_combo()
        if self.fleet_combo.count() > 0:
            self._on_fleet_changed(self.fleet_combo.currentText())

    @property
    def templates(self) -> dict[str, list[dict]]:
        return self._templates

    def _refresh_combo(self) -> None:
        names = sorted(self._templates.keys())
        self.fleet_combo.blockSignals(True)
        self.fleet_combo.clear()
        self.fleet_combo.addItems(names)
        self.fleet_combo.blockSignals(False)
        if names:
            self.fleet_combo.setCurrentText(names[0])

    def _rows_from_template(self, name: str) -> list[SetupRow]:
        out: list[SetupRow] = []
        for item in self._templates.get(name, []):
            quality_text = str(item.get("quality", "REGULAR")).strip().upper()
            try:
                quality = QualityLevel(quality_text)
            except Exception:
                quality = QualityLevel.REGULAR
            row = SetupRow(
                team=Team.BLUE,
                squad_id=str(item.get("squad_id", "ALPHA")),
                quality=quality,
                quantity=max(1, int(float(item.get("quantity", 1)))),
                fit_text=str(item.get("fit_text", "")),
                is_leader=bool(item.get("is_leader", False)) and max(1, int(float(item.get("quantity", 1)))) == 1,
            )
            try:
                parsed = self._parser.parse(row.fit_text)
                row.fit_name = f"{parsed.ship_name} / {parsed.fit_name}"
            except Exception:
                row.fit_name = ""
            out.append(row)
        return out

    def _template_from_rows(self) -> list[dict]:
        rows = [
            {
                "squad_id": row.squad_id,
                "quality": row.quality.value,
                "quantity": int(max(1, row.quantity)),
                "fit_text": row.fit_text,
                "is_leader": bool(row.is_leader) and int(max(1, row.quantity)) == 1,
            }
            for row in self.table_model.rows
        ]
        return self._normalize_leaders(rows)

    @staticmethod
    def _normalize_leaders(rows: list[dict]) -> list[dict]:
        squad_to_indices: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            squad = str(row.get("squad_id", "")).strip().upper()
            if not squad:
                squad = "ALPHA"
                row["squad_id"] = "ALPHA"
            squad_to_indices.setdefault(squad, []).append(idx)

        for indices in squad_to_indices.values():
            leader_indices = [
                i
                for i in indices
                if bool(rows[i].get("is_leader", False)) and int(rows[i].get("quantity", 1) or 1) == 1
            ]
            single_indices = [i for i in indices if int(rows[i].get("quantity", 1) or 1) == 1]
            chosen = leader_indices[0] if leader_indices else (single_indices[0] if single_indices else -1)
            for i in indices:
                rows[i]["is_leader"] = (i == chosen) and int(rows[i].get("quantity", 1) or 1) == 1
        return rows

    def _on_fleet_changed(self, name: str) -> None:
        if self._current_name:
            self._templates[self._current_name] = self._template_from_rows()
        self._current_name = name.strip()
        self.table_model.replace_rows(self._rows_from_template(self._current_name))
        if self.table_model.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self.fit_editor.blockSignals(True)
            self.fit_editor.setPlainText("")
            self.fit_editor.blockSignals(False)

    def _new_fleet(self) -> None:
        name, ok = QInputDialog.getText(self, tr(self._lang, "fleet_lib_new"), tr(self._lang, "setup_fleet_name_label"))
        if not ok:
            return
        fleet_name = name.strip()
        if not fleet_name:
            return
        if fleet_name in self._templates:
            QMessageBox.warning(
                self,
                tr(self._lang, "fleet_lib_exists_title"),
                tr(self._lang, "fleet_lib_exists_msg", name=fleet_name),
            )
            return
        self._templates[fleet_name] = []
        self._refresh_combo()
        self.fleet_combo.setCurrentText(fleet_name)
        self._on_fleet_changed(fleet_name)

    def _delete_fleet(self) -> None:
        name = self.fleet_combo.currentText().strip()
        if not name:
            return
        self._templates.pop(name, None)
        self._current_name = ""
        self._refresh_combo()

    def _save_current(self) -> None:
        name = self.fleet_combo.currentText().strip()
        if not name:
            return
        self._templates[name] = self._template_from_rows()

    def _add_row(self) -> None:
        previous: SetupRow | None = None
        current = self.table.currentIndex()
        if current.isValid() and 0 <= current.row() < len(self.table_model.rows):
            previous = self.table_model.rows[current.row()]
        elif self.table_model.rows:
            previous = self.table_model.rows[-1]
        self.table_model.add_row(
            SetupRow(
                team=Team.BLUE,
                squad_id=previous.squad_id if previous else "ALPHA",
                quality=QualityLevel.REGULAR,
                quantity=1,
                fit_text=previous.fit_text if previous else "",
                fit_name=previous.fit_name if previous else "",
            )
        )
        self.table.selectRow(self.table_model.rowCount() - 1)

    def _on_table_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        if index.column() in (2, 5):
            self.table.edit(index)

    def _remove_row(self) -> None:
        idx = self.table.currentIndex()
        if not idx.isValid():
            return
        self.table_model.remove_row(idx.row())
        if self.table_model.rowCount() > 0:
            self.table.selectRow(max(0, idx.row() - 1))

    def _on_row_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        del previous
        if not current.isValid() or not (0 <= current.row() < len(self.table_model.rows)):
            self.fit_editor.blockSignals(True)
            self.fit_editor.setPlainText("")
            self.fit_editor.blockSignals(False)
            return
        self.fit_editor.blockSignals(True)
        self.fit_editor.setPlainText(self.table_model.rows[current.row()].fit_text)
        self.fit_editor.blockSignals(False)

    def _on_fit_changed(self) -> None:
        idx = self.table.currentIndex()
        if not idx.isValid() or not (0 <= idx.row() < len(self.table_model.rows)):
            return
        text = self.fit_editor.toPlainText().strip()
        self.table_model.rows[idx.row()].fit_text = text
        fit_name = ""
        try:
            parsed = self._parser.parse(text)
            self._factory.build(parsed)
            fit_name = f"{parsed.ship_name} / {parsed.fit_name}"
        except Exception:
            pass
        self.table_model.update_fit_meta(idx.row(), fit_name)

    def _on_accept(self) -> None:
        self._save_current()
        self.accept()



class OverviewOptionsDialog(QDialog):
    _VALID_FILTER_TEAMS: tuple[Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"], ...] = (
        "ALL",
        "FRIENDLY",
        "ENEMY",
        "BLUE",
        "RED",
    )

    def __init__(self, prefs: UiPreferences, lang: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lang = lang
        self.setWindowTitle(tr(lang, "overview_filter_title"))
        self.setModal(True)
        self.resize(420, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.filter_team = QComboBox()
        self.filter_team.addItem(tr(lang, "filter_all"), "ALL")
        self.filter_team.addItem(tr(lang, "filter_friendly"), "FRIENDLY")
        self.filter_team.addItem(tr(lang, "filter_enemy_team"), "ENEMY")
        team_idx = self.filter_team.findData(prefs.filter_team)
        self.filter_team.setCurrentIndex(0 if team_idx < 0 else team_idx)

        self.filter_role = QComboBox()
        self.filter_role.addItem(tr(lang, "filter_all"), "ALL")
        self.filter_role.addItem("DPS", "DPS")
        self.filter_role.addItem("LOGI", "LOGI")
        role_idx = self.filter_role.findData(prefs.filter_role)
        self.filter_role.setCurrentIndex(0 if role_idx < 0 else role_idx)

        self.filter_alive = QComboBox()
        self.filter_alive.addItem(tr(lang, "filter_all"), "ALL")
        self.filter_alive.addItem(tr(lang, "filter_alive"), "ALIVE")
        self.filter_alive.addItem(tr(lang, "filter_destroyed"), "DESTROYED")
        alive_idx = self.filter_alive.findData(prefs.filter_alive)
        self.filter_alive.setCurrentIndex(0 if alive_idx < 0 else alive_idx)

        self.filter_squad = QLineEdit(prefs.filter_squad)

        form.addRow(tr(lang, "filter_team"), self.filter_team)
        form.addRow(tr(lang, "filter_role"), self.filter_role)
        form.addRow(tr(lang, "filter_status"), self.filter_alive)
        form.addRow(tr(lang, "filter_squad_contains"), self.filter_squad)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _selected_filter_team(self) -> Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"]:
        value = str(self.filter_team.currentData() or "ALL").upper()
        if value in self._VALID_FILTER_TEAMS:
            return cast(Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"], value)
        return "ALL"

    def to_preferences(self, prefs: UiPreferences) -> UiPreferences:
        return UiPreferences(
            config_version=prefs.config_version,
            selected_squad=prefs.selected_squad,
            filter_team=self._selected_filter_team(),
            filter_role=str(self.filter_role.currentData() or "ALL"),
            filter_alive=str(self.filter_alive.currentData() or "ALL"),
            filter_squad=self.filter_squad.text().strip(),
            sort_key=prefs.sort_key,
            sort_order=prefs.sort_order,
            zoom=prefs.zoom,
            language=prefs.language,
            engine_tick_rate=prefs.engine_tick_rate,
            engine_physics_substeps=prefs.engine_physics_substeps,
            engine_lockstep=prefs.engine_lockstep,
            engine_battlefield_radius=prefs.engine_battlefield_radius,
            engine_detailed_logging=prefs.engine_detailed_logging,
            engine_hotspot_logging=prefs.engine_hotspot_logging,
            engine_detail_log_file=prefs.engine_detail_log_file,
            engine_hotspot_log_file=prefs.engine_hotspot_log_file,
            engine_log_merge_window_sec=prefs.engine_log_merge_window_sec,
        )



class ShipStatusDialog(QDialog):
    def __init__(
        self,
        engine: SimulationEngine,
        ship_id: str,
        language_getter: Callable[[], str],
        fit_text_getter: Callable[[str], str | None],
        lock_charge_getter: Callable[[str, str], str | None],
        lock_charge_setter: Callable[[str, str, str], tuple[bool, str]],
        lock_charge_clearer: Callable[[str, str], tuple[bool, str]],
        module_mode_getter: Callable[[str, str], str],
        module_mode_setter: Callable[[str, str, str], tuple[bool, str]],
        module_mode_sync_setter: Callable[[str, str, str], tuple[bool, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.ship_id = ship_id
        self._language_getter = language_getter
        self._fit_text_getter = fit_text_getter
        self._lock_charge_getter = lock_charge_getter
        self._lock_charge_setter = lock_charge_setter
        self._lock_charge_clearer = lock_charge_clearer
        self._module_mode_getter = module_mode_getter
        self._module_mode_setter = module_mode_setter
        self._module_mode_sync_setter = module_mode_sync_setter
        self._parser = EftFitParser()
        self._runtime_engine = RuntimeStatEngine()
        self._cached_fit_text: str = ""
        self._cached_module_specs: list = []
        self._lock_module_specs: dict[str, ParsedModuleSpec] = {}
        self._lock_ammo_draft_by_module: dict[str, str] = {}
        self._module_mode_row_contexts: list[tuple[str, bool, str]] = []
        self._stable_profile_cache = None
        self._stable_profile_cache_key: tuple[Any, ...] | None = None
        self._lock_controls_signature: tuple[Any, ...] | None = None
        self._tab_signatures: dict[str, tuple[Any, ...] | str] = {}
        self._tab_keys = ["overview", "combat", "defense", "cap_target", "modules", "debug"]
        self._module_mode_popup_open: bool = False
        self.setWindowTitle(f"{tr(self._language_getter(), 'ship_status_title')} - {ship_id}")
        self.resize(920, 720)

        layout = QVBoxLayout(self)
        lock_row1 = QHBoxLayout()
        self.lbl_lock_module = QLabel(self)
        lock_row1.addWidget(self.lbl_lock_module)
        self.lock_module_combo = QComboBox(self)
        self.lock_module_combo.setMinimumWidth(260)
        lock_row1.addWidget(self.lock_module_combo, 1)
        layout.addLayout(lock_row1)

        lock_row2 = QHBoxLayout()
        self.lbl_lock_ammo = QLabel(self)
        lock_row2.addWidget(self.lbl_lock_ammo)
        self.lock_ammo_combo = QComboBox(self)
        self.lock_ammo_combo.setMinimumWidth(220)
        lock_row2.addWidget(self.lock_ammo_combo, 1)
        self.btn_lock_apply = QPushButton(self)
        lock_row2.addWidget(self.btn_lock_apply)
        self.btn_lock_clear = QPushButton(self)
        lock_row2.addWidget(self.btn_lock_clear)
        layout.addLayout(lock_row2)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        self.overview_table = self._create_read_only_table(2)
        overview_page = QWidget(self)
        overview_layout = QVBoxLayout(overview_page)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.addWidget(self.overview_table)
        self.tabs.addTab(overview_page, "")

        self.combat_table = self._create_read_only_table(2)
        combat_page = QWidget(self)
        combat_layout = QVBoxLayout(combat_page)
        combat_layout.setContentsMargins(0, 0, 0, 0)
        combat_layout.addWidget(self.combat_table)
        self.tabs.addTab(combat_page, "")

        self.defense_summary_table = self._create_read_only_table(2)
        self.defense_resistance_table = self._create_read_only_table(6, stretch_last=False)
        defense_page = QWidget(self)
        defense_layout = QVBoxLayout(defense_page)
        defense_layout.setContentsMargins(0, 0, 0, 0)
        defense_layout.addWidget(self.defense_summary_table)
        defense_layout.addWidget(self.defense_resistance_table, 1)
        self.tabs.addTab(defense_page, "")

        self.capacitor_table = self._create_read_only_table(2)
        self.targeting_table = self._create_read_only_table(2)
        cap_target_page = QWidget(self)
        cap_target_layout = QVBoxLayout(cap_target_page)
        cap_target_layout.setContentsMargins(0, 0, 0, 0)
        cap_target_layout.addWidget(self.capacitor_table)
        cap_target_layout.addWidget(self.targeting_table, 1)
        self.tabs.addTab(cap_target_page, "")

        self.modules_table = self._create_read_only_table(9, stretch_last=False)
        modules_page = QWidget(self)
        modules_layout = QVBoxLayout(modules_page)
        modules_layout.setContentsMargins(0, 0, 0, 0)
        modules_layout.addWidget(self.modules_table)
        self.tabs.addTab(modules_page, "")

        self.info = QPlainTextEdit(self)
        self.info.setReadOnly(True)
        debug_page = QWidget(self)
        debug_layout = QVBoxLayout(debug_page)
        debug_layout.setContentsMargins(0, 0, 0, 0)
        debug_layout.addWidget(self.info)
        self.tabs.addTab(debug_page, "")

        self.lock_module_combo.currentIndexChanged.connect(self._on_lock_module_changed)
        self.lock_ammo_combo.currentTextChanged.connect(self._on_lock_ammo_changed)
        self.btn_lock_apply.clicked.connect(self._on_lock_apply_clicked)
        self.btn_lock_clear.clicked.connect(self._on_lock_clear_clicked)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(500)
        self._retitle_tabs()
        self.refresh_status(force=True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start(500)
        self.refresh_status(force=True)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)

    def _on_tab_changed(self, _index: int) -> None:
        self._retitle_tabs()
        self.refresh_status(force=True)

    def _retitle_tabs(self) -> None:
        lang = self._language_getter()
        titles = [
            tr(lang, "status_tab_overview"),
            tr(lang, "status_tab_combat"),
            tr(lang, "status_tab_defense"),
            tr(lang, "status_tab_cap_target"),
            tr(lang, "status_tab_modules"),
            tr(lang, "status_tab_debug"),
        ]
        for index, title in enumerate(titles):
            if index < self.tabs.count():
                self.tabs.setTabText(index, title)

    def _current_tab_key(self) -> str:
        index = max(0, min(self.tabs.currentIndex(), len(self._tab_keys) - 1))
        return self._tab_keys[index]

    @staticmethod
    def _create_read_only_table(column_count: int, *, stretch_last: bool = True) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(column_count)
        table.setRowCount(0)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(stretch_last)
        return table

    @staticmethod
    def _table_signature(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> tuple[Any, ...]:
        return (headers, tuple(rows))

    @staticmethod
    def _apply_table(table: QTableWidget, headers: list[str], rows: list[tuple[str, ...]]) -> None:
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_idx, row_values in enumerate(rows):
            for col_idx, value in enumerate(row_values):
                item = table.item(row_idx, col_idx)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsSelectable)
                    table.setItem(row_idx, col_idx, item)
                item.setText(str(value))
        header = table.horizontalHeader()
        if len(headers) == 2:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        elif len(headers) == 6:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for index in range(1, len(headers)):
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Stretch)
        elif len(headers) == 7:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        elif len(headers) == 8:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        elif len(headers) == 9:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)

    def _refresh_lock_controls_if_needed(self) -> None:
        lang = self._language_getter()
        fit_text = self._fit_text_getter(self.ship_id) or ""
        signature = (lang, fit_text)
        if signature != self._lock_controls_signature:
            self._lock_controls_signature = signature
            self._refresh_lock_controls()

    @staticmethod
    def _module_index_from_id(module_id: str) -> int | None:
        parts = str(module_id).rsplit("-", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return None
        idx = int(parts[1])
        return idx if idx > 0 else None

    def _refresh_lock_controls(self) -> None:
        lang = self._language_getter()
        self.lbl_lock_module.setText(tr(lang, "status_lock_module"))
        self.lbl_lock_ammo.setText(tr(lang, "status_lock_ammo"))
        self.btn_lock_apply.setText(tr(lang, "status_lock_apply"))
        self.btn_lock_clear.setText(tr(lang, "status_lock_clear"))

        fit_text = self._fit_text_getter(self.ship_id) or ""
        module_specs = self._get_module_specs_cached(fit_text)
        previous_module_id = str(self.lock_module_combo.currentData() or "")
        previous_ammo_text = self.lock_ammo_combo.currentText().strip()
        if previous_module_id and previous_ammo_text:
            self._lock_ammo_draft_by_module[previous_module_id] = previous_ammo_text

        self._lock_module_specs = {}
        module_rows: list[tuple[str, str]] = []
        for idx, spec in enumerate(module_specs, start=1):
            ammo_options = get_charge_options_for_module(spec.module_name, language=lang)
            if not ammo_options:
                continue
            module_id = f"mod-{idx}"
            module_label = get_type_display_name(spec.module_name, language=lang)
            if spec.charge_name:
                charge_label = get_type_display_name(spec.charge_name, language=lang)
                module_label = f"[{idx:02d}] {module_label} ({charge_label})"
            else:
                module_label = f"[{idx:02d}] {module_label}"
            module_rows.append((module_id, module_label))
            self._lock_module_specs[module_id] = spec

        active_module_ids = {module_id for module_id, _ in module_rows}
        self._lock_ammo_draft_by_module = {
            module_id: ammo_text
            for module_id, ammo_text in self._lock_ammo_draft_by_module.items()
            if module_id in active_module_ids and ammo_text
        }

        self.lock_module_combo.blockSignals(True)
        self.lock_module_combo.clear()
        if not module_rows:
            self.lock_module_combo.addItem(tr(lang, "status_lock_none"), "")
            self.lock_module_combo.setEnabled(False)
        else:
            self.lock_module_combo.setEnabled(True)
            for module_id, module_label in module_rows:
                self.lock_module_combo.addItem(module_label, module_id)
            select_idx = self.lock_module_combo.findData(previous_module_id)
            if select_idx < 0:
                select_idx = 0
            self.lock_module_combo.setCurrentIndex(select_idx)
        self.lock_module_combo.blockSignals(False)
        self._on_lock_module_changed()

    def _on_lock_module_changed(self, _index: int = -1) -> None:
        lang = self._language_getter()
        module_id = str(self.lock_module_combo.currentData() or "")
        spec = self._lock_module_specs.get(module_id)

        self.lock_ammo_combo.blockSignals(True)
        self.lock_ammo_combo.clear()
        self.lock_ammo_combo.blockSignals(False)

        if spec is None:
            self.lock_ammo_combo.setEnabled(False)
            self.btn_lock_apply.setEnabled(False)
            self.btn_lock_clear.setEnabled(False)
            return

        ammo_options = get_charge_options_for_module(spec.module_name, language=lang)
        self.lock_ammo_combo.blockSignals(True)
        self.lock_ammo_combo.addItems(ammo_options)

        locked_ammo = self._lock_charge_getter(self.ship_id, module_id)
        selected_ammo = ""
        if locked_ammo:
            selected_ammo = get_type_display_name(locked_ammo, language=lang)
        elif module_id and self._lock_ammo_draft_by_module.get(module_id):
            selected_ammo = self._lock_ammo_draft_by_module[module_id]
        elif spec.charge_name:
            selected_ammo = get_type_display_name(spec.charge_name, language=lang)

        if selected_ammo:
            if self.lock_ammo_combo.findText(selected_ammo) < 0:
                self.lock_ammo_combo.addItem(selected_ammo)
            self.lock_ammo_combo.setCurrentText(selected_ammo)
        elif ammo_options:
            self.lock_ammo_combo.setCurrentIndex(0)

        self.lock_ammo_combo.blockSignals(False)
        has_ammo = self.lock_ammo_combo.count() > 0
        self.lock_ammo_combo.setEnabled(has_ammo)
        self.btn_lock_apply.setEnabled(has_ammo)
        self.btn_lock_clear.setEnabled(bool(locked_ammo))
        self._on_lock_ammo_changed(self.lock_ammo_combo.currentText())

    def _on_lock_ammo_changed(self, text: str) -> None:
        module_id = str(self.lock_module_combo.currentData() or "")
        ammo_text = str(text or "").strip()
        if not module_id:
            return
        if ammo_text:
            self._lock_ammo_draft_by_module[module_id] = ammo_text
        else:
            self._lock_ammo_draft_by_module.pop(module_id, None)

    def _on_lock_apply_clicked(self) -> None:
        lang = self._language_getter()
        module_id = str(self.lock_module_combo.currentData() or "")
        ammo_name = self.lock_ammo_combo.currentText().strip()
        if not module_id or not ammo_name:
            return
        ok, message = self._lock_charge_setter(self.ship_id, module_id, ammo_name)
        if ok:
            QMessageBox.information(self, tr(lang, "ammo_title"), message)
            self._lock_controls_signature = None
            self._tab_signatures.pop("modules", None)
            self.refresh_status(force=True)
            return
        QMessageBox.warning(self, tr(lang, "ammo_title"), tr(lang, "status_lock_failed", error=message))

    def _on_lock_clear_clicked(self) -> None:
        lang = self._language_getter()
        module_id = str(self.lock_module_combo.currentData() or "")
        if not module_id:
            return
        ok, message = self._lock_charge_clearer(self.ship_id, module_id)
        if ok:
            QMessageBox.information(self, tr(lang, "ammo_title"), message)
            self._lock_controls_signature = None
            self._tab_signatures.pop("modules", None)
            self.refresh_status(force=True)
            return
        QMessageBox.warning(self, tr(lang, "ammo_title"), tr(lang, "status_lock_failed", error=message))

    @staticmethod
    def _res_pct(resonance: float) -> float:
        return max(0.0, min(99.9, (1.0 - float(resonance)) * 100.0))

    @staticmethod
    def _is_weapon_group(group_name: str) -> bool:
        g = (group_name or "").lower()
        return ("weapon" in g) or ("turret" in g) or ("launcher" in g)

    @staticmethod
    def _is_ecm_group(group_name: str) -> bool:
        return "ecm" in (group_name or "").lower()

    @staticmethod
    def _is_area_effect_group(group_name: str) -> bool:
        group = (group_name or "").strip().lower()
        return group in {"command burst", "smart bomb", "structure area denial module", "burst jammer"}

    @staticmethod
    def _module_has_projected_effects(module: ModuleRuntime) -> bool:
        return any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects)

    @staticmethod
    def _display_module_state(module: ModuleRuntime, raw_state: str, current_target_id: str | None) -> str:
        try:
            normalized = module.normalized_state(ModuleState[str(raw_state or "ONLINE").upper()])
        except Exception:
            normalized = module.normalized_state(ModuleState.ONLINE)
        effective_state = normalized.value
        if effective_state == "ACTIVE":
            has_projected = ShipStatusDialog._module_has_projected_effects(module)
            is_area_effect = ShipStatusDialog._is_area_effect_group(module.group)
            target_required = ShipStatusDialog._module_requires_target(module)
            if target_required and not current_target_id and (ShipStatusDialog._is_weapon_group(module.group) or (has_projected and not is_area_effect)):
                effective_state = "ONLINE"
        return effective_state

    @staticmethod
    def _module_requires_target(module: ModuleRuntime) -> bool:
        if ShipStatusDialog._is_weapon_group(module.group):
            return True
        if ShipStatusDialog._is_area_effect_group(module.group):
            return False
        return ShipStatusDialog._module_has_projected_effects(module)

    @staticmethod
    def _normalize_module_manual_mode(mode: str | None) -> str:
        normalized = str(mode or "auto").strip().lower()
        return normalized if normalized in {"auto", "active", "online"} else "auto"

    @staticmethod
    def _module_supports_manual_mode(module: ModuleRuntime | None) -> bool:
        if module is None:
            return False
        if module.state == ModuleState.OFFLINE:
            return False
        try:
            return bool(module.can_be_active())
        except Exception:
            return False

    @staticmethod
    def _module_manual_mode_options(lang: str) -> list[tuple[str, str]]:
        return [
            ("auto", tr(lang, "status_module_mode_auto")),
            ("active", tr(lang, "status_module_mode_active")),
            ("online", tr(lang, "status_module_mode_online")),
        ]

    @classmethod
    def _module_manual_mode_label(cls, lang: str, mode: str | None) -> str:
        normalized = cls._normalize_module_manual_mode(mode)
        labels = dict(cls._module_manual_mode_options(lang))
        return labels.get(normalized, labels["auto"])

    def _clear_module_mode_widgets(self) -> None:
        self._module_mode_popup_open = False
        for row in range(max(0, self.modules_table.rowCount())):
            for column in (7, 8):
                widget = self.modules_table.cellWidget(row, column)
                if widget is not None:
                    self.modules_table.removeCellWidget(row, column)
                    widget.deleteLater()

    def _on_module_mode_popup_visibility_changed(self, visible: bool) -> None:
        self._module_mode_popup_open = bool(visible)

    def _refresh_module_mode_widgets(self, lang: str) -> None:
        self._clear_module_mode_widgets()
        options = self._module_manual_mode_options(lang)
        for row_index, (module_id, can_override, current_mode) in enumerate(self._module_mode_row_contexts):
            if not can_override or not module_id:
                continue
            combo = _PopupAwareComboBox(self.modules_table)
            combo.popup_visibility_changed.connect(self._on_module_mode_popup_visibility_changed)
            for mode_value, label in options:
                combo.addItem(label, mode_value)
            selected_index = combo.findData(self._normalize_module_manual_mode(current_mode))
            if selected_index < 0:
                selected_index = combo.findData("auto")
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, selected_index))
            combo.blockSignals(False)
            combo.currentIndexChanged.connect(
                lambda _index, c=combo, module_key=module_id: self._on_module_mode_changed(module_key, c)
            )
            self.modules_table.setCellWidget(row_index, 7, combo)
            mode_item = self.modules_table.item(row_index, 7)
            if mode_item is not None:
                mode_item.setText("")

            sync_button = QPushButton(tr(lang, "status_module_mode_sync_button"), self.modules_table)
            sync_button.setAutoDefault(False)
            sync_button.setDefault(False)
            sync_button.clicked.connect(
                lambda _checked=False, c=combo, module_key=module_id: self._on_module_mode_sync_clicked(module_key, c)
            )
            self.modules_table.setCellWidget(row_index, 8, sync_button)
            sync_item = self.modules_table.item(row_index, 8)
            if sync_item is not None:
                sync_item.setText("")

    def _on_module_mode_changed(self, module_id: str, combo: QComboBox) -> None:
        lang = self._language_getter()
        requested_mode = self._normalize_module_manual_mode(str(combo.currentData() or "auto"))
        ok, message = self._module_mode_setter(self.ship_id, module_id, requested_mode)
        self._module_mode_popup_open = False
        if ok:
            self._tab_signatures.pop("modules", None)
            self.refresh_status(force=True)
            return
        QMessageBox.warning(self, tr(lang, "ship_status_title"), tr(lang, "status_lock_failed", error=message))
        self._tab_signatures.pop("modules", None)
        self.refresh_status(force=True)

    def _on_module_mode_sync_clicked(self, module_id: str, combo: QComboBox) -> None:
        lang = self._language_getter()
        requested_mode = self._normalize_module_manual_mode(str(combo.currentData() or "auto"))
        ok, message = self._module_mode_sync_setter(self.ship_id, module_id, requested_mode)
        self._module_mode_popup_open = False
        if ok:
            if message:
                QMessageBox.information(self, tr(lang, "ship_status_title"), message)
            self._tab_signatures.pop("modules", None)
            self.refresh_status(force=True)
            return
        QMessageBox.warning(self, tr(lang, "ship_status_title"), tr(lang, "status_lock_failed", error=message))
        self._tab_signatures.pop("modules", None)
        self.refresh_status(force=True)

    @staticmethod
    def _fmt_charge_amount(value: float) -> str:
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-6:
            return str(int(rounded))
        return f"{float(value):.1f}"

    @staticmethod
    def _fmt_time_pair(remaining: float, total: float) -> str:
        left = max(0.0, float(remaining))
        base = max(0.0, float(total))
        if base <= 0.0:
            base = left
        return f"{left:.1f}/{base:.1f}s"

    @staticmethod
    def _fmt_distance(value: float) -> str:
        distance = max(0.0, float(value))
        if distance >= 1000.0:
            return f"{distance / 1000.0:.1f} km"
        return f"{distance:.0f} m"

    @staticmethod
    def _fmt_percent(value: float) -> str:
        return f"{float(value):.1f}%"

    @staticmethod
    def _fmt_rate(value: float, unit: str) -> str:
        return f"{float(value):.2f} {unit}/s"

    @staticmethod
    def _peak_cap_recharge(profile: ShipProfile) -> float:
        if float(profile.cap_recharge_time) <= 0.0:
            return 0.0
        return 2.5 * float(profile.max_cap) / float(profile.cap_recharge_time)

    @staticmethod
    def _sensor_strength_summary(profile: ShipProfile) -> str:
        parts: list[str] = []
        for label, value in (
            ("Grav", profile.sensor_strength_gravimetric),
            ("Ladar", profile.sensor_strength_ladar),
            ("Mag", profile.sensor_strength_magnetometric),
            ("Radar", profile.sensor_strength_radar),
        ):
            if float(value) > 0.0:
                parts.append(f"{label} {float(value):.1f}")
        return ", ".join(parts) if parts else "-"

    @staticmethod
    def _align_time_for_profile(profile: ShipProfile) -> float:
        mass = max(0.0, float(getattr(profile, "mass", 0.0) or 0.0))
        agility = max(0.0, float(getattr(profile, "agility", 0.0) or 0.0))
        if mass > 0.0 and agility > 0.0:
            return max(0.25, (mass * agility) / 1_000_000.0)
        speed = max(150.0, float(getattr(profile, "max_speed", 0.0) or 0.0))
        return max(2.5, min(14.0, 14_000.0 / speed))

    @staticmethod
    def _format_ship_id_summary(lang: str, ship_ids: list[str]) -> str:
        normalized = sorted(str(ship_id) for ship_id in ship_ids if str(ship_id).strip())
        if not normalized:
            return tr(lang, "status_target_none")
        return ", ".join(normalized)

    @staticmethod
    def _format_lock_timer_summary(
        lang: str,
        timer_entries: list[tuple[str, float]],
    ) -> str:
        normalized = [
            (str(ship_id), max(0.0, float(remaining)))
            for ship_id, remaining in timer_entries
            if str(ship_id).strip() and max(0.0, float(remaining)) > 0.0
        ]
        if not normalized:
            return tr(lang, "status_target_none")
        normalized.sort(key=lambda item: (item[1], item[0]))
        return ", ".join(f"{ship_id} ({remaining:.1f}s)" for ship_id, remaining in normalized)

    def _incoming_lock_status(self, ship, lang: str) -> tuple[str, str]:
        locked_by: list[str] = []
        locking_by: list[tuple[str, float]] = []
        for other in self.engine.world.ships.values():
            if other.ship_id == ship.ship_id or not other.vital.alive:
                continue
            if ship.ship_id in other.combat.lock_targets:
                locked_by.append(str(other.ship_id))
                continue
            remaining = max(0.0, float(other.combat.lock_timers.get(ship.ship_id, 0.0) or 0.0))
            if remaining > 0.0:
                locking_by.append((str(other.ship_id), remaining))
        return (
            self._format_ship_id_summary(lang, locked_by),
            self._format_lock_timer_summary(lang, locking_by),
        )

    @classmethod
    def _layer_omni_ehp(
        cls,
        hp: float,
        resonances: tuple[float, float, float, float],
    ) -> float:
        avg_resonance = max(0.01, sum(float(value) for value in resonances) / 4.0)
        return max(0.0, float(hp)) / avg_resonance

    @classmethod
    def _total_omni_ehp(cls, profile: ShipProfile) -> float:
        return (
            cls._layer_omni_ehp(
                profile.shield_hp,
                (
                    profile.shield_resonance_em,
                    profile.shield_resonance_thermal,
                    profile.shield_resonance_kinetic,
                    profile.shield_resonance_explosive,
                ),
            )
            + cls._layer_omni_ehp(
                profile.armor_hp,
                (
                    profile.armor_resonance_em,
                    profile.armor_resonance_thermal,
                    profile.armor_resonance_kinetic,
                    profile.armor_resonance_explosive,
                ),
            )
            + cls._layer_omni_ehp(
                profile.structure_hp,
                (
                    profile.structure_resonance_em,
                    profile.structure_resonance_thermal,
                    profile.structure_resonance_kinetic,
                    profile.structure_resonance_explosive,
                ),
            )
        )

    @staticmethod
    def _damage_split(profile: ShipProfile) -> str:
        values = {
            "EM": max(0.0, float(profile.damage_em)),
            "TH": max(0.0, float(profile.damage_thermal)),
            "KI": max(0.0, float(profile.damage_kinetic)),
            "EX": max(0.0, float(profile.damage_explosive)),
        }
        total = sum(values.values())
        if total <= 1e-9:
            return "-"
        return " / ".join(f"{label} {amount / total * 100.0:.1f}%" for label, amount in values.items() if amount > 0.0)

    @staticmethod
    def _support_projection_summary(runtime) -> tuple[float, float, float]:
        shield_rep_per_sec = 0.0
        armor_rep_per_sec = 0.0
        cap_warfare_per_sec = 0.0
        if runtime is None:
            return shield_rep_per_sec, armor_rep_per_sec, cap_warfare_per_sec
        for module in runtime.modules:
            for effect in module.effects:
                if effect.effect_class != EffectClass.PROJECTED:
                    continue
                cycle_time = max(0.1, float(effect.cycle_time) or 0.1)
                shield_rep_per_sec += max(0.0, float(effect.projected_add.get("shield_rep", 0.0) or 0.0)) / cycle_time
                armor_rep_per_sec += max(0.0, float(effect.projected_add.get("armor_rep", 0.0) or 0.0)) / cycle_time
                cap_warfare_per_sec += max(0.0, float(effect.projected_add.get("cap_drain", 0.0) or 0.0)) / cycle_time
        return shield_rep_per_sec, armor_rep_per_sec, cap_warfare_per_sec

    @staticmethod
    def _module_cycle_time(module: ModuleRuntime) -> float:
        return min(
            (
                max(0.1, float(effect.cycle_time))
                for effect in module.effects
                if str(effect.state_required.value).upper() == "ACTIVE" and float(effect.cycle_time) > 0.0
            ),
            default=0.0,
        )

    @staticmethod
    def _module_reactivation_delay(module: ModuleRuntime) -> float:
        return max(
            (
                max(0.0, float(getattr(effect, "reactivation_delay", 0.0) or 0.0))
                for effect in module.effects
                if str(effect.state_required.value).upper() == "ACTIVE"
            ),
            default=0.0,
        )

    def _format_module_charge_status(self, lang: str, ship, module: ModuleRuntime) -> str:
        if int(module.charge_capacity) <= 0:
            return ""

        remaining = max(0.0, float(module.charge_remaining))
        capacity = max(0, int(module.charge_capacity))
        return f" | {tr(lang, 'status_charge')}={self._fmt_charge_amount(remaining)}/{capacity}"

    def _format_module_time_status(
        self,
        lang: str,
        ship,
        module: ModuleRuntime,
        effective_state: str,
        cooldown_left: float,
    ) -> str:
        cooldown_left = max(0.0, float(cooldown_left))
        if cooldown_left > 0.0:
            delay_total = self._module_reactivation_delay(module)
            return f" | {tr(lang, 'status_reactivation_time')}={self._fmt_time_pair(cooldown_left, delay_total)}"

        reloading_left = max(
            0.0,
            float(ship.combat.module_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
        )
        if reloading_left > 0.0:
            reload_total = max(0.0, float(module.charge_reload_time))
            return f" | {tr(lang, 'status_reload_time')}={self._fmt_time_pair(reloading_left, reload_total)}"

        if str(effective_state).upper() == "ACTIVE":
            cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module.module_id, 0.0) or 0.0))
            if cycle_left > 0.0:
                cycle_total = self._module_cycle_time(module)
                return f" | {tr(lang, 'status_cycle_time')}={self._fmt_time_pair(cycle_left, cycle_total)}"

        return ""

    def _format_ecm_incoming_status(self, lang: str, ship, now: float) -> str:
        active_sources: list[tuple[str, float]] = []
        for source_id, jam_until in ship.combat.ecm_jam_sources.items():
            remaining = float(jam_until) - now
            if remaining > 0.0:
                active_sources.append((str(source_id), remaining))
        if not active_sources:
            return tr(lang, "status_ecm_none")
        active_sources.sort(key=lambda item: item[1], reverse=True)
        return ", ".join(f"{source_id}({remaining:.1f}s)" for source_id, remaining in active_sources)

    def _format_ecm_attempt_status(self, lang: str, ship, now: float) -> str:
        target_id = str(ship.combat.ecm_last_attempt_target or "").strip()
        if not target_id:
            return tr(lang, "status_ecm_attempt_none")

        module_id = str(ship.combat.ecm_last_attempt_module or "").strip()
        success = ship.combat.ecm_last_attempt_success
        if success is True:
            result_label = tr(lang, "status_ecm_result_success")
        elif success is False:
            result_label = tr(lang, "status_ecm_result_failed")
        else:
            result_label = tr(lang, "status_ecm_result_unknown")

        chance = max(0.0, min(1.0, float(ship.combat.ecm_last_attempt_chance or 0.0)))
        raw_last_attempt_at = ship.combat.ecm_last_attempt_at
        age = max(0.0, now - float(raw_last_attempt_at if raw_last_attempt_at is not None else -1e9))
        head = f"{module_id} -> {target_id}" if module_id else target_id
        return (
            f"{head} | {result_label} | {tr(lang, 'status_ecm_chance')}={chance * 100.0:.1f}% | "
            f"{tr(lang, 'status_ecm_elapsed')}={age:.1f}s"
        )

    def _format_ecm_cycle_result_for_module(self, lang: str, ship, module_id: str, target_id: str | None) -> str:
        success = ship.combat.ecm_last_attempt_success_by_module.get(module_id)
        if success is None:
            return ""
        module_target = str(ship.combat.ecm_last_attempt_target_by_module.get(module_id, "") or "").strip()
        shown_target = str(target_id or "").strip()
        if shown_target and module_target and shown_target != module_target:
            return ""
        result_label = tr(lang, "status_ecm_result_success") if success else tr(lang, "status_ecm_result_failed")
        return f" | {tr(lang, 'status_ecm_cycle_result')}={result_label}"

    def _stable_profile(self, ship):
        if ship.runtime is None:
            return ship.profile
        fit_text = self._fit_text_getter(self.ship_id) or ""
        runtime_state_key = tuple((module.module_id, module.state.value) for module in ship.runtime.modules)
        diagnostics = ship.runtime.diagnostics if isinstance(ship.runtime.diagnostics, dict) else {}
        cache_key = (
            fit_text,
            id(ship.runtime),
            runtime_state_key,
            diagnostics.get("pyfa_resolve_signature"),
            diagnostics.get("pyfa_command_booster_signature"),
            diagnostics.get("pyfa_projected_sources_signature"),
        )
        if self._stable_profile_cache is not None and self._stable_profile_cache_key == cache_key:
            return self._stable_profile_cache

        cached_pyfa_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
        if isinstance(cached_pyfa_profile, ShipProfile):
            base = replace(cached_pyfa_profile)
        else:
            pyfa_profile = recompute_profile_from_pyfa_runtime(ship.runtime)
            if pyfa_profile is not None:
                base = replace(pyfa_profile)
            else:
                base = replace(self._runtime_engine.compute_base_profile(ship.runtime))
        self._stable_profile_cache = base
        self._stable_profile_cache_key = cache_key
        return base

    def _get_module_specs_cached(self, fit_text: str):
        if fit_text == self._cached_fit_text:
            return self._cached_module_specs
        self._cached_fit_text = fit_text
        if not fit_text:
            self._cached_module_specs = []
            return self._cached_module_specs
        try:
            self._cached_module_specs = self._parser.parse(fit_text).module_specs
        except Exception:
            self._cached_module_specs = []
        return self._cached_module_specs

    def _runtime_maps(self, ship) -> tuple[dict[int, str], dict[int, ModuleRuntime], dict[int, str], dict[str, str], dict[str, float]]:
        runtime_by_slot: dict[int, str] = {}
        projected_by_slot: dict[int, str] = {}
        projected_by_module: dict[str, str] = {}
        runtime_module_by_slot: dict[int, ModuleRuntime] = {}
        reactivation_by_module: dict[str, float] = {}
        if ship.runtime is not None:
            for module in ship.runtime.modules:
                slot_index = self._module_index_from_id(module.module_id)
                if slot_index is None:
                    continue
                runtime_by_slot[slot_index] = module.normalized_state().value
                runtime_module_by_slot[slot_index] = module
            reactivation_by_module = {
                str(mid): float(left)
                for mid, left in ship.combat.module_reactivation_timers.items()
                if float(left) > 0.0
            }
            for module_id, target_id in ship.combat.projected_targets.items():
                projected_by_module[module_id] = target_id
                slot_index = self._module_index_from_id(module_id)
                if slot_index is not None:
                    projected_by_slot[slot_index] = target_id
        return runtime_by_slot, runtime_module_by_slot, projected_by_slot, projected_by_module, reactivation_by_module

    def _module_timer_label(
        self,
        lang: str,
        ship,
        module: ModuleRuntime,
        effective_state: str,
        cooldown_left: float,
    ) -> str:
        cooldown_left = max(0.0, float(cooldown_left))
        if cooldown_left > 0.0:
            delay_total = self._module_reactivation_delay(module)
            return f"{tr(lang, 'status_reactivation_time')} {self._fmt_time_pair(cooldown_left, delay_total)}"

        reloading_left = max(
            0.0,
            float(ship.combat.module_ammo_reload_timers.get(module.module_id, 0.0) or 0.0),
        )
        if reloading_left > 0.0:
            reload_total = max(0.0, float(module.charge_reload_time))
            return f"{tr(lang, 'status_reload_time')} {self._fmt_time_pair(reloading_left, reload_total)}"

        if str(effective_state).upper() == "ACTIVE":
            cycle_left = max(0.0, float(ship.combat.module_cycle_timers.get(module.module_id, 0.0) or 0.0))
            if cycle_left > 0.0:
                cycle_total = self._module_cycle_time(module)
                return f"{tr(lang, 'status_cycle_time')} {self._fmt_time_pair(cycle_left, cycle_total)}"

        return "-"

    def _module_rows(self, ship, lang: str) -> list[tuple[str, ...]]:
        fit_text = self._fit_text_getter(self.ship_id) or ""
        module_specs = self._get_module_specs_cached(fit_text)
        runtime_by_slot, runtime_module_by_slot, projected_by_slot, projected_by_module, reactivation_by_module = self._runtime_maps(ship)
        rows: list[tuple[str, ...]] = []
        self._module_mode_row_contexts = []

        if module_specs:
            for idx, spec in enumerate(module_specs, start=1):
                runtime_module = runtime_module_by_slot.get(idx)
                state_key = "OFFLINE" if spec.offline else runtime_by_slot.get(idx, "UNMODELED")
                group_name = runtime_module.group if runtime_module is not None else spec.module_name
                module_name = get_type_display_name(spec.module_name, language=lang)
                if spec.charge_name:
                    module_name = f"{module_name} ({get_type_display_name(spec.charge_name, language=lang)})"
                target_label = "-"
                charge_label = "-"
                timer_label = "-"
                can_override = self._module_supports_manual_mode(runtime_module)
                manual_mode = self._module_manual_mode_label(
                    lang,
                    self._module_mode_getter(self.ship_id, runtime_module.module_id) if can_override and runtime_module is not None else None,
                ) if can_override else "-"
                if runtime_module is not None:
                    has_projected = self._module_has_projected_effects(runtime_module)
                    is_area_effect = self._is_area_effect_group(runtime_module.group)
                    if self._is_weapon_group(runtime_module.group):
                        current_target_id = ship.combat.current_target
                    elif has_projected and not is_area_effect:
                        current_target_id = projected_by_slot.get(idx)
                    else:
                        current_target_id = None
                    state_key = self._display_module_state(runtime_module, state_key, current_target_id)
                    cooldown_left = reactivation_by_module.get(runtime_module.module_id, 0.0)
                    if cooldown_left > 0.0:
                        state_label = tr(lang, "state_REACTIVATING")
                    else:
                        state_label = tr(lang, f"state_{state_key}")
                    charge_label = (
                        f"{self._fmt_charge_amount(runtime_module.charge_remaining)}/{int(runtime_module.charge_capacity)}"
                        if int(runtime_module.charge_capacity) > 0
                        else "-"
                    )
                    timer_label = self._module_timer_label(lang, ship, runtime_module, state_key, cooldown_left)
                    if has_projected and not is_area_effect:
                        target_label = str(projected_by_slot.get(idx) or tr(lang, "status_target_none"))
                    elif self._is_weapon_group(runtime_module.group):
                        target_label = str(ship.combat.current_target or tr(lang, "status_target_none"))
                else:
                    state_label = tr(lang, f"state_{state_key}")
                self._module_mode_row_contexts.append(
                    (
                        str(runtime_module.module_id) if runtime_module is not None else "",
                        can_override,
                        self._module_mode_getter(self.ship_id, runtime_module.module_id) if can_override and runtime_module is not None else "auto",
                    )
                )
                rows.append(
                    (
                        f"[{idx:02d}]",
                        module_name,
                        str(group_name or "-"),
                        state_label,
                        target_label,
                        charge_label,
                        timer_label,
                        manual_mode,
                        tr(lang, "status_module_mode_sync_button") if can_override else "-",
                    )
                )
            return rows

        if ship.runtime is None:
            self._module_mode_row_contexts = [("", False, "auto")]
            return [(tr(lang, "status_module_none"), "-", "-", "-", "-", "-", "-", "-", "-")]

        ordered_modules = sorted(
            ship.runtime.modules,
            key=lambda module: (self._module_index_from_id(module.module_id) or 9999, module.module_id),
        )
        for module in ordered_modules:
            target_label = "-"
            has_projected = self._module_has_projected_effects(module)
            is_area_effect = self._is_area_effect_group(module.group)
            if self._is_weapon_group(module.group):
                current_target_id = ship.combat.current_target
            elif has_projected and not is_area_effect:
                current_target_id = projected_by_module.get(module.module_id)
            else:
                current_target_id = None
            effective_state = self._display_module_state(module, module.normalized_state().value, current_target_id)
            cooldown_left = reactivation_by_module.get(module.module_id, 0.0)
            if cooldown_left > 0.0:
                state_label = tr(lang, "state_REACTIVATING")
            else:
                state_label = tr(lang, f"state_{effective_state}")
            if has_projected and not is_area_effect:
                target_label = str(projected_by_module.get(module.module_id) or tr(lang, "status_target_none"))
            elif self._is_weapon_group(module.group):
                target_label = str(ship.combat.current_target or tr(lang, "status_target_none"))
            can_override = self._module_supports_manual_mode(module)
            manual_mode = (
                self._module_manual_mode_label(lang, self._module_mode_getter(self.ship_id, module.module_id))
                if can_override
                else "-"
            )
            self._module_mode_row_contexts.append(
                (
                    str(module.module_id),
                    can_override,
                    self._module_mode_getter(self.ship_id, module.module_id) if can_override else "auto",
                )
            )
            rows.append(
                (
                    str(self._module_index_from_id(module.module_id) or "-"),
                    module.module_id,
                    str(module.group or "-"),
                    state_label,
                    target_label,
                    (
                        f"{self._fmt_charge_amount(module.charge_remaining)}/{int(module.charge_capacity)}"
                        if int(module.charge_capacity) > 0
                        else "-"
                    ),
                    self._module_timer_label(lang, ship, module, effective_state, cooldown_left),
                    manual_mode,
                    tr(lang, "status_module_mode_sync_button") if can_override else "-",
                )
            )
        return rows

    def _refresh_overview_tab(self, ship, lang: str, force: bool) -> None:
        profile = self._stable_profile(ship)
        hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
        hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
        rows = [
            (tr(lang, "status_ship"), ship.ship_id),
            (tr(lang, "status_type"), get_type_display_name(ship.fit.ship_name, language=lang)),
            (tr(lang, "status_team_squad"), f"{ship.team.value} / {ship.squad_id}"),
            (tr(lang, "status_alive"), str(bool(ship.vital.alive))),
            (tr(lang, "status_backend"), get_fit_backend_status()),
            (tr(lang, "status_speed"), f"{ship.nav.velocity.length():.1f} / {profile.max_speed:.1f} m/s"),
            (tr(lang, "status_facing"), f"{ship.nav.facing_deg:.1f} deg"),
            (tr(lang, "status_hp"), f"{hp_cur:.1f} / {hp_max:.1f} ({hp_cur / max(1.0, hp_max) * 100.0:.1f}%)"),
            (
                tr(lang, "status_cap"),
                f"{ship.vital.cap:.1f} / {ship.vital.cap_max:.1f} ({ship.vital.cap / max(1.0, ship.vital.cap_max) * 100.0:.1f}%)",
            ),
            (tr(lang, "status_target"), str(ship.combat.current_target or tr(lang, "status_target_none"))),
            (tr(lang, "status_locked_targets"), f"{len(ship.combat.lock_targets)} / {max(0, int(profile.max_locked_targets))}"),
            (tr(lang, "status_ecm_incoming"), self._format_ecm_incoming_status(lang, ship, float(self.engine.world.now))),
            (tr(lang, "status_ecm_last_attempt"), self._format_ecm_attempt_status(lang, ship, float(self.engine.world.now))),
        ]
        headers = (tr(lang, "status_metric"), tr(lang, "status_value"))
        signature = self._table_signature(rows, headers)
        if not force and self._tab_signatures.get("overview") == signature:
            return
        self._tab_signatures["overview"] = signature
        self._apply_table(self.overview_table, list(headers), rows)

    def _refresh_combat_tab(self, ship, lang: str, force: bool) -> None:
        profile = self._stable_profile(ship)
        shield_rep_ps, armor_rep_ps, cap_warfare_ps = self._support_projection_summary(ship.runtime)
        rows = [
            (tr(lang, "status_dps"), f"{profile.dps:.2f}"),
            (tr(lang, "status_dph"), f"{profile.volley:.2f}"),
            (tr(lang, "status_turret_dps"), f"{profile.turret_dps:.2f}"),
            (tr(lang, "status_missile_dps"), f"{profile.missile_dps:.2f}"),
            (tr(lang, "status_optimal"), self._fmt_distance(profile.optimal)),
            (tr(lang, "status_falloff_short"), self._fmt_distance(profile.falloff)),
            (tr(lang, "status_tracking_short"), f"{profile.tracking:.4f}"),
            (tr(lang, "status_optimal_sig"), f"{profile.optimal_sig:.1f} m"),
            (tr(lang, "status_missile_range"), self._fmt_distance(profile.missile_max_range)),
            (tr(lang, "status_explosion_radius"), f"{profile.missile_explosion_radius:.1f} m"),
            (tr(lang, "status_explosion_velocity"), f"{profile.missile_explosion_velocity:.1f} m/s"),
            (tr(lang, "status_damage_split"), self._damage_split(profile)),
            (tr(lang, "status_remote_shield_rep_ps"), self._fmt_rate(shield_rep_ps, "HP")),
            (tr(lang, "status_remote_armor_rep_ps"), self._fmt_rate(armor_rep_ps, "HP")),
            (tr(lang, "status_cap_warfare_ps"), self._fmt_rate(cap_warfare_ps, "GJ")),
        ]
        headers = (tr(lang, "status_metric"), tr(lang, "status_value"))
        signature = self._table_signature(rows, headers)
        if not force and self._tab_signatures.get("combat") == signature:
            return
        self._tab_signatures["combat"] = signature
        self._apply_table(self.combat_table, list(headers), rows)

    def _refresh_defense_tab(self, ship, lang: str, force: bool) -> None:
        profile = self._stable_profile(ship)
        summary_rows = [
            (tr(lang, "status_shield"), f"{ship.vital.shield:.1f} / {ship.vital.shield_max:.1f}"),
            (tr(lang, "status_armor"), f"{ship.vital.armor:.1f} / {ship.vital.armor_max:.1f}"),
            (tr(lang, "status_structure"), f"{ship.vital.structure:.1f} / {ship.vital.structure_max:.1f}"),
            (tr(lang, "status_total_raw_hp"), f"{profile.shield_hp + profile.armor_hp + profile.structure_hp:.1f}"),
            (tr(lang, "status_total_ehp"), f"{self._total_omni_ehp(profile):.1f}"),
        ]
        resistance_rows = [
            (
                tr(lang, "status_shield"),
                self._fmt_percent(self._res_pct(profile.shield_resonance_em)),
                self._fmt_percent(self._res_pct(profile.shield_resonance_thermal)),
                self._fmt_percent(self._res_pct(profile.shield_resonance_kinetic)),
                self._fmt_percent(self._res_pct(profile.shield_resonance_explosive)),
                f"{self._layer_omni_ehp(profile.shield_hp, (profile.shield_resonance_em, profile.shield_resonance_thermal, profile.shield_resonance_kinetic, profile.shield_resonance_explosive)):.1f}",
            ),
            (
                tr(lang, "status_armor"),
                self._fmt_percent(self._res_pct(profile.armor_resonance_em)),
                self._fmt_percent(self._res_pct(profile.armor_resonance_thermal)),
                self._fmt_percent(self._res_pct(profile.armor_resonance_kinetic)),
                self._fmt_percent(self._res_pct(profile.armor_resonance_explosive)),
                f"{self._layer_omni_ehp(profile.armor_hp, (profile.armor_resonance_em, profile.armor_resonance_thermal, profile.armor_resonance_kinetic, profile.armor_resonance_explosive)):.1f}",
            ),
            (
                tr(lang, "status_structure"),
                self._fmt_percent(self._res_pct(profile.structure_resonance_em)),
                self._fmt_percent(self._res_pct(profile.structure_resonance_thermal)),
                self._fmt_percent(self._res_pct(profile.structure_resonance_kinetic)),
                self._fmt_percent(self._res_pct(profile.structure_resonance_explosive)),
                f"{self._layer_omni_ehp(profile.structure_hp, (profile.structure_resonance_em, profile.structure_resonance_thermal, profile.structure_resonance_kinetic, profile.structure_resonance_explosive)):.1f}",
            ),
        ]
        summary_headers = (tr(lang, "status_metric"), tr(lang, "status_value"))
        resistance_headers = (tr(lang, "status_layer"), "EM", "TH", "KI", "EX", tr(lang, "status_omni_ehp"))
        signature = (
            self._table_signature(summary_rows, summary_headers),
            self._table_signature(resistance_rows, resistance_headers),
        )
        if not force and self._tab_signatures.get("defense") == signature:
            return
        self._tab_signatures["defense"] = signature
        self._apply_table(self.defense_summary_table, list(summary_headers), summary_rows)
        self._apply_table(self.defense_resistance_table, list(resistance_headers), resistance_rows)

    def _refresh_cap_target_tab(self, ship, lang: str, force: bool) -> None:
        profile = self._stable_profile(ship)
        outgoing_locking = [
            (str(target_id), float(remaining))
            for target_id, remaining in ship.combat.lock_timers.items()
            if str(target_id) not in ship.combat.lock_targets
        ]
        locked_by_text, locking_by_text = self._incoming_lock_status(ship, lang)
        capacitor_rows = [
            (tr(lang, "status_cap"), f"{ship.vital.cap:.1f} / {ship.vital.cap_max:.1f}"),
            (tr(lang, "status_cap_peak_recharge"), f"{self._peak_cap_recharge(profile):.2f} GJ/s"),
            (tr(lang, "status_cap_recharge_time"), f"{profile.cap_recharge_time:.2f}s"),
            (tr(lang, "status_cap_resistance"), self._fmt_percent((1.0 - float(profile.energy_warfare_resistance or 1.0)) * 100.0)),
        ]
        targeting_rows = [
            (tr(lang, "status_targets"), str(max(0, int(profile.max_locked_targets)))),
            (tr(lang, "status_target_range"), self._fmt_distance(profile.max_target_range)),
            (tr(lang, "status_scan_resolution"), f"{profile.scan_resolution:.1f} mm"),
            (tr(lang, "status_sensor_strengths"), self._sensor_strength_summary(profile)),
            (tr(lang, "status_signature_radius"), f"{profile.sig_radius:.1f} m"),
            (tr(lang, "status_align_time"), f"{self._align_time_for_profile(profile):.2f}s"),
            (tr(lang, "status_mass"), f"{profile.mass:,.0f} kg"),
            (tr(lang, "status_agility"), f"{profile.agility:.3f}"),
            (tr(lang, "status_warp_stability"), f"{profile.warp_stability:.1f}"),
            (tr(lang, "status_warp_scramble"), f"{profile.warp_scramble_status:.1f}"),
            (tr(lang, "status_current_target"), str(ship.combat.current_target or tr(lang, "status_target_none"))),
            (tr(lang, "status_locked_targets_detail"), self._format_ship_id_summary(lang, list(ship.combat.lock_targets))),
            (tr(lang, "status_locking_targets"), self._format_lock_timer_summary(lang, outgoing_locking)),
            (tr(lang, "status_locked_by"), locked_by_text),
            (tr(lang, "status_locking_by"), locking_by_text),
        ]
        headers = (tr(lang, "status_metric"), tr(lang, "status_value"))
        signature = (
            self._table_signature(capacitor_rows, headers),
            self._table_signature(targeting_rows, headers),
        )
        if not force and self._tab_signatures.get("cap_target") == signature:
            return
        self._tab_signatures["cap_target"] = signature
        self._apply_table(self.capacitor_table, list(headers), capacitor_rows)
        self._apply_table(self.targeting_table, list(headers), targeting_rows)

    def _refresh_modules_tab(self, ship, lang: str, force: bool) -> None:
        rows = self._module_rows(ship, lang)
        headers = (
            tr(lang, "status_col_slot"),
            tr(lang, "status_col_module"),
            tr(lang, "status_col_group"),
            tr(lang, "status_col_state"),
            tr(lang, "status_col_target"),
            tr(lang, "status_col_charge"),
            tr(lang, "status_col_timer"),
            tr(lang, "status_col_mode"),
            tr(lang, "status_col_sync"),
        )
        signature = self._table_signature(rows, headers)
        if not force and self._tab_signatures.get("modules") == signature:
            return
        self._tab_signatures["modules"] = signature
        self._apply_table(self.modules_table, list(headers), rows)
        self._refresh_module_mode_widgets(lang)

    def _refresh_debug_tab(self, ship, lang: str, force: bool) -> None:
        diagnostics = {}
        if ship.runtime is not None and isinstance(ship.runtime.diagnostics, dict):
            for key, value in ship.runtime.diagnostics.items():
                if key in {"pyfa_blueprint", "pyfa_command_boosters", "pyfa_projected_sources"}:
                    continue
                diagnostics[key] = value
            diagnostics["pyfa_command_boosters_count"] = len(ship.runtime.diagnostics.get("pyfa_command_boosters", []))
            diagnostics["pyfa_projected_sources_count"] = len(ship.runtime.diagnostics.get("pyfa_projected_sources", []))
        lines = [
            f"{tr(lang, 'status_backend')}: {get_fit_backend_status()}",
            f"{tr(lang, 'status_ship')}: {ship.ship_id}",
            f"runtime: {'yes' if ship.runtime is not None else 'no'}",
            f"profile: dps={ship.profile.dps:.2f}, volley={ship.profile.volley:.2f}, speed={ship.profile.max_speed:.2f}",
        ]
        if diagnostics:
            lines.append("")
            lines.append("diagnostics:")
            lines.append(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str))
        fit_text = (self._fit_text_getter(self.ship_id) or "").strip()
        if fit_text:
            lines.append("")
            lines.append("fit:")
            lines.append(fit_text)
        text = "\n".join(lines)
        if not force and self._tab_signatures.get("debug") == text:
            return
        self._tab_signatures["debug"] = text
        self.info.setPlainText(text)

    def _clear_views_for_missing_ship(self, lang: str) -> None:
        self._clear_module_mode_widgets()
        self._module_mode_row_contexts = []
        metric_headers = [tr(lang, "status_metric"), tr(lang, "status_value")]
        missing_rows = [(tr(lang, "ship_missing"), "-")]
        self._apply_table(self.overview_table, metric_headers, missing_rows)
        self._apply_table(self.combat_table, metric_headers, missing_rows)
        self._apply_table(self.defense_summary_table, metric_headers, missing_rows)
        self._apply_table(self.capacitor_table, metric_headers, missing_rows)
        self._apply_table(self.targeting_table, metric_headers, missing_rows)
        self._apply_table(
            self.defense_resistance_table,
            [tr(lang, "status_layer"), "EM", "TH", "KI", "EX", tr(lang, "status_omni_ehp")],
            [(tr(lang, "ship_missing"), "-", "-", "-", "-", "-")],
        )
        self._apply_table(
            self.modules_table,
            [
                tr(lang, "status_col_slot"),
                tr(lang, "status_col_module"),
                tr(lang, "status_col_group"),
                tr(lang, "status_col_state"),
                tr(lang, "status_col_target"),
                tr(lang, "status_col_charge"),
                tr(lang, "status_col_timer"),
                tr(lang, "status_col_mode"),
                tr(lang, "status_col_sync"),
            ],
            [(tr(lang, "ship_missing"), "-", "-", "-", "-", "-", "-", "-", "-")],
        )
        self.info.setPlainText(tr(lang, "ship_missing"))

    def refresh_status(self, force: bool = False) -> None:
        lang = self._language_getter()
        self.setWindowTitle(f"{tr(lang, 'ship_status_title')} - {self.ship_id}")
        self._retitle_tabs()
        self._refresh_lock_controls_if_needed()
        ship = self.engine.world.ships.get(self.ship_id)
        if ship is None:
            self._clear_views_for_missing_ship(lang)
            return
        tab_key = self._current_tab_key()
        if tab_key == "modules" and self._module_mode_popup_open and not force:
            return
        if tab_key == "overview":
            self._refresh_overview_tab(ship, lang, force)
        elif tab_key == "combat":
            self._refresh_combat_tab(ship, lang, force)
        elif tab_key == "defense":
            self._refresh_defense_tab(ship, lang, force)
        elif tab_key == "cap_target":
            self._refresh_cap_target_tab(ship, lang, force)
        elif tab_key == "modules":
            self._refresh_modules_tab(ship, lang, force)
        else:
            self._refresh_debug_tab(ship, lang, force)



