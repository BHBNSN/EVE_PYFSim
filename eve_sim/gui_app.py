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

from .agents import CommanderAgent
from .config import EngineConfig, UiConfig
from .fleet_setup import (
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
from .fit_runtime import EffectClass, ModuleRuntime, ModuleState, RuntimeStatEngine
from .i18n import tr
from .lan_session import ClientLanSession, HostLanSession
from .lan_commands import (
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
from .math2d import Vector2
from .models import (
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
from .pyfa_bridge import PyfaBridge
from .sim_logging import get_sim_logger, log_sim_event
from .simulation_engine import SimulationEngine
from .systems import CombatSystem


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


@dataclass(slots=True)
class UiState:
    selected_squad: str = "BLUE-ALPHA"
    selected_enemy_target: str | None = None


@dataclass(slots=True)
class UiPreferences:
    config_version: int = 6
    selected_squad: str = "BLUE-ALPHA"
    filter_team: Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"] = "ALL"
    filter_role: str = "ALL"
    filter_alive: str = "ALL"
    filter_squad: str = ""
    sort_key: str = "Distance"
    sort_order: str = "ASC"
    zoom: float | None = None
    language: str = "zh_CN"
    engine_tick_rate: int = 30
    engine_physics_substeps: int = 1
    engine_lockstep: bool = True
    engine_battlefield_radius: float = 800_000.0
    engine_detailed_logging: bool = True
    engine_hotspot_logging: bool = False
    engine_detail_log_file: str = "logs/sim_detail.log"
    engine_hotspot_log_file: str = "logs/sim_hotspot.log"
    engine_log_merge_window_sec: float = 1.0


@dataclass(slots=True)
class SetupRow:
    team: Team
    squad_id: str
    quality: QualityLevel
    quantity: int
    fit_text: str
    fit_name: str = ""
    is_leader: bool = False


@dataclass(slots=True)
class AreaCycleOverlay:
    ship_id: str
    module_id: str
    center: Vector2
    radius_m: float
    fill_color: QColor
    border_color: QColor
    expires_at: float


class FleetSetupTableModel(QAbstractTableModel):
    HEADER_KEYS = [
        "setup_col_team",
        "setup_col_squad",
        "setup_col_quality",
        "setup_col_quantity",
        "setup_col_leader",
        "setup_col_fit",
    ]

    def __init__(self, rows: list[SetupRow], language_getter: Callable[[], str]) -> None:
        super().__init__()
        self.rows = rows
        self._language_getter = language_getter

    def _headers(self) -> list[str]:
        lang = self._language_getter()
        return [tr(lang, key) for key in self.HEADER_KEYS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADER_KEYS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            headers = self._headers()
            if 0 <= section < len(headers):
                return headers[section]
            return None
        return str(section + 1)

    def notify_headers_changed(self) -> None:
        if self.columnCount() > 0:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        editable_cols = {0, 1, 2, 3, 4}
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if index.column() in editable_cols:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.rows)):
            return None
        row = self.rows[index.row()]
        col = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == 0:
                return row.team.value
            if col == 1:
                return row.squad_id
            if col == 2:
                return row.quality.value
            if col == 3:
                return int(max(1, row.quantity))
            if col == 4:
                return tr(self._language_getter(), "yes_short") if row.is_leader else tr(self._language_getter(), "no_short")
            if col == 5:
                return row.fit_name
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        row = self.rows[index.row()]
        col = index.column()
        try:
            if col == 0:
                row.team = Team(str(value).strip().upper())
            elif col == 1:
                text = str(value).strip()
                if not text:
                    return False
                row.squad_id = text
            elif col == 2:
                row.quality = QualityLevel(str(value).strip().upper())
            elif col == 3:
                row.quantity = max(1, int(float(value)))
                if row.quantity != 1:
                    row.is_leader = False
            elif col == 4:
                text = str(value).strip().upper()
                row.is_leader = (row.quantity == 1) and (text in ("1", "Y", "YES", "TRUE", "T", "队长", "LEADER"))
            else:
                return False
        except Exception:
            return False
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        return True

    def add_row(self, row: SetupRow) -> None:
        self.beginInsertRows(QModelIndex(), len(self.rows), len(self.rows))
        self.rows.append(row)
        self.endInsertRows()

    def remove_row(self, row_index: int) -> None:
        if not (0 <= row_index < len(self.rows)):
            return
        self.beginRemoveRows(QModelIndex(), row_index, row_index)
        self.rows.pop(row_index)
        self.endRemoveRows()

    def update_fit_meta(self, row_index: int, fit_name: str) -> None:
        if not (0 <= row_index < len(self.rows)):
            return
        self.rows[row_index].fit_name = fit_name
        idx = self.index(row_index, 5)
        self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DisplayRole])

    def replace_rows(self, rows: list[SetupRow]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


class SetupRowDelegate(QStyledItemDelegate):
    def __init__(self, language_getter: Callable[[], str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._language_getter = language_getter

    def createEditor(self, parent, option, index):
        col = index.column()
        if col == 2:
            combo = QComboBox(parent)
            combo.setFrame(False)
            combo.setStyleSheet("QComboBox { padding-left: 0px; }")
            combo.addItem(QualityLevel.ELITE.value)
            combo.addItem(QualityLevel.REGULAR.value)
            combo.addItem(QualityLevel.IRREGULAR.value)
            QTimer.singleShot(0, combo.showPopup)
            return combo
        if col == 4:
            combo = QComboBox(parent)
            combo.setFrame(False)
            combo.setStyleSheet("QComboBox { padding-left: 0px; }")
            combo.addItem(tr(self._language_getter(), "yes_short"), True)
            combo.addItem(tr(self._language_getter(), "no_short"), False)
            QTimer.singleShot(0, combo.showPopup)
            return combo
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            col = index.column()
            value = index.model().data(index, Qt.ItemDataRole.EditRole)
            if col == 2:
                text = str(value or QualityLevel.REGULAR.value)
                pos = editor.findText(text)
                editor.setCurrentIndex(0 if pos < 0 else pos)
                return
            if col == 4:
                truthy = str(value).strip().upper() in ("1", "Y", "YES", "TRUE", "T", "是")
                pos = editor.findData(truthy)
                editor.setCurrentIndex(0 if pos < 0 else pos)
                return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            col = index.column()
            if col == 2:
                model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
                return
            if col == 4:
                flag = bool(editor.currentData())
                model.setData(index, "Y" if flag else "N", Qt.ItemDataRole.EditRole)
                return
        super().setModelData(editor, model, index)


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


class FleetSetupDialog(QDialog):
    def __init__(self, network_mode: str = "local", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = PreferencesStore()
        self._pref = self._store.load()
        self._network_mode = network_mode
        self._engine_pref_loading = False
        self._lang = self._detect_initial_language()
        self.setWindowTitle(tr(self._lang, "setup_title"))
        self.resize(1060, 680)
        self._parser = EftFitParser()
        self._factory = RuntimeFromEftFactory()
        self._fleet_store_path = Path.home() / ".eve_sim_fleet_configs.json"
        self._fleet_templates = self._load_fleet_templates()

        init_rows: list[SetupRow] = []
        self.model = FleetSetupTableModel(init_rows, lambda: self._lang)

        layout = QVBoxLayout(self)

        lang_row = QHBoxLayout()
        self.lbl_lang = QLabel(tr(self._lang, "lang_label"))
        self.lang_combo = QComboBox(self)
        self.lang_combo.addItem("中文", "zh_CN")
        self.lang_combo.addItem("English", "en_US")
        idx = self.lang_combo.findData(self._lang)
        self.lang_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.lang_combo.currentIndexChanged.connect(self._on_setup_language_changed)
        lang_row.addWidget(self.lbl_lang)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch(1)
        layout.addLayout(lang_row)

        self.hint = QLabel(tr(self._lang, "setup_hint"))
        layout.addWidget(self.hint)
        self.backend = QLabel(f"{tr(self._lang, 'setup_backend')}: {get_fit_backend_status()}")
        layout.addWidget(self.backend)

        manager_row = QHBoxLayout()
        self.btn_open_fleet_library = QPushButton(tr(self._lang, "fleet_lib_open"))
        manager_row.addWidget(self.btn_open_fleet_library)
        manager_row.addStretch(1)
        layout.addLayout(manager_row)

        fleet_row = QHBoxLayout()
        self.lbl_blue_fleet = QLabel(tr(self._lang, "setup_blue_fleet"))
        self.blue_fleet_combo = QComboBox(self)
        self.lbl_red_fleet = QLabel(tr(self._lang, "setup_red_fleet"))
        self.red_fleet_combo = QComboBox(self)
        fleet_row.addWidget(self.lbl_blue_fleet)
        fleet_row.addWidget(self.blue_fleet_combo, 1)
        fleet_row.addSpacing(10)
        fleet_row.addWidget(self.lbl_red_fleet)
        fleet_row.addWidget(self.red_fleet_combo, 1)
        layout.addLayout(fleet_row)

        self.setup_tabs = QTabWidget(self)

        preview_tab = QWidget(self)
        preview_row = QHBoxLayout(preview_tab)
        blue_col = QVBoxLayout()
        self.lbl_blue_preview = QLabel(tr(self._lang, "setup_blue_preview"))
        self.blue_preview = QPlainTextEdit(self)
        self.blue_preview.setReadOnly(True)
        blue_col.addWidget(self.lbl_blue_preview)
        blue_col.addWidget(self.blue_preview)
        red_col = QVBoxLayout()
        self.lbl_red_preview = QLabel(tr(self._lang, "setup_red_preview"))
        self.red_preview = QPlainTextEdit(self)
        self.red_preview.setReadOnly(True)
        red_col.addWidget(self.lbl_red_preview)
        red_col.addWidget(self.red_preview)
        preview_row.addLayout(blue_col, 1)
        preview_row.addLayout(red_col, 1)
        self.setup_tabs.addTab(preview_tab, tr(self._lang, "setup_tab_preview"))

        settings_tab = QWidget(self)
        settings_layout = QVBoxLayout(settings_tab)
        self.lbl_engine_hint = QLabel(settings_tab)
        self.lbl_engine_hint.setWordWrap(True)
        settings_layout.addWidget(self.lbl_engine_hint)

        settings_form = QFormLayout()
        self.lbl_cfg_tick_rate = QLabel(settings_tab)
        self.spin_cfg_tick_rate = QSpinBox(settings_tab)
        self.spin_cfg_tick_rate.setRange(1, 240)

        self.lbl_cfg_substeps = QLabel(settings_tab)
        self.spin_cfg_substeps = QSpinBox(settings_tab)
        self.spin_cfg_substeps.setRange(1, 16)

        self.lbl_cfg_radius = QLabel(settings_tab)
        self.spin_cfg_radius = QDoubleSpinBox(settings_tab)
        self.spin_cfg_radius.setRange(1_000.0, 20_000_000.0)
        self.spin_cfg_radius.setDecimals(0)
        self.spin_cfg_radius.setSingleStep(10_000.0)

        self.lbl_cfg_lockstep = QLabel(settings_tab)
        self.chk_cfg_lockstep = QCheckBox(settings_tab)

        self.lbl_cfg_detailed_log = QLabel(settings_tab)
        self.chk_cfg_detailed_log = QCheckBox(settings_tab)

        self.lbl_cfg_hotspot_log = QLabel(settings_tab)
        self.chk_cfg_hotspot_log = QCheckBox(settings_tab)

        self.lbl_cfg_log_file = QLabel(settings_tab)
        self.edit_cfg_log_file = QLineEdit(settings_tab)

        self.lbl_cfg_hotspot_log_file = QLabel(settings_tab)
        self.edit_cfg_hotspot_log_file = QLineEdit(settings_tab)

        self.lbl_cfg_log_merge_window = QLabel(settings_tab)
        self.spin_cfg_log_merge_window = QDoubleSpinBox(settings_tab)
        self.spin_cfg_log_merge_window.setRange(0.1, 30.0)
        self.spin_cfg_log_merge_window.setDecimals(1)
        self.spin_cfg_log_merge_window.setSingleStep(0.1)

        settings_form.addRow(self.lbl_cfg_tick_rate, self.spin_cfg_tick_rate)
        settings_form.addRow(self.lbl_cfg_substeps, self.spin_cfg_substeps)
        settings_form.addRow(self.lbl_cfg_radius, self.spin_cfg_radius)
        settings_form.addRow(self.lbl_cfg_lockstep, self.chk_cfg_lockstep)
        settings_form.addRow(self.lbl_cfg_detailed_log, self.chk_cfg_detailed_log)
        settings_form.addRow(self.lbl_cfg_hotspot_log, self.chk_cfg_hotspot_log)
        settings_form.addRow(self.lbl_cfg_log_file, self.edit_cfg_log_file)
        settings_form.addRow(self.lbl_cfg_hotspot_log_file, self.edit_cfg_hotspot_log_file)
        settings_form.addRow(self.lbl_cfg_log_merge_window, self.spin_cfg_log_merge_window)
        settings_layout.addLayout(settings_form)
        settings_layout.addStretch(1)

        self.setup_tabs.addTab(settings_tab, tr(self._lang, "setup_tab_settings"))
        layout.addWidget(self.setup_tabs, 1)

        top = QHBoxLayout()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        top.addWidget(self.table, 3)

        side = QVBoxLayout()
        self.btn_add_blue = QPushButton(tr(self._lang, "setup_add_blue"))
        self.btn_add_red = QPushButton(tr(self._lang, "setup_add_red"))
        self.btn_delete = QPushButton(tr(self._lang, "setup_delete"))
        self.btn_validate = QPushButton(tr(self._lang, "setup_validate"))
        side.addWidget(self.btn_add_blue)
        side.addWidget(self.btn_add_red)
        side.addWidget(self.btn_delete)
        side.addWidget(self.btn_validate)
        side.addStretch(1)
        top.addLayout(side, 1)
        layout.addLayout(top)
        self.table.setVisible(False)
        self.btn_add_blue.setVisible(False)
        self.btn_add_red.setVisible(False)
        self.btn_delete.setVisible(False)
        self.btn_validate.setVisible(False)

        self.fit_editor = QPlainTextEdit()
        self.fit_editor.setPlaceholderText(tr(self._lang, "setup_fit_placeholder"))
        layout.addWidget(self.fit_editor, 2)
        self.fit_editor.setVisible(False)

        self.btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setText(tr(self._lang, "setup_start"))
        self.btns.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(self._lang, "setup_cancel"))
        layout.addWidget(self.btns)

        self.btn_add_blue.clicked.connect(self._add_blue)
        self.btn_add_red.clicked.connect(self._add_red)
        self.btn_delete.clicked.connect(self._delete_current)
        self.btn_validate.clicked.connect(self._validate_all)
        self.btn_open_fleet_library.clicked.connect(self._open_fleet_library)
        self.blue_fleet_combo.currentTextChanged.connect(self._on_blue_fleet_changed)
        self.red_fleet_combo.currentTextChanged.connect(self._on_red_fleet_changed)
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self.fit_editor.textChanged.connect(self._on_fit_changed)
        self.spin_cfg_tick_rate.valueChanged.connect(self._on_engine_pref_changed)
        self.spin_cfg_substeps.valueChanged.connect(self._on_engine_pref_changed)
        self.spin_cfg_radius.valueChanged.connect(self._on_engine_pref_changed)
        self.chk_cfg_lockstep.toggled.connect(self._on_engine_pref_changed)
        self.chk_cfg_detailed_log.toggled.connect(self._on_engine_pref_changed)
        self.chk_cfg_hotspot_log.toggled.connect(self._on_engine_pref_changed)
        self.edit_cfg_log_file.textChanged.connect(self._on_engine_pref_changed)
        self.edit_cfg_hotspot_log_file.textChanged.connect(self._on_engine_pref_changed)
        self.spin_cfg_log_merge_window.valueChanged.connect(self._on_engine_pref_changed)

        self._load_engine_preferences_into_controls()
        self._set_engine_settings_enabled(self._network_mode != "client")
        self._refresh_fleet_combo_items()
        self._apply_setup_language()
        self._rebuild_rows_from_selected_fleets()

    def _default_fleet_templates(self) -> dict[str, list[dict]]:
        return {}

    def _load_fleet_templates(self) -> dict[str, list[dict]]:
        try:
            if not self._fleet_store_path.exists():
                data = self._default_fleet_templates()
                self._fleet_store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return data
            raw = json.loads(self._fleet_store_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("invalid template store")
            out: dict[str, list[dict]] = {}
            for name, entries in raw.items():
                if not isinstance(name, str) or not isinstance(entries, list):
                    continue
                cleaned: list[dict] = []
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    quantity_raw = item.get("quantity", 1)
                    try:
                        quantity = max(1, int(float(quantity_raw)))
                    except Exception:
                        quantity = 1
                    cleaned.append(
                        {
                            "squad_id": str(item.get("squad_id", "ALPHA")).strip() or "ALPHA",
                            "quality": str(item.get("quality", "REGULAR")).strip() or "REGULAR",
                            "quantity": quantity,
                            "fit_text": str(item.get("fit_text", "")).strip(),
                            "is_leader": bool(item.get("is_leader", False)) and quantity == 1,
                        }
                    )
                out[name] = cleaned
            return out
        except Exception:
            return self._default_fleet_templates()

    def _save_fleet_templates(self) -> None:
        try:
            self._fleet_store_path.write_text(json.dumps(self._fleet_templates, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _refresh_fleet_combo_items(self) -> None:
        names = sorted(self._fleet_templates.keys())
        self.blue_fleet_combo.blockSignals(True)
        self.red_fleet_combo.blockSignals(True)
        self.blue_fleet_combo.clear()
        self.red_fleet_combo.clear()
        self.blue_fleet_combo.addItem("")
        self.red_fleet_combo.addItem("")
        self.blue_fleet_combo.addItems(names)
        self.red_fleet_combo.addItems(names)
        self.blue_fleet_combo.setCurrentText("")
        self.red_fleet_combo.setCurrentText("")
        self.blue_fleet_combo.blockSignals(False)
        self.red_fleet_combo.blockSignals(False)

    def _preview_text_for_fleet(self, fleet_name: str) -> str:
        entries = self._fleet_templates.get(fleet_name, []) if fleet_name else []
        if not entries:
            return tr(self._lang, "setup_preview_empty")
        lines: list[str] = []
        for idx, item in enumerate(entries, start=1):
            fit_text = str(item.get("fit_text", "")).strip()
            fit_name = ""
            if fit_text:
                try:
                    parsed = self._parser.parse(fit_text)
                    fit_name = f"{parsed.ship_name} / {parsed.fit_name}"
                except Exception:
                    fit_name = tr(self._lang, "setup_preview_invalid")
            lines.append(f"{idx:02d}. {item.get('squad_id', 'ALPHA')} | {item.get('quality', 'REGULAR')} | {fit_name}")
        return "\n".join(lines)

    def _refresh_previews(self) -> None:
        self.blue_preview.setPlainText(self._preview_text_for_fleet(self.blue_fleet_combo.currentText().strip()))
        self.red_preview.setPlainText(self._preview_text_for_fleet(self.red_fleet_combo.currentText().strip()))

    def _rows_for_team_fleet(self, team: Team, fleet_name: str) -> list[SetupRow]:
        rows: list[SetupRow] = []
        entries = self._fleet_templates.get(fleet_name, []) if fleet_name else []
        for item in entries:
            quality_text = str(item.get("quality", "REGULAR")).strip().upper()
            try:
                quality = QualityLevel(quality_text)
            except Exception:
                quality = QualityLevel.REGULAR
            row = SetupRow(
                team=team,
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
            rows.append(row)
        return rows

    def _rebuild_rows_from_selected_fleets(self) -> None:
        blue_name = self.blue_fleet_combo.currentText().strip()
        red_name = self.red_fleet_combo.currentText().strip()
        rows = self._rows_for_team_fleet(Team.BLUE, blue_name) + self._rows_for_team_fleet(Team.RED, red_name)
        self.model.replace_rows(rows)
        self._refresh_previews()

    def _open_fleet_library(self) -> None:
        dlg = FleetLibraryDialog(self._fleet_templates, self._lang, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._fleet_templates = dlg.templates
        self._save_fleet_templates()
        blue_before = self.blue_fleet_combo.currentText().strip()
        red_before = self.red_fleet_combo.currentText().strip()
        self._refresh_fleet_combo_items()
        if blue_before in self._fleet_templates:
            self.blue_fleet_combo.setCurrentText(blue_before)
        if red_before in self._fleet_templates:
            self.red_fleet_combo.setCurrentText(red_before)
        self._rebuild_rows_from_selected_fleets()

    def _on_blue_fleet_changed(self, fleet_name: str) -> None:
        self._rebuild_rows_from_selected_fleets()

    def _on_red_fleet_changed(self, fleet_name: str) -> None:
        self._rebuild_rows_from_selected_fleets()

    def _detect_initial_language(self) -> str:
        system_name = (QLocale.system().name() or "").lower()
        if system_name.startswith("zh"):
            return "zh_CN"
        if system_name.startswith("en"):
            return "en_US"
        lang = (self._pref.language or "").strip()
        if lang in ("zh_CN", "en_US"):
            return lang
        return "zh_CN"

    def _on_setup_language_changed(self, _index: int) -> None:
        lang = self.lang_combo.currentData()
        self._lang = str(lang) if lang in ("zh_CN", "en_US") else "zh_CN"
        self._pref.language = self._lang
        self._store.save(self._pref)
        self._apply_setup_language()

    def _apply_setup_language(self) -> None:
        lang = self._lang
        self.setWindowTitle(tr(lang, "setup_title"))
        self.lbl_lang.setText(tr(lang, "lang_label"))
        self.lbl_blue_fleet.setText(tr(lang, "setup_blue_fleet"))
        self.lbl_red_fleet.setText(tr(lang, "setup_red_fleet"))
        self.lbl_blue_preview.setText(tr(lang, "setup_blue_preview"))
        self.lbl_red_preview.setText(tr(lang, "setup_red_preview"))
        self.btn_open_fleet_library.setText(tr(lang, "fleet_lib_open"))
        self.hint.setText(tr(lang, "setup_hint"))
        self.backend.setText(f"{tr(lang, 'setup_backend')}: {get_fit_backend_status()}")
        self.btn_add_blue.setText(tr(lang, "setup_add_blue"))
        self.btn_add_red.setText(tr(lang, "setup_add_red"))
        self.btn_delete.setText(tr(lang, "setup_delete"))
        self.btn_validate.setText(tr(lang, "setup_validate"))
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setText(tr(lang, "setup_start"))
        self.btns.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(lang, "setup_cancel"))
        self.fit_editor.setPlaceholderText(tr(lang, "setup_fit_placeholder"))
        self.model.notify_headers_changed()
        self._refresh_previews()

        self.setup_tabs.setTabText(0, tr(lang, "setup_tab_preview"))
        self.setup_tabs.setTabText(1, tr(lang, "setup_tab_settings"))
        self.lbl_cfg_tick_rate.setText(tr(lang, "setup_cfg_tick_rate"))
        self.lbl_cfg_substeps.setText(tr(lang, "setup_cfg_physics_substeps"))
        self.lbl_cfg_radius.setText(tr(lang, "setup_cfg_battlefield_radius"))
        self.lbl_cfg_lockstep.setText(tr(lang, "setup_cfg_lockstep"))
        self.lbl_cfg_detailed_log.setText(tr(lang, "setup_cfg_detailed_logging"))
        self.lbl_cfg_hotspot_log.setText(tr(lang, "setup_cfg_hotspot_logging"))
        self.lbl_cfg_log_file.setText(tr(lang, "setup_cfg_log_file"))
        self.lbl_cfg_hotspot_log_file.setText(tr(lang, "setup_cfg_hotspot_log_file"))
        self.lbl_cfg_log_merge_window.setText(tr(lang, "setup_cfg_log_merge_window"))
        if self._network_mode == "client":
            self.lbl_engine_hint.setText(tr(lang, "setup_cfg_host_authority"))
        else:
            self.lbl_engine_hint.setText(tr(lang, "setup_cfg_local_persist"))

    def _set_engine_settings_enabled(self, enabled: bool) -> None:
        self.spin_cfg_tick_rate.setEnabled(enabled)
        self.spin_cfg_substeps.setEnabled(enabled)
        self.spin_cfg_radius.setEnabled(enabled)
        self.chk_cfg_lockstep.setEnabled(enabled)
        self.chk_cfg_detailed_log.setEnabled(enabled)
        self.chk_cfg_hotspot_log.setEnabled(enabled)
        self.edit_cfg_log_file.setEnabled(enabled)
        self.edit_cfg_hotspot_log_file.setEnabled(enabled)
        self.spin_cfg_log_merge_window.setEnabled(enabled)

    def _load_engine_preferences_into_controls(self) -> None:
        defaults = EngineConfig()
        self._engine_pref_loading = True
        try:
            self.spin_cfg_tick_rate.setValue(max(1, int(self._pref.engine_tick_rate)))
            self.spin_cfg_substeps.setValue(max(1, int(self._pref.engine_physics_substeps)))
            self.spin_cfg_radius.setValue(max(1_000.0, float(self._pref.engine_battlefield_radius)))
            self.chk_cfg_lockstep.setChecked(bool(self._pref.engine_lockstep))
            self.chk_cfg_detailed_log.setChecked(bool(self._pref.engine_detailed_logging))
            self.chk_cfg_hotspot_log.setChecked(bool(self._pref.engine_hotspot_logging))
            log_path = str(self._pref.engine_detail_log_file or "").strip() or defaults.detail_log_file
            hotspot_log_path = str(self._pref.engine_hotspot_log_file or "").strip() or defaults.hotspot_log_file
            self.edit_cfg_log_file.setText(log_path)
            self.edit_cfg_hotspot_log_file.setText(hotspot_log_path)
            self.spin_cfg_log_merge_window.setValue(max(0.1, float(self._pref.engine_log_merge_window_sec)))
        finally:
            self._engine_pref_loading = False

    def _on_engine_pref_changed(self, *_args) -> None:
        if self._engine_pref_loading:
            return
        defaults = EngineConfig()
        self._pref.engine_tick_rate = max(1, int(self.spin_cfg_tick_rate.value()))
        self._pref.engine_physics_substeps = max(1, int(self.spin_cfg_substeps.value()))
        self._pref.engine_battlefield_radius = max(1_000.0, float(self.spin_cfg_radius.value()))
        self._pref.engine_lockstep = bool(self.chk_cfg_lockstep.isChecked())
        self._pref.engine_detailed_logging = bool(self.chk_cfg_detailed_log.isChecked())
        self._pref.engine_hotspot_logging = bool(self.chk_cfg_hotspot_log.isChecked())
        self._pref.engine_detail_log_file = self.edit_cfg_log_file.text().strip() or defaults.detail_log_file
        self._pref.engine_hotspot_log_file = self.edit_cfg_hotspot_log_file.text().strip() or defaults.hotspot_log_file
        self._pref.engine_log_merge_window_sec = max(0.1, float(self.spin_cfg_log_merge_window.value()))
        self._store.save(self._pref)

    def to_engine_config(self) -> EngineConfig:
        defaults = EngineConfig()
        return EngineConfig(
            tick_rate=max(1, int(self.spin_cfg_tick_rate.value())),
            physics_substeps=max(1, int(self.spin_cfg_substeps.value())),
            lockstep=bool(self.chk_cfg_lockstep.isChecked()),
            battlefield_radius=max(1_000.0, float(self.spin_cfg_radius.value())),
            detailed_logging=bool(self.chk_cfg_detailed_log.isChecked()),
            hotspot_logging=bool(self.chk_cfg_hotspot_log.isChecked()),
            detail_log_file=self.edit_cfg_log_file.text().strip() or defaults.detail_log_file,
            hotspot_log_file=self.edit_cfg_hotspot_log_file.text().strip() or defaults.hotspot_log_file,
            log_merge_window_sec=max(0.1, float(self.spin_cfg_log_merge_window.value())),
        )

    def _default_fit_text(self, ship: str = "Ferox", fit_name: str = "Manual") -> str:
        return f"[{ship}, {fit_name}]\nMagnetic Field Stabilizer II\nTracking Enhancer II\n"

    def _current_row(self) -> int:
        idx = self.table.currentIndex()
        return idx.row() if idx.isValid() else -1

    def _load_row_fit(self, row: int) -> None:
        if not (0 <= row < len(self.model.rows)):
            self.fit_editor.blockSignals(True)
            self.fit_editor.setPlainText("")
            self.fit_editor.blockSignals(False)
            return
        self.fit_editor.blockSignals(True)
        self.fit_editor.setPlainText(self.model.rows[row].fit_text)
        self.fit_editor.blockSignals(False)

    def _on_row_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        del previous
        if current.isValid():
            self._load_row_fit(current.row())

    def _on_fit_changed(self) -> None:
        row = self._current_row()
        if not (0 <= row < len(self.model.rows)):
            return
        text = self.fit_editor.toPlainText().strip()
        self.model.rows[row].fit_text = text
        fit_name = ""
        try:
            parsed = self._parser.parse(text)
            self._factory.build(parsed)
            fit_name = f"{parsed.ship_name} / {parsed.fit_name}"
        except Exception:
            pass
        self.model.update_fit_meta(row, fit_name)

    def _add_blue(self) -> None:
        self.model.add_row(
            SetupRow(
                team=Team.BLUE,
                squad_id="BLUE-ALPHA",
                quality=QualityLevel.REGULAR,
                quantity=1,
                fit_text=self._default_fit_text("Ferox", "New Blue"),
            )
        )
        self.table.selectRow(self.model.rowCount() - 1)

    def _add_red(self) -> None:
        self.model.add_row(
            SetupRow(
                team=Team.RED,
                squad_id="RED-ALPHA",
                quality=QualityLevel.REGULAR,
                quantity=1,
                fit_text=self._default_fit_text("Ferox", "New Red"),
            )
        )
        self.table.selectRow(self.model.rowCount() - 1)

    def _delete_current(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        self.model.remove_row(row)
        if self.model.rowCount() > 0:
            self.table.selectRow(max(0, row - 1))
        else:
            self._load_row_fit(-1)

    def _validate_all(self) -> bool:
        lang = self._lang
        if not self.blue_fleet_combo.currentText().strip() or not self.red_fleet_combo.currentText().strip():
            QMessageBox.warning(self, tr(lang, "setup_validate_fail_title"), tr(lang, "setup_validate_need_fleets"))
            return False
        if self.model.rowCount() == 0:
            QMessageBox.warning(self, tr(lang, "setup_validate_fail_title"), tr(lang, "setup_validate_need_ship"))
            return False
        for idx, row in enumerate(self.model.rows, start=1):
            try:
                parsed = self._parser.parse(row.fit_text)
                self._factory.build(parsed)
                self.model.update_fit_meta(idx - 1, f"{parsed.ship_name} / {parsed.fit_name}")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    tr(lang, "setup_validate_fail_title"),
                    tr(lang, "setup_validate_row_invalid", row=idx, error=_localize_fit_error(lang, e)),
                )
                return False
        QMessageBox.information(self, tr(lang, "setup_validate_ok_title"), tr(lang, "setup_validate_all_ok"))
        return True

    def _on_accept(self) -> None:
        if self._validate_all():
            self.accept()

    def to_manual_setup(self) -> list[ManualShipSetup]:
        rows: list[ManualShipSetup] = []
        for row in self.model.rows:
            quantity = max(1, int(row.quantity))
            for i in range(quantity):
                rows.append(
                    ManualShipSetup(
                        team=row.team,
                        squad_id=row.squad_id,
                        quality=row.quality,
                        position=Vector2(0.0, 0.0),
                        fit_text=row.fit_text,
                        is_leader=bool(row.is_leader) and quantity == 1 and i == 0,
                    )
                )
        group_indices: dict[tuple[str, str], list[int]] = {}
        for idx, row in enumerate(rows):
            key = (row.team.value, row.squad_id)
            group_indices.setdefault(key, []).append(idx)
        for indices in group_indices.values():
            flagged = [idx for idx in indices if rows[idx].is_leader]
            chosen = flagged[0] if flagged else indices[0]
            for idx in indices:
                rows[idx].is_leader = (idx == chosen)
        return rows


class PreferencesStore:
    CURRENT_VERSION = 6

    def __init__(self) -> None:
        self.path = Path.home() / ".eve_sim_gui_config.json"

    def _migrate_data(self, data: dict[str, object]) -> dict[str, object]:
        migrated = dict(data)
        raw_version = migrated.get("config_version", 1)
        version = int(raw_version) if isinstance(raw_version, (int, float, str)) else 1
        if version < 2:
            legacy_team = str(migrated.get("filter_team", "ALL") or "ALL").upper()
            legacy_enemy_only = bool(migrated.get("filter_enemy_only", False))
            if legacy_enemy_only:
                migrated["filter_team"] = "ENEMY"
            elif legacy_team in ("FRIENDLY", "ENEMY", "ALL", "BLUE", "RED"):
                migrated["filter_team"] = legacy_team
            else:
                migrated["filter_team"] = "ALL"
        migrated["config_version"] = self.CURRENT_VERSION
        return migrated

    @staticmethod
    def _read_str(data: dict[str, object], key: str, default: str) -> str:
        value = data.get(key, default)
        return str(value).strip() if value is not None else default

    @staticmethod
    def _read_float_or_none(data: dict[str, object], key: str, default: float | None) -> float | None:
        value = data.get(key, default)
        if value is None:
            return None
        if not isinstance(value, (int, float, str)):
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _read_float(data: dict[str, object], key: str, default: float, min_value: float) -> float:
        value = data.get(key, default)
        if not isinstance(value, (int, float, str)):
            return default
        try:
            return max(min_value, float(value))
        except Exception:
            return default

    @staticmethod
    def _read_int(data: dict[str, object], key: str, default: int, min_value: int) -> int:
        value = data.get(key, default)
        if not isinstance(value, (int, float, str)):
            return default
        try:
            return max(min_value, int(float(value)))
        except Exception:
            return default

    @staticmethod
    def _read_bool(data: dict[str, object], key: str, default: bool) -> bool:
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off", ""):
                return False
        return default

    @staticmethod
    def _read_filter_team(data: dict[str, object], default: Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"]) -> Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"]:
        value = str(data.get("filter_team", default) or default).upper()
        if value in ("ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"):
            return cast(Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"], value)
        return default

    def load(self) -> UiPreferences:
        try:
            if not self.path.exists():
                return UiPreferences()
            data = json.loads(self.path.read_text(encoding="utf-8"))
            migrated = self._migrate_data(data if isinstance(data, dict) else {})
            defaults = UiPreferences()
            return UiPreferences(
                config_version=self.CURRENT_VERSION,
                selected_squad=self._read_str(migrated, "selected_squad", defaults.selected_squad),
                filter_team=self._read_filter_team(migrated, defaults.filter_team),
                filter_role=self._read_str(migrated, "filter_role", defaults.filter_role),
                filter_alive=self._read_str(migrated, "filter_alive", defaults.filter_alive),
                filter_squad=self._read_str(migrated, "filter_squad", defaults.filter_squad),
                sort_key=self._read_str(migrated, "sort_key", defaults.sort_key),
                sort_order=self._read_str(migrated, "sort_order", defaults.sort_order),
                zoom=self._read_float_or_none(migrated, "zoom", defaults.zoom),
                language=self._read_str(migrated, "language", defaults.language),
                engine_tick_rate=self._read_int(migrated, "engine_tick_rate", defaults.engine_tick_rate, 1),
                engine_physics_substeps=self._read_int(migrated, "engine_physics_substeps", defaults.engine_physics_substeps, 1),
                engine_lockstep=self._read_bool(migrated, "engine_lockstep", defaults.engine_lockstep),
                engine_battlefield_radius=self._read_float(
                    migrated,
                    "engine_battlefield_radius",
                    defaults.engine_battlefield_radius,
                    1_000.0,
                ),
                engine_detailed_logging=self._read_bool(
                    migrated,
                    "engine_detailed_logging",
                    defaults.engine_detailed_logging,
                ),
                engine_hotspot_logging=self._read_bool(
                    migrated,
                    "engine_hotspot_logging",
                    defaults.engine_hotspot_logging,
                ),
                engine_detail_log_file=self._read_str(
                    migrated,
                    "engine_detail_log_file",
                    defaults.engine_detail_log_file,
                ),
                engine_hotspot_log_file=self._read_str(
                    migrated,
                    "engine_hotspot_log_file",
                    defaults.engine_hotspot_log_file,
                ),
                engine_log_merge_window_sec=self._read_float(
                    migrated,
                    "engine_log_merge_window_sec",
                    defaults.engine_log_merge_window_sec,
                    0.1,
                ),
            )
        except Exception:
            return UiPreferences()

    def save(self, prefs: UiPreferences) -> None:
        try:
            prefs.config_version = self.CURRENT_VERSION
            self.path.write_text(json.dumps(asdict(prefs), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


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


class OverviewTableModel(QAbstractTableModel):
    HEADER_KEYS = ["overview_col_name", "overview_col_type", "overview_col_distance", "overview_col_team"]

    def __init__(
        self,
        selected_squad_getter: Callable[[], str],
        selected_target_getter: Callable[[], str | None],
        language_getter: Callable[[], str],
        controlled_team_getter: Callable[[], Team],
    ) -> None:
        super().__init__()
        self._rows: list[dict] = []
        self._selected_squad_getter = selected_squad_getter
        self._selected_target_getter = selected_target_getter
        self._language_getter = language_getter
        self._controlled_team_getter = controlled_team_getter

    def _headers(self) -> list[str]:
        lang = self._language_getter()
        return [tr(lang, key) for key in self.HEADER_KEYS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADER_KEYS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            headers = self._headers()
            if 0 <= section < len(headers):
                return headers[section]
            return None
        return str(section + 1)

    def notify_headers_changed(self) -> None:
        if self.columnCount() > 0:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return row["id"]
            if col == 1:
                return row.get("ship_type_display", row["ship_type"])
            if col == 2:
                return f"{row['dist']:.1f}"
            if col == 3:
                return row["team"]

        if role == Qt.ItemDataRole.BackgroundRole:
            selected_squad = self._selected_squad_getter()
            selected_target = self._selected_target_getter()
            controlled_team = self._controlled_team_getter().value
            if row["team"] == controlled_team and row["squad"] == selected_squad:
                return QColor(28, 45, 68)
            if selected_target and row["id"] == selected_target:
                return QColor(85, 70, 24)
        return None

    def set_rows(self, rows: list[dict]) -> None:
        if rows == self._rows:
            return
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def get_row(self, row_index: int) -> dict | None:
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None

    def notify_visual_state_changed(self) -> None:
        if not self._rows:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._rows) - 1, len(self.HEADER_KEYS) - 1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])


class BlueRosterTableModel(QAbstractTableModel):
    HEADER_KEYS = ["fleet_col_ship", "fleet_col_squad", "fleet_col_role", "fleet_col_alive", "fleet_col_hp"]

    def __init__(self, language_getter: Callable[[], str]) -> None:
        super().__init__()
        self._rows: list[dict] = []
        self._language_getter = language_getter

    def _headers(self) -> list[str]:
        lang = self._language_getter()
        return [tr(lang, key) for key in self.HEADER_KEYS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADER_KEYS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            headers = self._headers()
            if 0 <= section < len(headers):
                return headers[section]
            return None
        return str(section + 1)

    def notify_headers_changed(self) -> None:
        if self.columnCount() > 0:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return row["ship_id"]
            if col == 1:
                return row["squad"]
            if col == 2:
                return row["role"]
            if col == 3:
                return tr(self._language_getter(), "yes_short") if row["alive"] else tr(self._language_getter(), "no_short")
            if col == 4:
                return f"{row['hp']:.1f}"
        return None

    def set_rows(self, rows: list[dict]) -> None:
        if rows == self._rows:
            return
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def get_row(self, row_index: int) -> dict | None:
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None


class OverviewFilterProxyModel(QSortFilterProxyModel):
    def __init__(
        self,
        prefs_getter: Callable[[], UiPreferences],
        controlled_team_getter: Callable[[], Team],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._prefs_getter = prefs_getter
        self._controlled_team_getter = controlled_team_getter
        self.setDynamicSortFilter(True)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None or not isinstance(model, OverviewTableModel):
            return True
        row = model.get_row(source_row)
        if row is None:
            return False

        prefs = self._prefs_getter()
        enemy_team = Team.RED if self._controlled_team_getter() == Team.BLUE else Team.BLUE
        controlled_team = self._controlled_team_getter()
        team_filter = str(prefs.filter_team or "ALL").upper()
        if team_filter == "ENEMY":
            if row["team"] != enemy_team.value:
                return False
        elif team_filter == "FRIENDLY":
            if row["team"] != controlled_team.value:
                return False
        elif team_filter == "BLUE":
            mapped = "FRIENDLY" if controlled_team == Team.BLUE else "ENEMY"
            if mapped == "FRIENDLY" and row["team"] != controlled_team.value:
                return False
            if mapped == "ENEMY" and row["team"] != enemy_team.value:
                return False
        elif team_filter == "RED":
            mapped = "ENEMY" if controlled_team == Team.BLUE else "FRIENDLY"
            if mapped == "FRIENDLY" and row["team"] != controlled_team.value:
                return False
            if mapped == "ENEMY" and row["team"] != enemy_team.value:
                return False
        if prefs.filter_role != "ALL" and row["role"] != prefs.filter_role:
            return False
        if prefs.filter_alive == "ALIVE" and not row["alive"]:
            return False
        if prefs.filter_alive == "DESTROYED" and row["alive"]:
            return False
        squad_filter = prefs.filter_squad.strip().upper()
        if squad_filter and squad_filter not in row["squad"].upper():
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None or not isinstance(model, OverviewTableModel):
            return super().lessThan(left, right)
        lrow = model.get_row(left.row())
        rrow = model.get_row(right.row())
        if lrow is None or rrow is None:
            return super().lessThan(left, right)

        field_by_col = {
            0: "id",
            1: "ship_type",
            2: "dist",
            3: "team",
        }
        field = field_by_col.get(left.column(), "id")
        lv = lrow[field]
        rv = rrow[field]
        if isinstance(lv, str) and isinstance(rv, str):
            return lv.lower() < rv.lower()
        return lv < rv

    def apply_preferences(self) -> None:
        self.invalidateFilter()

    def get_row(self, proxy_row: int) -> dict | None:
        source = self.sourceModel()
        if source is None or not isinstance(source, OverviewTableModel):
            return None
        proxy_idx = self.index(proxy_row, 0)
        if not proxy_idx.isValid():
            return None
        source_idx = self.mapToSource(proxy_idx)
        if not source_idx.isValid():
            return None
        return source.get_row(source_idx.row())


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
        self._parser = EftFitParser()
        self._runtime_engine = RuntimeStatEngine()
        self._cached_fit_text: str = ""
        self._cached_module_specs: list = []
        self._lock_module_specs: dict[str, ParsedModuleSpec] = {}
        self._lock_ammo_draft_by_module: dict[str, str] = {}
        self._stable_profile_cache = None
        self._stable_profile_cache_key: tuple[Any, ...] | None = None
        self.setWindowTitle(f"{tr(self._language_getter(), 'ship_status_title')} - {ship_id}")
        self.resize(520, 480)

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

        self.info = QPlainTextEdit(self)
        self.info.setReadOnly(True)
        layout.addWidget(self.info)

        self.lock_module_combo.currentIndexChanged.connect(self._on_lock_module_changed)
        self.lock_ammo_combo.currentTextChanged.connect(self._on_lock_ammo_changed)
        self.btn_lock_apply.clicked.connect(self._on_lock_apply_clicked)
        self.btn_lock_clear.clicked.connect(self._on_lock_clear_clicked)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(500)
        self.refresh_status()

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
            self._refresh_lock_controls()
            self.refresh_status()
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
            self._refresh_lock_controls()
            self.refresh_status()
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

    def _module_requires_target(self, module: ModuleRuntime) -> bool:
        if self._is_weapon_group(module.group):
            return True
        if self._is_area_effect_group(module.group):
            return False
        return self._module_has_projected_effects(module)

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
        age = max(0.0, now - float(ship.combat.ecm_last_attempt_at or -1e9))
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
        cache_key = (fit_text, id(ship.runtime), runtime_state_key)
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

    def refresh_status(self) -> None:
        lang = self._language_getter()
        self.setWindowTitle(f"{tr(lang, 'ship_status_title')} - {self.ship_id}")
        self._refresh_lock_controls()
        ship = self.engine.world.ships.get(self.ship_id)
        if ship is None:
            self.info.setPlainText(tr(lang, "ship_missing"))
            return

        profile = ship.profile
        speed = ship.nav.velocity.length()
        dph = float(profile.volley)
        dps = float(profile.dps)
        hp_cur = ship.vital.shield + ship.vital.armor + ship.vital.structure
        hp_max = ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max
        hp_pct = 100.0 * hp_cur / max(1.0, hp_max)
        ship_type_name = get_type_display_name(ship.fit.ship_name, language=lang)
        stable_profile = self._stable_profile(ship)

        lines: list[str] = []
        lines.append(f"{tr(lang, 'status_ship')}: {ship.ship_id}")
        lines.append(f"{tr(lang, 'status_type')}: {ship_type_name}")
        lines.append(f"{tr(lang, 'status_team_squad')}: {ship.team.value} / {ship.squad_id}")
        lines.append(f"{tr(lang, 'status_alive')}: {ship.vital.alive}")
        lines.append(f"{tr(lang, 'status_speed')}: {speed:.2f} m/s")
        lines.append(f"{tr(lang, 'status_facing')}: {ship.nav.facing_deg:.2f} deg")
        lines.append(f"{tr(lang, 'status_dph')}: {stable_profile.volley:.2f}")
        lines.append(f"{tr(lang, 'status_dps')}: {stable_profile.dps:.2f}")
        lines.append(f"{tr(lang, 'status_hp')}: {hp_cur:.1f}/{hp_max:.1f} ({hp_pct:.2f}%)")
        lines.append(
            f"{tr(lang, 'status_shield')}: {ship.vital.shield:.1f}/{ship.vital.shield_max:.1f} | "
            f"{tr(lang, 'status_armor')}: {ship.vital.armor:.1f}/{ship.vital.armor_max:.1f} | "
            f"{tr(lang, 'status_structure')}: {ship.vital.structure:.1f}/{ship.vital.structure_max:.1f}"
        )
        lines.append(f"{tr(lang, 'status_cap')}: {ship.vital.cap:.1f}/{stable_profile.max_cap:.1f}")
        lines.append(f"{tr(lang, 'status_target')}: {ship.combat.current_target or tr(lang, 'status_target_none')}")
        now = float(self.engine.world.now)
        lines.append(f"{tr(lang, 'status_ecm_incoming')}: {self._format_ecm_incoming_status(lang, ship, now)}")
        lines.append(
            f"{tr(lang, 'status_res')}: "
            f"S[{self._res_pct(profile.shield_resonance_em):.1f}/{self._res_pct(profile.shield_resonance_thermal):.1f}/{self._res_pct(profile.shield_resonance_kinetic):.1f}/{self._res_pct(profile.shield_resonance_explosive):.1f}] "
            f"A[{self._res_pct(profile.armor_resonance_em):.1f}/{self._res_pct(profile.armor_resonance_thermal):.1f}/{self._res_pct(profile.armor_resonance_kinetic):.1f}/{self._res_pct(profile.armor_resonance_explosive):.1f}] "
            f"H[{self._res_pct(profile.structure_resonance_em):.1f}/{self._res_pct(profile.structure_resonance_thermal):.1f}/{self._res_pct(profile.structure_resonance_kinetic):.1f}/{self._res_pct(profile.structure_resonance_explosive):.1f}]"
        )
        lines.append("")
        runtime_by_slot: dict[int, str] = {}
        projected_by_slot: dict[int, str] = {}
        projected_by_module: dict[str, str] = {}
        runtime_module_by_slot: dict[int, ModuleRuntime] = {}
        reactivation_by_module: dict[str, float] = {}
        if ship.runtime is not None:
            for module in ship.runtime.modules:
                parts = module.module_id.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    runtime_by_slot[int(parts[1])] = module.state.value
                    runtime_module_by_slot[int(parts[1])] = module
            reactivation_by_module = {
                str(mid): float(left)
                for mid, left in ship.combat.module_reactivation_timers.items()
                if float(left) > 0.0
            }
            for module_id, target_id in ship.combat.projected_targets.items():
                projected_by_module[module_id] = target_id
                parts = module_id.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    projected_by_slot[int(parts[1])] = target_id

        fit_text = self._fit_text_getter(self.ship_id) or ""
        module_specs = self._get_module_specs_cached(fit_text)

        lines.append(f"{tr(lang, 'status_modules')}: ")
        lines.append(f"{tr(lang, 'status_modules_fitted')}: {len(module_specs)} | {tr(lang, 'status_modules_runtime')}: {len(runtime_by_slot)}")
        if not module_specs:
            if ship.runtime is None:
                lines.append(f"  {tr(lang, 'status_module_none')}")
            else:
                for module in ship.runtime.modules:
                    effective_state = module.state.value
                    has_projected = self._module_has_projected_effects(module)
                    is_area_effect = self._is_area_effect_group(module.group)
                    target_required = self._module_requires_target(module)
                    if self._is_weapon_group(module.group):
                        current_target_id = ship.combat.current_target
                    elif has_projected and not is_area_effect:
                        current_target_id = projected_by_module.get(module.module_id)
                    else:
                        current_target_id = None
                    if effective_state == "ACTIVE" and target_required and not current_target_id:
                        effective_state = "ONLINE"
                    cooldown_left = reactivation_by_module.get(module.module_id, 0.0)
                    if cooldown_left > 0.0:
                        state_label = tr(lang, "state_REACTIVATING")
                    else:
                        state_label = tr(lang, f"state_{effective_state}")
                    line = f"  - {module.module_id} | {get_type_display_name(module.group, language=lang)} | {tr(lang, 'status_state')}={state_label}"
                    line += self._format_module_charge_status(lang, ship, module)
                    line += self._format_module_time_status(lang, ship, module, effective_state, cooldown_left)
                    if has_projected and not is_area_effect:
                        target_id = projected_by_module.get(module.module_id, tr(lang, "status_target_none"))
                        line += f" | {tr(lang, 'status_target')}={target_id}"
                        if self._is_ecm_group(module.group):
                            line += self._format_ecm_cycle_result_for_module(lang, ship, module.module_id, target_id)
                    elif self._is_weapon_group(module.group):
                        line += f" | {tr(lang, 'status_target')}={ship.combat.current_target or tr(lang, 'status_target_none')}"
                    lines.append(line)
        else:
            for idx, spec in enumerate(module_specs, start=1):
                module_name = get_type_display_name(spec.module_name, language=lang)
                if spec.charge_name:
                    charge_name = get_type_display_name(spec.charge_name, language=lang)
                    module_name = f"{module_name} ({charge_name})"
                if spec.offline:
                    state_key = "OFFLINE"
                else:
                    state_key = runtime_by_slot.get(idx, "UNMODELED")
                runtime_module = runtime_module_by_slot.get(idx)
                if runtime_module is not None:
                    has_projected = self._module_has_projected_effects(runtime_module)
                    is_area_effect = self._is_area_effect_group(runtime_module.group)
                    target_required = self._module_requires_target(runtime_module)
                    if self._is_weapon_group(runtime_module.group):
                        current_target_id = ship.combat.current_target
                    elif has_projected and not is_area_effect:
                        current_target_id = projected_by_slot.get(idx)
                    else:
                        current_target_id = None
                    if state_key == "ACTIVE" and target_required and not current_target_id:
                        state_key = "ONLINE"
                cooldown_left = 0.0
                if runtime_module is not None:
                    cooldown_left = reactivation_by_module.get(runtime_module.module_id, 0.0)
                if cooldown_left > 0.0:
                    state_label = tr(lang, "state_REACTIVATING")
                else:
                    state_label = tr(lang, f"state_{state_key}")
                line = f"  - [{idx:02d}] {module_name} | {tr(lang, 'status_state')}={state_label}"
                if runtime_module is not None:
                    line += self._format_module_charge_status(lang, ship, runtime_module)
                    line += self._format_module_time_status(lang, ship, runtime_module, state_key, cooldown_left)
                    has_projected = self._module_has_projected_effects(runtime_module)
                    if has_projected and not self._is_area_effect_group(runtime_module.group):
                        target_id = projected_by_slot.get(idx, tr(lang, "status_target_none"))
                        line += f" | {tr(lang, 'status_target')}={target_id}"
                        if self._is_ecm_group(runtime_module.group):
                            line += self._format_ecm_cycle_result_for_module(lang, ship, runtime_module.module_id, target_id)
                    elif self._is_weapon_group(runtime_module.group):
                        line += f" | {tr(lang, 'status_target')}={ship.combat.current_target or tr(lang, 'status_target_none')}"
                lines.append(line)

        self.info.setPlainText("\n".join(lines))


class BattleCanvas(QWidget):
    def __init__(
        self,
        engine: SimulationEngine,
        ui_cfg: UiConfig,
        on_issue_move: Callable[[str, Vector2], None],
        on_issue_approach: Callable[[str, str], None],
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
        self._bg_cache = None
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
            lang = self.language_getter()
            menu = QMenu(self)
            if clicked is not None and clicked.vital.alive:
                action_status = QAction(tr(lang, "menu_show_status", ship=clicked.ship_id), self)
                action_status.triggered.connect(lambda: self.on_show_status(clicked.ship_id))
                menu.addAction(action_status)
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
                    prelocked = clicked.ship_id in self.engine.world.squad_prelocked_targets.get(focus_key, set())
                    prelocking = clicked.ship_id in self.engine.world.squad_prelock_timers.get(focus_key, {})
                    if in_prequeue or prelocked or prelocking:
                        action_cancel_prefocus = QAction(
                            tr(lang, "menu_cancel_prefocus", squad=self.selected_squad, ship=clicked.ship_id),
                            self,
                        )
                        action_cancel_prefocus.triggered.connect(lambda: self.on_cancel_prefocus(clicked.ship_id))
                        menu.addAction(action_cancel_prefocus)

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
        if group == "burst jammer":
            return QColor(102, 204, 255, 13), QColor(122, 214, 255, 13)
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
            if module is None or module.state != ModuleState.ACTIVE:
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
                overlay = self._area_cycle_overlays.get(key)
                remaining = max(0.0, overlay.expires_at - now) if overlay is not None else 0.0
                if overlay is not None and cycle_left <= remaining + cycle_restart_margin:
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


class MainWindow(QMainWindow):
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

            prelocked = self.engine.world.squad_prelocked_targets.get(focus_key)
            if prelocked is not None:
                prelocked.discard(target_id)

            timers = self.engine.world.squad_prelock_timers.get(focus_key)
            if timers is not None:
                timers.pop(target_id, None)

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
                prelocked = self.engine.world.squad_prelocked_targets.get(focus_key)
                if prelocked is not None:
                    prelocked.discard(target_id)
                timers = self.engine.world.squad_prelock_timers.get(focus_key)
                if timers is not None:
                    timers.pop(target_id, None)
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

    def _ensure_remote_ship(self, ship_id: str, data: dict) -> ShipEntity:
        existing = self.engine.world.ships.get(ship_id)
        if existing is not None:
            fit_text_existing = str(data.get("fit_text", "") or "").strip()
            if fit_text_existing:
                self._ship_fit_texts[ship_id] = fit_text_existing
            return existing
        team_text = str(data.get("team", "BLUE"))
        team = Team.BLUE if team_text == "BLUE" else Team.RED
        fit_text = str(data.get("fit_text", "") or "").strip()
        if fit_text:
            self._ship_fit_texts[ship_id] = fit_text
        runtime = None
        if fit_text:
            try:
                parsed = self._parser.parse(fit_text)
                runtime_template, parsed_fit = self._factory.build(parsed)
                runtime = deepcopy(runtime_template)
                fit = parsed_fit
                profile = self._factory.build_profile(parsed)
            except Exception:
                fit = FitDescriptor(
                    fit_key=f"remote-{ship_id}",
                    ship_name=str(data.get("ship_name", ship_id)),
                    role=str(data.get("fit_role", "REMOTE")),
                    base_dps=float(data.get("base_dps", 0.0)),
                    volley=float(data.get("fit_volley", 0.0)),
                    optimal_range=float(data.get("fit_optimal_range", 0.0)),
                    falloff=float(data.get("fit_falloff", 0.0)),
                    tracking=float(data.get("fit_tracking", 0.0)),
                )
                profile = ShipProfile(
                    dps=float(data.get("profile_dps", 0.0)),
                    volley=float(data.get("profile_volley", 0.0)),
                    optimal=float(data.get("profile_optimal", 0.0)),
                    falloff=float(data.get("profile_falloff", 0.0)),
                    tracking=float(data.get("profile_tracking", 0.0)),
                    sig_radius=float(data.get("profile_sig_radius", 120.0)),
                    scan_resolution=float(data.get("profile_scan_resolution", 300.0)),
                    max_target_range=float(data.get("profile_max_target_range", 120000.0)),
                    max_locked_targets=int(data.get("profile_max_locked_targets", 0) or 0),
                    scan_strength=float(data.get("profile_scan_strength", 0.0)),
                    ecm_jam_chance=float(data.get("profile_ecm_jam_chance", 0.0)),
                    warp_scramble_status=float(data.get("profile_warp_scramble_status", 0.0)),
                    warp_stability=float(data.get("profile_warp_stability", 0.0)),
                    max_speed=float(data.get("profile_max_speed", 1800.0)),
                    max_cap=float(data.get("profile_max_cap", 4000.0)),
                    cap_recharge_time=float(data.get("profile_cap_recharge_time", 450.0)),
                    shield_hp=float(data.get("profile_shield_hp", 5000.0)),
                    armor_hp=float(data.get("profile_armor_hp", 4000.0)),
                    structure_hp=float(data.get("profile_structure_hp", 4000.0)),
                    rep_amount=float(data.get("profile_rep_amount", 0.0)),
                    rep_cycle=float(data.get("profile_rep_cycle", 5.0)),
                )
        else:
            fit = FitDescriptor(
                fit_key=f"remote-{ship_id}",
                ship_name=str(data.get("ship_name", ship_id)),
                role=str(data.get("fit_role", "REMOTE")),
                base_dps=float(data.get("base_dps", 0.0)),
                volley=float(data.get("fit_volley", 0.0)),
                optimal_range=float(data.get("fit_optimal_range", 0.0)),
                falloff=float(data.get("fit_falloff", 0.0)),
                tracking=float(data.get("fit_tracking", 0.0)),
            )
            profile = ShipProfile(
                dps=float(data.get("profile_dps", 0.0)),
                volley=float(data.get("profile_volley", 0.0)),
                optimal=float(data.get("profile_optimal", 0.0)),
                falloff=float(data.get("profile_falloff", 0.0)),
                tracking=float(data.get("profile_tracking", 0.0)),
                sig_radius=float(data.get("profile_sig_radius", 120.0)),
                scan_resolution=float(data.get("profile_scan_resolution", 300.0)),
                max_target_range=float(data.get("profile_max_target_range", 120000.0)),
                max_locked_targets=int(data.get("profile_max_locked_targets", 0) or 0),
                scan_strength=float(data.get("profile_scan_strength", 0.0)),
                ecm_jam_chance=float(data.get("profile_ecm_jam_chance", 0.0)),
                warp_scramble_status=float(data.get("profile_warp_scramble_status", 0.0)),
                warp_stability=float(data.get("profile_warp_stability", 0.0)),
                max_speed=float(data.get("profile_max_speed", 1800.0)),
                max_cap=float(data.get("profile_max_cap", 4000.0)),
                cap_recharge_time=float(data.get("profile_cap_recharge_time", 450.0)),
                shield_hp=float(data.get("profile_shield_hp", 5000.0)),
                armor_hp=float(data.get("profile_armor_hp", 4000.0)),
                structure_hp=float(data.get("profile_structure_hp", 4000.0)),
                rep_amount=float(data.get("profile_rep_amount", 0.0)),
                rep_cycle=float(data.get("profile_rep_cycle", 5.0)),
            )
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
                ship.combat.ecm_last_attempt_at = float(raw.get("ecm_last_attempt_at", -1e9) or -1e9)
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
                        module.state = ModuleState[state_name]

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
            round(float(raw.get("ecm_last_attempt_at", -1e9) or -1e9), 2),
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


def run_gui() -> None:
    app = QApplication.instance() or QApplication([])

    mode_options = ["Local", "Host LAN", "Join LAN"]
    mode, ok = QInputDialog.getItem(None, "Battle Mode", "Select mode", mode_options, 0, False)
    if not ok:
        return

    network_mode = "local"
    controlled_team = Team.BLUE
    lan_server: HostLanSession | None = None
    lan_client: ClientLanSession | None = None

    if mode == "Host LAN":
        port, ok = QInputDialog.getInt(None, "Host LAN", "Port", 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_server = HostLanSession(host="0.0.0.0", port=int(port))
        try:
            lan_server.start()
        except OSError as exc:
            QMessageBox.critical(None, "Host LAN", f"Failed to open LAN host: {exc}")
            return
        network_mode = "host"
        controlled_team = Team.BLUE
    elif mode == "Join LAN":
        host, ok = QInputDialog.getText(None, "Join LAN", "Host IP", text="127.0.0.1")
        if not ok or not host.strip():
            return
        port, ok = QInputDialog.getInt(None, "Join LAN", "Port", 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_client = ClientLanSession(host=host.strip(), port=int(port))
        if not lan_client.connect(timeout=5.0):
            QMessageBox.critical(None, "Join LAN", "Failed to connect to host")
            lan_client.close()
            return
        network_mode = "client"
        controlled_team = Team.RED

    setup_dialog = FleetSetupDialog(network_mode=network_mode)
    if setup_dialog.exec() != QDialog.DialogCode.Accepted:
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return

    manual_setup = setup_dialog.to_manual_setup()
    if network_mode == "host":
        manual_setup = [row for row in manual_setup if row.team == Team.BLUE]
    elif network_mode == "client":
        manual_setup = [row for row in manual_setup if row.team == Team.RED]
    if not manual_setup:
        QMessageBox.critical(None, "Fleet Setup", "No ships found for current side in this mode")
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return
    pyfa = PyfaBridge()
    world = build_world_from_manual_setup(manual_setup)
    cfg = setup_dialog.to_engine_config()
    engine = SimulationEngine(world=world, config=cfg, combat_system=CombatSystem(pyfa))

    blue_squads = sorted({s.squad_id for s in world.ships.values() if s.team == Team.BLUE})
    red_squads = sorted({s.squad_id for s in world.ships.values() if s.team == Team.RED})
    blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=blue_squads)
    red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=red_squads)
    engine.register_commander(blue_commander)
    engine.register_commander(red_commander)
    for ship_id in world.ships:
        engine.register_ship(ship_id)

    win = MainWindow(
        engine=engine,
        ui_cfg=UiConfig(),
        blue_commander=blue_commander,
        red_commander=red_commander,
        manual_setup=manual_setup,
        network_mode=network_mode,
        controlled_team=controlled_team,
        lan_server=lan_server,
        lan_client=lan_client,
    )
    win.show()
    app.exec()
