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
from .dialogs import FleetLibraryDialog
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
        for idx, setup_item in enumerate(rows):
            key = (setup_item.team.value, setup_item.squad_id)
            group_indices.setdefault(key, []).append(idx)
        for indices in group_indices.values():
            flagged = [idx for idx in indices if rows[idx].is_leader]
            chosen = flagged[0] if flagged else indices[0]
            for idx in indices:
                rows[idx].is_leader = (idx == chosen)
        return rows



