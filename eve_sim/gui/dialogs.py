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
                    runtime_by_slot[int(parts[1])] = module.normalized_state().value
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
                    effective_state = module.normalized_state().value
                    has_projected = self._module_has_projected_effects(module)
                    is_area_effect = self._is_area_effect_group(module.group)
                    if self._is_weapon_group(module.group):
                        current_target_id = ship.combat.current_target
                    elif has_projected and not is_area_effect:
                        current_target_id = projected_by_module.get(module.module_id)
                    else:
                        current_target_id = None
                    effective_state = self._display_module_state(module, effective_state, current_target_id)
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
                    if self._is_weapon_group(runtime_module.group):
                        current_target_id = ship.combat.current_target
                    elif has_projected and not is_area_effect:
                        current_target_id = projected_by_slot.get(idx)
                    else:
                        current_target_id = None
                    state_key = self._display_module_state(runtime_module, state_key, current_target_id)
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



