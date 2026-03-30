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



from .models import SetupRow, UiPreferences
class FleetSetupTableModel(QAbstractTableModel):
    HEADERS = [
        "Team",
        "Squad",
        "Quality",
        "Count",
        "Leader",
        "Fit Name",
    ]

    def __init__(self, rows: list[SetupRow], language_getter: Callable[[], str]) -> None:
        super().__init__()
        self.rows = rows
        self._language_getter = language_getter

    def _headers(self) -> list[str]:
        return [QCoreApplication.translate("eve_sim", header) for header in self.HEADERS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADERS)

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
                return QCoreApplication.translate("eve_sim", 'Y') if row.is_leader else QCoreApplication.translate("eve_sim", 'N')
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
                row.is_leader = (row.quantity == 1) and (text in ("1", "Y", "YES", "TRUE", "T", "闃熼暱", "LEADER"))
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
            combo.addItem(QCoreApplication.translate("eve_sim", 'Y'), True)
            combo.addItem(QCoreApplication.translate("eve_sim", 'N'), False)
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
                truthy = str(value).strip().upper() in ("1", "Y", "YES", "TRUE", "T")
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



class OverviewTableModel(QAbstractTableModel):
    HEADERS = ["Name", "Ship Type", "Distance (km)", "Team"]

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
        return [QCoreApplication.translate("eve_sim", header) for header in self.HEADERS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADERS)

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
        bottom_right = self.index(len(self._rows) - 1, len(self.HEADERS) - 1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])



class BlueRosterTableModel(QAbstractTableModel):
    HEADERS = ["Ship ID", "Squad", "Role", "Alive", "HP%"]

    def __init__(self, language_getter: Callable[[], str]) -> None:
        super().__init__()
        self._rows: list[dict] = []
        self._language_getter = language_getter

    def _headers(self) -> list[str]:
        return [QCoreApplication.translate("eve_sim", header) for header in self.HEADERS]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        del parent
        return len(self.HEADERS)

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
                return QCoreApplication.translate("eve_sim", 'Y') if row["alive"] else QCoreApplication.translate("eve_sim", 'N')
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
        begin_filter_change = getattr(self, "beginFilterChange", None)
        end_filter_change = getattr(self, "endFilterChange", None)
        if callable(begin_filter_change) and callable(end_filter_change):
            begin_filter_change()
            end_filter_change()
            return
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




