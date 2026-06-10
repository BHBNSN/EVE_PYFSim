from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QPoint, Qt, QCoreApplication
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..maps import (
    MapBuildingDefinition,
    MapDefinition,
    MapSpawnAnchorDefinition,
    MapSystemDefinition,
    is_stargate_building,
    normalize_stargate_links,
)
from ..math2d import Vector2
from .system_graph_canvas import SystemGraphCanvas


AU_METERS = 149_597_870_700.0


class _MapSystemCanvas(QWidget):
    def __init__(self, owner: "MapEditorDialog", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._owner = owner
        self.setMinimumSize(360, 360)

    def _system_to_screen(self, system: MapSystemDefinition, position: Vector2) -> tuple[int, int]:
        radius = max(1_000.0, float(system.radius_m or 1_000.0))
        usable = max(40.0, float(min(self.width(), self.height()) - 40.0))
        scale = usable / (radius * 2.0)
        cx = self.width() // 2
        cy = self.height() // 2
        return (
            int(cx + float(position.x) * scale),
            int(cy + float(position.y) * scale),
        )

    def _screen_to_system(self, system: MapSystemDefinition, point: QPoint) -> Vector2:
        radius = max(1_000.0, float(system.radius_m or 1_000.0))
        usable = max(40.0, float(min(self.width(), self.height()) - 40.0))
        scale = usable / (radius * 2.0)
        cx = self.width() // 2
        cy = self.height() // 2
        return Vector2((point.x() - cx) / scale, (point.y() - cy) / scale)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        system = self._owner.current_system()
        building = self._owner.current_building()
        if system is None or building is None:
            return
        local = self._screen_to_system(system, event.position().toPoint())
        building.position = local
        self._owner.sync_forms_from_selection()
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(18, 21, 28))
            system = self._owner.current_system()
            if system is None:
                painter.setPen(QColor(210, 210, 210))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, QCoreApplication.translate("eve_sim", "No system selected"))
                return
            cx = self.width() // 2
            cy = self.height() // 2
            radius = max(1_000.0, float(system.radius_m or 1_000.0))
            usable = max(40.0, float(min(self.width(), self.height()) - 40.0))
            scale = usable / (radius * 2.0)
            radius_px = max(10, int(radius * scale))
            painter.setPen(QPen(QColor(92, 110, 140), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(cx - radius_px, cy - radius_px, radius_px * 2, radius_px * 2)

            selected_building_id = self._owner.current_building_id()
            for building in system.buildings:
                x, y = self._system_to_screen(system, building.position)
                circle_px = max(2, int(max(0.0, float(building.radius_m or 0.0)) * scale))
                border = QColor(255, 194, 92) if building.kind.upper() == "STARGATE" else QColor(184, 194, 214)
                fill = QColor(border)
                fill.setAlpha(28)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(x - circle_px, y - circle_px, circle_px * 2, circle_px * 2)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(border, 1))
                painter.drawEllipse(x - circle_px, y - circle_px, circle_px * 2, circle_px * 2)
                painter.setPen(QPen(border if building.building_id != selected_building_id else QColor(255, 230, 90), 2))
                if building.kind.upper() == "STARGATE":
                    painter.drawRect(x - 7, y - 7, 14, 14)
                else:
                    painter.drawEllipse(x - 5, y - 5, 10, 10)
        finally:
            if painter.isActive():
                painter.end()


class MapEditorDialog(QDialog):
    def __init__(self, map_definition: MapDefinition, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._map = deepcopy(map_definition)
        normalize_stargate_links(self._map)
        self._loading = False
        self.setWindowTitle(QCoreApplication.translate("eve_sim", "Map Editor"))
        self.resize(980, 620)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter, 1)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)

        map_form = QFormLayout()
        self.edit_map_id = QLineEdit(self._map.map_id, self)
        self.edit_map_name = QLineEdit(self._map.name, self)
        self.edit_map_desc = QLineEdit(self._map.description, self)
        map_form.addRow(QCoreApplication.translate("eve_sim", "Map ID"), self.edit_map_id)
        map_form.addRow(QCoreApplication.translate("eve_sim", "Map Name"), self.edit_map_name)
        map_form.addRow(QCoreApplication.translate("eve_sim", "Description"), self.edit_map_desc)
        left_layout.addLayout(map_form)

        system_row = QHBoxLayout()
        self.system_combo = QComboBox(self)
        self.btn_add_system = QPushButton(QCoreApplication.translate("eve_sim", "Add System"), self)
        self.btn_remove_system = QPushButton(QCoreApplication.translate("eve_sim", "Remove System"), self)
        system_row.addWidget(QLabel(QCoreApplication.translate("eve_sim", "System"), self))
        system_row.addWidget(self.system_combo, 1)
        system_row.addWidget(self.btn_add_system)
        system_row.addWidget(self.btn_remove_system)
        left_layout.addLayout(system_row)

        system_form = QFormLayout()
        self.edit_system_name = QLineEdit(self)
        self.spin_system_radius = QDoubleSpinBox(self)
        self.spin_system_radius.setRange(0.1, 200.0)
        self.spin_system_radius.setDecimals(2)
        self.spin_system_radius.setSingleStep(1.0)
        system_form.addRow(QCoreApplication.translate("eve_sim", "System Name"), self.edit_system_name)
        system_form.addRow(QCoreApplication.translate("eve_sim", "System Radius (AU)"), self.spin_system_radius)
        left_layout.addLayout(system_form)

        building_row = QHBoxLayout()
        self.building_combo = QComboBox(self)
        self.btn_add_structure = QPushButton(QCoreApplication.translate("eve_sim", "Add Building"), self)
        self.btn_remove_structure = QPushButton(QCoreApplication.translate("eve_sim", "Remove Building"), self)
        building_row.addWidget(QLabel(QCoreApplication.translate("eve_sim", "Building"), self))
        building_row.addWidget(self.building_combo, 1)
        building_row.addWidget(self.btn_add_structure)
        building_row.addWidget(self.btn_remove_structure)
        left_layout.addLayout(building_row)

        building_form = QFormLayout()
        self.edit_building_name = QLineEdit(self)
        self.combo_building_kind = QComboBox(self)
        self.combo_building_kind.addItem(QCoreApplication.translate("eve_sim", "Structure"), "STRUCTURE")
        self.combo_building_kind.addItem(QCoreApplication.translate("eve_sim", "Stargate"), "STARGATE")
        self.spin_building_x = QDoubleSpinBox(self)
        self.spin_building_y = QDoubleSpinBox(self)
        self.spin_building_radius = QDoubleSpinBox(self)
        self.spin_building_interaction = QDoubleSpinBox(self)
        self.combo_building_link = QComboBox(self)
        for spin in (self.spin_building_x, self.spin_building_y):
            spin.setRange(-200.0, 200.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
        for spin in (self.spin_building_radius, self.spin_building_interaction):
            spin.setRange(0.0, 5_000_000.0)
            spin.setDecimals(1)
            spin.setSingleStep(1_000.0)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Building Name"), self.edit_building_name)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Kind"), self.combo_building_kind)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Local X (AU)"), self.spin_building_x)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Local Y (AU)"), self.spin_building_y)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Radius (m)"), self.spin_building_radius)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Interaction Range (m)"), self.spin_building_interaction)
        building_form.addRow(QCoreApplication.translate("eve_sim", "Gate Link"), self.combo_building_link)
        left_layout.addLayout(building_form)
        left_layout.addStretch(1)

        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.system_graph_canvas = SystemGraphCanvas(
            lambda: self._map,
            lambda: str(getattr(self.current_system(), "system_id", "") or ""),
            self._select_system_by_id,
            right,
        )
        self.system_graph_canvas.setMinimumHeight(180)
        right_layout.addWidget(QLabel(QCoreApplication.translate("eve_sim", "System Map"), right))
        right_layout.addWidget(self.system_graph_canvas, 1)
        self.canvas = _MapSystemCanvas(self, right)
        right_layout.addWidget(QLabel(QCoreApplication.translate("eve_sim", "Click canvas to place selected building"), right))
        right_layout.addWidget(self.canvas, 2)
        splitter.addWidget(right)
        splitter.setSizes([430, 550])

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.system_combo.currentIndexChanged.connect(self.sync_forms_from_selection)
        self.building_combo.currentIndexChanged.connect(self.sync_forms_from_selection)
        self.btn_add_system.clicked.connect(self._add_system)
        self.btn_remove_system.clicked.connect(self._remove_system)
        self.btn_add_structure.clicked.connect(self._add_building)
        self.btn_remove_structure.clicked.connect(self._remove_building)
        self.edit_system_name.textChanged.connect(self._apply_form_changes)
        self.spin_system_radius.valueChanged.connect(self._apply_form_changes)
        self.edit_building_name.textChanged.connect(self._apply_form_changes)
        self.combo_building_kind.currentIndexChanged.connect(self._apply_form_changes)
        self.spin_building_x.valueChanged.connect(self._apply_form_changes)
        self.spin_building_y.valueChanged.connect(self._apply_form_changes)
        self.spin_building_radius.valueChanged.connect(self._apply_form_changes)
        self.spin_building_interaction.valueChanged.connect(self._apply_form_changes)
        self.combo_building_link.currentIndexChanged.connect(self._apply_form_changes)

        self._refresh_systems()
        self.sync_forms_from_selection()

    def current_system(self) -> MapSystemDefinition | None:
        system_id = str(self.system_combo.currentData() or "").strip()
        for system in self._map.systems:
            if system.system_id == system_id:
                return system
        return self._map.systems[0] if self._map.systems else None

    def current_building(self) -> MapBuildingDefinition | None:
        system = self.current_system()
        if system is None:
            return None
        building_id = str(self.building_combo.currentData() or "").strip()
        for building in system.buildings:
            if building.building_id == building_id:
                return building
        return system.buildings[0] if system.buildings else None

    def current_building_id(self) -> str:
        building = self.current_building()
        return "" if building is None else building.building_id

    def map_definition(self) -> MapDefinition:
        normalized = deepcopy(self._map)
        normalize_stargate_links(normalized)
        return normalized

    def _select_system_by_id(self, system_id: str) -> None:
        idx = self.system_combo.findData(str(system_id or "").strip())
        if idx < 0:
            return
        self.system_combo.setCurrentIndex(idx)
        self.sync_forms_from_selection()

    def _building_system_id(self, building_id: str) -> str:
        target = str(building_id or "").strip()
        system = self._building_system(target)
        return "" if system is None else str(system.system_id or "")

    def _building_system(self, building_id: str) -> MapSystemDefinition | None:
        target = str(building_id or "").strip()
        for system in self._map.systems:
            for building in system.buildings:
                if building.building_id == target:
                    return system
        return None

    def _all_stargates(self) -> dict[str, MapBuildingDefinition]:
        gates: dict[str, MapBuildingDefinition] = {}
        for system in self._map.systems:
            for building in system.buildings:
                if is_stargate_building(building):
                    gates[building.building_id] = building
        return gates

    def _building_option_label(self, building_id: str) -> str:
        target = str(building_id or "").strip()
        for system in self._map.systems:
            for building in system.buildings:
                if building.building_id == target:
                    return f"{system.name} / {building.name or building.building_id}"
        return target

    def _stargate_link_option_label(self, system: MapSystemDefinition, building: MapBuildingDefinition, selected: MapBuildingDefinition) -> tuple[bool, str]:
        label = f"{system.name} / {building.name or building.building_id}"
        linked_id = str(building.linked_building_id or "").strip()
        linked_to_other = bool(linked_id and linked_id != selected.building_id)
        if linked_id:
            linked_label = self._building_option_label(linked_id)
            label = f"{label} - {QCoreApplication.translate('eve_sim', 'Already linked to {target}').format(target=linked_label)}"
        return linked_to_other, label

    def _clear_link_pair(self, building_id: str) -> None:
        target = str(building_id or "").strip()
        if not target:
            return
        for gate in self._all_stargates().values():
            if gate.building_id == target or gate.linked_building_id == target:
                gate.linked_building_id = None

    def _set_stargate_link(self, building: MapBuildingDefinition, target_id: str | None) -> None:
        source_id = str(building.building_id or "").strip()
        target_key = str(target_id or "").strip()
        self._clear_link_pair(source_id)
        if not is_stargate_building(building) or not target_key:
            building.linked_building_id = None
            return
        gates = self._all_stargates()
        target = gates.get(target_key)
        if target is None or target.building_id == source_id:
            building.linked_building_id = None
            return
        if self._building_system_id(source_id) == self._building_system_id(target.building_id):
            building.linked_building_id = None
            return
        self._clear_link_pair(target.building_id)
        building.linked_building_id = target.building_id
        target.linked_building_id = source_id
        source_system = self._building_system(source_id)
        target_system = self._building_system(target.building_id)
        if source_system is not None and target_system is not None:
            building.name = f"To {target_system.name or target_system.system_id}"
            target.name = f"To {source_system.name or source_system.system_id}"

    def _sync_linked_stargate_names(self) -> None:
        gates = self._all_stargates()
        for source_id, gate in gates.items():
            target_id = str(gate.linked_building_id or "").strip()
            target = gates.get(target_id)
            if target is None:
                continue
            source_system = self._building_system(source_id)
            target_system = self._building_system(target_id)
            if source_system is None or target_system is None:
                continue
            if source_system.system_id == target_system.system_id:
                continue
            gate.name = f"To {target_system.name or target_system.system_id}"

    def _sync_current_building_name_field(self) -> None:
        building = self.current_building()
        if building is None:
            return
        if self.edit_building_name.text() == building.name:
            return
        self.edit_building_name.blockSignals(True)
        self.edit_building_name.setText(building.name)
        self.edit_building_name.blockSignals(False)

    def _refresh_systems(self) -> None:
        normalize_stargate_links(self._map)
        current = str(self.system_combo.currentData() or "")
        self.system_combo.blockSignals(True)
        self.system_combo.clear()
        for system in self._map.systems:
            self.system_combo.addItem(system.name, system.system_id)
        idx = self.system_combo.findData(current)
        self.system_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.system_combo.blockSignals(False)
        self._refresh_buildings()
        self.system_graph_canvas.sync_layout()

    def _refresh_buildings(self) -> None:
        current = str(self.building_combo.currentData() or "")
        system = self.current_system()
        self.building_combo.blockSignals(True)
        self.building_combo.clear()
        if system is not None:
            for building in system.buildings:
                self.building_combo.addItem(building.name or building.building_id, building.building_id)
        idx = self.building_combo.findData(current)
        self.building_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.building_combo.blockSignals(False)
        self._refresh_building_link_combo()

    def _refresh_building_link_combo(self) -> None:
        selected = self.current_building()
        current = str(self.combo_building_link.currentData() or "")
        self.combo_building_link.blockSignals(True)
        self.combo_building_link.clear()
        self.combo_building_link.addItem(QCoreApplication.translate("eve_sim", "None"), "")
        selected_is_gate = selected is not None and is_stargate_building(selected)
        selected_system_id = self._building_system_id(selected.building_id) if selected is not None else ""
        if selected_is_gate:
            options: list[tuple[bool, str, str, str, str]] = []
            for system in self._map.systems:
                system_id = str(system.system_id or "")
                if system_id == selected_system_id:
                    continue
                for building in system.buildings:
                    if not is_stargate_building(building):
                        continue
                    linked_to_other, label = self._stargate_link_option_label(system, building, selected)
                    options.append((linked_to_other, system.name, building.name or building.building_id, building.building_id, label))
            for _linked_to_other, _system_name, _building_name, building_id, label in sorted(options):
                self.combo_building_link.addItem(label, building_id)
        link_id = getattr(selected, "linked_building_id", "") or current or ""
        idx = self.combo_building_link.findData(link_id)
        self.combo_building_link.setCurrentIndex(0 if idx < 0 else idx)
        self.combo_building_link.setEnabled(selected_is_gate)
        self.combo_building_link.blockSignals(False)

    def _apply_form_changes(self, *_args) -> None:
        if self._loading:
            return
        system = self.current_system()
        if system is not None:
            system.name = self.edit_system_name.text().strip() or system.system_id
            system.origin = Vector2(0.0, 0.0)
            system.radius_m = max(0.1 * AU_METERS, float(self.spin_system_radius.value()) * AU_METERS)
        building = self.current_building()
        if building is not None:
            building.name = self.edit_building_name.text().strip() or building.building_id
            building.kind = str(self.combo_building_kind.currentData() or "STRUCTURE")
            building.position = Vector2(float(self.spin_building_x.value()) * AU_METERS, float(self.spin_building_y.value()) * AU_METERS)
            building.radius_m = max(0.0, float(self.spin_building_radius.value()))
            building.interaction_range_m = max(0.0, float(self.spin_building_interaction.value()))
            building.icon_key = "stargate" if building.kind.upper() == "STARGATE" else "structure"
            self._set_stargate_link(building, str(self.combo_building_link.currentData() or "").strip())
        self._map.map_id = self.edit_map_id.text().strip() or self._map.map_id
        self._map.name = self.edit_map_name.text().strip() or self._map.name
        self._map.description = self.edit_map_desc.text().strip()
        normalize_stargate_links(self._map)
        self._sync_linked_stargate_names()
        self._sync_current_building_name_field()
        self._refresh_systems()
        self.canvas.update()
        self.system_graph_canvas.sync_layout()

    def sync_forms_from_selection(self) -> None:
        self._loading = True
        try:
            self.edit_map_id.setText(self._map.map_id)
            self.edit_map_name.setText(self._map.name)
            self.edit_map_desc.setText(self._map.description)
            system = self.current_system()
            if system is not None:
                self.edit_system_name.setText(system.name)
                self.spin_system_radius.setValue(float(system.radius_m) / AU_METERS)
            building = self.current_building()
            self._refresh_buildings()
            if building is not None:
                self.edit_building_name.setText(building.name or building.building_id)
                kind_idx = self.combo_building_kind.findData(building.kind)
                self.combo_building_kind.setCurrentIndex(0 if kind_idx < 0 else kind_idx)
                self.spin_building_x.setValue(float(building.position.x) / AU_METERS)
                self.spin_building_y.setValue(float(building.position.y) / AU_METERS)
                self.spin_building_radius.setValue(float(building.radius_m))
                self.spin_building_interaction.setValue(float(building.interaction_range_m))
                link_idx = self.combo_building_link.findData(building.linked_building_id or "")
                self.combo_building_link.setCurrentIndex(0 if link_idx < 0 else link_idx)
        finally:
            self._loading = False
            self.canvas.update()
            self.system_graph_canvas.sync_layout()

    def _add_system(self) -> None:
        next_index = len(self._map.systems) + 1
        system_id = f"system_{next_index}"
        system = MapSystemDefinition(
            system_id=system_id,
            name=QCoreApplication.translate("eve_sim", "System {index}").format(index=next_index),
            origin=Vector2(0.0, 0.0),
            radius_m=30.0 * AU_METERS,
            buildings=[],
            spawn_anchors=[
                MapSpawnAnchorDefinition(
                    anchor_id=f"{system_id}_blue",
                    system_id=system_id,
                    position=Vector2(-200_000.0, 0.0),
                    radius_m=15_000.0,
                    team="BLUE",
                    label=QCoreApplication.translate("eve_sim", "Blue Spawn"),
                ),
                MapSpawnAnchorDefinition(
                    anchor_id=f"{system_id}_red",
                    system_id=system_id,
                    position=Vector2(200_000.0, 0.0),
                    radius_m=15_000.0,
                    team="RED",
                    label=QCoreApplication.translate("eve_sim", "Red Spawn"),
                ),
            ],
        )
        self._map.systems.append(system)
        self._refresh_systems()
        idx = self.system_combo.findData(system.system_id)
        self.system_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.sync_forms_from_selection()

    def _remove_system(self) -> None:
        system = self.current_system()
        if system is None:
            return
        self._map.systems = [candidate for candidate in self._map.systems if candidate.system_id != system.system_id]
        normalize_stargate_links(self._map)
        self._refresh_systems()
        self.sync_forms_from_selection()

    def _add_building(self) -> None:
        system = self.current_system()
        if system is None:
            return
        next_index = len(system.buildings) + 1
        building = MapBuildingDefinition(
            building_id=f"{system.system_id}_building_{next_index}",
            system_id=system.system_id,
            position=Vector2(0.0, 0.0),
            radius_m=10_000.0,
            kind="STRUCTURE",
            name=QCoreApplication.translate("eve_sim", "Building {index}").format(index=next_index),
            interaction_range_m=0.0,
            icon_key="structure",
        )
        system.buildings.append(building)
        self._refresh_buildings()
        idx = self.building_combo.findData(building.building_id)
        self.building_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.sync_forms_from_selection()

    def _remove_building(self) -> None:
        system = self.current_system()
        building = self.current_building()
        if system is None or building is None:
            return
        self._clear_link_pair(building.building_id)
        system.buildings = [candidate for candidate in system.buildings if candidate.building_id != building.building_id]
        normalize_stargate_links(self._map)
        self._refresh_buildings()
        self.sync_forms_from_selection()

    def _on_accept(self) -> None:
        self._apply_form_changes()
        normalize_stargate_links(self._map)
        self._sync_linked_stargate_names()
        if not self.edit_map_id.text().strip():
            QMessageBox.warning(self, QCoreApplication.translate("eve_sim", "Map Editor"), QCoreApplication.translate("eve_sim", "Map ID cannot be empty"))
            return
        if not self._map.systems:
            QMessageBox.warning(self, QCoreApplication.translate("eve_sim", "Map Editor"), QCoreApplication.translate("eve_sim", "At least one system is required"))
            return
        self.accept()


__all__ = ["MapEditorDialog"]
