from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .adapters.runtime_view import WorldViewSource
from .system_graph_canvas import SystemGraphCanvas


class SystemGraphWindow(QWidget):
    def __init__(
        self,
        runtime_view: WorldViewSource,
        current_system_getter: Callable[[], str],
        jump_to_system: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.runtime_view = runtime_view
        self.current_system_getter = current_system_getter
        self.jump_to_system = jump_to_system
        self.setWindowTitle(QCoreApplication.translate("eve_sim", "System Map"))
        self.resize(360, 320)

        layout = QVBoxLayout(self)
        self.canvas = SystemGraphCanvas(
            lambda: getattr(self.runtime_view.world, "map_definition", None),
            self.current_system_getter,
            self.jump_to_system,
            self,
        )
        layout.addWidget(self.canvas, 1)

    def sync_from_world(self) -> None:
        self.canvas.sync_layout()


__all__ = ["SystemGraphWindow"]
