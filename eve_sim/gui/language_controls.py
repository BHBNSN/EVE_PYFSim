from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtWidgets import QLabel, QWidget


_LANGUAGE_ICON_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="{color}" xmlns="http://www.w3.org/2000/svg"><path d="M12.87 15.07l-2.54-2.51.03-.03A17.52 17.52 0 0 0 14.07 6H17V4h-7V2H8v2H1v2h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"></path></svg>"""


class LanguageIconLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip("Language")
        self._refresh_icon()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._refresh_icon()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        color = self.palette().color(QPalette.ColorRole.WindowText).name()
        svg = _LANGUAGE_ICON_SVG.format(color=color).encode("utf-8")
        pixmap = QPixmap()
        if pixmap.loadFromData(svg, "SVG"):
            self.setPixmap(pixmap)
            self.setText("")
        else:
            self.clear()
            self.setText("L")


def language_icon_label(parent: QWidget | None = None) -> QLabel:
    return LanguageIconLabel(parent)
