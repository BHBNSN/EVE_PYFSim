from __future__ import annotations

from typing import Any

from PySide6.QtCore import QCoreApplication


class UserFacingError(Exception):
    def __init__(self, source_text: str, /, **params: Any) -> None:
        super().__init__(source_text)
        self.source_text = str(source_text)
        self.params = dict(params)

    def __str__(self) -> str:
        return self.source_text

    def to_display_text(self) -> str:
        text = QCoreApplication.translate("eve_sim", self.source_text)
        return text.format(**self.params) if self.params else text


def display_user_error(error: Exception | str) -> str:
    if isinstance(error, UserFacingError):
        return error.to_display_text()
    return str(error).strip()


__all__ = ["UserFacingError", "display_user_error"]
