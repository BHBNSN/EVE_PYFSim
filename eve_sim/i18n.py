from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QApplication

_INSTALLED_LANGUAGE = "en_US"
_APP_TRANSLATOR: QTranslator | None = None


def _translations_dir() -> Path:
    return Path(__file__).resolve().parent / "translations"


def _ensure_qt_application() -> QCoreApplication | None:
    app = QApplication.instance()
    if app is not None:
        return app
    return QCoreApplication.instance()


def install_language(lang: str) -> str:
    global _APP_TRANSLATOR, _INSTALLED_LANGUAGE

    normalized = str(lang or "en_US").strip() or "en_US"
    if normalized not in {"zh_CN", "en_US"}:
        normalized = "en_US"

    app = _ensure_qt_application()
    if app is None:
        _INSTALLED_LANGUAGE = normalized
        return normalized

    if _APP_TRANSLATOR is not None:
        app.removeTranslator(_APP_TRANSLATOR)
        _APP_TRANSLATOR = None

    if normalized == "zh_CN":
        translator = QTranslator(app)
        qm_path = _translations_dir() / "eve_sim_zh_CN.qm"
        if translator.load(str(qm_path)):
            app.installTranslator(translator)
            _APP_TRANSLATOR = translator
        else:
            normalized = "en_US"

    _INSTALLED_LANGUAGE = normalized
    return normalized


def current_language() -> str:
    return _INSTALLED_LANGUAGE


__all__ = ["current_language", "install_language"]
