from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLocale, QTranslator
from PySide6.QtWidgets import QApplication

_INSTALLED_LANGUAGE = "en_US"
_APP_TRANSLATOR: QTranslator | None = None
_SUPPORTED_LANGUAGES = {"zh_CN", "en_US"}


def _translations_dir() -> Path:
    return Path(__file__).resolve().parent / "translations"


def _ensure_qt_application() -> QCoreApplication | None:
    app = QApplication.instance()
    if app is not None:
        return app
    return QCoreApplication.instance()


def normalize_language(lang: str | None, fallback: str = "en_US") -> str:
    normalized = str(lang or "").strip()
    if normalized in _SUPPORTED_LANGUAGES:
        return normalized
    return fallback if fallback in _SUPPORTED_LANGUAGES else "en_US"


def detect_system_language() -> str:
    system_name = (QLocale.system().name() or "").lower()
    if system_name.startswith("zh"):
        return "zh_CN"
    if system_name.startswith("en"):
        return "en_US"
    return "en_US"


def language_options() -> tuple[tuple[str, str], ...]:
    return (
        ("简体中文", "zh_CN"),
        ("English", "en_US"),
    )


def install_language(lang: str) -> str:
    global _APP_TRANSLATOR, _INSTALLED_LANGUAGE

    normalized = normalize_language(lang, "en_US")

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


__all__ = ["current_language", "detect_system_language", "install_language", "language_options", "normalize_language"]
