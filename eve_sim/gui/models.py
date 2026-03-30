from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, cast

from PySide6.QtGui import QColor

from ..math2d import Vector2
from ..models import QualityLevel, Team


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
    def _read_filter_team(
        data: dict[str, object],
        default: Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"],
    ) -> Literal["ALL", "FRIENDLY", "ENEMY", "BLUE", "RED"]:
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
                engine_physics_substeps=self._read_int(
                    migrated,
                    "engine_physics_substeps",
                    defaults.engine_physics_substeps,
                    1,
                ),
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


__all__ = [
    "AreaCycleOverlay",
    "PreferencesStore",
    "SetupRow",
    "UiPreferences",
    "UiState",
]
