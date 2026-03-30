from __future__ import annotations

MODULE_MANUAL_MODES = frozenset({"auto", "active", "online"})
MODULE_TARGET_MODES = frozenset(
    {
        "auto",
        "weapon_focus_prefocus",
        "enemy_nearest",
        "enemy_random",
        "ally_repair_queue",
        "ally_nearest",
    }
)


def normalize_module_manual_mode(mode: str | None) -> str:
    normalized = str(mode or "auto").strip().lower()
    return normalized if normalized in MODULE_MANUAL_MODES else "auto"


def normalize_module_target_mode(mode: str | None) -> str:
    normalized = str(mode or "auto").strip().lower()
    return normalized if normalized in MODULE_TARGET_MODES else "auto"


def effective_module_target_mode(mode: str | None, default_mode: str | None) -> str:
    normalized = normalize_module_target_mode(mode)
    if normalized != "auto":
        return normalized
    normalized_default = normalize_module_target_mode(default_mode)
    return normalized_default if normalized_default != "auto" else "auto"


def stored_module_target_mode(mode: str | None, default_mode: str | None) -> str:
    normalized = normalize_module_target_mode(mode)
    normalized_default = normalize_module_target_mode(default_mode)
    if normalized_default != "auto" and normalized == normalized_default:
        return "auto"
    return normalized
