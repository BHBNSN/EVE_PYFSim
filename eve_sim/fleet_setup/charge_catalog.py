from __future__ import annotations

from .engine import (
    get_charge_option_values_for_module,
    get_module_reload_time_sec,
    module_supports_unloaded_charge,
    resolve_module_type_name,
)


class FitChargeCatalog:
    """Infrastructure adapter exposing fit metadata to domain fit rules."""

    @staticmethod
    def resolve_type_name(type_name: str) -> str:
        return resolve_module_type_name(type_name)

    @staticmethod
    def charge_options(module_name: str) -> tuple[str, ...]:
        return tuple(get_charge_option_values_for_module(module_name))

    @staticmethod
    def supports_unloaded(module_name: str) -> bool:
        return module_supports_unloaded_charge(module_name)

    @staticmethod
    def reload_time(module_name: str) -> float:
        return float(get_module_reload_time_sec(module_name))
