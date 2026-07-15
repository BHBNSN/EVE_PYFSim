from __future__ import annotations

from typing import Protocol

from ..fit_runtime import ModuleRuntime, ModuleState
from ..module_control import normalize_module_manual_mode, normalize_module_target_mode, stored_module_target_mode
from ..models import ShipEntity, Team
from ..world import WorldState


class ModuleMetadataPort(Protocol):
    def module_target_metadata(self, module: ModuleRuntime):
        ...

    def module_target_mode_choices(self, module: ModuleRuntime) -> tuple[str, ...] | list[str]:
        ...


class ShipModuleService:
    """Business rules for per-ship module overrides at command boundaries."""

    def __init__(self, metadata: ModuleMetadataPort) -> None:
        self._metadata = metadata

    @staticmethod
    def _controlled_ship(world: WorldState, team: Team, ship_id: str) -> ShipEntity:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError("ship does not exist")
        if ship.team != team:
            raise ValueError(f"ship is not controlled by {team.value}")
        if ship.runtime is None:
            raise ValueError("ship has no runtime")
        return ship

    @staticmethod
    def _module(ship: ShipEntity, module_id: str) -> ModuleRuntime:
        module = next(
            (candidate for candidate in ship.runtime.modules if str(candidate.module_id) == str(module_id)),
            None,
        ) if ship.runtime is not None else None
        if module is None:
            raise ValueError("module slot does not exist")
        return module

    @staticmethod
    def _require_active_capability(module: ModuleRuntime) -> None:
        if not module.can_be_active() or module.state == ModuleState.OFFLINE:
            raise ValueError("module does not support mode overrides")

    @staticmethod
    def _apply_manual(ship: ShipEntity, module_id: str, mode: str) -> None:
        if mode == "auto":
            ship.combat.module_manual_modes.pop(module_id, None)
        else:
            ship.combat.module_manual_modes[module_id] = mode
        ship.combat.module_decision_pending.add(str(module_id))

    @staticmethod
    def _apply_target(ship: ShipEntity, module_id: str, mode: str) -> None:
        if mode == "auto":
            ship.combat.module_target_modes.pop(module_id, None)
        else:
            ship.combat.module_target_modes[module_id] = mode
        ship.combat.module_decision_pending.add(str(module_id))

    def set_manual_mode(self, world: WorldState, team: Team, ship_id: str, module_id: str, mode: str) -> str:
        ship = self._controlled_ship(world, team, ship_id)
        module = self._module(ship, module_id)
        self._require_active_capability(module)
        normalized = normalize_module_manual_mode(mode)
        self._apply_manual(ship, module_id, normalized)
        return normalized

    def _validated_target_mode(self, module: ModuleRuntime, mode: str) -> tuple[str, str]:
        valid_modes = tuple(self._metadata.module_target_mode_choices(module))
        if not valid_modes:
            raise ValueError("module does not support target rule overrides")
        normalized = normalize_module_target_mode(mode)
        if normalized != "auto" and normalized not in valid_modes:
            raise ValueError("target rule is invalid for this module")
        metadata = self._metadata.module_target_metadata(module)
        default_mode = normalize_module_target_mode(getattr(metadata.decision_rule, "target_mode", "auto"))
        return normalized, stored_module_target_mode(normalized, default_mode)

    def set_target_mode(self, world: WorldState, team: Team, ship_id: str, module_id: str, mode: str) -> str:
        ship = self._controlled_ship(world, team, ship_id)
        module = self._module(ship, module_id)
        normalized, applied = self._validated_target_mode(module, mode)
        self._apply_target(ship, module_id, applied)
        return normalized

    def target_rules(self, world: WorldState, ship_id: str, module_id: str) -> tuple[tuple[str, ...], str]:
        ship = world.ships.get(str(ship_id))
        if ship is None or ship.runtime is None:
            return (), "auto"
        try:
            module = self._module(ship, module_id)
        except ValueError:
            return (), "auto"
        choices = tuple(self._metadata.module_target_mode_choices(module))
        metadata = self._metadata.module_target_metadata(module)
        default_mode = normalize_module_target_mode(getattr(metadata.decision_rule, "target_mode", "auto"))
        return choices, default_mode

    @staticmethod
    def _initial_fit_key(ship: ShipEntity) -> str:
        runtime = ship.runtime
        if runtime is not None and isinstance(runtime.diagnostics, dict):
            key = str(runtime.diagnostics.get("initial_fit_key", "") or "")
            if key:
                return key
        return str(ship.fit.fit_key or "")

    def sync_squad_controls(
        self,
        world: WorldState,
        team: Team,
        ship_id: str,
        module_id: str,
        manual_mode: str,
        target_mode: str,
    ) -> tuple[str, ...]:
        source = self._controlled_ship(world, team, ship_id)
        source_module = self._module(source, module_id)
        self._require_active_capability(source_module)
        normalized_manual = normalize_module_manual_mode(manual_mode)
        normalized_target = normalize_module_target_mode(target_mode)
        source_fit_key = self._initial_fit_key(source)

        planned: list[tuple[ShipEntity, str]] = []
        for candidate in sorted(world.ships.values(), key=lambda item: item.ship_id):
            if candidate.team != team or candidate.squad_id != source.squad_id:
                continue
            if candidate.runtime is None or self._initial_fit_key(candidate) != source_fit_key:
                continue
            try:
                candidate_module = self._module(candidate, module_id)
                self._require_active_capability(candidate_module)
            except ValueError:
                continue
            valid_modes = tuple(self._metadata.module_target_mode_choices(candidate_module))
            if valid_modes:
                candidate_metadata = self._metadata.module_target_metadata(candidate_module)
                default_mode = normalize_module_target_mode(getattr(candidate_metadata.decision_rule, "target_mode", "auto"))
                requested = normalized_target if normalized_target in valid_modes or normalized_target == "auto" else "auto"
                applied_target = stored_module_target_mode(requested, default_mode)
            else:
                applied_target = "auto"
            planned.append((candidate, applied_target))

        if not planned:
            raise ValueError("no matching squad ships can be updated")
        for candidate, applied_target in planned:
            self._apply_manual(candidate, module_id, normalized_manual)
            self._apply_target(candidate, module_id, applied_target)
        return tuple(candidate.ship_id for candidate, _ in planned)
