from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from ..models import ShipEntity, Team
from ..world import WorldState


class FitParserPort(Protocol):
    def parse(self, fit_text: str):
        ...


class FitFactoryPort(Protocol):
    def build(self, parsed):
        ...

    def build_profile(self, parsed):
        ...


class FitRuntimePort(Protocol):
    def copy_runtime_dynamic_state(self, source_runtime, target_runtime) -> None:
        ...

    def request_module_reload(self, ship: ShipEntity, module_id: str, reload_time: float, *, now: float) -> None:
        ...


class ChargeCatalogPort(Protocol):
    def resolve_type_name(self, type_name: str) -> str:
        ...

    def charge_options(self, module_name: str) -> tuple[str, ...]:
        ...

    def supports_unloaded(self, module_name: str) -> bool:
        ...

    def reload_time(self, module_name: str) -> float:
        ...


class ShipFitService:
    """Atomically rebuilds a ship fit and preserves compatible runtime state."""

    def __init__(
        self,
        parser: FitParserPort,
        factory: FitFactoryPort,
        runtime: FitRuntimePort,
        charge_catalog: ChargeCatalogPort | None = None,
    ) -> None:
        self._parser = parser
        self._factory = factory
        self._runtime = runtime
        self._charge_catalog = charge_catalog

    @staticmethod
    def _initial_fit_key(ship: ShipEntity) -> str:
        runtime = ship.runtime
        if runtime is not None and isinstance(runtime.diagnostics, dict):
            key = str(runtime.diagnostics.get("initial_fit_key", "") or "")
            if key:
                return key
        return str(ship.fit.fit_key or "")

    def replace_fit(
        self,
        world: WorldState,
        team: Team,
        ship_id: str,
        fit_text: str,
        reloads: tuple[tuple[str, float], ...] = (),
    ) -> str:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError("ship does not exist")
        if ship.team != team:
            raise ValueError(f"ship is not controlled by {team.value}")

        parsed = self._parser.parse(str(fit_text))
        runtime_template, fit = self._factory.build(parsed)
        runtime = deepcopy(runtime_template)
        profile = self._factory.build_profile(parsed)
        if hasattr(self._factory, "build_deployable_manifest"):
            drone_bay, fighter_bay, deployable_control = self._factory.build_deployable_manifest(parsed)
        else:
            from ..models import DeployableControlState

            drone_bay, fighter_bay, deployable_control = [], [], DeployableControlState()

        initial_fit_key = self._initial_fit_key(ship)
        if ship.runtime is not None:
            self._runtime.copy_runtime_dynamic_state(ship.runtime, runtime)
        runtime_module_ids = {module.module_id for module in runtime.modules}
        normalized_reloads = tuple(
            (str(module_id), max(0.0, float(seconds)))
            for module_id, seconds in reloads
            if str(module_id) in runtime_module_ids and float(seconds) > 0.0
        )

        ship.runtime = runtime
        ship.fit = fit
        ship.profile = profile
        ship.drone_bay = list(drone_bay)
        ship.fighter_bay = list(fighter_bay)
        ship.deployable_control = deepcopy(deployable_control)
        ship.fit_text = str(fit_text)
        if isinstance(runtime.diagnostics, dict) and initial_fit_key:
            runtime.diagnostics["initial_fit_key"] = initial_fit_key

        for timer_map in (
            ship.combat.module_cycle_timers,
            ship.combat.module_reactivation_timers,
            ship.combat.module_ammo_reload_timers,
            ship.combat.module_pending_ammo_reload_timers,
        ):
            for module_id in tuple(timer_map):
                if module_id not in runtime_module_ids:
                    timer_map.pop(module_id, None)
        for mode_map in (ship.combat.module_manual_modes, ship.combat.module_target_modes):
            for module_id in tuple(mode_map):
                if module_id not in runtime_module_ids:
                    mode_map.pop(module_id, None)
        for module_id in tuple(ship.locked_module_charges):
            if module_id not in runtime_module_ids:
                ship.locked_module_charges.pop(module_id, None)

        for module_id, seconds in normalized_reloads:
            self._runtime.request_module_reload(ship, module_id, seconds, now=float(world.now))
        return str(getattr(fit, "fit_key", "") or "")

    def _validate_replacement(self, world: WorldState, team: Team, ship_id: str, fit_text: str) -> None:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError(f"ship does not exist: {ship_id}")
        if ship.team != team:
            raise ValueError(f"ship is not controlled by {team.value}: {ship_id}")
        parsed = self._parser.parse(str(fit_text))
        self._factory.build(parsed)
        self._factory.build_profile(parsed)
        if hasattr(self._factory, "build_deployable_manifest"):
            self._factory.build_deployable_manifest(parsed)

    def replace_many(
        self,
        world: WorldState,
        team: Team,
        replacements: tuple[tuple[str, str, tuple[tuple[str, float], ...]], ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            (str(ship_id), str(fit_text), tuple(reloads))
            for ship_id, fit_text, reloads in replacements
        )
        if not normalized:
            raise ValueError("at least one fit replacement is required")
        for ship_id, fit_text, _reloads in normalized:
            self._validate_replacement(world, team, ship_id, fit_text)
        for ship_id, fit_text, reloads in normalized:
            self.replace_fit(world, team, ship_id, fit_text, reloads)
        return tuple(ship_id for ship_id, _fit_text, _reloads in normalized)

    @staticmethod
    def _module_index(module_id: str) -> int | None:
        prefix, separator, suffix = str(module_id).rpartition("-")
        if separator and prefix == "mod" and suffix.isdigit() and int(suffix) > 0:
            return int(suffix)
        return None

    def _resolved(self, value: str) -> str:
        if self._charge_catalog is None:
            raise RuntimeError("charge catalog is not configured")
        return self._charge_catalog.resolve_type_name(str(value or "")).strip()

    @staticmethod
    def _format_module_line(module_name: str, charge_name: str, offline_suffix: str) -> str:
        charge = str(charge_name or "").strip()
        return f"{module_name}, {charge}{offline_suffix}" if charge else f"{module_name}{offline_suffix}"

    def _rewrite_charge_lines(
        self,
        ship: ShipEntity,
        fit_text: str,
        *,
        target_module_name: str = "",
        target_charge_name: str = "",
        forced_module_id: str = "",
        forced_charge_name: str = "",
    ) -> str:
        canonical_target_module = self._resolved(target_module_name).lower() if target_module_name else ""
        canonical_target_charge = self._resolved(target_charge_name) if target_charge_name else ""
        canonical_forced_charge = self._resolved(forced_charge_name) if forced_charge_name else ""
        output: list[str] = []
        module_index = 0
        for line in str(fit_text).splitlines():
            raw = line.strip()
            lowered = raw.lower()
            if not raw or raw.startswith("[") or lowered.startswith("dna:") or lowered.startswith("x-") or " x" in raw:
                output.append(line)
                continue

            base = raw
            offline_suffix = ""
            if lowered.endswith("/offline"):
                offline_suffix = raw[-9:]
                base = raw[:-9].rstrip()
            if not base or base.startswith("[Empty"):
                output.append(line)
                continue

            module_index += 1
            module_id = f"mod-{module_index}"
            module_name = base.split(",", 1)[0].strip()
            if forced_module_id and module_id == forced_module_id:
                output.append(self._format_module_line(module_name, canonical_forced_charge, offline_suffix))
                continue
            if module_id in ship.locked_module_charges:
                output.append(
                    self._format_module_line(
                        module_name,
                        str(ship.locked_module_charges[module_id] or ""),
                        offline_suffix,
                    )
                )
                continue
            if canonical_target_module and self._resolved(module_name).lower() == canonical_target_module:
                output.append(self._format_module_line(module_name, canonical_target_charge, offline_suffix))
            else:
                output.append(line)
        return "\n".join(output)

    def _changed_charge_module_ids(self, old_fit_text: str, new_fit_text: str) -> tuple[str, ...]:
        old_specs = list(getattr(self._parser.parse(old_fit_text), "module_specs", ()) or ())
        new_specs = list(getattr(self._parser.parse(new_fit_text), "module_specs", ()) or ())
        return tuple(
            f"mod-{index + 1}"
            for index, (old_spec, new_spec) in enumerate(zip(old_specs, new_specs))
            if self._resolved(str(old_spec.charge_name or "")).lower()
            != self._resolved(str(new_spec.charge_name or "")).lower()
        )

    def set_charge_lock(
        self,
        world: WorldState,
        team: Team,
        ship_id: str,
        module_id: str,
        charge_name: str,
    ) -> tuple[str, float]:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError("ship does not exist")
        if ship.team != team:
            raise ValueError(f"ship is not controlled by {team.value}")
        fit_text = str(ship.fit_text or "")
        if not fit_text.strip():
            raise ValueError("ship fit text is unavailable")
        parsed = self._parser.parse(fit_text)
        module_index = self._module_index(module_id)
        module_specs = list(getattr(parsed, "module_specs", ()) or ())
        if module_index is None or module_index > len(module_specs):
            raise ValueError("module slot does not exist")
        module_name = str(module_specs[module_index - 1].module_name or "")
        canonical_charge = self._resolved(charge_name) if str(charge_name or "").strip() else ""
        if self._charge_catalog is None:
            raise RuntimeError("charge catalog is not configured")
        valid_charges = {self._resolved(value).lower() for value in self._charge_catalog.charge_options(module_name)}
        if self._charge_catalog.supports_unloaded(module_name):
            valid_charges.add("")
        if canonical_charge.lower() not in valid_charges:
            raise ValueError("charge does not match module")

        rewritten = self._rewrite_charge_lines(
            ship,
            fit_text,
            forced_module_id=str(module_id),
            forced_charge_name=canonical_charge,
        )
        reload_time = max(0.0, float(self._charge_catalog.reload_time(module_name)))
        reloads = tuple(
            (changed_module_id, reload_time)
            for changed_module_id in self._changed_charge_module_ids(fit_text, rewritten)
            if reload_time > 0.0
        )
        self.replace_fit(world, team, ship_id, rewritten, reloads)
        ship.locked_module_charges[str(module_id)] = canonical_charge
        return module_name, reload_time

    def set_fleet_charge(
        self,
        world: WorldState,
        team: Team,
        module_name: str,
        charge_name: str,
    ) -> tuple[tuple[str, ...], float]:
        if self._charge_catalog is None:
            raise RuntimeError("charge catalog is not configured")
        canonical_module = self._resolved(module_name)
        canonical_charge = self._resolved(charge_name) if str(charge_name or "").strip() else ""
        valid_charges = {self._resolved(value).lower() for value in self._charge_catalog.charge_options(canonical_module)}
        if self._charge_catalog.supports_unloaded(canonical_module):
            valid_charges.add("")
        if canonical_charge.lower() not in valid_charges:
            raise ValueError("charge does not match module")

        reload_time = max(0.0, float(self._charge_catalog.reload_time(canonical_module)))
        replacements: list[tuple[str, str, tuple[tuple[str, float], ...]]] = []
        for ship in sorted(world.ships.values(), key=lambda item: item.ship_id):
            if ship.team != team or not str(ship.fit_text or "").strip():
                continue
            rewritten = self._rewrite_charge_lines(
                ship,
                ship.fit_text,
                target_module_name=canonical_module,
                target_charge_name=canonical_charge,
            )
            if rewritten == ship.fit_text:
                continue
            reloads = tuple(
                (module_id, reload_time)
                for module_id in self._changed_charge_module_ids(ship.fit_text, rewritten)
                if reload_time > 0.0
            )
            replacements.append((ship.ship_id, rewritten, reloads))
        if not replacements:
            raise ValueError("no matching module entries were found")
        ship_ids = self.replace_many(world, team, tuple(replacements))
        return ship_ids, reload_time

    @staticmethod
    def clear_charge_lock(world: WorldState, team: Team, ship_id: str, module_id: str) -> None:
        ship = world.ships.get(str(ship_id))
        if ship is None:
            raise ValueError("ship does not exist")
        if ship.team != team:
            raise ValueError(f"ship is not controlled by {team.value}")
        ship.locked_module_charges.pop(str(module_id), None)
