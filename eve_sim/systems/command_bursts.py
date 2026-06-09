from __future__ import annotations

from .combat_common import *  # noqa: F403


class CommandBurstsMixin:
    def _iter_area_targets_in_range(self, world: WorldState, source, module, effect, *, candidates: list | None = None) -> list:
        targets: list = []
        metadata = self._module_static_metadata(module)
        include_self = metadata.is_command_burst
        same_team_only = metadata.is_command_burst
        max_range = self._projected_max_range(effect)
        source_system_id = self._ship_system_id(source)

        candidate_iterable = candidates if candidates is not None else world.ships.values()
        for candidate in candidate_iterable:
            if not candidate.vital.alive:
                continue
            if self._ship_in_warp(candidate):
                continue
            if source_system_id and self._ship_system_id(candidate) != source_system_id:
                continue
            if candidate.ship_id == source.ship_id and not include_self:
                continue
            if same_team_only and candidate.team != source.team:
                continue
            distance = source.nav.position.distance_to(candidate.nav.position)
            if max_range > 0.0 and distance > max_range:
                continue
            targets.append(candidate)

        return targets

    def _collect_command_booster_snapshots(self, world: WorldState) -> dict[str, list[dict[str, Any]]]:
        snapshots_by_ship: dict[str, list[dict[str, Any]]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_combat_suppressed(source):
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue

            command_entries = self._runtime_module_buckets(source.runtime).command_entries
            if not command_entries:
                continue

            base_state_by_module_id: dict[str, str] = {}
            active_state_by_module_id: dict[str, str] = {}
            active_targets_by_module_id: dict[str, set[str]] = {}
            covered_targets: set[str] = set()

            for module, _metadata in command_entries:
                state_value = str(module.state.value or "ONLINE").upper()
                base_state_by_module_id[module.module_id] = "ONLINE" if state_value in {"ACTIVE", "OVERHEATED"} else state_value
                if state_value not in {"ACTIVE", "OVERHEATED"}:
                    continue

                # Command burst recipients are frozen at cycle start. Once the cycle begins,
                # leaving the burst radius does not remove a target and newly-entering ships
                # do not gain the burst until the next cycle starts.
                target_ids = set(
                    self._live_cycle_snapshot_target_ids(
                        world,
                        source.ship_id,
                        module.module_id,
                        team=source.team,
                        require_runtime=True,
                    )
                )
                if not target_ids:
                    continue

                active_state_by_module_id[module.module_id] = state_value
                active_targets_by_module_id[module.module_id] = target_ids
                covered_targets.update(target_ids)

            for target_id in sorted(covered_targets):
                state_by_module_id = dict(base_state_by_module_id)
                has_active_in_range = False
                for module_id, target_ids in active_targets_by_module_id.items():
                    if target_id not in target_ids:
                        continue
                    state_by_module_id[module_id] = active_state_by_module_id[module_id]
                    has_active_in_range = True

                if not has_active_in_range:
                    continue

                snapshots_by_ship.setdefault(target_id, []).append(
                    {
                        "fit_key": str(source.runtime.fit_key or ""),
                        "blueprint": blueprint,
                        "state_by_module_id": state_by_module_id,
                    }
                )

        return snapshots_by_ship
