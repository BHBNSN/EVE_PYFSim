from __future__ import annotations

from typing import Any

from ..models import Team
from ..world import WorldState


class LogisticsSystem:
    @staticmethod
    def _apply_repair(target, amount: float) -> None:
        remaining = max(0.0, float(amount))
        if remaining <= 0.0:
            return
        missing_shield = max(0.0, float(target.vital.shield_max) - float(target.vital.shield))
        if missing_shield > 0.0:
            restored = min(remaining, missing_shield)
            target.vital.shield += restored
            remaining -= restored
        if remaining <= 0.0:
            return
        missing_armor = max(0.0, float(target.vital.armor_max) - float(target.vital.armor))
        if missing_armor > 0.0:
            target.vital.armor += min(remaining, missing_armor)

    def run(self, world: WorldState, dt: float) -> None:
        alive_by_team: dict[Team, list] = {Team.BLUE: [], Team.RED: []}
        for ship in world.ships.values():
            if ship.vital.alive:
                alive_by_team[ship.team].append(ship)

        weakest_by_team: dict[Team, Any | None] = {Team.BLUE: None, Team.RED: None}
        for team, members in alive_by_team.items():
            if not members:
                weakest_by_team[team] = None
                continue
            weakest_by_team[team] = min(members, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            if ship.runtime is not None:
                # Runtime-backed fits already apply remote repair effects through CombatSystem.
                continue
            if ship.profile.rep_amount <= 0 or ship.profile.rep_cycle <= 0:
                continue

            target = weakest_by_team.get(ship.team)
            if target is None or target.ship_id == ship.ship_id:
                allies = [a for a in alive_by_team.get(ship.team, []) if a.ship_id != ship.ship_id]
                if not allies:
                    continue
                target = min(allies, key=lambda a: (a.vital.shield + a.vital.armor + a.vital.structure))

            dist = ship.nav.position.distance_to(target.nav.position)
            if dist > ship.profile.max_target_range:
                continue

            repair = ship.profile.rep_amount * (dt / ship.profile.rep_cycle)
            self._apply_repair(target, repair)


from .combat_common import *  # noqa: F403


class CombatLogisticsMixin:
    def _mark_all_repair_queues_dirty(self) -> None:
        self._repair_queue_cache.clear()
        self._repair_queue_dirty.clear()

    def _mark_team_repair_queues_dirty(self, team: Team, *layers: str) -> None:
        if team is None:
            return
        dirty_layers = tuple(str(layer) for layer in (layers or _REPAIR_QUEUE_LAYERS) if str(layer) in _REPAIR_QUEUE_LAYERS)
        for layer in dirty_layers:
            stale_keys = [cache_key for cache_key in self._repair_queue_cache.keys() if cache_key[0] == team and cache_key[1] == layer]
            for cache_key in stale_keys:
                self._repair_queue_cache.pop(cache_key, None)
                self._repair_queue_dirty.add(cache_key)

    @staticmethod
    def _hp_ratio(ship) -> float:
        hp_max = max(1.0, ship.vital.shield_max + ship.vital.armor_max + ship.vital.structure_max)
        hp_now = ship.vital.shield + ship.vital.armor + ship.vital.structure
        return hp_now / hp_max

    @staticmethod
    def _ship_layer_values(ship, layer: str) -> tuple[float, float]:
        if layer == "shield":
            return float(ship.vital.shield), max(1.0, float(ship.vital.shield_max))
        if layer == "armor":
            return float(ship.vital.armor), max(1.0, float(ship.vital.armor_max))
        return float(ship.vital.structure), max(1.0, float(ship.vital.structure_max))

    @classmethod
    def _ship_layer_fraction(cls, ship, layer: str) -> float:
        current, maximum = cls._ship_layer_values(ship, layer)
        return max(0.0, min(1.0, current / maximum))

    @classmethod
    def _ship_needs_layer_repair(cls, ship, layer: str) -> bool:
        if not ship.vital.alive:
            return False
        current, maximum = cls._ship_layer_values(ship, layer)
        return (maximum - current) > 1e-6

    @staticmethod
    def _ship_disallows_assistance(ship) -> bool:
        profile = getattr(ship, "profile", None)
        if profile is None:
            return False
        return bool(getattr(profile, "disallow_assistance", False))

    def _team_repair_queue(self, world: WorldState, team: Team, layer: str, system_id: str = "") -> tuple[str, ...]:
        cache_key = (team, layer, str(system_id or ""))
        if cache_key not in self._repair_queue_dirty:
            cached = self._repair_queue_cache.get(cache_key)
            if cached is not None:
                return cached

        ranked: list[tuple[float, str]] = []
        for ship in world.ships.values():
            if ship.team != team:
                continue
            if self._ship_hidden_from_targeting(ship):
                continue
            if system_id and self._ship_system_id(ship) != str(system_id):
                continue
            if self._ship_disallows_assistance(ship):
                continue
            if not self._ship_needs_layer_repair(ship, layer):
                continue
            ranked.append((self._ship_layer_fraction(ship, layer), str(ship.ship_id)))
        ranked.sort(key=lambda item: (item[0], item[1]))
        queue = tuple(ship_id for _fraction, ship_id in ranked)
        self._repair_queue_cache[cache_key] = queue
        self._repair_queue_dirty.discard(cache_key)
        return queue

    def _select_repair_queue_target(self, world: WorldState, source, module, metadata: ModuleStaticMetadata) -> str | None:
        source_system_id = self._ship_system_id(source)
        for layer in metadata.repair_layers:
            for target_id in self._team_repair_queue(world, source.team, layer, source_system_id):
                if target_id == source.ship_id:
                    continue
                target = world.ships.get(target_id)
                if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target) or target.team != source.team:
                    continue
                if self._ship_disallows_assistance(target):
                    continue
                if not self._ship_needs_layer_repair(target, layer):
                    continue
                if not self._target_within_lock_range(source, target):
                    continue
                if not self._module_in_projected_range(source, target, module):
                    continue
                return target_id
        return None

    def _module_in_projected_range(
        self,
        source,
        target,
        module,
        *,
        metadata: ModuleStaticMetadata | None = None,
        source_system_id: str | None = None,
        distance: float | None = None,
    ) -> bool:
        if source_system_id is None:
            source_system_id = self._ship_system_id(source)
        target_system_id = self._ship_system_id(target)
        if source_system_id and target_system_id and source_system_id != target_system_id:
            return False
        projected_max_range = self._module_projected_max_range(module, metadata)
        if projected_max_range is None or projected_max_range <= 0.0:
            return True
        distance_value = distance if distance is not None else source.nav.position.distance_to(target.nav.position)
        return distance_value <= projected_max_range

    @staticmethod
    def _cap_ratio(ship) -> float:
        return max(0.0, float(ship.vital.cap) / max(1.0, float(ship.vital.cap_max)))
