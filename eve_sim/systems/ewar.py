from __future__ import annotations

from .combat_common import *  # noqa: F403


class EwarMixin:
    @staticmethod
    def _ecm_strength_from_effect(effect) -> dict[str, float]:
        return {
            "gravimetric": max(0.0, float(effect.projected_add.get("ecm_gravimetric", 0.0) or 0.0)),
            "ladar": max(0.0, float(effect.projected_add.get("ecm_ladar", 0.0) or 0.0)),
            "magnetometric": max(0.0, float(effect.projected_add.get("ecm_magnetometric", 0.0) or 0.0)),
            "radar": max(0.0, float(effect.projected_add.get("ecm_radar", 0.0) or 0.0)),
        }

    @staticmethod
    def _target_sensor_type_and_strength(profile: ShipProfile) -> tuple[str, float, bool]:
        strengths = {
            "gravimetric": max(0.0, float(getattr(profile, "sensor_strength_gravimetric", 0.0) or 0.0)),
            "ladar": max(0.0, float(getattr(profile, "sensor_strength_ladar", 0.0) or 0.0)),
            "magnetometric": max(0.0, float(getattr(profile, "sensor_strength_magnetometric", 0.0) or 0.0)),
            "radar": max(0.0, float(getattr(profile, "sensor_strength_radar", 0.0) or 0.0)),
        }
        sensor_type, sensor_strength = max(strengths.items(), key=lambda item: item[1])
        has_known_sensor_type = sensor_strength > 0.0
        if sensor_strength <= 0.0:
            sensor_strength = 1.0
        return sensor_type, sensor_strength, has_known_sensor_type

    @staticmethod
    def _ecm_duration_seconds(module_group: str) -> float:
        group = (module_group or "").lower()
        if "burst jammer" in group:
            return 5.0
        if "drone" in group:
            return 5.0
        return 20.0

    @staticmethod
    def _prune_ecm_sources(ship, now: float) -> set[str]:
        active_sources: set[str] = set()
        for source_id, jam_until in list(ship.combat.ecm_jam_sources.items()):
            if float(jam_until) > now:
                active_sources.add(str(source_id))
                continue
            ship.combat.ecm_jam_sources.pop(source_id, None)
        return active_sources

    def _can_target_under_ecm(self, ship, target_id: str | None, now: float) -> bool:
        if not target_id:
            return False
        active_sources = self._prune_ecm_sources(ship, now)
        if not active_sources:
            return True
        return str(target_id) in active_sources

    def _enforce_ecm_restrictions(self, ship, now: float) -> None:
        active_sources = self._prune_ecm_sources(ship, now)
        if not active_sources:
            return
        engaged_target_ids = set(self._lock_slot_targets(ship))
        engaged_target_ids.update(str(target_id) for target_id in ship.combat.fire_delay_timers.keys() if str(target_id))
        engaged_target_ids.update(
            str(target_id)
            for target_id in ship.combat.projected_targets.values()
            if str(target_id)
        )
        if ship.combat.current_target:
            engaged_target_ids.add(str(ship.combat.current_target))
        for target_id in engaged_target_ids:
            if target_id not in active_sources:
                self._drop_lock_target(ship, target_id)

    def _update_ecm_restrictions(self, world: WorldState, now: float | None = None) -> None:
        now_value = self._decision_now(world, now)
        for ship in world.ships.values():
            if not ship.vital.alive:
                ship.combat.ecm_jam_sources.clear()
                continue
            self._enforce_ecm_restrictions(ship, now_value)

    def _resolve_ecm_cycle(self, world: WorldState, source, module, target_id: str) -> None:
        target = world.ships.get(target_id)
        if target is None or not target.vital.alive:
            return
        if target_id not in source.combat.lock_targets:
            return
        now = float(world.now)
        distance = source.nav.position.distance_to(target.nav.position)
        target_sensor_type, target_sensor_strength, has_known_sensor_type = self._target_sensor_type_and_strength(target.profile)
        if target_sensor_strength <= 0.0:
            return

        jammed = False
        ecm_attempted = False
        jam_chance = 0.0
        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            strengths = self._ecm_strength_from_effect(effect)
            module_jam_strength = strengths.get(target_sensor_type, 0.0)
            if module_jam_strength <= 0.0 and not has_known_sensor_type:
                module_jam_strength = max(strengths.values(), default=0.0)
            if module_jam_strength <= 0.0:
                continue
            ecm_attempted = True

            if effect.falloff_m > 0.0:
                max_range = effect.range_m + 3.0 * effect.falloff_m
            else:
                max_range = effect.range_m
            if max_range > 0 and distance > max_range:
                continue

            if effect.range_m > 0.0 or effect.falloff_m > 0.0:
                range_factor = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
            else:
                range_factor = 1.0

            effective_strength = module_jam_strength * max(0.0, min(1.0, range_factor))
            chance = max(0.0, min(1.0, effective_strength / max(1e-9, target_sensor_strength)))
            jam_chance = max(jam_chance, chance)
            if random.random() < chance:
                jammed = True
                break

        if not ecm_attempted:
            return

        source.combat.ecm_last_attempt_target = target_id
        source.combat.ecm_last_attempt_module = module.module_id
        source.combat.ecm_last_attempt_success = jammed
        source.combat.ecm_last_attempt_chance = max(0.0, min(1.0, float(jam_chance)))
        source.combat.ecm_last_attempt_at = now
        source.combat.ecm_last_attempt_target_by_module[module.module_id] = target_id
        source.combat.ecm_last_attempt_success_by_module[module.module_id] = bool(jammed)
        source.combat.ecm_last_attempt_at_by_module[module.module_id] = now

        if not jammed:
            return

        jam_until = now + self._ecm_duration_seconds(module.group)
        target.combat.ecm_jam_sources[source.ship_id] = max(
            float(target.combat.ecm_jam_sources.get(source.ship_id, 0.0) or 0.0),
            jam_until,
        )
        self._enforce_ecm_restrictions(target, now)
        self._queue_merged_event(
            "ecm_jam_applied",
            merge_fields={
                "source": source.ship_id,
                "target": target.ship_id,
                "module": module.module_id,
                "sensor_type": target_sensor_type,
            },
            sum_fields={
                "chance": jam_chance,
                "duration_s": self._ecm_duration_seconds(module.group),
            },
        )

    def _resolve_area_ecm_cycle(self, world: WorldState, source, module) -> None:
        now = float(world.now)

        for effect in module.effects:
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            strengths = self._ecm_strength_from_effect(effect)
            if max(strengths.values(), default=0.0) <= 0.0:
                continue

            for target in self._iter_area_targets_in_range(world, source, module, effect):
                if target.ship_id == source.ship_id:
                    continue

                distance = source.nav.position.distance_to(target.nav.position)
                target_sensor_type, target_sensor_strength, has_known_sensor_type = self._target_sensor_type_and_strength(target.profile)
                if target_sensor_strength <= 0.0:
                    continue

                module_jam_strength = strengths.get(target_sensor_type, 0.0)
                if module_jam_strength <= 0.0 and not has_known_sensor_type:
                    module_jam_strength = max(strengths.values(), default=0.0)
                if module_jam_strength <= 0.0:
                    continue

                if effect.range_m > 0.0 or effect.falloff_m > 0.0:
                    range_factor = self.pyfa.turret_range_factor(effect.range_m, effect.falloff_m, distance)
                else:
                    range_factor = 1.0

                effective_strength = module_jam_strength * max(0.0, min(1.0, range_factor))
                jam_chance = max(0.0, min(1.0, effective_strength / max(1e-9, target_sensor_strength)))
                jammed = random.random() < jam_chance

                source.combat.ecm_last_attempt_target = target.ship_id
                source.combat.ecm_last_attempt_module = module.module_id
                source.combat.ecm_last_attempt_success = jammed
                source.combat.ecm_last_attempt_chance = jam_chance
                source.combat.ecm_last_attempt_at = now
                source.combat.ecm_last_attempt_target_by_module[module.module_id] = target.ship_id
                source.combat.ecm_last_attempt_success_by_module[module.module_id] = bool(jammed)
                source.combat.ecm_last_attempt_at_by_module[module.module_id] = now

                if not jammed:
                    continue

                self._break_all_locks(target)
                self._queue_merged_event(
                    "ecm_burst_lock_break",
                    merge_fields={
                        "source": source.ship_id,
                        "target": target.ship_id,
                        "module": module.module_id,
                        "sensor_type": target_sensor_type,
                    },
                    sum_fields={
                        "chance": jam_chance,
                    },
                )
