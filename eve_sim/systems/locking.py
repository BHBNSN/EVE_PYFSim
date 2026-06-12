from __future__ import annotations

from .combat_common import *  # noqa: F403


class LockingMixin:
    @staticmethod
    def _ship_in_warp(ship) -> bool:
        return str(getattr(getattr(ship.nav, "warp", None), "phase", "idle") or "idle") == "warp"

    @staticmethod
    def _ship_is_gate_cloaked(ship, now: float | None = None) -> bool:
        cloak = getattr(getattr(ship, "nav", None), "cloak", None)
        if cloak is None or not bool(getattr(cloak, "active", False)):
            return False
        if now is not None and float(getattr(cloak, "expires_at", 0.0) or 0.0) <= float(now):
            cloak.active = False
            cloak.expires_at = 0.0
            cloak.source = ""
            return False
        return True

    @classmethod
    def _ship_combat_suppressed(cls, ship, now: float | None = None) -> bool:
        return cls._ship_in_warp(ship) or cls._ship_is_gate_cloaked(ship, now)

    @classmethod
    def _ship_hidden_from_targeting(cls, ship, now: float | None = None) -> bool:
        return cls._ship_in_warp(ship) or cls._ship_is_gate_cloaked(ship, now)

    def _clear_ship_warp_engagement_state(self, ship, runtime: FitRuntime | None = None) -> None:
        self._break_all_locks(ship)
        ship.combat.last_attack_target = None
        if runtime is None:
            return

        controlled_entries = self._runtime_module_buckets(runtime).controlled_entries
        for module, metadata in controlled_entries:
            if not (
                metadata.has_projected
                or metadata.is_weapon
                or metadata.is_command_burst
                or metadata.is_area_effect
                or metadata.has_projected_rep
                or metadata.is_cap_warfare
                or metadata.is_target_ewar
                or metadata.is_ecm
            ):
                continue
            module_id = str(module.module_id)
            if module.state in {module.state.ACTIVE, module.state.OVERHEATED}:
                module.state = module.state.ONLINE
            self._clear_module_cycle_snapshots(ship.ship_id, module_id)
            self._clear_module_cycle_timer(ship, module_id)
            self._clear_module_reactivation_timer(ship, module_id)
        ship.combat.module_decision_propulsion_active = None
        ship.combat.module_decision_recent_enemy_damage_active = None
        ship.combat.module_decision_enemy_targets_active = None
        ship.combat.module_decision_ally_targets_active = None
        ship.combat.module_decision_pending.update(
            str(module.module_id) for module, _metadata in controlled_entries
        )

    @staticmethod
    def _lock_slot_targets(ship) -> set[str]:
        targets = {str(target_id) for target_id in ship.combat.lock_targets if str(target_id)}
        targets.update(str(target_id) for target_id in ship.combat.lock_timers.keys() if str(target_id))
        targets.update(str(target_id) for target_id in ship.combat.lock_deadlines.keys() if str(target_id))
        return targets

    @staticmethod
    def _lock_started_at(ship, target_id: str) -> float:
        try:
            return float(ship.combat.lock_started_at.get(str(target_id), float("-inf")))
        except Exception:
            return float("-inf")

    @staticmethod
    def _remember_lock_started(ship, target_id: str, now: float) -> None:
        normalized_target_id = str(target_id or "")
        if not normalized_target_id:
            return
        ship.combat.lock_started_at.setdefault(normalized_target_id, float(now))

    @staticmethod
    def _locked_target_is_in_use(ship, target_id: str) -> bool:
        normalized_target_id = str(target_id or "")
        if not normalized_target_id:
            return False
        if ship.combat.current_target == normalized_target_id:
            return True
        if normalized_target_id in ship.combat.fire_delay_timers:
            return True
        return normalized_target_id in ship.combat.projected_targets.values()

    def _drop_lock_target(self, ship, target_id: str | None) -> None:
        normalized_target_id = str(target_id or "")
        if not normalized_target_id:
            return
        ship.combat.lock_targets.discard(normalized_target_id)
        self._clear_lock_timer(ship, normalized_target_id)
        ship.combat.lock_started_at.pop(normalized_target_id, None)
        ship.combat.fire_delay_timers.pop(normalized_target_id, None)
        if ship.combat.current_target == normalized_target_id:
            ship.combat.current_target = None
        if ship.combat.last_attack_target == normalized_target_id:
            ship.combat.last_attack_target = None
        for module_id, projected_target_id in list(ship.combat.projected_targets.items()):
            if projected_target_id == normalized_target_id:
                ship.combat.projected_targets.pop(module_id, None)

    def _select_lock_eviction_target(self, ship, preserve_target_id: str | None) -> str | None:
        normalized_preserve_target_id = str(preserve_target_id or "")
        occupied = [
            target_id
            for target_id in self._lock_slot_targets(ship)
            if target_id and target_id != normalized_preserve_target_id
        ]
        if not occupied:
            return None
        idle_targets = [target_id for target_id in occupied if not self._locked_target_is_in_use(ship, target_id)]
        candidate_pool = idle_targets or occupied
        return min(candidate_pool, key=lambda target_id: (self._lock_started_at(ship, target_id), target_id))

    def _ensure_lock_slot_capacity(self, ship, preserve_target_id: str | None) -> bool:
        max_locked_targets = max(0, int(getattr(ship.profile, "max_locked_targets", 0) or 0))
        normalized_preserve_target_id = str(preserve_target_id or "")
        if max_locked_targets <= 0:
            return True
        while True:
            occupied = self._lock_slot_targets(ship)
            others = [target_id for target_id in occupied if target_id != normalized_preserve_target_id]
            if len(others) < max_locked_targets:
                return True
            eviction_target_id = self._select_lock_eviction_target(ship, normalized_preserve_target_id)
            if not eviction_target_id:
                return False
            self._drop_lock_target(ship, eviction_target_id)

    @staticmethod
    def _clear_lock_timer(ship, target_id: str) -> None:
        ship.combat.lock_timers.pop(target_id, None)
        ship.combat.lock_deadlines.pop(target_id, None)

    def _cached_lock_time(self, attacker_profile, defender_profile) -> float:
        key = (
            round(float(getattr(attacker_profile, "scan_resolution", 0.0) or 0.0), 4),
            round(float(getattr(defender_profile, "sig_radius", 0.0) or 0.0), 4),
        )
        cached = self._lock_time_cache.get(key)
        if cached is not None:
            return cached
        value = max(0.0, float(self.pyfa.calculate_lock_time(attacker_profile, defender_profile)))
        self._lock_time_cache[key] = value
        return value

    def _ensure_target_lock(
        self,
        world: WorldState,
        ship,
        target_id: str | None,
        target,
        *,
        lock_context: str,
        target_profile: ShipProfile | None = None,
        now: float | None = None,
    ) -> bool:
        if not target_id or target is None or not target.vital.alive or self._ship_hidden_from_targeting(target, self._decision_now(world, now)):
            if target_id:
                self._drop_lock_target(ship, target_id)
            return False
        if not self._target_within_lock_range(ship, target):
            self._drop_lock_target(ship, target_id)
            return False
        now_value = self._decision_now(world, now)
        if not self._can_target_under_ecm(ship, target_id, now_value):
            self._drop_lock_target(ship, target_id)
            return False
        if target_id in ship.combat.lock_targets:
            return True
        if target_id not in ship.combat.lock_deadlines and target_id not in ship.combat.lock_timers:
            if not self._ensure_lock_slot_capacity(ship, target_id):
                return False
            profile_for_lock = target_profile if target_profile is not None else target.profile
            lock_time = self._cached_lock_time(ship.profile, profile_for_lock)
            if lock_time <= 1e-9:
                ship.combat.lock_targets.add(target_id)
                self._remember_lock_started(ship, target_id, now_value)
                self._clear_lock_timer(ship, target_id)
                return True
            self._schedule_lock_deadline(
                ship,
                target_id,
                duration=lock_time,
                now=now_value,
            )
            if self.detailed_logging and self.logger is not None:
                self.logger.debug(
                    f"{lock_context}_start source={ship.ship_id} target={target_id} lock_time={ship.combat.lock_timers[target_id]:.2f}"
                )
        elif target_id not in ship.combat.lock_deadlines:
            remaining = max(0.0, float(ship.combat.lock_timers.get(target_id, 0.0) or 0.0))
            if remaining > 0.0:
                self._schedule_lock_deadline(ship, target_id, duration=remaining, now=now_value)
        return False

    def _advance_target_locks(self, world: WorldState, dt: float, now: float | None = None) -> None:
        now_value = self._decision_now(world, now)
        for ship in world.iter_combat_entities():
            if not ship.vital.alive:
                continue
            if self._ship_combat_suppressed(ship, now_value):
                self._break_all_locks(ship)
                continue
            if not ship.combat.lock_timers and not ship.combat.lock_deadlines and not ship.combat.lock_targets:
                continue
            self._prepare_ship_timer_views(ship, now_value)
            for target_id in list(ship.combat.lock_targets):
                target = world.combat_entity(target_id)
                if (
                    target is None
                    or not target.vital.alive
                    or self._ship_hidden_from_targeting(target, now_value)
                    or not self._can_target_under_ecm(ship, target_id, now_value)
                    or not self._target_within_lock_range(ship, target)
                ):
                    self._drop_lock_target(ship, target_id)
            for target_id, left in list(ship.combat.lock_timers.items()):
                target = world.combat_entity(target_id)
                if (
                    target is None
                    or not target.vital.alive
                    or self._ship_hidden_from_targeting(target, now_value)
                    or not self._can_target_under_ecm(ship, target_id, now_value)
                    or not self._target_within_lock_range(ship, target)
                ):
                    self._drop_lock_target(ship, target_id)
                    continue
                if target_id in ship.combat.lock_deadlines:
                    if float(left) > 0.0:
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(
                                f"lock_progress attacker={ship.ship_id} target={target_id} remaining={float(left):.2f}"
                            )
                        continue
                else:
                    left = max(0.0, float(left) or 0.0)
                    if left > 0.0:
                        self._schedule_lock_deadline(ship, target_id, duration=left, now=now_value)
                        if self.detailed_logging and self.logger is not None:
                            self.logger.debug(
                                f"lock_progress attacker={ship.ship_id} target={target_id} remaining={left:.2f}"
                            )
                        continue
                if float(ship.combat.lock_timers.get(target_id, 0.0) or 0.0) <= 0.0:
                    if not self._ensure_lock_slot_capacity(ship, target_id):
                        self._drop_lock_target(ship, target_id)
                        continue
                    ship.combat.lock_targets.add(target_id)
                    self._remember_lock_started(ship, target_id, now_value)
                    self._clear_lock_timer(ship, target_id)
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(f"lock_complete attacker={ship.ship_id} target={target_id}")

    @staticmethod
    def _lock_range_m(profile: ShipProfile | None) -> float:
        if profile is None:
            return 0.0
        return max(0.0, float(getattr(profile, "max_target_range", 0.0) or 0.0))

    @classmethod
    def _target_within_lock_range(
        cls,
        source,
        target,
        *,
        source_profile: ShipProfile | None = None,
    ) -> bool:
        if source is None or target is None:
            return False
        source_nav = getattr(source, "nav", None)
        target_nav = getattr(target, "nav", None)
        if source_nav is None or target_nav is None:
            return False
        source_system_id = getattr(source_nav, "system_id", "") or ""
        target_system_id = getattr(target_nav, "system_id", "") or ""
        if source_system_id and target_system_id and source_system_id != target_system_id:
            return False
        max_target_range = cls._lock_range_m(source_profile if source_profile is not None else getattr(source, "profile", None))
        if max_target_range <= 0.0:
            return False
        return source_nav.position.distance_to(target_nav.position) <= max_target_range

    def _ship_target_candidate_pools(
        self,
        world: WorldState,
        ship,
        *,
        focus_queue: tuple[str, ...],
        include_allies: bool = True,
        include_enemies: bool = True,
    ) -> tuple[list, list, set[str], set[str]]:
        if not include_allies and not include_enemies:
            return [], [], set(), set()

        source_system_id = self._ship_system_id(ship)
        source_nav = getattr(ship, "nav", None)
        if source_nav is None:
            return [], [], set(), set()
        max_target_range = self._lock_range_m(getattr(ship, "profile", None))
        if max_target_range <= 0.0:
            return [], [], set(), set()
        max_target_range_sq = max_target_range * max_target_range
        source_x = float(source_nav.position.x)
        source_y = float(source_nav.position.y)
        source_team = ship.team

        visible_ids: set[str] = set()
        if bool(getattr(ship, "perception_split_ready", False)):
            if include_allies:
                visible_ids.update(str(target_id) for target_id in getattr(ship, "perception_allies", ()) if str(target_id))
            if include_enemies:
                visible_ids.update(str(target_id) for target_id in getattr(ship, "perception_enemies", ()) if str(target_id))
        else:
            visible_ids.update(str(target_id) for target_id in getattr(ship, "perception", ()) if str(target_id))
        visible_ids.update(str(target_id) for target_id in focus_queue[:2] if str(target_id))
        visible_ids.update(str(target_id) for target_id in ship.combat.lock_targets if str(target_id))
        visible_ids.update(str(target_id) for target_id in ship.combat.lock_timers.keys() if str(target_id))
        visible_ids.update(str(target_id) for target_id in ship.combat.projected_targets.values() if str(target_id))
        current_target_id = str(getattr(ship.combat, "current_target", "") or "")
        if current_target_id:
            visible_ids.add(current_target_id)

        allies_pool: list[Any] = []
        enemies_pool: list[Any] = []
        ally_ids: set[str] = set()
        enemy_ids: set[str] = set()
        for candidate_id in visible_ids:
            candidate = world.combat_entity(candidate_id)
            if candidate is None or not candidate.vital.alive or self._ship_hidden_from_targeting(candidate):
                continue
            candidate_nav = getattr(candidate, "nav", None)
            if candidate_nav is None:
                continue
            candidate_system_id = getattr(candidate_nav, "system_id", "") or ""
            if source_system_id and candidate_system_id and candidate_system_id != source_system_id:
                continue

            is_ally = candidate.team == source_team
            if is_ally:
                if not include_allies:
                    continue
            elif not include_enemies:
                continue

            dx = source_x - float(candidate_nav.position.x)
            dy = source_y - float(candidate_nav.position.y)
            if (dx * dx) + (dy * dy) > max_target_range_sq:
                continue

            if is_ally:
                allies_pool.append(candidate)
                ally_ids.add(candidate_id)
            else:
                enemies_pool.append(candidate)
                enemy_ids.add(candidate_id)
        return allies_pool, enemies_pool, ally_ids, enemy_ids

    @staticmethod
    def _break_all_locks(ship) -> None:
        ship.combat.lock_targets.clear()
        ship.combat.lock_timers.clear()
        ship.combat.lock_deadlines.clear()
        ship.combat.lock_started_at.clear()
        ship.combat.fire_delay_timers.clear()
        ship.combat.projected_targets.clear()
        ship.combat.current_target = None
        ship.combat.last_attack_target = None
