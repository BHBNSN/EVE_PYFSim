from __future__ import annotations

from .combat_common import *  # noqa: F403


class AuthoritativeTickMixin:
    def _log_hotspot(self, name: str, start_time: float, **fields: Any) -> None:
        if not self.hotspot_logging_enabled:
            return
        if self.logger is None or self.logger.disabled:
            return
        log_sim_event(
            self.logger,
            "hotspot",
            name=name,
            duration_ms=(time.perf_counter() - start_time) * 1000.0,
            **fields,
        )

    def recommended_time_slice(self, world: WorldState, max_dt: float) -> float:
        slice_dt = max(1e-6, float(max_dt))
        now = float(world.now)
        epsilon = 1e-6

        def note_duration(value: float | None) -> None:
            nonlocal slice_dt
            if value is None:
                return
            try:
                duration = float(value)
            except Exception:
                return
            if epsilon < duration < slice_dt:
                slice_dt = duration

        next_due_time = self._next_timer_event_time(world)
        if next_due_time is not None:
            note_duration(max(epsilon, float(next_due_time) - now))

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            for target_id, remaining in ship.combat.lock_timers.items():
                if target_id in ship.combat.lock_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_cycle_timers.items():
                if module_id in ship.combat.module_cycle_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_reactivation_timers.items():
                if module_id in ship.combat.module_reactivation_deadlines:
                    continue
                note_duration(remaining)

            for module_id, remaining in ship.combat.module_ammo_reload_timers.items():
                if module_id in ship.combat.module_ammo_reload_deadlines:
                    continue
                note_duration(remaining)

            for remaining in ship.combat.module_pending_ammo_reload_timers.values():
                note_duration(remaining)

            for ready_at in ship.combat.fire_delay_timers.values():
                note_duration(float(ready_at) - now)

            for jam_until in ship.combat.ecm_jam_sources.values():
                note_duration(float(jam_until) - now)

            raw_last_enemy_damage = getattr(ship.combat, "last_enemy_weapon_damaged_at", -1e9)
            last_enemy_damage = float(raw_last_enemy_damage if raw_last_enemy_damage is not None else -1e9)
            note_duration(30.0 - (now - last_enemy_damage))

        if abs(slice_dt - float(max_dt)) <= epsilon:
            note_duration(self._minimum_potential_cycle_time(world))

        return max(epsilon, min(slice_dt, float(max_dt)))

    def _log_event(self, event: str, **fields: Any) -> None:
        if not self.event_logging_enabled:
            return
        log_sim_event(self.logger, event, **fields)

    @staticmethod
    def _event_payload_id(payload: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            normalized = str(value)
            if normalized:
                return normalized
        return None

    def _emit_combat_event(self, event: str, *, tick: int, at: float, payload: dict[str, Any]) -> None:
        if self._combat_event_sink is None:
            return
        source_id = self._event_payload_id(payload, "source", "source_id", "source_ship_id") or ""
        target_id = self._event_payload_id(payload, "target", "target_id", "target_ship_id")
        module_id = self._event_payload_id(payload, "module", "module_id", "source_module_id")
        combat_event = CombatEvent(
            tick=int(tick),
            at=float(at),
            kind=str(event),
            source_id=source_id,
            target_id=target_id,
            module_id=module_id,
            rng_seed=int(self._event_rng_seed),
            rng_counter=int(self._event_rng_counter),
            payload=dict(payload),
        )
        self._event_rng_counter += 1
        self._combat_event_sink(combat_event)

    @classmethod
    def _normalize_merge_value(cls, value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, (list, tuple, set)):
            return tuple(cls._normalize_merge_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), cls._normalize_merge_value(v)) for k, v in value.items()))
        return value

    def _queue_merged_event(
        self,
        event: str,
        merge_fields: dict[str, Any],
        sum_fields: dict[str, float] | None = None,
        count: int = 1,
    ) -> None:
        if not self.event_logging_enabled and self._combat_event_sink is None:
            return
        key = (event,) + tuple(
            (k, self._normalize_merge_value(v))
            for k, v in sorted(merge_fields.items())
        )
        bucket = self._merged_event_buckets.get(key)
        if bucket is None:
            bucket = {
                "event": event,
                "tick": int(self._current_event_tick),
                "at": float(self._current_event_at),
                "merge_fields": dict(merge_fields),
                "sum_fields": {},
                "count": 0,
            }
            self._merged_event_buckets[key] = bucket
        bucket["count"] = int(bucket["count"]) + max(1, int(count))
        if sum_fields:
            sums = bucket["sum_fields"]
            for field, value in sum_fields.items():
                sums[field] = float(sums.get(field, 0.0)) + float(value)

    def _flush_merged_events(self, window_start: float | None = None, window_end: float | None = None) -> None:
        if not self._merged_event_buckets:
            return
        for bucket in self._merged_event_buckets.values():
            payload = dict(bucket["merge_fields"])
            event_count = int(bucket.get("count", 0))
            if event_count > 1:
                payload["count"] = event_count
            for field, value in bucket.get("sum_fields", {}).items():
                payload[field] = float(value)
            if window_start is not None and window_end is not None:
                payload["window_start"] = float(window_start)
                payload["window_end"] = float(window_end)
            event_name = str(bucket["event"])
            self._log_event(event_name, **payload)
            self._emit_combat_event(
                event_name,
                tick=int(bucket.get("tick", self._current_event_tick)),
                at=float(bucket.get("at", window_end if window_end is not None else self._current_event_at)),
                payload=payload,
            )
        self._merged_event_buckets.clear()

    def _advance_merge_window(self, now: float) -> None:
        window = max(0.1, float(self.event_merge_window_sec))
        if self._merge_window_end_time is None or self._merge_window_start_time is None:
            self._merge_window_start_time = float(now)
            self._merge_window_end_time = float(now) + window
            return
        while now >= self._merge_window_end_time:
            self._flush_merged_events(self._merge_window_start_time, self._merge_window_end_time)
            self._merge_window_start_time = self._merge_window_end_time
            self._merge_window_end_time = self._merge_window_start_time + window

    def flush_pending_events(self) -> None:
        self._flush_merged_events(self._merge_window_start_time, self._merge_window_end_time)

    def run(self, world: WorldState, dt: float) -> None:
        self._projected_cycle_starts_this_tick.clear()
        self._prune_cycle_effect_snapshots(world)
        self._refresh_alive_runtime_ship_ids(world)
        step_end = float(world.now)
        step_start = max(0.0, step_end - max(0.0, float(dt)))
        self._decision_reference_time = step_start
        self._current_event_tick = int(world.tick)
        self._current_event_at = float(step_end)
        event_stream_enabled = self.event_logging_enabled or self._combat_event_sink is not None
        if event_stream_enabled:
            self._advance_merge_window(world.now)
        else:
            self._merged_event_buckets.clear()
            self._merge_window_start_time = None
            self._merge_window_end_time = None

        started = time.perf_counter()
        self._update_ecm_restrictions(world, now=step_start)
        self._log_hotspot("combat.update_ecm_restrictions", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._process_due_timer_events(world, current_time=step_start)
        self._log_hotspot("combat.process_due_timer_events", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._advance_target_locks(world, dt, now=step_start)
        self._log_hotspot("combat.advance_target_locks", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        if self._update_module_states(world, dt, now=step_start):
            self._mark_pyfa_remote_inputs_dirty()
        self._log_hotspot("combat.update_module_states", started, tick=int(world.tick), dt=dt)

        started = time.perf_counter()
        self._sync_dynamic_bubble_fields(world)
        self._log_hotspot("combat.sync_bubbles", started, tick=int(world.tick), bubbles=len(world.bubble_fields))

        reusable_cached_profiles: list[tuple[Any, ShipProfile]] = []
        can_restore_cached_pyfa_bases = False
        if (not self._pyfa_remote_inputs_dirty) and self._cached_pyfa_remote_inputs_available():
            can_restore_cached_pyfa_bases, remote_recollect_required, reusable_cached_profiles = self._validate_cached_pyfa_base_profiles(world)
            if remote_recollect_required:
                self._mark_pyfa_remote_inputs_dirty()
                can_restore_cached_pyfa_bases = False
                reusable_cached_profiles = []

        reuse_remote_pyfa_inputs = (not self._pyfa_remote_inputs_dirty) and self._cached_pyfa_remote_inputs_available()

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs:
            command_boosters = self._cached_command_booster_snapshots or {}
        else:
            command_boosters = self._collect_command_booster_snapshots(world)
        self._log_hotspot("combat.collect_command_boosters", started, tick=int(world.tick), ships=len(command_boosters))

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs:
            projected_sources = self._cached_projected_source_snapshots or {}
        else:
            projected_sources = self._collect_projected_source_snapshots(world, command_boosters)
            self._cached_command_booster_snapshots = command_boosters
            self._cached_projected_source_snapshots = projected_sources
            self._pyfa_remote_inputs_dirty = False
        self._log_hotspot("combat.collect_projected_sources", started, tick=int(world.tick), ships=len(projected_sources))

        started = time.perf_counter()
        if reuse_remote_pyfa_inputs and can_restore_cached_pyfa_bases:
            for ship, cached_profile in reusable_cached_profiles:
                ship.profile = cached_profile
        else:
            self._refresh_effective_runtimes_from_pyfa(world, command_boosters, projected_sources)
        self._log_hotspot("combat.refresh_effective_runtimes", started, tick=int(world.tick), ships=len(world.ships))

        started = time.perf_counter()
        projected = self._collect_projected_impacts(world, dt)
        self._log_hotspot(
            "combat.collect_projected_impacts",
            started,
            tick=int(world.tick),
            targets=sum(len(v) for v in projected.values()),
        )

        started = time.perf_counter()
        effective_profiles: dict[str, ShipProfile] = {}
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            effective_profiles[ship.ship_id] = self._effective_profile(ship, projected)
        for ship in world.ships.values():
            if not ship.vital.alive:
                continue
            effective = effective_profiles.get(ship.ship_id)
            if effective is not None:
                ship.profile = effective
        self._log_hotspot("combat.apply_effective_profiles", started, tick=int(world.tick), ships=len(effective_profiles))

        self._update_squad_prelocks(world, dt, effective_profiles)

        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_combat_suppressed(source):
                continue
            for module in source.runtime.modules:
                metadata = self._module_static_metadata(module)
                if module.state != module.state.ACTIVE:
                    continue
                if metadata.is_command_burst or metadata.is_burst_jammer:
                    continue
                if (
                    metadata.cycle_time > 0.0
                    and self._uses_cycle_start_projected_application(metadata)
                    and not self._projected_cycle_started_this_tick(source.ship_id, module.module_id)
                ):
                    continue

                cycle_target_snapshots = self._module_cycle_snapshots_for(source.ship_id, module.module_id)
                if not cycle_target_snapshots and not metadata.is_smart_bomb:
                    continue

                for effect_index, effect in metadata.projected_effects:

                    targets: list[tuple[Any, CycleTargetSnapshot, float]] = []
                    if metadata.is_smart_bomb:
                        for target_id, target_snapshot in cycle_target_snapshots.items():
                            target = world.ships.get(target_id)
                            if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target):
                                continue
                            strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                            if strength > 0.0:
                                targets.append((target, target_snapshot, strength))
                    else:
                        tgt_id = source.combat.projected_targets.get(module.module_id)
                        if not tgt_id:
                            continue
                        target = world.ships.get(tgt_id)
                        if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target):
                            continue
                        got_snapshot = cycle_target_snapshots.get(tgt_id)
                        if got_snapshot is None:
                            continue
                        target_snapshot = got_snapshot
                        if False:
                            continue
                        strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                        if strength <= 0.0:
                            continue
                        targets.append((target, target_snapshot, strength))

                    for target, target_snapshot, strength in targets:
                        target_profile = effective_profiles.get(target.ship_id) or target.profile
                        damage_factor_override = self._cycle_effect_damage_factor(
                            source=source,
                            target=target,
                            target_profile=target_profile,
                            effect=effect,
                            effect_index=effect_index,
                            target_snapshot=target_snapshot,
                            strength=strength,
                        )
                        hp_before = target.vital.shield + target.vital.armor + target.vital.structure
                        (
                            shield_repaired,
                            armor_repaired,
                            cap_drained,
                            em_damage,
                            thermal_damage,
                            kinetic_damage,
                            explosive_damage,
                            total_damage,
                        ) = self._apply_projected_cycle_effects(
                            world=world,
                            source=source,
                            target=target,
                            target_profile=target_profile,
                            effect=effect,
                            strength=strength,
                            damage_factor_override=damage_factor_override,
                            module_id=module.module_id,
                        )
                        hp_after = target.vital.shield + target.vital.armor + target.vital.structure
                        applied_damage = max(0.0, hp_before - hp_after)
                        if (
                            applied_damage > 0.0
                            and source.team != target.team
                            and metadata.is_weapon
                        ):
                            target.combat.last_enemy_weapon_damaged_at = float(world.now)
                        if (
                            shield_repaired > 0.0
                            or armor_repaired > 0.0
                            or cap_drained > 0.0
                            or total_damage > 0.0
                        ):
                            self._add_projected_cycle_total(
                                source_ship_id=source.ship_id,
                                module_id=module.module_id,
                                target_ship_id=target.ship_id,
                                shield_repaired=shield_repaired,
                                armor_repaired=armor_repaired,
                                cap_drained=cap_drained,
                                em_damage=em_damage,
                                thermal_damage=thermal_damage,
                                kinetic_damage=kinetic_damage,
                                explosive_damage=explosive_damage,
                                total_damage=total_damage,
                            )
                    if metadata.is_smart_bomb:
                        self._destroy_projectiles_in_area(
                            world,
                            center=Vector2(source.nav.position.x, source.nav.position.y),
                            radius_m=max(0.0, float(effect.range_m or 0.0)),
                            damage=self._effect_damage_tuple(effect),
                        )
                        self._destroy_bubbles_in_area(
                            world,
                            center=Vector2(source.nav.position.x, source.nav.position.y),
                            radius_m=max(0.0, float(effect.range_m or 0.0)),
                            damage=self._effect_damage_tuple(effect),
                        )

        self._advance_projectiles(world, dt)

        if self.detailed_logging and self.logger is not None:
            total_impacts = sum(len(v) for v in projected.values())
            self.logger.debug(f"combat_tick dt={dt:.4f} projected_impacts={total_impacts}")

        for ship in world.ships.values():
            if not ship.vital.alive:
                continue

            ship_profile = effective_profiles.get(ship.ship_id, ship.profile)
            self._sync_vital_max_with_profile(ship, ship_profile)
            if (
                self.detailed_logging
                and self.logger is not None
                and ship.runtime is not None
                and ship.ship_id not in self._diag_logged_ships
            ):
                unmodeled = ship.runtime.diagnostics.get("unmodeled_modules", [])
                if unmodeled:
                    self.logger.debug(
                        f"fit_diagnostics ship={ship.ship_id} unmodeled_modules={unmodeled}"
                    )
                self._diag_logged_ships.add(ship.ship_id)
            ship.nav.max_speed = ship_profile.max_speed

            ship.vital.cap_max = ship_profile.max_cap
            if ship.vital.cap > ship.vital.cap_max:
                ship.vital.cap = ship.vital.cap_max

            ship.vital.cap = self._resolve_cap_recharge(
                cap_now=ship.vital.cap,
                cap_max=ship.vital.cap_max,
                recharge_time=ship_profile.cap_recharge_time,
                dt=dt,
            )

            if self._ship_combat_suppressed(ship):
                ship.combat.current_target = None
                continue

            current_target_id = ship.combat.current_target
            if current_target_id:
                current_target = world.ships.get(current_target_id)
                if (
                    current_target is None
                    or not current_target.vital.alive
                    or self._ship_hidden_from_targeting(current_target)
                    or current_target.team == ship.team
                    or not self._target_within_lock_range(ship, current_target, source_profile=ship_profile)
                ):
                    ship.combat.current_target = None

            if not ship.combat.current_target:
                queue = list(world.squad_focus_queues.get(self._focus_key(ship.team, ship.squad_id), []))
                for candidate_id in queue:
                    candidate = world.ships.get(candidate_id)
                    if (
                        candidate is None
                        or not candidate.vital.alive
                        or self._ship_hidden_from_targeting(candidate)
                        or candidate.team == ship.team
                        or not self._target_within_lock_range(ship, candidate, source_profile=ship_profile)
                    ):
                        continue
                    ship.combat.current_target = candidate_id
                    break

        started = time.perf_counter()
        self._process_due_timer_events(world, current_time=step_end)
        self._log_hotspot("combat.commit_due_timer_events", started, tick=int(world.tick), dt=dt)
        self._decision_reference_time = None

        if event_stream_enabled:
            self._advance_merge_window(world.now)
