from __future__ import annotations

import sys

from .combat_common import *  # noqa: F403


class RuntimeProjectionMixin:
    @staticmethod
    def _compat_combat_core_function(name: str, fallback):
        combat_core = sys.modules.get("eve_sim.systems.combat_core")
        if combat_core is None:
            return fallback
        return getattr(combat_core, name, fallback)

    def _get_runtime_resolve_cache_key(self, runtime, command_boosters, projected_sources):
        resolver = self._compat_combat_core_function("get_runtime_resolve_cache_key", get_runtime_resolve_cache_key)
        return resolver(runtime, command_boosters, projected_sources)

    def _resolve_runtime_from_pyfa_runtime(self, runtime, command_boosters, projected_sources):
        resolver = self._compat_combat_core_function("resolve_runtime_from_pyfa_runtime", resolve_runtime_from_pyfa_runtime)
        return resolver(runtime, command_boosters, projected_sources)

    @staticmethod
    def _copy_profile_passthrough_fields(base: ShipProfile, target: ShipProfile) -> None:
        for attr in _PROFILE_PASSTHROUGH_ATTRS:
            setattr(target, attr, getattr(base, attr, getattr(target, attr, 0.0)))

    @staticmethod
    def _copy_formula_base_fields(base: ShipProfile, target: ShipProfile) -> None:
        for attr in (
            "dps",
            "volley",
            "optimal",
            "falloff",
            "tracking",
            "sig_radius",
            "scan_resolution",
            "max_target_range",
            "max_speed",
            "max_cap",
            "cap_recharge_time",
            "shield_hp",
            "armor_hp",
            "structure_hp",
            "rep_amount",
            "rep_cycle",
            "mass",
            "agility",
            "warp_speed_au_s",
            "warp_capacitor_need",
            "max_warp_distance_au",
            "disallow_assistance",
            "warp_bubble_immune",
            "is_shuttle",
        ):
            setattr(target, attr, getattr(base, attr))

    def _apply_runtime_projected_impacts(self, base: ShipProfile, impacts: list[ProjectedImpact], runtime=None) -> ShipProfile:
        penalty_context = None
        weapon_penalty_context = None
        if runtime is not None:
            raw_context = getattr(runtime, "diagnostics", {}).get("pyfa_ship_attribute_penalty_context")
            if isinstance(raw_context, dict):
                penalty_context = raw_context
            raw_weapon_context = getattr(runtime, "diagnostics", {}).get("pyfa_weapon_attribute_penalty_context")
            if isinstance(raw_weapon_context, list):
                weapon_penalty_context = raw_weapon_context
        effective = self.runtime.apply_projected_effects(replace(base), impacts, base_penalty_context=penalty_context)
        self._apply_weighted_weapon_projection_context(effective, impacts, weapon_penalty_context)
        self._copy_profile_passthrough_fields(base, effective)
        return effective

    def _apply_weighted_weapon_projection_context(
        self,
        effective: ShipProfile,
        impacts: list[ProjectedImpact],
        weapon_penalty_context: list[dict[str, Any]] | None,
    ) -> None:
        if not weapon_penalty_context:
            return

        grouped_multipliers: dict[str, dict[str, list[float]]] = {
            "optimal": {},
            "falloff": {},
            "tracking": {},
        }
        affected_keys: set[str] = set()
        for impact in impacts:
            effect = impact.effect
            strength = max(0.0, min(1.0, float(impact.strength)))
            mult_groups = getattr(effect, "projected_mult_groups", {}) or {}
            for key in ("optimal", "falloff", "tracking"):
                value = effect.projected_mult.get(key)
                if value is None:
                    continue
                group_name = mult_groups.get(key, "default")
                if group_name is None:
                    group_name = f"__unstacked__:{len(grouped_multipliers[key])}"
                grouped_multipliers[key].setdefault(str(group_name), []).append(1.0 + (float(value) - 1.0) * strength)
                affected_keys.add(key)

        if not affected_keys:
            return

        attr_map = {
            "optimal": "optimal",
            "falloff": "falloff",
            "tracking": "tracking",
        }
        for key in affected_keys:
            weighted_total = 0.0
            total_weight = 0.0
            for entry in weapon_penalty_context:
                if str(entry.get("kind", "")) != "gunnery":
                    continue
                weight = float(entry.get("weight", 0.0) or 0.0)
                if weight <= 0.0:
                    continue
                context = entry.get(key)
                if isinstance(context, dict):
                    current_value = float(context.get("current", getattr(effective, attr_map[key])) or 0.0)
                else:
                    current_value = float(getattr(effective, attr_map[key]) or 0.0)
                    context = None
                value = self.runtime._apply_penalized_projection(
                    current_value,
                    0.0,
                    grouped_multipliers[key],
                    context,
                )
                weighted_total += weight * value
                total_weight += weight
            if total_weight <= 0.0:
                continue
            resolved_value = weighted_total / total_weight
            if key == "optimal":
                effective.optimal = max(1.0, resolved_value)
            elif key == "falloff":
                effective.falloff = max(1.0, resolved_value)
            elif key == "tracking":
                effective.tracking = max(0.0001, resolved_value)

    def _fallback_unprojected_profile(self, ship) -> ShipProfile:
        runtime = getattr(ship, "runtime", None)
        if runtime is None:
            return replace(ship.profile)

        fallback = replace(ship.profile)
        fit_base = replace(self.pyfa.build_profile(ship.fit))
        runtime_base = replace(self.runtime.compute_base_profile(runtime))
        self._copy_formula_base_fields(runtime_base, fallback)
        for attr in (
            "sensor_strength_gravimetric",
            "sensor_strength_ladar",
            "sensor_strength_magnetometric",
            "sensor_strength_radar",
            "shield_resonance_em",
            "shield_resonance_thermal",
            "shield_resonance_kinetic",
            "shield_resonance_explosive",
            "armor_resonance_em",
            "armor_resonance_thermal",
            "armor_resonance_kinetic",
            "armor_resonance_explosive",
            "structure_resonance_em",
            "structure_resonance_thermal",
            "structure_resonance_kinetic",
            "structure_resonance_explosive",
            "missile_explosion_radius",
            "missile_explosion_velocity",
            "missile_max_range",
            "warp_speed_au_s",
            "warp_capacitor_need",
            "max_warp_distance_au",
            "disallow_assistance",
            "warp_bubble_immune",
            "is_shuttle",
        ):
            setattr(fallback, attr, getattr(fit_base, attr))
        return fallback

    @staticmethod
    def _local_runtime_state_signature(runtime) -> tuple[tuple[str, str], ...] | None:
        blueprint = runtime.diagnostics.get("pyfa_blueprint")
        if not isinstance(blueprint, dict):
            return None
        return _runtime_local_profile_state_signature(runtime)

    def _local_runtime_state_signature_from_metadata(self, runtime) -> tuple[tuple[str, str], ...] | None:
        blueprint = runtime.diagnostics.get("pyfa_blueprint")
        if not isinstance(blueprint, dict):
            return None
        observed_signature = runtime.diagnostics.get("runtime_observed_module_state_signature")
        cached_observed_signature = runtime.diagnostics.get("runtime_local_state_observed_signature")
        cached_signature = runtime.diagnostics.get("runtime_local_state_signature")
        if (
            isinstance(observed_signature, tuple)
            and cached_observed_signature == observed_signature
            and isinstance(cached_signature, tuple)
        ):
            return cached_signature
        tracked_ids = runtime.diagnostics.get("runtime_local_stateful_module_ids")
        if isinstance(tracked_ids, tuple):
            tracked_id_set = {str(module_id) for module_id in tracked_ids}
            signature = tuple(
                (str(module.module_id), str(module.state.value or "ONLINE").upper())
                for module in runtime.modules
                if str(module.module_id) in tracked_id_set
            )
        else:
            signature = tuple(
                (str(module.module_id), str(module.state.value or "ONLINE").upper())
                for module in runtime.modules
                if self._module_static_metadata(module).affects_local_pyfa_profile
            )
        runtime.diagnostics["runtime_local_state_signature"] = signature
        if isinstance(observed_signature, tuple):
            runtime.diagnostics["runtime_local_state_observed_signature"] = observed_signature
        return signature

    @staticmethod
    def _runtime_observed_module_state_signature(runtime) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(module.module_id), str(module.state.value or "ONLINE").upper())
            for module in runtime.modules
        )

    def _mark_pyfa_remote_inputs_dirty(self) -> None:
        self._pyfa_remote_inputs_dirty = True

    def _refresh_alive_runtime_ship_ids(self, world: WorldState) -> None:
        current_alive_runtime_ship_ids = {
            ship.ship_id
            for ship in world.ships.values()
            if ship.vital.alive and ship.runtime is not None and not self._ship_combat_suppressed(ship, float(world.now))
        }
        if current_alive_runtime_ship_ids != self._alive_runtime_ship_ids:
            self._alive_runtime_ship_ids = current_alive_runtime_ship_ids
            self._mark_pyfa_remote_inputs_dirty()
            self._mark_all_repair_queues_dirty()

    def _cached_pyfa_remote_inputs_available(self) -> bool:
        return self._cached_command_booster_snapshots is not None and self._cached_projected_source_snapshots is not None

    @staticmethod
    def _module_tags(module) -> frozenset[str]:
        return frozenset(str(tag) for tag in (getattr(module, "tags", ()) or ()))

    def _module_static_metadata(self, module) -> ModuleStaticMetadata:
        key = id(module)
        cached_entry = self._module_static_metadata_by_object_id.get(key)
        if cached_entry is not None:
            cached_ref, cached_metadata = cached_entry
            if cached_ref() is module:
                return cached_metadata
            self._module_static_metadata_by_object_id.pop(key, None)

        tags = self._module_tags(module)
        active_effects = tuple(
            effect
            for effect in module.effects
            if str(effect.state_required.value).upper() == "ACTIVE"
        )
        projected_effects = tuple(
            (effect_index, effect)
            for effect_index, effect in enumerate(module.effects)
            if effect.effect_class == EffectClass.PROJECTED
        )
        has_projected = bool(projected_effects)
        target_side = "ally" if "support" in tags and "hostile" not in tags else "enemy"
        is_command_burst = "command_burst" in tags
        is_smart_bomb = "smart_bomb" in tags
        is_burst_jammer = "burst_jammer" in tags
        is_area_effect = "area_effect" in tags
        is_cap_booster = "cap_booster" in tags
        is_propulsion = "propulsion" in tags
        is_damage_control = "damage_control" in tags
        is_hardener = "hardener" in tags
        is_cap_warfare = "cap_warfare" in tags
        is_target_ewar = "target_ewar" in tags
        is_ecm = "ecm" in tags
        is_weapon = "weapon" in tags
        is_missile_weapon = any(
            float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5
            for _effect_index, effect in projected_effects
        )
        is_bomb_launcher = any(
            float(effect.projected_add.get("weapon_is_bomb", 0.0) or 0.0) > 0.5
            for _effect_index, effect in projected_effects
        )
        has_projected_rep = "remote_repair" in tags
        repair_layers: list[str] = []
        if has_projected_rep:
            for _effect_index, effect in projected_effects:
                if float(effect.projected_add.get("shield_rep", 0.0) or 0.0) > 0.0 and "shield" not in repair_layers:
                    repair_layers.append("shield")
                if float(effect.projected_add.get("armor_rep", 0.0) or 0.0) > 0.0 and "armor" not in repair_layers:
                    repair_layers.append("armor")
                if float(effect.projected_add.get("structure_rep", 0.0) or 0.0) > 0.0 and "structure" not in repair_layers:
                    repair_layers.append("structure")
        is_offensive_ewar = "offensive_ewar" in tags or is_ecm or is_cap_warfare
        supports_formula_projected_profile = (
            has_projected
            and any(effect.projected_mult or effect.projected_add for _effect_index, effect in projected_effects)
            and all(self._effect_supports_runtime_formula_projection(effect) for _effect_index, effect in projected_effects)
        )
        uses_pyfa_projected_profile = (
            has_projected
            and not is_command_burst
            and not is_smart_bomb
            and not is_burst_jammer
            and not is_ecm
            and not is_weapon
            and not has_projected_rep
            and not is_cap_warfare
            and not supports_formula_projected_profile
        )

        if is_command_burst:
            decision_rule = ModuleDecisionRule(
                rule_id="area_command_burst",
                activation_mode="always",
                target_mode="none",
            )
        elif is_smart_bomb:
            decision_rule = ModuleDecisionRule(
                rule_id="area_smart_bomb",
                activation_mode="enemy_in_area",
                target_mode="none",
                cap_threshold=0.15,
            )
        elif is_burst_jammer:
            decision_rule = ModuleDecisionRule(
                rule_id="area_burst_jammer",
                activation_mode="enemy_in_area",
                target_mode="none",
                cap_threshold=0.15,
            )
        elif is_weapon:
            decision_rule = ModuleDecisionRule(
                rule_id="weapon_focus_only",
                activation_mode="weapon_focus_only",
                target_mode="weapon_focus_prefocus",
            )
        elif has_projected:
            if has_projected_rep:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_remote_repair",
                    activation_mode="always",
                    target_mode="ally_repair_queue",
                )
            elif is_offensive_ewar:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_offensive_ewar",
                    activation_mode="cap_min",
                    target_mode="enemy_random",
                    cap_threshold=0.15,
                )
            elif is_target_ewar:
                decision_rule = ModuleDecisionRule(
                    rule_id="weapon_focus_only",
                    activation_mode="weapon_focus_only",
                    target_mode="weapon_focus_prefocus",
                )
            elif target_side == "ally":
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_support_generic",
                    activation_mode="always",
                    target_mode="ally_nearest",
                )
            else:
                decision_rule = ModuleDecisionRule(
                    rule_id="projected_hostile_generic",
                    activation_mode="never",
                    target_mode="none",
                )
        elif is_propulsion:
            decision_rule = ModuleDecisionRule(
                rule_id="local_propulsion",
                activation_mode="propulsion_command",
                target_mode="none",
            )
        elif is_damage_control:
            decision_rule = ModuleDecisionRule(
                rule_id="local_damage_control",
                activation_mode="recent_enemy_weapon_damage",
                target_mode="none",
            )
        elif is_hardener:
            decision_rule = ModuleDecisionRule(
                rule_id="local_hardener",
                activation_mode="cap_or_low_hp",
                target_mode="none",
                cap_threshold=0.10,
            )
        elif is_cap_booster:
            decision_rule = ModuleDecisionRule(
                rule_id="local_cap_booster",
                activation_mode="cap_max",
                target_mode="none",
                cap_threshold=0.85,
            )
        else:
            decision_rule = ModuleDecisionRule(
                rule_id="local_active_default",
                activation_mode="never",
                target_mode="none",
            )

        projected_max_range: float | None = None
        if has_projected:
            computed_projected_max_range = 0.0
            for _effect_index, effect in projected_effects:
                effect_max_range = self._projected_max_range(effect)
                if effect_max_range <= 0.0:
                    computed_projected_max_range = 0.0
                    break
                computed_projected_max_range = max(computed_projected_max_range, effect_max_range)
            projected_max_range = computed_projected_max_range

        metadata = ModuleStaticMetadata(
            active_effects=active_effects,
            projected_effects=projected_effects,
            cycle_cost=sum(max(0.0, effect.cap_need) for effect in active_effects),
            cycle_time=min((max(0.1, effect.cycle_time) for effect in active_effects if effect.cycle_time > 0), default=0.0),
            reactivation_delay=max((max(0.0, float(getattr(effect, "reactivation_delay", 0.0) or 0.0)) for effect in active_effects), default=0.0),
            has_projected=has_projected,
            projected_max_range=projected_max_range,
            target_side=target_side,
            is_command_burst=is_command_burst,
            is_smart_bomb=is_smart_bomb,
            is_burst_jammer=is_burst_jammer,
            is_area_effect=is_area_effect,
            is_weapon=is_weapon,
            is_missile_weapon=is_missile_weapon,
            is_bomb_launcher=is_bomb_launcher,
            has_projected_rep=has_projected_rep,
            is_cap_warfare=is_cap_warfare,
            is_target_ewar=is_target_ewar,
            is_ecm=is_ecm,
            uses_pyfa_projected_profile=uses_pyfa_projected_profile,
            is_hardener=is_hardener,
            is_cap_booster=is_cap_booster,
            is_propulsion=is_propulsion,
            is_damage_control=is_damage_control,
            affects_local_pyfa_profile=_module_affects_local_pyfa_profile(module),
            repair_layers=tuple(repair_layers),
            decision_rule=decision_rule,
        )
        module_ref = weakref.ref(
            module,
            lambda ref, *, cache=self._module_static_metadata_by_object_id, cache_key=key: (
                cache.pop(cache_key, None)
                if cache.get(cache_key, (None, None))[0] is ref
                else None
            ),
        )
        self._module_static_metadata_by_object_id[key] = (module_ref, metadata)
        return metadata

    @staticmethod
    def _effect_supports_runtime_formula_projection(effect) -> bool:
        projected_mult_keys = {str(key) for key in effect.projected_mult.keys()}
        projected_add_keys = {str(key) for key in effect.projected_add.keys()}
        modeled_keys = projected_mult_keys | projected_add_keys
        if not modeled_keys:
            return False
        return modeled_keys.issubset(_FORMULA_PROJECTED_KEYS)

    @staticmethod
    def _round_projection_signature_value(value: float) -> float:
        return round(float(value or 0.0), 6)

    @classmethod
    def _projected_effect_signature(cls, effect) -> tuple[Any, ...]:
        return (
            str(getattr(effect, "name", "") or ""),
            str(getattr(getattr(effect, "effect_class", None), "value", getattr(effect, "effect_class", "")) or ""),
            str(getattr(getattr(effect, "state_required", None), "value", getattr(effect, "state_required", "")) or ""),
            cls._round_projection_signature_value(float(getattr(effect, "range_m", 0.0) or 0.0)),
            cls._round_projection_signature_value(float(getattr(effect, "falloff_m", 0.0) or 0.0)),
            tuple(
                sorted(
                    (str(key), cls._round_projection_signature_value(float(value or 0.0)))
                    for key, value in effect.projected_mult.items()
                )
            ),
            tuple(
                sorted(
                    (str(key), cls._round_projection_signature_value(float(value or 0.0)))
                    for key, value in effect.projected_add.items()
                )
            ),
            tuple(
                sorted(
                    (str(key), None if value is None else str(value))
                    for key, value in getattr(effect, "projected_mult_groups", {}).items()
                )
            ),
            tuple(getattr(effect, "projected_signature", ()) or ()),
        )

    @classmethod
    def _projected_impact_signature(cls, impacts: list[ProjectedImpact]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                str(impact.source_ship_id or ""),
                str(impact.target_ship_id or ""),
                cls._round_projection_signature_value(float(impact.strength or 0.0)),
                cls._projected_effect_signature(impact.effect),
            )
            for impact in impacts
        )

    @classmethod
    def _projected_module_runtime_signature(
        cls,
        module,
        module_blueprint: dict[str, Any] | None,
        module_state: str,
        active_effect_indices: set[int] | None = None,
    ) -> tuple[Any, ...]:
        projected_effect_signatures: list[tuple[Any, ...]] = []
        for effect_index, effect in enumerate(module.effects):
            if effect.effect_class != EffectClass.PROJECTED:
                continue
            if active_effect_indices and effect_index not in active_effect_indices:
                continue
            projected_effect_signatures.append(cls._projected_effect_signature(effect))

        blueprint_signature = (
            str((module_blueprint or {}).get("module_name", "") or ""),
            str((module_blueprint or {}).get("charge_name", "") or ""),
            tuple(sorted(str(name) for name in ((module_blueprint or {}).get("effect_names") or ()))),
        )
        return (
            "module_projection",
            blueprint_signature,
            str(getattr(module, "group", "") or ""),
            str(module_state or "ONLINE").upper(),
            tuple(projected_effect_signatures),
        )

    def _runtime_module_metadata_list(self, runtime) -> tuple[ModuleStaticMetadata, ...]:
        cached = runtime.diagnostics.get("runtime_module_static_metadata")
        if isinstance(cached, tuple) and len(cached) == len(runtime.modules):
            return cached
        metadata_list = tuple(self._module_static_metadata(module) for module in runtime.modules)
        runtime.diagnostics["runtime_module_static_metadata"] = metadata_list
        return metadata_list

    def _runtime_module_buckets(self, runtime) -> RuntimeModuleBuckets:
        cached = runtime.diagnostics.get("runtime_module_buckets")
        if isinstance(cached, RuntimeModuleBuckets) and cached.module_count == len(runtime.modules):
            return cached

        controlled_ids = runtime.diagnostics.get("runtime_controlled_module_ids")
        controlled_id_set = {str(module_id) for module_id in controlled_ids} if isinstance(controlled_ids, tuple) else None
        controlled_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        command_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        runtime_projected_entries: list[tuple[Any, ModuleStaticMetadata]] = []
        pyfa_projected_entries: list[tuple[Any, ModuleStaticMetadata]] = []

        for module in runtime.modules:
            module_id = str(module.module_id)
            if controlled_id_set is not None and module_id not in controlled_id_set:
                if not any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects):
                    continue
            metadata = self._module_static_metadata(module)
            if controlled_id_set is not None:
                if module_id in controlled_id_set:
                    controlled_entries.append((module, metadata))
            elif metadata.active_effects:
                controlled_entries.append((module, metadata))
            if metadata.is_command_burst:
                command_entries.append((module, metadata))
            if metadata.projected_effects:
                if metadata.uses_pyfa_projected_profile:
                    pyfa_projected_entries.append((module, metadata))
                else:
                    runtime_projected_entries.append((module, metadata))

        buckets = RuntimeModuleBuckets(
            module_count=len(runtime.modules),
            controlled_entries=tuple(controlled_entries),
            command_entries=tuple(command_entries),
            runtime_projected_entries=tuple(runtime_projected_entries),
            pyfa_projected_entries=tuple(pyfa_projected_entries),
        )
        runtime.diagnostics["runtime_module_buckets"] = buckets
        return buckets

    def _runtime_controlled_module_ids(self, runtime) -> tuple[str, ...]:
        return runtime_controlled_module_ids(runtime, self._runtime_module_buckets(runtime).controlled_entries)

    def _runtime_controlled_entry_lookup(self, runtime) -> dict[str, tuple[Any, ModuleStaticMetadata]]:
        controlled_entries = self._runtime_module_buckets(runtime).controlled_entries
        controlled_ids = self._runtime_controlled_module_ids(runtime)
        return runtime_controlled_entry_lookup(runtime, controlled_entries, controlled_ids)

    def _runtime_decision_rule_groups(self, runtime) -> dict[str, dict[str, tuple[str, ...]]]:
        return runtime_decision_rule_groups(runtime, self._runtime_module_buckets(runtime).controlled_entries)

    def _validate_cached_pyfa_base_profiles(
        self,
        world: WorldState,
    ) -> tuple[bool, bool, list[tuple[Any, ShipProfile]]]:
        reusable_profiles: list[tuple[Any, ShipProfile]] = []
        reusable = True
        remote_recollect_required = False

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            local_signature = self._local_runtime_state_signature_from_metadata(ship.runtime)
            cached_local_signature = ship.runtime.diagnostics.get("pyfa_local_state_signature")
            cached_base_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
            if local_signature != cached_local_signature or not isinstance(cached_base_profile, ShipProfile):
                reusable = False
                if (
                    local_signature != cached_local_signature
                    and self._runtime_has_active_pyfa_remote_inputs(ship.runtime)
                ):
                    remote_recollect_required = True
                continue

            reusable_profiles.append((ship, cached_base_profile))

        return reusable, remote_recollect_required, reusable_profiles

    @staticmethod
    def _command_snapshot_list_signature(snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        signature: list[tuple[Any, ...]] = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            state_raw = snapshot.get("state_by_module_id")
            state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
            signature.append(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple((str(module_id), str(state)) for module_id, state in state_by_module_id.items()),
                )
            )
        return tuple(signature)

    @classmethod
    def _projected_snapshot_list_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        return shared_projected_snapshot_list_signature(
            snapshots,
            module_signature_builder=cls._projected_snapshot_module_signature,
            bucket_m=_PYFA_PROJECTION_RANGE_BUCKET_M,
        )

    @classmethod
    def _projected_snapshot_legacy_module_signature(cls, snapshot: dict[str, Any]) -> tuple[Any, ...]:
        state_raw = snapshot.get("state_by_module_id")
        state_by_module_id: dict[str, Any] = state_raw if isinstance(state_raw, dict) else {}
        command_raw = snapshot.get("command_booster_snapshots")
        command_snapshots = [snap for snap in command_raw if isinstance(snap, dict)] if isinstance(command_raw, list) else []
        return (
            "legacy_source",
            str(snapshot.get("fit_key", "") or ""),
            tuple((str(module_id), str(state)) for module_id, state in state_by_module_id.items()),
            cls._command_snapshot_list_signature(command_snapshots),
        )

    @classmethod
    def _projected_snapshot_module_signature(cls, snapshot: dict[str, Any]) -> tuple[Any, ...]:
        return shared_projected_snapshot_module_signature(
            snapshot,
            legacy_builder=cls._projected_snapshot_legacy_module_signature,
        )

    def _module_affects_pyfa_remote_inputs(self, module) -> bool:
        metadata = self._module_static_metadata(module)
        return metadata.is_command_burst or metadata.uses_pyfa_projected_profile

    def _runtime_has_active_pyfa_remote_inputs(self, runtime) -> bool:
        observed_signature = runtime.diagnostics.get("runtime_observed_module_state_signature")
        cached_signature = runtime.diagnostics.get("runtime_has_active_pyfa_remote_inputs_signature")
        cached_value = runtime.diagnostics.get("runtime_has_active_pyfa_remote_inputs")
        if (
            isinstance(observed_signature, tuple)
            and cached_signature == observed_signature
            and isinstance(cached_value, bool)
        ):
            return cached_value

        buckets = self._runtime_module_buckets(runtime)
        for module, _metadata in buckets.command_entries:
            if str(module.state.value or "ONLINE").upper() not in {"ACTIVE", "OVERHEATED"}:
                continue
            runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = True
            if isinstance(observed_signature, tuple):
                runtime.diagnostics["runtime_has_active_pyfa_remote_inputs_signature"] = observed_signature
            return True
        for module, _metadata in buckets.pyfa_projected_entries:
            if str(module.state.value or "ONLINE").upper() not in {"ACTIVE", "OVERHEATED"}:
                continue
            runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = True
            if isinstance(observed_signature, tuple):
                runtime.diagnostics["runtime_has_active_pyfa_remote_inputs_signature"] = observed_signature
            return True
        runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = False
        if isinstance(observed_signature, tuple):
            runtime.diagnostics["runtime_has_active_pyfa_remote_inputs_signature"] = observed_signature
        return False

    def _reconcile_external_module_state_changes(self, world: WorldState, ship, runtime) -> bool:
        current_signature = self._runtime_observed_module_state_signature(runtime)
        cached_signature = runtime.diagnostics.get("runtime_observed_module_state_signature")
        runtime.diagnostics["runtime_observed_module_state_signature"] = current_signature
        if not isinstance(cached_signature, tuple) or cached_signature == current_signature:
            return False

        previous_states = {str(module_id): str(state) for module_id, state in cached_signature}
        current_states = {str(module_id): str(state) for module_id, state in current_signature}
        changed_module_ids = {
            module_id
            for module_id, state in current_states.items()
            if previous_states.get(module_id) != state
        } | {
            module_id
            for module_id in previous_states.keys()
            if module_id not in current_states
        }
        if not changed_module_ids:
            return False

        runtime.diagnostics.pop("runtime_local_state_signature", None)
        runtime.diagnostics.pop("runtime_has_active_pyfa_remote_inputs", None)

        pyfa_remote_inputs_dirty = False
        for module in runtime.modules:
            module_id = str(module.module_id)
            if module_id not in changed_module_ids:
                continue

            metadata = self._module_static_metadata(module)
            state_name = str(module.state.value or "ONLINE").upper()
            previous_projected_target = ship.combat.projected_targets.get(module_id)

            if state_name not in {"ACTIVE", "OVERHEATED"}:
                self._flush_projected_cycle_total(world, ship.ship_id, module, previous_projected_target)
                self._clear_module_cycle_snapshots(ship.ship_id, module_id)
                self._clear_module_cycle_timer(ship, module_id)
                self._clear_module_reactivation_timer(ship, module_id)
                if state_name == "OFFLINE":
                    ship.combat.projected_targets.pop(module_id, None)

            if metadata.affects_local_pyfa_profile:
                runtime.diagnostics.pop("pyfa_local_state_signature", None)

            if self._module_affects_pyfa_remote_inputs(module):
                pyfa_remote_inputs_dirty = True

        return pyfa_remote_inputs_dirty

    @classmethod
    def _projected_snapshot_structure_signature(cls, snapshots: list[dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(item[:-1] for item in cls._projected_snapshot_list_signature(snapshots))

    @staticmethod
    def _quantize_pyfa_projection_range(distance: float) -> float:
        safe_distance = max(0.0, float(distance or 0.0))
        if _PYFA_PROJECTION_RANGE_BUCKET_M <= 0.0:
            return safe_distance
        return math.floor(safe_distance / _PYFA_PROJECTION_RANGE_BUCKET_M) * _PYFA_PROJECTION_RANGE_BUCKET_M

    @classmethod
    def _normalized_snapshot_projection_signature(cls, snapshot: dict[str, Any]) -> tuple[str, Any]:
        return shared_normalized_snapshot_projection_signature(snapshot, bucket_m=_PYFA_PROJECTION_RANGE_BUCKET_M)

    def _pyfa_projection_snapshot_params(self, module, target_snapshot: CycleTargetSnapshot) -> tuple[str, float]:
        projected_effects = [
            effect
            for effect_index, effect in enumerate(module.effects)
            if effect.effect_class == EffectClass.PROJECTED and effect_index in target_snapshot.active_effect_indices
        ]
        if not projected_effects:
            projected_effects = [effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED]
        falloff_effects = [
            effect
            for effect in projected_effects
            if max(0.0, float(getattr(effect, "falloff_m", 0.0) or 0.0)) > 0.0
        ]
        if not falloff_effects:
            return "in_range", 0.0
        if all(target_snapshot.distance <= max(0.0, float(getattr(effect, "range_m", 0.0) or 0.0)) for effect in falloff_effects):
            return "in_range", 0.0
        return "exact_range", self._quantize_pyfa_projection_range(target_snapshot.distance)

    @staticmethod
    def _runtime_state_rank(state: ModuleState) -> int:
        return {
            ModuleState.OFFLINE: 0,
            ModuleState.ONLINE: 1,
            ModuleState.ACTIVE: 2,
            ModuleState.OVERHEATED: 3,
        }.get(state, 0)

    @classmethod
    def _runtime_module_max_state(cls, runtime: FitRuntime | None, module_id: str) -> ModuleState:
        if runtime is None:
            return ModuleState.OVERHEATED
        raw_map = runtime.diagnostics.get("pyfa_max_state_by_module_id")
        if not isinstance(raw_map, dict):
            return ModuleState.OVERHEATED
        state_name = str(raw_map.get(str(module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
        if state_name in ModuleState.__members__:
            return ModuleState[state_name]
        return ModuleState.OVERHEATED

    @classmethod
    def _clamp_runtime_state_to_pyfa_max(cls, requested_state: ModuleState, max_state: ModuleState) -> ModuleState:
        return requested_state if cls._runtime_state_rank(requested_state) <= cls._runtime_state_rank(max_state) else max_state

    @classmethod
    def _runtime_inactive_module_state(cls, runtime: FitRuntime | None, module_id: str) -> ModuleState:
        max_state = cls._runtime_module_max_state(runtime, module_id)
        if cls._runtime_state_rank(max_state) < cls._runtime_state_rank(ModuleState.ONLINE):
            return ModuleState.OFFLINE
        return ModuleState.ONLINE

    @classmethod
    def _copy_runtime_dynamic_state(cls, source_runtime, target_runtime) -> None:
        raw_max_state_map = target_runtime.diagnostics.get("pyfa_max_state_by_module_id")
        max_state_map = raw_max_state_map if isinstance(raw_max_state_map, dict) else {}
        if len(source_runtime.modules) == len(target_runtime.modules):
            for source_module, target_module in zip(source_runtime.modules, target_runtime.modules):
                target_module.module_id = source_module.module_id
                max_state_name = str(max_state_map.get(str(target_module.module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
                max_state = ModuleState[max_state_name] if max_state_name in ModuleState.__members__ else ModuleState.OVERHEATED
                target_module.state = cls._clamp_runtime_state_to_pyfa_max(source_module.state, max_state)
                if source_module.charge_capacity > 0:
                    target_module.charge_remaining = max(
                        0.0,
                        min(float(source_module.charge_remaining), float(target_module.charge_capacity)),
                    )
            return

        source_by_module_id = {module.module_id: module for module in source_runtime.modules}
        for module in target_runtime.modules:
            source_module = source_by_module_id.get(module.module_id)
            if source_module is None:
                continue
            max_state_name = str(max_state_map.get(str(module.module_id), ModuleState.OVERHEATED.value) or ModuleState.OVERHEATED.value).upper()
            max_state = ModuleState[max_state_name] if max_state_name in ModuleState.__members__ else ModuleState.OVERHEATED
            module.state = cls._clamp_runtime_state_to_pyfa_max(source_module.state, max_state)
            if module.charge_capacity > 0:
                module.charge_remaining = max(0.0, min(float(source_module.charge_remaining), float(module.charge_capacity)))

    def _apply_runtime_activation_limit_transitions(
        self,
        world: WorldState,
        ship,
        source_runtime: FitRuntime,
        target_runtime: FitRuntime,
    ) -> bool:
        source_by_module_id = {str(module.module_id): module for module in source_runtime.modules}
        pyfa_remote_inputs_dirty = False

        for target_module in target_runtime.modules:
            module_id = str(target_module.module_id)
            source_module = source_by_module_id.get(module_id)
            if source_module is None:
                continue
            source_max_state = self._runtime_module_max_state(source_runtime, module_id)
            target_max_state = self._runtime_module_max_state(target_runtime, module_id)
            if source_max_state != target_max_state:
                ship.combat.module_decision_pending.add(module_id)
            if source_module.state not in {ModuleState.ACTIVE, ModuleState.OVERHEATED}:
                continue
            if target_module.state in {ModuleState.ACTIVE, ModuleState.OVERHEATED}:
                continue

            previous_projected_target = ship.combat.projected_targets.get(module_id)
            self._flush_projected_cycle_total(world, ship.ship_id, target_module, previous_projected_target)
            self._clear_module_cycle_snapshots(ship.ship_id, module_id)
            self._clear_module_cycle_timer(ship, module_id)
            self._clear_module_reactivation_timer(ship, module_id)
            if target_module.state == ModuleState.OFFLINE:
                ship.combat.projected_targets.pop(module_id, None)
            if self._module_affects_pyfa_remote_inputs(target_module):
                pyfa_remote_inputs_dirty = True

        return pyfa_remote_inputs_dirty

    @staticmethod
    def _clone_resolved_runtime_for_ship(source_runtime, resolved_runtime) -> FitRuntime:
        # Clone only the mutable runtime shell. Avoid generic deepcopy so batched Pyfa refresh
        # does not recursively copy cached metadata and immutable fit graph fragments.
        diagnostics = {
            key: value
            for key, value in source_runtime.diagnostics.items()
            if key not in _RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS
        }
        diagnostics.update(
            {
                key: value
                for key, value in resolved_runtime.diagnostics.items()
                if key not in _RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS
            }
        )
        modules = [
            ModuleRuntime(
                module_id=str(module.module_id),
                group=str(module.group),
                state=module.state,
                effects=list(module.effects),
                charge_capacity=int(module.charge_capacity),
                charge_rate=float(module.charge_rate),
                charge_remaining=float(module.charge_remaining),
                charge_reload_time=float(module.charge_reload_time),
                tags=tuple(str(tag) for tag in getattr(module, "tags", ()) or ()),
            )
            for module in resolved_runtime.modules
        ]
        return FitRuntime(
            fit_key=str(source_runtime.fit_key),
            hull=resolved_runtime.hull,
            skills=resolved_runtime.skills,
            modules=modules,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _runtime_offline_module_signature(runtime) -> int:
        signature = 0
        for index, module in enumerate(runtime.modules):
            if module.state == module.state.OFFLINE:
                signature |= 1 << index
        return signature

    def _runtime_minimum_potential_cycle_time(self, runtime) -> float | None:
        signature = self._runtime_offline_module_signature(runtime)
        cached_signature = runtime.diagnostics.get("runtime_minimum_potential_cycle_signature")
        cached_minimum = runtime.diagnostics.get("runtime_minimum_potential_cycle_time")
        if cached_signature == signature:
            if cached_minimum is None:
                return None
            return float(cached_minimum)

        minimum: float | None = None
        for module, metadata in self._runtime_module_buckets(runtime).controlled_entries:
            if module.state == module.state.OFFLINE:
                continue
            cycle_time = metadata.cycle_time
            if cycle_time <= 0.0:
                continue
            if minimum is None or cycle_time < minimum:
                minimum = cycle_time

        runtime.diagnostics["runtime_minimum_potential_cycle_signature"] = signature
        runtime.diagnostics["runtime_minimum_potential_cycle_time"] = minimum
        return minimum

    def _minimum_potential_cycle_time(self, world: WorldState) -> float | None:
        minimum: float | None = None
        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue
            cycle_time = self._runtime_minimum_potential_cycle_time(ship.runtime)
            if cycle_time is None:
                continue
            if minimum is None or cycle_time < minimum:
                minimum = cycle_time
        return minimum

    def _decision_now(self, world: WorldState, fallback: float | None = None) -> float:
        if fallback is not None:
            return float(fallback)
        if self._decision_reference_time is not None:
            return float(self._decision_reference_time)
        return float(world.now)

    def _collect_projected_impacts(self, world: WorldState, dt: float) -> dict[str, list[ProjectedImpact]]:
        del dt
        impacts: dict[str, list[ProjectedImpact]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_combat_suppressed(source):
                continue
            runtime_projected_entries = self._runtime_module_buckets(source.runtime).runtime_projected_entries
            if not runtime_projected_entries:
                continue
            if not source.combat.projected_targets:
                has_active_area_effect = False
                for module, metadata in runtime_projected_entries:
                    if not metadata.is_area_effect:
                        continue
                    if any(module.is_active_for(effect.state_required) for _effect_index, effect in metadata.projected_effects):
                        has_active_area_effect = True
                        break
                if not has_active_area_effect:
                    continue
            for module, metadata in runtime_projected_entries:
                if metadata.is_missile_weapon or metadata.is_bomb_launcher:
                    continue
                for effect_index, effect in metadata.projected_effects:
                    if not module.is_active_for(effect.state_required):
                        continue

                    target_id = source.combat.projected_targets.get(module.module_id)
                    if not target_id:
                        continue
                    target = world.combat_entity(target_id)
                    if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target):
                        continue

                    if not self._ensure_target_lock(
                        world,
                        source,
                        target_id,
                        target,
                        lock_context="projected_lock",
                    ):
                        continue

                    target_snapshot = self._module_cycle_snapshot_for_target(source.ship_id, module.module_id, target_id)
                    if target_snapshot is None:
                        continue
                    strength = self._cycle_effect_strength(effect, effect_index, target_snapshot)
                    if strength <= 0:
                        continue
                    if self.detailed_logging and self.logger is not None:
                        self.logger.debug(
                            f"projected_formula source={source.ship_id} target={target_id} module={module.module_id} dist={target_snapshot.distance:.1f} range={effect.range_m:.1f} falloff={effect.falloff_m:.1f} strength={strength:.4f}"
                        )
                    impacts.setdefault(target_id, []).append(
                        ProjectedImpact(source_ship_id=source.ship_id, target_ship_id=target_id, effect=effect, strength=strength)
                    )
        return impacts

    def _collect_projected_source_snapshots(
        self,
        world: WorldState,
        command_boosters_by_ship: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        snapshots_by_ship: dict[str, list[dict[str, Any]]] = {}
        for source in world.ships.values():
            if not source.vital.alive or source.runtime is None or self._ship_combat_suppressed(source):
                continue

            blueprint = source.runtime.diagnostics.get("pyfa_blueprint")
            if not isinstance(blueprint, dict):
                continue
            blueprint_modules_raw = blueprint.get("modules")
            blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
            blueprint_modules_by_id = {
                str(raw.get("module_id", "") or ""): raw
                for raw in blueprint_modules
                if isinstance(raw, dict)
            }

            source_command_snapshots = command_boosters_by_ship.get(source.ship_id, [])
            base_state_by_module_id: dict[str, str] = {}
            active_projected_modules: list[tuple[Any, str]] = []
            projected_module_ids: set[str] = set()

            for module, metadata in zip(source.runtime.modules, self._runtime_module_metadata_list(source.runtime)):
                state_value = str(module.state.value or "ONLINE").upper()
                projected_state = state_value
                if metadata.has_projected:
                    projected_module_ids.add(str(module.module_id))

                if state_value in {"ACTIVE", "OVERHEATED"}:
                    if metadata.is_command_burst:
                        projected_state = state_value
                    elif metadata.uses_pyfa_projected_profile:
                        projected_state = "ONLINE"
                    elif metadata.is_area_effect or metadata.is_weapon or metadata.has_projected_rep or metadata.is_cap_warfare:
                        projected_state = "ONLINE"

                base_state_by_module_id[module.module_id] = projected_state

                if metadata.uses_pyfa_projected_profile and state_value in {"ACTIVE", "OVERHEATED"}:
                    active_projected_modules.append((module, state_value))

            for active_projected_module, active_state in active_projected_modules:
                target_id = source.combat.projected_targets.get(active_projected_module.module_id)
                if not target_id:
                    continue
                target = world.combat_entity(target_id)
                if target is None or not target.vital.alive or self._ship_hidden_from_targeting(target) or target.runtime is None:
                    continue
                target_snapshot = self._module_cycle_snapshot_for_target(source.ship_id, active_projected_module.module_id, target_id)
                if target_snapshot is None:
                    continue
                projection_key_mode, projection_range = self._pyfa_projection_snapshot_params(
                    active_projected_module,
                    target_snapshot,
                )

                state_by_module_id = dict(base_state_by_module_id)
                for projected_module_id in projected_module_ids:
                    if projected_module_id != str(active_projected_module.module_id):
                        state_by_module_id[projected_module_id] = "OFFLINE"
                state_by_module_id[active_projected_module.module_id] = active_state
                snapshots_by_ship.setdefault(target_id, []).append(
                    {
                        "fit_key": f"{source.runtime.fit_key}:{active_projected_module.module_id}",
                        "blueprint": blueprint,
                        "state_by_module_id": state_by_module_id,
                        "command_booster_snapshots": source_command_snapshots,
                        "pyfa_projection_key_mode": projection_key_mode,
                        "pyfa_projection_range": projection_range,
                        "projection_range": projection_range,
                        "pyfa_projection_module_signature": self._projected_module_runtime_signature(
                            active_projected_module,
                            blueprint_modules_by_id.get(str(active_projected_module.module_id)),
                            active_state,
                            active_effect_indices=target_snapshot.active_effect_indices,
                        ),
                    }
                )

        return snapshots_by_ship

    def _refresh_effective_runtimes_from_pyfa(
        self,
        world: WorldState,
        command_boosters_by_ship: dict[str, list[dict[str, Any]]],
        projected_sources_by_ship: dict[str, list[dict[str, Any]]],
    ) -> None:
        pending_batches: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None:
                continue

            booster_snapshots = command_boosters_by_ship.get(ship.ship_id, [])
            projected_snapshots = projected_sources_by_ship.get(ship.ship_id, [])
            cache_key = self._get_runtime_resolve_cache_key(ship.runtime, booster_snapshots, projected_snapshots)
            local_signature = self._local_runtime_state_signature_from_metadata(ship.runtime)
            booster_signature = self._command_snapshot_list_signature(booster_snapshots)
            projected_signature = self._projected_snapshot_list_signature(projected_snapshots)
            projected_structure_signature = self._projected_snapshot_structure_signature(projected_snapshots)
            cached_signature = ship.runtime.diagnostics.get("pyfa_resolve_signature")
            cached_base_profile = ship.runtime.diagnostics.get("pyfa_base_profile")
            if cache_key is not None and cached_signature == cache_key and isinstance(cached_base_profile, ShipProfile):
                if local_signature is not None:
                    ship.runtime.diagnostics["pyfa_local_state_signature"] = local_signature
                ship.runtime.diagnostics["pyfa_command_boosters"] = booster_snapshots
                ship.runtime.diagnostics["pyfa_projected_sources"] = projected_snapshots
                ship.runtime.diagnostics["pyfa_command_booster_signature"] = booster_signature
                ship.runtime.diagnostics["pyfa_projected_sources_signature"] = projected_signature
                ship.runtime.diagnostics["pyfa_projected_sources_structure_signature"] = projected_structure_signature
                ship.profile = cached_base_profile
                continue

            batch_key = cache_key if cache_key is not None else ("ship", ship.ship_id)
            pending_batches.setdefault(batch_key, []).append(
                {
                    "ship": ship,
                    "runtime": ship.runtime,
                    "booster_snapshots": booster_snapshots,
                    "projected_snapshots": projected_snapshots,
                    "booster_signature": booster_signature,
                    "projected_signature": projected_signature,
                    "local_signature": local_signature,
                    "cache_key": cache_key,
                }
            )

        for pending_group in pending_batches.values():
            first_pending = pending_group[0]
            resolve_started = time.perf_counter()
            resolved = self._resolve_runtime_from_pyfa_runtime(
                first_pending["runtime"],
                first_pending["booster_snapshots"],
                first_pending["projected_snapshots"],
            )
            resolve_cache = "error"
            projected_fit_cache = "error"
            if resolved is not None:
                resolve_cache = str(resolved[0].diagnostics.get("pyfa_runtime_resolve_cache", "unknown") or "unknown")
                projected_fit_cache = str(resolved[0].diagnostics.get("pyfa_projected_target_fit_cache", "not_applicable") or "not_applicable")
            self._log_hotspot(
                "combat.pyfa_resolve_batch",
                resolve_started,
                tick=int(world.tick),
                batch_size=len(pending_group),
                ship_ids=tuple(str(pending["ship"].ship_id) for pending in pending_group),
                fit_key=str(first_pending["runtime"].fit_key or ""),
                command_sources=len(first_pending["booster_snapshots"]),
                projected_sources=len(first_pending["projected_snapshots"]),
                success=resolved is not None,
                resolve_cache=resolve_cache,
                projected_fit_cache=projected_fit_cache,
            )
            if resolved is None:
                for pending in pending_group:
                    cached_base_profile = pending["runtime"].diagnostics.get("pyfa_base_profile")
                    if isinstance(cached_base_profile, ShipProfile):
                        pending["ship"].profile = cached_base_profile
                    else:
                        pending["ship"].profile = self._fallback_unprojected_profile(pending["ship"])
                continue

            resolved_runtime, resolved_profile = resolved
            resolved_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

            for index, pending in enumerate(pending_group):
                source_runtime = pending["runtime"]
                ship = pending["ship"]
                target_runtime = resolved_runtime if index == 0 else self._clone_resolved_runtime_for_ship(source_runtime, resolved_runtime)
                target_runtime.fit_key = source_runtime.fit_key

                blueprint = source_runtime.diagnostics.get("pyfa_blueprint")
                if isinstance(blueprint, dict):
                    target_runtime.diagnostics["pyfa_blueprint"] = deepcopy(blueprint)

                target_runtime.diagnostics["pyfa_command_boosters"] = pending["booster_snapshots"]
                target_runtime.diagnostics["pyfa_projected_sources"] = pending["projected_snapshots"]
                target_runtime.diagnostics["pyfa_command_booster_signature"] = pending["booster_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_signature"] = pending["projected_signature"]
                target_runtime.diagnostics["pyfa_projected_sources_structure_signature"] = self._projected_snapshot_structure_signature(pending["projected_snapshots"])
                target_runtime.diagnostics["pyfa_base_profile"] = resolved_profile

                self._copy_runtime_dynamic_state(source_runtime, target_runtime)
                if self._apply_runtime_activation_limit_transitions(world, ship, source_runtime, target_runtime):
                    self._mark_pyfa_remote_inputs_dirty()
                resolved_local_signature = self._local_runtime_state_signature_from_metadata(target_runtime)
                resolved_cache_key = self._get_runtime_resolve_cache_key(
                    target_runtime,
                    pending["booster_snapshots"],
                    pending["projected_snapshots"],
                )
                if resolved_cache_key is not None:
                    target_runtime.diagnostics["pyfa_resolve_signature"] = resolved_cache_key
                else:
                    target_runtime.diagnostics.pop("pyfa_resolve_signature", None)
                if resolved_local_signature is not None:
                    target_runtime.diagnostics["pyfa_local_state_signature"] = resolved_local_signature
                else:
                    target_runtime.diagnostics.pop("pyfa_local_state_signature", None)
                target_runtime.diagnostics["runtime_local_state_signature"] = resolved_local_signature
                target_runtime.diagnostics["runtime_has_active_pyfa_remote_inputs"] = self._runtime_has_active_pyfa_remote_inputs(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_minimum_potential_cycle_signature"] = self._runtime_offline_module_signature(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_minimum_potential_cycle_time"] = self._runtime_minimum_potential_cycle_time(
                    target_runtime
                )
                target_runtime.diagnostics["runtime_observed_module_state_signature"] = self._runtime_observed_module_state_signature(
                    target_runtime
                )
                ship.runtime = target_runtime
                ship.profile = resolved_profile

    def _effective_profile(self, ship, impacts: dict[str, list[ProjectedImpact]]):
        if ship.runtime is None:
            return ship.profile

        applied = impacts.get(ship.ship_id)
        if not applied:
            return ship.profile

        runtime = ship.runtime
        projected_signature = self._projected_impact_signature(applied)
        cache_signature = (
            runtime.diagnostics.get("pyfa_resolve_signature"),
            runtime.diagnostics.get("runtime_local_state_signature"),
            projected_signature,
        )
        cached_signature = runtime.diagnostics.get("runtime_projected_effective_profile_signature")
        cached_profile = runtime.diagnostics.get("runtime_projected_effective_profile")
        if cached_signature == cache_signature and isinstance(cached_profile, ShipProfile):
            return cached_profile

        effective = self._apply_runtime_projected_impacts(ship.profile, applied, runtime=runtime)
        runtime.diagnostics["runtime_projected_effective_profile_signature"] = cache_signature
        runtime.diagnostics["runtime_projected_effective_profile"] = effective
        return effective
