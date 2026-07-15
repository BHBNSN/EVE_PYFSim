from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from eve_sim.fleet_setup import (
    EftFitParser,
    _PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE,
    RuntimeFromEftFactory,
    _PYFA_RUNTIME_RESOLVED_CACHE,
    _get_static_backend,
    get_runtime_resolve_cache_key,
    prewarm_runtime_base_cache,
    _runtime_local_profile_state_map,
    resolve_runtime_from_pyfa_runtime,
)
from eve_sim.fit_runtime import ModuleState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.systems import CombatSystem


class PyfaResolveCacheDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _get_static_backend().fit_engine_ready:
            self.skipTest("pyfa static fit engine unavailable")
        _PYFA_RUNTIME_RESOLVED_CACHE.clear()
        _PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE.clear()

    def tearDown(self) -> None:
        _PYFA_RUNTIME_RESOLVED_CACHE.clear()
        _PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE.clear()

    def test_resolve_runtime_records_cache_hit_and_miss(self) -> None:
        eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        first = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(first)
        assert first is not None
        first_runtime, _first_profile = first
        self.assertEqual(first_runtime.diagnostics.get("pyfa_runtime_resolve_cache"), "miss")

        second = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(second)
        assert second is not None
        second_runtime, _second_profile = second
        self.assertEqual(second_runtime.diagnostics.get("pyfa_runtime_resolve_cache"), "hit")

    def test_resolve_runtime_reuses_local_base_fit_cache_across_cache_misses(self) -> None:
        eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        first = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(first)
        self.assertGreater(len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE), 0)
        local_base_cache_size = len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE)

        _PYFA_RUNTIME_RESOLVED_CACHE.clear()

        second = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(second)
        assert second is not None
        second_runtime, _second_profile = second
        self.assertEqual(second_runtime.diagnostics.get("pyfa_runtime_resolve_cache"), "miss")
        self.assertEqual(len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE), local_base_cache_size)

    def test_prewarm_runtime_base_cache_tracks_unique_fit_kinds(self) -> None:
        parser = EftFitParser()
        factory = RuntimeFromEftFactory()
        ferox_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        drake_eft = """[Drake, Burst]
Shield Command Burst II, Shield Harmonizing Charge
10MN Afterburner II
"""

        runtime_a, _fit_a = factory.build(parser.parse(ferox_eft))
        runtime_b, _fit_b = factory.build(parser.parse(ferox_eft))
        runtime_c, _fit_c = factory.build(parser.parse(drake_eft))

        self.assertTrue(prewarm_runtime_base_cache(runtime_a))
        self.assertTrue(prewarm_runtime_base_cache(runtime_b))
        self.assertTrue(prewarm_runtime_base_cache(runtime_c))
        self.assertEqual(len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE), 2)

    def test_build_prewarms_precalculated_local_base_fit(self) -> None:
        eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        self.assertIsNotNone(runtime)
        self.assertGreaterEqual(len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE), 1)
        cached_fit, _charge_names = next(iter(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE.values()))
        online_state = _get_static_backend()._fitting_module_state_online
        self.assertTrue(all(module.state == online_state for module in cached_fit.modules))

    def test_build_downgrades_modules_pyfa_would_not_allow_online(self) -> None:
        eft = """[Drake, Dual Burst]
Shield Command Burst II, Shield Harmonizing Charge
Armor Command Burst II, Armor Energizing Charge
10MN Afterburner II
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        burst_states = [module.state.value for module in runtime.modules if module.group == "Command Burst"]
        self.assertEqual(sorted(burst_states), ["OFFLINE", "ONLINE"])

    def test_build_preclassifies_controlled_and_local_stateful_modules(self) -> None:
        eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        self.assertEqual(
            runtime.diagnostics.get("runtime_controlled_module_ids"),
            ("mod-1", "mod-2", "mod-3", "mod-7"),
        )
        self.assertEqual(
            runtime.diagnostics.get("runtime_local_stateful_module_ids"),
            ("mod-7",),
        )
        self.assertEqual(
            _runtime_local_profile_state_map(runtime),
            {"mod-7": "ONLINE"},
        )

    def test_unscripted_hic_generator_stays_unscripted_and_builds_local_bubble_marker(self) -> None:
        eft = """[Devoter, Bubble]
Warp Disruption Field Generator I
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        module = runtime.modules[0]
        blueprint_modules = runtime.diagnostics.get("pyfa_blueprint", {}).get("modules", [])
        self.assertEqual(module.group, "Warp Disrupt Field Generator")
        self.assertEqual(module.effects[0].effect_class.value, "LOCAL")
        self.assertAlmostEqual(module.effects[0].local_add.get("bubble_follow_owner", 0.0), 1.0)
        self.assertAlmostEqual(module.effects[0].local_add.get("bubble_blocks_warp", 0.0), 1.0)
        self.assertEqual(blueprint_modules[0].get("charge_name"), None)

    def test_interdiction_sphere_probe_builds_static_bubble_marker_from_charge_stats(self) -> None:
        eft = """[Heretic, Probe]
Interdiction Sphere Launcher I, Warp Disrupt Probe
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        module = runtime.modules[0]
        effect = module.effects[0]
        self.assertEqual(module.group, "Interdiction Sphere Launcher")
        self.assertEqual(effect.effect_class.value, "LOCAL")
        self.assertAlmostEqual(effect.local_add.get("bubble_radius_m", 0.0), 20_000.0, delta=1e-6)
        self.assertAlmostEqual(effect.local_add.get("bubble_duration_sec", 0.0), 120.0, delta=1e-6)
        self.assertAlmostEqual(effect.local_add.get("bubble_blocks_warp", 0.0), 1.0)
        self.assertAlmostEqual(effect.local_add.get("bubble_structure_hp", 0.0), 1_000.0, delta=1e-6)

    def test_stasis_webification_probe_builds_static_web_bubble_marker(self) -> None:
        eft = """[Heretic, Web Probe]
Interdiction Sphere Launcher I, Stasis Webification Probe
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))

        module = runtime.modules[0]
        effect = module.effects[0]
        self.assertEqual(module.group, "Interdiction Sphere Launcher")
        self.assertEqual(effect.effect_class.value, "LOCAL")
        self.assertAlmostEqual(effect.local_add.get("bubble_radius_m", 0.0), 15_000.0, delta=1e-6)
        self.assertAlmostEqual(effect.local_add.get("bubble_duration_sec", 0.0), 30.0, delta=1e-6)
        self.assertAlmostEqual(effect.local_add.get("bubble_blocks_warp", 0.0), 0.0)
        self.assertAlmostEqual(effect.local_add.get("bubble_speed_factor_mult", 1.0), 0.8, delta=1e-6)

    def test_resolve_miss_calculates_target_fit_once_from_neutral_base(self) -> None:
        target_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        booster_eft = """[Drake, Burst]
Shield Command Burst II, Shield Harmonizing Charge
"""
        projected_eft = """[Blackbird, EWAR]
Remote Sensor Dampener II
"""

        factory = RuntimeFromEftFactory()
        target_runtime, _fit = factory.build(EftFitParser().parse(target_eft))
        booster_runtime, _booster_fit = factory.build(EftFitParser().parse(booster_eft))
        projected_runtime, _projected_fit = factory.build(EftFitParser().parse(projected_eft))

        for module in booster_runtime.modules:
            module.state = module.state.ACTIVE
        for module in projected_runtime.modules:
            module.state = module.state.ACTIVE

        command_snapshot = {
            "fit_key": booster_runtime.fit_key,
            "blueprint": deepcopy(booster_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in booster_runtime.modules},
            "command_booster_snapshots": [],
        }
        projected_snapshot = {
            "fit_key": projected_runtime.fit_key,
            "blueprint": deepcopy(projected_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in projected_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_range": 1_000.0,
            "pyfa_projection_module_signature": ("test-projected-source",),
        }

        _PYFA_RUNTIME_RESOLVED_CACHE.clear()
        fit_cls = _get_static_backend()._fit_cls
        self.assertIsNotNone(fit_cls)
        assert fit_cls is not None
        original_calculate = fit_cls.calculateModifiedAttributes
        calculate_calls: list[tuple[bool, str]] = []

        def _wrapped_calculate(self, *args, **kwargs):
            target_fit = kwargs.get("targetFit", args[0] if len(args) >= 1 else None)
            calc_type = kwargs.get("type", args[1] if len(args) >= 2 else None)
            calc_name = getattr(calc_type, "name", str(calc_type) if calc_type is not None else "LOCAL")
            calculate_calls.append((target_fit is None, calc_name))
            return original_calculate(self, *args, **kwargs)

        with unittest.mock.patch.object(fit_cls, "calculateModifiedAttributes", _wrapped_calculate):
            resolved = resolve_runtime_from_pyfa_runtime(target_runtime, [command_snapshot], [projected_snapshot])

        self.assertIsNotNone(resolved)
        local_calls = [(is_local, calc_name) for is_local, calc_name in calculate_calls if is_local and calc_name == "LOCAL"]
        self.assertEqual(len(local_calls), 1)
        self.assertIn((False, "COMMAND"), calculate_calls)
        self.assertIn((False, "PROJECTED"), calculate_calls)

    def test_resolve_records_pyfa_group_limited_max_state_downgrades(self) -> None:
        eft = """[Drake, Dual Prop]
10MN Afterburner II
50MN Quad LiF Restrained Microwarpdrive
"""

        runtime, _fit = RuntimeFromEftFactory().build(EftFitParser().parse(eft))
        for module in runtime.modules:
            module.state = module.state.ACTIVE

        first = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(first)
        assert first is not None
        first_runtime, _first_profile = first
        self.assertEqual(first_runtime.diagnostics.get("pyfa_runtime_resolve_cache"), "miss")
        self.assertEqual(
            sorted(first_runtime.diagnostics.get("pyfa_max_state_by_module_id", {}).values()),
            ["ONLINE", "OVERHEATED"],
        )

        second = resolve_runtime_from_pyfa_runtime(runtime, [], [])
        self.assertIsNotNone(second)
        assert second is not None
        second_runtime, _second_profile = second
        self.assertEqual(second_runtime.diagnostics.get("pyfa_runtime_resolve_cache"), "hit")
        self.assertEqual(
            sorted(second_runtime.diagnostics.get("pyfa_max_state_by_module_id", {}).values()),
            ["ONLINE", "OVERHEATED"],
        )

    def test_resolve_scrammed_propulsion_module_stays_online_and_unboosted(self) -> None:
        target_eft = """[Ferox, Prop]
50MN Quad LiF Restrained Microwarpdrive
"""
        source_eft = """[Keres, Scram]
Warp Scrambler II
"""

        factory = RuntimeFromEftFactory()
        target_parsed = EftFitParser().parse(target_eft)
        target_runtime, _target_fit = factory.build(target_parsed)
        base_profile = factory.build_profile(target_parsed)
        source_runtime, _source_fit = factory.build(EftFitParser().parse(source_eft))

        for module in source_runtime.modules:
            module.state = module.state.ACTIVE

        projected_snapshot = {
            "fit_key": source_runtime.fit_key,
            "blueprint": deepcopy(source_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in source_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_range": 5_000.0,
            "pyfa_projection_key_mode": "exact_range",
            "pyfa_projection_module_signature": ("test-scram-source",),
        }

        target_runtime.modules[0].state = target_runtime.modules[0].state.ACTIVE
        resolved = resolve_runtime_from_pyfa_runtime(target_runtime, [], [projected_snapshot])

        self.assertIsNotNone(resolved)
        assert resolved is not None
        resolved_runtime, resolved_profile = resolved
        self.assertEqual(resolved_runtime.modules[0].state, ModuleState.ONLINE)
        self.assertEqual(resolved_runtime.diagnostics.get("pyfa_max_state_by_module_id", {}).get("mod-1"), "ONLINE")
        self.assertAlmostEqual(resolved_profile.warp_scramble_status, 2.0)
        self.assertAlmostEqual(resolved_profile.max_speed, base_profile.max_speed)

    def test_resolve_local_state_change_derives_cached_local_fit_without_rebuild(self) -> None:
        eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""

        parser = EftFitParser()
        parsed = parser.parse(eft)
        runtime, fit_descriptor = RuntimeFromEftFactory().build(parsed)
        initial_local_cache_size = len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE)
        self.assertGreaterEqual(initial_local_cache_size, 1)

        runtime.modules[-1].state = runtime.modules[-1].state.ACTIVE
        _PYFA_RUNTIME_RESOLVED_CACHE.clear()

        original_build_pyfa_fit = RuntimeFromEftFactory._build_pyfa_fit
        build_call_count = 0

        def _wrapped_build_pyfa_fit(self, *args, **kwargs):
            nonlocal build_call_count
            build_call_count += 1
            return original_build_pyfa_fit(self, *args, **kwargs)

        with patch.object(RuntimeFromEftFactory, "_build_pyfa_fit", _wrapped_build_pyfa_fit):
            resolved = resolve_runtime_from_pyfa_runtime(runtime, [], [])

        self.assertIsNotNone(resolved)
        assert resolved is not None
        _resolved_runtime, resolved_profile = resolved
        fresh_factory = RuntimeFromEftFactory()
        fresh_fit, _fresh_modules = fresh_factory._build_pyfa_fit(
            parsed,
            state_by_module_id=_runtime_local_profile_state_map(runtime),
            calculate_modified_attributes=True,
        )
        fresh_max_speed = fresh_factory._compute_pyfa_final_stats(fresh_fit)["max_speed"]
        self.assertEqual(build_call_count, 0)
        self.assertEqual(len(_PYFA_PRECALCULATED_LOCAL_BASE_FIT_CACHE), initial_local_cache_size)
        self.assertGreater(resolved_profile.max_speed, fit_descriptor.max_speed)
        self.assertAlmostEqual(resolved_profile.max_speed, fresh_max_speed, places=6)

    def test_resolve_projected_miss_does_not_use_incremental_remote_helper_path(self) -> None:
        target_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        projected_a_eft = """[Blackbird, Damp A]
Remote Sensor Dampener II
"""
        projected_b_eft = """[Blackbird, Damp B]
Remote Sensor Dampener II
"""

        factory = RuntimeFromEftFactory()
        target_runtime, _fit = factory.build(EftFitParser().parse(target_eft))
        projected_a_runtime, _fit_a = factory.build(EftFitParser().parse(projected_a_eft))
        projected_b_runtime, _fit_b = factory.build(EftFitParser().parse(projected_b_eft))
        for projected_runtime in (projected_a_runtime, projected_b_runtime):
            for module in projected_runtime.modules:
                module.state = module.state.ACTIVE

        snapshot_a = {
            "fit_key": projected_a_runtime.fit_key,
            "blueprint": deepcopy(projected_a_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in projected_a_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_range": 1_000.0,
            "pyfa_projection_module_signature": ("test-projected-a",),
        }
        snapshot_b = {
            "fit_key": projected_b_runtime.fit_key,
            "blueprint": deepcopy(projected_b_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in projected_b_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_range": 2_000.0,
            "pyfa_projection_module_signature": ("test-projected-b",),
        }

        _PYFA_RUNTIME_RESOLVED_CACHE.clear()
        first = resolve_runtime_from_pyfa_runtime(target_runtime, [], [snapshot_a])
        second = resolve_runtime_from_pyfa_runtime(target_runtime, [], [snapshot_a, snapshot_b])

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first[0].diagnostics.get("pyfa_projected_target_fit_cache"), "single_pass")
        self.assertEqual(second[0].diagnostics.get("pyfa_projected_target_fit_cache"), "single_pass")

    def test_resolve_cache_key_ignores_in_range_projection_distance_for_constant_pyfa_effects(self) -> None:
        target_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        projected_eft = """[Blackbird, Damp]
Remote Sensor Dampener II
"""

        factory = RuntimeFromEftFactory()
        target_runtime, _fit = factory.build(EftFitParser().parse(target_eft))
        projected_runtime, _projected_fit = factory.build(EftFitParser().parse(projected_eft))
        for module in projected_runtime.modules:
            module.state = module.state.ACTIVE

        snapshot_near = {
            "fit_key": projected_runtime.fit_key,
            "blueprint": deepcopy(projected_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in projected_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_key_mode": "in_range",
            "pyfa_projection_range": 1_000.0,
            "pyfa_projection_module_signature": ("test-constant-projection",),
        }
        snapshot_far = {
            **snapshot_near,
            "pyfa_projection_range": 50_000.0,
        }

        near_key = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_near])
        far_key = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_far])

        self.assertEqual(near_key, far_key)

    def test_resolve_cache_key_buckets_exact_projection_distance_for_falloff_pyfa_effects(self) -> None:
        target_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        projected_eft = """[Blackbird, Damp]
Remote Sensor Dampener II
"""

        factory = RuntimeFromEftFactory()
        target_runtime, _fit = factory.build(EftFitParser().parse(target_eft))
        projected_runtime, _projected_fit = factory.build(EftFitParser().parse(projected_eft))
        for module in projected_runtime.modules:
            module.state = module.state.ACTIVE

        snapshot_bucket_a = {
            "fit_key": projected_runtime.fit_key,
            "blueprint": deepcopy(projected_runtime.diagnostics["pyfa_blueprint"]),
            "state_by_module_id": {str(module.module_id): str(module.state.value) for module in projected_runtime.modules},
            "command_booster_snapshots": [],
            "pyfa_projection_key_mode": "exact_range",
            "pyfa_projection_range": 45_000.0,
            "pyfa_projection_module_signature": ("projected-module",),
        }
        snapshot_same_bucket = {
            **snapshot_bucket_a,
            "pyfa_projection_range": 45_099.0,
        }
        snapshot_next_bucket = {
            **snapshot_bucket_a,
            "pyfa_projection_range": 45_100.0,
        }

        bucket_a_key = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_bucket_a])
        same_bucket_key = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_same_bucket])
        next_bucket_key = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_next_bucket])

        self.assertEqual(bucket_a_key, same_bucket_key)
        self.assertNotEqual(bucket_a_key, next_bucket_key)

    def test_resolve_cache_key_uses_module_projection_signature_not_unrelated_source_state(self) -> None:
        target_eft = """[Ferox, Rail DPS]
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II
10MN Afterburner II
"""
        projected_eft = """[Blackbird, EWAR]
Remote Sensor Dampener II
10MN Afterburner II
"""

        factory = RuntimeFromEftFactory()
        target_runtime, _fit = factory.build(EftFitParser().parse(target_eft))
        projected_runtime, _projected_fit = factory.build(EftFitParser().parse(projected_eft))
        combat = CombatSystem(PyfaBridge())

        projected_runtime.modules[0].state = projected_runtime.modules[0].state.ACTIVE
        blueprint = deepcopy(projected_runtime.diagnostics["pyfa_blueprint"])
        blueprint_modules_raw = blueprint.get("modules")
        blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
        blueprint_by_id = {
            str(raw.get("module_id", "") or ""): raw
            for raw in blueprint_modules
            if isinstance(raw, dict)
        }
        module_signature = combat._projected_module_runtime_signature(
            projected_runtime.modules[0],
            blueprint_by_id.get("mod-1"),
            "ACTIVE",
            active_effect_indices={0},
        )

        snapshot_a = {
            "fit_key": "ewar-a",
            "blueprint": deepcopy(blueprint),
            "state_by_module_id": {"mod-1": "ACTIVE", "mod-2": "ONLINE"},
            "command_booster_snapshots": [],
            "pyfa_projection_key_mode": "exact_range",
            "pyfa_projection_range": 45_000.0,
            "pyfa_projection_module_signature": module_signature,
        }
        snapshot_b = {
            **snapshot_a,
            "fit_key": "ewar-b",
            "state_by_module_id": {"mod-1": "ACTIVE", "mod-2": "ACTIVE"},
        }
        snapshot_c = {
            **snapshot_a,
            "pyfa_projection_module_signature": (
                module_signature[0],
                module_signature[1],
                module_signature[2],
                module_signature[3],
                tuple([("changed",)]),
            ),
        }

        key_a = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_a])
        key_b = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_b])
        key_c = get_runtime_resolve_cache_key(target_runtime, [], [snapshot_c])

        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_resolve_local_active_state_matches_fresh_pyfa_speed(self) -> None:
        eft = """[Ferox, Prop Test]
10MN Afterburner II
Magnetic Field Stabilizer II
250mm Railgun II, Antimatter Charge M
"""

        parser = EftFitParser()
        parsed = parser.parse(eft)
        runtime, _fit = RuntimeFromEftFactory().build(parsed)
        runtime.modules[0].state = runtime.modules[0].state.ACTIVE
        _PYFA_RUNTIME_RESOLVED_CACHE.clear()

        resolved = resolve_runtime_from_pyfa_runtime(runtime, [], [])

        self.assertIsNotNone(resolved)
        assert resolved is not None
        _resolved_runtime, resolved_profile = resolved

        fresh_factory = RuntimeFromEftFactory()
        fresh_fit, _fresh_modules = fresh_factory._build_pyfa_fit(
            parsed,
            state_by_module_id=_runtime_local_profile_state_map(runtime),
            calculate_modified_attributes=True,
        )
        fresh_max_speed = fresh_factory._compute_pyfa_final_stats(fresh_fit)["max_speed"]
        self.assertAlmostEqual(resolved_profile.max_speed, fresh_max_speed, places=6)


if __name__ == "__main__":
    unittest.main()
