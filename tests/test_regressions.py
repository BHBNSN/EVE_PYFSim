from __future__ import annotations

import logging
from types import SimpleNamespace
import unittest

from eve_sim.fit_runtime import EffectClass, FitRuntime, HullProfile, ModuleEffect, ModuleRuntime, ModuleState, SkillProfile
from eve_sim.domain.ship_fit_service import ShipFitService
from eve_sim.serialization.runtime_ship_factory import RuntimeReplicaShipFactory
from eve_sim.serialization.snapshot_loader import SnapshotLoader
from eve_sim.gui.main_window import MainWindow
from eve_sim.application.contracts import ShipSetupSpec
from eve_sim.lan_match_coordinator import LanMatchCoordinator
from eve_sim.lan_commands import CMD_SYNC_SETUP
from eve_sim.math2d import Vector2
from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, ShipProfile, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.systems import CombatSystem, LogisticsSystem
from eve_sim.world import WorldState


def _make_profile(
    *,
    dps: float = 0.0,
    sig_radius: float = 120.0,
    scan_resolution: float = 300.0,
    max_target_range: float = 120_000.0,
    max_speed: float = 1_000.0,
    shield_hp: float = 100.0,
    armor_hp: float = 100.0,
    structure_hp: float = 100.0,
    rep_amount: float = 0.0,
    rep_cycle: float = 5.0,
) -> ShipProfile:
    return ShipProfile(
        dps=dps,
        volley=dps,
        optimal=10_000.0,
        falloff=0.0,
        tracking=1.0,
        sig_radius=sig_radius,
        scan_resolution=scan_resolution,
        max_target_range=max_target_range,
        max_speed=max_speed,
        max_cap=100.0,
        cap_recharge_time=100.0,
        shield_hp=shield_hp,
        armor_hp=armor_hp,
        structure_hp=structure_hp,
        rep_amount=rep_amount,
        rep_cycle=rep_cycle,
    )


def _make_ship(
    ship_id: str,
    *,
    team: Team,
    squad_id: str = "SQ1",
    profile: ShipProfile | None = None,
    runtime: FitRuntime | None = None,
    position: Vector2 | None = None,
) -> ShipEntity:
    effective_profile = profile or _make_profile()
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name=ship_id,
        role="test",
        base_dps=effective_profile.dps,
        volley=effective_profile.volley,
        optimal_range=effective_profile.optimal,
        falloff=effective_profile.falloff,
        tracking=effective_profile.tracking,
        signature_radius=effective_profile.sig_radius,
        scan_resolution=effective_profile.scan_resolution,
        max_target_range=effective_profile.max_target_range,
        max_speed=effective_profile.max_speed,
        max_cap=effective_profile.max_cap,
        cap_recharge_time=effective_profile.cap_recharge_time,
        shield_hp=effective_profile.shield_hp,
        armor_hp=effective_profile.armor_hp,
        structure_hp=effective_profile.structure_hp,
        rep_amount=effective_profile.rep_amount,
        rep_cycle=effective_profile.rep_cycle,
    )
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=fit,
        profile=effective_profile,
        nav=NavigationState(
            position=position or Vector2(0.0, 0.0),
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            max_speed=effective_profile.max_speed,
        ),
        combat=CombatState(),
        vital=VitalState(
            shield=effective_profile.shield_hp,
            armor=effective_profile.armor_hp,
            structure=effective_profile.structure_hp,
            shield_max=effective_profile.shield_hp,
            armor_max=effective_profile.armor_hp,
            structure_max=effective_profile.structure_hp,
            cap=effective_profile.max_cap,
            cap_max=effective_profile.max_cap,
        ),
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
        runtime=runtime,
    )


def _make_runtime(fit_key: str) -> FitRuntime:
    return FitRuntime(
        fit_key=fit_key,
        hull=HullProfile(
            ship_name=fit_key,
            role="test",
            base_dps=0.0,
            volley=0.0,
            optimal=0.0,
            falloff=0.0,
            tracking=0.0,
            sig_radius=120.0,
            scan_resolution=300.0,
            max_target_range=120_000.0,
            max_speed=1_000.0,
            cap_max=100.0,
            cap_recharge_time=100.0,
            shield_hp=100.0,
            armor_hp=100.0,
            structure_hp=100.0,
            rep_amount=0.0,
            rep_cycle=5.0,
        ),
        skills=SkillProfile(),
        modules=[],
    )


class _FakeLanClient:
    def __init__(self) -> None:
        self.connected = False
        self.sent_commands: list[dict] = []

    def send_command(self, command: dict) -> None:
        self.sent_commands.append(dict(command))

    def consume_latest_state(self) -> dict | None:
        return None


class _FakeParsedFit:
    def __init__(self, fit_text: str) -> None:
        self.fit_text = fit_text


class _FakeParser:
    def parse(self, fit_text: str) -> _FakeParsedFit:
        return _FakeParsedFit(fit_text)


class _FakeFactory:
    @staticmethod
    def _stats(parsed: _FakeParsedFit) -> tuple[str, float]:
        fit_key = parsed.fit_text.strip().lower()
        if fit_key == "fit-b":
            return fit_key, 200.0
        return fit_key, 100.0

    def build(self, parsed: _FakeParsedFit) -> tuple[FitRuntime, FitDescriptor]:
        fit_key, dps = self._stats(parsed)
        fit = FitDescriptor(
            fit_key=fit_key,
            ship_name=f"Hull-{fit_key}",
            role="REMOTE",
            base_dps=dps,
            volley=dps,
            optimal_range=15_000.0,
            falloff=5_000.0,
            tracking=0.5,
            max_speed=1_500.0 + dps,
            max_cap=500.0,
            cap_recharge_time=100.0,
            shield_hp=150.0,
            armor_hp=125.0,
            structure_hp=100.0,
        )
        runtime = _make_runtime(fit_key)
        runtime.fit_key = fit_key
        return runtime, fit

    def build_profile(self, parsed: _FakeParsedFit) -> ShipProfile:
        fit_key, dps = self._stats(parsed)
        return _make_profile(
            dps=dps,
            max_speed=1_500.0 + dps,
            shield_hp=150.0,
            armor_hp=125.0,
            structure_hp=100.0,
        )


class _FakeAmmoFactory(_FakeFactory):
    def build(self, parsed: _FakeParsedFit) -> tuple[FitRuntime, FitDescriptor]:
        fit_key, dps = self._stats(parsed)
        fit = FitDescriptor(
            fit_key=fit_key,
            ship_name=f"Hull-{fit_key}",
            role="REMOTE",
            base_dps=dps,
            volley=dps,
            optimal_range=15_000.0,
            falloff=5_000.0,
            tracking=0.5,
            max_speed=1_500.0 + dps,
            max_cap=500.0,
            cap_recharge_time=100.0,
            shield_hp=150.0,
            armor_hp=125.0,
            structure_hp=100.0,
        )
        runtime = FitRuntime(
            fit_key=fit_key,
            hull=HullProfile(
                ship_name=fit.ship_name,
                role=fit.role,
                base_dps=fit.base_dps,
                volley=fit.volley,
                optimal=fit.optimal_range,
                falloff=fit.falloff,
                tracking=fit.tracking,
                sig_radius=120.0,
                scan_resolution=300.0,
                max_target_range=120_000.0,
                max_speed=fit.max_speed,
                cap_max=fit.max_cap,
                cap_recharge_time=fit.cap_recharge_time,
                shield_hp=fit.shield_hp,
                armor_hp=fit.armor_hp,
                structure_hp=fit.structure_hp,
                rep_amount=0.0,
                rep_cycle=5.0,
            ),
            skills=SkillProfile(),
            modules=[
                ModuleRuntime(
                    module_id="mod-1",
                    group="missile launcher",
                    state=ModuleState.ONLINE,
                    charge_capacity=20,
                    charge_rate=1.0,
                    charge_remaining=20.0,
                    charge_reload_time=10.0,
                    effects=[
                        ModuleEffect(
                            name="launcher-a",
                            effect_class=EffectClass.PROJECTED,
                            state_required=ModuleState.ACTIVE,
                            cycle_time=5.0,
                        )
                    ],
                ),
                ModuleRuntime(
                    module_id="mod-2",
                    group="turret weapon",
                    state=ModuleState.ONLINE,
                    charge_capacity=10,
                    charge_rate=1.0,
                    charge_remaining=10.0,
                    charge_reload_time=5.0,
                    effects=[
                        ModuleEffect(
                            name="gun-a",
                            effect_class=EffectClass.PROJECTED,
                            state_required=ModuleState.ACTIVE,
                            cycle_time=5.0,
                        )
                    ],
                ),
            ],
            diagnostics={},
        )
        return runtime, fit


class RegressionTests(unittest.TestCase):
    def test_client_resends_setup_after_reconnect(self) -> None:
        lan_client = _FakeLanClient()
        query_service = SimpleNamespace(scenario_ship_specs=lambda _team: (
            ShipSetupSpec(
                ship_id="RED-001",
                squad_id="RED-ALPHA",
                ship_group_id="test",
                fit_text="[Test, Fit]",
                position=Vector2(0.0, 0.0),
                velocity=Vector2(0.0, 0.0),
                facing_deg=0.0,
                system_id="alpha",
                deployed=False,
                alive=False,
                shield=0.0,
                armor=0.0,
                structure=0.0,
                cap=0.0,
                quality_level="REGULAR",
                quality_reaction_delay=0.0,
                quality_ignore_order_probability=0.0,
                quality_formation_jitter=0.0,
            ),
        ))
        coordinator = LanMatchCoordinator(
            SimpleNamespace(query_service=query_service),
            mode="client",
            client=lan_client,
        )

        coordinator.poll_client(Team.RED)
        self.assertEqual(lan_client.sent_commands, [])

        lan_client.connected = True
        coordinator.poll_client(Team.RED)

        self.assertEqual(len(lan_client.sent_commands), 1)
        self.assertEqual(lan_client.sent_commands[0]["kind"], CMD_SYNC_SETUP)

    def test_existing_remote_ship_rebuilds_when_fit_text_changes(self) -> None:
        engine = SimpleNamespace(world=WorldState(), register_ship=lambda ship_id: None)
        replica_factory = RuntimeReplicaShipFactory(_FakeParser(), _FakeFactory())

        first = replica_factory.ensure_ship(
            engine.world,
            "RED-001",
            {"team": "RED", "squad_id": "SQ1", "fit_text": "fit-a", "quality_level": "REGULAR"},
        )
        updated = replica_factory.ensure_ship(
            engine.world,
            "RED-001",
            {"team": "RED", "squad_id": "SQ1", "fit_text": "fit-b", "quality_level": "REGULAR"},
        )

        self.assertIs(first, updated)
        self.assertEqual(updated.fit.fit_key, "fit-b")
        self.assertAlmostEqual(updated.profile.dps, 200.0)
        self.assertIsNotNone(updated.runtime)
        self.assertEqual(updated.runtime.fit_key, "fit-b")
        self.assertAlmostEqual(updated.nav.max_speed, 1700.0)

    def test_snapshot_loader_syncs_locked_module_charges(self) -> None:
        engine = SimpleNamespace(world=WorldState(), register_ship=lambda ship_id: None)
        parser = _FakeParser()
        factory = _FakeFactory()
        loader = SnapshotLoader(RuntimeReplicaShipFactory(parser, factory))
        loader.apply_delta(
            engine.world,
            {
                "tick": 1,
                "now": 0.0,
                "ships": {
                    "RED-001": {
                        "team": "RED",
                        "squad_id": "SQ1",
                        "fit_text": "fit-a",
                        "locked_module_charges": {"mod-1": ""},
                    }
                }
            },
        )

        self.assertEqual(engine.world.ships["RED-001"].locked_module_charges["mod-1"], "")

    def test_attach_logger_keeps_detailed_logging_enabled(self) -> None:
        combat = CombatSystem(PyfaBridge())
        logger = logging.getLogger("eve_sim.tests.regressions")

        combat.attach_logger(logger, detailed_logging=True, merge_window_sec=1.0, hotspot_logging=False)

        self.assertTrue(combat.detailed_logging)
        self.assertTrue(combat.event_logging_enabled)

    def test_rebuild_preserves_other_module_runtime_state_during_charge_change(self) -> None:
        ship = _make_ship("BLUE-001", team=Team.BLUE, runtime=_make_runtime("fit-a"))
        ship.runtime = _FakeAmmoFactory().build(_FakeParsedFit("fit-a"))[0]
        ship.runtime.modules[0].state = ModuleState.ACTIVE
        ship.runtime.modules[0].charge_remaining = 7.0
        ship.runtime.modules[1].state = ModuleState.ACTIVE
        ship.runtime.modules[1].charge_remaining = 3.0
        ship.combat.module_cycle_timers["mod-1"] = 4.0
        ship.combat.module_cycle_timers["mod-2"] = 4.0

        engine = SimpleNamespace(
            world=WorldState(ships={ship.ship_id: ship}),
            register_ship=lambda ship_id: None,
            combat=CombatSystem(PyfaBridge()),
        )
        service = ShipFitService(_FakeParser(), _FakeAmmoFactory(), engine.combat)
        service.replace_fit(engine.world, Team.BLUE, ship.ship_id, "fit-b")

        self.assertEqual(ship.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(ship.runtime.modules[0].charge_remaining, 7.0)
        self.assertEqual(ship.runtime.modules[1].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(ship.runtime.modules[1].charge_remaining, 3.0)
        self.assertAlmostEqual(ship.combat.module_cycle_timers["mod-1"], 4.0)
        self.assertAlmostEqual(ship.combat.module_cycle_timers["mod-2"], 4.0)

    def test_prefocus_prelock_tracks_each_ship_separately(self) -> None:
        combat = CombatSystem(PyfaBridge())
        fast = _make_ship(
            "BLUE-FAST",
            team=Team.BLUE,
            profile=_make_profile(scan_resolution=2_000.0),
        )
        slow = _make_ship(
            "BLUE-SLOW",
            team=Team.BLUE,
            profile=_make_profile(scan_resolution=150.0),
            position=Vector2(200.0, 0.0),
        )
        primary = _make_ship("RED-PRIMARY", team=Team.RED)
        prefocus = _make_ship("RED-PREFOCUS", team=Team.RED, position=Vector2(400.0, 0.0))
        world = WorldState(
            ships={
                fast.ship_id: fast,
                slow.ship_id: slow,
                primary.ship_id: primary,
                prefocus.ship_id: prefocus,
            },
            squad_focus_queues={"BLUE:SQ1": [primary.ship_id, prefocus.ship_id]},
        )

        combat._update_squad_prelocks(world, 0.0, {})
        fast_lock = fast.combat.prelock_timers[prefocus.ship_id]
        slow_lock = slow.combat.prelock_timers[prefocus.ship_id]
        self.assertLess(fast_lock, slow_lock)

        combat._update_squad_prelocks(world, fast_lock + 0.05, {})

        self.assertIn(prefocus.ship_id, fast.combat.prelocked_targets)
        self.assertNotIn(prefocus.ship_id, fast.combat.prelock_timers)
        self.assertNotIn(prefocus.ship_id, slow.combat.prelocked_targets)
        self.assertGreater(slow.combat.prelock_timers[prefocus.ship_id], 0.0)

    def test_logistics_fallback_repairs_shield_then_armor(self) -> None:
        logi = _make_ship(
            "LOGI",
            team=Team.BLUE,
            profile=_make_profile(rep_amount=30.0, rep_cycle=5.0),
        )
        ally = _make_ship("ALLY", team=Team.BLUE)
        ally.vital.shield = 90.0
        ally.vital.armor = 50.0
        world = WorldState(ships={logi.ship_id: logi, ally.ship_id: ally})

        LogisticsSystem().run(world, 5.0)

        self.assertAlmostEqual(ally.vital.shield, 100.0)
        self.assertAlmostEqual(ally.vital.armor, 70.0)

    def test_logistics_skips_runtime_backed_repairs(self) -> None:
        logi = _make_ship(
            "LOGI",
            team=Team.BLUE,
            profile=_make_profile(rep_amount=40.0, rep_cycle=5.0),
            runtime=_make_runtime("runtime-logi"),
        )
        ally = _make_ship("ALLY", team=Team.BLUE)
        ally.vital.shield = 60.0
        world = WorldState(ships={logi.ship_id: logi, ally.ship_id: ally})

        LogisticsSystem().run(world, 5.0)

        self.assertAlmostEqual(ally.vital.shield, 60.0)


if __name__ == "__main__":
    unittest.main()
