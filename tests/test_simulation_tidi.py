from __future__ import annotations

import unittest
from types import SimpleNamespace

from eve_sim.config import EngineConfig
from eve_sim.gui.main_window import MainWindow
from eve_sim.lan_snapshot_adapter import LanSnapshotPublisher
from eve_sim.lan_match_coordinator import ClientPollResult, LanMatchCoordinator
from eve_sim.models import Team
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.world import WorldState


class _NoopCombat:
    def attach_logger(self, *_args, **_kwargs) -> None:
        pass

    def run(self, *_args, **_kwargs) -> None:
        pass


class _CountingMovement:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def run(self, _world, dt: float) -> None:
        self.calls.append(float(dt))


class _CountingCombat(_NoopCombat):
    def __init__(self) -> None:
        self.calls: list[float] = []

    def run(self, _world, dt: float) -> None:
        self.calls.append(float(dt))


class _CountingDeployables:
    def __init__(self) -> None:
        self.logic_calls: list[tuple[float, bool, bool]] = []
        self.physics_calls: list[float] = []

    def run(self, _world, dt: float, *, advance_physics: bool = True, apply_effects: bool = True) -> None:
        self.logic_calls.append((float(dt), bool(advance_physics), bool(apply_effects)))

    def run_physics(self, _world, dt: float) -> None:
        self.physics_calls.append(float(dt))


class _CountingLogistics:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def run(self, _world, dt: float) -> None:
        self.calls.append(float(dt))


class _FakeLanServer:
    client_connected = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_state(self, packet: dict) -> None:
        self.sent.append(packet)


class SimulationTidiTests(unittest.TestCase):
    def _make_engine(self, config: EngineConfig | None = None) -> SimulationEngine:
        return SimulationEngine(WorldState(), config or EngineConfig(), _NoopCombat())  # type: ignore[arg-type]

    def test_default_engine_tick_is_authoritative_one_hz(self) -> None:
        engine = self._make_engine()

        self.assertEqual(engine.config.tick_rate, 1)
        self.assertAlmostEqual(engine.nominal_tick_interval_ms(), 1000.0)
        engine.step()
        self.assertEqual(engine.world.tick, 1)
        self.assertAlmostEqual(engine.world.now, 1.0)

    def test_physics_substeps_only_repeat_position_velocity_systems(self) -> None:
        combat = _CountingCombat()
        engine = SimulationEngine(WorldState(), EngineConfig(tick_rate=1, physics_substeps=4), combat)  # type: ignore[arg-type]
        movement = _CountingMovement()
        deployables = _CountingDeployables()
        logistics = _CountingLogistics()
        engine.movement = movement  # type: ignore[assignment]
        engine.deployables = deployables  # type: ignore[assignment]
        engine.logistics = logistics  # type: ignore[assignment]

        engine.step()

        self.assertEqual(len(movement.calls), 4)
        self.assertEqual(len(deployables.physics_calls), 4)
        self.assertEqual(combat.calls, [1.0])
        self.assertEqual(logistics.calls, [1.0])
        self.assertEqual(
            deployables.logic_calls,
            [
                (1.0, False, False),
                (1.0, False, True),
            ],
        )

    def test_tidi_uses_elapsed_step_time_over_nominal_budget(self) -> None:
        engine = self._make_engine()

        engine.update_tidi_after_step(2500.0)

        self.assertAlmostEqual(engine.tidi_factor, 0.4)
        self.assertEqual(engine.tidi_tick_interval_ms(), 2500)
        self.assertEqual(engine.next_tick_delay_ms(), 1)
        self.assertAlmostEqual(engine.last_step_budget_ms, 1000.0)
        self.assertAlmostEqual(engine.last_step_ms, 2500.0)

    def test_tidi_clamps_to_configured_minimum(self) -> None:
        engine = self._make_engine(EngineConfig(tidi_min_factor=0.1))

        engine.update_tidi_after_step(20_000.0)

        self.assertAlmostEqual(engine.tidi_factor, 0.1)
        self.assertEqual(engine.tidi_tick_interval_ms(), 10_000)

    def test_tidi_recovers_to_full_speed_when_step_is_within_budget(self) -> None:
        engine = self._make_engine()
        engine.update_tidi_after_step(2500.0)

        engine.update_tidi_after_step(900.0)

        self.assertAlmostEqual(engine.tidi_factor, 1.0)
        self.assertEqual(engine.tidi_tick_interval_ms(), 1000)
        self.assertEqual(engine.next_tick_delay_ms(), 100)

    def test_main_window_tidi_helpers_format_and_schedule(self) -> None:
        self.assertEqual(MainWindow._format_tidi_percent(1.0), "100%")
        self.assertEqual(MainWindow._format_tidi_percent(0.375), "38%")

        local = SimpleNamespace(
            network_mode="local",
            application=SimpleNamespace(next_tick_delay_ms=lambda: 2500),
        )
        client = SimpleNamespace(network_mode="client", _client_poll_interval_ms=50)

        self.assertEqual(MainWindow._tick_timer_interval_ms(local), 2500)
        self.assertEqual(MainWindow._tick_timer_interval_ms(client), 50)

    def test_host_snapshot_sync_is_not_legacy_20hz_throttled(self) -> None:
        lan_server = _FakeLanServer()
        world = SimpleNamespace(tick=1, now=1.0, ships={}, map_definition=None, squad_focus_queues={})
        application = SimpleNamespace(
            snapshot=lambda: {"tick": world.tick, "now": world.now, "ships": {}, "squad_focus_queues": {}},
            tidi_factor=lambda: 1.0,
        )
        coordinator = LanMatchCoordinator(application, mode="host", server=lan_server)

        coordinator.publish(countdown_left=0.0, started=True)
        coordinator.publish(countdown_left=0.0, started=True)

        self.assertEqual(len(lan_server.sent), 2)
        self.assertEqual(lan_server.sent[0]["snapshot"]["tick"], 1)

    def test_lan_snapshot_publisher_owns_ship_delta_state(self) -> None:
        lan_server = _FakeLanServer()
        snapshot = {
            "tick": 1,
            "now": 1.0,
            "ships": {"blue": {"ship_id": "blue", "fit_text": "fit-a", "shield": 100.0}},
            "simulation_metadata": {"engine_config": {"tick_rate": 1}},
        }
        source = SimpleNamespace(snapshot=lambda: snapshot)
        publisher = LanSnapshotPublisher(lan_server, source, full_sync_interval_sec=30.0)

        first = publisher.publish(tidi_factor=1.0)
        second = publisher.publish(tidi_factor=1.0)
        snapshot["ships"]["blue"]["fit_text"] = "fit-b"
        third = publisher.publish(tidi_factor=1.0)

        self.assertTrue(first.full_sync)
        self.assertEqual(second.changed_ship_count, 0)
        self.assertEqual(lan_server.sent[1]["snapshot"]["ships"], {})
        self.assertEqual(third.changed_ship_count, 1)
        self.assertEqual(lan_server.sent[2]["snapshot"]["ships"]["blue"]["fit_text"], "fit-b")

    def test_client_records_only_when_authoritative_snapshot_arrives(self) -> None:
        recorded = {"count": 0}
        dummy = SimpleNamespace()
        dummy.network_mode = "client"
        poll_results = iter((ClientPollResult(False), ClientPollResult(True)))
        dummy.network = SimpleNamespace(poll_client=lambda _team: next(poll_results))
        dummy.controlled_team = Team.RED
        dummy._ui_tick_counter = 0
        dummy._ui_refresh_interval_ticks = 99
        dummy._overview_refresh_interval_ticks = 99
        dummy._sync_blue_squads = lambda: None
        dummy.request_overview_refresh = lambda force=False: None
        dummy.refresh_blue_roster = lambda: None
        dummy.runtime_view = SimpleNamespace(world=SimpleNamespace(tick=1), refresh=lambda: None)
        dummy._consume_removed_ship_ids = lambda _ids: None
        dummy._reschedule_tick_timer = lambda: None
        dummy.recording = SimpleNamespace(
            record_snapshot=lambda: recorded.__setitem__("count", recorded["count"] + 1)
        )

        MainWindow.on_tick(dummy)
        MainWindow.on_tick(dummy)

        self.assertEqual(recorded["count"], 1)


if __name__ == "__main__":
    unittest.main()
