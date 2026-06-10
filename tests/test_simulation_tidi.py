from __future__ import annotations

import unittest
from types import SimpleNamespace

from eve_sim.config import EngineConfig
from eve_sim.gui.main_window import MainWindow
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.world import WorldState


class _NoopCombat:
    def attach_logger(self, *_args, **_kwargs) -> None:
        pass

    def run(self, *_args, **_kwargs) -> None:
        pass


class _FakeLanServer:
    client_connected = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_state(self, packet: dict) -> None:
        self.sent.append(packet)


class _FakeLanClient:
    connected = True

    def __init__(self, packets: list[dict | None]) -> None:
        self.packets = list(packets)
        self.sent_commands: list[dict] = []

    def consume_latest_state(self) -> dict | None:
        if not self.packets:
            return None
        return self.packets.pop(0)

    def send_command(self, command: dict) -> None:
        self.sent_commands.append(command)


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

        local = SimpleNamespace(network_mode="local", engine=SimpleNamespace(tidi_tick_interval_ms=lambda: 2500))
        client = SimpleNamespace(network_mode="client", _client_poll_interval_ms=50)

        self.assertEqual(MainWindow._tick_timer_interval_ms(local), 2500)
        self.assertEqual(MainWindow._tick_timer_interval_ms(client), 50)

    def test_host_snapshot_sync_is_not_legacy_20hz_throttled(self) -> None:
        lan_server = _FakeLanServer()
        world = SimpleNamespace(tick=1, now=1.0, ships={}, map_definition=None, squad_focus_queues={})
        engine = SimpleNamespace(
            world=world,
            snapshot=lambda: {"tick": world.tick, "now": world.now, "ships": {}, "squad_focus_queues": {}},
        )
        dummy = SimpleNamespace(
            lan_server=lan_server,
            engine=engine,
            _last_full_snapshot_sync_at=0.0,
            _snapshot_full_sync_interval_sec=30.0,
            _last_sent_ship_signatures={},
            _last_sent_fit_texts={},
            _undeployed_ship_ids=set(),
            _ship_locked_module_charges={},
            _effective_tidi_factor=lambda: 1.0,
            _engine_config_payload=lambda: {"tick_rate": 1},
            _lan_debug=lambda _message: None,
        )

        MainWindow._send_host_state(dummy, countdown_left=0.0, started=True)
        MainWindow._send_host_state(dummy, countdown_left=0.0, started=True)

        self.assertEqual(len(lan_server.sent), 2)
        self.assertEqual(lan_server.sent[0]["snapshot"]["tick"], 1)

    def test_client_records_only_when_authoritative_snapshot_arrives(self) -> None:
        lan_client = _FakeLanClient([None, {"snapshot": {"tick": 2, "now": 2.0, "ships": {}}, "lan": {}}])
        recorded = {"count": 0}
        dummy = SimpleNamespace()
        dummy.network_mode = "client"
        dummy.lan_client = lan_client
        dummy._setup_synced = True
        dummy._flush_tick_ops = lambda: None
        dummy._build_setup_sync_payload = lambda: []
        dummy._apply_remote_snapshot = lambda _packet: None
        dummy._update_approach_targets = lambda: None
        dummy._ui_tick_counter = 0
        dummy._ui_refresh_interval_ticks = 99
        dummy._overview_refresh_interval_ticks = 99
        dummy._sync_blue_squads = lambda: None
        dummy.request_overview_refresh = lambda force=False: None
        dummy.refresh_blue_roster = lambda: None
        dummy.engine = SimpleNamespace(world=SimpleNamespace(tick=1))
        dummy._record_battle_snapshot = lambda: recorded.__setitem__("count", recorded["count"] + 1)

        MainWindow.on_tick(dummy)
        MainWindow.on_tick(dummy)

        self.assertEqual(recorded["count"], 1)


if __name__ == "__main__":
    unittest.main()
