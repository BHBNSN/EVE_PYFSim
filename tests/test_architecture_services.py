from __future__ import annotations

import json

import pytest

from eve_sim.battle_report import BattleReportService
from eve_sim.fleet_setup import RuntimeFromEftFactory as EngineRuntimeFromEftFactory
from eve_sim.fleet_setup.fit_factory import RuntimeFromEftFactory
from eve_sim.replay import CombatEvent
from eve_sim.scenario import load_scenario_library, validate_scenario
from eve_sim.sde_manifest import DataVersionManifest, validate_manifest
from eve_sim.sync_service import ClientSyncService, HostSyncService


def test_data_manifest_validation_skips_unpinned_fields() -> None:
    expected = DataVersionManifest(eve_db_sha256="abc", eve_db_size=None)
    current = DataVersionManifest(eve_db_sha256="abc", eve_db_size=10)

    assert validate_manifest(expected, current) == []
    assert validate_manifest(DataVersionManifest(eve_db_sha256="def"), current) == [
        "eve_db_sha256: expected 'def', got 'abc'"
    ]


def test_scenario_library_loads_combat_smoke() -> None:
    library = load_scenario_library()

    scenario = library["combat_runtime_smoke"]
    assert scenario.duration_s == pytest.approx(60.0)
    assert {fleet.team for fleet in scenario.fleets} == {"BLUE", "RED"}
    assert validate_scenario({"scenario_id": "", "fleets": []})


def test_sync_service_wraps_host_and_client_sessions() -> None:
    class Host:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        def client_connected(self) -> bool:
            return True

        def poll_commands(self) -> list[dict]:
            return [{"kind": "PING"}]

        def send_state(self, snapshot: dict) -> None:
            self.sent.append(snapshot)

        def stop(self) -> None:
            self.stopped = True

    class Client:
        def __init__(self) -> None:
            self.commands: list[dict] = []

        def connected(self) -> bool:
            return True

        def consume_latest_state(self) -> dict | None:
            return {"tick": 1}

        def send_command(self, command: dict) -> None:
            self.commands.append(command)

        def close(self) -> None:
            self.closed = True

    host = Host()
    client = Client()
    host_service = HostSyncService(host)
    client_service = ClientSyncService(client)

    assert host_service.connected()
    assert host_service.poll_commands() == [{"kind": "PING"}]
    host_service.publish_snapshot({"tick": 2})
    assert host.sent == [{"tick": 2}]
    assert client_service.connected()
    assert client_service.consume_snapshot() == {"tick": 1}
    client_service.send_command({"kind": "MOVE"})
    assert client.commands == [{"kind": "MOVE"}]


def test_fleet_setup_facade_keeps_runtime_factory_identity() -> None:
    assert RuntimeFromEftFactory is EngineRuntimeFromEftFactory


def test_battle_report_matches_golden_contract() -> None:
    events = [
        CombatEvent(1, 1.0, "active_module_cycle_effect", "blue-1", "red-1", "gun-1", 11, 0, {"team": "BLUE", "total_damage": 100.0}),
        CombatEvent(2, 2.0, "active_module_cycle_effect", "blue-1", "red-1", "gun-1", 11, 1, {"team": "BLUE", "em": 10.0, "thermal": 5.0}),
        CombatEvent(3, 3.0, "active_module_cycle_effect", "blue-logi", "blue-1", "rep-1", 11, 2, {"team": "BLUE", "shield_repaired": 20.0, "armor_repaired": 5.0}),
        CombatEvent(4, 4.0, "ecm_jam_applied", "blue-ewar", "red-1", "jam-1", 11, 3, {"duration_s": 20.0}),
        CombatEvent(5, 5.0, "active_module_cycle", "blue-boost", None, "burst-1", 11, 4, {"team": "BLUE", "group": "Shield Command Burst", "effects": "Shield Harmonizing", "cycle_time": 60.0}),
        CombatEvent(6, 6.0, "ship_death", "blue-1", "red-1", "gun-1", 11, 5, {"target_team": "RED"}),
    ]
    report = BattleReportService().build("combat_runtime_smoke", events).to_dict()
    golden = json.loads(open("tests/goldens/combat_report_smoke.json", encoding="utf-8").read())

    assert report == golden
