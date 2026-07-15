from __future__ import annotations

import pytest

from eve_sim.battle_report import BattleReportService
from eve_sim.math2d import Vector2
from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, ShipProfile, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.replay import CombatEvent, ReplayPlayer, ReplayRecorder
from eve_sim.serialization import SnapshotLoader
from eve_sim.systems import CombatSystem
from eve_sim.world import WorldState


def _profile(structure_hp: float = 10.0) -> ShipProfile:
    return ShipProfile(
        dps=0.0,
        volley=0.0,
        optimal=10_000.0,
        falloff=0.0,
        tracking=1.0,
        sig_radius=120.0,
        scan_resolution=300.0,
        max_target_range=120_000.0,
        max_speed=1_000.0,
        max_cap=100.0,
        cap_recharge_time=100.0,
        shield_hp=0.0,
        armor_hp=0.0,
        structure_hp=structure_hp,
        rep_amount=0.0,
        rep_cycle=5.0,
    )


def _ship(ship_id: str, team: Team, *, structure_hp: float = 10.0) -> ShipEntity:
    profile = _profile(structure_hp)
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name=ship_id,
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=10_000.0,
        falloff=0.0,
        tracking=1.0,
        shield_hp=0.0,
        armor_hp=0.0,
        structure_hp=structure_hp,
    )
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id="SQ1",
        fit=fit,
        profile=profile,
        nav=NavigationState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.0), facing_deg=0.0, max_speed=0.0),
        combat=CombatState(),
        vital=VitalState(shield=0.0, armor=0.0, structure=structure_hp, shield_max=0.0, armor_max=0.0, structure_max=structure_hp, cap=100.0, cap_max=100.0),
        quality=QualityState(QualityLevel.REGULAR, 0.0, 0.0, 0.0),
    )


def test_replay_recorder_round_trips_events(tmp_path) -> None:
    event = CombatEvent(
        tick=3,
        at=1.5,
        kind="active_module_cycle_effect",
        source_id="blue-1",
        target_id="red-1",
        module_id="gun-1",
        rng_seed=42,
        rng_counter=7,
        payload={"team": "BLUE", "total_damage": 125.0},
    )
    recorder = ReplayRecorder("smoke", rng_seed=42)
    recorder.record(event)
    recorder.record_snapshot(
        {
            "tick": 3,
            "now": 1.5,
            "ships": {
                "blue-1": {
                    "ship_id": "blue-1",
                    "team": "BLUE",
                    "squad_id": "SQ1",
                    "ship_name": "Rifter",
                    "alive": True,
                    "position": {"x": 10.0, "y": 20.0},
                    "velocity": {"x": 1.0, "y": 2.0},
                    "shield": 50.0,
                    "armor": 40.0,
                    "structure": 30.0,
                    "shield_max": 100.0,
                    "armor_max": 100.0,
                    "structure_max": 100.0,
                    "cap": 80.0,
                    "cap_max": 100.0,
                }
            },
        }
    )
    recorder.record_snapshot(
        {
            "tick": 4,
            "now": 1.6,
            "ships": {
                "blue-1": {
                    "ship_id": "blue-1",
                    "team": "BLUE",
                    "squad_id": "SQ1",
                    "ship_name": "Rifter",
                    "alive": True,
                    "position": {"x": 20.0, "y": 25.0},
                    "velocity": {"x": 2.0, "y": 2.0},
                    "shield": 45.0,
                    "armor": 40.0,
                    "structure": 30.0,
                    "shield_max": 100.0,
                    "armor_max": 100.0,
                    "structure_max": 100.0,
                    "cap": 80.0,
                    "cap_max": 100.0,
                }
            },
        }
    )

    replay_path = tmp_path / "smoke.replay.json"
    recorder.save(replay_path)
    raw_saved = replay_path.read_text(encoding="utf-8")

    loaded = ReplayRecorder.load(replay_path)
    player = ReplayPlayer.from_file(replay_path)

    assert loaded.to_dict() == recorder.to_dict()
    assert '"snapshots"' not in raw_saved
    assert '"schema_version"' not in raw_saved
    assert '"source_id"' not in raw_saved
    assert '"ship_name"' not in raw_saved
    assert '"v":5' in raw_saved
    assert '"si":"blue-1"' in raw_saved
    assert '"sn":"Rifter"' in raw_saved
    assert loaded.to_dict()["frames"][0]["kind"] == "keyframe"
    assert loaded.to_dict()["frames"][1]["kind"] == "delta"
    assert "ship_name" not in loaded.to_dict()["frames"][1]["patch"]["ships"]["blue-1"]
    assert [item.to_dict() for item in player.iter_events(kind="active_module_cycle_effect")] == [event.to_dict()]
    assert player.snapshot_count == 2
    assert player.duration_s == pytest.approx(1.6)
    assert player.snapshot_at_index(10).snapshot["ships"]["blue-1"]["ship_name"] == "Rifter"
    assert player.snapshot_at_index(10).snapshot["ships"]["blue-1"]["position"] == {"x": 20.0, "y": 25.0}


def test_replay_delta_recomputes_predictable_motion_and_module_timers() -> None:
    recorder = ReplayRecorder("predictive")
    first = {
        "tick": 1,
        "now": 1.0,
        "ships": {
            "blue-1": {
                "ship_id": "blue-1",
                "team": "BLUE",
                "squad_id": "SQ1",
                "ship_name": "Rifter",
                "alive": True,
                "position": {"x": 0.0, "y": 0.0},
                "velocity": {"x": 10.0, "y": -5.0},
                "module_cycle_timers": {"gun-1": 5.0},
                "shield": 50.0,
                "armor": 40.0,
                "structure": 30.0,
                "shield_max": 100.0,
                "armor_max": 100.0,
                "structure_max": 100.0,
            }
        },
    }
    second = {
        "tick": 2,
        "now": 2.0,
        "ships": {
            "blue-1": {
                "ship_id": "blue-1",
                "team": "BLUE",
                "squad_id": "SQ1",
                "ship_name": "Rifter",
                "alive": True,
                "position": {"x": 10.0, "y": -5.0},
                "velocity": {"x": 10.0, "y": -5.0},
                "module_cycle_timers": {"gun-1": 4.0},
                "shield": 50.0,
                "armor": 40.0,
                "structure": 30.0,
                "shield_max": 100.0,
                "armor_max": 100.0,
                "structure_max": 100.0,
            }
        },
    }

    recorder.record_snapshot(first)
    recorder.record_snapshot(second, force_frame=True)

    assert recorder.frame_count == 2
    assert recorder.to_dict()["frames"][1]["patch"] == {}

    player = ReplayPlayer.from_dict(recorder.to_dict())
    resolved = player.snapshot_at_index(1).snapshot["ships"]["blue-1"]
    assert resolved["position"] == {"x": 10.0, "y": -5.0}
    assert resolved["module_cycle_timers"] == {"gun-1": 4.0}


def test_battle_report_service_aggregates_event_stream() -> None:
    events = [
        CombatEvent(1, 1.0, "active_module_cycle_effect", "blue-1", "red-1", "gun-1", 11, 0, {"team": "BLUE", "total_damage": 100.0}),
        CombatEvent(2, 2.0, "active_module_cycle_effect", "blue-1", "red-1", "gun-1", 11, 1, {"team": "BLUE", "em": 10.0, "thermal": 5.0}),
        CombatEvent(3, 3.0, "active_module_cycle_effect", "blue-logi", "blue-1", "rep-1", 11, 2, {"team": "BLUE", "shield_repaired": 20.0, "armor_repaired": 5.0}),
        CombatEvent(4, 4.0, "ecm_jam_applied", "blue-ewar", "red-1", "jam-1", 11, 3, {"duration_s": 20.0}),
        CombatEvent(5, 5.0, "active_module_cycle", "blue-boost", None, "burst-1", 11, 4, {"team": "BLUE", "group": "Shield Command Burst", "effects": "Shield Harmonizing", "cycle_time": 60.0}),
        CombatEvent(6, 6.0, "ship_death", "blue-1", "red-1", "gun-1", 11, 5, {"target_team": "RED"}),
    ]

    report = BattleReportService().build("baseline", events)

    assert report.duration_s == pytest.approx(6.0)
    assert report.total_damage_by_team == {"BLUE": 115.0}
    assert report.rep_applied_by_team == {"BLUE": 25.0}
    assert report.jam_uptime_by_target == {"red-1": 20.0}
    assert report.burst_coverage_by_effect == {"Shield Harmonizing": 60.0}
    assert report.ship_deaths == [
        {
            "tick": 6,
            "at": 6.0,
            "ship_id": "red-1",
            "source_id": "blue-1",
            "module_id": "gun-1",
            "team": "RED",
        }
    ]


def test_snapshot_loader_rebuilds_display_world() -> None:
    world = WorldState()
    SnapshotLoader().apply_replica(
        world,
        {
            "tick": 12,
            "now": 4.0,
            "ships": {
                "blue-1": {
                    "team": "BLUE",
                    "squad_id": "SQ1",
                    "ship_name": "Rifter",
                    "alive": True,
                    "position": {"x": 100.0, "y": 200.0},
                    "velocity": {"x": 10.0, "y": 0.0},
                    "facing_deg": 45.0,
                    "system_id": "sys-a",
                    "shield": 50.0,
                    "armor": 40.0,
                    "structure": 30.0,
                    "shield_max": 100.0,
                    "armor_max": 100.0,
                    "structure_max": 100.0,
                    "cap": 80.0,
                    "cap_max": 100.0,
                    "target": "red-1",
                    "projected_targets": {"web-1": "red-1"},
                    "ecm_jam_sources": {"red-ecm": 7.0},
                }
            },
            "projectiles": {
                "proj:1": {
                    "kind": "missile",
                    "source_ship_id": "blue-1",
                    "source_module_id": "launcher-1",
                    "team": "BLUE",
                    "position": {"x": 150.0, "y": 210.0},
                    "system_id": "sys-a",
                    "target_ship_id": "red-1",
                    "distance_traveled": 500.0,
                    "flight_time": 10.0,
                    "age": 1.0,
                    "blast_radius": 1_000.0,
                }
            },
            "projectile_blasts": {
                "blast:1": {
                    "kind": "bomb",
                    "position": {"x": 200.0, "y": 220.0},
                    "system_id": "sys-a",
                    "radius_m": 2_000.0,
                    "expires_at": 5.0,
                }
            },
            "bubble_fields": {
                "bubble:1": {
                    "kind": "warp_disruption",
                    "interdiction_kind": "bubble",
                    "source_ship_id": "blue-1",
                    "source_module_id": "hic-1",
                    "team": "BLUE",
                    "position": {"x": 300.0, "y": 400.0},
                    "system_id": "sys-a",
                    "radius_m": 12_000.0,
                    "expires_at": 30.0,
                    "blocks_warp": True,
                    "speed_factor_mult": 0.5,
                    "alive": True,
                }
            },
            "squad_focus_queues": {"BLUE:SQ1": ["red-1"]},
        },
    )

    assert world.tick == 12
    assert world.now == pytest.approx(4.0)
    assert world.ships["blue-1"].nav.position == Vector2(100.0, 200.0)
    assert world.ships["blue-1"].combat.current_target == "red-1"
    assert world.projectiles["proj:1"].blast_radius == pytest.approx(1_000.0)
    assert world.projectile_blasts["blast:1"].radius_m == pytest.approx(2_000.0)
    assert world.bubble_fields["bubble:1"].blocks_warp is True
    assert world.squad_focus_queues == {"BLUE:SQ1": ["red-1"]}


def test_combat_system_event_sink_receives_merged_events_without_detailed_logging() -> None:
    recorder = ReplayRecorder("combat-smoke")
    combat = CombatSystem(PyfaBridge(), combat_event_sink=recorder)
    combat._current_event_tick = 7
    combat._current_event_at = 3.5

    combat._queue_merged_event(
        "active_module_cycle_effect",
        merge_fields={"source": "blue-1", "target": "red-1", "module": "gun-1", "team": "BLUE"},
        sum_fields={"total_damage": 25.0},
    )
    combat.flush_pending_events()

    assert combat.event_logging_enabled is False
    assert len(recorder.events) == 1
    assert recorder.events[0].kind == "active_module_cycle_effect"
    assert recorder.events[0].source_id == "blue-1"
    assert recorder.events[0].target_id == "red-1"
    assert recorder.events[0].module_id == "gun-1"
    assert recorder.events[0].payload["total_damage"] == pytest.approx(25.0)


def test_combat_system_records_ship_death_from_direct_damage() -> None:
    recorder = ReplayRecorder("death-smoke")
    combat = CombatSystem(PyfaBridge(), combat_event_sink=recorder)
    combat._current_event_tick = 9
    combat._current_event_at = 4.0
    source = _ship("blue-1", Team.BLUE)
    target = _ship("red-1", Team.RED)
    world = WorldState(tick=9, now=4.0, ships={source.ship_id: source, target.ship_id: target})

    applied = combat._apply_direct_damage(
        world,
        source=source,
        target=target,
        target_profile=target.profile,
        damage=(20.0, 0.0, 0.0, 0.0),
        damage_factor=1.0,
        module_id="launcher-1",
    )
    combat.flush_pending_events()

    assert applied == pytest.approx(10.0)
    assert target.vital.alive is False
    assert [event.kind for event in recorder.events] == ["direct_damage_applied", "ship_death"]
    damage = recorder.events[0]
    assert damage.source_id == "blue-1"
    assert damage.target_id == "red-1"
    assert damage.module_id == "launcher-1"
    assert damage.payload["total_damage"] == pytest.approx(10.0)
    death = recorder.events[1]
    assert death.kind == "ship_death"
    assert death.source_id == "blue-1"
    assert death.target_id == "red-1"
    assert death.module_id == "launcher-1"
    assert death.payload["applied_damage"] == pytest.approx(10.0)
