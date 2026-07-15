from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
from copy import deepcopy
from types import SimpleNamespace

import pytest
import eve_sim.simulation_engine as simulation_engine_module

from eve_sim.config import EngineConfig
from eve_sim.agents import ShipAgent
from eve_sim.domain.squad_follow_service import (
    FOLLOW_LEADER_SYSTEM,
    FORMATION_FOLLOW,
    WARP_TO_LEADER,
)
from eve_sim.domain.squad_service import SquadLeadershipService
from eve_sim.math2d import Vector2
from eve_sim.models import (
    BubbleField,
    CombatState,
    DamageProfile,
    DroneBayEntry,
    DroneEntity,
    FitDescriptor,
    FighterAbilityProfile,
    FighterBayEntry,
    FighterEntity,
    ProjectileBlast,
    NavigationState,
    ProjectileEntity,
    QualityLevel,
    QualityState,
    ShipEntity,
    StructureEntity,
    Team,
    VitalState,
)
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.serialization import SnapshotBuilder
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.system_isolation import (
    DuplicateEntityIdError,
    SystemExecutionMode,
    SystemShardResult,
    SystemTransferOut,
    active_system_pressures,
    build_system_shard,
    merge_system_results,
    plan_system_execution,
    stable_system_seed,
)
from eve_sim.systems import CombatStateCloneError, CombatSystem, DeployableSystem, MovementSystem
from eve_sim.world import WorldState


class _RandomConsumingAgent(ShipAgent):
    def think(self, world: WorldState) -> None:
        world.ships[self.ship_id].nav.position.x += random.random()


def _ship(
    ship_id: str,
    system_id: str,
    *,
    team: Team = Team.BLUE,
    squad_id: str = "A",
    rep_amount: float = 0.0,
    shield: float = 100.0,
) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name="Test Hull",
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_target_range=100_000.0,
        max_cap=100.0,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
        rep_amount=rep_amount,
        rep_cycle=1.0,
    )
    profile = PyfaBridge().build_profile(fit)
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=fit,
        profile=profile,
        nav=NavigationState(
            position=Vector2(0.0, 0.0),
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            max_speed=profile.max_speed,
            system_id=system_id,
        ),
        combat=CombatState(),
        vital=VitalState(
            shield=shield,
            armor=100.0,
            structure=100.0,
            shield_max=100.0,
            armor_max=100.0,
            structure_max=100.0,
            cap=100.0,
            cap_max=100.0,
            alive=True,
        ),
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
    )


def _projectile(projectile_id: str, system_id: str, *, kind: str = "missile", target_id: str | None = None) -> ProjectileEntity:
    return ProjectileEntity(
        projectile_id=projectile_id,
        kind=kind,
        source_ship_id="source",
        source_module_id="module",
        team=Team.BLUE,
        position=Vector2(0.0, 0.0),
        velocity=Vector2(1_000.0, 0.0),
        facing_deg=0.0,
        target_ship_id=target_id,
        speed=1_000.0,
        max_speed=1_000.0,
        max_range=100_000.0,
        distance_traveled=0.0,
        flight_time=100.0,
        age=0.0,
        acceleration_time=0.0,
        damage_em=100.0,
        damage_thermal=0.0,
        damage_kinetic=0.0,
        damage_explosive=0.0,
        explosion_radius=1_000.0,
        explosion_velocity=1_000.0,
        damage_reduction_factor=0.5,
        shield=0.0,
        armor=0.0,
        structure=100.0,
        shield_max=0.0,
        armor_max=0.0,
        structure_max=100.0,
        blast_radius=5_000.0,
        system_id=system_id,
    )


def _bubble(field_id: str, system_id: str, *, speed_factor_mult: float = 1.0, destructible: bool = True) -> BubbleField:
    return BubbleField(
        field_id=field_id,
        kind="warp_disrupt_probe",
        interdiction_kind="probe",
        source_ship_id="source",
        source_module_id="module",
        team=Team.RED,
        position=Vector2(0.0, 0.0),
        radius_m=5_000.0,
        expires_at=1_000.0,
        blocks_warp=True,
        speed_factor_mult=speed_factor_mult,
        destructible=destructible,
        shield=0.0,
        armor=0.0,
        structure=100.0,
        shield_max=0.0,
        armor_max=0.0,
        structure_max=100.0,
        alive=True,
        system_id=system_id,
    )


def _gate(gate_id: str, system_id: str, linked_id: str, position: Vector2 | None = None) -> StructureEntity:
    pos = position if position is not None else Vector2(0.0, 0.0)
    return StructureEntity(
        structure_id=gate_id,
        position=Vector2(pos.x, pos.y),
        radius=1_000.0,
        interaction_range=2_500.0,
        kind="STARGATE",
        system_id=system_id,
        linked_structure_id=linked_id,
    )


def _blast(blast_id: str, system_id: str) -> ProjectileBlast:
    return ProjectileBlast(
        blast_id=blast_id,
        kind="bomb",
        position=Vector2(0.0, 0.0),
        radius_m=5_000.0,
        expires_at=10.0,
        system_id=system_id,
    )


def _drone_entry() -> DroneBayEntry:
    return DroneBayEntry(
        type_name="Test Drone",
        quantity=1,
        group_name="Combat Drone",
        bandwidth_mbit=5.0,
        volume_m3=5.0,
        max_velocity=1_000.0,
        orbit_range_m=1_000.0,
        control_range_m=20_000.0,
        cycle_time_s=1.0,
        optimal_range_m=20_000.0,
        falloff_m=0.0,
        tracking=10.0,
        damage=DamageProfile(thermal=50.0),
        shield_hp=10.0,
        armor_hp=10.0,
        structure_hp=10.0,
        signature_radius=25.0,
    )


def _fighter_entry() -> FighterBayEntry:
    ability = FighterAbilityProfile(
        ability_id="normal",
        name="Pulse Cannon",
        effect_name="normal",
        kind="normal_attack",
        cycle_time_s=1.0,
        optimal_range_m=20_000.0,
        falloff_m=0.0,
        tracking=10.0,
        damage=DamageProfile(thermal=50.0),
    )
    return FighterBayEntry(
        type_name="Test Fighter",
        quantity=1,
        group_name="Light Fighter",
        slot_kind="light",
        squadron_size=9,
        max_velocity=1_000.0,
        orbit_range_m=1_000.0,
        shield_hp=20.0,
        armor_hp=20.0,
        structure_hp=20.0,
        signature_radius=80.0,
        scan_resolution=500.0,
        abilities=(ability,),
    )


def _drone(ship_id: str, owner_id: str, system_id: str, target_id: str | None) -> DroneEntity:
    definition = _drone_entry()
    return DroneEntity(
        ship_id=ship_id,
        owner_ship_id=owner_id,
        team=Team.BLUE,
        squad_id="A",
        definition=definition,
        fit=DeployableSystem._drone_fit(definition),
        profile=DeployableSystem._drone_profile(definition),
        nav=NavigationState(Vector2(0.0, 0.0), Vector2(0.0, 0.0), 0.0, 1_000.0, system_id=system_id),
        combat=CombatState(),
        vital=VitalState(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0),
        target_id=target_id,
    )


def _fighter(ship_id: str, owner_id: str, system_id: str, target_id: str | None) -> FighterEntity:
    definition = _fighter_entry()
    return FighterEntity(
        ship_id=ship_id,
        owner_ship_id=owner_id,
        team=Team.BLUE,
        squad_id=DeployableSystem.fighter_squad_id("A"),
        definition=definition,
        fit=DeployableSystem._fighter_fit(definition),
        profile=DeployableSystem._fighter_profile(definition),
        nav=NavigationState(Vector2(0.0, 0.0), Vector2(0.0, 0.0), 0.0, 1_000.0, system_id=system_id),
        combat=CombatState(),
        vital=VitalState(20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0),
        target_id=target_id,
        owner_squad_id="A",
    )


def test_isolated_systems_prevent_cross_system_legacy_logistics() -> None:
    logi = _ship("BLUE-ALPHA-001", "alpha", rep_amount=50.0)
    damaged_ally = _ship("BLUE-BETA-001", "beta", shield=10.0)
    world = WorldState(ships={logi.ship_id: logi, damaged_ally.ship_id: damaged_ally})
    engine = SimulationEngine(
        world,
        EngineConfig(tick_rate=1, physics_substeps=1, parallel_systems=False),
        CombatSystem(PyfaBridge()),
    )
    engine.register_ship(logi.ship_id)
    engine.register_ship(damaged_ally.ship_id)

    engine.step()

    assert len(engine.last_system_execution_plan.active_systems) == 2
    assert world.ships[damaged_ally.ship_id].vital.shield == 10.0


def test_pressure_plan_skips_empty_systems_groups_small_systems_and_splits_large_work() -> None:
    world = WorldState(
        ships={
            "alpha": _ship("alpha", "alpha"),
            "beta": _ship("beta", "beta"),
            "gamma": _ship("gamma", "gamma"),
        },
        structures={
            "empty_gate": StructureEntity(
                structure_id="empty_gate",
                position=Vector2(0.0, 0.0),
                radius=1_000.0,
                interaction_range=2_500.0,
                kind="STARGATE",
                system_id="empty",
            )
        },
    )

    small_plan = plan_system_execution(
        world,
        EngineConfig(parallel_system_target_pressure=1_000.0),
        cpu_count=8,
    )
    assert len(small_plan.active_systems) == 3
    assert {item.system_id for item in small_plan.active_systems} == {"alpha", "beta", "gamma"}
    assert len(small_plan.groups) == 1
    assert small_plan.groups[0].system_ids == ("alpha", "beta", "gamma")
    assert not small_plan.use_processes

    split_plan = plan_system_execution(
        world,
        EngineConfig(parallel_system_workers=4, parallel_system_target_pressure=1.0),
        cpu_count=8,
    )
    assert len(split_plan.groups) == 3
    assert not split_plan.use_processes

    parallel_plan = plan_system_execution(
        world,
        EngineConfig(parallel_systems=True, parallel_system_workers=4, parallel_system_target_pressure=1.0),
        cpu_count=8,
    )
    assert len(parallel_plan.groups) == 3
    assert parallel_plan.use_processes


def test_pressure_plan_includes_transient_only_systems_and_skips_blank_system_id() -> None:
    world = WorldState(
        ships={
            "alpha": _ship("alpha", "alpha"),
            "beta": _ship("beta", "beta"),
        },
        projectiles={
            "gamma_projectile": _projectile("gamma_projectile", "gamma"),
            "blank_projectile": _projectile("blank_projectile", ""),
        },
        projectile_blasts={"delta_blast": _blast("delta_blast", "delta")},
        bubble_fields={"epsilon_bubble": _bubble("epsilon_bubble", "epsilon")},
    )

    pressures = active_system_pressures(world)
    plan = plan_system_execution(world, EngineConfig(parallel_system_target_pressure=1_000.0), cpu_count=8)

    assert set(pressures) == {"alpha", "beta", "gamma", "delta", "epsilon"}
    assert pressures["gamma"].projectile_count == 1
    assert pressures["delta"].projectile_count == 1
    assert pressures["epsilon"].bubble_count == 1
    assert {item.system_id for item in plan.active_systems} == {"alpha", "beta", "gamma", "delta", "epsilon"}


def test_isolated_step_ticks_projectile_only_system() -> None:
    alpha = _ship("alpha", "alpha")
    beta = _ship("beta", "beta")
    bomb = _projectile("gamma_bomb", "gamma", kind="bomb")
    bomb.flight_time = 0.01
    world = WorldState(
        ships={alpha.ship_id: alpha, beta.ship_id: beta},
        projectiles={bomb.projectile_id: bomb},
    )
    engine = SimulationEngine(
        world,
        EngineConfig(tick_rate=1, physics_substeps=1, parallel_systems=False),
        CombatSystem(PyfaBridge()),
    )
    engine.register_ship(alpha.ship_id)
    engine.register_ship(beta.ship_id)

    engine.step()

    assert {item.system_id for item in engine.last_system_execution_plan.active_systems} == {"alpha", "beta", "gamma"}
    assert "gamma_bomb" not in world.projectiles
    assert any(blast.system_id == "gamma" for blast in world.projectile_blasts.values())


def test_unassigned_active_entity_is_rejected_without_switching_to_global_execution() -> None:
    alpha = _ship("alpha", "alpha")
    beta = _ship("beta", "beta")
    bomb = _projectile("blank_bomb", "", kind="bomb")
    bomb.flight_time = 0.01
    world = WorldState(
        ships={alpha.ship_id: alpha, beta.ship_id: beta},
        projectiles={bomb.projectile_id: bomb},
    )
    engine = SimulationEngine(
        world,
        EngineConfig(tick_rate=1, physics_substeps=1, parallel_systems=False),
        CombatSystem(PyfaBridge()),
    )
    engine.register_ship(alpha.ship_id)
    engine.register_ship(beta.ship_id)

    with pytest.raises(ValueError, match="system_id"):
        engine.step()

    assert "blank_bomb" in world.projectiles
    assert world.tick == 0
    assert world.now == 0.0
    assert engine.system_execution_mode is SystemExecutionMode.SHARD_SERIAL


def test_legacy_world_requires_explicit_global_mode() -> None:
    world = WorldState(ships={"legacy": _ship("legacy", "")})
    strict_engine = SimulationEngine(world, EngineConfig(), CombatSystem(PyfaBridge()))

    with pytest.raises(ValueError, match="system_id"):
        strict_engine.step()

    legacy_world = WorldState(ships={"legacy": _ship("legacy", "")})
    legacy_engine = SimulationEngine(
        legacy_world,
        EngineConfig(isolate_systems=False),
        CombatSystem(PyfaBridge()),
    )
    legacy_engine.step()
    assert legacy_engine.system_execution_mode is SystemExecutionMode.GLOBAL_SERIAL
    assert legacy_world.tick == 1


def test_bomb_explosion_ignores_targets_in_other_systems() -> None:
    source = _ship("source", "alpha", team=Team.BLUE)
    target = _ship("target", "beta", team=Team.RED)
    world = WorldState(ships={source.ship_id: source, target.ship_id: target})
    bomb = _projectile("bomb", "alpha", kind="bomb")
    combat = CombatSystem(PyfaBridge())

    combat._resolve_bomb_explosion(world, bomb)

    assert target.vital.shield == target.vital.shield_max


def test_smartbomb_area_cleanup_ignores_projectiles_and_bubbles_in_other_systems() -> None:
    world = WorldState(
        projectiles={
            "alpha_proj": _projectile("alpha_proj", "alpha"),
            "beta_proj": _projectile("beta_proj", "beta"),
        },
        bubble_fields={
            "alpha_bubble": _bubble("alpha_bubble", "alpha"),
            "beta_bubble": _bubble("beta_bubble", "beta"),
        },
    )
    combat = CombatSystem(PyfaBridge())

    combat._destroy_projectiles_in_area(
        world,
        center=Vector2(0.0, 0.0),
        radius_m=5_000.0,
        damage=(500.0, 0.0, 0.0, 0.0),
        system_id="alpha",
    )
    combat._destroy_bubbles_in_area(
        world,
        center=Vector2(0.0, 0.0),
        radius_m=5_000.0,
        damage=(500.0, 0.0, 0.0, 0.0),
        system_id="alpha",
    )

    assert "alpha_proj" not in world.projectiles
    assert "alpha_bubble" not in world.bubble_fields
    assert "beta_proj" in world.projectiles
    assert "beta_bubble" in world.bubble_fields


def test_cross_system_bubble_does_not_intercept_warp_or_reduce_speed() -> None:
    ship = _ship("ship", "alpha")
    ship.profile.warp_speed_au_s = 3.0
    ship.profile.warp_capacitor_need = 0.0
    ship.nav.warp.phase = "align"
    ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
    ship.nav.warp.align_timeout = 0.0
    world = WorldState(ships={ship.ship_id: ship}, bubble_fields={"bubble": _bubble("bubble", "beta", speed_factor_mult=0.1, destructible=False)})
    movement = MovementSystem()

    movement._prepare_warp_alignment(world, ship)
    movement._finalize_warp_alignment(world, ship, 1.0)

    assert ship.nav.warp.phase == "warp"
    assert len(ship.nav.warp.interdiction_snapshots) == 0
    assert movement._bubble_speed_multiplier(world, ship) == 1.0


def test_missile_loses_target_that_moves_to_another_system() -> None:
    source = _ship("source", "alpha", team=Team.BLUE)
    target = _ship("target", "beta", team=Team.RED)
    projectile = _projectile("missile", "alpha", target_id=target.ship_id)
    world = WorldState(
        ships={source.ship_id: source, target.ship_id: target},
        projectiles={projectile.projectile_id: projectile},
    )
    combat = CombatSystem(PyfaBridge())

    combat._advance_projectiles(world, 0.1)

    assert target.vital.shield == target.vital.shield_max
    assert world.projectiles[projectile.projectile_id].target_ship_id is None


def test_deployable_targets_crossing_systems_are_cleared() -> None:
    owner = _ship("owner", "alpha", team=Team.BLUE)
    target = _ship("target", "beta", team=Team.RED)
    drone = _drone("drone", owner.ship_id, "alpha", target.ship_id)
    fighter = _fighter("fighter", owner.ship_id, "alpha", target.ship_id)
    world = WorldState(
        ships={owner.ship_id: owner, target.ship_id: target},
        drones={drone.ship_id: drone},
        fighters={fighter.ship_id: fighter},
    )
    deployables = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())

    assert deployables._target_for_asset(world, drone) is None
    assert deployables._target_for_asset(world, fighter) is None
    assert drone.target_id is None
    assert fighter.target_id is None


def test_transfer_survives_destination_shard_merge() -> None:
    moved = _ship("moved", "beta")
    world = WorldState(ships={"moved": _ship("moved", "alpha")})
    combat = CombatSystem(PyfaBridge())
    alpha_result = SystemShardResult(
        system_id="alpha",
        world=WorldState(ships={"moved": moved}),
        combat=combat,
        owned_entity_ids={"ships": {"moved"}},
        transfer_outs=[
            SystemTransferOut(
                collection_name="ships",
                entity_id="moved",
                source_system_id="alpha",
                destination_system_id="beta",
                entity=moved,
            )
        ],
    )
    beta_result = SystemShardResult(
        system_id="beta",
        world=WorldState(ships={}),
        combat=CombatSystem(PyfaBridge()),
        owned_entity_ids={"ships": set()},
    )
    transfer_ins = []

    merge_system_results(world, [alpha_result, beta_result], {}, transfer_sink=transfer_ins)

    assert "moved" in world.ships
    assert world.ships["moved"].nav.system_id == "beta"
    assert len(transfer_ins) == 1
    assert transfer_ins[0].entity_id == "moved"
    assert transfer_ins[0].source_system_id == "alpha"
    assert transfer_ins[0].destination_system_id == "beta"


def test_stargate_jump_emits_transfer_out_and_transfer_in() -> None:
    jumper = _ship("jumper", "alpha", squad_id="A")
    beta_ship = _ship("beta-guard", "beta", squad_id="B")
    jumper.nav.position = Vector2(0.0, 0.0)
    jumper.nav.gate.target_structure_id = "alpha_gate"
    world = WorldState(
        ships={jumper.ship_id: jumper, beta_ship.ship_id: beta_ship},
        structures={
            "alpha_gate": _gate("alpha_gate", "alpha", "beta_gate", Vector2(0.0, 0.0)),
            "beta_gate": _gate("beta_gate", "beta", "alpha_gate", Vector2(100_000.0, 0.0)),
        },
    )
    engine = SimulationEngine(
        world,
        EngineConfig(tick_rate=1, physics_substeps=1, parallel_systems=False),
        CombatSystem(PyfaBridge()),
    )
    engine.register_ship(jumper.ship_id)
    engine.register_ship(beta_ship.ship_id)

    engine.step()

    assert world.ships[jumper.ship_id].nav.system_id == "beta"
    assert len(engine.last_system_transfers) == 1
    transfer = engine.last_system_transfers[0]
    assert transfer.collection_name == "ships"
    assert transfer.entity_id == jumper.ship_id
    assert transfer.source_system_id == "alpha"
    assert transfer.destination_system_id == "beta"


def test_single_active_system_stays_on_shard_path_and_preserves_combat_state() -> None:
    alpha = _ship("alpha", "alpha")
    beta = _ship("beta", "beta")
    world = WorldState(ships={"alpha": alpha, "beta": beta})
    engine = SimulationEngine(world, EngineConfig(parallel_systems=False), CombatSystem(PyfaBridge()))
    engine.register_ship("alpha")
    engine.register_ship("beta")

    engine.step()
    alpha_combat = engine._system_combats["alpha"]
    alpha_combat._projectile_seq = 17
    beta.vital.alive = False
    world.ships["beta"].vital.alive = False
    engine.step()

    assert engine.system_execution_mode is SystemExecutionMode.SHARD_SERIAL
    assert tuple(item.system_id for item in engine.last_system_execution_plan.active_systems) == ("alpha",)
    assert engine._system_combats["alpha"]._projectile_seq == 17


def test_empty_world_advances_without_switching_execution_mode() -> None:
    engine = SimulationEngine(WorldState(), EngineConfig(), CombatSystem(PyfaBridge()))

    engine.step()

    assert engine.world.tick == 1
    assert engine.world.now == pytest.approx(1.0)
    assert engine.system_execution_mode is SystemExecutionMode.SHARD_SERIAL
    assert engine._system_combats == {}


def test_process_configuration_reports_single_group_effective_serial_mode() -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha")})
    engine = SimulationEngine(
        world,
        EngineConfig(parallel_systems=True, parallel_system_workers=2),
        CombatSystem(PyfaBridge()),
    )

    engine.step()

    assert engine.system_execution_mode is SystemExecutionMode.SHARD_PROCESS
    assert engine.last_effective_system_execution_mode == "shard_serial_single_group"
    engine.close()


def test_combat_clone_preserves_authority_state_and_is_independent() -> None:
    combat = CombatSystem(PyfaBridge())
    combat._projectile_seq = 9
    combat._projectile_blast_seq = 8
    combat._bubble_seq = 7
    combat._event_rng_counter = 6
    combat._merged_event_buckets[("probe",)] = {"count": 1}

    cloned = combat.clone_for_system("alpha")
    cloned._projectile_seq += 1
    cloned._merged_event_buckets[("probe",)]["count"] = 2

    assert cloned._projectile_blast_seq == 8
    assert cloned._bubble_seq == 7
    assert cloned._event_rng_counter == 6
    assert combat._projectile_seq == 9
    assert combat._merged_event_buckets[("probe",)]["count"] == 1
    assert cloned.logger is None
    assert cloned._combat_event_sink is None


def test_shard_worker_does_not_commit_temporary_logging_configuration(tmp_path) -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha")})
    engine = SimulationEngine(
        world,
        EngineConfig(
            detailed_logging=True,
            hotspot_logging=True,
            detail_log_file=str(tmp_path / "detail.log"),
            hotspot_log_file=str(tmp_path / "hotspot.log"),
        ),
        CombatSystem(PyfaBridge()),
    )

    engine.step()
    combat = engine._system_combats["alpha"]

    assert combat.detailed_logging is True
    assert combat.hotspot_logging_enabled is True
    assert combat.event_logging_enabled is True


def test_system_seed_is_stable_and_namespaced() -> None:
    assert stable_system_seed(42, "alpha") == stable_system_seed(42, "alpha")
    assert stable_system_seed(42, "alpha") != stable_system_seed(42, "beta")
    assert stable_system_seed(42, "alpha") != stable_system_seed(43, "alpha")


def test_two_system_blast_ids_do_not_collide() -> None:
    alpha_bomb = _projectile("alpha-bomb", "alpha", kind="bomb")
    beta_bomb = _projectile("beta-bomb", "beta", kind="bomb")
    alpha_bomb.flight_time = 0.01
    beta_bomb.flight_time = 0.01
    world = WorldState(projectiles={alpha_bomb.projectile_id: alpha_bomb, beta_bomb.projectile_id: beta_bomb})
    engine = SimulationEngine(world, EngineConfig(parallel_systems=False), CombatSystem(PyfaBridge()))

    engine.step()

    assert set(world.projectile_blasts) == {"blast:alpha:1", "blast:beta:1"}


def test_two_system_projectile_and_static_bubble_ids_do_not_collide() -> None:
    world = WorldState()
    projectile_effect = SimpleNamespace(
        range_m=1_000.0,
        projected_add={"weapon_projectile_speed": 1_000.0, "weapon_projectile_flight_time": 1.0},
    )
    bubble_effect = SimpleNamespace(
        local_add={"bubble_radius_m": 1_000.0, "bubble_duration_sec": 5.0}
    )
    module = SimpleNamespace(module_id="module")
    metadata = SimpleNamespace(is_bomb_launcher=False)

    for system_id in ("alpha", "beta"):
        source = _ship(f"{system_id}-source", system_id)
        combat = CombatSystem(PyfaBridge()).clone_for_system(system_id)
        combat._spawn_projectile(
            world,
            source=source,
            module=module,
            metadata=metadata,
            effect=projectile_effect,
            target_id=None,
        )
        combat._spawn_static_bubble_field(
            world,
            source=source,
            module=module,
            effect=bubble_effect,
        )

    assert set(world.projectiles) == {"proj:alpha:1", "proj:beta:1"}
    assert set(world.bubble_fields) == {"bubble:alpha:1", "bubble:beta:1"}


def test_duplicate_entity_id_aborts_atomic_merge() -> None:
    original = _ship("duplicate", "alpha")
    world = WorldState(ships={"duplicate": original})
    before = deepcopy(world)
    result = SystemShardResult(
        system_id="beta",
        world=WorldState(ships={"duplicate": _ship("duplicate", "beta")}),
        combat=CombatSystem(PyfaBridge()),
        owned_entity_ids={"ships": set()},
    )

    with pytest.raises(DuplicateEntityIdError):
        merge_system_results(world, [result], {})

    assert world == before
    assert world.ships["duplicate"] is original


def test_same_squad_shards_read_one_global_leader() -> None:
    alpha = _ship("alpha-leader", "alpha", squad_id="A")
    beta = _ship("beta-member", "beta", squad_id="A")
    target = _ship("alpha-target", "alpha", team=Team.RED, squad_id="B")
    world = WorldState(
        ships={alpha.ship_id: alpha, beta.ship_id: beta, target.ship_id: target},
        squad_leaders={"BLUE:A": alpha.ship_id},
        squad_focus_queues={"BLUE:A": [target.ship_id]},
    )
    SquadLeadershipService().refresh(world)

    alpha_task = build_system_shard(world, "alpha", {})
    beta_task = build_system_shard(world, "beta", {})

    assert alpha_task.world.squad_leaders["BLUE:A"] == alpha.ship_id
    assert beta_task.world.squad_leaders["BLUE:A"] == alpha.ship_id
    assert alpha_task.world.squad_leader_locations["BLUE:A"].system_id == "alpha"
    assert beta_task.world.squad_leader_locations["BLUE:A"].system_id == "alpha"
    assert alpha_task.world.squad_focus_queues["BLUE:A"] == [target.ship_id]
    assert beta_task.world.squad_focus_queues["BLUE:A"] == []


def test_parallel_failure_circuit_breaks_and_falls_back_once(monkeypatch) -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha"), "beta": _ship("beta", "beta")})
    engine = SimulationEngine(
        world,
        EngineConfig(parallel_systems=True, parallel_system_workers=2, parallel_system_target_pressure=1.0),
        CombatSystem(PyfaBridge()),
    )
    attempts = {"count": 0}

    def fail_parallel(*_args, **_kwargs):
        attempts["count"] += 1
        raise TimeoutError("worker timed out")

    monkeypatch.setattr(engine, "_run_parallel_groups", fail_parallel)
    engine.step()
    engine.step()

    assert attempts["count"] == 1
    assert world.tick == 2
    assert engine.system_execution_mode is SystemExecutionMode.SHARD_SERIAL_DEGRADED
    assert engine.parallel_failure_count == 1
    assert engine.parallel_disabled_at_tick == 1
    assert "timed out" in str(engine.parallel_disabled_reason)


def test_shard_rng_is_continuous_and_does_not_mutate_process_rng() -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha")})
    engine = SimulationEngine(world, EngineConfig(simulation_seed=1234), CombatSystem(PyfaBridge()))
    engine.ship_agents["alpha"] = _RandomConsumingAgent(agent_id="random:alpha", ship_id="alpha")
    process_state = random.getstate()
    initial_state = engine._random_state_for_system("alpha")

    engine.step()
    first_state = engine._system_random_states["alpha"]
    engine.step()

    assert random.getstate() == process_state
    assert first_state != initial_state
    assert engine._system_random_states["alpha"] != first_state


def test_invalid_combat_commit_is_rejected_before_world_commit(monkeypatch) -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha"), "beta": _ship("beta", "beta")})
    before = deepcopy(world)
    engine = SimulationEngine(world, EngineConfig(parallel_systems=False), CombatSystem(PyfaBridge()))
    original_run = simulation_engine_module.run_system_group

    def corrupt_second_combat(*args, **kwargs):
        group_result = original_run(*args, **kwargs)
        if len(group_result.results) > 1:
            group_result.results[1].combat._system_id = "wrong-system"
        return group_result

    monkeypatch.setattr(simulation_engine_module, "run_system_group", corrupt_second_combat)

    with pytest.raises(CombatStateCloneError):
        engine.step()

    assert world == before
    assert world.tick == 0
    assert world.now == 0.0


def test_event_sink_failure_does_not_rollback_committed_tick(monkeypatch) -> None:
    world = WorldState(ships={"alpha": _ship("alpha", "alpha")})
    engine = SimulationEngine(world, EngineConfig(), CombatSystem(PyfaBridge()))

    def fail_delivery(_events):
        raise RuntimeError("recorder unavailable")

    monkeypatch.setattr(engine, "_emit_isolated_events", fail_delivery)
    engine.step()

    assert world.tick == 1
    assert world.now == pytest.approx(1.0)
    assert engine._isolated_commit_completed


def test_leader_system_change_increments_version_and_clears_focus() -> None:
    leader = _ship("leader", "alpha")
    target = _ship("target", "alpha", team=Team.RED, squad_id="B")
    world = WorldState(
        ships={leader.ship_id: leader, target.ship_id: target},
        squad_leaders={"BLUE:A": leader.ship_id},
        squad_focus_queues={"BLUE:A": [target.ship_id]},
        squad_focus_updated_at={"BLUE:A": 1.0},
    )
    SquadLeadershipService().refresh(world)
    initial_version = world.squad_leader_location_versions["BLUE:A"]

    leader.nav.system_id = "beta"
    SquadLeadershipService().refresh(world)

    assert world.squad_leader_location_versions["BLUE:A"] == initial_version + 1
    assert "BLUE:A" not in world.squad_focus_queues
    assert "BLUE:A" not in world.squad_focus_updated_at


def test_dead_leader_is_replaced_deterministically_by_local_same_group() -> None:
    dead = _ship("leader", "alpha")
    dead.ship_group_id = "command"
    dead.vital.alive = False
    local_other = _ship("local-other", "alpha")
    remote_same = _ship("remote-same", "beta")
    remote_same.ship_group_id = "command"
    local_same = _ship("local-same", "alpha")
    local_same.ship_group_id = "command"
    world = WorldState(
        ships={ship.ship_id: ship for ship in (dead, local_other, remote_same, local_same)},
        squad_leaders={"BLUE:A": dead.ship_id},
    )

    SquadLeadershipService().refresh(world)

    assert world.squad_leaders["BLUE:A"] == local_same.ship_id


def test_cross_system_member_routes_to_global_leader_and_drops_combat() -> None:
    follower = _ship("follower", "alpha")
    leader = _ship("leader", "gamma")
    enemy = _ship("enemy", "alpha", team=Team.RED, squad_id="B")
    follower.combat.current_target = enemy.ship_id
    world = WorldState(
        ships={ship.ship_id: ship for ship in (follower, leader, enemy)},
        structures={
            "alpha-beta": _gate("alpha-beta", "alpha", "beta-alpha", Vector2(100_000.0, 0.0)),
            "beta-alpha": _gate("beta-alpha", "beta", "alpha-beta"),
            "beta-gamma": _gate("beta-gamma", "beta", "gamma-beta"),
            "gamma-beta": _gate("gamma-beta", "gamma", "beta-gamma"),
        },
        squad_leaders={"BLUE:A": leader.ship_id},
        squad_focus_queues={"BLUE:A": [enemy.ship_id]},
    )
    SquadLeadershipService().refresh(world)
    agent = ShipAgent(agent_id="agent:follower", ship_id=follower.ship_id)

    agent.think(world)

    assert follower.nav.squad_follow_state == FOLLOW_LEADER_SYSTEM
    assert follower.nav.gate.target_structure_id == "alpha-beta"
    assert follower.nav.squad_follow_leader_id == leader.ship_id
    assert follower.combat.current_target is None
    assert not any(order.kind == "ATTACK" for order in follower.order_queue)


def test_same_system_follow_warp_uses_170_150_km_hysteresis() -> None:
    leader = _ship("leader", "alpha")
    follower = _ship("follower", "alpha")
    leader.nav.position = Vector2(200_000.0, 0.0)
    world = WorldState(
        ships={leader.ship_id: leader, follower.ship_id: follower},
        squad_leaders={"BLUE:A": leader.ship_id},
    )
    SquadLeadershipService().refresh(world)
    agent = ShipAgent(agent_id="agent:follower", ship_id=follower.ship_id)

    agent.think(world)
    assert follower.nav.squad_follow_state == WARP_TO_LEADER
    assert follower.nav.warp.phase == "align"
    assert follower.nav.warp.target_ship_id == leader.ship_id
    assert not follower.nav.squad_follow_warp_ready

    follower.nav.position = Vector2(40_000.0, 0.0)
    agent.think(world)
    assert follower.nav.squad_follow_state == WARP_TO_LEADER

    follower.nav.position = Vector2(51_000.0, 0.0)
    agent.think(world)
    assert follower.nav.squad_follow_state == FORMATION_FOLLOW
    assert follower.nav.warp.phase == "idle"
    assert follower.nav.squad_follow_warp_ready


def test_same_system_follow_waits_for_existing_warp_to_finish() -> None:
    leader = _ship("leader", "alpha")
    follower = _ship("follower", "alpha")
    leader.nav.position = Vector2(200_000.0, 0.0)
    follower.nav.warp.phase = "warp"
    follower.nav.warp.target_position = Vector2(-200_000.0, 0.0)
    world = WorldState(
        ships={leader.ship_id: leader, follower.ship_id: follower},
        squad_leaders={"BLUE:A": leader.ship_id},
    )
    SquadLeadershipService().refresh(world)

    ShipAgent(agent_id="agent:follower", ship_id=follower.ship_id).think(world)

    assert follower.nav.warp.phase == "warp"
    assert follower.nav.warp.target_ship_id is None
    assert follower.nav.squad_follow_state == WARP_TO_LEADER


def test_leader_system_change_cancels_old_local_warp_and_replans_gate() -> None:
    leader = _ship("leader", "alpha")
    follower = _ship("follower", "alpha")
    leader.nav.position = Vector2(200_000.0, 0.0)
    world = WorldState(
        ships={leader.ship_id: leader, follower.ship_id: follower},
        structures={
            "alpha-beta": _gate("alpha-beta", "alpha", "beta-alpha", Vector2(100_000.0, 0.0)),
            "beta-alpha": _gate("beta-alpha", "beta", "alpha-beta"),
        },
        squad_leaders={"BLUE:A": leader.ship_id},
    )
    SquadLeadershipService().refresh(world)
    agent = ShipAgent(agent_id="agent:follower", ship_id=follower.ship_id)
    agent.think(world)
    old_version = follower.nav.squad_follow_leader_location_version
    assert follower.nav.warp.target_ship_id == leader.ship_id

    leader.nav.system_id = "beta"
    SquadLeadershipService().refresh(world)
    agent.think(world)

    assert follower.nav.squad_follow_state == FOLLOW_LEADER_SYSTEM
    assert follower.nav.squad_follow_leader_location_version == old_version + 1
    assert follower.nav.warp.target_ship_id is None
    assert follower.nav.gate.target_structure_id == "alpha-beta"


def test_shards_cannot_overwrite_global_squad_authority() -> None:
    world = WorldState(
        squad_leaders={"BLUE:A": "leader"},
        squad_propulsion_commands={"BLUE:A": True},
        squad_leader_speed_limits={"BLUE:A": 321.0},
        squad_focus_queues={"BLUE:A": ["target"]},
        squad_focus_updated_at={"BLUE:A": 12.0},
    )
    alpha_result = SystemShardResult(
        system_id="alpha",
        world=WorldState(
            squad_leaders={"BLUE:A": "alpha-local"},
            squad_propulsion_commands={"BLUE:A": False},
            squad_leader_speed_limits={"BLUE:A": 1.0},
            squad_focus_queues={"BLUE:A": ["alpha-target"]},
            squad_focus_updated_at={"BLUE:A": 1.0},
        ),
        combat=CombatSystem(PyfaBridge()),
    )
    beta_result = SystemShardResult(
        system_id="beta",
        world=WorldState(
            squad_leaders={"BLUE:A": "beta-local"},
            squad_propulsion_commands={"BLUE:A": False},
            squad_leader_speed_limits={"BLUE:A": 2.0},
            squad_focus_queues={"BLUE:A": ["beta-target"]},
            squad_focus_updated_at={"BLUE:A": 2.0},
        ),
        combat=CombatSystem(PyfaBridge()),
    )

    merge_system_results(world, [alpha_result, beta_result], {})

    assert world.squad_propulsion_commands["BLUE:A"] is True
    assert world.squad_leaders["BLUE:A"] == "leader"
    assert world.squad_leader_speed_limits["BLUE:A"] == pytest.approx(321.0)
    assert world.squad_focus_queues["BLUE:A"] == ["target"]
    assert world.squad_focus_updated_at["BLUE:A"] == pytest.approx(12.0)


def test_close_is_idempotent_and_shuts_down_executor() -> None:
    class FakeExecutor:
        _processes = {}

        def __init__(self) -> None:
            self.calls = 0

        def shutdown(self, **_kwargs) -> None:
            self.calls += 1

    engine = SimulationEngine(WorldState(), EngineConfig(), CombatSystem(PyfaBridge()))
    executor = FakeExecutor()
    engine._system_executor = executor  # type: ignore[assignment]
    engine._system_executor_workers = 2

    engine.close()
    engine.close()

    assert executor.calls == 1
    assert engine.system_executor_workers == 0


def test_close_terminates_worker_that_does_not_join() -> None:
    class HungProcess:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False

        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    class FakeExecutor:
        def __init__(self, process) -> None:
            self._processes = {1: process}

        def shutdown(self, **_kwargs) -> None:
            return None

    process = HungProcess()
    engine = SimulationEngine(WorldState(), EngineConfig(), CombatSystem(PyfaBridge()))
    engine._system_executor = FakeExecutor(process)  # type: ignore[assignment]

    engine.close(timeout_sec=0.01)

    assert process.terminated
    assert engine.system_executor_workers == 0


def test_executor_worker_capacity_is_fixed_after_first_creation() -> None:
    engine = SimulationEngine(WorldState(), EngineConfig(parallel_systems=True), CombatSystem(PyfaBridge()))
    first_plan = SimpleNamespace(worker_count=2, groups=(object(), object()))
    second_plan = SimpleNamespace(worker_count=3, groups=(object(), object(), object()))

    first_executor = engine._executor_for_plan(first_plan)  # type: ignore[arg-type]
    second_executor = engine._executor_for_plan(second_plan)  # type: ignore[arg-type]

    assert second_executor is first_executor
    assert engine.system_executor_workers == 2
    engine.close()


def test_parallel_systems_true_runs_real_process_pool(tmp_path) -> None:
    script = tmp_path / "parallel_smoke.py"
    script.write_text(
        textwrap.dedent(
            """
            from eve_sim.config import EngineConfig
            from eve_sim.agents import ShipAgent
            from eve_sim.math2d import Vector2
            from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, Team, VitalState
            from eve_sim.pyfa_bridge import PyfaBridge
            from eve_sim.serialization import SnapshotBuilder
            from eve_sim.simulation_engine import SimulationEngine
            from eve_sim.system_isolation import SystemExecutionMode
            from eve_sim.systems import CombatSystem
            from eve_sim.world import WorldState

            class SlowAgent(ShipAgent):
                def think(self, world):
                    import time
                    time.sleep(0.5)

            class RandomAgent(ShipAgent):
                def think(self, world):
                    import random
                    world.ships[self.ship_id].nav.position.x += random.random()

            def hang_forever():
                import time
                while True:
                    time.sleep(10.0)

            def ship(ship_id, system_id):
                fit = FitDescriptor(fit_key=ship_id, ship_name="Test Hull", role="test", base_dps=0.0, volley=0.0, optimal_range=0.0, falloff=0.0, tracking=0.0, max_target_range=100000.0, max_cap=100.0, shield_hp=100.0, armor_hp=100.0, structure_hp=100.0)
                profile = PyfaBridge().build_profile(fit)
                return ShipEntity(
                    ship_id=ship_id,
                    team=Team.BLUE,
                    squad_id=ship_id,
                    fit=fit,
                    profile=profile,
                    nav=NavigationState(Vector2(0.0, 0.0), Vector2(0.0, 0.0), 0.0, profile.max_speed, system_id=system_id),
                    combat=CombatState(),
                    vital=VitalState(100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                    quality=QualityState(QualityLevel.REGULAR, 0.0, 0.0, 0.0),
                )

            def main():
                world = WorldState(ships={"alpha": ship("alpha", "alpha"), "beta": ship("beta", "beta")})
                serial_world = WorldState(ships={"alpha": ship("alpha", "alpha"), "beta": ship("beta", "beta")})
                engine = SimulationEngine(
                    world,
                    EngineConfig(parallel_systems=True, parallel_system_workers=2, parallel_system_target_pressure=1.0),
                    CombatSystem(PyfaBridge()),
                )
                engine.register_ship("alpha")
                engine.register_ship("beta")
                engine.ship_agents["alpha"] = RandomAgent(agent_id="random:alpha", ship_id="alpha")
                engine.ship_agents["beta"] = RandomAgent(agent_id="random:beta", ship_id="beta")
                serial_engine = SimulationEngine(
                    serial_world,
                    EngineConfig(parallel_systems=False, parallel_system_target_pressure=1.0),
                    CombatSystem(PyfaBridge()),
                )
                serial_engine.register_ship("alpha")
                serial_engine.register_ship("beta")
                serial_engine.ship_agents["alpha"] = RandomAgent(agent_id="random:alpha", ship_id="alpha")
                serial_engine.ship_agents["beta"] = RandomAgent(agent_id="random:beta", ship_id="beta")
                for _ in range(100):
                    engine.step()
                    serial_engine.step()
                assert engine.last_system_execution_plan.use_processes
                assert engine.last_system_parallel_error is None
                parallel_snapshot = SnapshotBuilder().build(world)
                serial_snapshot = SnapshotBuilder().build(serial_world)
                for key in (
                        "ships", "drones", "fighters", "projectiles", "projectile_blasts",
                        "bubble_fields", "intents", "squad_leaders",
                        "squad_leader_location_versions", "squad_propulsion_commands",
                        "squad_leader_speed_limits", "squad_focus_queues",
                        "squad_focus_updated_at",
                    ):
                    assert parallel_snapshot[key] == serial_snapshot[key], key
                assert engine._system_random_states == serial_engine._system_random_states
                assert {
                    key: (value._projectile_seq, value._projectile_blast_seq, value._bubble_seq, value._event_rng_counter)
                    for key, value in engine._system_combats.items()
                } == {
                    key: (value._projectile_seq, value._projectile_blast_seq, value._bubble_seq, value._event_rng_counter)
                    for key, value in serial_engine._system_combats.items()
                }
                serial_engine.close()
                engine.config.parallel_system_timeout_sec = 0.05
                engine.ship_agents["alpha"] = SlowAgent(agent_id="slow:alpha", ship_id="alpha")
                engine.step()
                assert engine.system_execution_mode is SystemExecutionMode.SHARD_SERIAL_DEGRADED
                assert engine.parallel_failure_count == 1
                assert engine.parallel_disabled_at_tick == 101
                engine.close()
                assert engine.system_executor_workers == 0

                import multiprocessing
                import time
                from concurrent.futures import ProcessPoolExecutor
                hung_engine = SimulationEngine(WorldState(), EngineConfig(), CombatSystem(PyfaBridge()))
                hung_executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                hung_engine._system_executor = hung_executor
                hung_engine._system_executor_workers = 1
                hung_executor.submit(hang_forever)
                deadline = time.monotonic() + 5.0
                while not getattr(hung_executor, "_processes", {}) and time.monotonic() < deadline:
                    time.sleep(0.01)
                started = time.monotonic()
                hung_engine.close(timeout_sec=0.1)
                assert time.monotonic() - started < 3.0

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
