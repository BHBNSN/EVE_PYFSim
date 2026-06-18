from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from eve_sim.config import EngineConfig
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
    FleetIntent,
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
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.system_isolation import SystemShardResult, SystemTransferOut, active_system_pressures, merge_system_results, plan_system_execution
from eve_sim.systems import CombatSystem, DeployableSystem, MovementSystem
from eve_sim.world import WorldState


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


def test_unassigned_active_entity_uses_global_fallback_instead_of_being_dropped_from_isolated_plan() -> None:
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

    engine.step()

    assert "blank_bomb" not in world.projectiles


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


def test_intent_merge_only_clears_consumed_intents() -> None:
    consumed = FleetIntent(squad_id="A", target_position=Vector2(1.0, 0.0))
    persistent = FleetIntent(squad_id="B", target_position=Vector2(2.0, 0.0))
    world = WorldState(intents={"BLUE:A": consumed, "BLUE:B": persistent})
    result = SystemShardResult(
        system_id="alpha",
        world=WorldState(),
        combat=CombatSystem(PyfaBridge()),
        consumed_intent_keys={"BLUE:A"},
    )

    merge_system_results(world, [result], {})

    assert "BLUE:A" not in world.intents
    assert world.intents["BLUE:B"] == persistent


def test_parallel_systems_true_runs_real_process_pool(tmp_path) -> None:
    script = tmp_path / "parallel_smoke.py"
    script.write_text(
        textwrap.dedent(
            """
            from eve_sim.config import EngineConfig
            from eve_sim.math2d import Vector2
            from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, Team, VitalState
            from eve_sim.pyfa_bridge import PyfaBridge
            from eve_sim.simulation_engine import SimulationEngine
            from eve_sim.systems import CombatSystem
            from eve_sim.world import WorldState

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
                engine = SimulationEngine(
                    world,
                    EngineConfig(parallel_systems=True, parallel_system_workers=2, parallel_system_target_pressure=1.0),
                    CombatSystem(PyfaBridge()),
                )
                engine.register_ship("alpha")
                engine.register_ship("beta")
                engine.step()
                assert engine.last_system_execution_plan.use_processes
                assert engine.last_system_parallel_error is None
                engine.shutdown_parallel_workers()

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
