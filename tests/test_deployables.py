from __future__ import annotations

import pytest

from eve_sim.fleet_setup import EftFitParser, RuntimeFromEftFactory
from eve_sim.math2d import Vector2
from eve_sim.models import (
    CombatState,
    DamageProfile,
    DeployableControlState,
    DroneBayEntry,
    FitDescriptor,
    FighterAbilityProfile,
    FighterBayEntry,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipEntity,
    ShipProfile,
    Team,
    VitalState,
)
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.config import EngineConfig
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem, DeployableSystem, MovementSystem
from eve_sim.world import WorldState


def _fit(name: str = "Test Hull") -> FitDescriptor:
    return FitDescriptor(
        fit_key=name,
        ship_name=name,
        role="TEST",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_speed=1_000.0,
        shield_hp=1_000.0,
        armor_hp=1_000.0,
        structure_hp=1_000.0,
    )


def _profile(max_speed: float = 1_000.0) -> ShipProfile:
    return ShipProfile(
        dps=0.0,
        volley=0.0,
        optimal=0.0,
        falloff=0.0,
        tracking=0.0,
        sig_radius=120.0,
        scan_resolution=300.0,
        max_target_range=120_000.0,
        max_speed=max_speed,
        max_cap=1_000.0,
        cap_recharge_time=100.0,
        shield_hp=1_000.0,
        armor_hp=1_000.0,
        structure_hp=1_000.0,
        rep_amount=0.0,
        rep_cycle=5.0,
    )


def _ship(ship_id: str, team: Team, squad_id: str, position: Vector2) -> ShipEntity:
    profile = _profile()
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=_fit(ship_id),
        profile=profile,
        nav=NavigationState(position=position, velocity=Vector2(0.0, 0.0), facing_deg=0.0, max_speed=profile.max_speed),
        combat=CombatState(),
        vital=VitalState(
            shield=profile.shield_hp,
            armor=profile.armor_hp,
            structure=profile.structure_hp,
            shield_max=profile.shield_hp,
            armor_max=profile.armor_hp,
            structure_max=profile.structure_hp,
            cap=profile.max_cap,
            cap_max=profile.max_cap,
        ),
        quality=QualityState(QualityLevel.REGULAR, 0.0, 0.0, 0.0),
    )


def _drone_entry(*, quantity: int = 10, bandwidth: float = 5.0, sentry: bool = False) -> DroneBayEntry:
    return DroneBayEntry(
        type_name="Test Drone",
        quantity=quantity,
        group_name="Sentry Drone" if sentry else "Combat Drone",
        bandwidth_mbit=bandwidth,
        volume_m3=5.0,
        max_velocity=0.0 if sentry else 2_000.0,
        orbit_range_m=0.0 if sentry else 1_000.0,
        control_range_m=20_000.0,
        cycle_time_s=1.0,
        optimal_range_m=20_000.0,
        falloff_m=0.0,
        tracking=10.0,
        damage=DamageProfile(thermal=100.0),
        shield_hp=10.0,
        armor_hp=10.0,
        structure_hp=10.0,
        signature_radius=25.0,
        is_sentry=sentry,
    )


def _fighter_entry(*, quantity: int = 1) -> FighterBayEntry:
    normal = FighterAbilityProfile(
        ability_id="normal",
        name="Pulse Cannon",
        effect_name="normal",
        kind="normal_attack",
        cycle_time_s=1.0,
        optimal_range_m=30_000.0,
        falloff_m=0.0,
        tracking=10.0,
        damage=DamageProfile(thermal=40.0),
    )
    heavy = FighterAbilityProfile(
        ability_id="heavy",
        name="Heavy Rocket",
        effect_name="heavy",
        kind="heavy_attack",
        cycle_time_s=1.0,
        optimal_range_m=30_000.0,
        falloff_m=0.0,
        tracking=10.0,
        damage=DamageProfile(thermal=120.0),
        ammo_capacity=1,
        reload_time_s=10.0,
    )
    mwd = FighterAbilityProfile(
        ability_id="mwd",
        name="Microwarpdrive",
        effect_name="mwd",
        kind="mwd",
        cycle_time_s=10.0,
        optimal_range_m=0.0,
        falloff_m=0.0,
        tracking=0.0,
        damage=DamageProfile(),
        speed_bonus_pct=500.0,
        duration_s=5.0,
        cooldown_s=20.0,
    )
    return FighterBayEntry(
        type_name="Test Fighter",
        quantity=quantity,
        group_name="Light Fighter",
        slot_kind="light",
        squadron_size=9,
        max_velocity=3_000.0,
        orbit_range_m=1_000.0,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
        signature_radius=80.0,
        scan_resolution=500.0,
        warp_speed_au_s=3.0,
        abilities=(normal, heavy, mwd),
    )


def test_eft_parser_preserves_cargo_stack_quantities() -> None:
    parsed = EftFitParser().parse("[Vexor, Drones]\nHobgoblin II x5\nCurator II x2")

    assert [(item.item_name, item.quantity) for item in parsed.cargo_specs or []] == [
        ("Hobgoblin II", 5),
        ("Curator II", 2),
    ]


def test_eft_parser_ignores_pyfa_mutation_detail_blocks() -> None:
    fit_text = """[Typhoon Fleet Issue, Mutated]
Heavy Gremlin Compact Energy Neutralizer [1]
Hobgoblin II x5 [2]


[1] Heavy Gremlin Compact Energy Neutralizer
  Unstable Heavy Energy Neutralizer Mutaplasmid
  capacitorNeed 500.0, cpu 32.0

[2] Hobgoblin II
  Decayed Drone Mutaplasmid
  maxVelocity 5000.0
"""

    parsed = EftFitParser().parse(fit_text)

    assert parsed.module_names == ["Heavy Gremlin Compact Energy Neutralizer"]
    assert parsed.module_specs[0].module_name == "Heavy Gremlin Compact Energy Neutralizer"
    assert parsed.module_specs[0].mutation_ref == 1
    assert [(item.item_name, item.quantity) for item in parsed.cargo_specs or []] == [("Hobgoblin II", 5)]
    assert parsed.mutation_specs is not None
    assert parsed.mutation_specs[1].mutaplasmid_name == "Unstable Heavy Energy Neutralizer Mutaplasmid"
    assert parsed.mutation_specs[1].attributes == {"capacitorNeed": 500.0, "cpu": 32.0}
    assert all("Mutaplasmid" not in spec.module_name for spec in parsed.module_specs)


def test_eft_parser_rejects_non_english_names_in_mutation_blocks() -> None:
    parser = EftFitParser()
    fit_text = (
        "[Typhoon Fleet Issue, Localized Mutation]\n"
        "Heavy Ghoul Compact Energy Nosferatu [1]\n\n"
        "[1] 重型盗墓者紧凑型掠能器\n"
        "  不稳定的重型掠能器突变质体\n"
        "  cpu 40.0, maxRange 16000.0, power 1800.0, powerTransferAmount 110.0\n"
    )

    with pytest.raises(Exception) as ctx:
        parser.parse(fit_text)

    assert "Mutation block contains non-English item names" in str(ctx.value)


def test_eft_parser_recognizes_pyfa_implant_and_booster_sections_when_db_available() -> None:
    parser = EftFitParser()
    if parser._classifier.kind_for("High-grade Amulet Alpha") is None:
        pytest.skip("pyfa static database is unavailable")

    parsed = parser.parse(
        "[Vexor, Additions]\n"
        "Drone Damage Amplifier II\n\n\n"
        "High-grade Amulet Alpha\n"
        "Synth Crash Booster\n\n\n"
        "Nanite Repair Paste x10\n"
    )

    assert parsed.module_names == ["Drone Damage Amplifier II"]
    assert parsed.implant_names == ["High-grade Amulet Alpha"]
    assert parsed.booster_names == ["Synth Crash Booster"]
    assert [(item.item_name, item.quantity) for item in parsed.cargo_specs or []] == [("Nanite Repair Paste", 10)]


def test_pyfa_factory_builds_fit_with_pyfa_exported_implants_and_boosters() -> None:
    factory = RuntimeFromEftFactory()
    if not factory._pyfa.available or not factory._pyfa.fit_engine_ready:
        pytest.skip(factory.backend_status)

    parsed = EftFitParser().parse(
        "[Vexor, Additions]\n"
        "Drone Damage Amplifier II\n\n\n"
        "High-grade Amulet Alpha\n"
        "Synth Crash Booster\n"
    )

    runtime, fit = factory.build(parsed)

    assert runtime.fit_key == parsed.fit_key
    assert fit.ship_name == "Vexor"
    assert runtime.diagnostics["pyfa_blueprint"]["implants"] == ["High-grade Amulet Alpha"]
    assert runtime.diagnostics["pyfa_blueprint"]["boosters"] == ["Synth Crash Booster"]


def test_pyfa_factory_applies_pyfa_exported_mutated_module_attributes() -> None:
    factory = RuntimeFromEftFactory()
    if not factory._pyfa.available or not factory._pyfa.fit_engine_ready:
        pytest.skip(factory.backend_status)

    parsed = EftFitParser().parse(
        "[Typhoon Fleet Issue, Mutated]\n"
        "Heavy Gremlin Compact Energy Neutralizer [1]\n\n"
        "[1] Heavy Gremlin Compact Energy Neutralizer\n"
        "  Unstable Heavy Energy Neutralizer Mutaplasmid\n"
        "  capacitorNeed 500.0, cpu 32.0, energyNeutralizerAmount 550.0, maxRange 16000.0, power 1800.0\n"
    )

    runtime, _fit = factory.build(parsed)
    effect = runtime.modules[0].effects[0]

    assert effect.range_m == pytest.approx(16_000.0)
    assert effect.projected_add["cap_drain"] == pytest.approx(550.0)
    assert runtime.diagnostics["pyfa_blueprint"]["modules"][0]["mutation"]["attributes"]["maxRange"] == 16000.0


def test_pyfa_deployable_manifest_extracts_drones_and_fighters() -> None:
    factory = RuntimeFromEftFactory()
    if not factory._pyfa.available or not factory._pyfa.fit_engine_ready:
        pytest.skip(factory.backend_status)

    parser = EftFitParser()
    drones, fighters, control = factory.build_deployable_manifest(
        parser.parse("[Vexor, Drones]\nHobgoblin II x5\nCurator II x2")
    )
    by_name = {entry.type_name: entry for entry in drones}

    assert control.max_active_drones >= 5
    assert by_name["Hobgoblin II"].quantity == 5
    assert by_name["Hobgoblin II"].bandwidth_mbit == pytest.approx(5.0)
    assert by_name["Curator II"].is_sentry

    _drones, fighters, fighter_control = factory.build_deployable_manifest(
        parser.parse("[Thanatos, Fighters]\nEinherji I x2")
    )
    assert fighter_control.fighter_tubes >= 1
    assert fighters[0].type_name == "Einherji I"
    assert any(ability.kind == "mwd" for ability in fighters[0].abilities)
    assert any(ability.kind == "heavy_attack" and ability.ammo_capacity > 0 for ability in fighters[0].abilities)


def test_launch_squad_drones_respects_bandwidth_and_active_limit() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=10, bandwidth=5.0)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=15.0, max_active_drones=5)
    world.ships[owner.ship_id] = owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    launched = system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")

    assert launched == 3
    assert len(world.drones) == 3


def test_sentry_drone_far_recall_does_not_keep_trying() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1, sentry=True)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=25.0, max_active_drones=1)
    world.ships[owner.ship_id] = owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    drone = next(iter(world.drones.values()))
    drone.nav.position = Vector2(20_000.0, 0.0)
    system.recall_squad_deployables(world, Team.BLUE, "A")

    system.run(world, 1.0)

    assert drone.ship_id in world.drones
    assert drone.state == "idle"
    assert drone.nav.command_target is None


def test_drone_attack_applies_damage_to_focus_target() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1, bandwidth=5.0)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target
    world.squad_focus_queues["BLUE:A"] = [target.ship_id]
    world.squad_focus_updated_at["BLUE:A"] = 0.0
    owner.combat.lock_targets.add(target.ship_id)

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    assert system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone") == 1
    before = target.vital.shield + target.vital.armor + target.vital.structure

    system.run(world, 1.0)

    after = target.vital.shield + target.vital.armor + target.vital.structure
    assert after < before


def test_drone_attack_command_requires_owner_lock() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    assert system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone") == 1
    drone = next(iter(world.drones.values()))

    assert system.set_squad_drone_target(world, Team.BLUE, "A", target.ship_id)
    assert drone.target_id is None
    assert owner.deployable_control.pending_drone_attack_target_id == target.ship_id

    owner.combat.lock_targets.add(target.ship_id)
    system.run(world, 1.0)

    assert drone.target_id == target.ship_id
    assert owner.deployable_control.pending_drone_attack_target_id is None


def test_drone_command_rejects_targets_outside_owner_signal_range() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.profile.max_target_range = 1_000.0
    target = _ship("RED-A-001", Team.RED, "A", Vector2(5_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    assert system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone") == 1

    assert not system.set_squad_drone_target(world, Team.BLUE, "A", target.ship_id)
    assert owner.deployable_control.pending_drone_attack_target_id is None
    assert next(iter(world.drones.values())).target_id is None


def test_drone_keeps_accepted_target_after_owner_signal_range_is_lost() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.profile.max_target_range = 1_000.0
    target = _ship("RED-A-001", Team.RED, "A", Vector2(500.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    assert system.set_squad_drone_target(world, Team.BLUE, "A", target.ship_id)
    drone = next(iter(world.drones.values()))
    assert drone.target_id == target.ship_id

    owner.nav.position = Vector2(-10_000.0, 0.0)
    owner.combat.lock_targets.clear()
    system.run(world, 1.0)

    assert drone.target_id == target.ship_id


def test_deployable_disconnect_clears_target_and_recovers_within_500km() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    system.set_squad_drone_target(world, Team.BLUE, "A", target.ship_id)
    drone = next(iter(world.drones.values()))
    drone.nav.position = Vector2(600_000.0, 0.0)

    system.run(world, 1.0)

    assert not drone.connected
    assert drone.state == "disconnected"
    assert drone.target_id is None
    assert drone.nav.velocity == Vector2(0.0, 0.0)

    drone.nav.position = Vector2(1_000.0, 0.0)
    system.run(world, 1.0)

    assert drone.connected
    assert drone.state == "guarding"


def test_drones_orbit_owner_and_fighters_wait_for_manual_navigation() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(
        drone_bandwidth_mbit=5.0,
        max_active_drones=1,
        fighter_tubes=1,
        fighter_light_slots=1,
    )
    world.ships[owner.ship_id] = owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")

    system.run(world, 1.0, advance_physics=False, apply_effects=False)

    drone = next(iter(world.drones.values()))
    fighter = next(iter(world.fighters.values()))
    assert drone.state == "guarding"
    assert fighter.state == "idle"
    assert drone.nav.command_mode == "orbit"
    assert drone.nav.command_target_ship_id == owner.ship_id
    assert fighter.nav.command_target_ship_id is None
    assert fighter.nav.command_target is None


def test_recall_recovers_deployables_at_edge_range_with_physics_substeps() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(
        drone_bandwidth_mbit=5.0,
        max_active_drones=1,
        fighter_tubes=1,
        fighter_light_slots=1,
    )
    world.ships[owner.ship_id] = owner
    combat = CombatSystem(PyfaBridge())
    engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=4), combat)

    engine.deployables.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    engine.deployables.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")
    drone = next(iter(world.drones.values()))
    fighter = next(iter(world.fighters.values()))
    drone.nav.position = Vector2(owner.nav.radius + drone.nav.radius + engine.deployables.RECOVERY_RANGE_M, 0.0)
    fighter.nav.position = Vector2(owner.nav.radius + fighter.nav.radius + engine.deployables.RECOVERY_RANGE_M, 0.0)

    engine.deployables.recall_squad_deployables(world, Team.BLUE, "A")
    engine.step()

    assert not world.drones
    assert not world.fighters


def test_drone_attacks_at_best_attack_orbit_range() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    system.set_squad_drone_target(world, Team.BLUE, "A", target.ship_id)

    system.run(world, 1.0, advance_physics=False, apply_effects=False)

    drone = next(iter(world.drones.values()))
    assert drone.nav.command_mode == "orbit"
    assert drone.nav.command_target_ship_id == target.ship_id
    assert drone.nav.command_range_m == pytest.approx(_drone_entry().optimal_range_m)


def test_fighter_target_does_not_create_auto_navigation() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(20_000.0, 0.0))
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")
    assert system.set_squad_fighter_target(world, Team.BLUE, "A", target.ship_id)

    system.run(world, 1.0, advance_physics=False, apply_effects=False)

    fighter = next(iter(world.fighters.values()))
    assert fighter.state == "engaging"
    assert fighter.nav.command_target is None
    assert fighter.nav.command_target_ship_id is None


def test_fighter_squad_manual_navigation_sets_each_fighter_without_leader() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    world.ships[owner.ship_id] = owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")
    fighter = next(iter(world.fighters.values()))

    changed = system.command_fighter_squad_move(world, Team.BLUE, fighter.squad_id, Vector2(5_000.0, 1_000.0))

    assert changed == 1
    assert world.squad_leaders == {}
    assert fighter.nav.command_mode == "move"
    assert fighter.nav.command_target == Vector2(5_000.0, 1_000.0)


def test_drone_can_attack_fighter_target_without_error() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    fighter_owner = _ship("RED-A-001", Team.RED, "B", Vector2(1_000.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(drone_bandwidth_mbit=5.0, max_active_drones=1)
    fighter_owner.fighter_bay = [_fighter_entry(quantity=1)]
    fighter_owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    world.ships[owner.ship_id] = owner
    world.ships[fighter_owner.ship_id] = fighter_owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    system.launch_squad_fighters(world, Team.RED, "B", "Test Fighter")
    fighter = next(iter(world.fighters.values()))
    fighter.nav.position = Vector2(1_500.0, 0.0)
    owner.combat.lock_targets.add(fighter.ship_id)

    assert system.set_squad_drone_target(world, Team.BLUE, "A", fighter.ship_id)
    before = fighter.vital.shield + fighter.vital.armor + fighter.vital.structure
    system.run(world, 1.0)

    after = fighter.vital.shield + fighter.vital.armor + fighter.vital.structure
    assert after < before


def test_owner_death_kills_deployables() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    owner.drone_bay = [_drone_entry(quantity=1)]
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(
        drone_bandwidth_mbit=5.0,
        max_active_drones=1,
        fighter_tubes=1,
        fighter_light_slots=1,
    )
    world.ships[owner.ship_id] = owner

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_drones(world, Team.BLUE, "A", "Test Drone")
    system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")
    owner.vital.alive = False

    system.run(world, 1.0)

    assert all(not drone.vital.alive for drone in world.drones.values())
    assert all(not fighter.vital.alive for fighter in world.fighters.values())


def test_fighter_launches_into_selectable_squad_and_manual_heavy_attack() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    assert system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter") == 1
    fighter = next(iter(world.fighters.values()))
    assert fighter.squad_id == DeployableSystem.fighter_squad_id("A")
    assert fighter.owner_squad_id == "A"

    assert system.set_squad_fighter_target(world, Team.BLUE, "A", target.ship_id)
    system.run(world, 1.0)
    assert fighter.ability_ammo_remaining["heavy"] == 1

    fighter.combat.lock_targets.add(target.ship_id)
    assert system.activate_fighter_ability(world, Team.BLUE, fighter.squad_id, "heavy") == 1
    system.run(world, 1.0)

    assert fighter.ability_ammo_remaining["heavy"] == 0


def test_fighter_attack_requires_fighter_self_lock_after_owner_command() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    target = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    target.profile.sig_radius = 1.0
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    owner.combat.lock_targets.add(target.ship_id)
    world.ships[owner.ship_id] = owner
    world.ships[target.ship_id] = target

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    assert system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter") == 1
    fighter = next(iter(world.fighters.values()))
    assert system.set_squad_fighter_target(world, Team.BLUE, "A", target.ship_id)
    before = target.vital.shield + target.vital.armor + target.vital.structure

    system.run(world, 1.0)

    after_without_lock = target.vital.shield + target.vital.armor + target.vital.structure
    assert after_without_lock == pytest.approx(before)
    assert target.ship_id not in fighter.combat.lock_targets
    assert target.ship_id in fighter.combat.lock_timers or target.ship_id in fighter.combat.lock_deadlines

    fighter.combat.lock_targets.add(target.ship_id)
    fighter.combat.lock_timers.pop(target.ship_id, None)
    fighter.combat.lock_deadlines.pop(target.ship_id, None)
    system.run(world, 1.0)

    after_with_lock = target.vital.shield + target.vital.armor + target.vital.structure
    assert after_with_lock < after_without_lock


def test_fighter_target_uses_latest_mother_or_fighter_squad_command() -> None:
    world = WorldState()
    owner = _ship("BLUE-A-001", Team.BLUE, "A", Vector2(0.0, 0.0))
    first = _ship("RED-A-001", Team.RED, "A", Vector2(2_000.0, 0.0))
    second = _ship("RED-A-002", Team.RED, "A", Vector2(2_500.0, 0.0))
    owner.fighter_bay = [_fighter_entry(quantity=1)]
    owner.deployable_control = DeployableControlState(fighter_tubes=1, fighter_light_slots=1)
    owner.combat.lock_targets.update({first.ship_id, second.ship_id})
    world.ships[owner.ship_id] = owner
    world.ships[first.ship_id] = first
    world.ships[second.ship_id] = second

    system = DeployableSystem(CombatSystem(PyfaBridge()), MovementSystem())
    system.launch_squad_fighters(world, Team.BLUE, "A", "Test Fighter")
    fighter = next(iter(world.fighters.values()))

    world.now = 1.0
    system.set_squad_fighter_target(world, Team.BLUE, "A", first.ship_id)
    system.run(world, 1.0)
    assert fighter.target_id == first.ship_id

    world.now = 5.0
    fighter_focus_key = f"BLUE:{fighter.squad_id}"
    world.squad_focus_queues[fighter_focus_key] = [second.ship_id]
    world.squad_focus_updated_at[fighter_focus_key] = world.now
    system.run(world, 1.0)
    assert fighter.target_id == second.ship_id

    world.now = 10.0
    system.set_squad_fighter_target(world, Team.BLUE, "A", first.ship_id)
    system.run(world, 1.0)
    assert fighter.target_id == first.ship_id
