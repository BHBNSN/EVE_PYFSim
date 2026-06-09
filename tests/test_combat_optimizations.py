from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import unittest
import weakref
from typing import Callable, TypedDict
from unittest.mock import patch

from eve_sim.agents import ShipAgent
from eve_sim import fleet_setup as fleet_setup_module
from eve_sim.config import EngineConfig
from eve_sim.fleet_setup import EftFitParser, RuntimeFromEftFactory, get_runtime_resolve_cache_key, resolve_runtime_from_pyfa_runtime
from eve_sim.fit_runtime import EffectClass, FitRuntime, HullProfile, ModuleEffect, ModuleRuntime, ModuleState, ProjectedImpact, SkillProfile
from eve_sim.gui.dialogs import ShipStatusDialog
from eve_sim.math2d import Vector2
from eve_sim.models import BubbleField, CombatState, FitDescriptor, NavigationState, Order, ProjectileEntity, QualityLevel, QualityState, ShipEntity, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem, MovementSystem, PerceptionSystem
from eve_sim.systems.models import CycleTargetSnapshot
from eve_sim.world import WorldState


class ResolveCall(TypedDict):
    fit_key: str
    module_states: tuple[tuple[str, str], ...]
    command_fit_keys: tuple[str, ...]
    projected_fit_keys: tuple[str, ...]


class CountedEffects(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _projection_distance_signature(snapshot: dict[str, object]) -> float | None:
    if str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range") != "exact_range":
        return None
    distance = max(0.0, float(snapshot.get("pyfa_projection_range", snapshot.get("projection_range", 0.0)) or 0.0))
    return math.floor(distance / 100.0) * 100.0


_ROKH_PROPULSION_FIT = '''[Rokh, 2506 WC Rokh]
Damage Control II
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II

Large Shield Extender II
Large Shield Extender II
Large Shield Extender II
Medium Capacitor Booster II
Multispectrum Shield Hardener II
Multispectrum Shield Hardener II
500MN Quad LiF Restrained Microwarpdrive

425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L
425mm Railgun II, Caldari Navy Antimatter Charge L

Large Core Defense Field Extender I
Large Core Defense Field Extender I
Large Core Defense Field Extender I
'''

_FEROX_RAIL_FIT = '''[Ferox, Rail DPS]
Magnetic Field Stabilizer II
Magnetic Field Stabilizer II
Tracking Enhancer II

10MN Afterburner II
Large Shield Extender II
Large Shield Extender II
Multispectrum Shield Hardener II
Multispectrum Shield Hardener II
Multispectrum Shield Hardener II

250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M
250mm Railgun II, Antimatter Charge M

Medium Core Defense Field Extender I
Medium Core Defense Field Extender I
Medium Core Defense Field Extender I
'''

_BLACKBIRD_DAMP_FIT = '''[Blackbird, Damp]
Remote Sensor Dampener II
Remote Sensor Dampener II
'''

_BELLICOSE_PAINTER_FIT = '''[Bellicose, Painter]
Target Painter II
Target Painter II
'''

_SCYTHE_REMOTE_TRACKING_FIT = '''[Scythe, Remote Tracking]
Remote Tracking Computer II
'''

_CARACAL_NAVY_HAM_FIT = '''[Caracal Navy Issue, 自爆吉普车]
Reactor Control Unit II
Ballistic Control System II
Ballistic Control System II

Large Shield Extender II
Large Shield Extender II
Large Shield Extender II
Large Shield Extender II
Large Shield Extender II
50MN Quad LiF Restrained Microwarpdrive

Polarized Heavy Assault Missile Launcher
Polarized Heavy Assault Missile Launcher
Polarized Heavy Assault Missile Launcher
Polarized Heavy Assault Missile Launcher
Polarized Heavy Assault Missile Launcher
Polarized Heavy Assault Missile Launcher

Medium Ancillary Current Router I
Medium Core Defense Field Extender I
Medium Core Defense Field Extender I

Acolyte II x5

Nova Rage Heavy Assault Missile x300
Mjolnir Rage Heavy Assault Missile x300
Nanite Repair Paste x300
Inferno Rage Heavy Assault Missile x300
Scourge Rage Heavy Assault Missile x300
'''

_NESTOR_SMARTBOMB_FIT = '''[Nestor, *炸弹大白]
Damage Control II
Syndicate 1600mm Steel Plates
Syndicate 1600mm Steel Plates
Corpii A-Type Multispectrum Coating
Corpus X-Type Explosive Armor Hardener
Centii A-Type Multispectrum Coating

Heavy Capacitor Booster II
Republic Fleet Large Cap Battery
Sentient Burst Jammer
Gist X-Type 500MN Microwarpdrive
Large Micro Jump Drive
Signature Radius Suppressor I

Domination Large Proton Smartbomb
Caldari Navy Large Graviton Smartbomb
Domination Large Proton Smartbomb
Caldari Navy Large Graviton Smartbomb
Domination Large Proton Smartbomb
Caldari Navy Large Graviton Smartbomb
Caldari Navy Large Graviton Smartbomb

Large Trimark Armor Pump II
Large Trimark Armor Pump II
Large Trimark Armor Pump II

Hornet EC-300 x15
Warrior I x10

Navy Cap Booster 3200 x7
Navy Cap Booster 800 x6
Warp Scrambler II x1
Improved Cloaking Device II x1
Reactive Armor Hardener x1
Stasis Webifier II x1
'''


def _make_runtime(fit: FitDescriptor, modules: list[ModuleRuntime]) -> FitRuntime:
    hull = HullProfile(
        ship_name=fit.ship_name,
        role=fit.role,
        base_dps=fit.base_dps,
        volley=fit.volley,
        optimal=fit.optimal_range,
        falloff=fit.falloff,
        tracking=fit.tracking,
        sig_radius=fit.signature_radius,
        scan_resolution=fit.scan_resolution,
        max_target_range=fit.max_target_range,
        max_speed=fit.max_speed,
        cap_max=fit.max_cap,
        cap_recharge_time=fit.cap_recharge_time,
        shield_hp=fit.shield_hp,
        armor_hp=fit.armor_hp,
        structure_hp=fit.structure_hp,
        rep_amount=fit.rep_amount,
        rep_cycle=fit.rep_cycle,
    )
    runtime = FitRuntime(fit_key=fit.fit_key, hull=hull, skills=SkillProfile(), modules=modules)
    runtime.diagnostics["pyfa_blueprint"] = {
        "fit_key": fit.fit_key,
        "ship_name": fit.ship_name,
        "modules": [
            {
                "module_id": module.module_id,
                "module_name": module.group,
                "charge_name": "",
                "offline": module.state == ModuleState.OFFLINE,
            }
            for module in modules
        ],
    }
    runtime.diagnostics["motion_params"] = {
        "mass": float(getattr(fit, "mass", 0.0) or 0.0),
        "agility": float(getattr(fit, "agility", 0.0) or 0.0),
    }
    return runtime


def _make_test_fit_descriptor(fit_key: str, role: str = "test") -> FitDescriptor:
    return FitDescriptor(
        fit_key=fit_key,
        ship_name="Test Hull",
        role=role,
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_speed=0.0,
        max_cap=100.0,
        cap_recharge_time=1e12,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
    )


def _make_ship(
    ship_id: str,
    modules: list[ModuleRuntime],
    *,
    fit_key: str | None = None,
    team: Team = Team.BLUE,
    squad_id: str = "SQ1",
) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=fit_key or ship_id,
        ship_name="Test Hull",
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_speed=0.0,
        max_cap=100.0,
        cap_recharge_time=1e12,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
    )

    bridge = PyfaBridge()
    profile = bridge.build_profile(fit)
    runtime = _make_runtime(fit, modules)
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=fit,
        profile=replace(profile),
        nav=NavigationState(
            position=Vector2(0.0, 0.0),
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            max_speed=0.0,
        ),
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
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
        runtime=runtime,
    )


def _make_pyfa_ship_from_fit_text(
    fit_text: str,
    *,
    ship_id: str = "pyfa-ship",
    team: Team = Team.BLUE,
    squad_id: str = "SQ1",
) -> ShipEntity:
    parser = EftFitParser()
    factory = RuntimeFromEftFactory()
    parsed = parser.parse(fit_text)
    runtime_template, fit = factory.build(parsed)
    profile = factory.build_profile(parsed)
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=fit,
        profile=replace(profile),
        nav=NavigationState(
            position=Vector2(0.0, 0.0),
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            max_speed=profile.max_speed,
        ),
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
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
        runtime=deepcopy(runtime_template),
    )


def _bubble_field(
    field_id: str,
    *,
    position: Vector2,
    radius_m: float,
    kind: str = "warp_disrupt_probe",
    interdiction_kind: str = "probe",
    blocks_warp: bool = True,
    speed_factor_mult: float = 1.0,
    anchor_ship_id: str | None = None,
) -> BubbleField:
    return BubbleField(
        field_id=field_id,
        kind=kind,
        interdiction_kind=interdiction_kind,
        source_ship_id="source",
        source_module_id="module",
        team=Team.RED,
        position=Vector2(position.x, position.y),
        radius_m=radius_m,
        expires_at=1_000.0,
        blocks_warp=blocks_warp,
        speed_factor_mult=speed_factor_mult,
        anchor_ship_id=anchor_ship_id,
        destructible=False,
        alive=True,
    )


def _command_burst_module(module_id: str, cycle_time: float, cap_need: float, *, charge_capacity: int = 0, charge_rate: float = 0.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="command burst",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-burst",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=100_000.0,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"shield_rep": 0.0},
            )
        ],
        charge_capacity=charge_capacity,
        charge_rate=charge_rate,
        charge_remaining=float(charge_capacity),
        tags=("affects_local_pyfa_profile", "area_effect", "command_burst", "controlled", "projected", "support"),
    )


def _remote_sensor_damp_module(
    module_id: str,
    cycle_time: float,
    cap_need: float,
    *,
    range_m: float = 100_000.0,
    falloff_m: float = 0.0,
) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="sensor dampener",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-damp",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=range_m,
                falloff_m=falloff_m,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_mult={"scan": 0.8, "range": 0.8},
            )
        ],
        tags=("controlled", "hostile", "offensive_ewar", "projected"),
    )


def _weapon_module(module_id: str, cycle_time: float, cap_need: float, *, damage: float = 50.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="weapon",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-weapon",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"damage_em": damage},
            )
        ],
        tags=("controlled", "hostile", "projected", "weapon"),
    )


def _remote_shield_rep_module(module_id: str, cycle_time: float, cap_need: float, *, amount: float = 30.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="remote shield booster",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-shield-rep",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"shield_rep": amount},
            )
        ],
        tags=("controlled", "projected", "remote_repair", "support"),
    )


def _remote_armor_rep_module(module_id: str, cycle_time: float, cap_need: float, *, amount: float = 30.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="remote armor repairer",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-armor-rep",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"armor_rep": amount},
            )
        ],
        tags=("controlled", "projected", "remote_repair", "support"),
    )


def _ally_support_module(module_id: str, cycle_time: float, cap_need: float) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="remote tracking computer",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-support",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                range_m=100_000.0,
                projected_mult={"tracking": 1.2},
            )
        ],
        tags=("controlled", "projected", "support"),
    )


def _energy_neutralizer_module(module_id: str, cycle_time: float, cap_need: float, *, amount: float = 20.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="energy neutralizer",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-neutralizer",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"cap_drain": amount},
            )
        ],
        tags=("cap_warfare", "controlled", "hostile", "offensive_ewar", "projected"),
    )


def _nosferatu_module(module_id: str, cycle_time: float, cap_need: float, *, amount: float = 20.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="energy nosferatu",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-nosferatu",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"cap_drain": amount},
            )
        ],
        tags=("cap_warfare", "controlled", "hostile", "offensive_ewar", "projected"),
    )


def _smart_bomb_module(
    module_id: str,
    cycle_time: float,
    cap_need: float,
    *,
    damage: float = 40.0,
    range_m: float = 10_000.0,
) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="smart bomb",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-smart-bomb",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                range_m=range_m,
                projected_add={"damage_em": damage},
            )
        ],
        tags=("area_effect", "controlled", "hostile", "projected", "smart_bomb", "weapon"),
    )


def _missile_weapon_module(
    module_id: str,
    cycle_time: float,
    cap_need: float,
    *,
    damage: float = 80.0,
    range_m: float = 8_000.0,
    projectile_speed: float = 1_000.0,
    flight_time: float = 8.0,
    explosion_radius: float = 120.0,
    explosion_velocity: float = 90.0,
    drf: float = 0.5,
    projectile_shield_hp: float = 0.0,
    projectile_armor_hp: float = 0.0,
    projectile_structure_hp: float = 20.0,
    projectile_shield_resonance_em: float = 1.0,
) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="missile launcher",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-missile",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=range_m,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={
                    "damage_em": damage,
                    "weapon_is_missile": 1.0,
                    "weapon_explosion_radius": explosion_radius,
                    "weapon_explosion_velocity": explosion_velocity,
                    "weapon_drf": drf,
                    "weapon_projectile_speed": projectile_speed,
                    "weapon_projectile_flight_time": flight_time,
                    "weapon_projectile_shield_hp": projectile_shield_hp,
                    "weapon_projectile_armor_hp": projectile_armor_hp,
                    "weapon_projectile_structure_hp": projectile_structure_hp,
                    "weapon_projectile_shield_resonance_em": projectile_shield_resonance_em,
                },
            )
        ],
        tags=("controlled", "hostile", "projected", "weapon"),
    )


def _bomb_launcher_module(
    module_id: str,
    cycle_time: float,
    cap_need: float,
    *,
    damage: float = 400.0,
    range_m: float = 6_000.0,
    projectile_speed: float = 2_000.0,
    flight_time: float = 3.0,
    explosion_radius: float = 400.0,
    blast_radius: float = 900.0,
    projectile_structure_hp: float = 20.0,
) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="bomb launcher",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-bomb",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=range_m,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={
                    "damage_explosive": damage,
                    "weapon_is_bomb": 1.0,
                    "weapon_blast_radius": blast_radius,
                    "weapon_explosion_radius": explosion_radius,
                    "weapon_projectile_speed": projectile_speed,
                    "weapon_projectile_flight_time": flight_time,
                    "weapon_projectile_structure_hp": projectile_structure_hp,
                },
            )
        ],
        tags=("controlled", "hostile", "projected", "weapon"),
    )


def _propulsion_module(module_id: str, cycle_time: float, cap_need: float, *, speed_mult: float = 1.5) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="propulsion module",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-prop",
                effect_class=EffectClass.LOCAL,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                local_mult={"speed": speed_mult},
            )
        ],
        tags=("affects_local_pyfa_profile", "controlled", "propulsion"),
    )


def _damage_control_module(module_id: str, cycle_time: float, cap_need: float) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="damage control",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-damage-control",
                effect_class=EffectClass.LOCAL,
                state_required=ModuleState.ACTIVE,
                cycle_time=cycle_time,
                cap_need=cap_need,
                local_mult={"speed": 1.0},
            )
        ],
        tags=("affects_local_pyfa_profile", "controlled", "damage_control"),
    )


def _set_motion_state(ship: ShipEntity, *, max_speed: float, mass: float, agility: float, sync_runtime: bool = True) -> None:
    ship.profile = replace(ship.profile, max_speed=max_speed, mass=mass, agility=agility)
    ship.nav.max_speed = max_speed
    if sync_runtime and ship.runtime is not None:
        ship.runtime.diagnostics["motion_params"] = {
            "mass": mass,
            "agility": agility,
        }


def _ecm_module(
    module_id: str,
    cycle_time: float,
    cap_need: float,
    *,
    range_m: float = 100_000.0,
    falloff_m: float = 0.0,
    jam_strength: float = 20.0,
) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="ecm",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-ecm",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=range_m,
                falloff_m=falloff_m,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"ecm_gravimetric": jam_strength},
            )
        ],
        tags=("controlled", "ecm", "hostile", "offensive_ewar", "projected"),
    )


def _warp_scrambler_module(module_id: str, cycle_time: float, cap_need: float, *, amount: float = 2.0) -> ModuleRuntime:
    return ModuleRuntime(
        module_id=module_id,
        group="warp scrambler",
        state=ModuleState.ONLINE,
        effects=[
            ModuleEffect(
                name=f"{module_id}-scram",
                effect_class=EffectClass.PROJECTED,
                state_required=ModuleState.ACTIVE,
                range_m=24_000.0,
                cycle_time=cycle_time,
                cap_need=cap_need,
                projected_add={"warp_scramble_status": amount},
            )
        ],
        tags=("controlled", "hostile", "projected", "target_ewar"),
    )


class CombatOptimizationTests(unittest.TestCase):
    class _DummyPyfaFit:
        def __init__(self) -> None:
            self.ID = 1
            self.ship = None
            self.calculated = False
            self.modules: list[object] = []
            self.projectedModules: list[object] = []
            self.drones: list[object] = []
            self.projectedDrones: list[object] = []
            self.fighters: list[object] = []
            self.projectedFighters: list[object] = []
            self.implants: list[object] = []
            self.boosters: list[object] = []
            self.victimOf: dict[int, object] = {}
            self.boostedOf: dict[int, object] = {}
            self.projectedOnto: dict[int, object] = {}
            self.applied_projected: list[str] = []
            self.applied_commands: list[str] = []

        def calculateModifiedAttributes(self, *args, **kwargs) -> None:
            del args, kwargs
            self.calculated = True

    def _make_engine(self, world: WorldState, *, tick_rate: int = 1, physics_substeps: int = 1) -> SimulationEngine:
        combat = CombatSystem(PyfaBridge())
        engine = SimulationEngine(world, EngineConfig(tick_rate=tick_rate, physics_substeps=physics_substeps), combat)
        for ship_id in world.ships:
            engine.register_ship(ship_id)
        return engine

    def _run_world_steps_and_capture_resolves(
        self,
        world: WorldState,
        *,
        step_dts: list[float],
        before_step_callbacks: list[Callable[[WorldState], None] | None] | None = None,
    ) -> list[list[ResolveCall]]:
        combat = CombatSystem(PyfaBridge())
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), combat)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        base_profiles = {
            ship.runtime.fit_key: replace(ship.profile)
            for ship in world.ships.values()
            if ship.runtime is not None
        }
        recorded_steps: list[list[ResolveCall]] = []
        current_step_calls: list[ResolveCall] = []

        def fake_cache_key(runtime, command_boosters, projected_sources):
            booster_sig = tuple(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple(sorted((str(module_id), str(state)) for module_id, state in (snapshot.get("state_by_module_id") or {}).items())),
                )
                for snapshot in (command_boosters or [])
            )
            projected_sig = tuple(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple(sorted((str(module_id), str(state)) for module_id, state in (snapshot.get("state_by_module_id") or {}).items())),
                    str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range"),
                    _projection_distance_signature(snapshot),
                )
                for snapshot in (projected_sources or [])
            )
            state_sig = tuple((module.module_id, str(module.state.value)) for module in runtime.modules)
            return runtime.fit_key, state_sig, booster_sig, projected_sig

        def fake_resolve(runtime, command_boosters, projected_sources):
            current_step_calls.append(
                {
                    "fit_key": runtime.fit_key,
                    "module_states": tuple((module.module_id, str(module.state.value)) for module in runtime.modules),
                    "command_fit_keys": tuple(str(snapshot.get("fit_key", "") or "") for snapshot in (command_boosters or [])),
                    "projected_fit_keys": tuple(str(snapshot.get("fit_key", "") or "") for snapshot in (projected_sources or [])),
                }
            )
            runtime.diagnostics["pyfa_command_boosters"] = deepcopy(command_boosters or [])
            runtime.diagnostics["pyfa_projected_sources"] = deepcopy(projected_sources or [])
            runtime.diagnostics["pyfa_base_profile"] = replace(base_profiles[runtime.fit_key])
            return runtime, replace(base_profiles[runtime.fit_key])

        with patch("eve_sim.systems.combat_core.get_runtime_resolve_cache_key", side_effect=fake_cache_key), patch(
            "eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime",
            side_effect=fake_resolve,
        ):
            for step_index, step_dt in enumerate(step_dts):
                if before_step_callbacks is not None and step_index < len(before_step_callbacks):
                    callback = before_step_callbacks[step_index]
                    if callback is not None:
                        callback(world)
                current_step_calls = []
                engine._dt = float(step_dt)
                engine.step()
                recorded_steps.append(list(current_step_calls))

        return recorded_steps

    def _run_world_step_and_count_resolves(self, world: WorldState, *, step_dt: float) -> int:
        return len(self._run_world_steps_and_capture_resolves(world, step_dts=[step_dt])[0])

    def _run_step_and_count_resolves(self, ship: ShipEntity, *, step_dt: float) -> int:
        return self._run_world_step_and_count_resolves(WorldState(ships={ship.ship_id: ship}), step_dt=step_dt)

    def test_local_signature_ignores_projected_weapon_state_but_tracks_propulsion(self) -> None:
        runtime = _make_runtime(
            _make_test_fit_descriptor("local-sig-fit"),
            [
                _weapon_module("weapon-a", cycle_time=5.0, cap_need=1.0),
                _propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0),
            ],
        )

        base_signature = CombatSystem._local_runtime_state_signature(runtime)
        runtime.modules[0].state = ModuleState.ACTIVE
        self.assertEqual(CombatSystem._local_runtime_state_signature(runtime), base_signature)

        runtime.modules[1].state = ModuleState.ACTIVE
        self.assertNotEqual(CombatSystem._local_runtime_state_signature(runtime), base_signature)

    def test_control_signal_wakes_propulsion_module_when_pending_is_empty(self) -> None:
        ship = _make_ship(
            "BLUE-1",
            [_propulsion_module("prop-a", cycle_time=10.0, cap_need=0.0)],
            team=Team.BLUE,
            squad_id="SQ1",
        )
        ship.combat.module_decision_pending_signature = ("prop-a",)
        ship.combat.module_decision_pending.clear()
        ship.combat.module_decision_propulsion_active = False
        ship.nav.propulsion_command_active = True
        world = WorldState(ships={ship.ship_id: ship})

        CombatSystem(PyfaBridge())._update_module_states(world, 1.0, now=0.0)

        self.assertEqual(ship.runtime.modules[0].state, ModuleState.ACTIVE)

    def test_focus_signal_wakes_weapon_target_selection_when_pending_is_empty(self) -> None:
        source = _make_ship(
            "BLUE-1",
            [_weapon_module("weapon-a", cycle_time=10.0, cap_need=0.0)],
            team=Team.BLUE,
            squad_id="SQ1",
        )
        target = _make_ship("RED-1", [], team=Team.RED, squad_id="SQ2")
        target.nav.position = Vector2(1_000.0, 0.0)
        source.combat.module_decision_pending_signature = ("weapon-a",)
        source.combat.module_decision_pending.clear()
        world = WorldState(
            ships={source.ship_id: source, target.ship_id: target},
            squad_focus_queues={"BLUE:SQ1": [target.ship_id]},
        )

        CombatSystem(PyfaBridge())._update_module_states(world, 1.0, now=0.0)

        self.assertEqual(source.combat.projected_targets.get("weapon-a"), target.ship_id)

    def test_resolve_cache_key_ignores_remote_only_module_state(self) -> None:
        runtime = _make_runtime(
            _make_test_fit_descriptor("ewar-fit", role="ewar"),
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
        )

        online_key = get_runtime_resolve_cache_key(runtime, [], [])
        runtime.modules[0].state = ModuleState.ACTIVE
        active_key = get_runtime_resolve_cache_key(runtime, [], [])

        self.assertIsNotNone(online_key)
        self.assertEqual(active_key, online_key)

    def test_resolve_cache_key_still_tracks_command_burst_state(self) -> None:
        runtime = _make_runtime(
            _make_test_fit_descriptor("burst-fit", role="burst"),
            [_command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0)],
        )

        online_key = get_runtime_resolve_cache_key(runtime, [], [])
        runtime.modules[0].state = ModuleState.ACTIVE
        active_key = get_runtime_resolve_cache_key(runtime, [], [])

        self.assertIsNotNone(online_key)
        self.assertNotEqual(active_key, online_key)

    def test_multi_module_activation_resolves_once_per_tick(self) -> None:
        ship = _make_ship(
            "ship-alpha",
            [
                _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0),
                _command_burst_module("burst-b", cycle_time=10.0, cap_need=5.0),
            ],
        )

        resolve_count = self._run_step_and_count_resolves(ship, step_dt=1.0)

        runtime = ship.runtime
        assert runtime is not None
        self.assertEqual(resolve_count, 1)
        self.assertEqual(runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(runtime.modules[1].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(ship.vital.cap, 90.0)

    def test_large_step_defers_cycle_completion_to_next_substep_without_extra_resolve(self) -> None:
        ship = _make_ship(
            "ship-beta",
            [
                _command_burst_module(
                    "burst-main",
                    cycle_time=5.0,
                    cap_need=10.0,
                    charge_capacity=10,
                    charge_rate=1.0,
                )
            ],
        )

        resolve_count = self._run_step_and_count_resolves(ship, step_dt=15.0)

        runtime = ship.runtime
        assert runtime is not None
        module = runtime.modules[0]
        self.assertEqual(resolve_count, 1)
        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertAlmostEqual(ship.vital.cap, 90.0)
        self.assertAlmostEqual(module.charge_remaining, 10.0)
        self.assertAlmostEqual(ship.combat.module_cycle_timers[module.module_id], 0.0)
        self.assertIn(module.module_id, ship.combat.module_decision_pending)

    def test_identical_fleet_activation_batches_pyfa_resolve(self) -> None:
        world = WorldState()
        for idx in range(8):
            ship = _make_ship(
                f"ship-gamma-{idx}",
                [
                    _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0),
                    _command_burst_module("burst-b", cycle_time=10.0, cap_need=5.0),
                ],
                fit_key="shared-burst-fit",
                team=Team.BLUE,
                squad_id="SQ-BULK",
            )
            world.ships[ship.ship_id] = ship

        resolve_count = self._run_world_step_and_count_resolves(world, step_dt=1.0)

        self.assertEqual(resolve_count, 1)
        for ship in world.ships.values():
            runtime = ship.runtime
            assert runtime is not None
            self.assertEqual(runtime.modules[0].state, ModuleState.ACTIVE)
            self.assertEqual(runtime.modules[1].state, ModuleState.ACTIVE)
            self.assertAlmostEqual(ship.vital.cap, 90.0)

    def test_batched_runtime_refresh_keeps_ship_local_module_ids(self) -> None:
        world = WorldState()
        expected_module_ids: dict[str, list[str]] = {}
        for idx in range(2):
            module_ids = [f"burst-{idx}-a", f"burst-{idx}-b"]
            ship = _make_ship(
                f"ship-batch-{idx}",
                [
                    _command_burst_module(module_ids[0], cycle_time=5.0, cap_need=5.0),
                    _command_burst_module(module_ids[1], cycle_time=5.0, cap_need=5.0),
                ],
                fit_key="shared-burst-fit",
                team=Team.BLUE,
                squad_id="SQ-BATCH",
            )
            expected_module_ids[ship.ship_id] = module_ids
            world.ships[ship.ship_id] = ship

        combat = CombatSystem(PyfaBridge())
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), combat)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        shared_profile = replace(next(iter(world.ships.values())).profile)

        def fake_cache_key(runtime, command_boosters, projected_sources):
            del command_boosters, projected_sources
            return runtime.fit_key

        def fake_resolve(runtime, command_boosters, projected_sources):
            runtime.diagnostics["pyfa_command_boosters"] = deepcopy(command_boosters or [])
            runtime.diagnostics["pyfa_projected_sources"] = deepcopy(projected_sources or [])
            runtime.diagnostics["pyfa_base_profile"] = replace(shared_profile)
            runtime.diagnostics["pyfa_runtime_resolve_cache"] = "miss"
            return runtime, replace(shared_profile)

        with patch("eve_sim.systems.combat_core.get_runtime_resolve_cache_key", side_effect=fake_cache_key), patch(
            "eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime",
            side_effect=fake_resolve,
        ):
            engine._dt = 1.0
            engine.step()

            for ship_id, expected_ids in expected_module_ids.items():
                runtime = world.ships[ship_id].runtime
                assert runtime is not None
                self.assertEqual([module.module_id for module in runtime.modules], expected_ids)

            engine.step()

        for ship_id, expected_ids in expected_module_ids.items():
            ship = world.ships[ship_id]
            runtime = ship.runtime
            assert runtime is not None
            self.assertEqual([module.module_id for module in runtime.modules], expected_ids)
            self.assertEqual([module.state for module in runtime.modules], [ModuleState.ACTIVE, ModuleState.ACTIVE])
            self.assertAlmostEqual(ship.combat.module_cycle_timers[expected_ids[0]], 4.0)
            self.assertAlmostEqual(ship.combat.module_cycle_timers[expected_ids[1]], 4.0)

    def test_remote_projected_activation_reuses_source_profile_across_cycles(self) -> None:
        source = _make_ship(
            "ship-remote-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="remote-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-remote-target",
            [],
            fit_key="remote-target-fit",
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        recorded_steps = self._run_world_steps_and_capture_resolves(world, step_dts=[1.0, 10.0, 10.0])

        first_step = recorded_steps[0]
        self.assertEqual(len(first_step), 2)
        self.assertEqual(sum(1 for call in first_step if call["fit_key"] == "remote-source-fit"), 1)
        target_calls = [call for call in first_step if call["fit_key"] == "remote-target-fit"]
        self.assertEqual(len(target_calls), 1)
        self.assertEqual(target_calls[0]["projected_fit_keys"], ())
        self.assertEqual(recorded_steps[1], [])
        self.assertEqual(recorded_steps[2], [])
        runtime = source.runtime
        assert runtime is not None
        self.assertEqual(runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.projected_targets.get("damp-a"), target.ship_id)
        self.assertAlmostEqual(source.vital.cap, 90.0, places=4)

    def test_module_target_mode_override_can_select_nearest_enemy_for_projected_module(self) -> None:
        source = _make_ship(
            "ship-target-mode-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="target-mode-source-fit",
            team=Team.BLUE,
        )
        target_near = _make_ship(
            "ship-target-near",
            [],
            fit_key="target-mode-near-fit",
            team=Team.RED,
        )
        target_far = _make_ship(
            "ship-target-far",
            [],
            fit_key="target-mode-far-fit",
            team=Team.RED,
        )
        target_near.nav.position = Vector2(1_000.0, 0.0)
        target_far.nav.position = Vector2(8_000.0, 0.0)
        source.combat.lock_targets.update({target_near.ship_id, target_far.ship_id})
        source.combat.module_target_modes["damp-a"] = "enemy_nearest"
        world = WorldState(ships={source.ship_id: source, target_near.ship_id: target_near, target_far.ship_id: target_far})
        engine = self._make_engine(world, physics_substeps=1)

        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.projected_targets.get("damp-a"), target_near.ship_id)

    def test_remote_sensor_damp_scripts_apply_distinct_live_profile_effects(self) -> None:
        target_fit = """[Ferox, Damp Target]
Damage Control II
"""

        def _run_case(source_fit: str, ship_id_suffix: str) -> tuple[float, float]:
            source = _make_pyfa_ship_from_fit_text(
                source_fit,
                ship_id=f"ship-damp-source-{ship_id_suffix}",
                team=Team.BLUE,
            )
            target = _make_pyfa_ship_from_fit_text(
                target_fit,
                ship_id=f"ship-damp-target-{ship_id_suffix}",
                team=Team.RED,
            )
            source.nav.position = Vector2(0.0, 0.0)
            target.nav.position = Vector2(10_000.0, 0.0)
            source.combat.lock_targets.add(target.ship_id)
            source.combat.module_manual_modes["mod-1"] = "active"

            world = WorldState(ships={source.ship_id: source, target.ship_id: target})
            engine = self._make_engine(world, physics_substeps=1)
            base_scan = float(target.profile.scan_resolution)
            base_range = float(target.profile.max_target_range)

            engine.step()

            assert source.runtime is not None
            self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
            return (
                float(target.profile.scan_resolution) / max(1e-6, base_scan),
                float(target.profile.max_target_range) / max(1e-6, base_range),
            )

        scan_ratio, scan_range_ratio = _run_case(
            """[Celestis, Scan Script]
Remote Sensor Dampener II, Scan Resolution Dampening Script
""",
            "scan",
        )
        range_scan_ratio, range_ratio = _run_case(
            """[Celestis, Range Script]
Remote Sensor Dampener II, Targeting Range Dampening Script
""",
            "range",
        )
        unscripted_scan_ratio, unscripted_range_ratio = _run_case(
            """[Celestis, Unscripted]
Remote Sensor Dampener II
""",
            "none",
        )

        self.assertLess(scan_ratio, 1.0)
        self.assertAlmostEqual(scan_range_ratio, 1.0, places=6)
        self.assertAlmostEqual(range_scan_ratio, 1.0, places=6)
        self.assertLess(range_ratio, 1.0)
        self.assertLess(unscripted_scan_ratio, 1.0)
        self.assertLess(unscripted_range_ratio, 1.0)

    def test_resolve_none_without_cached_base_profile_does_not_compound_formula_projection(self) -> None:
        source = _make_ship(
            "ship-resolve-none-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="resolve-none-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-resolve-none-target",
            [],
            fit_key="resolve-none-target-fit",
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        assert source.runtime is not None
        assert target.runtime is not None
        source.runtime.diagnostics.pop("pyfa_base_profile", None)
        target.runtime.diagnostics.pop("pyfa_base_profile", None)

        with patch("eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime", return_value=None):
            engine.step()
            first_scan = target.profile.scan_resolution
            engine.step()
            second_scan = target.profile.scan_resolution
            engine.step()
            third_scan = target.profile.scan_resolution

        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(first_scan, 240.0)
        self.assertAlmostEqual(second_scan, first_scan)
        self.assertAlmostEqual(third_scan, first_scan)

    def test_formula_projected_modules_skip_pyfa_snapshot_collection(self) -> None:
        source = _make_ship(
            "ship-projected-bucket-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0, range_m=30_000.0, falloff_m=10_000.0)],
            fit_key="projected-bucket-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-projected-bucket-target",
            [],
            fit_key="projected-bucket-target-fit",
            team=Team.RED,
        )
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        combat = CombatSystem(PyfaBridge())

        runtime = source.runtime
        assert runtime is not None
        module = runtime.modules[0]
        module.state = ModuleState.ACTIVE
        source.combat.projected_targets[module.module_id] = target.ship_id
        snapshot_key = combat._module_cycle_snapshot_key(source.ship_id, module.module_id)
        combat._module_cycle_target_snapshots[snapshot_key] = {
            target.ship_id: CycleTargetSnapshot(distance=30_150.0, active_effect_indices={0})
        }

        snapshots = combat._collect_projected_source_snapshots(world, {})
        metadata = combat._module_static_metadata(module)
        buckets = combat._runtime_module_buckets(runtime)

        self.assertEqual(snapshots, {})
        self.assertFalse(metadata.uses_pyfa_projected_profile)
        self.assertEqual([entry[0].module_id for entry in buckets.pyfa_projected_entries], [])

    def test_effective_profile_reuses_cached_runtime_projection_result(self) -> None:
        ship = _make_ship("effective-cache-target", [], fit_key="effective-cache-fit")
        combat = CombatSystem(PyfaBridge())
        impact_effect = _remote_sensor_damp_module("damp-a", cycle_time=5.0, cap_need=1.0).effects[0]
        impacts = {
            ship.ship_id: [
                ProjectedImpact(
                    source_ship_id="source",
                    target_ship_id=ship.ship_id,
                    effect=impact_effect,
                    strength=1.0,
                )
            ]
        }

        with patch.object(combat.runtime, "apply_projected_effects", wraps=combat.runtime.apply_projected_effects) as wrapped:
            first = combat._effective_profile(ship, impacts)
            second = combat._effective_profile(ship, impacts)

        self.assertEqual(wrapped.call_count, 1)
        self.assertEqual(first.scan_resolution, second.scan_resolution)
        self.assertEqual(first.max_target_range, second.max_target_range)

    def test_effective_profile_cache_invalidates_when_projection_changes(self) -> None:
        ship = _make_ship("effective-cache-target", [], fit_key="effective-cache-fit")
        combat = CombatSystem(PyfaBridge())
        impact_effect = _remote_sensor_damp_module("damp-a", cycle_time=5.0, cap_need=1.0).effects[0]
        first_impacts = {
            ship.ship_id: [
                ProjectedImpact(
                    source_ship_id="source",
                    target_ship_id=ship.ship_id,
                    effect=impact_effect,
                    strength=1.0,
                )
            ]
        }
        second_impacts = {
            ship.ship_id: [
                ProjectedImpact(
                    source_ship_id="source",
                    target_ship_id=ship.ship_id,
                    effect=impact_effect,
                    strength=0.5,
                )
            ]
        }

        with patch.object(combat.runtime, "apply_projected_effects", wraps=combat.runtime.apply_projected_effects) as wrapped:
            first = combat._effective_profile(ship, first_impacts)
            second = combat._effective_profile(ship, second_impacts)

        self.assertEqual(wrapped.call_count, 2)
        self.assertNotEqual(first.scan_resolution, second.scan_resolution)

    def test_substep_steady_state_reuses_cached_pyfa_remote_inputs(self) -> None:
        source = _make_ship(
            "ship-remote-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="remote-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-remote-target",
            [],
            fit_key="remote-target-fit",
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        combat = CombatSystem(PyfaBridge())
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=4), combat)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        base_profiles = {
            ship.runtime.fit_key: replace(ship.profile)
            for ship in world.ships.values()
            if ship.runtime is not None
        }

        def fake_cache_key(runtime, command_boosters, projected_sources):
            booster_sig = tuple(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple(sorted((str(module_id), str(state)) for module_id, state in (snapshot.get("state_by_module_id") or {}).items())),
                )
                for snapshot in (command_boosters or [])
            )
            projected_sig = tuple(
                (
                    str(snapshot.get("fit_key", "") or ""),
                    tuple(sorted((str(module_id), str(state)) for module_id, state in (snapshot.get("state_by_module_id") or {}).items())),
                    str(snapshot.get("pyfa_projection_key_mode", "in_range") or "in_range"),
                    _projection_distance_signature(snapshot),
                )
                for snapshot in (projected_sources or [])
            )
            state_sig = tuple((module.module_id, str(module.state.value)) for module in runtime.modules)
            return runtime.fit_key, state_sig, booster_sig, projected_sig

        def fake_resolve(runtime, command_boosters, projected_sources):
            runtime.diagnostics["pyfa_command_boosters"] = deepcopy(command_boosters or [])
            runtime.diagnostics["pyfa_projected_sources"] = deepcopy(projected_sources or [])
            runtime.diagnostics["pyfa_base_profile"] = replace(base_profiles[runtime.fit_key])
            runtime.diagnostics["pyfa_runtime_resolve_cache"] = "miss"
            return runtime, replace(base_profiles[runtime.fit_key])

        with patch("eve_sim.systems.combat_core.get_runtime_resolve_cache_key", side_effect=fake_cache_key), patch(
            "eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime",
            side_effect=fake_resolve,
        ), patch.object(
            combat,
            "_collect_command_booster_snapshots",
            wraps=combat._collect_command_booster_snapshots,
        ) as collect_command, patch.object(
            combat,
            "_collect_projected_source_snapshots",
            wraps=combat._collect_projected_source_snapshots,
        ) as collect_projected, patch.object(
            combat,
            "_refresh_effective_runtimes_from_pyfa",
            wraps=combat._refresh_effective_runtimes_from_pyfa,
        ) as refresh:
            engine._dt = 1.0
            engine.step()

        self.assertEqual(collect_command.call_count, 1)
        self.assertEqual(collect_projected.call_count, 1)
        self.assertEqual(refresh.call_count, 1)

    def test_engine_uses_fixed_substeps_without_dynamic_slice_query(self) -> None:
        ship = _make_ship("ship-fixed-substeps", [])
        world = WorldState(ships={ship.ship_id: ship})
        combat = CombatSystem(PyfaBridge())
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=4), combat)
        engine.register_ship(ship.ship_id)

        with patch.object(combat, "recommended_time_slice", side_effect=AssertionError("dynamic slice disabled")), patch.object(
            combat,
            "run",
            wraps=combat.run,
        ) as run:
            engine.step()

        self.assertEqual(run.call_count, 4)

    def test_engine_prewarms_world_base_cache_on_init(self) -> None:
        world = WorldState()
        combat = CombatSystem(PyfaBridge())

        with patch("eve_sim.simulation_engine.prewarm_world_base_cache", return_value=0) as prewarm_world:
            SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), combat)

        prewarm_world.assert_called_once_with(world)

    def test_engine_register_ship_prewarms_runtime_base_cache(self) -> None:
        ship = _make_ship("ship-register", [])
        world = WorldState(ships={ship.ship_id: ship})
        combat = CombatSystem(PyfaBridge())

        with patch("eve_sim.simulation_engine.prewarm_world_base_cache", return_value=0), patch(
            "eve_sim.simulation_engine.prewarm_runtime_base_cache",
            return_value=True,
        ) as prewarm_runtime:
            engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), combat)
            engine.register_ship(ship.ship_id)

        prewarm_runtime.assert_called_once_with(ship.runtime)

    def test_lock_deadline_commits_at_substep_end(self) -> None:
        source = _make_ship("ship-lock-source", [], team=Team.BLUE)
        target = _make_ship("ship-lock-target", [], team=Team.RED)
        source.combat.lock_timers[target.ship_id] = 0.25
        world = WorldState(ships={source.ship_id: source, target.ship_id: target}, now=0.25)
        combat = CombatSystem(PyfaBridge())

        combat.run(world, 0.25)

        self.assertNotIn(target.ship_id, source.combat.lock_targets)
        self.assertAlmostEqual(source.combat.lock_timers[target.ship_id], 0.0)
        self.assertNotIn(target.ship_id, source.combat.lock_deadlines)

        world.now = 0.5
        combat.run(world, 0.25)

        self.assertIn(target.ship_id, source.combat.lock_targets)

    def test_target_lock_does_not_start_beyond_max_target_range(self) -> None:
        source = _make_ship("ship-lock-range-source", [], team=Team.BLUE)
        target = _make_ship("ship-lock-range-target", [], team=Team.RED)
        source.profile = replace(source.profile, max_target_range=50_000.0)
        target.nav.position = Vector2(60_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target}, now=0.0)
        combat = CombatSystem(PyfaBridge())

        acquired = combat._ensure_target_lock(
            world,
            source,
            target.ship_id,
            target,
            lock_context="test_lock_range",
            now=0.0,
        )

        self.assertFalse(acquired)
        self.assertNotIn(target.ship_id, source.combat.lock_targets)
        self.assertNotIn(target.ship_id, source.combat.lock_timers)
        self.assertNotIn(target.ship_id, source.combat.lock_deadlines)

    def test_existing_lock_drops_when_target_leaves_max_target_range(self) -> None:
        source = _make_ship(
            "ship-lock-drop-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=1.0)],
            team=Team.BLUE,
        )
        target = _make_ship("ship-lock-drop-target", [], team=Team.RED)
        source.profile = replace(source.profile, max_target_range=50_000.0)
        source.combat.lock_targets.add(target.ship_id)
        source.combat.current_target = target.ship_id
        source.combat.projected_targets["damp-a"] = target.ship_id
        target.nav.position = Vector2(65_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target}, now=1.0)
        combat = CombatSystem(PyfaBridge())

        combat._advance_target_locks(world, 0.1, now=1.0)

        self.assertNotIn(target.ship_id, source.combat.lock_targets)
        self.assertIsNone(source.combat.current_target)
        self.assertNotIn("damp-a", source.combat.projected_targets)

    def test_new_lock_evicts_oldest_idle_lock_before_used_lock(self) -> None:
        source = _make_ship(
            "ship-lock-evict-idle-source",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=1.0)],
            team=Team.BLUE,
        )
        target_a = _make_ship("ship-lock-evict-idle-a", [], team=Team.RED)
        target_b = _make_ship("ship-lock-evict-idle-b", [], team=Team.RED)
        target_c = _make_ship("ship-lock-evict-idle-c", [], team=Team.RED)
        target_d = _make_ship("ship-lock-evict-idle-d", [], team=Team.RED)
        source.profile.max_locked_targets = 3
        source.combat.lock_targets.update({target_a.ship_id, target_b.ship_id, target_c.ship_id})
        source.combat.lock_started_at.update(
            {
                target_a.ship_id: 1.0,
                target_b.ship_id: 2.0,
                target_c.ship_id: 3.0,
            }
        )
        source.combat.current_target = target_a.ship_id
        source.combat.projected_targets["damp-a"] = target_a.ship_id
        world = WorldState(
            ships={
                source.ship_id: source,
                target_a.ship_id: target_a,
                target_b.ship_id: target_b,
                target_c.ship_id: target_c,
                target_d.ship_id: target_d,
            },
            now=10.0,
        )
        combat = CombatSystem(PyfaBridge())

        acquired = combat._ensure_target_lock(
            world,
            source,
            target_d.ship_id,
            target_d,
            lock_context="test_lock",
            now=10.0,
        )

        self.assertFalse(acquired)
        self.assertIn(target_a.ship_id, source.combat.lock_targets)
        self.assertNotIn(target_b.ship_id, source.combat.lock_targets)
        self.assertIn(target_c.ship_id, source.combat.lock_targets)
        self.assertIn(target_d.ship_id, source.combat.lock_timers)
        self.assertIn(target_d.ship_id, source.combat.lock_deadlines)
        self.assertAlmostEqual(source.combat.lock_started_at[target_d.ship_id], 10.0)

    def test_new_lock_evicts_oldest_used_lock_when_all_locks_are_in_use(self) -> None:
        source = _make_ship(
            "ship-lock-evict-used-source",
            [
                _remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=1.0),
                _remote_sensor_damp_module("damp-b", cycle_time=10.0, cap_need=1.0),
            ],
            team=Team.BLUE,
        )
        target_a = _make_ship("ship-lock-evict-used-a", [], team=Team.RED)
        target_b = _make_ship("ship-lock-evict-used-b", [], team=Team.RED)
        target_c = _make_ship("ship-lock-evict-used-c", [], team=Team.RED)
        target_d = _make_ship("ship-lock-evict-used-d", [], team=Team.RED)
        source.profile.max_locked_targets = 3
        source.combat.lock_targets.update({target_a.ship_id, target_b.ship_id, target_c.ship_id})
        source.combat.lock_started_at.update(
            {
                target_a.ship_id: 1.0,
                target_b.ship_id: 2.0,
                target_c.ship_id: 3.0,
            }
        )
        source.combat.current_target = target_a.ship_id
        source.combat.projected_targets["damp-a"] = target_a.ship_id
        source.combat.projected_targets["damp-b"] = target_b.ship_id
        source.combat.fire_delay_timers[target_c.ship_id] = 15.0
        world = WorldState(
            ships={
                source.ship_id: source,
                target_a.ship_id: target_a,
                target_b.ship_id: target_b,
                target_c.ship_id: target_c,
                target_d.ship_id: target_d,
            },
            now=20.0,
        )
        combat = CombatSystem(PyfaBridge())

        acquired = combat._ensure_target_lock(
            world,
            source,
            target_d.ship_id,
            target_d,
            lock_context="test_lock",
            now=20.0,
        )

        self.assertFalse(acquired)
        self.assertNotIn(target_a.ship_id, source.combat.lock_targets)
        self.assertIsNone(source.combat.current_target)
        self.assertNotIn("damp-a", source.combat.projected_targets)
        self.assertIn(target_b.ship_id, source.combat.lock_targets)
        self.assertIn(target_c.ship_id, source.combat.lock_targets)
        self.assertIn("damp-b", source.combat.projected_targets)
        self.assertIn(target_c.ship_id, source.combat.fire_delay_timers)
        self.assertIn(target_d.ship_id, source.combat.lock_timers)
        self.assertAlmostEqual(source.combat.lock_started_at[target_d.ship_id], 20.0)

    def test_remote_projected_target_batches_multiple_sources_into_one_resolve(self) -> None:
        target = _make_ship(
            "ship-target-aggregate",
            [],
            fit_key="aggregate-target-fit",
            team=Team.RED,
        )
        source_a = _make_ship(
            "ship-source-a",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="source-a-fit",
            team=Team.BLUE,
        )
        source_b = _make_ship(
            "ship-source-b",
            [_remote_sensor_damp_module("damp-b", cycle_time=10.0, cap_need=5.0)],
            fit_key="source-b-fit",
            team=Team.BLUE,
        )
        source_a.combat.lock_targets.add(target.ship_id)
        source_b.combat.lock_targets.add(target.ship_id)
        world = WorldState(
            ships={
                source_a.ship_id: source_a,
                source_b.ship_id: source_b,
                target.ship_id: target,
            }
        )

        first_step = self._run_world_steps_and_capture_resolves(world, step_dts=[1.0])[0]

        self.assertEqual(sum(1 for call in first_step if call["fit_key"] == "aggregate-target-fit"), 1)
        target_call = next(call for call in first_step if call["fit_key"] == "aggregate-target-fit")
        self.assertEqual(target_call["projected_fit_keys"], ())
        self.assertEqual(sum(1 for call in first_step if call["fit_key"] == "source-a-fit"), 1)
        self.assertEqual(sum(1 for call in first_step if call["fit_key"] == "source-b-fit"), 1)

    def test_runtime_module_buckets_skip_passive_modules_in_hot_paths(self) -> None:
        passive_module = ModuleRuntime(
            module_id="passive-a",
            group="heat sink",
            state=ModuleState.ONLINE,
            effects=[
                ModuleEffect(
                    name="passive-a-local",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ONLINE,
                    local_mult={"dps": 1.1},
                )
            ],
        )
        pyfa_projected_module = ModuleRuntime(
            module_id="disrupt-a",
            group="weapon disruptor",
            state=ModuleState.ONLINE,
            effects=[
                ModuleEffect(
                    name="disrupt-a-projected",
                    effect_class=EffectClass.PROJECTED,
                    state_required=ModuleState.ACTIVE,
                    cycle_time=10.0,
                    cap_need=5.0,
                    projected_mult={"tracking": 0.8},
                )
            ],
            tags=("controlled", "hostile", "offensive_ewar", "projected"),
        )
        runtime = _make_runtime(
            _make_test_fit_descriptor("bucket-fit", role="ewar"),
            [
                passive_module,
                _propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0),
                _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0),
                _remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0),
                pyfa_projected_module,
            ],
        )

        combat = CombatSystem(PyfaBridge())

        buckets = combat._runtime_module_buckets(runtime)
        cached_buckets = combat._runtime_module_buckets(runtime)

        self.assertIs(buckets, cached_buckets)
        self.assertEqual(
            {module.module_id for module, _metadata in buckets.controlled_entries},
            {"prop-a", "burst-a", "damp-a", "disrupt-a"},
        )
        self.assertEqual(
            {module.module_id for module, _metadata in buckets.command_entries},
            {"burst-a"},
        )
        self.assertEqual(
            {module.module_id for module, _metadata in buckets.runtime_projected_entries},
            {"burst-a", "damp-a", "disrupt-a"},
        )
        self.assertEqual(
            {module.module_id for module, _metadata in buckets.pyfa_projected_entries},
            set(),
        )

    def test_module_static_metadata_cache_ignores_stale_object_id_entry(self) -> None:
        combat = CombatSystem(PyfaBridge())
        passive_module = ModuleRuntime(
            module_id="passive-a",
            group="heat sink",
            state=ModuleState.ONLINE,
            effects=[
                ModuleEffect(
                    name="passive-a-local",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ONLINE,
                    local_mult={"dps": 1.1},
                )
            ],
        )
        stale_metadata = combat._module_static_metadata(passive_module)
        weapon_module = _weapon_module("weapon-a", cycle_time=5.0, cap_need=0.0, damage=25.0)

        combat._module_static_metadata_by_object_id[id(weapon_module)] = (weakref.ref(passive_module), stale_metadata)

        metadata = combat._module_static_metadata(weapon_module)

        self.assertTrue(metadata.is_weapon)
        self.assertFalse(metadata.uses_pyfa_projected_profile)

    def test_factory_does_not_promote_nonactivatable_timed_passive_module(self) -> None:
        factory = RuntimeFromEftFactory()

        class FakeGroup:
            name = "Heat Sink"

        class FakeItem:
            def __init__(self) -> None:
                self.group = FakeGroup()
                self.effects = {}

            @staticmethod
            def isType(type_name: str) -> bool:
                return type_name == "passive"

            @staticmethod
            def getAttribute(_name: str, default=None):
                return default

        class FakeFittedModule:
            def __init__(self) -> None:
                self.item = FakeItem()
                self.charge = None
                self.itemModifiedAttributes = {}

            @staticmethod
            def getModifiedItemAttr(name: str):
                values = {
                    "duration": 10_000.0,
                    "capacitorNeed": 25.0,
                    "damageMultiplierBonus": 10.0,
                }
                return values.get(name, 0.0)

            @staticmethod
            def getModifiedChargeAttr(_name: str):
                return 0.0

            def isValidState(self, state) -> bool:
                active_state = getattr(factory._pyfa, "_fitting_module_state_active", None)
                return state != active_state

        runtime_module = factory._module_effect_pyfa(FakeFittedModule(), 1)

        self.assertIsNotNone(runtime_module)
        assert runtime_module is not None
        self.assertEqual(runtime_module.effects[0].state_required, ModuleState.ONLINE)
        self.assertEqual(runtime_module.effects[0].cap_need, 0.0)

        ship = _make_ship("ship-passive-runtime", [runtime_module])
        world = WorldState(ships={ship.ship_id: ship})
        combat = CombatSystem(PyfaBridge())

        self.assertFalse(combat._update_module_states(world, 1.0))
        self.assertEqual(ship.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertNotIn(runtime_module.module_id, ship.combat.module_cycle_timers)

    def test_snapshot_clamps_nonactivatable_module_active_state_to_online(self) -> None:
        passive_module = ModuleRuntime(
            module_id="passive-a",
            group="heat sink",
            state=ModuleState.ACTIVE,
            effects=[
                ModuleEffect(
                    name="passive-a-local",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ONLINE,
                    local_mult={"dps": 1.1},
                )
            ],
        )
        ship = _make_ship("ship-passive-snapshot", [passive_module])
        world = WorldState(ships={ship.ship_id: ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))

        snapshot = engine.snapshot()

        self.assertEqual(snapshot["ships"][ship.ship_id]["module_states"]["passive-a"], "ONLINE")

    def test_status_display_clamps_nonactivatable_module_active_state_to_online(self) -> None:
        passive_module = ModuleRuntime(
            module_id="passive-a",
            group="heat sink",
            state=ModuleState.ACTIVE,
            effects=[
                ModuleEffect(
                    name="passive-a-local",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ONLINE,
                    local_mult={"dps": 1.1},
                )
            ],
        )

        effective_state = ShipStatusDialog._display_module_state(passive_module, "ACTIVE", None)

        self.assertEqual(effective_state, "ONLINE")

    def test_pyfa_state_mapping_downgrades_invalid_active_request(self) -> None:
        factory = RuntimeFromEftFactory()

        class FakeFittedModule:
            def isValidState(self, state) -> bool:
                active_state = getattr(factory._pyfa, "_fitting_module_state_active", None)
                return state != active_state

        mapped = fleet_setup_module.engine._pyfa_module_state_from_runtime_state(factory, FakeFittedModule(), "ACTIVE")

        self.assertEqual(mapped, factory._pyfa._fitting_module_state_online)

    def test_pyfa_state_mapping_uses_get_max_state_for_invalid_online_request(self) -> None:
        factory = RuntimeFromEftFactory()

        class FakeFittedModule:
            def isValidState(self, state) -> bool:
                online_state = getattr(factory._pyfa, "_fitting_module_state_online", None)
                return state != online_state

            def getMaxState(self, proposedState=None):
                return getattr(factory._pyfa, "_fitting_module_state_offline", None)

        mapped = fleet_setup_module.engine._pyfa_module_state_from_runtime_state(factory, FakeFittedModule(), "ONLINE")

        self.assertEqual(mapped, factory._pyfa._fitting_module_state_offline)

    def test_active_midcycle_module_skips_effect_scan(self) -> None:
        module = _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0)
        counted_effects = CountedEffects(module.effects)
        module.effects = counted_effects
        module.state = ModuleState.ACTIVE

        ship = _make_ship("ship-active-midcycle", [module])
        ship.combat.module_cycle_timers[module.module_id] = 5.0
        world = WorldState(ships={ship.ship_id: ship})

        combat = CombatSystem(PyfaBridge())
        combat._module_static_metadata(module)
        counted_effects.iterations = 0
        for _ in range(4):
            self.assertFalse(combat._update_module_states(world, 0.25))

        self.assertEqual(counted_effects.iterations, 0)
        self.assertAlmostEqual(ship.combat.module_cycle_timers[module.module_id], 5.0)
        self.assertIn(module.module_id, ship.combat.module_cycle_deadlines)

    def test_control_candidates_sleep_midcycle_and_never_modules(self) -> None:
        burst_module = _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0)
        burst_module.state = ModuleState.ACTIVE
        never_module = ModuleRuntime(
            module_id="local-never",
            group="tracking enhancer",
            state=ModuleState.ONLINE,
            effects=[
                ModuleEffect(
                    name="local-never-effect",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ACTIVE,
                    cycle_time=10.0,
                    cap_need=1.0,
                    local_mult={"tracking": 1.1},
                )
            ],
            tags=("affects_local_pyfa_profile", "controlled"),
        )

        ship = _make_ship("ship-control-sleep", [burst_module, never_module])
        ship.combat.module_cycle_timers[burst_module.module_id] = 5.0
        world = WorldState(ships={ship.ship_id: ship})

        combat = CombatSystem(PyfaBridge())
        runtime = ship.runtime
        assert runtime is not None

        initial_candidates = combat._ship_candidate_control_entries(ship, runtime)
        self.assertEqual(
            [module.module_id for module, _metadata in initial_candidates],
            ["burst-a", "local-never"],
        )

        self.assertFalse(combat._update_module_states(world, 0.25))

        sleeping_candidates = combat._ship_candidate_control_entries(ship, runtime)
        self.assertEqual([module.module_id for module, _metadata in sleeping_candidates], [])
        self.assertEqual(ship.combat.module_decision_pending, set())
        self.assertEqual(ship.combat.module_decision_pending_signature, ("burst-a", "local-never"))
        self.assertIn("burst-a", ship.combat.module_cycle_deadlines)

    def test_command_burst_cycle_snapshot_keeps_original_targets_until_next_cycle(self) -> None:
        burst_module = _command_burst_module("burst-a", cycle_time=10.0, cap_need=5.0)
        burst_module.state = ModuleState.ACTIVE

        source = _make_ship("source-burst", [burst_module], team=Team.BLUE, squad_id="SQ1")
        ally_a = _make_ship("ally-a", [], team=Team.BLUE, squad_id="SQ2")
        ally_b = _make_ship("ally-b", [], team=Team.BLUE, squad_id="SQ3")
        source.nav.position = Vector2(0.0, 0.0)
        ally_a.nav.position = Vector2(50_000.0, 0.0)
        ally_b.nav.position = Vector2(150_000.0, 0.0)

        world = WorldState(ships={ship.ship_id: ship for ship in (source, ally_a, ally_b)})
        combat = CombatSystem(PyfaBridge())

        combat._capture_module_cycle_snapshots(
            world,
            source,
            burst_module,
            None,
            area_candidates=[source, ally_a, ally_b],
        )

        first = combat._collect_command_booster_snapshots(world)
        self.assertEqual(set(first.keys()), {source.ship_id, ally_a.ship_id})

        ally_a.nav.position = Vector2(160_000.0, 0.0)
        ally_b.nav.position = Vector2(40_000.0, 0.0)

        second = combat._collect_command_booster_snapshots(world)
        self.assertEqual(set(second.keys()), {source.ship_id, ally_a.ship_id})

    def test_support_projection_cycle_snapshot_keeps_original_target_after_range_change(self) -> None:
        support_module = ModuleRuntime(
            module_id="support-a",
            group="remote tracking computer",
            state=ModuleState.ACTIVE,
            effects=[
                ModuleEffect(
                    name="support-a-tracking",
                    effect_class=EffectClass.PROJECTED,
                    state_required=ModuleState.ACTIVE,
                    range_m=100_000.0,
                    cycle_time=10.0,
                    cap_need=5.0,
                    projected_mult={"tracking": 1.2},
                )
            ],
            tags=("controlled", "projected", "support"),
        )

        source = _make_ship("source-support", [support_module], team=Team.BLUE, squad_id="SQ1")
        target_a = _make_ship("ally-a", [], team=Team.BLUE, squad_id="SQ2")
        target_b = _make_ship("ally-b", [], team=Team.BLUE, squad_id="SQ3")
        source.profile = replace(source.profile, max_target_range=250_000.0)
        source.nav.position = Vector2(0.0, 0.0)
        target_a.nav.position = Vector2(50_000.0, 0.0)
        target_b.nav.position = Vector2(150_000.0, 0.0)
        source.combat.projected_targets[support_module.module_id] = target_a.ship_id
        source.combat.lock_targets.add(target_a.ship_id)

        world = WorldState(ships={ship.ship_id: ship for ship in (source, target_a, target_b)})
        combat = CombatSystem(PyfaBridge())

        combat._capture_module_cycle_snapshots(world, source, support_module, target_a.ship_id)

        first = combat._collect_projected_impacts(world, 1.0)
        self.assertEqual(set(first.keys()), {target_a.ship_id})
        self.assertEqual(len(first[target_a.ship_id]), 1)

        target_a.nav.position = Vector2(160_000.0, 0.0)
        target_b.nav.position = Vector2(40_000.0, 0.0)

        second = combat._collect_projected_impacts(world, 1.0)
        self.assertEqual(set(second.keys()), {target_a.ship_id})
        self.assertEqual(len(second[target_a.ship_id]), 1)

    def test_propulsion_module_wakes_from_event_signal(self) -> None:
        ship = _make_ship("ship-propulsion-signal", [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0)])
        world = WorldState(ships={ship.ship_id: ship})
        combat = CombatSystem(PyfaBridge())

        runtime = ship.runtime
        assert runtime is not None

        self.assertFalse(combat._update_module_states(world, 0.25))
        self.assertEqual(combat._ship_candidate_control_entries(ship, runtime), ())
        self.assertFalse(ship.combat.module_decision_propulsion_active)

        ship.nav.propulsion_command_active = True
        combat._enqueue_ship_control_signal_modules(world, ship, runtime, focus_changed=False)

        self.assertEqual(
            [module.module_id for module, _metadata in combat._ship_candidate_control_entries(ship, runtime)],
            ["prop-a"],
        )

        self.assertFalse(combat._update_module_states(world, 0.25))
        self.assertEqual(runtime.modules[0].state, ModuleState.ACTIVE)

    def test_manual_active_override_wakes_module_without_ai_signal(self) -> None:
        ship = _make_ship("ship-manual-active", [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0)])
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        assert ship.runtime is not None
        module = ship.runtime.modules[0]
        ship.combat.module_manual_modes[module.module_id] = "active"
        ship.combat.module_decision_pending.add(module.module_id)

        engine.step()

        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertIn(module.module_id, ship.combat.module_cycle_timers)

    def test_manual_online_override_waits_for_cycle_end_before_deactivating(self) -> None:
        ship = _make_ship("ship-manual-online", [_propulsion_module("prop-a", cycle_time=5.0, cap_need=5.0)])
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        assert ship.runtime is not None
        module = ship.runtime.modules[0]
        ship.nav.propulsion_command_active = True

        engine.step()
        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertIn(module.module_id, ship.combat.module_cycle_timers)

        ship.combat.module_manual_modes[module.module_id] = "online"
        ship.combat.module_decision_pending.add(module.module_id)
        engine.step()

        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertIn(module.module_id, ship.combat.module_cycle_timers)

        for _ in range(8):
            engine.step()

        self.assertEqual(module.state, ModuleState.ONLINE)
        self.assertNotIn(module.module_id, ship.combat.module_cycle_timers)
        self.assertNotIn(module.module_id, ship.combat.module_cycle_deadlines)

        engine.step()
        self.assertEqual(module.state, ModuleState.ONLINE)

    def test_propulsion_command_off_waits_for_cycle_end_before_deactivating(self) -> None:
        ship = _make_ship("ship-prop-command-off", [_propulsion_module("prop-a", cycle_time=5.0, cap_need=5.0)])
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        assert ship.runtime is not None
        module = ship.runtime.modules[0]
        ship.nav.propulsion_command_active = True

        engine.step()
        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertIn(module.module_id, ship.combat.module_cycle_timers)

        ship.nav.propulsion_command_active = False
        engine.step()

        self.assertEqual(module.state, ModuleState.ACTIVE)
        self.assertIn(module.module_id, ship.combat.module_cycle_timers)

        for _ in range(8):
            engine.step()

        self.assertEqual(module.state, ModuleState.ONLINE)
        self.assertNotIn(module.module_id, ship.combat.module_cycle_timers)
        self.assertNotIn(module.module_id, ship.combat.module_cycle_deadlines)

        engine.step()
        self.assertEqual(module.state, ModuleState.ONLINE)

    def test_damage_control_module_wakes_from_recent_damage_signal(self) -> None:
        ship = _make_ship("ship-dc-signal", [_damage_control_module("dc-a", cycle_time=10.0, cap_need=0.0)])
        world = WorldState(ships={ship.ship_id: ship})
        combat = CombatSystem(PyfaBridge())

        runtime = ship.runtime
        assert runtime is not None

        self.assertFalse(combat._update_module_states(world, 0.25))
        self.assertEqual(combat._ship_candidate_control_entries(ship, runtime), ())
        self.assertFalse(ship.combat.module_decision_recent_enemy_damage_active)

        ship.combat.last_enemy_weapon_damaged_at = float(world.now)
        combat._enqueue_ship_control_signal_modules(world, ship, runtime, focus_changed=False)

        self.assertEqual(
            [module.module_id for module, _metadata in combat._ship_candidate_control_entries(ship, runtime)],
            ["dc-a"],
        )

        self.assertFalse(combat._update_module_states(world, 0.25))
        self.assertEqual(runtime.modules[0].state, ModuleState.ACTIVE)

    def test_external_module_state_change_refreshes_local_pyfa_base_profile(self) -> None:
        neutral_speed = 387.5
        boosted_speed = 1042.7
        ship = _make_ship(
            "ship-external-local",
            [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0, speed_mult=boosted_speed / neutral_speed)],
        )
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        assert ship.runtime is not None
        module = ship.runtime.modules[0]
        ship.fit.max_speed = neutral_speed
        ship.runtime.hull.max_speed = neutral_speed
        boosted_profile = replace(ship.profile, max_speed=boosted_speed)
        neutral_profile = replace(ship.profile, max_speed=neutral_speed)
        ship.profile = boosted_profile
        module.state = ModuleState.ACTIVE
        active_signature = ((module.module_id, "ACTIVE"),)
        ship.runtime.diagnostics["pyfa_base_profile"] = replace(boosted_profile)
        ship.runtime.diagnostics["pyfa_local_state_signature"] = active_signature
        ship.runtime.diagnostics["runtime_local_state_signature"] = active_signature
        ship.runtime.diagnostics["runtime_observed_module_state_signature"] = active_signature
        ship.combat.module_decision_pending_signature = (module.module_id,)
        ship.combat.module_decision_pending = set()
        engine.combat._pyfa_remote_inputs_dirty = False
        engine.combat._cached_command_booster_snapshots = {}
        engine.combat._cached_projected_source_snapshots = {}

        def fake_resolve(runtime, command_boosters, projected_sources):
            del command_boosters, projected_sources
            state = runtime.modules[0].state
            speed = boosted_speed if state == ModuleState.ACTIVE else neutral_speed
            return runtime, replace(neutral_profile, max_speed=speed)

        module.state = ModuleState.ONLINE
        with patch("eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime", side_effect=fake_resolve):
            engine.step()

        self.assertAlmostEqual(ship.profile.max_speed, neutral_speed)

    def test_external_module_state_change_does_not_hard_clamp_velocity_to_new_speed_cap(self) -> None:
        neutral_speed = 387.5
        boosted_speed = 1042.7
        mass = 12_000_000.0
        agility = 0.5
        ship = _make_ship(
            "ship-external-local-velocity",
            [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0, speed_mult=boosted_speed / neutral_speed)],
        )
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        assert ship.runtime is not None
        module = ship.runtime.modules[0]
        ship.fit.max_speed = neutral_speed
        ship.runtime.hull.max_speed = neutral_speed
        neutral_profile = replace(ship.profile, max_speed=neutral_speed, mass=mass, agility=agility)
        boosted_profile = replace(ship.profile, max_speed=boosted_speed, mass=mass, agility=agility)
        ship.profile = boosted_profile
        ship.nav.max_speed = boosted_speed
        ship.nav.velocity = Vector2(boosted_speed, 0.0)
        module.state = ModuleState.ACTIVE
        active_signature = ((module.module_id, "ACTIVE"),)
        ship.runtime.diagnostics["pyfa_base_profile"] = replace(boosted_profile)
        ship.runtime.diagnostics["pyfa_local_state_signature"] = active_signature
        ship.runtime.diagnostics["runtime_local_state_signature"] = active_signature
        ship.runtime.diagnostics["runtime_observed_module_state_signature"] = active_signature
        ship.combat.module_decision_pending_signature = (module.module_id,)
        ship.combat.module_decision_pending = set()
        engine.combat._pyfa_remote_inputs_dirty = False
        engine.combat._cached_command_booster_snapshots = {}
        engine.combat._cached_projected_source_snapshots = {}

        def fake_resolve(runtime, command_boosters, projected_sources):
            del command_boosters, projected_sources
            state = runtime.modules[0].state
            speed = boosted_speed if state == ModuleState.ACTIVE else neutral_speed
            profile = replace(neutral_profile, max_speed=speed, mass=mass, agility=agility)
            return runtime, profile

        expected_tau = mass * agility / 1_000_000.0
        expected_speed_after_step = boosted_speed * math.exp(-1.0 / expected_tau)

        module.state = ModuleState.ONLINE
        with patch("eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime", side_effect=fake_resolve):
            engine.step()

        self.assertAlmostEqual(ship.profile.max_speed, neutral_speed)
        self.assertGreater(ship.nav.velocity.length(), neutral_speed)
        self.assertAlmostEqual(ship.nav.velocity.length(), expected_speed_after_step, places=6)

    def test_external_module_state_change_invalidates_cached_pyfa_projected_effect(self) -> None:
        source = _make_ship(
            "ship-external-scram-source",
            [_warp_scrambler_module("scram-a", cycle_time=5.0, cap_need=1.0)],
            fit_key="external-scram-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-external-scram-target",
            [],
            fit_key="external-scram-target-fit",
            team=Team.RED,
        )
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=1)

        assert source.runtime is not None
        assert target.runtime is not None
        module = source.runtime.modules[0]
        module.state = ModuleState.ACTIVE
        source.combat.lock_targets.add(target.ship_id)
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        source.combat.projected_targets[module.module_id] = target.ship_id
        snapshot_key = engine.combat._module_cycle_snapshot_key(source.ship_id, module.module_id)
        engine.combat._module_cycle_target_snapshots[snapshot_key] = {
            target.ship_id: CycleTargetSnapshot(distance=5_000.0, active_effect_indices={0})
        }

        base_profiles = {
            source.fit.fit_key: replace(source.profile),
            target.fit.fit_key: replace(target.profile, warp_scramble_status=0.0),
        }

        def fake_resolve(runtime, command_boosters, projected_sources):
            del command_boosters
            base = replace(base_profiles[runtime.fit_key])
            if runtime.fit_key == target.fit.fit_key:
                if projected_sources:
                    return runtime, replace(base, warp_scramble_status=2.0)
                return runtime, replace(base, warp_scramble_status=0.0)
            return runtime, base

        with patch("eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime", side_effect=fake_resolve):
            engine.step()
            self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)
            module.state = ModuleState.OFFLINE
            engine.step()

        self.assertAlmostEqual(target.profile.warp_scramble_status, 0.0)
        self.assertNotIn(module.module_id, source.combat.projected_targets)

    def test_scram_blocked_pyfa_propulsion_module_never_enters_active_state(self) -> None:
        source = _make_pyfa_ship_from_fit_text(
            """[Keres, Scram]
Warp Scrambler II
""",
            ship_id="ship-pyfa-scram-source",
            team=Team.RED,
        )
        target = _make_pyfa_ship_from_fit_text(
            """[Ferox, Prop]
50MN Quad LiF Restrained Microwarpdrive
""",
            ship_id="ship-pyfa-scram-target",
            team=Team.BLUE,
        )
        base_max_speed = float(target.profile.max_speed)
        source.nav.position = Vector2(0.0, 0.0)
        target.nav.position = Vector2(1_000.0, 0.0)
        source.combat.lock_targets.add(target.ship_id)
        source.combat.module_manual_modes["mod-1"] = "active"

        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=1)

        engine.step()
        target.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertEqual(target.runtime.diagnostics.get("pyfa_max_state_by_module_id", {}).get("mod-1"), "ONLINE")
        self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)
        self.assertAlmostEqual(target.profile.max_speed, base_max_speed)
        self.assertNotIn("mod-1", target.combat.module_cycle_timers)
        self.assertNotIn("mod-1", target.combat.module_cycle_deadlines)

    def test_scram_pyfa_refresh_immediately_drops_active_propulsion_cycle(self) -> None:
        source = _make_pyfa_ship_from_fit_text(
            """[Keres, Scram]
Warp Scrambler II
""",
            ship_id="ship-pyfa-scram-refresh-source",
            team=Team.RED,
        )
        target = _make_pyfa_ship_from_fit_text(
            """[Ferox, Prop]
50MN Quad LiF Restrained Microwarpdrive
""",
            ship_id="ship-pyfa-scram-refresh-target",
            team=Team.BLUE,
        )
        base_max_speed = float(target.profile.max_speed)
        source.nav.position = Vector2(0.0, 0.0)
        target.nav.position = Vector2(1_000.0, 0.0)

        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=1)

        target.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertIn("mod-1", target.combat.module_cycle_timers)
        self.assertGreater(target.profile.max_speed, base_max_speed)

        source.combat.lock_targets.add(target.ship_id)
        source.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertEqual(target.runtime.diagnostics.get("pyfa_max_state_by_module_id", {}).get("mod-1"), "ONLINE")
        self.assertNotIn("mod-1", target.combat.module_cycle_timers)
        self.assertNotIn("mod-1", target.combat.module_cycle_deadlines)
        self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)
        self.assertAlmostEqual(target.profile.max_speed, base_max_speed)

    def test_scram_release_requeues_manual_active_propulsion_module(self) -> None:
        source = _make_pyfa_ship_from_fit_text(
            """[Keres, Scram]
Warp Scrambler II
""",
            ship_id="ship-pyfa-scram-release-source",
            team=Team.RED,
        )
        target = _make_pyfa_ship_from_fit_text(
            """[Ferox, Prop]
50MN Quad LiF Restrained Microwarpdrive
""",
            ship_id="ship-pyfa-scram-release-target",
            team=Team.BLUE,
        )
        base_max_speed = float(target.profile.max_speed)
        source.nav.position = Vector2(0.0, 0.0)
        target.nav.position = Vector2(1_000.0, 0.0)

        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=1)

        target.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertGreater(target.profile.max_speed, base_max_speed)

        source.combat.lock_targets.add(target.ship_id)
        source.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)

        source.nav.position = Vector2(-100_000.0, 0.0)
        released = False
        for _ in range(8):
            engine.step()
            assert source.runtime is not None
            if source.runtime.modules[0].state == ModuleState.ONLINE and target.profile.warp_scramble_status <= 0.0:
                released = True
                break

        self.assertTrue(released)
        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)

        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertGreater(target.profile.max_speed, base_max_speed)

    def test_scram_release_requeues_propulsion_command_module(self) -> None:
        source = _make_pyfa_ship_from_fit_text(
            """[Keres, Scram]
Warp Scrambler II
""",
            ship_id="ship-pyfa-scram-release-prop-source",
            team=Team.RED,
        )
        target = _make_pyfa_ship_from_fit_text(
            """[Ferox, Prop]
50MN Quad LiF Restrained Microwarpdrive
""",
            ship_id="ship-pyfa-scram-release-prop-target",
            team=Team.BLUE,
        )
        base_max_speed = float(target.profile.max_speed)
        source.nav.position = Vector2(0.0, 0.0)
        target.nav.position = Vector2(1_000.0, 0.0)

        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=1)

        target.nav.propulsion_command_active = True
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertGreater(target.profile.max_speed, base_max_speed)

        source.combat.lock_targets.add(target.ship_id)
        source.combat.module_manual_modes["mod-1"] = "active"
        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)

        source.nav.position = Vector2(-100_000.0, 0.0)
        released = False
        for _ in range(8):
            engine.step()
            assert source.runtime is not None
            if source.runtime.modules[0].state == ModuleState.ONLINE and target.profile.warp_scramble_status <= 0.0:
                released = True
                break

        self.assertTrue(released)
        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)

        engine.step()

        assert target.runtime is not None
        self.assertEqual(target.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertGreater(target.profile.max_speed, base_max_speed)

    def test_scram_only_real_fit_does_not_leave_blocked_mwd_velocity_penalty(self) -> None:
        source = _make_pyfa_ship_from_fit_text(
            """[Vindicator, *模拟复仇者级*装配]
Capacitor Power Relay II
Naiyon's Modified Magnetic Field Stabilizer
Naiyon's Modified Magnetic Field Stabilizer
Naiyon's Modified Magnetic Field Stabilizer
Naiyon's Modified Magnetic Field Stabilizer
Naiyon's Modified Magnetic Field Stabilizer
Naiyon's Modified Magnetic Field Stabilizer

Tobias' Modified Stasis Webifier
Gotan's Modified Stasis Webifier
Hakim's Modified Stasis Webifier
Mizuro's Modified Stasis Webifier
Warp Scrambler II






Antimatter Charge L x1920
""",
            ship_id="ship-realfit-web-source",
            team=Team.RED,
        )
        target = _make_pyfa_ship_from_fit_text(
            """[Dramiel, *模拟德拉米尔级*装配]
Overdrive Injector System II
Overdrive Injector System II
Overdrive Injector System II

Asine's Modified 5MN Microwarpdrive


Small Auxiliary Thrusters II
Small Auxiliary Thrusters II
""",
            ship_id="ship-realfit-web-target",
            team=Team.BLUE,
        )
        base_max_speed = float(target.profile.max_speed)
        source.nav.position = Vector2(0.0, 0.0)
        target.nav.position = Vector2(1_000.0, 0.0)
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        engine = self._make_engine(world, physics_substeps=1)

        for module in source.runtime.modules:
            if "Stasis Web" in module.group or "Warp Scrambler" in module.group:
                source.combat.module_manual_modes[module.module_id] = "active"
        for module in target.runtime.modules:
            if "Propulsion Module" in module.group:
                target.combat.module_manual_modes[module.module_id] = "active"

        for _ in range(8):
            engine.step()

        self.assertLess(target.profile.max_speed, base_max_speed / 50.0)

        for module in source.runtime.modules:
            if "Stasis Web" in module.group:
                source.combat.module_manual_modes[module.module_id] = "online"
            elif "Warp Scrambler" in module.group:
                source.combat.module_manual_modes[module.module_id] = "active"

        for _ in range(5):
            engine.step()

        assert target.runtime is not None
        web_states = [
            module.state
            for module in source.runtime.modules
            if "Stasis Web" in module.group
        ]
        self.assertTrue(web_states)
        self.assertTrue(all(state == ModuleState.ONLINE for state in web_states))
        self.assertAlmostEqual(target.profile.warp_scramble_status, 2.0)
        self.assertAlmostEqual(target.profile.max_speed, base_max_speed)
        self.assertAlmostEqual(
            target.runtime.diagnostics.get("pyfa_base_profile").max_speed,
            base_max_speed,
        )

    def test_external_online_override_survives_runtime_refresh_without_old_cycle_rewake(self) -> None:
        ship = _make_ship(
            "ship-external-rewake",
            [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="external-rewake-fit",
        )
        ship.nav.propulsion_command_active = True
        world = WorldState(ships={ship.ship_id: ship})
        engine = self._make_engine(world, physics_substeps=1)

        neutral_profile = replace(ship.profile)
        boosted_speed = neutral_profile.max_speed * 1.5

        def fake_resolve(runtime, command_boosters, projected_sources):
            del command_boosters, projected_sources
            fresh_runtime = _make_runtime(ship.fit, [deepcopy(module) for module in runtime.modules])
            any_active = any(module.state == ModuleState.ACTIVE for module in runtime.modules)
            profile = replace(neutral_profile, max_speed=boosted_speed if any_active else neutral_profile.max_speed)
            return fresh_runtime, profile

        with patch("eve_sim.systems.combat_core.resolve_runtime_from_pyfa_runtime", side_effect=fake_resolve):
            engine.step()
            runtime_module = ship.runtime.modules[0]
            self.assertEqual(runtime_module.state, ModuleState.ACTIVE)
            self.assertIn(runtime_module.module_id, ship.combat.module_cycle_timers)
            self.assertIn("runtime_observed_module_state_signature", ship.runtime.diagnostics)

            runtime_module.state = ModuleState.ONLINE
            engine.step()

            runtime_module = ship.runtime.modules[0]
            self.assertEqual(runtime_module.state, ModuleState.ONLINE)
            self.assertNotIn(runtime_module.module_id, ship.combat.module_cycle_timers)
            self.assertNotIn(runtime_module.module_id, ship.combat.module_cycle_deadlines)

            for _ in range(12):
                engine.step()

            runtime_module = ship.runtime.modules[0]
            self.assertEqual(runtime_module.state, ModuleState.ONLINE)
            self.assertNotIn(runtime_module.module_id, ship.combat.module_cycle_timers)
            self.assertAlmostEqual(ship.profile.max_speed, neutral_profile.max_speed)

    def test_weapon_focus_module_wakes_when_focus_changes(self) -> None:
        source = _make_ship("ship-focus-source", [_weapon_module("weapon-a", cycle_time=5.0, cap_need=0.0)], team=Team.BLUE)
        target = _make_ship("ship-focus-target", [], team=Team.RED)
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        combat = CombatSystem(PyfaBridge())

        runtime = source.runtime
        assert runtime is not None

        self.assertFalse(combat._update_module_states(world, 0.25))
        self.assertEqual(combat._ship_candidate_control_entries(source, runtime), ())

        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        combat._enqueue_ship_control_signal_modules(world, source, runtime, focus_changed=True)

        self.assertEqual(
            [module.module_id for module, _metadata in combat._ship_candidate_control_entries(source, runtime)],
            ["weapon-a"],
        )

    def test_weapon_damage_applies_once_on_cycle_start(self) -> None:
        source = _make_ship(
            "ship-weapon-source",
            [_weapon_module("weapon-a", cycle_time=5.0, cap_need=0.0, damage=40.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-weapon-target",
            [],
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        engine = self._make_engine(world, physics_substeps=4)

        with patch.object(CombatSystem, "_sample_weapon_fire_delay", return_value=0.0):
            engine.step()
            shield_after_first_step = target.vital.shield
            engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(shield_after_first_step, target.vital.shield_max - 40.0)
        self.assertAlmostEqual(target.vital.shield, shield_after_first_step)

    def test_weapon_focus_without_existing_lock_starts_lock_before_damage(self) -> None:
        source = _make_ship(
            "ship-weapon-lock-source",
            [_weapon_module("weapon-a", cycle_time=5.0, cap_need=0.0, damage=40.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-weapon-lock-target",
            [],
            team=Team.RED,
        )
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        engine = self._make_engine(world, physics_substeps=4)

        with patch.object(CombatSystem, "_sample_weapon_fire_delay", return_value=0.0):
            engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)
        self.assertNotIn(target.ship_id, source.combat.lock_targets)
        self.assertIn(target.ship_id, source.combat.lock_timers)
        self.assertGreater(source.combat.lock_timers[target.ship_id], 0.0)

    def test_remote_rep_applies_once_on_cycle_start(self) -> None:
        source = _make_ship(
            "ship-logi-source",
            [_remote_shield_rep_module("rep-a", cycle_time=5.0, cap_need=0.0, amount=30.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-logi-target",
            [],
            team=Team.BLUE,
        )
        target.vital.shield = target.vital.shield_max - 60.0
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()
        shield_after_first_step = target.vital.shield
        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(shield_after_first_step, target.vital.shield_max - 30.0)
        self.assertAlmostEqual(target.vital.shield, shield_after_first_step)

    def test_remote_armor_rep_skips_shield_only_damage_targets(self) -> None:
        source = _make_ship(
            "ship-armor-logi-source",
            [_remote_armor_rep_module("armor-rep-a", cycle_time=5.0, cap_need=0.0, amount=30.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-armor-logi-target",
            [],
            team=Team.BLUE,
        )
        target.vital.shield = target.vital.shield_max - 60.0
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ONLINE)
        self.assertNotIn("armor-rep-a", source.combat.projected_targets)
        self.assertAlmostEqual(target.vital.armor, target.vital.armor_max)

    def test_remote_shield_rep_prioritizes_lowest_shield_fraction_queue(self) -> None:
        source = _make_ship(
            "ship-shield-logi-source",
            [_remote_shield_rep_module("shield-rep-a", cycle_time=5.0, cap_need=0.0, amount=30.0)],
            team=Team.BLUE,
        )
        shield_target = _make_ship(
            "ship-shield-logi-target",
            [],
            team=Team.BLUE,
        )
        armor_only_target = _make_ship(
            "ship-armor-only-target",
            [],
            team=Team.BLUE,
        )
        shield_target.vital.shield = shield_target.vital.shield_max - 60.0
        armor_only_target.vital.armor = 0.0
        source.combat.lock_targets.update({shield_target.ship_id, armor_only_target.ship_id})
        world = WorldState(
            ships={
                source.ship_id: source,
                shield_target.ship_id: shield_target,
                armor_only_target.ship_id: armor_only_target,
            }
        )
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.projected_targets.get("shield-rep-a"), shield_target.ship_id)
        self.assertAlmostEqual(shield_target.vital.shield, shield_target.vital.shield_max - 30.0)
        self.assertAlmostEqual(armor_only_target.vital.armor, 0.0)

    def test_remote_repair_queue_updates_after_damage_event_next_step(self) -> None:
        source = _make_ship(
            "ship-event-logi-source",
            [_remote_shield_rep_module("shield-rep-a", cycle_time=5.0, cap_need=0.0, amount=30.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-event-logi-target",
            [],
            team=Team.BLUE,
        )
        attacker = _make_ship(
            "ship-event-attacker",
            [_weapon_module("weapon-a", cycle_time=5.0, cap_need=0.0, damage=40.0)],
            team=Team.RED,
            squad_id="RSQ1",
        )
        source.combat.lock_targets.add(target.ship_id)
        attacker.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target, attacker.ship_id: attacker})
        world.squad_focus_queues[CombatSystem._focus_key(attacker.team, attacker.squad_id)] = [target.ship_id]
        engine = self._make_engine(world, physics_substeps=1)

        with patch.object(CombatSystem, "_sample_weapon_fire_delay", return_value=0.0):
            engine.step()
            shield_after_damage = target.vital.shield
            assert source.runtime is not None
            self.assertEqual(source.runtime.modules[0].state, ModuleState.ONLINE)
            engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.projected_targets.get("shield-rep-a"), target.ship_id)
        self.assertAlmostEqual(target.vital.shield, shield_after_damage + 30.0)

    def test_generic_ally_support_defaults_to_nearest_ally(self) -> None:
        source = _make_ship(
            "ship-ally-support-source",
            [_ally_support_module("support-a", cycle_time=5.0, cap_need=0.0)],
            team=Team.BLUE,
        )
        near_target = _make_ship(
            "ship-ally-support-near",
            [],
            team=Team.BLUE,
        )
        far_target = _make_ship(
            "ship-ally-support-far",
            [],
            team=Team.BLUE,
        )
        near_target.nav.position = Vector2(1_000.0, 0.0)
        far_target.nav.position = Vector2(8_000.0, 0.0)
        source.combat.lock_targets.update({near_target.ship_id, far_target.ship_id})
        world = WorldState(ships={source.ship_id: source, near_target.ship_id: near_target, far_target.ship_id: far_target})
        engine = self._make_engine(world, physics_substeps=1)

        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.projected_targets.get("support-a"), near_target.ship_id)

    def test_cap_warfare_applies_once_on_cycle_start(self) -> None:
        source = _make_ship(
            "ship-neut-source",
            [_energy_neutralizer_module("neut-a", cycle_time=5.0, cap_need=0.0, amount=18.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-neut-target",
            [],
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()
        cap_after_first_step = target.vital.cap
        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(cap_after_first_step, target.vital.cap_max - 18.0)
        self.assertAlmostEqual(target.vital.cap, cap_after_first_step)

    def test_nosferatu_cycle_start_uses_final_energy_warfare_resistance(self) -> None:
        source = _make_ship(
            "ship-nos-source",
            [_nosferatu_module("nos-a", cycle_time=5.0, cap_need=0.0, amount=20.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-nos-target",
            [],
            team=Team.RED,
        )
        target.profile = replace(target.profile, energy_warfare_resistance=0.5)
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()
        cap_after_first_step = target.vital.cap
        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(cap_after_first_step, target.vital.cap_max - 10.0)
        self.assertAlmostEqual(target.vital.cap, cap_after_first_step)

    def test_smart_bomb_damage_applies_once_on_cycle_start(self) -> None:
        source = _make_ship(
            "ship-smartbomb-source",
            [_smart_bomb_module("smartbomb-a", cycle_time=5.0, cap_need=0.0, damage=25.0, range_m=5_000.0)],
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-smartbomb-target",
            [],
            team=Team.RED,
        )
        target.nav.position = Vector2(1_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        engine = self._make_engine(world, physics_substeps=4)

        engine.step()
        shield_after_first_step = target.vital.shield
        engine.step()

        assert source.runtime is not None
        self.assertEqual(source.runtime.modules[0].state, ModuleState.ACTIVE)
        self.assertAlmostEqual(shield_after_first_step, target.vital.shield_max - 25.0)
        self.assertAlmostEqual(target.vital.shield, shield_after_first_step)

    def test_pyfa_smartbomb_effect_extracts_damage_while_online(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_NESTOR_SMARTBOMB_FIT, ship_id="pyfa-smartbomb-source", team=Team.BLUE)

        assert source.runtime is not None
        smartbombs = [module for module in source.runtime.modules if "smart_bomb" in module.tags]
        self.assertTrue(smartbombs)
        for module in smartbombs:
            self.assertTrue(module.effects)
            effect = module.effects[0]
            self.assertGreater(
                sum(float(effect.projected_add.get(key, 0.0) or 0.0) for key in ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive")),
                0.0,
            )

    def test_pyfa_smartbomb_manual_activation_after_start_damages_friendly_ship_without_enemies(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_NESTOR_SMARTBOMB_FIT, ship_id="pyfa-smartbomb-manual-source", team=Team.BLUE)
        ally = _make_ship("pyfa-smartbomb-manual-ally", [], team=Team.BLUE)
        ally.nav.position = Vector2(1_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, ally.ship_id: ally})
        engine = self._make_engine(world, physics_substeps=1)

        assert source.runtime is not None
        smartbomb_id = next(module.module_id for module in source.runtime.modules if "smart_bomb" in module.tags)

        engine.step()
        self.assertAlmostEqual(ally.vital.shield, ally.vital.shield_max)

        source.combat.module_manual_modes[smartbomb_id] = "active"
        source.combat.module_decision_pending.add(smartbomb_id)
        engine.step()

        assert source.runtime is not None
        smartbomb = next(module for module in source.runtime.modules if module.module_id == smartbomb_id)
        self.assertEqual(smartbomb.state, ModuleState.ACTIVE)
        self.assertLess(ally.vital.shield + ally.vital.armor + ally.vital.structure, ally.vital.shield_max + ally.vital.armor_max + ally.vital.structure_max)

    def test_pyfa_smartbomb_manual_activation_after_start_damages_projectiles(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_NESTOR_SMARTBOMB_FIT, ship_id="pyfa-smartbomb-projectile-source", team=Team.RED)
        world = WorldState(ships={source.ship_id: source})
        engine = self._make_engine(world, physics_substeps=1)

        assert source.runtime is not None
        smartbomb_id = next(module.module_id for module in source.runtime.modules if "smart_bomb" in module.tags)

        engine.step()
        world.projectiles["proj-smartbomb-test"] = ProjectileEntity(
            projectile_id="proj-smartbomb-test",
            kind="missile",
            source_ship_id="dummy-source",
            source_module_id="dummy-module",
            team=Team.BLUE,
            position=Vector2(1_000.0, 0.0),
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            target_ship_id=None,
            speed=0.0,
            max_speed=0.0,
            max_range=10_000.0,
            distance_traveled=0.0,
            flight_time=60.0,
            age=0.0,
            acceleration_time=0.0,
            damage_em=0.0,
            damage_thermal=0.0,
            damage_kinetic=0.0,
            damage_explosive=0.0,
            explosion_radius=0.0,
            explosion_velocity=0.0,
            damage_reduction_factor=0.5,
            shield=0.0,
            armor=0.0,
            structure=100.0,
            shield_max=0.0,
            armor_max=0.0,
            structure_max=100.0,
        )

        source.combat.module_manual_modes[smartbomb_id] = "active"
        source.combat.module_decision_pending.add(smartbomb_id)
        engine.step()

        assert source.runtime is not None
        smartbomb = next(module for module in source.runtime.modules if module.module_id == smartbomb_id)
        self.assertEqual(smartbomb.state, ModuleState.ACTIVE)
        self.assertNotIn("proj-smartbomb-test", world.projectiles)

    def test_missile_weapon_spawns_projectile_and_hits_after_travel(self) -> None:
        source = _make_ship(
            "ship-missile-source",
            [
                _missile_weapon_module(
                    "launcher-a",
                    cycle_time=5.0,
                    cap_need=0.0,
                    damage=40.0,
                    projectile_speed=1_000.0,
                    flight_time=5.0,
                    range_m=10_000.0,
                    explosion_radius=120.0,
                    explosion_velocity=1_000.0,
                )
            ],
            team=Team.BLUE,
        )
        target = _make_ship("ship-missile-target", [], team=Team.RED)
        target.nav.position = Vector2(2_500.0, 0.0)
        source.combat.lock_targets.add(target.ship_id)
        source.combat.projected_targets["launcher-a"] = target.ship_id
        source.combat.fire_delay_timers[target.ship_id] = 0.0
        world = WorldState(
            ships={source.ship_id: source, target.ship_id: target},
            squad_focus_queues={"BLUE:SQ1": [target.ship_id]},
        )
        engine = self._make_engine(world)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

        engine.step()
        self.assertEqual(len(world.projectiles), 0)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max - 40.0)

    def test_smart_bomb_destroys_in_flight_missiles(self) -> None:
        missile_source = _make_ship(
            "ship-missile-firewall-source",
            [_missile_weapon_module("launcher-a", cycle_time=5.0, cap_need=0.0, projectile_speed=1_000.0, flight_time=8.0, range_m=10_000.0)],
            team=Team.BLUE,
        )
        target = _make_ship("ship-missile-firewall-target", [], team=Team.RED)
        firewall = _make_ship(
            "ship-smartbomb-firewall",
            [_smart_bomb_module("smartbomb-a", cycle_time=5.0, cap_need=0.0, damage=25.0, range_m=1_500.0)],
            team=Team.RED,
        )
        target.nav.position = Vector2(6_000.0, 0.0)
        firewall.nav.position = Vector2(1_000.0, 0.0)
        missile_source.combat.lock_targets.add(target.ship_id)
        missile_source.combat.projected_targets["launcher-a"] = target.ship_id
        missile_source.combat.fire_delay_timers[target.ship_id] = 0.0
        world = WorldState(
            ships={
                missile_source.ship_id: missile_source,
                target.ship_id: target,
                firewall.ship_id: firewall,
            },
            squad_focus_queues={"BLUE:SQ1": [target.ship_id]},
        )
        engine = self._make_engine(world)

        engine.step()
        self.assertEqual(len(world.projectiles), 0)

        for _ in range(6):
            engine.step()
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

    def test_smart_bomb_respects_projectile_hp_and_resistance(self) -> None:
        missile_source = _make_ship(
            "ship-missile-durable-source",
            [
                _missile_weapon_module(
                    "launcher-a",
                    cycle_time=5.0,
                    cap_need=0.0,
                    projectile_speed=1_000.0,
                    flight_time=8.0,
                    range_m=10_000.0,
                    projectile_shield_hp=20.0,
                    projectile_structure_hp=20.0,
                    projectile_shield_resonance_em=0.5,
                )
            ],
            team=Team.BLUE,
        )
        target = _make_ship("ship-missile-durable-target", [], team=Team.RED)
        firewall = _make_ship(
            "ship-smartbomb-durable-firewall",
            [_smart_bomb_module("smartbomb-a", cycle_time=1.0, cap_need=0.0, damage=25.0, range_m=1_500.0)],
            team=Team.RED,
        )
        target.nav.position = Vector2(6_000.0, 0.0)
        firewall.nav.position = Vector2(1_000.0, 0.0)
        missile_source.combat.lock_targets.add(target.ship_id)
        missile_source.combat.projected_targets["launcher-a"] = target.ship_id
        missile_source.combat.fire_delay_timers[target.ship_id] = 0.0
        world = WorldState(
            ships={
                missile_source.ship_id: missile_source,
                target.ship_id: target,
                firewall.ship_id: firewall,
            },
            squad_focus_queues={"BLUE:SQ1": [target.ship_id]},
        )
        engine = self._make_engine(world)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        durable_projectile = next(iter(world.projectiles.values()))
        self.assertAlmostEqual(durable_projectile.shield, 7.5)
        self.assertAlmostEqual(durable_projectile.structure, 20.0)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        durable_projectile = next(iter(world.projectiles.values()))
        self.assertAlmostEqual(durable_projectile.shield, 0.0)
        self.assertAlmostEqual(durable_projectile.structure, 10.0)

        engine.step()
        self.assertEqual(len(world.projectiles), 0)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

    def test_bomb_launcher_launches_without_lock_and_explodes_at_endpoint(self) -> None:
        source = _make_ship(
            "ship-bomb-source",
            [
                _bomb_launcher_module(
                    "bomb-a",
                    cycle_time=10.0,
                    cap_need=0.0,
                    damage=500.0,
                    projectile_speed=2_000.0,
                    flight_time=3.0,
                    range_m=6_000.0,
                    explosion_radius=400.0,
                    blast_radius=900.0,
                )
            ],
            team=Team.BLUE,
        )
        target = _make_ship("ship-bomb-target", [], team=Team.RED)
        bystander = _make_ship("ship-bomb-bystander", [], team=Team.RED)
        target.nav.position = Vector2(5_900.0, 0.0)
        bystander.nav.position = Vector2(6_200.0, 0.0)
        source.combat.projected_targets["bomb-a"] = target.ship_id
        source.combat.fire_delay_timers[target.ship_id] = 0.0
        world = WorldState(
            ships={source.ship_id: source, target.ship_id: target, bystander.ship_id: bystander},
            squad_focus_queues={"BLUE:SQ1": [target.ship_id]},
        )
        engine = self._make_engine(world)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

        engine.step()
        self.assertEqual(len(world.projectiles), 1)
        self.assertAlmostEqual(target.vital.shield, target.vital.shield_max)

        engine.step()
        self.assertEqual(len(world.projectiles), 0)
        self.assertTrue(world.projectile_blasts)
        self.assertLess(target.vital.shield, target.vital.shield_max)
        self.assertLess(bystander.vital.shield, bystander.vital.shield_max)

    def test_pyfa_ham_launcher_without_explicit_loaded_charge_builds_projected_missile_weapon(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_CARACAL_NAVY_HAM_FIT, ship_id="ham-source", team=Team.BLUE)

        assert source.runtime is not None
        launcher_modules = [module for module in source.runtime.modules if "launcher" in module.group.lower()]
        self.assertEqual(len(launcher_modules), 6)
        for module in launcher_modules:
            self.assertIn("weapon", module.tags)
            self.assertIn("projected", module.tags)
            self.assertGreater(module.charge_capacity, 0)
            self.assertGreater(module.charge_remaining, 0.0)
            self.assertTrue(module.effects)
            effect = module.effects[0]
            self.assertEqual(effect.effect_class, EffectClass.PROJECTED)
            self.assertGreater(effect.range_m, 0.0)
            self.assertGreater(float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0), 0.5)
            self.assertGreater(
                sum(float(effect.projected_add.get(key, 0.0) or 0.0) for key in ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive")),
                0.0,
            )

    def test_pyfa_ham_fit_focus_fire_spawns_projectiles_and_applies_damage(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_CARACAL_NAVY_HAM_FIT, ship_id="ham-focus-source", team=Team.BLUE)
        target = _make_ship("ham-focus-target", [], team=Team.RED)
        target.nav.position = Vector2(10_000.0, 0.0)
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        engine = self._make_engine(world, physics_substeps=1)

        observed_projectiles = False
        with patch.object(CombatSystem, "_sample_weapon_fire_delay", return_value=0.0):
            for _ in range(8):
                engine.step()
                if world.projectiles:
                    observed_projectiles = True

        self.assertTrue(observed_projectiles)
        self.assertLess(target.vital.shield, target.vital.shield_max)

    def test_propulsion_command_without_target_accelerates_along_current_velocity(self) -> None:
        ship = _make_ship("ship-propulsion-burn", [_propulsion_module("prop-a", cycle_time=10.0, cap_need=5.0)])
        ship.nav.velocity = Vector2(181.25, 0.0)
        ship.nav.facing_deg = 0.0
        ship.nav.max_speed = 419.58
        ship.nav.propulsion_command_active = True
        world = WorldState(ships={ship.ship_id: ship})

        MovementSystem().run(world, 0.1)

        self.assertGreater(ship.nav.velocity.length(), 181.25)

    def test_movement_linear_acceleration_matches_exponential_response(self) -> None:
        ship = _make_ship("ship-linear-accel", [])
        _set_motion_state(ship, max_speed=500.0, mass=12_000_000.0, agility=0.5)
        ship.nav.command_target = Vector2(100_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        dt = 1.5
        tau = ship.profile.mass * ship.profile.agility / 1_000_000.0
        decay = math.exp(-dt / tau)
        expected_speed = ship.profile.max_speed * (1.0 - decay)
        expected_distance = ship.profile.max_speed * dt - ship.profile.max_speed * tau * (1.0 - decay)

        MovementSystem().run(world, dt)

        self.assertAlmostEqual(ship.nav.velocity.x, expected_speed, places=6)
        self.assertAlmostEqual(ship.nav.velocity.y, 0.0, places=6)
        self.assertAlmostEqual(ship.nav.position.x, expected_distance, places=6)
        self.assertAlmostEqual(ship.nav.position.y, 0.0, places=6)

    def test_movement_small_angle_turn_uses_stable_angular_velocity_limit(self) -> None:
        ship = _make_ship("ship-small-turn", [])
        _set_motion_state(ship, max_speed=500.0, mass=12_000_000.0, agility=0.5)
        ship.nav.velocity = Vector2(400.0, 0.0)
        ship.nav.facing_deg = 0.0
        desired_angle = 45.0
        ship.nav.command_target = Vector2(math.cos(math.radians(desired_angle)) * 100_000.0, math.sin(math.radians(desired_angle)) * 100_000.0)
        world = WorldState(ships={ship.ship_id: ship})

        dt = 1.0
        tau = ship.profile.mass * ship.profile.agility / 1_000_000.0
        expected_omega = math.sqrt(ship.profile.max_speed ** 2 - 400.0 ** 2) / (tau * 400.0)
        expected_heading = math.degrees(expected_omega * dt)

        MovementSystem().run(world, dt)

        self.assertAlmostEqual(ship.nav.facing_deg, expected_heading, places=6)
        self.assertLess(ship.nav.facing_deg, desired_angle)

    def test_movement_large_angle_turn_uses_exponential_vector_split(self) -> None:
        ship = _make_ship("ship-large-turn", [])
        _set_motion_state(ship, max_speed=500.0, mass=12_000_000.0, agility=0.5)
        ship.nav.velocity = Vector2(300.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.nav.command_target = Vector2(0.0, 100_000.0)
        world = WorldState(ships={ship.ship_id: ship})

        dt = 1.0
        tau = ship.profile.mass * ship.profile.agility / 1_000_000.0
        decay = math.exp(-dt / tau)
        expected_x = 300.0 * decay
        expected_y = ship.profile.max_speed * (1.0 - decay)

        MovementSystem().run(world, dt)

        self.assertAlmostEqual(ship.nav.velocity.x, expected_x, places=6)
        self.assertAlmostEqual(ship.nav.velocity.y, expected_y, places=6)

    def test_movement_uses_updated_profile_mass_and_speed_cap(self) -> None:
        ship = _make_ship("ship-mass-feedback", [])
        _set_motion_state(ship, max_speed=500.0, mass=10_000_000.0, agility=0.5)
        ship.nav.command_target = Vector2(100_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        MovementSystem().run(world, 1.0)
        first_speed = ship.nav.velocity.length()

        ship.profile = replace(ship.profile, max_speed=800.0, mass=20_000_000.0, agility=0.5)
        if ship.runtime is not None:
            ship.runtime.diagnostics["motion_params"] = {"mass": 10_000_000.0, "agility": 0.5}

        tau = ship.profile.mass * ship.profile.agility / 1_000_000.0
        expected_speed = ship.profile.max_speed + (first_speed - ship.profile.max_speed) * math.exp(-1.0 / tau)

        MovementSystem().run(world, 1.0)

        self.assertAlmostEqual(ship.nav.velocity.length(), expected_speed, places=6)

    def test_movement_speed_cap_drop_preserves_inertial_overspeed(self) -> None:
        ship = _make_ship("ship-speed-cap-drop", [])
        mass = 10_000_000.0
        agility = 0.5
        _set_motion_state(ship, max_speed=1_000.0, mass=mass, agility=agility)
        ship.nav.velocity = Vector2(800.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        ship.profile = replace(ship.profile, max_speed=300.0, mass=mass, agility=agility)

        tau = mass * agility / 1_000_000.0
        expected_speed = 800.0 * math.exp(-1.0 / tau)

        MovementSystem().run(world, 1.0)

        self.assertAlmostEqual(ship.nav.max_speed, 300.0)
        self.assertGreater(ship.nav.velocity.length(), ship.nav.max_speed)
        self.assertAlmostEqual(ship.nav.velocity.length(), expected_speed, places=6)

    def test_large_compact_fleet_uses_all_visible_perception_result(self) -> None:
        world = WorldState()
        for idx in range(30):
            ship = _make_ship(f"ship-perception-{idx}", [])
            ship.nav.position = Vector2(float(idx * 100.0), float((idx % 3) * 50.0))
            world.ships[ship.ship_id] = ship

        PerceptionSystem(sensor_range=10_000.0).run(world)

        for ship in world.ships.values():
            self.assertEqual(len(ship.perception), 29)
            self.assertNotIn(ship.ship_id, ship.perception)

    def test_target_candidate_pools_use_perception_and_lock_range_pruning(self) -> None:
        source = _make_ship("ship-pool-source", [], team=Team.BLUE)
        ally = _make_ship("ship-pool-ally", [], team=Team.BLUE)
        near_enemy = _make_ship("ship-pool-near", [], team=Team.RED)
        far_enemy = _make_ship("ship-pool-far", [], team=Team.RED)
        sticky_enemy = _make_ship("ship-pool-sticky", [], team=Team.RED)
        source.profile = replace(source.profile, max_target_range=50_000.0)
        ally.nav.position = Vector2(20_000.0, 0.0)
        near_enemy.nav.position = Vector2(40_000.0, 0.0)
        far_enemy.nav.position = Vector2(80_000.0, 0.0)
        sticky_enemy.nav.position = Vector2(10_000.0, 0.0)
        source.perception = [ally.ship_id, near_enemy.ship_id, far_enemy.ship_id]
        source.combat.current_target = sticky_enemy.ship_id
        world = WorldState(
            ships={
                source.ship_id: source,
                ally.ship_id: ally,
                near_enemy.ship_id: near_enemy,
                far_enemy.ship_id: far_enemy,
                sticky_enemy.ship_id: sticky_enemy,
            }
        )
        combat = CombatSystem(PyfaBridge())

        allies_pool, enemies_pool, ally_ids, enemy_ids = combat._ship_target_candidate_pools(
            world,
            source,
            focus_queue=(),
        )

        self.assertEqual({ship.ship_id for ship in allies_pool}, {ally.ship_id})
        self.assertEqual({ship.ship_id for ship in enemies_pool}, {near_enemy.ship_id, sticky_enemy.ship_id})
        self.assertEqual(ally_ids, {ally.ship_id})
        self.assertEqual(enemy_ids, {near_enemy.ship_id, sticky_enemy.ship_id})
        self.assertNotIn(far_enemy.ship_id, enemy_ids)

    def test_target_candidate_pools_respect_split_perception_without_raw_fallback(self) -> None:
        source = _make_ship("ship-pool-split-source", [], team=Team.BLUE)
        ally = _make_ship("ship-pool-split-ally", [], team=Team.BLUE)
        source.profile = replace(source.profile, max_target_range=50_000.0)
        ally.nav.position = Vector2(20_000.0, 0.0)
        source.perception = [ally.ship_id]
        source.perception_allies = [ally.ship_id]
        source.perception_enemies = []
        source.perception_split_ready = True
        world = WorldState(ships={source.ship_id: source, ally.ship_id: ally})
        combat = CombatSystem(PyfaBridge())

        allies_pool, enemies_pool, ally_ids, enemy_ids = combat._ship_target_candidate_pools(
            world,
            source,
            focus_queue=(),
            include_allies=False,
            include_enemies=True,
        )

        self.assertEqual(allies_pool, [])
        self.assertEqual(enemies_pool, [])
        self.assertEqual(ally_ids, set())
        self.assertEqual(enemy_ids, set())

    def test_perception_excludes_warping_ships(self) -> None:
        visible = _make_ship("ship-visible", [])
        warping = _make_ship("ship-warping", [])
        warping.nav.warp.phase = "warp"
        visible.nav.position = Vector2(0.0, 0.0)
        warping.nav.position = Vector2(500.0, 0.0)
        world = WorldState(ships={visible.ship_id: visible, warping.ship_id: warping})

        PerceptionSystem(sensor_range=10_000.0).run(world)

        self.assertEqual(visible.perception, [])
        self.assertEqual(warping.perception, [])

    def test_perception_is_split_by_team_and_system(self) -> None:
        blue = _make_ship("ship-blue", [], team=Team.BLUE)
        red_same_system = _make_ship("ship-red-same", [], team=Team.RED)
        red_other_system = _make_ship("ship-red-other", [], team=Team.RED)
        blue.nav.position = Vector2(0.0, 0.0)
        red_same_system.nav.position = Vector2(1_000.0, 0.0)
        red_other_system.nav.position = Vector2(1_000.0, 0.0)
        blue.nav.system_id = "alpha"
        red_same_system.nav.system_id = "alpha"
        red_other_system.nav.system_id = "beta"
        world = WorldState(
            ships={
                blue.ship_id: blue,
                red_same_system.ship_id: red_same_system,
                red_other_system.ship_id: red_other_system,
            }
        )

        PerceptionSystem(sensor_range=10_000.0).run(world)

        self.assertEqual(blue.perception_allies, [])
        self.assertEqual(blue.perception_enemies, [red_same_system.ship_id])
        self.assertEqual(blue.perception, [red_same_system.ship_id])
        self.assertTrue(blue.perception_split_ready)

    def test_warp_alignment_enters_warp_and_spends_capacitor(self) -> None:
        ship = _make_ship("ship-warp-align", [])
        _set_motion_state(ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        ship.profile = replace(ship.profile, warp_speed_au_s=3.0, warp_capacitor_need=1.0)
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = 1_000.0
        ship.vital.cap_max = 1_000.0
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        movement = MovementSystem()
        movement.run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "warp")
        self.assertLess(ship.vital.cap, 1_000.0)
        warp_start_position = ship.nav.position.x

        movement.run(world, 0.1)

        self.assertGreater(ship.nav.position.x, warp_start_position)
        for _ in range(200):
            if ship.nav.warp.phase == "idle":
                break
            movement.run(world, 0.5)

        self.assertEqual(ship.nav.warp.phase, "idle")
        self.assertAlmostEqual(ship.nav.position.x, 300_000.0, delta=1.0)
        self.assertAlmostEqual(ship.nav.position.y, 0.0, delta=1.0)

    def test_warp_uses_partial_distance_when_capacitor_is_insufficient(self) -> None:
        ship = _make_ship("ship-partial-warp", [])
        mass = 10_000_000.0
        warp_capacitor_need = 4.0
        available_cap = 50.0
        _set_motion_state(ship, max_speed=1_000.0, mass=mass, agility=0.5)
        ship.profile = replace(ship.profile, warp_speed_au_s=3.0, warp_capacitor_need=warp_capacitor_need)
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = available_cap
        ship.vital.cap_max = available_cap
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(400_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        expected_distance = (available_cap / (mass * warp_capacitor_need)) * MovementSystem.AU_METERS
        movement = MovementSystem()
        movement.run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "warp")
        self.assertAlmostEqual(ship.nav.warp.warp_distance_m, expected_distance, delta=1e-3)
        expected_final_x = ship.nav.position.x + ship.nav.warp.warp_distance_m

        for _ in range(200):
            if ship.nav.warp.phase == "idle":
                break
            movement.run(world, 0.5)

        self.assertEqual(ship.nav.warp.phase, "idle")
        self.assertAlmostEqual(ship.nav.position.x, expected_final_x, delta=1.0)
        self.assertAlmostEqual(ship.vital.cap, 0.0, delta=1e-6)

    def test_scram_cancels_warp_alignment_before_entry(self) -> None:
        ship = _make_ship("ship-warp-scrammed", [])
        _set_motion_state(ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        ship.profile = replace(
            ship.profile,
            warp_speed_au_s=3.0,
            warp_capacitor_need=1.0,
            warp_scramble_status=2.0,
        )
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = 1_000.0
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        MovementSystem().run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "idle")
        self.assertAlmostEqual(ship.vital.cap, 1_000.0, delta=1e-9)
        self.assertIsNone(ship.nav.command_target)

    def test_warp_bubble_blocks_warp_until_ship_leaves_field(self) -> None:
        ship = _make_ship("ship-warp-bubbled", [])
        _set_motion_state(ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        ship.profile = replace(ship.profile, warp_speed_au_s=3.0, warp_capacitor_need=1.0)
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = 1_000.0
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(
            ships={ship.ship_id: ship},
            bubble_fields={
                "bubble-a": _bubble_field(
                    "bubble-a",
                    position=Vector2(0.0, 0.0),
                    radius_m=20_000.0,
                )
            },
        )

        movement = MovementSystem()
        movement.run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "align")
        self.assertEqual(len(ship.nav.warp.interdiction_snapshots), 1)

        ship.nav.position = Vector2(25_000.0, 0.0)
        ship.nav.velocity = Vector2(800.0, 0.0)
        movement.run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "warp")

    def test_warp_is_intercepted_at_bubble_edge(self) -> None:
        ship = _make_ship("ship-warp-intercepted", [])
        _set_motion_state(ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        ship.profile = replace(ship.profile, warp_speed_au_s=3.0, warp_capacitor_need=1.0)
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = 1_000.0
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(
            ships={ship.ship_id: ship},
            bubble_fields={
                "bubble-a": _bubble_field(
                    "bubble-a",
                    position=Vector2(290_000.0, 0.0),
                    radius_m=20_000.0,
                )
            },
        )

        MovementSystem().run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "warp")
        self.assertIsNotNone(ship.nav.warp.destination)
        self.assertAlmostEqual(ship.nav.warp.destination.x, 270_000.0, delta=1.0)

    def test_bubble_added_after_warp_order_does_not_change_snapshotted_exit_point(self) -> None:
        ship = _make_ship("ship-warp-snapshot", [])
        _set_motion_state(ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        ship.profile = replace(ship.profile, warp_speed_au_s=3.0, warp_capacitor_need=1.0)
        ship.nav.max_speed = ship.profile.max_speed
        ship.nav.velocity = Vector2(0.0, 0.0)
        ship.nav.facing_deg = 0.0
        ship.vital.cap = 1_000.0
        ship.nav.warp.phase = "align"
        ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(ships={ship.ship_id: ship})

        movement = MovementSystem()
        movement.run(world, 0.1)
        self.assertEqual(len(ship.nav.warp.interdiction_snapshots), 0)
        world.bubble_fields["bubble-a"] = _bubble_field(
            "bubble-a",
            position=Vector2(290_000.0, 0.0),
            radius_m=20_000.0,
        )

        ship.nav.velocity = Vector2(800.0, 0.0)
        ship.nav.facing_deg = 0.0
        movement.run(world, 0.1)

        self.assertEqual(ship.nav.warp.phase, "warp")
        self.assertIsNotNone(ship.nav.warp.destination)
        self.assertAlmostEqual(ship.nav.warp.destination.x, 300_000.0, delta=1.0)

    def test_nullifier_snapshot_ignores_probe_bubbles_but_not_hic_fields(self) -> None:
        immune_ship = _make_ship("ship-nullified-probe", [])
        _set_motion_state(immune_ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        immune_ship.profile = replace(
            immune_ship.profile,
            warp_speed_au_s=3.0,
            warp_capacitor_need=1.0,
            warp_bubble_immune=True,
        )
        immune_ship.nav.max_speed = immune_ship.profile.max_speed
        immune_ship.nav.velocity = Vector2(800.0, 0.0)
        immune_ship.nav.facing_deg = 0.0
        immune_ship.vital.cap = 1_000.0
        immune_ship.nav.warp.phase = "align"
        immune_ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        probe_world = WorldState(
            ships={immune_ship.ship_id: immune_ship},
            bubble_fields={
                "bubble-probe": _bubble_field(
                    "bubble-probe",
                    position=Vector2(290_000.0, 0.0),
                    radius_m=20_000.0,
                    interdiction_kind="probe",
                )
            },
        )

        MovementSystem().run(probe_world, 0.1)

        self.assertEqual(immune_ship.nav.warp.phase, "warp")
        self.assertIsNotNone(immune_ship.nav.warp.destination)
        self.assertAlmostEqual(immune_ship.nav.warp.destination.x, 300_000.0, delta=1.0)

        hic_ship = _make_ship("ship-nullified-hic", [])
        _set_motion_state(hic_ship, max_speed=1_000.0, mass=10_000_000.0, agility=0.5)
        hic_ship.profile = replace(
            hic_ship.profile,
            warp_speed_au_s=3.0,
            warp_capacitor_need=1.0,
            warp_bubble_immune=True,
        )
        hic_ship.nav.max_speed = hic_ship.profile.max_speed
        hic_ship.nav.velocity = Vector2(800.0, 0.0)
        hic_ship.nav.facing_deg = 0.0
        hic_ship.vital.cap = 1_000.0
        hic_ship.nav.warp.phase = "align"
        hic_ship.nav.warp.target_position = Vector2(300_000.0, 0.0)
        hic_world = WorldState(
            ships={hic_ship.ship_id: hic_ship},
            bubble_fields={
                "bubble-hic": _bubble_field(
                    "bubble-hic",
                    position=Vector2(290_000.0, 0.0),
                    radius_m=20_000.0,
                    kind="hic_warp_field",
                    interdiction_kind="hic",
                )
            },
        )

        MovementSystem().run(hic_world, 0.1)

        self.assertEqual(hic_ship.nav.warp.phase, "warp")
        self.assertIsNotNone(hic_ship.nav.warp.destination)
        self.assertAlmostEqual(hic_ship.nav.warp.destination.x, 270_000.0, delta=1.0)

    def test_web_probe_reduces_effective_speed_cap_inside_field(self) -> None:
        ship = _make_ship("ship-web-probe", [])
        ship.nav.max_speed = 1_000.0
        movement = MovementSystem()
        world = WorldState(
            ships={ship.ship_id: ship},
            bubble_fields={
                "bubble-web": _bubble_field(
                    "bubble-web",
                    position=Vector2(0.0, 0.0),
                    radius_m=15_000.0,
                    kind="webification_probe",
                    blocks_warp=False,
                    speed_factor_mult=0.8,
                )
            },
        )

        self.assertAlmostEqual(movement._effective_speed_cap(world, ship), 800.0, delta=1e-6)

    def test_warp_order_keeps_existing_attack_target_during_alignment(self) -> None:
        ship = _make_ship("ship-warp-order", [])
        target = _make_ship("ship-warp-order-target", [], team=Team.RED)
        ship.combat.current_target = target.ship_id
        ship.combat.last_attack_target = target.ship_id
        ship.order_queue.append(
            Order(
                kind="WARP",
                payload={"x": 300_000.0, "y": 0.0, "immediate": True},
                issue_time=0.0,
            )
        )
        world = WorldState(ships={ship.ship_id: ship, target.ship_id: target})

        ShipAgent(agent_id="agent:ship-warp-order", ship_id=ship.ship_id).think(world)

        self.assertEqual(ship.nav.warp.phase, "align")
        self.assertEqual(ship.combat.current_target, target.ship_id)
        self.assertEqual(ship.combat.last_attack_target, target.ship_id)

    def test_aligning_ship_keeps_projected_effects_until_actual_warp_entry(self) -> None:
        source = _make_ship(
            "ship-align-scram-source",
            [_warp_scrambler_module("scram-a", cycle_time=5.0, cap_need=1.0)],
            team=Team.BLUE,
        )
        target = _make_ship("ship-align-scram-target", [], team=Team.RED)
        target.nav.position = Vector2(5_000.0, 0.0)
        scram = source.runtime.modules[0]
        scram.state = ModuleState.ACTIVE
        source.combat.lock_targets.add(target.ship_id)
        source.combat.current_target = target.ship_id
        source.combat.last_attack_target = target.ship_id
        source.combat.projected_targets[scram.module_id] = target.ship_id
        source.combat.module_cycle_timers[scram.module_id] = 4.0
        source.nav.warp.phase = "align"
        source.nav.warp.target_position = Vector2(300_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        CombatSystem(PyfaBridge()).run(world, 1.0)

        self.assertEqual(source.nav.warp.phase, "align")
        self.assertEqual(scram.state, ModuleState.ACTIVE)
        self.assertEqual(source.combat.current_target, target.ship_id)
        self.assertEqual(source.combat.projected_targets.get(scram.module_id), target.ship_id)
        self.assertAlmostEqual(source.combat.module_cycle_timers.get(scram.module_id, 0.0), 4.0)

    def test_warping_ship_immediately_drops_projected_engagement_state(self) -> None:
        source = _make_ship(
            "ship-warp-scram-source",
            [_warp_scrambler_module("scram-a", cycle_time=5.0, cap_need=1.0)],
            team=Team.BLUE,
        )
        target = _make_ship("ship-warp-scram-target", [], team=Team.RED)
        target.nav.position = Vector2(5_000.0, 0.0)
        scram = source.runtime.modules[0]
        scram.state = ModuleState.ACTIVE
        source.combat.lock_targets.add(target.ship_id)
        source.combat.current_target = target.ship_id
        source.combat.last_attack_target = target.ship_id
        source.combat.projected_targets[scram.module_id] = target.ship_id
        source.combat.module_cycle_timers[scram.module_id] = 4.0
        source.nav.warp.phase = "warp"
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        CombatSystem(PyfaBridge()).run(world, 1.0)

        self.assertEqual(scram.state, ModuleState.ONLINE)
        self.assertEqual(source.combat.lock_targets, set())
        self.assertEqual(source.combat.projected_targets, {})
        self.assertIsNone(source.combat.current_target)
        self.assertIsNone(source.combat.last_attack_target)
        self.assertAlmostEqual(target.profile.warp_scramble_status, 0.0)

    def test_resolve_projected_miss_rebuilds_from_neutral_base_each_time(self) -> None:
        fit = _make_test_fit_descriptor("projected-cache-fit")
        runtime = _make_runtime(fit, [])
        runtime.diagnostics["pyfa_blueprint"] = {
            "ship_name": fit.ship_name,
            "fit_name": fit.fit_key,
            "modules": [{"module_id": "mod-1", "module_name": "ewar-a", "charge_name": "", "offline": False}],
        }
        baseline_profile = PyfaBridge().build_profile(fit)
        attached_batches: list[tuple[str, ...]] = []

        def make_snapshot(label: str) -> dict[str, object]:
            return {
                "fit_key": label,
                "blueprint": {
                    "ship_name": f"Source {label}",
                    "fit_name": label,
                    "modules": [{"module_id": "mod-1", "module_name": f"ewar-{label}", "charge_name": "", "offline": False}],
                },
                "state_by_module_id": {"mod-1": "ACTIVE"},
                "projection_range": 25_000.0,
                "pyfa_projection_key_mode": "exact_range",
            }

        snapshot_a = make_snapshot("source-a")
        snapshot_b = make_snapshot("source-b")
        snapshot_c = make_snapshot("source-c")

        class FakeBackend:
            fit_engine_ready = True

        class FakeFactory:
            backend_status = "ready"

            def _build_runtime_artifacts_from_pyfa_fit(
                self,
                parsed_fit,
                fit_ctx,
                fitted_modules,
                state_by_module_id=None,
                command_booster_snapshots=None,
            ):
                del parsed_fit, fitted_modules, state_by_module_id, command_booster_snapshots
                resolved_runtime = _make_runtime(fit, [])
                resolved_runtime.diagnostics["pyfa_blueprint"] = deepcopy(runtime.diagnostics["pyfa_blueprint"])
                resolved_runtime.diagnostics["attached_projected"] = tuple(getattr(fit_ctx, "applied_projected", []))
                return resolved_runtime, fit_ctx, replace(baseline_profile)

        def fake_copy_precalculated_neutral_base_fit(factory, parsed_fit):
            del factory, parsed_fit
            return CombatOptimizationTests._DummyPyfaFit(), []

        def fake_build_transient_fit_from_snapshot(factory, snapshot, next_fit_id, fallback_runtime, fit_prefix):
            del factory, fallback_runtime, fit_prefix
            source_fit = CombatOptimizationTests._DummyPyfaFit()
            source_fit.label = str(snapshot.get("fit_key", "") or "")
            source_fit.ID = next_fit_id
            return source_fit, next_fit_id + 1

        def fake_attach_projected_fit(target_fit, source_fit, amount=1, active=True, projection_range=None):
            del amount, active, projection_range
            target_fit.applied_projected.append(source_fit.label)

        with (
            patch.dict(fleet_setup_module.engine._PYFA_RUNTIME_RESOLVED_CACHE, {}, clear=True),
            patch.object(fleet_setup_module.engine, "_get_static_backend", return_value=FakeBackend()),
            patch.object(fleet_setup_module.engine, "RuntimeFromEftFactory", return_value=FakeFactory()),
            patch.object(
                fleet_setup_module.engine,
                "_copy_precalculated_neutral_base_fit",
                side_effect=fake_copy_precalculated_neutral_base_fit,
            ),
            patch.object(
                fleet_setup_module.engine,
                "_build_transient_fit_from_snapshot",
                side_effect=fake_build_transient_fit_from_snapshot,
            ),
            patch.object(
                fleet_setup_module.engine,
                "_attach_projected_fit",
                side_effect=fake_attach_projected_fit,
            ),
        ):
            first = fleet_setup_module.resolve_runtime_from_pyfa_runtime(
                runtime,
                projected_source_snapshots=[snapshot_a, snapshot_b],
            )
            second = fleet_setup_module.resolve_runtime_from_pyfa_runtime(
                runtime,
                projected_source_snapshots=[snapshot_a, snapshot_b, snapshot_c],
            )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        attached_batches.append(tuple(first[0].diagnostics.get("attached_projected", ())))
        attached_batches.append(tuple(second[0].diagnostics.get("attached_projected", ())))
        self.assertEqual(attached_batches, [("source-a", "source-b"), ("source-a", "source-b", "source-c")])
        self.assertEqual(first[0].diagnostics.get("pyfa_projected_target_fit_cache"), "single_pass")
        self.assertEqual(second[0].diagnostics.get("pyfa_projected_target_fit_cache"), "single_pass")

    def test_real_fit_initializes_active_modules_online_and_keeps_passive_speed_clean(self) -> None:
        parser = EftFitParser()
        factory = RuntimeFromEftFactory()
        parsed = parser.parse(_ROKH_PROPULSION_FIT)
        runtime, _fit = factory.build(parsed)
        profile = factory.build_profile(parsed)

        modules_by_id = {module.module_id: module for module in runtime.modules}

        self.assertEqual(modules_by_id["mod-1"].state, ModuleState.ONLINE)
        self.assertEqual(modules_by_id["mod-8"].state, ModuleState.ONLINE)
        self.assertEqual(modules_by_id["mod-9"].state, ModuleState.ONLINE)
        self.assertEqual(modules_by_id["mod-11"].state, ModuleState.ONLINE)
        self.assertEqual(modules_by_id["mod-12"].state, ModuleState.ONLINE)
        self.assertNotIn("speed", modules_by_id["mod-1"].effects[0].local_mult)
        self.assertNotIn("speed", modules_by_id["mod-5"].effects[0].local_mult)
        self.assertGreater(modules_by_id["mod-11"].effects[0].local_mult.get("speed", 1.0), 1.0)
        self.assertLess(profile.max_speed, 200.0)

    def test_real_fit_propulsion_activation_updates_speed_cap(self) -> None:
        ship = _make_pyfa_ship_from_fit_text(_ROKH_PROPULSION_FIT, ship_id="rokh-propulsion-real")
        world = WorldState(ships={ship.ship_id: ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        engine.register_ship(ship.ship_id)

        assert ship.runtime is not None
        propulsion_module = next(module for module in ship.runtime.modules if module.module_id == "mod-11")
        initial_max_speed = ship.profile.max_speed

        self.assertEqual(propulsion_module.state, ModuleState.ONLINE)
        self.assertAlmostEqual(ship.nav.max_speed, initial_max_speed)

        ship.nav.velocity = Vector2(100.0, 0.0)
        ship.nav.propulsion_command_active = True
        engine.step()

        self.assertEqual(propulsion_module.state, ModuleState.ACTIVE)
        self.assertGreater(ship.profile.max_speed, initial_max_speed * 5.0)
        self.assertAlmostEqual(ship.nav.max_speed, ship.profile.max_speed)
        self.assertIn(propulsion_module.module_id, ship.combat.module_cycle_timers)
        self.assertNotIn("mod-1", ship.combat.module_cycle_timers)

    def test_real_fit_weapons_keep_weapon_role_after_resolve_and_deal_damage(self) -> None:
        source = _make_pyfa_ship_from_fit_text(_FEROX_RAIL_FIT, ship_id="ferox-weapon-source", team=Team.BLUE)
        target = _make_ship("ferox-weapon-target", [], team=Team.RED)
        target.nav.position = Vector2(1_000.0, 0.0)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})
        combat = CombatSystem(PyfaBridge())
        world.squad_focus_queues[CombatSystem._focus_key(source.team, source.squad_id)] = [target.ship_id]
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=4), combat)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        initial_weapon_ids = [
            module.module_id
            for module in source.runtime.modules
            if combat._module_static_metadata(module).is_weapon
        ]
        self.assertGreaterEqual(len(initial_weapon_ids), 5)

        with patch.object(CombatSystem, "_sample_weapon_fire_delay", return_value=0.0), patch.object(
            CombatSystem, "_cached_lock_time", return_value=0.0
        ):
            engine.step()

        resolved_weapon_ids = [
            module.module_id
            for module in source.runtime.modules
            if combat._module_static_metadata(module).is_weapon
        ]
        active_weapon_ids = [
            module.module_id
            for module in source.runtime.modules
            if module.module_id in initial_weapon_ids and module.state == ModuleState.ACTIVE
        ]

        self.assertEqual(resolved_weapon_ids, initial_weapon_ids)
        self.assertTrue(active_weapon_ids)
        self.assertLess(target.vital.shield, target.vital.shield_max)

    def test_formula_projected_runtime_matches_pyfa_for_falloff_ewar(self) -> None:
        parser = EftFitParser()
        factory = RuntimeFromEftFactory()
        target_parsed = parser.parse(_FEROX_RAIL_FIT)
        source_parsed = parser.parse(_BLACKBIRD_DAMP_FIT)
        target_runtime, _target_fit = factory.build(target_parsed)
        source_runtime, _source_fit = factory.build(source_parsed)
        target_profile = factory.build_profile(target_parsed)
        combat = CombatSystem(PyfaBridge())

        for module in source_runtime.modules:
            module.state = ModuleState.ACTIVE
            self.assertFalse(combat._module_static_metadata(module).uses_pyfa_projected_profile)

        distance = combat._quantize_pyfa_projection_range(
            max(
                float(source_runtime.modules[0].effects[0].range_m) + 100.0,
                float(source_runtime.modules[0].effects[0].range_m) + float(source_runtime.modules[0].effects[0].falloff_m) * 0.75,
            )
        )
        impacts: list[ProjectedImpact] = []
        projected_snapshots: list[dict[str, object]] = []
        blueprint = deepcopy(source_runtime.diagnostics["pyfa_blueprint"])
        state_online = {str(module.module_id): "ONLINE" for module in source_runtime.modules}
        blueprint_modules_raw = blueprint.get("modules")
        blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
        blueprint_by_id = {
            str(raw.get("module_id", "") or ""): raw
            for raw in blueprint_modules
            if isinstance(raw, dict)
        }

        for module in source_runtime.modules:
            effect = next(effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED)
            strength = combat._projected_strength(effect, distance)
            impacts.append(
                ProjectedImpact(
                    source_ship_id=f"src-{module.module_id}",
                    target_ship_id="target",
                    effect=effect,
                    strength=strength,
                )
            )
            state_by_module_id = dict(state_online)
            state_by_module_id[str(module.module_id)] = "ACTIVE"
            projected_snapshots.append(
                {
                    "fit_key": f"{source_runtime.fit_key}:{module.module_id}",
                    "blueprint": deepcopy(blueprint),
                    "state_by_module_id": state_by_module_id,
                    "command_booster_snapshots": [],
                    "pyfa_projection_key_mode": "exact_range",
                    "pyfa_projection_range": distance,
                    "projection_range": distance,
                    "pyfa_projection_module_signature": combat._projected_module_runtime_signature(
                        module,
                        blueprint_by_id.get(str(module.module_id)),
                        "ACTIVE",
                        active_effect_indices={0},
                    ),
                }
            )

        direct_profile = combat._apply_runtime_projected_impacts(replace(target_profile), impacts, runtime=target_runtime)
        resolved = resolve_runtime_from_pyfa_runtime(target_runtime, [], projected_snapshots)

        self.assertIsNotNone(resolved)
        assert resolved is not None
        _resolved_runtime, pyfa_profile = resolved
        self.assertAlmostEqual(direct_profile.max_target_range, pyfa_profile.max_target_range, places=5)
        self.assertAlmostEqual(direct_profile.scan_resolution, pyfa_profile.scan_resolution, places=5)

    def test_formula_target_painter_matches_pyfa_across_falloff_ranges(self) -> None:
        parser = EftFitParser()
        factory = RuntimeFromEftFactory()
        target_parsed = parser.parse(_FEROX_RAIL_FIT)
        source_parsed = parser.parse(_BELLICOSE_PAINTER_FIT)
        target_runtime, _target_fit = factory.build(target_parsed)
        source_runtime, _source_fit = factory.build(source_parsed)
        target_profile = factory.build_profile(target_parsed)
        combat = CombatSystem(PyfaBridge())

        projected_modules = []
        blueprint = deepcopy(source_runtime.diagnostics["pyfa_blueprint"])
        blueprint_modules_raw = blueprint.get("modules")
        blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
        blueprint_by_id = {
            str(raw.get("module_id", "") or ""): raw
            for raw in blueprint_modules
            if isinstance(raw, dict)
        }
        state_online = {str(module.module_id): "ONLINE" for module in source_runtime.modules}

        for module in source_runtime.modules:
            if not any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects):
                continue
            module.state = ModuleState.ACTIVE
            self.assertFalse(combat._module_static_metadata(module).uses_pyfa_projected_profile)
            projected_modules.append(module)

        for distance in (81_000.0, 120_000.0, 216_000.0):
            impacts: list[ProjectedImpact] = []
            projected_snapshots: list[dict[str, object]] = []

            for module in projected_modules:
                effect = next(effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED)
                strength = combat._projected_strength(effect, distance)
                impacts.append(
                    ProjectedImpact(
                        source_ship_id=f"src-{module.module_id}",
                        target_ship_id="target",
                        effect=effect,
                        strength=strength,
                    )
                )
                state_by_module_id = dict(state_online)
                state_by_module_id[str(module.module_id)] = "ACTIVE"
                projected_snapshots.append(
                    {
                        "fit_key": f"{source_runtime.fit_key}:{module.module_id}",
                        "blueprint": deepcopy(blueprint),
                        "state_by_module_id": state_by_module_id,
                        "command_booster_snapshots": [],
                        "pyfa_projection_key_mode": "exact_range",
                        "pyfa_projection_range": distance,
                        "projection_range": distance,
                        "pyfa_projection_module_signature": combat._projected_module_runtime_signature(
                            module,
                            blueprint_by_id.get(str(module.module_id)),
                            "ACTIVE",
                            active_effect_indices={0},
                        ),
                    }
                )

            direct_profile = combat._apply_runtime_projected_impacts(replace(target_profile), impacts, runtime=target_runtime)
            resolved = resolve_runtime_from_pyfa_runtime(target_runtime, [], projected_snapshots)

            self.assertIsNotNone(resolved)
            assert resolved is not None
            _resolved_runtime, pyfa_profile = resolved
            self.assertAlmostEqual(direct_profile.sig_radius, pyfa_profile.sig_radius, places=5)

    def test_formula_remote_tracking_computer_matches_pyfa_across_falloff_ranges(self) -> None:
        parser = EftFitParser()
        factory = RuntimeFromEftFactory()
        target_parsed = parser.parse(_FEROX_RAIL_FIT)
        source_parsed = parser.parse(_SCYTHE_REMOTE_TRACKING_FIT)
        target_runtime, _target_fit = factory.build(target_parsed)
        source_runtime, _source_fit = factory.build(source_parsed)
        target_profile = factory.build_profile(target_parsed)
        combat = CombatSystem(PyfaBridge())

        projected_modules = []
        blueprint = deepcopy(source_runtime.diagnostics["pyfa_blueprint"])
        blueprint_modules_raw = blueprint.get("modules")
        blueprint_modules = blueprint_modules_raw if isinstance(blueprint_modules_raw, list) else []
        blueprint_by_id = {
            str(raw.get("module_id", "") or ""): raw
            for raw in blueprint_modules
            if isinstance(raw, dict)
        }
        state_online = {str(module.module_id): "ONLINE" for module in source_runtime.modules}

        for module in source_runtime.modules:
            if not any(effect.effect_class == EffectClass.PROJECTED for effect in module.effects):
                continue
            module.state = ModuleState.ACTIVE
            self.assertFalse(combat._module_static_metadata(module).uses_pyfa_projected_profile)
            projected_modules.append(module)

        for distance in (30_000.0, 60_000.0, 120_000.0, 180_000.0):
            impacts: list[ProjectedImpact] = []
            projected_snapshots: list[dict[str, object]] = []

            for module in projected_modules:
                effect = next(effect for effect in module.effects if effect.effect_class == EffectClass.PROJECTED)
                strength = combat._projected_strength(effect, distance)
                impacts.append(
                    ProjectedImpact(
                        source_ship_id=f"src-{module.module_id}",
                        target_ship_id="target",
                        effect=effect,
                        strength=strength,
                    )
                )
                state_by_module_id = dict(state_online)
                state_by_module_id[str(module.module_id)] = "ACTIVE"
                projected_snapshots.append(
                    {
                        "fit_key": f"{source_runtime.fit_key}:{module.module_id}",
                        "blueprint": deepcopy(blueprint),
                        "state_by_module_id": state_by_module_id,
                        "command_booster_snapshots": [],
                        "pyfa_projection_key_mode": "exact_range",
                        "pyfa_projection_range": distance,
                        "projection_range": distance,
                        "pyfa_projection_module_signature": combat._projected_module_runtime_signature(
                            module,
                            blueprint_by_id.get(str(module.module_id)),
                            "ACTIVE",
                            active_effect_indices={0},
                        ),
                    }
                )

            direct_profile = combat._apply_runtime_projected_impacts(replace(target_profile), impacts, runtime=target_runtime)
            resolved = resolve_runtime_from_pyfa_runtime(target_runtime, [], projected_snapshots)

            self.assertIsNotNone(resolved)
            assert resolved is not None
            _resolved_runtime, pyfa_profile = resolved
            self.assertAlmostEqual(direct_profile.optimal, pyfa_profile.optimal, places=5)
            self.assertAlmostEqual(direct_profile.falloff, pyfa_profile.falloff, places=5)
            self.assertAlmostEqual(direct_profile.tracking, pyfa_profile.tracking, places=5)

    def test_remote_projected_constant_range_change_skips_target_pyfa(self) -> None:
        source = _make_ship(
            "ship-constant-source",
            [_remote_sensor_damp_module("damp-constant", cycle_time=10.0, cap_need=5.0)],
            fit_key="constant-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-constant-target",
            [],
            fit_key="constant-target-fit",
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        recorded_steps = self._run_world_steps_and_capture_resolves(
            world,
            step_dts=[1.0, 1.0],
            before_step_callbacks=[None, lambda current_world: current_world.ships[target.ship_id].nav.position.__setattr__("x", 80_000.0)],
        )

        self.assertEqual(len(recorded_steps[0]), 2)
        self.assertEqual(recorded_steps[1], [])

    def test_remote_projected_formula_range_change_only_resolves_when_signature_changes(self) -> None:
        source = _make_ship(
            "ship-formula-source",
            [_remote_sensor_damp_module("damp-formula", cycle_time=10.0, cap_need=5.0, range_m=40_000.0, falloff_m=40_000.0)],
            fit_key="formula-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-formula-target",
            [],
            fit_key="formula-target-fit",
            team=Team.RED,
        )
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        def warmup(_world: WorldState) -> None:
            return None

        def activate(current_world: WorldState) -> None:
            current_world.ships[source.ship_id].combat.lock_targets.add(target.ship_id)

        def move_farther(current_world: WorldState) -> None:
            current_world.ships[target.ship_id].nav.position = Vector2(60_000.0, 0.0)

        recorded_steps = self._run_world_steps_and_capture_resolves(
            world,
            step_dts=[1.0, 1.0, 10.0],
            before_step_callbacks=[warmup, activate, move_farther],
        )

        self.assertEqual(len(recorded_steps[0]), 2)
        self.assertEqual(recorded_steps[1], [])
        self.assertEqual(recorded_steps[2], [])

    def test_remote_projected_formula_source_set_change_triggers_exact_resolve(self) -> None:
        source_a = _make_ship(
            "ship-formula-source-a",
            [_remote_sensor_damp_module("damp-a", cycle_time=10.0, cap_need=5.0, range_m=40_000.0, falloff_m=40_000.0)],
            fit_key="formula-source-a-fit",
            team=Team.BLUE,
        )
        source_b = _make_ship(
            "ship-formula-source-b",
            [_remote_sensor_damp_module("damp-b", cycle_time=10.0, cap_need=5.0, range_m=40_000.0, falloff_m=40_000.0)],
            fit_key="formula-source-b-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-formula-target-set",
            [],
            fit_key="formula-target-set-fit",
            team=Team.RED,
        )
        world = WorldState(
            ships={
                source_a.ship_id: source_a,
                source_b.ship_id: source_b,
                target.ship_id: target,
            }
        )

        def activate_first(current_world: WorldState) -> None:
            current_world.ships[source_a.ship_id].combat.lock_targets.add(target.ship_id)

        def activate_second(current_world: WorldState) -> None:
            current_world.ships[source_b.ship_id].combat.lock_targets.add(target.ship_id)

        recorded_steps = self._run_world_steps_and_capture_resolves(
            world,
            step_dts=[1.0, 1.0, 1.0],
            before_step_callbacks=[None, activate_first, activate_second],
        )

        self.assertEqual(len(recorded_steps[0]), 3)
        self.assertEqual(recorded_steps[1], [])
        self.assertEqual(recorded_steps[2], [])

    def test_ecm_activation_skips_target_pyfa_profile_rebuild(self) -> None:
        source = _make_ship(
            "ship-ecm-source",
            [_ecm_module("ecm-a", cycle_time=10.0, cap_need=5.0)],
            fit_key="ecm-source-fit",
            team=Team.BLUE,
        )
        target = _make_ship(
            "ship-ecm-target",
            [],
            fit_key="ecm-target-fit",
            team=Team.RED,
        )
        source.combat.lock_targets.add(target.ship_id)
        world = WorldState(ships={source.ship_id: source, target.ship_id: target})

        first_step = self._run_world_steps_and_capture_resolves(world, step_dts=[1.0])[0]

        self.assertEqual(len(first_step), 2)
        target_call = next(call for call in first_step if call["fit_key"] == "ecm-target-fit")
        self.assertEqual(target_call["projected_fit_keys"], ())
        self.assertIn(source.ship_id, target.combat.ecm_jam_sources)


if __name__ == "__main__":
    unittest.main()


