from __future__ import annotations

from dataclasses import dataclass
import random

from .fit_runtime import (
    EffectClass,
    FitRuntime,
    HullProfile,
    ModuleEffect,
    ModuleRuntime,
    ModuleState,
    RuntimeStatEngine,
    SkillProfile,
)
from .math2d import Vector2
from .models import (
    Beacon,
    CombatState,
    FitDescriptor,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipEntity,
    Team,
    VitalState,
)
from .world import WorldState


QUALITY_PRESETS = {
    QualityLevel.ELITE: QualityState(QualityLevel.ELITE, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.REGULAR: QualityState(QualityLevel.REGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.IRREGULAR: QualityState(QualityLevel.IRREGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
}


@dataclass(slots=True)
class FleetTemplate:
    squad_id: str
    team: Team
    quality: QualityLevel
    count: int
    anchor: Vector2
    fit: FitDescriptor


def _mk_ship_id(team: Team, squad: str, idx: int) -> str:
    return f"{team.value}-{squad}-{idx:03d}"


def spawn_fleet(world: WorldState, stat_engine: RuntimeStatEngine, fleet: FleetTemplate, runtime: FitRuntime) -> list[str]:
    quality = QUALITY_PRESETS[fleet.quality]
    created: list[str] = []
    for idx in range(1, fleet.count + 1):
        jitter = Vector2(
            random.uniform(-quality.formation_jitter, quality.formation_jitter),
            random.uniform(-quality.formation_jitter, quality.formation_jitter),
        )
        ship_id = _mk_ship_id(fleet.team, fleet.squad_id, idx)
        profile = stat_engine.compute_base_profile(runtime)
        ship = ShipEntity(
            ship_id=ship_id,
            team=fleet.team,
            squad_id=fleet.squad_id,
            fit=fleet.fit,
            profile=profile,
            runtime=runtime,
            nav=NavigationState(position=fleet.anchor + jitter, velocity=Vector2(0.0, 0.0), facing_deg=0.0, max_speed=profile.max_speed),
            combat=CombatState(),
            vital=VitalState(
                shield=20_000,
                armor=12_000,
                structure=10_000,
                shield_max=20_000,
                armor_max=12_000,
                structure_max=10_000,
                cap=profile.max_cap,
                cap_max=profile.max_cap,
                alive=True,
            ),
            quality=QualityState(
                level=quality.level,
                reaction_delay=quality.reaction_delay,
                ignore_order_probability=quality.ignore_order_probability,
                formation_jitter=quality.formation_jitter,
            ),
        )
        world.ships[ship_id] = ship
        created.append(ship_id)
    return created


def _skills_combat() -> SkillProfile:
    return SkillProfile(
        levels={
            "Gunnery": 5,
            "Motion Prediction": 5,
            "Sharpshooter": 5,
            "Trajectory Analysis": 5,
            "Long Range Targeting": 5,
            "Signature Analysis": 5,
            "Navigation": 5,
            "Capacitor Management": 5,
            "Capacitor Systems Operation": 5,
        }
    )


def _ferox_runtime() -> FitRuntime:
    hull = HullProfile(
        ship_name="Ferox",
        role="DPS",
        base_dps=630,
        volley=3200,
        optimal=72_000,
        falloff=18_000,
        tracking=0.06,
        sig_radius=220,
        scan_resolution=250,
        max_target_range=110_000,
        max_speed=1450,
        cap_max=4200,
        cap_recharge_time=520,
        shield_hp=6000,
        armor_hp=5000,
        structure_hp=5000,
        rep_amount=0,
        rep_cycle=5,
    )
    return FitRuntime(
        fit_key="ferox-rail-v2",
        hull=hull,
        skills=_skills_combat(),
        modules=[
            ModuleRuntime(
                module_id="magstab-1",
                group="Damage",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="magstab-dps",
                        effect_class=EffectClass.LOCAL,
                        state_required=ModuleState.ONLINE,
                        local_mult={"dps": 1.12, "tracking": 1.08},
                    )
                ],
            ),
            ModuleRuntime(
                module_id="te-1",
                group="Range",
                state=ModuleState.ONLINE,
                effects=[
                    ModuleEffect(
                        name="tracking-enhancer",
                        effect_class=EffectClass.LOCAL,
                        state_required=ModuleState.ONLINE,
                        local_mult={"optimal": 1.15, "falloff": 1.15, "tracking": 1.08},
                    )
                ],
            ),
        ],
    )


def _scythe_runtime() -> FitRuntime:
    hull = HullProfile(
        ship_name="Scythe",
        role="LOGI",
        base_dps=80,
        volley=300,
        optimal=45_000,
        falloff=12_000,
        tracking=0.08,
        sig_radius=90,
        scan_resolution=420,
        max_target_range=95_000,
        max_speed=2100,
        cap_max=3600,
        cap_recharge_time=380,
        shield_hp=4500,
        armor_hp=3200,
        structure_hp=3200,
        rep_amount=900,
        rep_cycle=5.0,
    )
    return FitRuntime(
        fit_key="scythe-logi-v2",
        hull=hull,
        skills=_skills_combat(),
        modules=[
            ModuleRuntime(
                module_id="ab-1",
                group="Propulsion",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="afterburner",
                        effect_class=EffectClass.LOCAL,
                        state_required=ModuleState.ACTIVE,
                        cap_need=80,
                        cycle_time=10,
                        local_mult={"speed": 1.65, "sig": 1.05},
                    )
                ],
            )
        ],
    )


def _ewar_runtime() -> FitRuntime:
    hull = HullProfile(
        ship_name="Blackbird",
        role="EWAR",
        base_dps=120,
        volley=600,
        optimal=40_000,
        falloff=20_000,
        tracking=0.05,
        sig_radius=140,
        scan_resolution=350,
        max_target_range=120_000,
        max_speed=1650,
        cap_max=3900,
        cap_recharge_time=430,
        shield_hp=4200,
        armor_hp=3800,
        structure_hp=3800,
        rep_amount=0,
        rep_cycle=5,
    )
    return FitRuntime(
        fit_key="blackbird-ewar-v1",
        hull=hull,
        skills=_skills_combat(),
        modules=[
            ModuleRuntime(
                module_id="web-1",
                group="Stasis Web",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="stasis-web",
                        effect_class=EffectClass.PROJECTED,
                        state_required=ModuleState.ACTIVE,
                        range_m=13_000,
                        cap_need=45,
                        cycle_time=5,
                        projected_mult={"speed": 0.45},
                    )
                ],
            ),
            ModuleRuntime(
                module_id="damp-1",
                group="Sensor Dampener",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="sensor-damp",
                        effect_class=EffectClass.PROJECTED,
                        state_required=ModuleState.ACTIVE,
                        range_m=60_000,
                        falloff_m=20_000,
                        cap_need=55,
                        cycle_time=8,
                        projected_mult={"scan": 0.72, "range": 0.76},
                    )
                ],
            ),
            ModuleRuntime(
                module_id="paint-1",
                group="Target Painter",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="target-painter",
                        effect_class=EffectClass.PROJECTED,
                        state_required=ModuleState.ACTIVE,
                        range_m=55_000,
                        cap_need=50,
                        cycle_time=6,
                        projected_mult={"sig": 1.28},
                    )
                ],
            ),
            ModuleRuntime(
                module_id="neut-1",
                group="Energy Neutralizer",
                state=ModuleState.ACTIVE,
                effects=[
                    ModuleEffect(
                        name="neut",
                        effect_class=EffectClass.PROJECTED,
                        state_required=ModuleState.ACTIVE,
                        range_m=25_000,
                        cap_need=140,
                        cycle_time=12,
                        projected_add={"cap_max": -220},
                    )
                ],
            ),
        ],
    )


def build_default_world(pyfa=None) -> WorldState:
    del pyfa
    world = WorldState()
    stat_engine = RuntimeStatEngine()
    world.beacons["gate-1"] = Beacon(
        beacon_id="gate-1",
        position=Vector2(-80_000, 0),
        radius=2000,
        interaction_range=2500,
        kind="STARGATE",
    )
    world.beacons["gate-2"] = Beacon(
        beacon_id="gate-2",
        position=Vector2(80_000, 0),
        radius=2000,
        interaction_range=2500,
        kind="STARGATE",
    )

    rail_dps = FitDescriptor(
        fit_key="ferox-rail-v1",
        ship_name="Ferox",
        role="DPS",
        base_dps=630,
        volley=3200,
        optimal_range=72_000,
        falloff=18_000,
        tracking=0.06,
        signature_radius=220,
        scan_resolution=250,
        max_target_range=110_000,
        max_speed=1450,
        max_cap=4200,
        cap_recharge_time=520,
    )
    logi = FitDescriptor(
        fit_key="scythe-logi-v1",
        ship_name="Scythe",
        role="LOGI",
        base_dps=80,
        volley=300,
        optimal_range=45_000,
        falloff=12_000,
        tracking=0.08,
        signature_radius=90,
        scan_resolution=420,
        max_target_range=95_000,
        max_speed=2100,
        max_cap=3600,
        cap_recharge_time=380,
        rep_amount=900,
        rep_cycle=5.0,
    )
    ewar = FitDescriptor(
        fit_key="blackbird-ewar-v1",
        ship_name="Blackbird",
        role="EWAR",
        base_dps=120,
        volley=600,
        optimal_range=40_000,
        falloff=20_000,
        tracking=0.05,
        signature_radius=140,
        scan_resolution=350,
        max_target_range=120_000,
        max_speed=1650,
        max_cap=3900,
        cap_recharge_time=430,
    )

    ferox_runtime = _ferox_runtime()
    scythe_runtime = _scythe_runtime()
    ewar_runtime = _ewar_runtime()

    spawn_fleet(
        world,
        stat_engine,
        FleetTemplate(
            squad_id="BLUE-ALPHA",
            team=Team.BLUE,
            quality=QualityLevel.REGULAR,
            count=20,
            anchor=Vector2(-45_000, -8_000),
            fit=rail_dps,
        ),
        ferox_runtime,
    )
    spawn_fleet(
        world,
        stat_engine,
        FleetTemplate(
            squad_id="BLUE-LOGI",
            team=Team.BLUE,
            quality=QualityLevel.ELITE,
            count=8,
            anchor=Vector2(-48_000, 6_000),
            fit=logi,
        ),
        scythe_runtime,
    )
    spawn_fleet(
        world,
        stat_engine,
        FleetTemplate(
            squad_id="BLUE-EWAR",
            team=Team.BLUE,
            quality=QualityLevel.REGULAR,
            count=4,
            anchor=Vector2(-52_000, 2_000),
            fit=ewar,
        ),
        ewar_runtime,
    )
    spawn_fleet(
        world,
        stat_engine,
        FleetTemplate(
            squad_id="RED-ALPHA",
            team=Team.RED,
            quality=QualityLevel.IRREGULAR,
            count=28,
            anchor=Vector2(42_000, 0),
            fit=rail_dps,
        ),
        ferox_runtime,
    )
    spawn_fleet(
        world,
        stat_engine,
        FleetTemplate(
            squad_id="RED-EWAR",
            team=Team.RED,
            quality=QualityLevel.REGULAR,
            count=6,
            anchor=Vector2(38_000, 6_000),
            fit=ewar,
        ),
        ewar_runtime,
    )
    return world
