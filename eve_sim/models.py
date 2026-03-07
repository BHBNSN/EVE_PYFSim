from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .math2d import Vector2

if TYPE_CHECKING:
    from .fit_runtime import FitRuntime


class Team(str, Enum):
    BLUE = "BLUE"
    RED = "RED"


class QualityLevel(str, Enum):
    ELITE = "ELITE"
    REGULAR = "REGULAR"
    IRREGULAR = "IRREGULAR"


@dataclass(slots=True)
class FitDescriptor:
    fit_key: str
    ship_name: str
    role: str
    base_dps: float
    volley: float
    optimal_range: float
    falloff: float
    tracking: float
    missile_explosion_radius: float = 0.0
    missile_explosion_velocity: float = 0.0
    signature_radius: float = 120.0
    scan_resolution: float = 300.0
    max_target_range: float = 120_000.0
    sensor_strength_gravimetric: float = 0.0
    sensor_strength_ladar: float = 0.0
    sensor_strength_magnetometric: float = 0.0
    sensor_strength_radar: float = 0.0
    max_speed: float = 1800.0
    max_cap: float = 4000.0
    cap_recharge_time: float = 450.0
    shield_hp: float = 5000.0
    armor_hp: float = 4000.0
    structure_hp: float = 4000.0
    rep_amount: float = 0.0
    rep_cycle: float = 5.0


@dataclass(slots=True)
class ShipProfile:
    dps: float
    volley: float
    optimal: float
    falloff: float
    tracking: float
    sig_radius: float
    scan_resolution: float
    max_target_range: float
    max_speed: float
    max_cap: float
    cap_recharge_time: float
    shield_hp: float
    armor_hp: float
    structure_hp: float
    rep_amount: float
    rep_cycle: float
    weapon_system: str = "turret"
    optimal_sig: float = 40.0
    turret_dps: float = 0.0
    missile_dps: float = 0.0
    turret_cycle: float = 0.0
    missile_cycle: float = 0.0
    damage_em: float = 0.0
    damage_thermal: float = 0.0
    damage_kinetic: float = 0.0
    damage_explosive: float = 0.0
    turret_em_dps: float = 0.0
    turret_thermal_dps: float = 0.0
    turret_kinetic_dps: float = 0.0
    turret_explosive_dps: float = 0.0
    missile_em_dps: float = 0.0
    missile_thermal_dps: float = 0.0
    missile_kinetic_dps: float = 0.0
    missile_explosive_dps: float = 0.0
    missile_explosion_radius: float = 0.0
    missile_explosion_velocity: float = 0.0
    missile_max_range: float = 0.0
    missile_damage_reduction_factor: float = 0.5
    sensor_strength_gravimetric: float = 0.0
    sensor_strength_ladar: float = 0.0
    sensor_strength_magnetometric: float = 0.0
    sensor_strength_radar: float = 0.0
    shield_resonance_em: float = 1.0
    shield_resonance_thermal: float = 1.0
    shield_resonance_kinetic: float = 1.0
    shield_resonance_explosive: float = 1.0
    armor_resonance_em: float = 1.0
    armor_resonance_thermal: float = 1.0
    armor_resonance_kinetic: float = 1.0
    armor_resonance_explosive: float = 1.0
    structure_resonance_em: float = 1.0
    structure_resonance_thermal: float = 1.0
    structure_resonance_kinetic: float = 1.0
    structure_resonance_explosive: float = 1.0


@dataclass(slots=True)
class NavigationState:
    position: Vector2
    velocity: Vector2
    facing_deg: float
    max_speed: float
    radius: float = 60.0
    command_target: Vector2 | None = None
    propulsion_command_active: bool = False


@dataclass(slots=True)
class CombatState:
    lock_targets: set[str] = field(default_factory=set)
    current_target: str | None = None
    last_attack_target: str | None = None
    lock_timers: dict[str, float] = field(default_factory=dict)
    module_ammo_reload_timers: dict[str, float] = field(default_factory=dict)
    module_pending_ammo_reload_timers: dict[str, float] = field(default_factory=dict)
    fire_delay_timers: dict[str, float] = field(default_factory=dict)
    projected_targets: dict[str, str] = field(default_factory=dict)
    ecm_jam_sources: dict[str, float] = field(default_factory=dict)
    ecm_last_attempt_target: str | None = None
    ecm_last_attempt_module: str | None = None
    ecm_last_attempt_success: bool | None = None
    ecm_last_attempt_chance: float = 0.0
    ecm_last_attempt_at: float = -1e9
    ecm_last_attempt_target_by_module: dict[str, str] = field(default_factory=dict)
    ecm_last_attempt_success_by_module: dict[str, bool] = field(default_factory=dict)
    ecm_last_attempt_at_by_module: dict[str, float] = field(default_factory=dict)
    last_damaged_at: float = -1e9
    module_cycle_timers: dict[str, float] = field(default_factory=dict)
    module_reactivation_timers: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class VitalState:
    shield: float
    armor: float
    structure: float
    shield_max: float
    armor_max: float
    structure_max: float
    cap: float
    cap_max: float
    alive: bool = True


@dataclass(slots=True)
class QualityState:
    level: QualityLevel
    reaction_delay: float
    ignore_order_probability: float
    formation_jitter: float


@dataclass(slots=True)
class Order:
    kind: str
    payload: dict
    issue_time: float


@dataclass(slots=True)
class ShipEntity:
    ship_id: str
    team: Team
    squad_id: str
    fit: FitDescriptor
    profile: ShipProfile
    nav: NavigationState
    combat: CombatState
    vital: VitalState
    quality: QualityState
    runtime: "FitRuntime | None" = None
    order_queue: list[Order] = field(default_factory=list)
    perception: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Beacon:
    beacon_id: str
    position: Vector2
    radius: float
    interaction_range: float
    kind: str


@dataclass(slots=True)
class FleetIntent:
    squad_id: str
    target_position: Vector2 | None = None
    focus_target: str | None = None
    propulsion_active: bool | None = None
