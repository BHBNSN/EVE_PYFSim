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
    energy_warfare_resistance: float = 1.0
    mass: float = 0.0
    agility: float = 0.0
    warp_speed_au_s: float = 0.0
    warp_capacitor_need: float = 0.0
    max_warp_distance_au: float = 0.0
    disallow_assistance: bool = False
    warp_bubble_immune: bool = False
    is_shuttle: bool = False


@dataclass(frozen=True, slots=True)
class DamageProfile:
    em: float = 0.0
    thermal: float = 0.0
    kinetic: float = 0.0
    explosive: float = 0.0

    @property
    def total(self) -> float:
        return max(0.0, self.em) + max(0.0, self.thermal) + max(0.0, self.kinetic) + max(0.0, self.explosive)


@dataclass(frozen=True, slots=True)
class DeployableEwarProfile:
    cycle_time_s: float = 5.0
    optimal_range_m: float = 0.0
    falloff_m: float = 0.0
    duration_s: float = 0.0
    speed_factor_mult: float = 1.0
    signature_radius_bonus_pct: float = 0.0
    scan_resolution_bonus_pct: float = 0.0
    max_target_range_bonus_pct: float = 0.0
    tracking_bonus_pct: float = 0.0
    optimal_bonus_pct: float = 0.0
    falloff_bonus_pct: float = 0.0
    capacitor_neutralized: float = 0.0
    warp_disrupt_strength: float = 0.0
    ecm_gravimetric: float = 0.0
    ecm_ladar: float = 0.0
    ecm_magnetometric: float = 0.0
    ecm_radar: float = 0.0

    @property
    def has_effect(self) -> bool:
        return (
            self.speed_factor_mult < 0.999
            or abs(self.signature_radius_bonus_pct) > 1e-9
            or abs(self.scan_resolution_bonus_pct) > 1e-9
            or abs(self.max_target_range_bonus_pct) > 1e-9
            or abs(self.tracking_bonus_pct) > 1e-9
            or abs(self.optimal_bonus_pct) > 1e-9
            or abs(self.falloff_bonus_pct) > 1e-9
            or self.capacitor_neutralized > 0.0
            or self.warp_disrupt_strength > 0.0
            or max(self.ecm_gravimetric, self.ecm_ladar, self.ecm_magnetometric, self.ecm_radar) > 0.0
        )


@dataclass(frozen=True, slots=True)
class DroneBayEntry:
    type_name: str
    quantity: int
    group_name: str
    bandwidth_mbit: float
    volume_m3: float
    max_velocity: float
    orbit_range_m: float
    control_range_m: float
    cycle_time_s: float
    optimal_range_m: float
    falloff_m: float
    tracking: float
    damage: DamageProfile
    shield_hp: float
    armor_hp: float
    structure_hp: float
    signature_radius: float
    scan_resolution: float = 200.0
    sensor_strength_gravimetric: float = 0.0
    sensor_strength_ladar: float = 0.0
    sensor_strength_magnetometric: float = 0.0
    sensor_strength_radar: float = 0.0
    is_sentry: bool = False
    ewar: DeployableEwarProfile = field(default_factory=DeployableEwarProfile)


@dataclass(frozen=True, slots=True)
class FighterAbilityProfile:
    ability_id: str
    name: str
    effect_name: str
    kind: str
    cycle_time_s: float
    optimal_range_m: float
    falloff_m: float
    tracking: float
    damage: DamageProfile
    explosion_radius: float = 0.0
    explosion_velocity: float = 0.0
    damage_reduction_factor: float = 0.5
    ammo_capacity: int = 0
    reload_time_s: float = 0.0
    speed_bonus_pct: float = 0.0
    duration_s: float = 0.0
    cooldown_s: float = 0.0
    ewar: DeployableEwarProfile = field(default_factory=DeployableEwarProfile)

    @property
    def has_damage(self) -> bool:
        return self.damage.total > 0.0


@dataclass(frozen=True, slots=True)
class FighterBayEntry:
    type_name: str
    quantity: int
    group_name: str
    slot_kind: str
    squadron_size: int
    max_velocity: float
    orbit_range_m: float
    shield_hp: float
    armor_hp: float
    structure_hp: float
    signature_radius: float
    scan_resolution: float
    sensor_strength_gravimetric: float = 0.0
    sensor_strength_ladar: float = 0.0
    sensor_strength_magnetometric: float = 0.0
    sensor_strength_radar: float = 0.0
    warp_speed_au_s: float = 0.0
    abilities: tuple[FighterAbilityProfile, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class DeployableControlState:
    drone_bandwidth_mbit: float = 0.0
    max_active_drones: int = 0
    fighter_tubes: int = 0
    fighter_light_slots: int = 0
    fighter_support_slots: int = 0
    fighter_heavy_slots: int = 0
    drone_attack_target_id: str | None = None
    drone_attack_command_at: float = 0.0
    pending_drone_attack_target_id: str | None = None
    pending_drone_attack_command_at: float = 0.0
    fighter_attack_target_id: str | None = None
    fighter_attack_command_at: float = 0.0
    pending_fighter_attack_target_id: str | None = None
    pending_fighter_attack_command_at: float = 0.0


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
    max_locked_targets: int = 0
    scan_strength: float = 0.0
    ecm_jam_chance: float = 0.0
    warp_scramble_status: float = 0.0
    warp_stability: float = 0.0
    energy_warfare_resistance: float = 1.0
    mass: float = 0.0
    agility: float = 0.0
    warp_speed_au_s: float = 0.0
    warp_capacitor_need: float = 0.0
    max_warp_distance_au: float = 0.0
    disallow_assistance: bool = False
    warp_bubble_immune: bool = False
    is_shuttle: bool = False


@dataclass(frozen=True, slots=True)
class WarpInterdictionSnapshot:
    field_id: str
    kind: str
    interdiction_kind: str
    position: Vector2
    radius_m: float
    blocks_warp: bool


@dataclass(slots=True)
class WarpState:
    phase: str = "idle"
    target_position: Vector2 | None = None
    target_ship_id: str | None = None
    target_beacon_id: str | None = None
    align_elapsed: float = 0.0
    align_timeout: float = 180.0
    origin: Vector2 | None = None
    destination: Vector2 | None = None
    warp_distance_m: float = 0.0
    warp_duration: float = 0.0
    warp_elapsed: float = 0.0
    capacitor_cost: float = 0.0
    bubble_immune_snapshot: bool = False
    interdiction_snapshots_captured: bool = False
    interdiction_snapshots: tuple[WarpInterdictionSnapshot, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class GateTransitState:
    target_structure_id: str | None = None
    activation_range_m: float = 2_500.0


@dataclass(slots=True)
class CloakState:
    active: bool = False
    expires_at: float = 0.0
    source: str = ""


@dataclass(slots=True)
class NavigationState:
    position: Vector2
    velocity: Vector2
    facing_deg: float
    max_speed: float
    system_id: str = ""
    radius: float = 60.0
    command_target: Vector2 | None = None
    command_mode: str = "move"
    command_target_ship_id: str | None = None
    command_target_structure_id: str | None = None
    command_range_m: float = 0.0
    command_orbit_clockwise: bool = True
    propulsion_command_active: bool = False
    warp: WarpState = field(default_factory=WarpState)
    gate: GateTransitState = field(default_factory=GateTransitState)
    cloak: CloakState = field(default_factory=CloakState)
    squad_follow_state: str = "FORMATION_FOLLOW"
    squad_follow_leader_id: str | None = None
    squad_follow_leader_location_version: int = 0
    squad_follow_warp_ready: bool = True


@dataclass(slots=True)
class CombatState:
    lock_targets: set[str] = field(default_factory=set)
    current_target: str | None = None
    last_attack_target: str | None = None
    lock_started_at: dict[str, float] = field(default_factory=dict)
    lock_timers: dict[str, float] = field(default_factory=dict)
    lock_deadlines: dict[str, float] = field(default_factory=dict)
    prelocked_targets: set[str] = field(default_factory=set)
    prelock_timers: dict[str, float] = field(default_factory=dict)
    module_ammo_reload_timers: dict[str, float] = field(default_factory=dict)
    module_ammo_reload_deadlines: dict[str, float] = field(default_factory=dict)
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
    last_enemy_weapon_damaged_at: float = -1e9
    module_cycle_timers: dict[str, float] = field(default_factory=dict)
    module_cycle_deadlines: dict[str, float] = field(default_factory=dict)
    module_reactivation_timers: dict[str, float] = field(default_factory=dict)
    module_reactivation_deadlines: dict[str, float] = field(default_factory=dict)
    module_manual_modes: dict[str, str] = field(default_factory=dict)
    module_target_modes: dict[str, str] = field(default_factory=dict)
    module_decision_pending: set[str] = field(default_factory=set)
    module_decision_pending_signature: tuple[str, ...] = field(default_factory=tuple)
    module_decision_propulsion_active: bool | None = None
    module_decision_recent_enemy_damage_active: bool | None = None
    module_decision_enemy_targets_active: bool | None = None
    module_decision_ally_targets_active: bool | None = None


@dataclass(slots=True)
class ProjectileEntity:
    projectile_id: str
    kind: str
    source_ship_id: str
    source_module_id: str
    team: Team
    position: Vector2
    velocity: Vector2
    facing_deg: float
    target_ship_id: str | None
    speed: float
    max_speed: float
    max_range: float
    distance_traveled: float
    flight_time: float
    age: float
    acceleration_time: float
    damage_em: float
    damage_thermal: float
    damage_kinetic: float
    damage_explosive: float
    explosion_radius: float
    explosion_velocity: float
    damage_reduction_factor: float
    shield: float = 0.0
    armor: float = 0.0
    structure: float = 1.0
    shield_max: float = 0.0
    armor_max: float = 0.0
    structure_max: float = 1.0
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
    blast_radius: float = 0.0
    alive: bool = True
    system_id: str = ""


@dataclass(slots=True)
class ProjectileBlast:
    blast_id: str
    kind: str
    position: Vector2
    radius_m: float
    expires_at: float
    system_id: str = ""


@dataclass(slots=True)
class BubbleField:
    field_id: str
    kind: str
    interdiction_kind: str
    source_ship_id: str
    source_module_id: str
    team: Team
    position: Vector2
    radius_m: float
    expires_at: float
    blocks_warp: bool = False
    speed_factor_mult: float = 1.0
    anchor_ship_id: str | None = None
    destructible: bool = False
    shield: float = 0.0
    armor: float = 0.0
    structure: float = 0.0
    shield_max: float = 0.0
    armor_max: float = 0.0
    structure_max: float = 0.0
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
    alive: bool = True
    system_id: str = ""


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
    perception_allies: list[str] = field(default_factory=list)
    perception_enemies: list[str] = field(default_factory=list)
    perception_split_ready: bool = False
    drone_bay: list[DroneBayEntry] = field(default_factory=list)
    fighter_bay: list[FighterBayEntry] = field(default_factory=list)
    deployable_control: DeployableControlState = field(default_factory=DeployableControlState)
    ship_group_id: str = ""
    command_priority: int = 0
    deployed: bool = True
    fit_text: str = ""
    locked_module_charges: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SquadLeaderLocation:
    leader_id: str
    system_id: str
    location_version: int


@dataclass(slots=True)
class DroneEntity:
    ship_id: str
    owner_ship_id: str
    team: Team
    squad_id: str
    definition: DroneBayEntry
    fit: FitDescriptor
    profile: ShipProfile
    nav: NavigationState
    combat: CombatState
    vital: VitalState
    state: str = "idle"
    target_id: str | None = None
    connected: bool = True
    target_command_at: float = 0.0
    cycle_timer: float = 0.0
    ewar_cycle_timer: float = 0.0

    @property
    def type_name(self) -> str:
        return self.definition.type_name


@dataclass(slots=True)
class FighterEntity:
    ship_id: str
    owner_ship_id: str
    team: Team
    squad_id: str
    definition: FighterBayEntry
    fit: FitDescriptor
    profile: ShipProfile
    nav: NavigationState
    combat: CombatState
    vital: VitalState
    state: str = "idle"
    target_id: str | None = None
    owner_squad_id: str = ""
    connected: bool = True
    target_command_at: float = 0.0
    ability_cycle_timers: dict[str, float] = field(default_factory=dict)
    ability_ammo_remaining: dict[str, int] = field(default_factory=dict)
    ability_reload_timers: dict[str, float] = field(default_factory=dict)
    pending_manual_abilities: set[str] = field(default_factory=set)
    mwd_active_timer: float = 0.0
    mwd_cooldown_timer: float = 0.0

    @property
    def type_name(self) -> str:
        return self.definition.type_name


@dataclass(slots=True)
class StructureEntity:
    structure_id: str
    position: Vector2
    radius: float
    interaction_range: float
    kind: str
    system_id: str = ""
    display_name: str = ""
    icon_key: str = ""
    linked_structure_id: str | None = None

    @property
    def beacon_id(self) -> str:
        return self.structure_id


Beacon = StructureEntity


@dataclass(slots=True)
class FleetIntent:
    squad_id: str
    target_position: Vector2 | None = None
    movement_mode: str = "move"
    target_ship_id: str | None = None
    target_structure_id: str | None = None
    target_range_m: float = 0.0
    focus_target: str | None = None
    propulsion_active: bool | None = None
