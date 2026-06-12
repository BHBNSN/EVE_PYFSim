from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..math2d import Vector2
from ..models import (
    BubbleField,
    CombatState,
    DamageProfile,
    FitDescriptor,
    DroneBayEntry,
    DroneEntity,
    FighterBayEntry,
    FighterEntity,
    NavigationState,
    ProjectileBlast,
    ProjectileEntity,
    QualityLevel,
    QualityState,
    ShipEntity,
    ShipProfile,
    Team,
    VitalState,
)
from ..world import WorldState


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _team(value: Any, default: Team = Team.BLUE) -> Team:
    try:
        return Team(str(value))
    except ValueError:
        return default


def _vector(raw: Any) -> Vector2:
    if isinstance(raw, Mapping):
        return Vector2(_float(raw.get("x")), _float(raw.get("y")))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return Vector2(_float(raw[0]), _float(raw[1]))
    return Vector2(0.0, 0.0)


def _str_dict(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _float_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): _float(v) for k, v in raw.items()}


def _bool_dict(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): bool(v) for k, v in raw.items()}


def _fit_from_ship_snapshot(ship_id: str, data: Mapping[str, Any]) -> FitDescriptor:
    shield_max = max(0.0, _float(data.get("shield_max")))
    armor_max = max(0.0, _float(data.get("armor_max")))
    structure_max = max(0.0, _float(data.get("structure_max"), 1.0))
    return FitDescriptor(
        fit_key=str(ship_id),
        ship_name=str(data.get("ship_name") or ship_id),
        role="replay",
        base_dps=0.0,
        volley=0.0,
        optimal_range=120_000.0,
        falloff=0.0,
        tracking=1.0,
        shield_hp=shield_max,
        armor_hp=armor_max,
        structure_hp=structure_max,
        max_cap=max(0.0, _float(data.get("cap_max"))),
    )


def _profile_from_ship_snapshot(data: Mapping[str, Any]) -> ShipProfile:
    max_speed = max(_vector(data.get("velocity")).length(), 1.0)
    shield_max = max(0.0, _float(data.get("shield_max")))
    armor_max = max(0.0, _float(data.get("armor_max")))
    structure_max = max(0.0, _float(data.get("structure_max"), 1.0))
    cap_max = max(0.0, _float(data.get("cap_max")))
    return ShipProfile(
        dps=0.0,
        volley=0.0,
        optimal=120_000.0,
        falloff=0.0,
        tracking=1.0,
        sig_radius=120.0,
        scan_resolution=300.0,
        max_target_range=120_000.0,
        max_speed=max_speed,
        max_cap=cap_max,
        cap_recharge_time=100.0,
        shield_hp=shield_max,
        armor_hp=armor_max,
        structure_hp=structure_max,
        rep_amount=0.0,
        rep_cycle=5.0,
    )


def _ship_from_snapshot(ship_id: str, data: Mapping[str, Any]) -> ShipEntity:
    fit = _fit_from_ship_snapshot(ship_id, data)
    profile = _profile_from_ship_snapshot(data)
    nav = NavigationState(
        position=_vector(data.get("position")),
        velocity=_vector(data.get("velocity")),
        facing_deg=_float(data.get("facing_deg")),
        max_speed=profile.max_speed,
        system_id=str(data.get("system_id") or ""),
    )
    nav.gate.target_structure_id = str(data.get("gate_target_structure_id") or "") or None
    nav.cloak.active = _bool(data.get("gate_cloak_active"))
    nav.cloak.expires_at = _float(data.get("gate_cloak_expires_at"))
    nav.cloak.source = str(data.get("gate_cloak_source") or "")
    nav.follow_hold_active = _bool(data.get("follow_hold_active"))
    nav.follow_hold_leader_id = str(data.get("follow_hold_leader_id") or "") or None

    combat = CombatState(current_target=str(data.get("target") or "") or None)
    combat.projected_targets = _str_dict(data.get("projected_targets"))
    combat.module_cycle_timers = _float_dict(data.get("module_cycle_timers"))
    combat.ecm_jam_sources = _float_dict(data.get("ecm_jam_sources"))
    combat.ecm_last_attempt_target = str(data.get("ecm_last_attempt_target") or "") or None
    combat.ecm_last_attempt_module = str(data.get("ecm_last_attempt_module") or "") or None
    if data.get("ecm_last_attempt_success") is not None:
        combat.ecm_last_attempt_success = bool(data.get("ecm_last_attempt_success"))
    combat.ecm_last_attempt_chance = _float(data.get("ecm_last_attempt_chance"))
    combat.ecm_last_attempt_at = _float(data.get("ecm_last_attempt_at"), -1e9)
    combat.ecm_last_attempt_target_by_module = _str_dict(data.get("ecm_last_attempt_target_by_module"))
    combat.ecm_last_attempt_success_by_module = _bool_dict(data.get("ecm_last_attempt_success_by_module"))
    combat.ecm_last_attempt_at_by_module = _float_dict(data.get("ecm_last_attempt_at_by_module"))

    shield_max = max(0.0, _float(data.get("shield_max")))
    armor_max = max(0.0, _float(data.get("armor_max")))
    structure_max = max(0.0, _float(data.get("structure_max"), 1.0))
    return ShipEntity(
        ship_id=str(ship_id),
        team=_team(data.get("team")),
        squad_id=str(data.get("squad_id") or ""),
        fit=fit,
        profile=profile,
        nav=nav,
        combat=combat,
        vital=VitalState(
            shield=max(0.0, _float(data.get("shield"))),
            armor=max(0.0, _float(data.get("armor"))),
            structure=max(0.0, _float(data.get("structure"))),
            shield_max=shield_max,
            armor_max=armor_max,
            structure_max=structure_max,
            cap=max(0.0, _float(data.get("cap"))),
            cap_max=max(0.0, _float(data.get("cap_max"))),
            alive=_bool(data.get("alive"), True),
        ),
        quality=QualityState(QualityLevel.REGULAR, 0.0, 0.0, 0.0),
    )


def _fit_from_deployable_snapshot(entity_id: str, data: Mapping[str, Any], role: str) -> FitDescriptor:
    shield_max = max(0.0, _float(data.get("shield_max")))
    armor_max = max(0.0, _float(data.get("armor_max")))
    structure_max = max(0.0, _float(data.get("structure_max"), 1.0))
    max_speed = max(1.0, _vector(data.get("velocity")).length())
    return FitDescriptor(
        fit_key=f"replay:{role.lower()}:{entity_id}",
        ship_name=str(data.get("type_name") or entity_id),
        role=role,
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        signature_radius=25.0 if role == "DRONE" else 100.0,
        scan_resolution=200.0,
        max_speed=max_speed,
        max_cap=1.0,
        cap_recharge_time=1.0,
        shield_hp=shield_max,
        armor_hp=armor_max,
        structure_hp=structure_max,
    )


def _profile_from_deployable_snapshot(data: Mapping[str, Any], role: str) -> ShipProfile:
    shield_max = max(0.0, _float(data.get("shield_max")))
    armor_max = max(0.0, _float(data.get("armor_max")))
    structure_max = max(0.0, _float(data.get("structure_max"), 1.0))
    max_speed = max(1.0, _vector(data.get("velocity")).length())
    return ShipProfile(
        dps=0.0,
        volley=0.0,
        optimal=0.0,
        falloff=0.0,
        tracking=0.0,
        sig_radius=25.0 if role == "DRONE" else 100.0,
        scan_resolution=200.0,
        max_target_range=120_000.0,
        max_speed=max_speed,
        max_cap=1.0,
        cap_recharge_time=1.0,
        shield_hp=shield_max,
        armor_hp=armor_max,
        structure_hp=structure_max,
        rep_amount=0.0,
        rep_cycle=1.0,
    )


def _deployable_nav_from_snapshot(data: Mapping[str, Any], profile: ShipProfile) -> NavigationState:
    return NavigationState(
        position=_vector(data.get("position")),
        velocity=_vector(data.get("velocity")),
        facing_deg=_float(data.get("facing_deg")),
        max_speed=profile.max_speed,
        system_id=str(data.get("system_id") or ""),
        radius=12.0,
    )


def _deployable_vital_from_snapshot(data: Mapping[str, Any]) -> VitalState:
    return VitalState(
        shield=max(0.0, _float(data.get("shield"))),
        armor=max(0.0, _float(data.get("armor"))),
        structure=max(0.0, _float(data.get("structure"))),
        shield_max=max(0.0, _float(data.get("shield_max"))),
        armor_max=max(0.0, _float(data.get("armor_max"))),
        structure_max=max(0.0, _float(data.get("structure_max"), 1.0)),
        cap=1.0,
        cap_max=1.0,
        alive=_bool(data.get("alive"), True),
    )


def _drone_from_snapshot(drone_id: str, data: Mapping[str, Any]) -> DroneEntity:
    fit = _fit_from_deployable_snapshot(drone_id, data, "DRONE")
    profile = _profile_from_deployable_snapshot(data, "DRONE")
    definition = DroneBayEntry(
        type_name=str(data.get("type_name") or drone_id),
        quantity=1,
        group_name=str(data.get("group_name") or ""),
        bandwidth_mbit=0.0,
        volume_m3=0.0,
        max_velocity=profile.max_speed,
        orbit_range_m=0.0,
        control_range_m=120_000.0,
        cycle_time_s=5.0,
        optimal_range_m=0.0,
        falloff_m=0.0,
        tracking=0.0,
        damage=DamageProfile(),
        shield_hp=profile.shield_hp,
        armor_hp=profile.armor_hp,
        structure_hp=profile.structure_hp,
        signature_radius=profile.sig_radius,
        is_sentry=_bool(data.get("is_sentry")),
    )
    return DroneEntity(
        ship_id=str(drone_id),
        owner_ship_id=str(data.get("owner_ship_id") or ""),
        team=_team(data.get("team")),
        squad_id=str(data.get("squad_id") or ""),
        definition=definition,
        fit=fit,
        profile=profile,
        nav=_deployable_nav_from_snapshot(data, profile),
        combat=CombatState(),
        vital=_deployable_vital_from_snapshot(data),
        state=str(data.get("state") or "idle"),
        target_id=str(data.get("target_id") or "") or None,
        connected=_bool(data.get("connected"), True),
        target_command_at=_float(data.get("target_command_at")),
        cycle_timer=_float(data.get("cycle_timer")),
        ewar_cycle_timer=_float(data.get("ewar_cycle_timer")),
    )


def _fighter_from_snapshot(fighter_id: str, data: Mapping[str, Any]) -> FighterEntity:
    fit = _fit_from_deployable_snapshot(fighter_id, data, "FIGHTER")
    profile = _profile_from_deployable_snapshot(data, "FIGHTER")
    definition = FighterBayEntry(
        type_name=str(data.get("type_name") or fighter_id),
        quantity=1,
        group_name=str(data.get("group_name") or ""),
        slot_kind=str(data.get("slot_kind") or "support"),
        squadron_size=max(1, int(_float(data.get("squadron_size"), 1.0))),
        max_velocity=profile.max_speed,
        orbit_range_m=5_000.0,
        shield_hp=profile.shield_hp,
        armor_hp=profile.armor_hp,
        structure_hp=profile.structure_hp,
        signature_radius=profile.sig_radius,
        scan_resolution=profile.scan_resolution,
    )
    return FighterEntity(
        ship_id=str(fighter_id),
        owner_ship_id=str(data.get("owner_ship_id") or ""),
        team=_team(data.get("team")),
        squad_id=str(data.get("squad_id") or ""),
        definition=definition,
        fit=fit,
        profile=profile,
        nav=_deployable_nav_from_snapshot(data, profile),
        combat=CombatState(),
        vital=_deployable_vital_from_snapshot(data),
        state=str(data.get("state") or "idle"),
        target_id=str(data.get("target_id") or "") or None,
        owner_squad_id=str(data.get("owner_squad_id") or ""),
        connected=_bool(data.get("connected"), True),
        target_command_at=_float(data.get("target_command_at")),
        ability_cycle_timers=_float_dict(data.get("ability_cycle_timers")),
        ability_ammo_remaining={str(k): int(_float(v)) for k, v in (data.get("ability_ammo_remaining") or {}).items()} if isinstance(data.get("ability_ammo_remaining"), Mapping) else {},
        ability_reload_timers=_float_dict(data.get("ability_reload_timers")),
        pending_manual_abilities={str(item) for item in data.get("pending_manual_abilities", []) if str(item)} if isinstance(data.get("pending_manual_abilities"), list) else set(),
        mwd_active_timer=_float(data.get("mwd_active_timer")),
        mwd_cooldown_timer=_float(data.get("mwd_cooldown_timer")),
    )


def _projectile_from_snapshot(projectile_id: str, data: Mapping[str, Any]) -> ProjectileEntity:
    return ProjectileEntity(
        projectile_id=str(projectile_id),
        kind=str(data.get("kind") or "missile"),
        source_ship_id=str(data.get("source_ship_id") or ""),
        source_module_id=str(data.get("source_module_id") or ""),
        team=_team(data.get("team")),
        position=_vector(data.get("position")),
        velocity=_vector(data.get("velocity")),
        facing_deg=0.0,
        target_ship_id=str(data.get("target_ship_id") or "") or None,
        speed=max(0.0, _float(data.get("speed"))),
        max_speed=max(0.0, _float(data.get("max_speed"))),
        max_range=max(_float(data.get("distance_traveled")), 0.0),
        distance_traveled=max(0.0, _float(data.get("distance_traveled"))),
        flight_time=max(0.0, _float(data.get("flight_time"))),
        age=max(0.0, _float(data.get("age"))),
        acceleration_time=0.0,
        damage_em=0.0,
        damage_thermal=0.0,
        damage_kinetic=0.0,
        damage_explosive=0.0,
        explosion_radius=0.0,
        explosion_velocity=0.0,
        damage_reduction_factor=0.5,
        blast_radius=max(0.0, _float(data.get("blast_radius"))),
        alive=True,
        system_id=str(data.get("system_id") or ""),
    )


def _blast_from_snapshot(blast_id: str, data: Mapping[str, Any]) -> ProjectileBlast:
    return ProjectileBlast(
        blast_id=str(blast_id),
        kind=str(data.get("kind") or "blast"),
        position=_vector(data.get("position")),
        radius_m=max(0.0, _float(data.get("radius_m"))),
        expires_at=_float(data.get("expires_at")),
        system_id=str(data.get("system_id") or ""),
    )


def _bubble_from_snapshot(field_id: str, data: Mapping[str, Any]) -> BubbleField:
    return BubbleField(
        field_id=str(field_id),
        kind=str(data.get("kind") or "bubble"),
        interdiction_kind=str(data.get("interdiction_kind") or ""),
        source_ship_id=str(data.get("source_ship_id") or ""),
        source_module_id=str(data.get("source_module_id") or ""),
        team=_team(data.get("team")),
        position=_vector(data.get("position")),
        radius_m=max(0.0, _float(data.get("radius_m"))),
        expires_at=_float(data.get("expires_at")),
        blocks_warp=_bool(data.get("blocks_warp")),
        speed_factor_mult=max(0.0, _float(data.get("speed_factor_mult"), 1.0)),
        anchor_ship_id=str(data.get("anchor_ship_id") or "") or None,
        alive=_bool(data.get("alive"), True),
        system_id=str(data.get("system_id") or ""),
    )


def apply_snapshot_to_world(world: WorldState, snapshot: Mapping[str, Any]) -> WorldState:
    world.tick = int(_float(snapshot.get("tick"), 0.0))
    world.now = _float(snapshot.get("now", snapshot.get("at", 0.0)))

    raw_ships = snapshot.get("ships")
    world.ships = {
        str(ship_id): _ship_from_snapshot(str(ship_id), ship_data)
        for ship_id, ship_data in (raw_ships.items() if isinstance(raw_ships, Mapping) else ())
        if isinstance(ship_data, Mapping)
    }

    raw_drones = snapshot.get("drones")
    world.drones = {
        str(drone_id): _drone_from_snapshot(str(drone_id), drone_data)
        for drone_id, drone_data in (raw_drones.items() if isinstance(raw_drones, Mapping) else ())
        if isinstance(drone_data, Mapping)
    }

    raw_fighters = snapshot.get("fighters")
    world.fighters = {
        str(fighter_id): _fighter_from_snapshot(str(fighter_id), fighter_data)
        for fighter_id, fighter_data in (raw_fighters.items() if isinstance(raw_fighters, Mapping) else ())
        if isinstance(fighter_data, Mapping)
    }

    raw_projectiles = snapshot.get("projectiles")
    world.projectiles = {
        str(projectile_id): _projectile_from_snapshot(str(projectile_id), projectile_data)
        for projectile_id, projectile_data in (raw_projectiles.items() if isinstance(raw_projectiles, Mapping) else ())
        if isinstance(projectile_data, Mapping)
    }

    raw_blasts = snapshot.get("projectile_blasts")
    world.projectile_blasts = {
        str(blast_id): _blast_from_snapshot(str(blast_id), blast_data)
        for blast_id, blast_data in (raw_blasts.items() if isinstance(raw_blasts, Mapping) else ())
        if isinstance(blast_data, Mapping)
    }

    raw_bubbles = snapshot.get("bubble_fields")
    world.bubble_fields = {
        str(field_id): _bubble_from_snapshot(str(field_id), field_data)
        for field_id, field_data in (raw_bubbles.items() if isinstance(raw_bubbles, Mapping) else ())
        if isinstance(field_data, Mapping)
    }

    raw_focus_queues = snapshot.get("squad_focus_queues")
    world.squad_focus_queues = {
        str(key): [str(item) for item in value]
        for key, value in (raw_focus_queues.items() if isinstance(raw_focus_queues, Mapping) else ())
        if isinstance(value, list)
    }
    raw_focus_updated = snapshot.get("squad_focus_updated_at")
    world.squad_focus_updated_at = {
        str(key): _float(value)
        for key, value in (raw_focus_updated.items() if isinstance(raw_focus_updated, Mapping) else ())
    }
    return world
