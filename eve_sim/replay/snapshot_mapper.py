from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..math2d import Vector2
from ..models import (
    BubbleField,
    CombatState,
    FitDescriptor,
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
    return world
