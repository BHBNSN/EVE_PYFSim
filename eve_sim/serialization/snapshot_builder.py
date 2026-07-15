from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..domain.squad_follow_service import FORMATION_FOLLOW
from ..maps import serialize_map_definition
from .schemas import SnapshotOptions

if TYPE_CHECKING:
    from ..world import WorldState

MatchSnapshot = dict[str, Any]


class SnapshotBuilder:
    """Build stable match snapshots without making the simulation kernel serialize itself."""

    def build(
        self,
        world: "WorldState",
        options: SnapshotOptions | None = None,
        *,
        simulation_metadata: Mapping[str, Any] | None = None,
    ) -> MatchSnapshot:
        options = options or SnapshotOptions()
        payload = self._build_payload(world, simulation_metadata)
        if not options.include_transient_entities:
            payload["projectiles"] = {}
            payload["projectile_blasts"] = {}
            payload["bubble_fields"] = {}
        if not options.include_modules:
            for ship in payload.get("ships", {}).values():
                if isinstance(ship, dict):
                    ship.pop("module_states", None)
                    ship.pop("module_cycle_timers", None)
        if not options.include_diagnostics:
            payload.pop("simulation_metadata", None)
        return payload

    @staticmethod
    def _build_payload(
        world: "WorldState",
        simulation_metadata: Mapping[str, Any] | None,
    ) -> MatchSnapshot:
        ships: dict[str, dict[str, Any]] = {}
        for ship_id, ship in world.ships.items():
            module_states: dict[str, str] = {}
            if ship.runtime is not None:
                module_states = {
                    module.module_id: module.normalized_state().value
                    for module in ship.runtime.modules
                }
            ships[ship_id] = {
                "ship_id": ship_id,
                "team": ship.team.value,
                "squad_id": ship.squad_id,
                "ship_group_id": str(getattr(ship, "ship_group_id", "") or ""),
                "command_priority": int(getattr(ship, "command_priority", 0) or 0),
                "ship_name": ship.fit.ship_name,
                "role": ship.fit.role,
                "fit_text": str(ship.fit_text or ""),
                "locked_module_charges": dict(ship.locked_module_charges),
                "alive": ship.vital.alive,
                "deployed": bool(ship.deployed),
                "position": {"x": ship.nav.position.x, "y": ship.nav.position.y},
                "velocity": {"x": ship.nav.velocity.x, "y": ship.nav.velocity.y},
                "facing_deg": ship.nav.facing_deg,
                "system_id": str(getattr(ship.nav, "system_id", "") or ""),
                "command_mode": str(getattr(ship.nav, "command_mode", "") or ""),
                "command_target": (
                    {
                        "x": float(ship.nav.command_target.x),
                        "y": float(ship.nav.command_target.y),
                    }
                    if ship.nav.command_target is not None
                    else None
                ),
                "command_target_ship_id": str(getattr(ship.nav, "command_target_ship_id", "") or ""),
                "command_target_structure_id": str(getattr(ship.nav, "command_target_structure_id", "") or ""),
                "command_range_m": float(getattr(ship.nav, "command_range_m", 0.0) or 0.0),
                "command_orbit_clockwise": bool(getattr(ship.nav, "command_orbit_clockwise", True)),
                "gate_target_structure_id": str(getattr(getattr(ship.nav, "gate", None), "target_structure_id", "") or ""),
                "gate_cloak_active": bool(getattr(getattr(ship.nav, "cloak", None), "active", False)),
                "gate_cloak_expires_at": float(getattr(getattr(ship.nav, "cloak", None), "expires_at", 0.0) or 0.0),
                "gate_cloak_source": str(getattr(getattr(ship.nav, "cloak", None), "source", "") or ""),
                "squad_follow_state": str(getattr(ship.nav, "squad_follow_state", FORMATION_FOLLOW) or FORMATION_FOLLOW),
                "squad_follow_leader_id": str(getattr(ship.nav, "squad_follow_leader_id", "") or ""),
                "squad_follow_leader_location_version": int(
                    getattr(ship.nav, "squad_follow_leader_location_version", 0) or 0
                ),
                "squad_follow_warp_ready": bool(getattr(ship.nav, "squad_follow_warp_ready", True)),
                "shield": ship.vital.shield,
                "armor": ship.vital.armor,
                "structure": ship.vital.structure,
                "shield_max": ship.vital.shield_max,
                "armor_max": ship.vital.armor_max,
                "structure_max": ship.vital.structure_max,
                "cap": ship.vital.cap,
                "cap_max": ship.vital.cap_max,
                "target": ship.combat.current_target,
                "projected_targets": dict(ship.combat.projected_targets),
                "prelocked_targets": sorted(str(target_id) for target_id in ship.combat.prelocked_targets),
                "prelock_timers": {k: float(v) for k, v in ship.combat.prelock_timers.items()},
                "module_cycle_timers": {k: float(v) for k, v in ship.combat.module_cycle_timers.items()},
                "ecm_jam_sources": {k: float(v) for k, v in ship.combat.ecm_jam_sources.items()},
                "ecm_last_attempt_target": ship.combat.ecm_last_attempt_target,
                "ecm_last_attempt_module": ship.combat.ecm_last_attempt_module,
                "ecm_last_attempt_success": ship.combat.ecm_last_attempt_success,
                "ecm_last_attempt_chance": float(ship.combat.ecm_last_attempt_chance),
                "ecm_last_attempt_at": float(ship.combat.ecm_last_attempt_at),
                "ecm_last_attempt_target_by_module": {
                    k: str(v) for k, v in ship.combat.ecm_last_attempt_target_by_module.items()
                },
                "ecm_last_attempt_success_by_module": {
                    k: bool(v) for k, v in ship.combat.ecm_last_attempt_success_by_module.items()
                },
                "ecm_last_attempt_at_by_module": {
                    k: float(v) for k, v in ship.combat.ecm_last_attempt_at_by_module.items()
                },
                "module_states": module_states,
                "module_manual_modes": dict(ship.combat.module_manual_modes),
                "module_target_modes": dict(ship.combat.module_target_modes),
            }
        return {
            "tick": world.tick,
            "now": world.now,
            "ships": ships,
            "drones": {
                drone_id: {
                    "ship_id": drone_id,
                    "owner_ship_id": drone.owner_ship_id,
                    "team": drone.team.value,
                    "squad_id": drone.squad_id,
                    "type_name": drone.definition.type_name,
                    "group_name": drone.definition.group_name,
                    "max_velocity": float(drone.definition.max_velocity),
                    "state": drone.state,
                    "target_id": drone.target_id,
                    "connected": bool(drone.connected),
                    "target_command_at": float(drone.target_command_at),
                    "alive": drone.vital.alive,
                    "is_sentry": bool(drone.definition.is_sentry),
                    "position": {"x": drone.nav.position.x, "y": drone.nav.position.y},
                    "velocity": {"x": drone.nav.velocity.x, "y": drone.nav.velocity.y},
                    "facing_deg": drone.nav.facing_deg,
                    "system_id": str(getattr(drone.nav, "system_id", "") or ""),
                    "shield": drone.vital.shield,
                    "armor": drone.vital.armor,
                    "structure": drone.vital.structure,
                    "shield_max": drone.vital.shield_max,
                    "armor_max": drone.vital.armor_max,
                    "structure_max": drone.vital.structure_max,
                    "cycle_timer": float(drone.cycle_timer),
                    "ewar_cycle_timer": float(drone.ewar_cycle_timer),
                }
                for drone_id, drone in world.drones.items()
            },
            "fighters": {
                fighter_id: {
                    "ship_id": fighter_id,
                    "owner_ship_id": fighter.owner_ship_id,
                    "team": fighter.team.value,
                    "squad_id": fighter.squad_id,
                    "owner_squad_id": fighter.owner_squad_id,
                    "type_name": fighter.definition.type_name,
                    "group_name": fighter.definition.group_name,
                    "slot_kind": fighter.definition.slot_kind,
                    "squadron_size": int(fighter.definition.squadron_size),
                    "max_velocity": float(fighter.definition.max_velocity),
                    "state": fighter.state,
                    "target_id": fighter.target_id,
                    "connected": bool(fighter.connected),
                    "target_command_at": float(fighter.target_command_at),
                    "alive": fighter.vital.alive,
                    "position": {"x": fighter.nav.position.x, "y": fighter.nav.position.y},
                    "velocity": {"x": fighter.nav.velocity.x, "y": fighter.nav.velocity.y},
                    "facing_deg": fighter.nav.facing_deg,
                    "system_id": str(getattr(fighter.nav, "system_id", "") or ""),
                    "shield": fighter.vital.shield,
                    "armor": fighter.vital.armor,
                    "structure": fighter.vital.structure,
                    "shield_max": fighter.vital.shield_max,
                    "armor_max": fighter.vital.armor_max,
                    "structure_max": fighter.vital.structure_max,
                    "ability_cycle_timers": {k: float(v) for k, v in fighter.ability_cycle_timers.items()},
                    "ability_ammo_remaining": {k: int(v) for k, v in fighter.ability_ammo_remaining.items()},
                    "ability_reload_timers": {k: float(v) for k, v in fighter.ability_reload_timers.items()},
                    "pending_manual_abilities": sorted(str(k) for k in fighter.pending_manual_abilities),
                    "mwd_active_timer": float(fighter.mwd_active_timer),
                    "mwd_cooldown_timer": float(fighter.mwd_cooldown_timer),
                }
                for fighter_id, fighter in world.fighters.items()
            },
            "projectiles": {
                projectile_id: {
                    "projectile_id": projectile.projectile_id,
                    "kind": projectile.kind,
                    "source_ship_id": projectile.source_ship_id,
                    "source_module_id": projectile.source_module_id,
                    "team": projectile.team.value,
                    "position": {"x": projectile.position.x, "y": projectile.position.y},
                    "velocity": {"x": projectile.velocity.x, "y": projectile.velocity.y},
                    "system_id": str(getattr(projectile, "system_id", "") or ""),
                    "target_ship_id": projectile.target_ship_id,
                    "speed": float(projectile.speed),
                    "max_speed": float(projectile.max_speed),
                    "distance_traveled": float(projectile.distance_traveled),
                    "flight_time": float(projectile.flight_time),
                    "age": float(projectile.age),
                    "blast_radius": float(projectile.blast_radius),
                }
                for projectile_id, projectile in world.projectiles.items()
            },
            "projectile_blasts": {
                blast_id: {
                    "blast_id": blast.blast_id,
                    "kind": blast.kind,
                    "position": {"x": blast.position.x, "y": blast.position.y},
                    "system_id": str(getattr(blast, "system_id", "") or ""),
                    "radius_m": float(blast.radius_m),
                    "expires_at": float(blast.expires_at),
                }
                for blast_id, blast in world.projectile_blasts.items()
            },
            "bubble_fields": {
                field_id: {
                    "field_id": field.field_id,
                    "kind": field.kind,
                    "interdiction_kind": field.interdiction_kind,
                    "source_ship_id": field.source_ship_id,
                    "source_module_id": field.source_module_id,
                    "team": field.team.value,
                    "position": {"x": field.position.x, "y": field.position.y},
                    "system_id": str(getattr(field, "system_id", "") or ""),
                    "radius_m": float(field.radius_m),
                    "expires_at": float(field.expires_at),
                    "blocks_warp": bool(field.blocks_warp),
                    "speed_factor_mult": float(field.speed_factor_mult),
                    "anchor_ship_id": field.anchor_ship_id,
                    "alive": bool(field.alive),
                }
                for field_id, field in world.bubble_fields.items()
            },
            "intents": {k: asdict(v) for k, v in world.intents.items()},
            "squad_leaders": {k: str(v) for k, v in world.squad_leaders.items()},
            "squad_leader_location_versions": {
                k: int(v) for k, v in world.squad_leader_location_versions.items()
            },
            "squad_propulsion_commands": {k: bool(v) for k, v in world.squad_propulsion_commands.items()},
            "squad_leader_speed_limits": {k: float(v) for k, v in world.squad_leader_speed_limits.items()},
            "squad_focus_queues": {k: list(v) for k, v in world.squad_focus_queues.items()},
            "squad_focus_updated_at": {k: float(v) for k, v in world.squad_focus_updated_at.items()},
            "map": (
                serialize_map_definition(world.map_definition)
                if world.map_definition is not None
                else None
            ),
            "simulation_metadata": dict(simulation_metadata or {}),
        }
