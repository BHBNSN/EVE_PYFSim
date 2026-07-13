from __future__ import annotations

import math
import random

from ..math2d import Vector2
from ..models import (
    CombatState,
    DamageProfile,
    DeployableEwarProfile,
    DroneBayEntry,
    DroneEntity,
    FighterAbilityProfile,
    FighterBayEntry,
    FighterEntity,
    FitDescriptor,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipProfile,
    Team,
    VitalState,
)
from ..world import WorldState
from .movement import MovementSystem


class DeployableSystem:
    RECOVERY_RANGE_M = 2_500.0
    DISCONNECT_RANGE_M = 500_000.0
    FIGHTER_SQUAD_SUFFIX = " Fighters"

    def __init__(self, combat_system, movement_system: MovementSystem) -> None:
        self.combat = combat_system
        self.movement = movement_system
        self._sequence = 0

    @staticmethod
    def _focus_key(team: Team, squad_id: str) -> str:
        return f"{team.value}:{squad_id}"

    @staticmethod
    def _entity_system_id(entity) -> str:
        nav = getattr(entity, "nav", None)
        if nav is not None:
            return str(getattr(nav, "system_id", "") or "")
        return str(getattr(entity, "system_id", "") or "")

    @classmethod
    def fighter_squad_id(cls, owner_squad_id: str) -> str:
        squad = str(owner_squad_id or "").strip()
        return f"{squad}{cls.FIGHTER_SQUAD_SUFFIX}" if squad else cls.FIGHTER_SQUAD_SUFFIX.strip()

    @staticmethod
    def _damage_tuple(damage: DamageProfile) -> tuple[float, float, float, float]:
        return (
            max(0.0, float(damage.em)),
            max(0.0, float(damage.thermal)),
            max(0.0, float(damage.kinetic)),
            max(0.0, float(damage.explosive)),
        )

    @staticmethod
    def _type_slug(type_name: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(type_name or "deployable"))
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "deployable"

    @staticmethod
    def _deploy_offset(sequence: int, radius_m: float = 350.0) -> Vector2:
        angle = math.radians((sequence * 137.507764) % 360.0)
        return Vector2(math.cos(angle) * radius_m, math.sin(angle) * radius_m)

    @staticmethod
    def _quality_state() -> QualityState:
        return QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        )

    @classmethod
    def _drone_fit(cls, definition: DroneBayEntry) -> FitDescriptor:
        cycle = max(0.1, float(definition.cycle_time_s or 0.0))
        dps = definition.damage.total / cycle if cycle > 0.0 else 0.0
        return FitDescriptor(
            fit_key=f"drone:{definition.type_name}",
            ship_name=definition.type_name,
            role="DRONE",
            base_dps=dps,
            volley=definition.damage.total,
            optimal_range=definition.optimal_range_m,
            falloff=definition.falloff_m,
            tracking=definition.tracking,
            signature_radius=definition.signature_radius,
            scan_resolution=definition.scan_resolution,
            max_target_range=max(1_000.0, definition.control_range_m),
            sensor_strength_gravimetric=definition.sensor_strength_gravimetric,
            sensor_strength_ladar=definition.sensor_strength_ladar,
            sensor_strength_magnetometric=definition.sensor_strength_magnetometric,
            sensor_strength_radar=definition.sensor_strength_radar,
            max_speed=definition.max_velocity,
            max_cap=1.0,
            cap_recharge_time=1.0,
            shield_hp=definition.shield_hp,
            armor_hp=definition.armor_hp,
            structure_hp=definition.structure_hp,
            mass=1_000.0,
            agility=1.0,
        )

    @classmethod
    def _drone_profile(cls, definition: DroneBayEntry) -> ShipProfile:
        cycle = max(0.1, float(definition.cycle_time_s or 0.0))
        dps = definition.damage.total / cycle if cycle > 0.0 else 0.0
        return ShipProfile(
            dps=dps,
            volley=definition.damage.total,
            optimal=definition.optimal_range_m,
            falloff=definition.falloff_m,
            tracking=definition.tracking,
            sig_radius=definition.signature_radius,
            scan_resolution=definition.scan_resolution,
            max_target_range=max(1_000.0, definition.control_range_m),
            max_speed=definition.max_velocity,
            max_cap=1.0,
            cap_recharge_time=1.0,
            shield_hp=definition.shield_hp,
            armor_hp=definition.armor_hp,
            structure_hp=definition.structure_hp,
            rep_amount=0.0,
            rep_cycle=1.0,
            weapon_system="turret",
            optimal_sig=40.0,
            turret_dps=dps,
            turret_cycle=cycle,
            damage_em=definition.damage.em,
            damage_thermal=definition.damage.thermal,
            damage_kinetic=definition.damage.kinetic,
            damage_explosive=definition.damage.explosive,
            sensor_strength_gravimetric=definition.sensor_strength_gravimetric,
            sensor_strength_ladar=definition.sensor_strength_ladar,
            sensor_strength_magnetometric=definition.sensor_strength_magnetometric,
            sensor_strength_radar=definition.sensor_strength_radar,
            mass=1_000.0,
            agility=1.0,
        )

    @classmethod
    def _fighter_fit(cls, definition: FighterBayEntry) -> FitDescriptor:
        normal = next((ability for ability in definition.abilities if ability.kind == "normal_attack"), None)
        cycle = max(0.1, float(getattr(normal, "cycle_time_s", 0.0) or 0.0))
        damage = normal.damage if normal is not None else DamageProfile()
        dps = damage.total / cycle if normal is not None and cycle > 0.0 else 0.0
        return FitDescriptor(
            fit_key=f"fighter:{definition.type_name}",
            ship_name=definition.type_name,
            role=f"FIGHTER_{definition.slot_kind.upper()}",
            base_dps=dps,
            volley=damage.total,
            optimal_range=float(getattr(normal, "optimal_range_m", 0.0) or 0.0),
            falloff=float(getattr(normal, "falloff_m", 0.0) or 0.0),
            tracking=float(getattr(normal, "tracking", 0.0) or 0.0),
            missile_explosion_radius=float(getattr(normal, "explosion_radius", 0.0) or 0.0),
            missile_explosion_velocity=float(getattr(normal, "explosion_velocity", 0.0) or 0.0),
            signature_radius=definition.signature_radius,
            scan_resolution=definition.scan_resolution,
            max_target_range=120_000.0,
            sensor_strength_gravimetric=definition.sensor_strength_gravimetric,
            sensor_strength_ladar=definition.sensor_strength_ladar,
            sensor_strength_magnetometric=definition.sensor_strength_magnetometric,
            sensor_strength_radar=definition.sensor_strength_radar,
            max_speed=definition.max_velocity,
            max_cap=1.0,
            cap_recharge_time=1.0,
            shield_hp=definition.shield_hp,
            armor_hp=definition.armor_hp,
            structure_hp=definition.structure_hp,
            mass=5_000.0,
            agility=1.0,
            warp_speed_au_s=definition.warp_speed_au_s,
        )

    @classmethod
    def _fighter_profile(cls, definition: FighterBayEntry) -> ShipProfile:
        normal = next((ability for ability in definition.abilities if ability.kind == "normal_attack"), None)
        cycle = max(0.1, float(getattr(normal, "cycle_time_s", 0.0) or 0.0))
        damage = normal.damage if normal is not None else DamageProfile()
        dps = damage.total / cycle if normal is not None and cycle > 0.0 else 0.0
        return ShipProfile(
            dps=dps,
            volley=damage.total,
            optimal=float(getattr(normal, "optimal_range_m", 0.0) or 0.0),
            falloff=float(getattr(normal, "falloff_m", 0.0) or 0.0),
            tracking=float(getattr(normal, "tracking", 0.0) or 0.0),
            sig_radius=definition.signature_radius,
            scan_resolution=definition.scan_resolution,
            max_target_range=120_000.0,
            max_locked_targets=1,
            max_speed=definition.max_velocity,
            max_cap=1.0,
            cap_recharge_time=1.0,
            shield_hp=definition.shield_hp,
            armor_hp=definition.armor_hp,
            structure_hp=definition.structure_hp,
            rep_amount=0.0,
            rep_cycle=1.0,
            weapon_system="missile" if normal is not None and normal.explosion_radius > 0.0 else "turret",
            optimal_sig=40.0,
            turret_dps=0.0 if normal is not None and normal.explosion_radius > 0.0 else dps,
            missile_dps=dps if normal is not None and normal.explosion_radius > 0.0 else 0.0,
            turret_cycle=cycle,
            missile_cycle=cycle,
            damage_em=damage.em,
            damage_thermal=damage.thermal,
            damage_kinetic=damage.kinetic,
            damage_explosive=damage.explosive,
            missile_explosion_radius=float(getattr(normal, "explosion_radius", 0.0) or 0.0),
            missile_explosion_velocity=float(getattr(normal, "explosion_velocity", 0.0) or 0.0),
            missile_max_range=float(getattr(normal, "optimal_range_m", 0.0) or 0.0)
            + float(getattr(normal, "falloff_m", 0.0) or 0.0),
            missile_damage_reduction_factor=float(getattr(normal, "damage_reduction_factor", 0.5) or 0.5),
            sensor_strength_gravimetric=definition.sensor_strength_gravimetric,
            sensor_strength_ladar=definition.sensor_strength_ladar,
            sensor_strength_magnetometric=definition.sensor_strength_magnetometric,
            sensor_strength_radar=definition.sensor_strength_radar,
            mass=5_000.0,
            agility=1.0,
            warp_speed_au_s=definition.warp_speed_au_s,
        )

    @staticmethod
    def _vital_from_hp(shield: float, armor: float, structure: float) -> VitalState:
        return VitalState(
            shield=max(0.0, float(shield)),
            armor=max(0.0, float(armor)),
            structure=max(1.0, float(structure)),
            shield_max=max(0.0, float(shield)),
            armor_max=max(0.0, float(armor)),
            structure_max=max(1.0, float(structure)),
            cap=1.0,
            cap_max=1.0,
            alive=True,
        )

    def _next_id(self, world: WorldState, prefix: str, owner_ship_id: str, type_name: str) -> str:
        slug = self._type_slug(type_name)
        while True:
            self._sequence += 1
            entity_id = f"{prefix}:{owner_ship_id}:{slug}:{self._sequence:04d}"
            if world.combat_entity(entity_id) is None:
                return entity_id

    @staticmethod
    def _squad_members(world: WorldState, team: Team, squad_id: str):
        members = [
            ship
            for ship in world.ships.values()
            if ship.team == team and ship.squad_id == squad_id and ship.vital.alive
        ]
        members.sort(key=lambda ship: ship.ship_id)
        return members

    @staticmethod
    def _entry_matches_type(entry, type_name: str) -> bool:
        return str(entry.type_name).strip().lower() == str(type_name).strip().lower()

    def _active_drone_usage(self, world: WorldState, owner_ship_id: str) -> tuple[int, float]:
        active = [
            drone
            for drone in world.drones.values()
            if drone.owner_ship_id == owner_ship_id and drone.vital.alive
        ]
        bandwidth = sum(max(0.0, float(drone.definition.bandwidth_mbit)) for drone in active)
        return len(active), bandwidth

    def _created_drone_count(self, world: WorldState, owner_ship_id: str, type_name: str) -> int:
        return sum(
            1
            for drone in world.drones.values()
            if drone.owner_ship_id == owner_ship_id and self._entry_matches_type(drone.definition, type_name)
        )

    def launch_squad_drones(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int:
        launched = 0
        for owner in self._squad_members(world, team, squad_id):
            entry = next((item for item in owner.drone_bay if self._entry_matches_type(item, type_name)), None)
            if entry is None or entry.quantity <= 0:
                continue
            active_count, used_bandwidth = self._active_drone_usage(world, owner.ship_id)
            control = owner.deployable_control
            max_active = max(0, int(control.max_active_drones or 0))
            if max_active <= 0:
                continue
            remaining_slots = max(0, max_active - active_count)
            remaining_inventory = max(0, int(entry.quantity) - self._created_drone_count(world, owner.ship_id, entry.type_name))
            remaining_bandwidth = max(0.0, float(control.drone_bandwidth_mbit or 0.0) - used_bandwidth)
            if entry.bandwidth_mbit > 0.0:
                bandwidth_slots = int(remaining_bandwidth // max(1e-6, float(entry.bandwidth_mbit)))
            else:
                bandwidth_slots = remaining_slots
            count = max(0, min(remaining_slots, remaining_inventory, bandwidth_slots))
            for _ in range(count):
                entity_id = self._next_id(world, "drone", owner.ship_id, entry.type_name)
                offset = self._deploy_offset(self._sequence, max(150.0, owner.nav.radius + 150.0))
                position = owner.nav.position + offset
                profile = self._drone_profile(entry)
                drone = DroneEntity(
                    ship_id=entity_id,
                    owner_ship_id=owner.ship_id,
                    team=owner.team,
                    squad_id=owner.squad_id,
                    definition=entry,
                    fit=self._drone_fit(entry),
                    profile=profile,
                    nav=NavigationState(
                        position=position,
                        velocity=Vector2(owner.nav.velocity.x, owner.nav.velocity.y),
                        facing_deg=owner.nav.facing_deg,
                        max_speed=profile.max_speed,
                        system_id=str(getattr(owner.nav, "system_id", "") or ""),
                        radius=max(8.0, min(80.0, entry.signature_radius * 0.25)),
                    ),
                    combat=CombatState(),
                    vital=self._vital_from_hp(entry.shield_hp, entry.armor_hp, entry.structure_hp),
                    state="idle",
                    target_id=None,
                    connected=True,
                    target_command_at=0.0,
                    cycle_timer=0.0,
                    ewar_cycle_timer=0.0,
                )
                world.drones[entity_id] = drone
                launched += 1
        if launched:
            self._queue_event("deployable_launch", world, count=float(launched), deployable_kind="drone", type_name=type_name)
        return launched

    def _active_fighter_usage(self, world: WorldState, owner_ship_id: str) -> tuple[int, dict[str, int]]:
        fighters = [
            fighter
            for fighter in world.fighters.values()
            if fighter.owner_ship_id == owner_ship_id and fighter.vital.alive
        ]
        by_slot: dict[str, int] = {"light": 0, "support": 0, "heavy": 0}
        for fighter in fighters:
            key = str(fighter.definition.slot_kind or "support").lower()
            by_slot[key] = by_slot.get(key, 0) + 1
        return len(fighters), by_slot

    def _created_fighter_count(self, world: WorldState, owner_ship_id: str, type_name: str) -> int:
        return sum(
            1
            for fighter in world.fighters.values()
            if fighter.owner_ship_id == owner_ship_id and self._entry_matches_type(fighter.definition, type_name)
        )

    @staticmethod
    def _slot_limit(control, slot_kind: str) -> int:
        key = str(slot_kind or "support").lower()
        if key == "light":
            return max(0, int(control.fighter_light_slots or 0))
        if key == "heavy":
            return max(0, int(control.fighter_heavy_slots or 0))
        return max(0, int(control.fighter_support_slots or 0))

    def launch_squad_fighters(self, world: WorldState, team: Team, squad_id: str, type_name: str) -> int:
        launched = 0
        for owner in self._squad_members(world, team, squad_id):
            entry = next((item for item in owner.fighter_bay if self._entry_matches_type(item, type_name)), None)
            if entry is None or entry.quantity <= 0:
                continue
            control = owner.deployable_control
            active_count, by_slot = self._active_fighter_usage(world, owner.ship_id)
            tube_slots = max(0, int(control.fighter_tubes or 0) - active_count)
            slot_limit = self._slot_limit(control, entry.slot_kind)
            slot_slots = max(0, slot_limit - by_slot.get(str(entry.slot_kind).lower(), 0)) if slot_limit > 0 else tube_slots
            remaining_inventory = max(0, int(entry.quantity) - self._created_fighter_count(world, owner.ship_id, entry.type_name))
            count = max(0, min(tube_slots, slot_slots, remaining_inventory))
            for _ in range(count):
                entity_id = self._next_id(world, "fighter", owner.ship_id, entry.type_name)
                offset = self._deploy_offset(self._sequence, max(300.0, owner.nav.radius + 300.0))
                position = owner.nav.position + offset
                profile = self._fighter_profile(entry)
                ammo = {
                    ability.ability_id: max(0, int(ability.ammo_capacity))
                    for ability in entry.abilities
                    if ability.ammo_capacity > 0
                }
                fighter = FighterEntity(
                    ship_id=entity_id,
                    owner_ship_id=owner.ship_id,
                    team=owner.team,
                    squad_id=self.fighter_squad_id(owner.squad_id),
                    definition=entry,
                    fit=self._fighter_fit(entry),
                    profile=profile,
                    nav=NavigationState(
                        position=position,
                        velocity=Vector2(owner.nav.velocity.x, owner.nav.velocity.y),
                        facing_deg=owner.nav.facing_deg,
                        max_speed=profile.max_speed,
                        system_id=str(getattr(owner.nav, "system_id", "") or ""),
                        radius=max(25.0, min(160.0, entry.signature_radius * 0.4)),
                    ),
                    combat=CombatState(),
                    vital=self._vital_from_hp(entry.shield_hp, entry.armor_hp, entry.structure_hp),
                    state="idle",
                    target_id=None,
                    owner_squad_id=owner.squad_id,
                    connected=True,
                    target_command_at=0.0,
                    ability_ammo_remaining=ammo,
                )
                world.fighters[entity_id] = fighter
                launched += 1
        if launched:
            self._queue_event("deployable_launch", world, count=float(launched), deployable_kind="fighter", type_name=type_name)
        return launched

    def recall_squad_deployables(self, world: WorldState, team: Team, squad_id: str) -> int:
        changed = 0
        owner_ids = {
            ship.ship_id
            for ship in world.ships.values()
            if ship.team == team and ship.squad_id == squad_id and ship.vital.alive
        }
        if not owner_ids:
            owner_ids = {
                fighter.owner_ship_id
                for fighter in world.fighters.values()
                if fighter.team == team and fighter.squad_id == squad_id and fighter.vital.alive
            }
        for drone in world.drones.values():
            if drone.owner_ship_id in owner_ids and drone.vital.alive:
                drone.state = "recalling"
                drone.target_id = None
                changed += 1
        for fighter in world.fighters.values():
            if fighter.owner_ship_id in owner_ids and fighter.vital.alive:
                fighter.state = "recalling"
                fighter.target_id = None
                changed += 1
        if changed:
            self._queue_event("deployable_recall", world, count=float(changed), deployable_kind="all", type_name="")
        return changed

    @staticmethod
    def _fighter_squad_members(world: WorldState, team: Team, squad_id: str) -> list[FighterEntity]:
        squad = str(squad_id or "").strip()
        members = [
            fighter
            for fighter in world.fighters.values()
            if fighter.team == team and fighter.squad_id == squad and fighter.vital.alive
        ]
        members.sort(key=lambda fighter: fighter.ship_id)
        return members

    def command_fighter_squad_move(self, world: WorldState, team: Team, squad_id: str, target: Vector2) -> int:
        changed = 0
        for fighter in self._fighter_squad_members(world, team, squad_id):
            if not getattr(fighter, "connected", True) or fighter.state == "recalling":
                continue
            self.movement._cancel_warp(fighter)
            fighter.nav.command_target = Vector2(target.x, target.y)
            fighter.nav.command_mode = "move"
            fighter.nav.command_target_ship_id = None
            fighter.nav.command_target_structure_id = None
            fighter.nav.command_range_m = 0.0
            fighter.state = "moving"
            changed += 1
        return changed

    def command_fighter_squad_navigation(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        *,
        target_kind: str,
        target_id: str,
        movement_mode: str,
        range_m: float = 0.0,
    ) -> int:
        mode = str(movement_mode or "").strip().lower()
        kind = str(target_kind or "").strip().lower()
        target_key = str(target_id or "").strip()
        if mode not in {"approach", "keep_range", "orbit"} or kind not in {"ship", "structure"} or not target_key:
            return 0
        target_position: Vector2 | None = None
        target_system_id = ""
        if kind == "ship":
            target = world.combat_entity(target_key)
            if target is None or not target.vital.alive:
                return 0
            target_position = Vector2(target.nav.position.x, target.nav.position.y)
            target_system_id = self._entity_system_id(target)
        else:
            structure = world.structures.get(target_key)
            if structure is None:
                return 0
            target_position = Vector2(structure.position.x, structure.position.y)
            target_system_id = self._entity_system_id(structure)

        changed = 0
        for fighter in self._fighter_squad_members(world, team, squad_id):
            if not getattr(fighter, "connected", True) or fighter.state == "recalling":
                continue
            if self._entity_system_id(fighter) != target_system_id:
                continue
            fighter.nav.command_mode = mode
            fighter.nav.command_target = Vector2(target_position.x, target_position.y)
            fighter.nav.command_target_ship_id = target_key if kind == "ship" else None
            fighter.nav.command_target_structure_id = target_key if kind == "structure" else None
            fighter.nav.command_range_m = max(0.0, float(range_m or 0.0))
            fighter.state = "moving" if not fighter.target_id else "engaging"
            changed += 1
        return changed

    def command_fighter_squad_warp(
        self,
        world: WorldState,
        team: Team,
        squad_id: str,
        target_position: Vector2,
        *,
        target_ship_id: str | None = None,
        target_beacon_id: str | None = None,
    ) -> int:
        changed = 0
        target_ship = world.combat_entity(target_ship_id) if target_ship_id else None
        target_beacon = world.structures.get(str(target_beacon_id)) if target_beacon_id else None
        if target_ship_id and (target_ship is None or not target_ship.vital.alive):
            return 0
        if target_beacon_id and target_beacon is None:
            return 0
        for fighter in self._fighter_squad_members(world, team, squad_id):
            if not getattr(fighter, "connected", True) or fighter.state == "recalling":
                continue
            if target_ship is not None and not self._same_system(fighter, target_ship):
                continue
            if target_beacon is not None and self._entity_system_id(fighter) != self._entity_system_id(target_beacon):
                continue
            distance = fighter.nav.position.distance_to(target_position)
            if distance < self.movement.MIN_WARP_DISTANCE_M or self.movement._ship_is_scrammed(fighter):
                fighter.nav.command_target = Vector2(target_position.x, target_position.y)
                fighter.nav.command_mode = "approach"
                fighter.nav.command_target_ship_id = target_ship_id
                fighter.nav.command_target_structure_id = target_beacon_id
                fighter.nav.command_range_m = 0.0
            else:
                fighter.nav.warp.phase = "align"
                fighter.nav.warp.target_ship_id = target_ship_id
                fighter.nav.warp.target_beacon_id = target_beacon_id
                fighter.nav.warp.target_position = Vector2(target_position.x, target_position.y)
                fighter.nav.warp.align_elapsed = 0.0
                fighter.nav.warp.origin = None
                fighter.nav.warp.destination = None
                fighter.nav.warp.warp_distance_m = 0.0
                fighter.nav.warp.warp_duration = 0.0
                fighter.nav.warp.warp_elapsed = 0.0
                fighter.nav.warp.capacitor_cost = 0.0
                fighter.nav.warp.bubble_immune_snapshot = False
                fighter.nav.warp.interdiction_snapshots_captured = False
                fighter.nav.warp.interdiction_snapshots = tuple()
            fighter.state = "moving" if not fighter.target_id else "engaging"
            changed += 1
        return changed

    def set_squad_drone_target(self, world: WorldState, team: Team, squad_id: str, target_id: str | None) -> bool:
        target_key = str(target_id or "").strip()
        target = world.combat_entity(target_key) if target_key else None
        if target_key and (target is None or not target.vital.alive or target.team == team):
            return False
        changed = False
        command_at = float(world.now)
        for owner in self._squad_members(world, team, squad_id):
            control = owner.deployable_control
            if not target_key:
                control.pending_drone_attack_target_id = None
                control.pending_drone_attack_command_at = 0.0
                self._accept_owner_drone_command(world, owner, None, command_at)
                changed = True
            elif target is not None:
                if not self._same_system(owner, target):
                    control.pending_drone_attack_target_id = None
                    control.pending_drone_attack_command_at = 0.0
                    continue
                control.pending_drone_attack_target_id = target_key
                control.pending_drone_attack_command_at = command_at
                accepted = self._try_accept_owner_drone_command(world, owner, target, command_at)
                changed = changed or accepted or self._pending_lock_still_active(owner, target.ship_id)
        return changed

    def set_squad_fighter_target(self, world: WorldState, team: Team, squad_id: str, target_id: str | None) -> bool:
        target_key = str(target_id or "").strip()
        target = world.combat_entity(target_key) if target_key else None
        if target_key and (target is None or not target.vital.alive or target.team == team):
            return False
        changed = False
        command_at = float(world.now)
        for owner in self._squad_members(world, team, squad_id):
            control = owner.deployable_control
            if not target_key:
                control.pending_fighter_attack_target_id = None
                control.pending_fighter_attack_command_at = 0.0
                self._accept_owner_fighter_command(world, owner, None, command_at)
                changed = True
            elif target is not None:
                if not self._same_system(owner, target):
                    control.pending_fighter_attack_target_id = None
                    control.pending_fighter_attack_command_at = 0.0
                    continue
                control.pending_fighter_attack_target_id = target_key
                control.pending_fighter_attack_command_at = command_at
                accepted = self._try_accept_owner_fighter_command(world, owner, target, command_at)
                changed = changed or accepted or self._pending_lock_still_active(owner, target.ship_id)
        return changed

    def activate_fighter_ability(self, world: WorldState, team: Team, fighter_squad_id: str, ability_id: str) -> int:
        squad = str(fighter_squad_id or "").strip()
        ability_key = str(ability_id or "").strip()
        if not squad or not ability_key:
            return 0
        changed = 0
        for fighter in world.fighters.values():
            if fighter.team != team or fighter.squad_id != squad or not fighter.vital.alive or not fighter.connected:
                continue
            ability = next((item for item in fighter.definition.abilities if item.ability_id == ability_key), None)
            if ability is None:
                continue
            if not self._fighter_manual_ability_ready(world, fighter, ability):
                continue
            fighter.pending_manual_abilities.add(ability_key)
            changed += 1
        return changed

    def _fighter_manual_ability_ready(self, world: WorldState, fighter: FighterEntity, ability: FighterAbilityProfile) -> bool:
        ability_id = str(ability.ability_id)
        if ability_id in fighter.pending_manual_abilities:
            return False
        if ability.kind == "mwd":
            return fighter.mwd_active_timer <= 0.0 and fighter.mwd_cooldown_timer <= 0.0
        if max(0.0, float(fighter.ability_cycle_timers.get(ability_id, 0.0) or 0.0)) > 0.0:
            return False
        if max(0.0, float(fighter.ability_reload_timers.get(ability_id, 0.0) or 0.0)) > 0.0:
            return False
        if ability.ammo_capacity > 0 and int(fighter.ability_ammo_remaining.get(ability_id, ability.ammo_capacity)) <= 0:
            return False
        target = self._target_for_asset(world, fighter)
        if target is None or target.ship_id not in fighter.combat.lock_targets:
            return False
        distance = fighter.nav.position.distance_to(target.nav.position)
        max_range = max(
            ability.optimal_range_m + max(0.0, ability.falloff_m) * 3.0,
            ability.ewar.optimal_range_m + max(0.0, ability.ewar.falloff_m) * 3.0,
        )
        return max_range <= 0.0 or distance <= max_range

    def clear_fighter_squad_target(self, world: WorldState, team: Team, fighter_squad_id: str) -> int:
        squad = str(fighter_squad_id or "").strip()
        if not squad:
            return 0
        changed = 0
        owner_ids: set[str] = set()
        for fighter in world.fighters.values():
            if fighter.team != team or fighter.squad_id != squad or not fighter.vital.alive:
                continue
            owner_ids.add(fighter.owner_ship_id)
            if fighter.target_id is not None:
                changed += 1
            fighter.target_id = None
            fighter.target_command_at = 0.0
            if fighter.state == "engaging":
                fighter.state = "idle"
        for owner_id in owner_ids:
            owner = world.ships.get(owner_id)
            if owner is None:
                continue
            owner.deployable_control.fighter_attack_target_id = None
            owner.deployable_control.fighter_attack_command_at = 0.0
            owner.deployable_control.pending_fighter_attack_target_id = None
            owner.deployable_control.pending_fighter_attack_command_at = 0.0
        return changed

    def _candidate_target(self, world: WorldState, team: Team, target_id: str | None, source=None):
        target_key = str(target_id or "").strip()
        if not target_key:
            return None
        target = world.combat_entity(target_key)
        if target is None or not target.vital.alive or target.team == team:
            return None
        if source is not None and not self._same_system(source, target):
            return None
        return target

    def _owner_lock_ready_for_command(self, world: WorldState, owner, target) -> bool:
        if owner is None or target is None or not owner.vital.alive or not target.vital.alive or target.team == owner.team:
            return False
        if not self._same_system(owner, target):
            return False
        return bool(
            self.combat._ensure_target_lock(
                world,
                owner,
                target.ship_id,
                target,
                lock_context="deployable_command_lock",
                now=float(world.now),
            )
        )

    def _pending_lock_still_active(self, owner, target_id: str | None) -> bool:
        target_key = str(target_id or "").strip()
        if not target_key:
            return False
        return target_key in owner.combat.lock_timers or target_key in owner.combat.lock_deadlines

    def _accept_owner_drone_command(self, world: WorldState, owner, target_id: str | None, command_at: float) -> None:
        target_key = str(target_id or "").strip() or None
        target = world.combat_entity(target_key) if target_key else None
        control = owner.deployable_control
        control.drone_attack_target_id = target_key
        control.drone_attack_command_at = float(command_at)
        control.pending_drone_attack_target_id = None
        control.pending_drone_attack_command_at = 0.0
        for drone in world.drones.values():
            if drone.owner_ship_id == owner.ship_id and drone.vital.alive and drone.connected:
                if not self._same_system(drone, owner):
                    continue
                if target is not None and not self._same_system(drone, target):
                    continue
                drone.target_id = target_key
                drone.target_command_at = float(command_at) if target_key else 0.0
                if target_key:
                    drone.state = "engaging"

    def _apply_owner_drone_focus(self, world: WorldState, owner, target_id: str, command_at: float) -> None:
        target_key = str(target_id or "").strip()
        if not target_key:
            return
        target = world.combat_entity(target_key)
        if target is None or not self._same_system(owner, target):
            return
        for drone in world.drones.values():
            if drone.owner_ship_id == owner.ship_id and drone.vital.alive and drone.connected:
                if not self._same_system(drone, owner) or not self._same_system(drone, target):
                    continue
                drone.target_id = target_key
                drone.target_command_at = float(command_at)
                drone.state = "engaging"

    def _accept_owner_fighter_command(self, world: WorldState, owner, target_id: str | None, command_at: float) -> None:
        target_key = str(target_id or "").strip() or None
        target = world.combat_entity(target_key) if target_key else None
        control = owner.deployable_control
        control.fighter_attack_target_id = target_key
        control.fighter_attack_command_at = float(command_at)
        control.pending_fighter_attack_target_id = None
        control.pending_fighter_attack_command_at = 0.0
        for fighter in world.fighters.values():
            if fighter.owner_ship_id == owner.ship_id and fighter.vital.alive and fighter.connected:
                if not self._same_system(fighter, owner):
                    continue
                if target is not None and not self._same_system(fighter, target):
                    continue
                fighter.target_id = target_key
                fighter.target_command_at = float(command_at) if target_key else 0.0
                if target_key:
                    fighter.state = "engaging"

    def _apply_owner_fighter_focus(self, world: WorldState, owner, target_id: str, command_at: float) -> None:
        target_key = str(target_id or "").strip()
        if not target_key:
            return
        target = world.combat_entity(target_key)
        if target is None or not self._same_system(owner, target):
            return
        for fighter in world.fighters.values():
            if fighter.owner_ship_id == owner.ship_id and fighter.vital.alive and fighter.connected:
                if not self._same_system(fighter, owner) or not self._same_system(fighter, target):
                    continue
                fighter.target_id = target_key
                fighter.target_command_at = float(command_at)
                fighter.state = "engaging"

    def _try_accept_owner_drone_command(self, world: WorldState, owner, target, command_at: float) -> bool:
        if not self._same_system(owner, target):
            owner.deployable_control.pending_drone_attack_target_id = None
            owner.deployable_control.pending_drone_attack_command_at = 0.0
            self.combat._drop_lock_target(owner, target.ship_id)
            return False
        if self._owner_lock_ready_for_command(world, owner, target):
            self._accept_owner_drone_command(world, owner, target.ship_id, command_at)
            return True
        if not self._pending_lock_still_active(owner, target.ship_id):
            owner.deployable_control.pending_drone_attack_target_id = None
            owner.deployable_control.pending_drone_attack_command_at = 0.0
        return False

    def _try_accept_owner_fighter_command(self, world: WorldState, owner, target, command_at: float) -> bool:
        if not self._same_system(owner, target):
            owner.deployable_control.pending_fighter_attack_target_id = None
            owner.deployable_control.pending_fighter_attack_command_at = 0.0
            self.combat._drop_lock_target(owner, target.ship_id)
            return False
        if self._owner_lock_ready_for_command(world, owner, target):
            self._accept_owner_fighter_command(world, owner, target.ship_id, command_at)
            return True
        if not self._pending_lock_still_active(owner, target.ship_id):
            owner.deployable_control.pending_fighter_attack_target_id = None
            owner.deployable_control.pending_fighter_attack_command_at = 0.0
        return False

    def _process_pending_owner_commands(self, world: WorldState, owner) -> None:
        control = owner.deployable_control
        pending_drone = str(control.pending_drone_attack_target_id or "").strip()
        if pending_drone:
            target = self._candidate_target(world, owner.team, pending_drone, source=owner)
            if target is None:
                control.pending_drone_attack_target_id = None
                control.pending_drone_attack_command_at = 0.0
            else:
                self._try_accept_owner_drone_command(world, owner, target, control.pending_drone_attack_command_at)

        pending_fighter = str(control.pending_fighter_attack_target_id or "").strip()
        if pending_fighter:
            target = self._candidate_target(world, owner.team, pending_fighter, source=owner)
            if target is None:
                control.pending_fighter_attack_target_id = None
                control.pending_fighter_attack_command_at = 0.0
            else:
                self._try_accept_owner_fighter_command(world, owner, target, control.pending_fighter_attack_command_at)

    def _first_valid_queue_target(self, world: WorldState, team: Team, queue: list[str] | tuple[str, ...] | None, source=None) -> str | None:
        for target_id in queue or []:
            target_key = str(target_id or "").strip()
            target = self._candidate_target(world, team, target_key, source=source)
            if target is not None:
                return target_key
        return None

    def _resolve_owner_drone_focus(self, world: WorldState, owner) -> None:
        explicit_id = str(getattr(owner.deployable_control, "drone_attack_target_id", "") or "").strip()
        if explicit_id:
            explicit = self._candidate_target(world, owner.team, explicit_id, source=owner)
            if explicit is not None:
                return
            owner.deployable_control.drone_attack_target_id = None
            owner.deployable_control.drone_attack_command_at = 0.0
        if str(getattr(owner.deployable_control, "pending_drone_attack_target_id", "") or "").strip():
            return
        focus_key = self._focus_key(owner.team, owner.squad_id)
        target_id = self._first_valid_queue_target(world, owner.team, world.squad_focus_queues.get(focus_key, []), source=owner)
        if not target_id:
            return
        target = world.combat_entity(target_id)
        command_at = float(world.squad_focus_updated_at.get(focus_key, world.now) or world.now)
        if target is not None and self._owner_lock_ready_for_command(world, owner, target):
            self._apply_owner_drone_focus(world, owner, target_id, command_at)

    def _latest_fighter_command_candidate(self, world: WorldState, fighter: FighterEntity, owner) -> tuple[str | None, float, str]:
        candidates: list[tuple[str, float, str]] = []
        control = owner.deployable_control
        explicit = self._candidate_target(world, owner.team, control.fighter_attack_target_id, source=owner)
        if explicit is not None:
            candidates.append((explicit.ship_id, float(control.fighter_attack_command_at or 0.0), "command"))

        fighter_focus_key = self._focus_key(fighter.team, fighter.squad_id)
        fighter_focus = self._first_valid_queue_target(world, fighter.team, world.squad_focus_queues.get(fighter_focus_key, []), source=fighter)
        if fighter_focus:
            candidates.append((fighter_focus, float(world.squad_focus_updated_at.get(fighter_focus_key, world.now) or world.now), "command"))

        if candidates:
            return max(candidates, key=lambda item: item[1])

        mother_focus_key = self._focus_key(owner.team, getattr(fighter, "owner_squad_id", "") or owner.squad_id)
        mother_focus = self._first_valid_queue_target(world, owner.team, world.squad_focus_queues.get(mother_focus_key, []), source=owner)
        if mother_focus:
            return mother_focus, float(world.squad_focus_updated_at.get(mother_focus_key, world.now) or world.now), "mother_focus"
        return None, 0.0, ""

    def _resolve_fighter_command(self, world: WorldState, fighter: FighterEntity, owner) -> None:
        target_id, command_at, source = self._latest_fighter_command_candidate(world, fighter, owner)
        if not target_id:
            return
        if target_id == fighter.target_id and command_at <= float(fighter.target_command_at or 0.0):
            return
        target = world.combat_entity(target_id)
        if target is None:
            return
        if not self._same_system(fighter, target):
            return
        if source == "mother_focus":
            if self._owner_lock_ready_for_command(world, owner, target):
                self._apply_owner_fighter_focus(world, owner, target_id, command_at)
        else:
            self._try_accept_owner_fighter_command(world, owner, target, command_at)

    def _target_for_asset(self, world: WorldState, asset) -> object | None:
        target_id = str(getattr(asset, "target_id", "") or "").strip()
        target = world.combat_entity(target_id) if target_id else None
        if (
            target is not None
            and target.vital.alive
            and target.team != asset.team
            and self._same_system(asset, target)
            and self.combat._can_target_under_ecm(asset, target.ship_id, float(world.now))
        ):
            return target
        asset.target_id = None
        return None

    def _fighter_target_lock_ready(self, world: WorldState, fighter: FighterEntity, target) -> bool:
        target_id = str(getattr(target, "ship_id", "") or "").strip()
        if not target_id:
            return False
        if not self._same_system(fighter, target):
            return False
        return bool(
            self.combat._ensure_target_lock(
                world,
                fighter,
                target_id,
                target,
                lock_context="fighter_target_lock",
                now=float(world.now),
            )
        )

    @staticmethod
    def _same_system(a, b) -> bool:
        return DeployableSystem._entity_system_id(a) == DeployableSystem._entity_system_id(b)

    def _asset_attack_orbit_range(self, asset) -> float:
        if isinstance(asset, DroneEntity):
            ranges = [
                float(getattr(asset.definition, "optimal_range_m", 0.0) or 0.0),
                float(getattr(getattr(asset.definition, "ewar", None), "optimal_range_m", 0.0) or 0.0),
                float(getattr(asset.definition, "orbit_range_m", 0.0) or 0.0),
            ]
            return max(0.0, next((value for value in ranges if value > 0.0), 0.0))
        if isinstance(asset, FighterEntity):
            for ability in asset.definition.abilities:
                if ability.kind == "normal_attack" and ability.optimal_range_m > 0.0:
                    return max(0.0, float(ability.optimal_range_m))
            return max(0.0, float(getattr(asset.definition, "orbit_range_m", 0.0) or 0.0))
        return 0.0

    def _asset_owner_orbit_range(self, asset) -> float:
        if isinstance(asset, DroneEntity):
            if asset.definition.is_sentry:
                return 0.0
            return max(500.0, float(asset.definition.orbit_range_m or 0.0), self.RECOVERY_RANGE_M)
        if isinstance(asset, FighterEntity):
            return max(1_000.0, float(asset.definition.orbit_range_m or 0.0), self.RECOVERY_RANGE_M)
        return self.RECOVERY_RANGE_M

    def _set_asset_target_navigation(self, world: WorldState, asset, target, orbit_range_m: float) -> None:
        if not asset.vital.alive:
            return
        if str(getattr(asset.nav.warp, "phase", "idle") or "idle") == "warp":
            return
        if target is None or not target.vital.alive or not self._same_system(asset, target):
            self.movement._clear_navigation_command(asset)
            asset.nav.velocity = Vector2(0.0, 0.0)
            return
        distance = asset.nav.position.distance_to(target.nav.position)
        if isinstance(asset, FighterEntity) and distance >= self.movement.MIN_WARP_DISTANCE_M:
            if str(asset.nav.warp.phase or "idle") == "idle" and not self.movement._ship_is_scrammed(asset):
                asset.nav.warp.phase = "align"
                asset.nav.warp.target_ship_id = target.ship_id
                asset.nav.warp.target_position = Vector2(target.nav.position.x, target.nav.position.y)
                asset.nav.warp.align_elapsed = 0.0
                asset.nav.warp.origin = None
                asset.nav.warp.destination = None
                asset.nav.warp.warp_distance_m = 0.0
                asset.nav.warp.warp_duration = 0.0
                asset.nav.warp.warp_elapsed = 0.0
                asset.nav.warp.capacitor_cost = 0.0
                asset.nav.warp.bubble_immune_snapshot = False
                asset.nav.warp.interdiction_snapshots_captured = False
                asset.nav.warp.interdiction_snapshots = tuple()

        if str(getattr(asset.nav.warp, "phase", "idle") or "idle") == "warp":
            return

        asset.nav.command_mode = "orbit"
        asset.nav.command_target_ship_id = target.ship_id
        asset.nav.command_target_structure_id = None
        asset.nav.command_range_m = max(0.0, float(orbit_range_m or 0.0))
        asset.nav.command_target = Vector2(target.nav.position.x, target.nav.position.y)

    def _set_asset_recall_navigation(self, world: WorldState, asset, owner) -> None:
        if not asset.vital.alive:
            return
        if str(getattr(asset.nav.warp, "phase", "idle") or "idle") == "warp":
            return
        if owner is None or not owner.vital.alive:
            self.movement._clear_navigation_command(asset)
            asset.nav.velocity = Vector2(0.0, 0.0)
            return
        if not self._same_system(asset, owner):
            self.movement._clear_navigation_command(asset)
            asset.nav.velocity = Vector2(0.0, 0.0)
            return
        distance = asset.nav.position.distance_to(owner.nav.position)
        if isinstance(asset, FighterEntity) and distance >= self.movement.MIN_WARP_DISTANCE_M:
            if str(asset.nav.warp.phase or "idle") == "idle" and not self.movement._ship_is_scrammed(asset):
                asset.nav.warp.phase = "align"
                asset.nav.warp.target_ship_id = owner.ship_id
                asset.nav.warp.target_position = Vector2(owner.nav.position.x, owner.nav.position.y)
                asset.nav.warp.align_elapsed = 0.0
                asset.nav.warp.origin = None
                asset.nav.warp.destination = None
                asset.nav.warp.warp_distance_m = 0.0
                asset.nav.warp.warp_duration = 0.0
                asset.nav.warp.warp_elapsed = 0.0
                asset.nav.warp.capacitor_cost = 0.0
                asset.nav.warp.bubble_immune_snapshot = False
                asset.nav.warp.interdiction_snapshots_captured = False
                asset.nav.warp.interdiction_snapshots = tuple()
        asset.nav.command_mode = "approach"
        asset.nav.command_target_ship_id = owner.ship_id
        asset.nav.command_target_structure_id = None
        asset.nav.command_range_m = self.RECOVERY_RANGE_M
        asset.nav.command_target = Vector2(owner.nav.position.x, owner.nav.position.y)

    def _advance_asset_navigation(self, world: WorldState, asset, dt: float) -> None:
        if not asset.vital.alive:
            return
        if str(getattr(asset.nav.warp, "phase", "idle") or "idle") == "warp":
            self.movement._advance_in_warp(world, asset, dt)
            return
        self.movement._prepare_warp_alignment(world, asset)
        if str(getattr(asset.nav.warp, "phase", "idle") or "idle") == "warp":
            self.movement._advance_in_warp(world, asset, dt)
            return
        displacement = self.movement._update_velocity_with_inertia(world, asset, dt)
        next_pos = asset.nav.position + displacement
        system_center, system_radius = self.movement._system_center_and_radius(world, asset)
        relative_next = next_pos - system_center
        if relative_next.length() > system_radius:
            next_pos = system_center + relative_next.normalized() * system_radius
            asset.nav.velocity = Vector2(0.0, 0.0)
        asset.nav.position = next_pos
        self.movement._finalize_warp_alignment(world, asset, dt)

    def _drive_asset_to_target(self, world: WorldState, asset, target, orbit_range_m: float, dt: float) -> None:
        self._set_asset_target_navigation(world, asset, target, orbit_range_m)
        self._advance_asset_navigation(world, asset, dt)

    def _drive_asset_to_owner(self, world: WorldState, asset, owner, dt: float) -> bool:
        if owner is None or not owner.vital.alive:
            asset.state = "idle"
            self.movement._clear_navigation_command(asset)
            return False
        if not self._same_system(asset, owner):
            asset.state = "idle"
            self.movement._clear_navigation_command(asset)
            return False
        if self._asset_recovery_ready(asset, owner):
            return True
        if isinstance(asset, DroneEntity) and asset.definition.is_sentry:
            asset.state = "idle"
            self.movement._clear_navigation_command(asset)
            return False
        self._set_asset_recall_navigation(world, asset, owner)
        self._advance_asset_navigation(world, asset, dt)
        return self._asset_recovery_ready(asset, owner)

    def _asset_recovery_ready(self, asset, owner) -> bool:
        if owner is None or not owner.vital.alive or not self._same_system(asset, owner):
            return False
        distance = asset.nav.position.distance_to(owner.nav.position)
        own_radius = max(0.0, float(getattr(owner.nav, "radius", 0.0) or 0.0))
        asset_radius = max(0.0, float(getattr(asset.nav, "radius", 0.0) or 0.0))
        edge_distance = max(0.0, distance - own_radius - asset_radius)
        return edge_distance <= self.RECOVERY_RANGE_M

    def _update_asset_connection(self, world: WorldState, asset, owner) -> bool:
        if owner is None or not owner.vital.alive:
            asset.vital.alive = False
            asset.vital.shield = 0.0
            asset.vital.armor = 0.0
            asset.vital.structure = 0.0
            asset.connected = False
            asset.target_id = None
            asset.state = "destroyed"
            self.movement._clear_navigation_command(asset)
            asset.nav.velocity = Vector2(0.0, 0.0)
            return False

        same_system = self._same_system(asset, owner)
        distance = asset.nav.position.distance_to(owner.nav.position) if same_system else float("inf")
        connected = same_system and distance <= self.DISCONNECT_RANGE_M
        if not connected:
            asset.connected = False
            asset.target_id = None
            asset.target_command_at = 0.0
            asset.state = "disconnected"
            if isinstance(asset, FighterEntity):
                asset.pending_manual_abilities.clear()
            self.movement._clear_navigation_command(asset)
            asset.nav.velocity = Vector2(0.0, 0.0)
            return False

        if not getattr(asset, "connected", True):
            asset.connected = True
            asset.state = "idle"
            asset.target_id = None
            asset.target_command_at = 0.0
        return True

    def _damage_factor_turret(self, source, target, tracking: float, optimal: float, falloff: float) -> float:
        relative_velocity = source.nav.velocity - target.nav.velocity
        radial = (target.nav.position - source.nav.position).normalized()
        tangential = Vector2(-radial.y, radial.x)
        transversal = abs(relative_velocity.x * tangential.x + relative_velocity.y * tangential.y)
        distance = max(0.0, source.nav.position.distance_to(target.nav.position))
        chance = self.combat.pyfa.turret_chance_to_hit(
            tracking=max(0.0, float(tracking)),
            optimal_sig=40.0,
            distance=distance,
            optimal=max(0.0, float(optimal)),
            falloff=max(0.0, float(falloff)),
            transversal_speed=transversal,
            target_sig=max(1.0, float(target.profile.sig_radius or 1.0)),
            attacker_radius=max(0.0, float(source.nav.radius or 0.0)),
            target_radius=max(0.0, float(target.nav.radius or 0.0)),
        )
        return max(0.0, float(self.combat.pyfa.turret_damage_multiplier(chance)))

    @staticmethod
    def _damage_factor_missile(target, explosion_radius: float, explosion_velocity: float, drf: float) -> float:
        radius = max(0.0, float(explosion_radius or 0.0))
        if radius <= 0.0:
            return 1.0
        sig_factor = max(0.0, float(target.profile.sig_radius or 0.0)) / max(1.0, radius)
        target_speed = max(0.0, target.nav.velocity.length())
        velocity = max(0.0, float(explosion_velocity or 0.0))
        if target_speed <= 1e-9 or velocity <= 0.0:
            velocity_factor = 1.0
        else:
            velocity_factor = ((sig_factor * velocity) / max(1.0, target_speed)) ** max(0.1, float(drf or 0.5))
        return max(0.0, min(1.0, sig_factor, velocity_factor))

    def _range_factor(self, optimal: float, falloff: float, distance: float) -> float:
        if optimal <= 0.0 and falloff <= 0.0:
            return 1.0
        return max(0.0, min(1.0, float(self.combat.pyfa.turret_range_factor(optimal, falloff, distance))))

    def _apply_ewar(self, world: WorldState, source, target, ewar: DeployableEwarProfile, source_id: str) -> None:
        if not ewar.has_effect:
            return
        if not self._same_system(source, target):
            return
        distance = source.nav.position.distance_to(target.nav.position)
        if ewar.optimal_range_m > 0.0 or ewar.falloff_m > 0.0:
            if distance > ewar.optimal_range_m + max(0.0, ewar.falloff_m) * 3.0:
                return
        strength = self._range_factor(ewar.optimal_range_m, ewar.falloff_m, distance)
        if strength <= 0.0:
            return

        if ewar.speed_factor_mult < 0.999:
            target.profile.max_speed = max(1.0, target.profile.max_speed * (1.0 + (ewar.speed_factor_mult - 1.0) * strength))
            target.nav.max_speed = min(target.nav.max_speed, target.profile.max_speed)
        if abs(ewar.signature_radius_bonus_pct) > 1e-9:
            target.profile.sig_radius = max(1.0, target.profile.sig_radius * (1.0 + ewar.signature_radius_bonus_pct * strength / 100.0))
        if abs(ewar.scan_resolution_bonus_pct) > 1e-9:
            target.profile.scan_resolution = max(1.0, target.profile.scan_resolution * (1.0 + ewar.scan_resolution_bonus_pct * strength / 100.0))
        if abs(ewar.max_target_range_bonus_pct) > 1e-9:
            target.profile.max_target_range = max(1_000.0, target.profile.max_target_range * (1.0 + ewar.max_target_range_bonus_pct * strength / 100.0))
        if abs(ewar.tracking_bonus_pct) > 1e-9:
            target.profile.tracking = max(0.0, target.profile.tracking * (1.0 + ewar.tracking_bonus_pct * strength / 100.0))
        if abs(ewar.optimal_bonus_pct) > 1e-9:
            target.profile.optimal = max(0.0, target.profile.optimal * (1.0 + ewar.optimal_bonus_pct * strength / 100.0))
        if abs(ewar.falloff_bonus_pct) > 1e-9:
            target.profile.falloff = max(0.0, target.profile.falloff * (1.0 + ewar.falloff_bonus_pct * strength / 100.0))
        if ewar.warp_disrupt_strength > 0.0:
            target.profile.warp_scramble_status = max(target.profile.warp_scramble_status, ewar.warp_disrupt_strength * strength)
        if ewar.capacitor_neutralized > 0.0:
            resistance = max(0.0, float(getattr(target.profile, "energy_warfare_resistance", 1.0) or 1.0))
            target.vital.cap = max(0.0, target.vital.cap - ewar.capacitor_neutralized * strength * resistance)
        self._apply_ecm(world, source, target, ewar, source_id, strength)

    def _apply_ecm(self, world: WorldState, source, target, ewar: DeployableEwarProfile, source_id: str, strength_factor: float) -> None:
        strengths = {
            "gravimetric": ewar.ecm_gravimetric,
            "ladar": ewar.ecm_ladar,
            "magnetometric": ewar.ecm_magnetometric,
            "radar": ewar.ecm_radar,
        }
        if max(strengths.values(), default=0.0) <= 0.0:
            return
        sensor_type, sensor_strength, has_known = self.combat._target_sensor_type_and_strength(target.profile)
        module_strength = max(0.0, strengths.get(sensor_type, 0.0))
        if module_strength <= 0.0 and not has_known:
            module_strength = max(strengths.values(), default=0.0)
        if module_strength <= 0.0 or sensor_strength <= 0.0:
            return
        chance = max(0.0, min(1.0, module_strength * max(0.0, strength_factor) / max(1e-9, sensor_strength)))
        source.combat.ecm_last_attempt_target = target.ship_id
        source.combat.ecm_last_attempt_module = source_id
        source.combat.ecm_last_attempt_success = False
        source.combat.ecm_last_attempt_chance = chance
        source.combat.ecm_last_attempt_at = float(world.now)
        source.combat.ecm_last_attempt_target_by_module[source_id] = target.ship_id
        source.combat.ecm_last_attempt_success_by_module[source_id] = False
        source.combat.ecm_last_attempt_at_by_module[source_id] = float(world.now)
        if random.random() >= chance:
            return
        source.combat.ecm_last_attempt_success = True
        source.combat.ecm_last_attempt_success_by_module[source_id] = True
        jam_until = float(world.now) + max(0.1, float(ewar.duration_s or 5.0))
        target.combat.ecm_jam_sources[source.ship_id] = max(
            float(target.combat.ecm_jam_sources.get(source.ship_id, 0.0) or 0.0),
            jam_until,
        )
        self.combat._enforce_ecm_restrictions(target, float(world.now))
        self._queue_event(
            "ecm_jam_applied",
            world,
            source=source.ship_id,
            target=target.ship_id,
            module=source_id,
            sensor_type=sensor_type,
            chance=chance,
            duration_s=max(0.1, float(ewar.duration_s or 5.0)),
        )

    def _drone_attack(self, world: WorldState, drone: DroneEntity, target, dt: float) -> None:
        if not self._same_system(drone, target):
            drone.target_id = None
            return
        if drone.definition.damage.total <= 0.0 and not drone.definition.ewar.has_effect:
            return
        drone.cycle_timer = max(0.0, float(drone.cycle_timer or 0.0) - max(0.0, dt))
        drone.ewar_cycle_timer = max(0.0, float(drone.ewar_cycle_timer or 0.0) - max(0.0, dt))
        distance = drone.nav.position.distance_to(target.nav.position)
        max_range = max(
            drone.definition.optimal_range_m + max(0.0, drone.definition.falloff_m) * 3.0,
            drone.definition.control_range_m,
            drone.definition.ewar.optimal_range_m + max(0.0, drone.definition.ewar.falloff_m) * 3.0,
        )
        if max_range > 0.0 and distance > max_range:
            return
        if drone.definition.ewar.has_effect and drone.ewar_cycle_timer <= 0.0:
            self._apply_ewar(world, drone, target, drone.definition.ewar, f"{drone.ship_id}:ewar")
            drone.ewar_cycle_timer = max(0.1, drone.definition.ewar.cycle_time_s)
        if drone.definition.damage.total <= 0.0 or drone.cycle_timer > 0.0:
            return
        damage_factor = self._damage_factor_turret(
            drone,
            target,
            drone.definition.tracking,
            drone.definition.optimal_range_m,
            drone.definition.falloff_m,
        )
        self.combat._apply_direct_damage(
            world,
            source=drone,
            target=target,
            target_profile=target.profile,
            damage=self._damage_tuple(drone.definition.damage),
            damage_factor=damage_factor,
            module_id=f"{drone.ship_id}:attack",
        )
        drone.cycle_timer = max(0.1, drone.definition.cycle_time_s)

    def _activate_fighter_mwd(self, fighter: FighterEntity, ability: FighterAbilityProfile) -> bool:
        if fighter.mwd_active_timer > 0.0 or fighter.mwd_cooldown_timer > 0.0:
            return False
        fighter.mwd_active_timer = max(0.1, ability.duration_s or ability.cycle_time_s or 10.0)
        fighter.mwd_cooldown_timer = max(fighter.mwd_active_timer, ability.cooldown_s or ability.cycle_time_s or fighter.mwd_active_timer)
        return True

    def _activate_pending_fighter_mwd(self, fighter: FighterEntity) -> None:
        if not fighter.pending_manual_abilities:
            return
        for ability in fighter.definition.abilities:
            if ability.kind != "mwd" or ability.ability_id not in fighter.pending_manual_abilities:
                continue
            if self._activate_fighter_mwd(fighter, ability):
                fighter.pending_manual_abilities.discard(ability.ability_id)
            return

    def _advance_fighter_timers(self, fighter: FighterEntity, dt: float) -> None:
        fighter.mwd_active_timer = max(0.0, float(fighter.mwd_active_timer or 0.0) - max(0.0, dt))
        fighter.mwd_cooldown_timer = max(0.0, float(fighter.mwd_cooldown_timer or 0.0) - max(0.0, dt))
        for ability_id in list(fighter.ability_cycle_timers.keys()):
            fighter.ability_cycle_timers[ability_id] = max(
                0.0,
                float(fighter.ability_cycle_timers.get(ability_id, 0.0) or 0.0) - max(0.0, dt),
            )
        for ability_id in list(fighter.ability_reload_timers.keys()):
            remaining = max(0.0, float(fighter.ability_reload_timers.get(ability_id, 0.0) or 0.0) - max(0.0, dt))
            if remaining > 0.0:
                fighter.ability_reload_timers[ability_id] = remaining
                continue
            fighter.ability_reload_timers.pop(ability_id, None)
            ability = next((item for item in fighter.definition.abilities if item.ability_id == ability_id), None)
            if ability is not None and ability.ammo_capacity > 0:
                fighter.ability_ammo_remaining[ability_id] = int(ability.ammo_capacity)

    def _apply_fighter_speed(self, fighter: FighterEntity) -> None:
        base = max(1.0, float(fighter.profile.max_speed or fighter.definition.max_velocity or 1.0))
        if fighter.mwd_active_timer <= 0.0:
            fighter.nav.max_speed = base
            return
        ability = next((item for item in fighter.definition.abilities if item.kind == "mwd"), None)
        bonus = max(0.0, float(getattr(ability, "speed_bonus_pct", 0.0) or 0.0)) if ability is not None else 0.0
        factor = 1.0 + bonus / 100.0
        fighter.profile.max_speed = max(base, base * factor)
        fighter.nav.max_speed = fighter.profile.max_speed

    def _fighter_attack(self, world: WorldState, fighter: FighterEntity, target) -> None:
        if not self._same_system(fighter, target):
            fighter.target_id = None
            return
        for ability in fighter.definition.abilities:
            if ability.kind == "mwd":
                continue
            manual = ability.kind != "normal_attack"
            if manual and ability.ability_id not in fighter.pending_manual_abilities:
                continue
            timer = max(0.0, float(fighter.ability_cycle_timers.get(ability.ability_id, 0.0) or 0.0))
            if timer > 0.0:
                continue
            if max(0.0, float(fighter.ability_reload_timers.get(ability.ability_id, 0.0) or 0.0)) > 0.0:
                continue
            if ability.ammo_capacity > 0:
                ammo_left = int(fighter.ability_ammo_remaining.get(ability.ability_id, ability.ammo_capacity))
                if ammo_left <= 0:
                    fighter.ability_reload_timers[ability.ability_id] = max(0.1, ability.reload_time_s)
                    continue
            distance = fighter.nav.position.distance_to(target.nav.position)
            max_range = max(
                ability.optimal_range_m + max(0.0, ability.falloff_m) * 3.0,
                ability.ewar.optimal_range_m + max(0.0, ability.ewar.falloff_m) * 3.0,
            )
            if max_range > 0.0 and distance > max_range:
                continue
            activated = False
            if ability.ewar.has_effect:
                self._apply_ewar(world, fighter, target, ability.ewar, f"{fighter.ship_id}:{ability.ability_id}")
                activated = True
            if ability.has_damage:
                if ability.explosion_radius > 0.0:
                    damage_factor = self._damage_factor_missile(
                        target,
                        ability.explosion_radius,
                        ability.explosion_velocity,
                        ability.damage_reduction_factor,
                    )
                else:
                    damage_factor = self._damage_factor_turret(
                        fighter,
                        target,
                        ability.tracking,
                        ability.optimal_range_m,
                        ability.falloff_m,
                    )
                self.combat._apply_direct_damage(
                    world,
                    source=fighter,
                    target=target,
                    target_profile=target.profile,
                    damage=self._damage_tuple(ability.damage),
                    damage_factor=damage_factor,
                    module_id=f"{fighter.ship_id}:{ability.ability_id}",
                )
                activated = True
            if manual and not activated:
                fighter.pending_manual_abilities.discard(ability.ability_id)
                fighter.ability_cycle_timers[ability.ability_id] = max(0.1, ability.cycle_time_s)
                continue
            if ability.ammo_capacity > 0:
                fighter.ability_ammo_remaining[ability.ability_id] = max(
                    0,
                    int(fighter.ability_ammo_remaining.get(ability.ability_id, ability.ammo_capacity)) - 1,
                )
                if fighter.ability_ammo_remaining[ability.ability_id] <= 0 and ability.reload_time_s > 0.0:
                    fighter.ability_reload_timers[ability.ability_id] = ability.reload_time_s
            fighter.ability_cycle_timers[ability.ability_id] = max(0.1, ability.cycle_time_s)
            if manual:
                fighter.pending_manual_abilities.discard(ability.ability_id)

    def _queue_event(self, kind: str, world: WorldState, **fields) -> None:
        queue = getattr(self.combat, "_queue_merged_event", None)
        if not callable(queue):
            return
        merge_fields = {key: str(value) for key, value in fields.items() if not isinstance(value, (int, float))}
        sum_fields = {key: float(value) for key, value in fields.items() if isinstance(value, (int, float))}
        try:
            queue(kind, merge_fields=merge_fields, sum_fields=sum_fields)
        except Exception:
            return

    def run_physics(self, world: WorldState, dt: float) -> None:
        for drone in world.drones.values():
            if not drone.vital.alive or not getattr(drone, "connected", True) or drone.definition.is_sentry:
                continue
            self._advance_asset_navigation(world, drone, dt)
        for fighter in world.fighters.values():
            if not fighter.vital.alive or not getattr(fighter, "connected", True):
                continue
            self._advance_asset_navigation(world, fighter, dt)

    def run(self, world: WorldState, dt: float, *, advance_physics: bool = True, apply_effects: bool = True) -> None:
        recovered_drones: list[str] = []
        recovered_fighters: list[str] = []

        for drone in world.drones.values():
            drone.profile = self._drone_profile(drone.definition)
        for fighter in world.fighters.values():
            fighter.profile = self._fighter_profile(fighter.definition)

        for owner in world.ships.values():
            if not owner.vital.alive:
                continue
            if str(getattr(owner.nav, "squad_follow_state", "FORMATION_FOLLOW") or "FORMATION_FOLLOW") in {
                "FOLLOW_LEADER_SYSTEM",
                "WARP_TO_LEADER",
            }:
                continue
            self._process_pending_owner_commands(world, owner)
            self._resolve_owner_drone_focus(world, owner)

        for drone_id, drone in list(world.drones.items()):
            if not drone.vital.alive:
                continue
            owner = world.ships.get(drone.owner_ship_id)
            if not self._update_asset_connection(world, drone, owner):
                continue
            if owner is not None and str(getattr(owner.nav, "squad_follow_state", "FORMATION_FOLLOW") or "FORMATION_FOLLOW") in {
                "FOLLOW_LEADER_SYSTEM",
                "WARP_TO_LEADER",
            }:
                drone.target_id = None
            if drone.state == "recalling":
                if advance_physics:
                    recovered = self._drive_asset_to_owner(world, drone, owner, dt)
                else:
                    recovered = self._asset_recovery_ready(drone, owner)
                    if not recovered and owner is not None and owner.vital.alive and not drone.definition.is_sentry:
                        self._set_asset_recall_navigation(world, drone, owner)
                if apply_effects and recovered:
                    recovered_drones.append(drone_id)
                continue
            target = self._target_for_asset(world, drone)
            if target is None:
                if owner is not None and owner.vital.alive and not drone.definition.is_sentry:
                    drone.state = "guarding"
                    if advance_physics:
                        self._drive_asset_to_target(world, drone, owner, self._asset_owner_orbit_range(drone), dt)
                    else:
                        self._set_asset_target_navigation(world, drone, owner, self._asset_owner_orbit_range(drone))
                else:
                    drone.state = "idle"
                    self.movement._clear_navigation_command(drone)
                continue
            drone.state = "engaging"
            if not drone.definition.is_sentry:
                if advance_physics:
                    self._drive_asset_to_target(world, drone, target, self._asset_attack_orbit_range(drone), dt)
                else:
                    self._set_asset_target_navigation(world, drone, target, self._asset_attack_orbit_range(drone))
            else:
                drone.nav.velocity = Vector2(0.0, 0.0)
            if apply_effects:
                self._drone_attack(world, drone, target, dt)

        for fighter_id, fighter in list(world.fighters.items()):
            if not fighter.vital.alive:
                continue
            owner = world.ships.get(fighter.owner_ship_id)
            if not self._update_asset_connection(world, fighter, owner):
                continue
            if owner is not None and str(getattr(owner.nav, "squad_follow_state", "FORMATION_FOLLOW") or "FORMATION_FOLLOW") in {
                "FOLLOW_LEADER_SYSTEM",
                "WARP_TO_LEADER",
            }:
                fighter.target_id = None
            if apply_effects:
                self._advance_fighter_timers(fighter, dt)
                self._activate_pending_fighter_mwd(fighter)
            if fighter.state == "recalling":
                self._apply_fighter_speed(fighter)
                if advance_physics:
                    recovered = self._drive_asset_to_owner(world, fighter, owner, dt)
                else:
                    recovered = self._asset_recovery_ready(fighter, owner)
                    if not recovered and owner is not None and owner.vital.alive:
                        self._set_asset_recall_navigation(world, fighter, owner)
                if apply_effects and recovered:
                    recovered_fighters.append(fighter_id)
                continue
            if owner is not None and owner.vital.alive:
                self._resolve_fighter_command(world, fighter, owner)
            target = self._target_for_asset(world, fighter)
            if target is None:
                self._apply_fighter_speed(fighter)
                fighter.state = "moving" if fighter.nav.command_target is not None or str(fighter.nav.warp.phase or "idle") != "idle" else "idle"
                continue
            fighter.state = "engaging"
            self._apply_fighter_speed(fighter)
            if apply_effects and self._fighter_target_lock_ready(world, fighter, target):
                self._fighter_attack(world, fighter, target)

        for drone_id in recovered_drones:
            world.drones.pop(drone_id, None)
        for fighter_id in recovered_fighters:
            world.fighters.pop(fighter_id, None)
