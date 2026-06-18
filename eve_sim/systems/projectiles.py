from __future__ import annotations

from .combat_common import *  # noqa: F403


class ProjectilesMixin:
    @staticmethod
    def _projectile_entity_system_id(entity: Any) -> str:
        nav = getattr(entity, "nav", None)
        if nav is not None:
            return str(getattr(nav, "system_id", "") or "")
        return str(getattr(entity, "system_id", "") or "")

    @classmethod
    def _same_projectile_system(cls, projectile: ProjectileEntity, entity: Any) -> bool:
        return str(getattr(projectile, "system_id", "") or "") == cls._projectile_entity_system_id(entity)

    @staticmethod
    def _vector_from_facing_deg(facing_deg: float) -> Vector2:
        radians = math.radians(float(facing_deg or 0.0))
        return Vector2(math.cos(radians), math.sin(radians))

    @staticmethod
    def _projectile_damage_tuple(projectile: ProjectileEntity) -> DamageTuple:
        return (
            max(0.0, float(projectile.damage_em or 0.0)),
            max(0.0, float(projectile.damage_thermal or 0.0)),
            max(0.0, float(projectile.damage_kinetic or 0.0)),
            max(0.0, float(projectile.damage_explosive or 0.0)),
        )

    @staticmethod
    def _effect_damage_tuple(effect) -> DamageTuple:
        return (
            max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
        )

    @staticmethod
    def _projectile_acceleration_time(flight_time: float, mass: float, agility: float) -> float:
        if flight_time <= 0.0 or mass <= 0.0 or agility <= 0.0:
            return 0.0
        return max(0.0, min(float(flight_time), (float(mass) * float(agility)) / 1_000_000.0))

    @classmethod
    def _projectile_max_range(
        cls,
        *,
        max_speed: float,
        flight_time: float,
        acceleration_time: float,
        fallback_range: float,
    ) -> float:
        if max_speed <= 0.0 or flight_time <= 0.0:
            return max(0.0, float(fallback_range or 0.0))
        accel_time = max(0.0, min(float(acceleration_time), float(flight_time)))
        during_acceleration = max_speed * 0.5 * accel_time
        at_full_speed = max_speed * max(0.0, float(flight_time) - accel_time)
        return max(max(0.0, float(fallback_range or 0.0)), during_acceleration + at_full_speed)

    @classmethod
    def _projectile_distance_for_interval(
        cls,
        *,
        max_speed: float,
        acceleration_time: float,
        start_age: float,
        dt: float,
    ) -> float:
        if max_speed <= 0.0 or dt <= 0.0:
            return 0.0
        if acceleration_time <= 1e-9:
            return max_speed * dt

        start = max(0.0, float(start_age))
        end = start + max(0.0, float(dt))
        if start >= acceleration_time:
            return max_speed * (end - start)

        accel_end = min(end, acceleration_time)
        start_speed = max_speed * max(0.0, min(1.0, start / acceleration_time))
        end_speed = max_speed * max(0.0, min(1.0, accel_end / acceleration_time))
        distance = 0.5 * (start_speed + end_speed) * max(0.0, accel_end - start)
        if end > acceleration_time:
            distance += max_speed * (end - acceleration_time)
        return max(0.0, distance)

    @classmethod
    def _projectile_speed_for_age(cls, *, max_speed: float, acceleration_time: float, age: float) -> float:
        if max_speed <= 0.0:
            return 0.0
        if acceleration_time <= 1e-9:
            return max_speed
        return max_speed * max(0.0, min(1.0, float(age) / float(acceleration_time)))

    def _create_projectile_blast(
        self,
        world: WorldState,
        *,
        kind: str,
        position: Vector2,
        radius_m: float,
        duration: float = 1.0,
        system_id: str = "",
    ) -> None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return
        self._projectile_blast_seq += 1
        blast_id = f"blast:{self._projectile_blast_seq}"
        world.projectile_blasts[blast_id] = ProjectileBlast(
            blast_id=blast_id,
            kind=str(kind),
            position=Vector2(position.x, position.y),
            radius_m=radius,
            expires_at=float(world.now) + max(0.1, float(duration)),
            system_id=str(system_id or ""),
        )

    @staticmethod
    def _clamp_projectile_layer_hp(projectile: ProjectileEntity) -> None:
        projectile.shield_max = max(0.0, float(projectile.shield_max or 0.0))
        projectile.armor_max = max(0.0, float(projectile.armor_max or 0.0))
        projectile.structure_max = max(0.0, float(projectile.structure_max or 0.0))
        projectile.shield = max(0.0, min(float(projectile.shield or 0.0), projectile.shield_max))
        projectile.armor = max(0.0, min(float(projectile.armor or 0.0), projectile.armor_max))
        projectile.structure = max(0.0, min(float(projectile.structure or 0.0), projectile.structure_max))
        projectile.alive = (projectile.shield + projectile.armor + projectile.structure) > 1e-6

    def _apply_damage_to_projectile(
        self,
        projectile: ProjectileEntity,
        damage: DamageTuple,
        *,
        damage_factor: float = 1.0,
    ) -> bool:
        scaled_damage = _scale_damage(damage, max(0.0, float(damage_factor)))
        if _sum_damage(scaled_damage) <= 0.0:
            return not projectile.alive

        self._clamp_projectile_layer_hp(projectile)
        remaining = scaled_damage
        layer_specs = (
            (
                "shield",
                float(projectile.shield),
                (
                    float(projectile.shield_resonance_em),
                    float(projectile.shield_resonance_thermal),
                    float(projectile.shield_resonance_kinetic),
                    float(projectile.shield_resonance_explosive),
                ),
            ),
            (
                "armor",
                float(projectile.armor),
                (
                    float(projectile.armor_resonance_em),
                    float(projectile.armor_resonance_thermal),
                    float(projectile.armor_resonance_kinetic),
                    float(projectile.armor_resonance_explosive),
                ),
            ),
            (
                "structure",
                float(projectile.structure),
                (
                    float(projectile.structure_resonance_em),
                    float(projectile.structure_resonance_thermal),
                    float(projectile.structure_resonance_kinetic),
                    float(projectile.structure_resonance_explosive),
                ),
            ),
        )
        new_values = {
            "shield": float(projectile.shield),
            "armor": float(projectile.armor),
            "structure": float(projectile.structure),
        }
        for layer_name, layer_hp, layer_resonances in layer_specs:
            if layer_hp <= 0.0:
                continue
            effective_damage = _layer_effective_damage(remaining, layer_resonances)
            if effective_damage <= 0.0:
                continue
            if effective_damage <= layer_hp:
                new_values[layer_name] = layer_hp - effective_damage
                remaining = (0.0, 0.0, 0.0, 0.0)
                break
            consumed_ratio = max(0.0, min(1.0, layer_hp / effective_damage))
            new_values[layer_name] = 0.0
            remaining = _scale_damage(remaining, 1.0 - consumed_ratio)

        projectile.shield = max(0.0, new_values["shield"])
        projectile.armor = max(0.0, new_values["armor"])
        projectile.structure = max(0.0, new_values["structure"])
        projectile.alive = (projectile.shield + projectile.armor + projectile.structure) > 1e-6
        return not projectile.alive

    def _destroy_projectiles_in_area(
        self,
        world: WorldState,
        *,
        center: Vector2,
        radius_m: float,
        exclude_ids: set[str] | None = None,
        damage: DamageTuple | None = None,
        damage_factor: float = 1.0,
        system_id: str = "",
    ) -> None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return
        excluded = exclude_ids or set()
        source_system_id = str(system_id or "")
        hit_radius = radius + 25.0
        for projectile_id, projectile in list(world.projectiles.items()):
            if projectile_id in excluded:
                continue
            if source_system_id and str(getattr(projectile, "system_id", "") or "") != source_system_id:
                continue
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)
                continue
            if projectile.position.distance_to(center) > hit_radius:
                continue
            if damage is None:
                projectile.alive = False
            else:
                self._apply_damage_to_projectile(
                    projectile,
                    damage,
                    damage_factor=damage_factor,
                )
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)

    def _apply_direct_damage(
        self,
        world: WorldState,
        *,
        source,
        target,
        target_profile: ShipProfile,
        damage: DamageTuple,
        damage_factor: float,
        module_id: str | None = None,
    ) -> float:
        self._clamp_ship_layer_hp(target)
        scaled_damage = _scale_damage(damage, max(0.0, float(damage_factor)))
        total_damage = _sum_damage(scaled_damage)
        if total_damage <= 0.0:
            return 0.0

        shield_before = target.vital.shield
        armor_before = target.vital.armor
        structure_before = target.vital.structure
        alive_before = bool(target.vital.alive)

        target.vital.shield, target.vital.armor, target.vital.structure = _apply_damage_sequence(
            target.vital.shield,
            target.vital.armor,
            target.vital.structure,
            scaled_damage,
            target_profile,
        )
        dirty_layers: list[str] = []
        if abs(target.vital.shield - shield_before) > 1e-6:
            dirty_layers.append("shield")
        if abs(target.vital.armor - armor_before) > 1e-6:
            dirty_layers.append("armor")
        if abs(target.vital.structure - structure_before) > 1e-6:
            dirty_layers.append("structure")

        applied = (shield_before + armor_before + structure_before) - (
            target.vital.shield + target.vital.armor + target.vital.structure
        )
        if applied > 0.0:
            target.combat.last_damaged_at = float(world.now)
            if source is not None and source.team != target.team:
                target.combat.last_enemy_weapon_damaged_at = float(world.now)
            self._queue_merged_event(
                "direct_damage_applied",
                merge_fields={
                    "source": source.ship_id if source is not None else "",
                    "target": target.ship_id,
                    "module": str(module_id or ""),
                    "team": source.team.value if source is not None else "",
                    "source_team": source.team.value if source is not None else "",
                    "target_team": target.team.value,
                },
                sum_fields={
                    "total_damage": applied,
                },
            )
        if target.vital.structure <= 0.0:
            target.vital.alive = False
            target.nav.velocity = Vector2(0.0, 0.0)
            if alive_before:
                self._queue_merged_event(
                    "ship_death",
                    merge_fields={
                        "source": source.ship_id if source is not None else "",
                        "target": target.ship_id,
                        "module": str(module_id or ""),
                        "source_team": source.team.value if source is not None else "",
                        "target_team": target.team.value,
                    },
                    sum_fields={
                        "applied_damage": applied,
                    },
                )
        if dirty_layers or (alive_before and not target.vital.alive):
            self._mark_team_repair_queues_dirty(target.team, *(dirty_layers or _REPAIR_QUEUE_LAYERS))
        return max(0.0, applied)

    def _spawn_projectile(
        self,
        world: WorldState,
        *,
        source,
        module,
        metadata: ModuleStaticMetadata,
        effect,
        target_id: str | None,
    ) -> None:
        max_speed = max(0.0, float(effect.projected_add.get("weapon_projectile_speed", 0.0) or 0.0))
        flight_time = max(0.0, float(effect.projected_add.get("weapon_projectile_flight_time", 0.0) or 0.0))
        acceleration_time = self._projectile_acceleration_time(
            flight_time,
            float(effect.projected_add.get("weapon_projectile_mass", 0.0) or 0.0),
            float(effect.projected_add.get("weapon_projectile_agility", 0.0) or 0.0),
        )
        max_range = self._projectile_max_range(
            max_speed=max_speed,
            flight_time=flight_time,
            acceleration_time=acceleration_time,
            fallback_range=float(effect.range_m or 0.0),
        )
        if max_speed <= 0.0 and flight_time > 0.0 and max_range > 0.0:
            max_speed = max_range / max(0.1, flight_time)
        if flight_time <= 0.0 and max_speed > 0.0 and max_range > 0.0:
            flight_time = max_range / max(1.0, max_speed)
        if max_range <= 0.0 and max_speed <= 0.0:
            return

        source_system_id = self._projectile_entity_system_id(source)
        target = world.combat_entity(target_id) if target_id else None
        if target is not None and (
            not target.vital.alive or self._projectile_entity_system_id(target) != source_system_id
        ):
            target = None
            target_id = None
        if target is not None and target.vital.alive:
            direction = (target.nav.position - source.nav.position).normalized()
        else:
            direction = self._vector_from_facing_deg(source.nav.facing_deg)
        if direction.length() <= 1e-9:
            direction = Vector2(1.0, 0.0)

        initial_speed = self._projectile_speed_for_age(
            max_speed=max_speed,
            acceleration_time=acceleration_time,
            age=0.0,
        )
        kind = "bomb" if metadata.is_bomb_launcher else "missile"
        self._projectile_seq += 1
        projectile_id = f"proj:{self._projectile_seq}"
        shield_max = max(0.0, float(effect.projected_add.get("weapon_projectile_shield_hp", 0.0) or 0.0))
        armor_max = max(0.0, float(effect.projected_add.get("weapon_projectile_armor_hp", 0.0) or 0.0))
        structure_max = max(0.0, float(effect.projected_add.get("weapon_projectile_structure_hp", 0.0) or 0.0))
        if (shield_max + armor_max + structure_max) <= 0.0:
            structure_max = 1.0
        world.projectiles[projectile_id] = ProjectileEntity(
            projectile_id=projectile_id,
            kind=kind,
            source_ship_id=str(source.ship_id),
            source_module_id=str(module.module_id),
            team=source.team,
            position=Vector2(source.nav.position.x, source.nav.position.y),
            velocity=direction * initial_speed,
            facing_deg=direction.angle_deg(),
            target_ship_id=str(target_id) if target_id else None,
            speed=initial_speed,
            max_speed=max_speed,
            max_range=max_range,
            distance_traveled=0.0,
            flight_time=max(0.0, flight_time),
            age=0.0,
            acceleration_time=acceleration_time,
            damage_em=max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            damage_thermal=max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            damage_kinetic=max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            damage_explosive=max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
            explosion_radius=max(0.0, float(effect.projected_add.get("weapon_explosion_radius", 0.0) or 0.0)),
            explosion_velocity=max(0.0, float(effect.projected_add.get("weapon_explosion_velocity", 0.0) or 0.0)),
            damage_reduction_factor=max(0.1, float(effect.projected_add.get("weapon_drf", 0.5) or 0.5)),
            shield=shield_max,
            armor=armor_max,
            structure=structure_max,
            shield_max=shield_max,
            armor_max=armor_max,
            structure_max=structure_max,
            shield_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_em", 1.0) or 1.0))),
            shield_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_thermal", 1.0) or 1.0))),
            shield_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_kinetic", 1.0) or 1.0))),
            shield_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_shield_resonance_explosive", 1.0) or 1.0))),
            armor_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_em", 1.0) or 1.0))),
            armor_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_thermal", 1.0) or 1.0))),
            armor_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_kinetic", 1.0) or 1.0))),
            armor_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_armor_resonance_explosive", 1.0) or 1.0))),
            structure_resonance_em=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_em", 1.0) or 1.0))),
            structure_resonance_thermal=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_thermal", 1.0) or 1.0))),
            structure_resonance_kinetic=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_kinetic", 1.0) or 1.0))),
            structure_resonance_explosive=max(0.0, min(1.0, float(effect.projected_add.get("weapon_projectile_structure_resonance_explosive", 1.0) or 1.0))),
            blast_radius=max(0.0, float(effect.projected_add.get("weapon_blast_radius", 0.0) or 0.0)),
            alive=True,
            system_id=str(getattr(source.nav, "system_id", "") or ""),
        )

    def _spawn_cycle_projectiles(
        self,
        world: WorldState,
        *,
        source,
        module,
        metadata: ModuleStaticMetadata,
        target_id: str | None,
    ) -> None:
        if not (metadata.is_missile_weapon or metadata.is_bomb_launcher):
            return
        for _effect_index, effect in metadata.projected_effects:
            damage_total = sum(
                max(0.0, float(effect.projected_add.get(key, 0.0) or 0.0))
                for key in ("damage_em", "damage_thermal", "damage_kinetic", "damage_explosive")
            )
            if damage_total <= 0.0:
                continue
            self._spawn_projectile(
                world,
                source=source,
                module=module,
                metadata=metadata,
                effect=effect,
                target_id=target_id,
            )

    def _resolve_projectile_hit(self, world: WorldState, projectile: ProjectileEntity) -> None:
        target = world.combat_entity(projectile.target_ship_id or "")
        if target is None or not target.vital.alive:
            return
        if not self._same_projectile_system(projectile, target):
            projectile.target_ship_id = None
            return
        source = world.combat_entity(projectile.source_ship_id)
        if source is not None and not self._same_projectile_system(projectile, source):
            return
        damage_factor = self._missile_damage_factor(projectile, target, target.profile)
        self._apply_direct_damage(
            world,
            source=source,
            target=target,
            target_profile=target.profile,
            damage=self._projectile_damage_tuple(projectile),
            damage_factor=damage_factor,
            module_id=projectile.source_module_id,
        )

    def _resolve_bomb_explosion(self, world: WorldState, projectile: ProjectileEntity) -> None:
        source = world.combat_entity(projectile.source_ship_id)
        if source is not None and not self._same_projectile_system(projectile, source):
            source = None
        blast_radius = max(0.0, float(projectile.blast_radius or 0.0))
        if blast_radius <= 0.0:
            return
        projectile_system_id = str(getattr(projectile, "system_id", "") or "")
        self._create_projectile_blast(
            world,
            kind="bomb",
            position=projectile.position,
            radius_m=blast_radius,
            system_id=projectile_system_id,
        )
        for target in world.iter_combat_entities():
            if not target.vital.alive:
                continue
            if self._projectile_entity_system_id(target) != projectile_system_id:
                continue
            if projectile.position.distance_to(target.nav.position) > (blast_radius + max(0.0, float(target.nav.radius or 0.0))):
                continue
            damage_factor = self._bomb_damage_factor(projectile, target.profile)
            self._apply_direct_damage(
                world,
                source=source,
                target=target,
                target_profile=target.profile,
                damage=self._projectile_damage_tuple(projectile),
                damage_factor=damage_factor,
                module_id=projectile.source_module_id,
            )
        self._destroy_projectiles_in_area(
            world,
            center=projectile.position,
            radius_m=blast_radius,
            exclude_ids={projectile.projectile_id},
            damage=self._projectile_damage_tuple(projectile),
            system_id=projectile_system_id,
        )
        self._destroy_bubbles_in_area(
            world,
            center=projectile.position,
            radius_m=blast_radius,
            damage=self._projectile_damage_tuple(projectile),
            system_id=projectile_system_id,
        )

    def _advance_projectiles(self, world: WorldState, dt: float) -> None:
        if dt <= 0.0:
            return
        for blast_id, blast in list(world.projectile_blasts.items()):
            if float(blast.expires_at) <= float(world.now):
                world.projectile_blasts.pop(blast_id, None)

        for projectile_id, projectile in list(world.projectiles.items()):
            if not projectile.alive:
                world.projectiles.pop(projectile_id, None)
                continue

            target = world.combat_entity(projectile.target_ship_id or "")
            if target is not None and not self._same_projectile_system(projectile, target):
                projectile.target_ship_id = None
                target = None
            if projectile.kind == "missile" and target is not None and target.vital.alive:
                direction = (target.nav.position - projectile.position).normalized()
                if direction.length() > 1e-9:
                    projectile.facing_deg = direction.angle_deg()
                else:
                    direction = self._vector_from_facing_deg(projectile.facing_deg)
            else:
                direction = projectile.velocity.normalized()
                if direction.length() <= 1e-9:
                    direction = self._vector_from_facing_deg(projectile.facing_deg)
            if direction.length() <= 1e-9:
                direction = Vector2(1.0, 0.0)

            step_distance = self._projectile_distance_for_interval(
                max_speed=float(projectile.max_speed or 0.0),
                acceleration_time=float(projectile.acceleration_time or 0.0),
                start_age=float(projectile.age or 0.0),
                dt=dt,
            )
            target_contact_distance = None
            if projectile.kind == "missile" and target is not None and target.vital.alive:
                target_contact_distance = max(
                    0.0,
                    projectile.position.distance_to(target.nav.position) - max(0.0, float(target.nav.radius or 0.0)),
                )

            if target_contact_distance is not None and target_contact_distance <= step_distance + 1e-6:
                projectile.position = Vector2(target.nav.position.x, target.nav.position.y)
                projectile.distance_traveled += max(0.0, target_contact_distance)
                projectile.age += dt
                self._resolve_projectile_hit(world, projectile)
                projectile.alive = False
                world.projectiles.pop(projectile_id, None)
                continue

            projectile.position = projectile.position + (direction * step_distance)
            projectile.distance_traveled += max(0.0, step_distance)
            projectile.age += dt
            projectile.speed = self._projectile_speed_for_age(
                max_speed=float(projectile.max_speed or 0.0),
                acceleration_time=float(projectile.acceleration_time or 0.0),
                age=float(projectile.age or 0.0),
            )
            projectile.velocity = direction * projectile.speed

            expired_by_time = projectile.flight_time > 0.0 and float(projectile.age) >= float(projectile.flight_time) - 1e-6
            expired_by_range = projectile.max_range > 0.0 and float(projectile.distance_traveled) >= float(projectile.max_range) - 1e-6
            if projectile.kind == "missile":
                if target is not None and target.vital.alive:
                    contact_distance = max(
                        0.0,
                        projectile.position.distance_to(target.nav.position) - max(0.0, float(target.nav.radius or 0.0)),
                    )
                    if contact_distance <= 1e-6:
                        self._resolve_projectile_hit(world, projectile)
                        projectile.alive = False
                        world.projectiles.pop(projectile_id, None)
                        continue
                if expired_by_time or expired_by_range:
                    projectile.alive = False
                    world.projectiles.pop(projectile_id, None)
                    continue
            else:
                if expired_by_time or expired_by_range:
                    self._resolve_bomb_explosion(world, projectile)
                    projectile.alive = False
                    world.projectiles.pop(projectile_id, None)
