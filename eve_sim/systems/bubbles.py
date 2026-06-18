from __future__ import annotations

from .combat_common import *  # noqa: F403


class BubblesMixin:
    @staticmethod
    def _module_bubble_effects(module) -> tuple[ModuleEffect, ...]:
        return tuple(
            effect
            for effect in module.effects
            if effect.effect_class == EffectClass.LOCAL
            and float(effect.local_add.get("bubble_radius_m", 0.0) or 0.0) > 0.0
        )

    @staticmethod
    def _bubble_blocks_warp(effect: ModuleEffect) -> bool:
        return float(effect.local_add.get("bubble_blocks_warp", 0.0) or 0.0) > 0.5

    @staticmethod
    def _bubble_follows_owner(effect: ModuleEffect) -> bool:
        return float(effect.local_add.get("bubble_follow_owner", 0.0) or 0.0) > 0.5

    @staticmethod
    def _bubble_kind(effect: ModuleEffect) -> str:
        if float(effect.local_add.get("bubble_web", 0.0) or 0.0) > 0.5:
            return "webification_probe"
        if float(effect.local_add.get("bubble_hic", 0.0) or 0.0) > 0.5:
            return "hic_warp_field"
        return "warp_disrupt_probe"

    @staticmethod
    def _bubble_interdiction_kind(effect: ModuleEffect) -> str:
        if float(effect.local_add.get("bubble_hic", 0.0) or 0.0) > 0.5:
            return "hic"
        return "probe"

    @staticmethod
    def _bubble_speed_factor_mult(effect: ModuleEffect) -> float:
        return max(0.01, float(effect.local_add.get("bubble_speed_factor_mult", 1.0) or 1.0))

    @staticmethod
    def _clamp_bubble_layer_hp(field: BubbleField) -> None:
        field.shield_max = max(0.0, float(field.shield_max or 0.0))
        field.armor_max = max(0.0, float(field.armor_max or 0.0))
        field.structure_max = max(0.0, float(field.structure_max or 0.0))
        field.shield = max(0.0, min(float(field.shield or 0.0), field.shield_max))
        field.armor = max(0.0, min(float(field.armor or 0.0), field.armor_max))
        field.structure = max(0.0, min(float(field.structure or 0.0), field.structure_max))
        field.alive = (field.shield + field.armor + field.structure) > 1e-6 or not field.destructible

    def _apply_damage_to_bubble(
        self,
        field: BubbleField,
        damage: DamageTuple,
        *,
        damage_factor: float = 1.0,
    ) -> bool:
        if not field.destructible:
            return False
        scaled_damage = _scale_damage(damage, max(0.0, float(damage_factor)))
        if _sum_damage(scaled_damage) <= 0.0:
            return False

        self._clamp_bubble_layer_hp(field)
        remaining = scaled_damage
        layer_specs = (
            (
                "shield",
                float(field.shield),
                (
                    float(field.shield_resonance_em),
                    float(field.shield_resonance_thermal),
                    float(field.shield_resonance_kinetic),
                    float(field.shield_resonance_explosive),
                ),
            ),
            (
                "armor",
                float(field.armor),
                (
                    float(field.armor_resonance_em),
                    float(field.armor_resonance_thermal),
                    float(field.armor_resonance_kinetic),
                    float(field.armor_resonance_explosive),
                ),
            ),
            (
                "structure",
                float(field.structure),
                (
                    float(field.structure_resonance_em),
                    float(field.structure_resonance_thermal),
                    float(field.structure_resonance_kinetic),
                    float(field.structure_resonance_explosive),
                ),
            ),
        )
        new_values = {
            "shield": float(field.shield),
            "armor": float(field.armor),
            "structure": float(field.structure),
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

        field.shield = max(0.0, new_values["shield"])
        field.armor = max(0.0, new_values["armor"])
        field.structure = max(0.0, new_values["structure"])
        field.alive = (field.shield + field.armor + field.structure) > 1e-6
        return not field.alive

    def _spawn_static_bubble_field(
        self,
        world: WorldState,
        *,
        source,
        module,
        effect: ModuleEffect,
    ) -> None:
        radius_m = max(0.0, float(effect.local_add.get("bubble_radius_m", 0.0) or 0.0))
        duration_sec = max(0.0, float(effect.local_add.get("bubble_duration_sec", 0.0) or 0.0))
        if radius_m <= 0.0 or duration_sec <= 0.0 or self._bubble_follows_owner(effect):
            return
        self._bubble_seq += 1
        field_id = f"bubble:{self._bubble_seq}"
        shield_max = max(0.0, float(effect.local_add.get("bubble_shield_hp", 0.0) or 0.0))
        armor_max = max(0.0, float(effect.local_add.get("bubble_armor_hp", 0.0) or 0.0))
        structure_max = max(0.0, float(effect.local_add.get("bubble_structure_hp", 0.0) or 0.0))
        if (shield_max + armor_max + structure_max) <= 0.0:
            structure_max = 1.0
        world.bubble_fields[field_id] = BubbleField(
            field_id=field_id,
            kind=self._bubble_kind(effect),
            interdiction_kind=self._bubble_interdiction_kind(effect),
            source_ship_id=str(source.ship_id),
            source_module_id=str(module.module_id),
            team=source.team,
            position=Vector2(source.nav.position.x, source.nav.position.y),
            radius_m=radius_m,
            expires_at=float(world.now) + duration_sec,
            system_id=str(getattr(source.nav, "system_id", "") or ""),
            blocks_warp=self._bubble_blocks_warp(effect),
            speed_factor_mult=self._bubble_speed_factor_mult(effect),
            anchor_ship_id=None,
            destructible=True,
            shield=shield_max,
            armor=armor_max,
            structure=structure_max,
            shield_max=shield_max,
            armor_max=armor_max,
            structure_max=structure_max,
            shield_resonance_em=max(0.0, min(1.0, float(effect.local_add.get("bubble_shield_resonance_em", 1.0) or 1.0))),
            shield_resonance_thermal=max(0.0, min(1.0, float(effect.local_add.get("bubble_shield_resonance_thermal", 1.0) or 1.0))),
            shield_resonance_kinetic=max(0.0, min(1.0, float(effect.local_add.get("bubble_shield_resonance_kinetic", 1.0) or 1.0))),
            shield_resonance_explosive=max(0.0, min(1.0, float(effect.local_add.get("bubble_shield_resonance_explosive", 1.0) or 1.0))),
            armor_resonance_em=max(0.0, min(1.0, float(effect.local_add.get("bubble_armor_resonance_em", 1.0) or 1.0))),
            armor_resonance_thermal=max(0.0, min(1.0, float(effect.local_add.get("bubble_armor_resonance_thermal", 1.0) or 1.0))),
            armor_resonance_kinetic=max(0.0, min(1.0, float(effect.local_add.get("bubble_armor_resonance_kinetic", 1.0) or 1.0))),
            armor_resonance_explosive=max(0.0, min(1.0, float(effect.local_add.get("bubble_armor_resonance_explosive", 1.0) or 1.0))),
            structure_resonance_em=max(0.0, min(1.0, float(effect.local_add.get("bubble_structure_resonance_em", 1.0) or 1.0))),
            structure_resonance_thermal=max(0.0, min(1.0, float(effect.local_add.get("bubble_structure_resonance_thermal", 1.0) or 1.0))),
            structure_resonance_kinetic=max(0.0, min(1.0, float(effect.local_add.get("bubble_structure_resonance_kinetic", 1.0) or 1.0))),
            structure_resonance_explosive=max(0.0, min(1.0, float(effect.local_add.get("bubble_structure_resonance_explosive", 1.0) or 1.0))),
            alive=True,
        )

    def _sync_dynamic_bubble_fields(self, world: WorldState) -> None:
        active_dynamic_field_ids: set[str] = set()
        for ship in world.ships.values():
            if not ship.vital.alive or ship.runtime is None or self._ship_combat_suppressed(ship):
                continue
            for module in ship.runtime.modules:
                if module.state != ModuleState.ACTIVE:
                    continue
                for effect in self._module_bubble_effects(module):
                    if not self._bubble_follows_owner(effect):
                        continue
                    field_id = f"bubble:hic:{ship.ship_id}:{module.module_id}"
                    active_dynamic_field_ids.add(field_id)
                    field = world.bubble_fields.get(field_id)
                    if field is None:
                        field = BubbleField(
                            field_id=field_id,
                            kind=self._bubble_kind(effect),
                            interdiction_kind=self._bubble_interdiction_kind(effect),
                            source_ship_id=str(ship.ship_id),
                            source_module_id=str(module.module_id),
                            team=ship.team,
                            position=Vector2(ship.nav.position.x, ship.nav.position.y),
                            radius_m=max(0.0, float(effect.local_add.get("bubble_radius_m", 0.0) or 0.0)),
                            expires_at=float(world.now) + 1.0,
                            system_id=str(getattr(ship.nav, "system_id", "") or ""),
                            blocks_warp=self._bubble_blocks_warp(effect),
                            speed_factor_mult=self._bubble_speed_factor_mult(effect),
                            anchor_ship_id=str(ship.ship_id),
                            destructible=False,
                            alive=True,
                        )
                        world.bubble_fields[field_id] = field
                    else:
                        field.position = Vector2(ship.nav.position.x, ship.nav.position.y)
                        field.system_id = str(getattr(ship.nav, "system_id", "") or "")
                        field.radius_m = max(0.0, float(effect.local_add.get("bubble_radius_m", field.radius_m) or field.radius_m))
                        field.expires_at = float(world.now) + 1.0
                        field.blocks_warp = self._bubble_blocks_warp(effect)
                        field.speed_factor_mult = self._bubble_speed_factor_mult(effect)
                        field.anchor_ship_id = str(ship.ship_id)
                        field.alive = True

        for field_id, field in list(world.bubble_fields.items()):
            if field.anchor_ship_id is not None and field_id not in active_dynamic_field_ids:
                world.bubble_fields.pop(field_id, None)
                continue
            if not field.alive:
                world.bubble_fields.pop(field_id, None)
                continue
            if field.anchor_ship_id is None and float(field.expires_at) <= float(world.now):
                world.bubble_fields.pop(field_id, None)

    def _destroy_bubbles_in_area(
        self,
        world: WorldState,
        *,
        center: Vector2,
        radius_m: float,
        damage: DamageTuple | None = None,
        damage_factor: float = 1.0,
        system_id: str = "",
    ) -> None:
        radius = max(0.0, float(radius_m or 0.0))
        if radius <= 0.0:
            return
        source_system_id = str(system_id or "")
        hit_radius = radius + 25.0
        for field_id, field in list(world.bubble_fields.items()):
            if source_system_id and str(getattr(field, "system_id", "") or "") != source_system_id:
                continue
            if not field.destructible:
                continue
            if not field.alive:
                world.bubble_fields.pop(field_id, None)
                continue
            if field.position.distance_to(center) > hit_radius:
                continue
            if damage is None:
                field.alive = False
            else:
                self._apply_damage_to_bubble(
                    field,
                    damage,
                    damage_factor=damage_factor,
                )
            if not field.alive:
                world.bubble_fields.pop(field_id, None)
