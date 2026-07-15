from __future__ import annotations

from ..math2d import Vector2
from ..models import ShipProfile
from ..world import WorldState
from .constants import REPAIR_QUEUE_LAYERS
from .models import CycleTargetSnapshot, _apply_damage_sequence, _scale_damage, _sum_damage


class DamageTurretMixin:
    def _compute_projected_damage_factor(
        self,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        strength: float,
        distance: float,
    ) -> float:
        damage_factor = strength
        if float(effect.projected_add.get("weapon_is_turret", 0.0) or 0.0) > 0.5:
            relative_velocity = source.nav.velocity - target.nav.velocity
            radial = (target.nav.position - source.nav.position).normalized()
            tangential = Vector2(-radial.y, radial.x)
            transversal = abs(relative_velocity.x * tangential.x + relative_velocity.y * tangential.y)
            chance = self.pyfa.turret_chance_to_hit(
                tracking=max(0.0, float(effect.projected_add.get("weapon_tracking", 0.0) or 0.0)),
                optimal_sig=max(1.0, float(effect.projected_add.get("weapon_optimal_sig", 40_000.0) or 40_000.0)),
                distance=distance,
                optimal=effect.range_m,
                falloff=effect.falloff_m,
                transversal_speed=transversal,
                target_sig=target_profile.sig_radius,
                attacker_radius=source.nav.radius,
                target_radius=target.nav.radius,
            )
            damage_factor = max(0.0, self.pyfa.turret_damage_multiplier(chance))
        elif float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5:
            target_speed = target.nav.velocity.length()
            explosion_radius = max(0.0, float(effect.projected_add.get("weapon_explosion_radius", 0.0) or 0.0))
            explosion_velocity = max(0.0, float(effect.projected_add.get("weapon_explosion_velocity", 0.0) or 0.0))
            drf = max(0.1, float(effect.projected_add.get("weapon_drf", 0.5) or 0.5))
            if explosion_radius > 0.0:
                sig_factor = target_profile.sig_radius / max(1.0, explosion_radius)
                vel_term = (sig_factor * explosion_velocity) / max(1.0, target_speed)
                vel_factor = vel_term ** drf
                application = max(0.0, min(1.0, min(sig_factor, vel_factor, 1.0)))
            else:
                application = 1.0
            damage_factor = max(0.0, min(1.0, application * strength))
        return max(0.0, damage_factor)

    def _cycle_effect_damage_factor(
        self,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        effect_index: int,
        target_snapshot: CycleTargetSnapshot,
        strength: float,
    ) -> float | None:
        cached = target_snapshot.effect_damage_factors.get(effect_index)
        if cached is not None:
            return cached
        is_turret = float(effect.projected_add.get("weapon_is_turret", 0.0) or 0.0) > 0.5
        is_missile = float(effect.projected_add.get("weapon_is_missile", 0.0) or 0.0) > 0.5
        if not (is_turret or is_missile):
            return None
        damage_factor = self._compute_projected_damage_factor(
            source=source,
            target=target,
            target_profile=target_profile,
            effect=effect,
            strength=strength,
            distance=target_snapshot.distance,
        )
        target_snapshot.effect_damage_factors[effect_index] = damage_factor
        return damage_factor

    def _apply_projected_cycle_effects(
        self,
        world: WorldState,
        source,
        target,
        target_profile: ShipProfile,
        effect,
        strength: float,
        damage_factor_override: float | None = None,
        module_id: str | None = None,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        if target is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Keep layer values bounded to prevent hidden overflow from masking later damage.
        self._clamp_ship_layer_hp(target)

        strength = max(0.0, min(1.0, strength))

        shield_repaired = 0.0
        armor_repaired = 0.0
        cap_drained = 0.0
        dirty_layers: set[str] = set()
        alive_before = bool(target.vital.alive)

        shield_rep = float(effect.projected_add.get("shield_rep", 0.0) or 0.0)
        if shield_rep > 0.0:
            amount = shield_rep * strength
            before = target.vital.shield
            target.vital.shield = min(target.vital.shield_max, target.vital.shield + amount)
            shield_repaired = max(0.0, target.vital.shield - before)
            if shield_repaired > 0.0:
                dirty_layers.add("shield")

        armor_rep = float(effect.projected_add.get("armor_rep", 0.0) or 0.0)
        if armor_rep > 0.0:
            amount = armor_rep * strength
            before = target.vital.armor
            target.vital.armor = min(target.vital.armor_max, target.vital.armor + amount)
            armor_repaired = max(0.0, target.vital.armor - before)
            if armor_repaired > 0.0:
                dirty_layers.add("armor")

        cap_drain = float(effect.projected_add.get("cap_drain", 0.0) or 0.0)
        if cap_drain > 0.0:
            resistance = max(0.0, float(getattr(target_profile, "energy_warfare_resistance", 1.0) or 1.0))
            amount = cap_drain * strength * resistance
            before_cap = target.vital.cap
            target.vital.cap = max(0.0, target.vital.cap - amount)
            cap_drained = max(0.0, before_cap - target.vital.cap)

        base_damage = (
            max(0.0, float(effect.projected_add.get("damage_em", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_thermal", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_kinetic", 0.0) or 0.0)),
            max(0.0, float(effect.projected_add.get("damage_explosive", 0.0) or 0.0)),
        )
        if _sum_damage(base_damage) <= 0.0:
            if dirty_layers:
                self._mark_team_repair_queues_dirty(target.team, *dirty_layers)
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        damage_factor = strength if damage_factor_override is None else max(0.0, float(damage_factor_override))

        dealt_damage = _scale_damage(base_damage, damage_factor)
        total_damage = _sum_damage(dealt_damage)
        if total_damage <= 0.0:
            if dirty_layers:
                self._mark_team_repair_queues_dirty(target.team, *dirty_layers)
            return shield_repaired, armor_repaired, cap_drained, 0.0, 0.0, 0.0, 0.0, 0.0

        shield_before = target.vital.shield
        armor_before = target.vital.armor
        structure_before = target.vital.structure
        target.vital.shield, target.vital.armor, target.vital.structure = _apply_damage_sequence(
            target.vital.shield,
            target.vital.armor,
            target.vital.structure,
            dealt_damage,
            target_profile,
        )
        if abs(target.vital.shield - shield_before) > 1e-6:
            dirty_layers.add("shield")
        if abs(target.vital.armor - armor_before) > 1e-6:
            dirty_layers.add("armor")
        if abs(target.vital.structure - structure_before) > 1e-6:
            dirty_layers.add("structure")
        applied = (shield_before + armor_before + structure_before) - (
            target.vital.shield + target.vital.armor + target.vital.structure
        )
        if applied > 0.0:
            target.combat.last_damaged_at = world.now
        if target.vital.structure <= 0:
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
        if alive_before and not target.vital.alive:
            dirty_layers.update(REPAIR_QUEUE_LAYERS)
        if dirty_layers:
            self._mark_team_repair_queues_dirty(target.team, *dirty_layers)

        return (
            shield_repaired,
            armor_repaired,
            cap_drained,
            dealt_damage[0],
            dealt_damage[1],
            dealt_damage[2],
            dealt_damage[3],
            total_damage,
        )
