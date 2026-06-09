from __future__ import annotations

from .combat_common import *  # noqa: F403


class DamageMissileMixin:
    def _missile_damage_factor(self, projectile: ProjectileEntity, target, target_profile: ShipProfile) -> float:
        explosion_radius = max(0.0, float(projectile.explosion_radius or 0.0))
        if explosion_radius <= 0.0:
            return 1.0
        sig_factor = float(target_profile.sig_radius or 0.0) / max(1.0, explosion_radius)
        explosion_velocity = max(0.0, float(projectile.explosion_velocity or 0.0))
        target_speed = target.nav.velocity.length()
        if target_speed <= 1e-9 or explosion_velocity <= 0.0:
            velocity_factor = 1.0
        else:
            velocity_factor = ((sig_factor * explosion_velocity) / max(1.0, target_speed)) ** max(
                0.1,
                float(projectile.damage_reduction_factor or 0.5),
            )
        return max(0.0, min(1.0, min(1.0, sig_factor, velocity_factor)))

    def _bomb_damage_factor(self, projectile: ProjectileEntity, target_profile: ShipProfile) -> float:
        explosion_radius = max(0.0, float(projectile.explosion_radius or 0.0))
        if explosion_radius <= 0.0:
            return 1.0
        return max(0.0, min(1.0, float(target_profile.sig_radius or 0.0) / explosion_radius))
