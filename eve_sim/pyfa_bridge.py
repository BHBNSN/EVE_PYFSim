from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable

from .config import resolve_pyfa_source_dir
from .models import FitDescriptor, ShipProfile


@dataclass(slots=True)
class PyfaMath:
    calculate_range_factor: Callable
    calculate_lock_time: Callable


class PyfaBridge:
    def __init__(self) -> None:
        self._pyfa_math = self._load_pyfa_math()
        self._profile_cache: dict[str, ShipProfile] = {}

    def _load_pyfa_math(self) -> PyfaMath:
        pyfa_path = resolve_pyfa_source_dir()
        if pyfa_path.exists():
            src = str(pyfa_path)
            if src not in sys.path:
                sys.path.insert(0, src)
        try:
            from eos.calc import calculateLockTime, calculateRangeFactor

            return PyfaMath(
                calculate_range_factor=calculateRangeFactor,
                calculate_lock_time=calculateLockTime,
            )
        except Exception:
            return PyfaMath(
                calculate_range_factor=self._fallback_range,
                calculate_lock_time=self._fallback_lock,
            )

    @staticmethod
    def _fallback_range(optimal: float, falloff: float, distance: float, restrictedRange: bool = True) -> float:
        if distance is None:
            return 1.0
        if falloff > 0:
            if restrictedRange and distance > optimal + 3 * falloff:
                return 0.0
            return 0.5 ** ((max(0.0, distance - optimal) / falloff) ** 2)
        return 1.0 if distance <= optimal else 0.0

    @staticmethod
    def _fallback_lock(scan_res: float, sig_radius: float) -> float | None:
        if not scan_res or not sig_radius:
            return None
        return min(40000 / scan_res / math.asinh(sig_radius) ** 2, 30 * 60)

    def build_profile(self, fit: FitDescriptor) -> ShipProfile:
        cached = self._profile_cache.get(fit.fit_key)
        if cached is not None:
            return cached
        profile = ShipProfile(
            dps=fit.base_dps,
            volley=fit.volley,
            optimal=fit.optimal_range,
            falloff=fit.falloff,
            tracking=fit.tracking,
            sig_radius=fit.signature_radius,
            scan_resolution=fit.scan_resolution,
            max_target_range=fit.max_target_range,
            max_speed=fit.max_speed,
            max_cap=fit.max_cap,
            cap_recharge_time=fit.cap_recharge_time,
            shield_hp=fit.shield_hp,
            armor_hp=fit.armor_hp,
            structure_hp=fit.structure_hp,
            rep_amount=fit.rep_amount,
            rep_cycle=fit.rep_cycle,
            sensor_strength_gravimetric=fit.sensor_strength_gravimetric,
            sensor_strength_ladar=fit.sensor_strength_ladar,
            sensor_strength_magnetometric=fit.sensor_strength_magnetometric,
            sensor_strength_radar=fit.sensor_strength_radar,
            mass=max(0.0, float(getattr(fit, "mass", 0.0) or 0.0)),
            agility=max(0.0, float(getattr(fit, "agility", 0.0) or 0.0)),
        )
        self._profile_cache[fit.fit_key] = profile
        return profile

    def calculate_lock_time(self, attacker: ShipProfile, defender: ShipProfile) -> float:
        result = self._pyfa_math.calculate_lock_time(attacker.scan_resolution, defender.sig_radius)
        return 2.0 if result is None else float(result)

    def turret_chance_to_hit(
        self,
        tracking: float,
        optimal_sig: float,
        distance: float,
        optimal: float,
        falloff: float,
        transversal_speed: float,
        target_sig: float,
        attacker_radius: float | None = None,
        target_radius: float | None = None,
    ) -> float:
        range_factor = self._pyfa_math.calculate_range_factor(optimal, falloff, distance, False)
        if tracking <= 0 or target_sig <= 0:
            return 0.0
        ctc_distance = distance
        if attacker_radius is not None and target_radius is not None:
            ctc_distance = attacker_radius + distance + target_radius
        angular_speed = 0.0 if ctc_distance <= 0 else abs(transversal_speed / ctc_distance)
        tracking_factor = 0.5 ** (((angular_speed * optimal_sig) / (tracking * target_sig)) ** 2)
        return max(0.0, min(1.0, range_factor * tracking_factor))

    def turret_range_factor(self, optimal: float, falloff: float, distance: float) -> float:
        return max(0.0, min(1.0, float(self._pyfa_math.calculate_range_factor(optimal, falloff, distance, False))))

    @staticmethod
    def turret_damage_multiplier(chance_to_hit: float) -> float:
        wrecking = min(chance_to_hit, 0.01)
        wrecking_part = wrecking * 3
        normal = chance_to_hit - wrecking
        normal_part = normal * (((0.01 + chance_to_hit) / 2 + 0.49) if normal > 0 else 0.0)
        return normal_part + wrecking_part
