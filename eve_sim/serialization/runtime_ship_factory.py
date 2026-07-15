from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..fleet_setup import EftFitParser, RuntimeFromEftFactory
from ..models import DeployableControlState, QualityLevel
from ..world import WorldState
from .snapshot_loader import BasicReplicaShipFactory


class RuntimeReplicaShipFactory(BasicReplicaShipFactory):
    """Enrich replica ships with parsed fit runtimes without engine or GUI coupling."""

    def __init__(self, parser: EftFitParser, runtime_factory: RuntimeFromEftFactory) -> None:
        self._parser = parser
        self._runtime_factory = runtime_factory

    def ensure_ship(self, world: WorldState, ship_id: str, data: Mapping[str, Any]):
        ship = super().ensure_ship(world, ship_id, data)
        fit_text = str(data.get("fit_text", "") or "").strip()
        if fit_text and (fit_text != str(ship.fit_text or "") or ship.runtime is None):
            parsed = self._parser.parse(fit_text)
            runtime_template, fit = self._runtime_factory.build(parsed)
            profile = self._runtime_factory.build_profile(parsed)
            if hasattr(self._runtime_factory, "build_deployable_manifest"):
                drone_bay, fighter_bay, control = self._runtime_factory.build_deployable_manifest(parsed)
            else:
                drone_bay, fighter_bay, control = [], [], DeployableControlState()
            ship.fit = fit
            ship.profile = profile
            ship.runtime = deepcopy(runtime_template)
            ship.drone_bay = list(drone_bay)
            ship.fighter_bay = list(fighter_bay)
            ship.deployable_control = deepcopy(control)
            ship.nav.max_speed = profile.max_speed
            ship.fit_text = fit_text
        try:
            ship.quality.level = QualityLevel(str(data.get("quality_level", ship.quality.level.value)))
        except ValueError:
            pass
        ship.quality.reaction_delay = float(data.get("quality_reaction_delay", ship.quality.reaction_delay) or 0.0)
        ship.quality.ignore_order_probability = float(
            data.get("quality_ignore_order_probability", ship.quality.ignore_order_probability) or 0.0
        )
        ship.quality.formation_jitter = float(data.get("quality_formation_jitter", ship.quality.formation_jitter) or 0.0)
        return ship
