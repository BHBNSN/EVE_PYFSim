from __future__ import annotations

from types import MappingProxyType

from ..math2d import Vector2
from ..models import Team
from ..squad_identity import squad_key
from .queries import MatchStatusView, ModuleTargetRulesView, OverviewQuery, OverviewView, ShipView, SimulationDiagnosticsView, SquadView
from .session import MatchSession
from .contracts import ShipSetupSpec


class QueryService:
    """Read-only DTO projections. Queries never refresh leaders or mutate WorldState."""

    def __init__(self, session: MatchSession) -> None:
        self._session = session

    def match_status(self) -> MatchStatusView:
        world = self._session.world
        return MatchStatusView(self._session.match_id, self._session.status.value, int(world.tick), float(world.now))

    def squad_view(self, team: Team, squad_id: str) -> SquadView:
        world = self._session.world
        key = squad_key(team, squad_id)
        members = tuple(sorted(ship.ship_id for ship in world.ships.values() if ship.team == team and ship.squad_id == squad_id))
        speed = world.squad_leader_speed_limits.get(key)
        return SquadView(
            team=team.value,
            squad_id=squad_id,
            leader_id=world.squad_leaders.get(key),
            member_ids=members,
            focus_queue=tuple(world.squad_focus_queues.get(key, ())),
            propulsion_active=bool(world.squad_propulsion_commands.get(key, False)),
            speed_limit=float(speed) if speed is not None else None,
        )

    def ship_view(self, ship_id: str) -> ShipView | None:
        ship = self._session.world.ships.get(ship_id)
        if ship is None:
            return None
        return ShipView(
            ship_id=ship.ship_id,
            team=ship.team.value,
            squad_id=ship.squad_id,
            alive=bool(ship.vital.alive),
            system_id=str(ship.nav.system_id or ""),
            position=(float(ship.nav.position.x), float(ship.nav.position.y)),
            current_target=ship.combat.current_target,
            deployed=bool(ship.deployed),
            fit_text=str(ship.fit_text or ""),
            locked_module_charges=MappingProxyType(dict(ship.locked_module_charges)),
            module_manual_modes=MappingProxyType(dict(ship.combat.module_manual_modes)),
            module_target_modes=MappingProxyType(dict(ship.combat.module_target_modes)),
        )

    def overview(self, query: OverviewQuery = OverviewQuery()) -> OverviewView:
        ships = []
        for ship in self._session.world.ships.values():
            if query.team is not None and ship.team.value != query.team:
                continue
            if query.system_id is not None and str(ship.nav.system_id or "") != query.system_id:
                continue
            if query.alive_only and not ship.vital.alive:
                continue
            view = self.ship_view(ship.ship_id)
            if view is not None:
                ships.append(view)
        return OverviewView(tuple(sorted(ships, key=lambda item: item.ship_id)))

    def diagnostics(self) -> SimulationDiagnosticsView:
        values = self._session.engine.runtime_diagnostics()
        return SimulationDiagnosticsView(MappingProxyType(values))

    def module_target_rules(self, ship_id: str, module_id: str) -> ModuleTargetRulesView:
        choices, default_mode = self._session.module_commands.target_rules(self._session.world, ship_id, module_id)
        return ModuleTargetRulesView(choices=choices, default_mode=default_mode)

    def scenario_ship_specs(self, team: Team) -> tuple[ShipSetupSpec, ...]:
        """Project a client-owned fleet into the immutable setup transfer contract."""
        return tuple(
            ShipSetupSpec(
                ship_id=ship.ship_id,
                squad_id=ship.squad_id,
                ship_group_id=str(ship.ship_group_id or ""),
                fit_text=str(ship.fit_text or ""),
                position=Vector2(float(ship.nav.position.x), float(ship.nav.position.y)),
                velocity=Vector2(float(ship.nav.velocity.x), float(ship.nav.velocity.y)),
                facing_deg=float(ship.nav.facing_deg),
                system_id=str(ship.nav.system_id or ""),
                deployed=bool(ship.deployed),
                alive=bool(ship.vital.alive),
                shield=float(ship.vital.shield),
                armor=float(ship.vital.armor),
                structure=float(ship.vital.structure),
                cap=float(ship.vital.cap),
                quality_level=ship.quality.level.value,
                quality_reaction_delay=float(ship.quality.reaction_delay),
                quality_ignore_order_probability=float(ship.quality.ignore_order_probability),
                quality_formation_jitter=float(ship.quality.formation_jitter),
            )
            for ship in sorted(self._session.world.ships.values(), key=lambda item: item.ship_id)
            if ship.team == team
        )

    def team_fit_texts(self, team: Team) -> tuple[str, ...]:
        return tuple(
            str(ship.fit_text).strip()
            for ship in sorted(self._session.world.ships.values(), key=lambda item: item.ship_id)
            if ship.team == team and str(ship.fit_text or "").strip()
        )
