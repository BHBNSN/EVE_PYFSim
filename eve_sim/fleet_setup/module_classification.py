from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any

from ..user_errors import UserFacingError


def _normalize_market_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _path(*names: str) -> tuple[str, ...]:
    return tuple(_normalize_market_name(name) for name in names)


@dataclass(frozen=True, slots=True)
class MarketDecisionProfile:
    rule_id: str
    activation_mode: str = "never"
    target_mode: str = "none"
    target_side: str = "enemy"
    cap_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketModuleRule:
    path_prefix: tuple[str, ...]
    classification_id: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    decision: MarketDecisionProfile = field(default_factory=lambda: MarketDecisionProfile("market_default"))
    weapon_kind: str = "none"
    reload_channel: str = "none"


@dataclass(frozen=True, slots=True)
class MarketModuleClassification:
    type_name: str
    category_name: str
    group_name: str
    market_group_id: int | None
    market_path_ids: tuple[int, ...]
    market_path_names: tuple[str, ...]
    classification_id: str
    tags: tuple[str, ...]
    decision: MarketDecisionProfile
    weapon_kind: str = "none"
    reload_channel: str = "none"

    def has_tag(self, tag: str) -> bool:
        return str(tag) in self.tags

    @property
    def is_weapon(self) -> bool:
        return self.weapon_kind in {"turret", "missile", "bomb"}

    @property
    def is_turret_weapon(self) -> bool:
        return self.weapon_kind == "turret"

    @property
    def is_missile_weapon(self) -> bool:
        return self.weapon_kind in {"missile", "bomb"}

    @property
    def is_bomb_launcher(self) -> bool:
        return self.weapon_kind == "bomb"


_DEFAULT_DECISION = MarketDecisionProfile("market_default", "never", "none", "enemy", 0.0)
_WEAPON_DECISION = MarketDecisionProfile("market_weapon", "weapon_focus_only", "weapon_focus_prefocus", "enemy", 0.0)
_AREA_HOSTILE_DECISION = MarketDecisionProfile("market_area_hostile", "enemy_in_area", "none", "enemy", 0.15)
_AREA_SUPPORT_DECISION = MarketDecisionProfile("market_area_support", "always", "none", "ally", 0.0)
_OFFENSIVE_EWAR_DECISION = MarketDecisionProfile("market_offensive_ewar", "cap_min", "enemy_random", "enemy", 0.15)
_REMOTE_REPAIR_DECISION = MarketDecisionProfile("market_remote_repair", "always", "ally_repair_queue", "ally", 0.0)
_REMOTE_SUPPORT_DECISION = MarketDecisionProfile("market_remote_support", "always", "ally_nearest", "ally", 0.0)
_PROPULSION_DECISION = MarketDecisionProfile("market_propulsion", "propulsion_command", "none", "enemy", 0.0)
_DAMAGE_CONTROL_DECISION = MarketDecisionProfile("market_damage_control", "recent_enemy_weapon_damage", "none", "enemy", 0.0)
_HARDENER_DECISION = MarketDecisionProfile("market_hardener", "cap_or_low_hp", "none", "enemy", 0.10)
_CAP_BOOSTER_DECISION = MarketDecisionProfile("market_cap_booster", "cap_max", "none", "enemy", 0.85)
_LOCAL_REPAIR_DECISION = MarketDecisionProfile("market_local_repair", "cap_or_low_hp", "none", "enemy", 0.10)

_RULES: tuple[MarketModuleRule, ...] = (
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Energy Turrets"),
        "weapon_turret_energy",
        ("hostile", "weapon", "turret_weapon"),
        _WEAPON_DECISION,
        weapon_kind="turret",
        reload_channel="turret",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Hybrid Turrets"),
        "weapon_turret_hybrid",
        ("hostile", "weapon", "turret_weapon"),
        _WEAPON_DECISION,
        weapon_kind="turret",
        reload_channel="turret",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Projectile Turrets"),
        "weapon_turret_projectile",
        ("hostile", "weapon", "turret_weapon"),
        _WEAPON_DECISION,
        weapon_kind="turret",
        reload_channel="turret",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Precursor Turrets"),
        "weapon_turret_precursor",
        ("hostile", "weapon", "turret_weapon"),
        _WEAPON_DECISION,
        weapon_kind="turret",
        reload_channel="turret",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Vorton Projectors"),
        "weapon_turret_vorton",
        ("hostile", "weapon", "turret_weapon"),
        _WEAPON_DECISION,
        weapon_kind="turret",
        reload_channel="turret",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Missile Launchers"),
        "weapon_launcher_missile",
        ("hostile", "launcher_weapon", "missile_weapon", "weapon"),
        _WEAPON_DECISION,
        weapon_kind="missile",
        reload_channel="launcher",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Turrets & Launchers", "Bomb Launchers"),
        "weapon_launcher_bomb",
        ("bomb_launcher", "hostile", "launcher_weapon", "missile_weapon", "weapon"),
        _WEAPON_DECISION,
        weapon_kind="bomb",
        reload_channel="launcher",
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Smartbombs"),
        "area_smart_bomb",
        ("area_effect", "hostile", "smart_bomb", "weapon"),
        _AREA_HOSTILE_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Fleet Assistance Modules", "Command Bursts"),
        "area_command_burst",
        ("area_effect", "command_burst", "support"),
        _AREA_SUPPORT_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare"),
        "projected_offensive_ewar",
        ("hostile", "offensive_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Electronic Counter Measures"),
        "projected_ecm",
        ("ecm", "hostile", "offensive_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "ECM Bursts"),
        "area_burst_jammer",
        ("area_effect", "burst_jammer", "ecm", "hostile", "offensive_ewar"),
        _AREA_HOSTILE_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Target Painters"),
        "projected_target_painter",
        ("hostile", "offensive_ewar", "target_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Stasis Webifiers"),
        "projected_stasis_webifier",
        ("hostile", "offensive_ewar", "target_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Stasis Grapplers"),
        "projected_stasis_grappler",
        ("hostile", "offensive_ewar", "target_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Warp Disruption Field Generators"),
        "projected_warp_disruption_field_generator",
        ("hostile", "offensive_ewar", "warp_disruption_field_generator"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Electronic Warfare", "Interdiction Sphere Launchers"),
        "interdiction_sphere_launcher",
        ("bubble", "hostile", "interdiction_sphere_launcher"),
        _AREA_HOSTILE_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Engineering Equipment", "Energy Neutralizers"),
        "projected_energy_neutralizer",
        ("cap_warfare", "energy_neutralizer", "hostile", "offensive_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Engineering Equipment", "Energy Nosferatu"),
        "projected_energy_nosferatu",
        ("cap_warfare", "energy_nosferatu", "hostile", "offensive_ewar"),
        _OFFENSIVE_EWAR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Engineering Equipment", "Remote Capacitor Transmitters"),
        "projected_remote_capacitor_transmitter",
        ("remote_capacitor", "support"),
        _REMOTE_SUPPORT_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Shield", "Remote Shield Boosters"),
        "projected_remote_shield_booster",
        ("remote_repair", "support"),
        _REMOTE_REPAIR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Remote Armor Repairers"),
        "projected_remote_armor_repairer",
        ("remote_repair", "support"),
        _REMOTE_REPAIR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Remote Hull Repairers"),
        "projected_remote_hull_repairer",
        ("remote_repair", "support"),
        _REMOTE_REPAIR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Propulsion"),
        "local_propulsion",
        ("propulsion",),
        _PROPULSION_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Damage Controls"),
        "local_damage_control",
        ("damage_control",),
        _DAMAGE_CONTROL_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Shield", "Shield Hardeners"),
        "local_shield_hardener",
        ("hardener",),
        _HARDENER_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Armor Hardeners"),
        "local_armor_hardener",
        ("hardener",),
        _HARDENER_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Energized Armor Resistance Membranes"),
        "local_energized_armor_resistance",
        ("hardener",),
        _HARDENER_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Engineering Equipment", "Capacitor Boosters"),
        "local_cap_booster",
        ("cap_booster",),
        _CAP_BOOSTER_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Shield", "Shield Boosters"),
        "local_shield_booster",
        ("local_repair",),
        _LOCAL_REPAIR_DECISION,
    ),
    MarketModuleRule(
        _path("Ship Equipment", "Hull & Armor", "Armor Repairers"),
        "local_armor_repairer",
        ("local_repair",),
        _LOCAL_REPAIR_DECISION,
    ),
)

_RULES_BY_DESCENDING_DEPTH = tuple(sorted(_RULES, key=lambda rule: len(rule.path_prefix), reverse=True))


def _path_matches_prefix(path_names: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    if len(path_names) < len(prefix):
        return False
    return path_names[: len(prefix)] == prefix


def _best_rule(path_names: tuple[str, ...]) -> MarketModuleRule | None:
    normalized_path = tuple(_normalize_market_name(name) for name in path_names)
    for rule in _RULES_BY_DESCENDING_DEPTH:
        if _path_matches_prefix(normalized_path, rule.path_prefix):
            return rule
    return None


def _item_identity(item: Any) -> tuple[int | None, str]:
    type_id = getattr(item, "ID", None)
    try:
        normalized_type_id = int(type_id) if type_id is not None else None
    except (TypeError, ValueError):
        normalized_type_id = None
    return normalized_type_id, str(getattr(item, "typeName", "") or "").strip()


class MarketTreeClassifier:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._cache: dict[tuple[int | None, str], MarketModuleClassification] = {}

    def classify_item(self, item: Any) -> MarketModuleClassification:
        type_id, type_name = _item_identity(item)
        cache_key = (type_id, type_name.lower())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        classification = self._classify_uncached(item, type_id, type_name)
        self._cache[cache_key] = classification
        return classification

    def _classify_uncached(
        self,
        item: Any,
        type_id: int | None,
        type_name: str,
    ) -> MarketModuleClassification:
        fallback_group = str(getattr(getattr(item, "group", None), "name", "") or "")
        fallback_category = str(getattr(getattr(item, "category", None), "name", "") or "")
        if not self._db_path.exists() or (type_id is None and not type_name):
            return self._unknown(type_name, fallback_category, fallback_group)

        conn = sqlite3.connect(str(self._db_path))
        try:
            cur = conn.cursor()
            if type_id is not None:
                cur.execute(
                    "SELECT t.typeID, t.typeName, g.name, c.name, t.marketGroupID "
                    "FROM invtypes t "
                    "JOIN invgroups g ON g.groupID=t.groupID "
                    "JOIN invcategories c ON c.categoryID=g.categoryID "
                    "WHERE t.typeID=? "
                    "LIMIT 1",
                    (type_id,),
                )
            else:
                cur.execute(
                    "SELECT t.typeID, t.typeName, g.name, c.name, t.marketGroupID "
                    "FROM invtypes t "
                    "JOIN invgroups g ON g.groupID=t.groupID "
                    "JOIN invcategories c ON c.categoryID=g.categoryID "
                    "WHERE LOWER(t.typeName)=LOWER(?) "
                    "LIMIT 1",
                    (type_name,),
                )
            row = cur.fetchone()
            if not row:
                return self._unknown(type_name, fallback_category, fallback_group)

            resolved_type_name = str(row[1] or type_name)
            group_name = str(row[2] or fallback_group)
            category_name = str(row[3] or fallback_category)
            market_group_id = int(row[4]) if row[4] is not None else None
            if category_name == "Module" and market_group_id is None:
                raise UserFacingError(
                    "Module has no market group and cannot be classified: {name}",
                    name=resolved_type_name,
                )

            path_ids: list[int] = []
            path_names: list[str] = []
            current_market_group_id = market_group_id
            while current_market_group_id is not None:
                cur.execute(
                    "SELECT marketGroupID, marketGroupName, parentGroupID "
                    "FROM invmarketgroups "
                    "WHERE marketGroupID=? "
                    "LIMIT 1",
                    (current_market_group_id,),
                )
                market_row = cur.fetchone()
                if not market_row:
                    break
                path_ids.append(int(market_row[0]))
                path_names.append(str(market_row[1] or ""))
                current_market_group_id = int(market_row[2]) if market_row[2] is not None else None
        finally:
            conn.close()

        path_ids_tuple = tuple(reversed(path_ids))
        path_names_tuple = tuple(reversed(path_names))
        rule = _best_rule(path_names_tuple)
        if rule is None:
            return MarketModuleClassification(
                type_name=resolved_type_name,
                category_name=category_name,
                group_name=group_name,
                market_group_id=market_group_id,
                market_path_ids=path_ids_tuple,
                market_path_names=path_names_tuple,
                classification_id="market_default",
                tags=tuple(),
                decision=_DEFAULT_DECISION,
                weapon_kind="none",
                reload_channel="none",
            )

        return MarketModuleClassification(
            type_name=resolved_type_name,
            category_name=category_name,
            group_name=group_name,
            market_group_id=market_group_id,
            market_path_ids=path_ids_tuple,
            market_path_names=path_names_tuple,
            classification_id=rule.classification_id,
            tags=tuple(sorted(set(rule.tags))),
            decision=rule.decision,
            weapon_kind=rule.weapon_kind,
            reload_channel=rule.reload_channel,
        )

    @staticmethod
    def _unknown(type_name: str, category_name: str, group_name: str) -> MarketModuleClassification:
        return MarketModuleClassification(
            type_name=type_name,
            category_name=category_name,
            group_name=group_name,
            market_group_id=None,
            market_path_ids=tuple(),
            market_path_names=tuple(),
            classification_id="market_unknown",
            tags=tuple(),
            decision=_DEFAULT_DECISION,
            weapon_kind="none",
            reload_channel="none",
        )


__all__ = [
    "MarketDecisionProfile",
    "MarketModuleClassification",
    "MarketTreeClassifier",
]
