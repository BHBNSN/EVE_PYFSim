from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import math
import random
import re
from pathlib import Path
import sqlite3
import sys
import types
from typing import Any, cast

from ..config import resolve_pyfa_source_dir

from ..fit_runtime import (
    EffectClass,
    FitRuntime,
    HullProfile,
    ModuleEffect,
    ModuleRuntime,
    ModuleState,
    SkillProfile,
)
from ..math2d import Vector2
from ..models import (
    Beacon,
    CombatState,
    FitDescriptor,
    NavigationState,
    QualityLevel,
    QualityState,
    ShipProfile,
    ShipEntity,
    Team,
    VitalState,
)
from ..world import WorldState


QUALITY_PRESETS = {
    QualityLevel.ELITE: QualityState(QualityLevel.ELITE, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.REGULAR: QualityState(QualityLevel.REGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
    QualityLevel.IRREGULAR: QualityState(QualityLevel.IRREGULAR, reaction_delay=0.0, ignore_order_probability=0.0, formation_jitter=0.0),
}


@dataclass(slots=True)
class ParsedEftFit:
    ship_name: str
    fit_name: str
    module_names: list[str]
    module_specs: list["ParsedModuleSpec"]
    cargo_item_names: list[str]
    fit_key: str


@dataclass(slots=True)
class ParsedModuleSpec:
    module_name: str
    charge_name: str | None = None
    offline: bool = False


@dataclass(slots=True)
class ManualShipSetup:
    team: Team
    squad_id: str
    quality: QualityLevel
    position: Vector2
    fit_text: str
    is_leader: bool = False

