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

from .models import *


class EftFitParser:
    _header_re = re.compile(r"^\[(?P<ship>[^,\]]+)\s*,\s*(?P<name>[^\]]+)\]$")
    _offline_suffixes = ("/offline", "/OFFLINE")

    def parse(self, fit_text: str) -> ParsedEftFit:
        lines = [line.strip() for line in fit_text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError("配装文本为空")

        m = self._header_re.match(lines[0])
        if not m:
            raise ValueError("EFT 头格式无效，需形如 [Ship, Fit Name]")

        ship_name = m.group("ship").strip()
        fit_name = m.group("name").strip()

        modules: list[str] = []
        module_specs: list[ParsedModuleSpec] = []
        cargo_item_names: list[str] = []

        for raw in lines[1:]:
            if raw.lower().startswith("dna:"):
                continue
            if raw.lower().startswith("x-"):
                continue
            line = raw
            is_quantity_line = " x" in line
            if is_quantity_line:
                qty_name = line.split(" x", 1)[0].strip()
                if qty_name:
                    cargo_item_names.append(qty_name)
                continue
            offline = False
            for suffix in self._offline_suffixes:
                if line.endswith(suffix):
                    offline = True
                    line = line[: -len(suffix)].strip()
                    break
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            charge_name: str | None = None
            if "," in line:
                mod, charge = line.split(",", 1)
                line = mod.strip()
                charge_name = charge.strip() or None
            if line.startswith("[Empty"):
                continue
            modules.append(line)
            module_specs.append(ParsedModuleSpec(module_name=line, charge_name=charge_name, offline=offline))

        fit_key = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        return ParsedEftFit(
            ship_name=ship_name,
            fit_name=fit_name,
            module_names=modules,
            module_specs=module_specs,
            cargo_item_names=cargo_item_names,
            fit_key=f"eft-{fit_key}",
        )

