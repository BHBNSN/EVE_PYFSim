from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import logging
import random
import time
import weakref
from typing import Any

import numpy as np

from ..combat_control_workset import (
    enqueue_control_signal_modules,
    ensure_ship_module_decision_pending,
    module_keeps_decision_pending,
    runtime_decision_rule_groups,
    runtime_controlled_entry_lookup,
    runtime_controlled_module_ids,
    ship_candidate_module_ids,
)
from ..fleet_setup import (
    _module_affects_local_pyfa_profile,
    _runtime_local_profile_state_signature,
    get_runtime_resolve_cache_key,
    resolve_runtime_from_pyfa_runtime,
)
from ..fit_runtime import EffectClass, FitRuntime, ModuleEffect, ModuleRuntime, ModuleState, ProjectedImpact, RuntimeStatEngine
from ..math2d import Vector2
from ..module_control import normalize_module_manual_mode, normalize_module_target_mode
from ..models import BubbleField, ProjectileBlast, ProjectileEntity, ShipProfile, Team
from ..pyfa_bridge import PyfaBridge
from ..remote_snapshot_signatures import (
    normalized_snapshot_projection_signature as shared_normalized_snapshot_projection_signature,
    projected_snapshot_list_signature as shared_projected_snapshot_list_signature,
    projected_snapshot_module_signature as shared_projected_snapshot_module_signature,
)
from ..replay.schema import CombatEvent, CombatEventSink
from ..sim_logging import log_sim_event
from ..timer_views import adopt_deadlines_from_remaining_view, deadline_remaining, sync_deadline_view
from ..timing_wheel import EventType, TimingWheel
from ..world import WorldState

_PYFA_PROJECTION_RANGE_BUCKET_M = 100.0
_REPAIR_QUEUE_LAYERS = ("shield", "armor", "structure")
from .models import (
    CycleTargetSnapshot,
    ModuleDecisionRule,
    ModuleStaticMetadata,
    RuntimeModuleBuckets,
    _FORMULA_PROJECTED_KEYS,
    _PROFILE_PASSTHROUGH_ATTRS,
    _RUNTIME_MODULE_OBJECT_CACHE_DIAGNOSTIC_KEYS,
    _apply_damage_sequence,
    _layer_effective_damage,
    _scale_damage,
    _sum_damage,
)

__all__ = [name for name in globals() if not name.startswith("__")]
