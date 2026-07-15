from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any
import weakref

from ..fit_runtime import RuntimeStatEngine
from ..models import Team
from ..pyfa_bridge import PyfaBridge
from ..replay.schema import CombatEventSink
from ..system_identity import normalize_system_namespace
from ..timing_wheel import TimingWheel
from .authoritative_tick import AuthoritativeTickMixin
from .bubbles import BubblesMixin
from .command_bursts import CommandBurstsMixin
from .damage_missile import DamageMissileMixin
from .damage_turret import DamageTurretMixin
from .ewar import EwarMixin
from .locking import LockingMixin
from .logistics import CombatLogisticsMixin
from .module_cycles import ModuleCyclesMixin
from .projectiles import ProjectilesMixin
from .runtime_projection import RuntimeProjectionMixin
from .models import CycleTargetSnapshot, ModuleStaticMetadata


class CombatStateCloneError(RuntimeError):
    pass

class CombatSystem(
    AuthoritativeTickMixin,
    LockingMixin,
    DamageTurretMixin,
    DamageMissileMixin,
    ProjectilesMixin,
    EwarMixin,
    CombatLogisticsMixin,
    BubblesMixin,
    CommandBurstsMixin,
    ModuleCyclesMixin,
    RuntimeProjectionMixin,
):
    def __init__(
        self,
        pyfa: PyfaBridge,
        combat_event_sink: CombatEventSink | None = None,
    ) -> None:
        self.pyfa = pyfa
        self.runtime = RuntimeStatEngine()
        self.logger: logging.Logger | None = None
        self.detailed_logging: bool = False
        self.hotspot_logging_enabled: bool = False
        self.event_logging_enabled: bool = False
        self.event_merge_window_sec: float = 1.0
        self._diag_logged_ships: set[str] = set()
        self._lock_time_cache: dict[tuple[float, float], float] = {}
        self._projected_cycle_totals: dict[tuple[str, str, str], dict[str, float]] = {}
        self._projected_cycle_starts_this_tick: set[tuple[str, str]] = set()
        self._module_cycle_target_snapshots: dict[tuple[str, str], dict[str, CycleTargetSnapshot]] = {}
        self._merged_event_buckets: dict[tuple, dict[str, Any]] = {}
        self._merge_window_start_time: float | None = None
        self._merge_window_end_time: float | None = None
        self._last_focus_queue_by_squad: dict[str, tuple[str, ...]] = {}
        self._pyfa_remote_inputs_dirty: bool = True
        self._alive_runtime_ship_ids: set[str] = set()
        self._cached_command_booster_snapshots: dict[str, list[dict[str, Any]]] | None = None
        self._cached_projected_source_snapshots: dict[str, list[dict[str, Any]]] | None = None
        self._module_static_metadata_by_object_id: dict[int, tuple[weakref.ReferenceType[Any], ModuleStaticMetadata]] = {}
        self._repair_queue_cache: dict[tuple[Team, str, str], tuple[str, ...]] = {}
        self._repair_queue_dirty: set[tuple[Team, str, str]] = set()
        self._projectile_seq: int = 0
        self._projectile_blast_seq: int = 0
        self._bubble_seq: int = 0
        self._timing_wheel = TimingWheel()
        self._decision_reference_time: float | None = None
        self._combat_event_sink: CombatEventSink | None = combat_event_sink
        self._event_rng_seed: int = 0
        self._event_rng_counter: int = 0
        self._current_event_tick: int = 0
        self._current_event_at: float = 0.0
        self._system_id: str = ""
        self._system_namespace: str = ""

    def attach_logger(
        self,
        logger: logging.Logger,
        detailed_logging: bool,
        merge_window_sec: float = 1.0,
        hotspot_logging: bool = False,
    ) -> None:
        self.logger = logger
        self.event_logging_enabled = bool(detailed_logging)
        self.detailed_logging = bool(detailed_logging)
        self.hotspot_logging_enabled = bool(hotspot_logging)
        try:
            self.event_merge_window_sec = max(0.1, float(merge_window_sec))
        except Exception:
            self.event_merge_window_sec = 1.0
        self._merge_window_start_time = None
        self._merge_window_end_time = None
        self._merged_event_buckets.clear()

    def attach_event_sink(self, event_sink: CombatEventSink | None) -> None:
        self._combat_event_sink = event_sink

    def set_event_rng_context(self, rng_seed: int, rng_counter: int = 0) -> None:
        self._event_rng_seed = int(rng_seed)
        self._event_rng_counter = int(rng_counter)

    def module_target_metadata(self, module):
        """Expose module targeting metadata without leaking mixin-private APIs."""
        return self._module_static_metadata(module)

    def module_target_mode_choices(self, module) -> tuple[str, ...]:
        metadata = self.module_target_metadata(module)
        return self._module_target_mode_choices(module, metadata)

    @classmethod
    def copy_runtime_dynamic_state(cls, source_runtime, target_runtime) -> None:
        cls._copy_runtime_dynamic_state(source_runtime, target_runtime)

    def clone_for_system(self, system_id: str) -> "CombatSystem":
        """Clone authoritative combat state without copying external resources."""
        namespace = normalize_system_namespace(system_id)
        excluded = {"logger", "_combat_event_sink", "_module_static_metadata_by_object_id"}
        try:
            cloned = object.__new__(type(self))
            for name, value in self.__dict__.items():
                if name in excluded:
                    continue
                setattr(cloned, name, deepcopy(value))
            cloned.logger = None
            cloned._combat_event_sink = None
            cloned._module_static_metadata_by_object_id = {}
            cloned._system_id = str(system_id).strip()
            cloned._system_namespace = namespace
            return cloned
        except Exception as exc:
            raise CombatStateCloneError(
                f"failed to clone CombatSystem for system {system_id!r}: {exc}"
            ) from exc

    def adopt_authoritative_state(self, completed: "CombatSystem") -> None:
        """Commit a completed shard while preserving this authority object's identity."""
        if not isinstance(completed, CombatSystem) or completed._system_id != self._system_id:
            raise CombatStateCloneError("cannot adopt combat state from a different system")
        preserved = {
            "logger": self.logger,
            "_combat_event_sink": self._combat_event_sink,
            "detailed_logging": self.detailed_logging,
            "hotspot_logging_enabled": self.hotspot_logging_enabled,
            "event_logging_enabled": self.event_logging_enabled,
        }
        for name, value in completed.__dict__.items():
            if name in preserved:
                continue
            setattr(self, name, value)
        for name, value in preserved.items():
            setattr(self, name, value)
