from __future__ import annotations

from .authoritative_tick import AuthoritativeTickMixin
from .bubbles import BubblesMixin
from .command_bursts import CommandBurstsMixin
from .combat_common import *  # noqa: F403
from .damage_missile import DamageMissileMixin
from .damage_turret import DamageTurretMixin
from .ewar import EwarMixin
from .locking import LockingMixin
from .logistics import CombatLogisticsMixin
from .module_cycles import ModuleCyclesMixin
from .projectiles import ProjectilesMixin
from .runtime_projection import RuntimeProjectionMixin

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
