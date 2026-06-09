from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QCoreApplication, QTimer, Qt
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ..config import EngineConfig, UiConfig
from ..maps import deserialize_map_definition, instantiate_structures
from ..math2d import Vector2
from ..models import Team
from ..pyfa_bridge import PyfaBridge
from ..replay import ReplayPlayer
from ..replay.snapshot_mapper import apply_snapshot_to_world
from ..simulation_engine import SimulationEngine
from ..systems import CombatSystem
from ..world import WorldState
from .battle_canvas import BattleCanvas


class ReplayPlaybackDialog(QDialog):
    def __init__(
        self,
        player: ReplayPlayer,
        ui_cfg: UiConfig,
        language_getter: Callable[[], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.player = player
        self._language_getter = language_getter
        self._current_index = 0
        self._playing = False
        self._frame_accumulator = 0.0

        self.world = WorldState()
        self._install_map_metadata()
        canvas_cfg = UiConfig(
            width=min(int(ui_cfg.width), 1200),
            height=min(int(ui_cfg.height), 760),
            world_to_screen_scale=float(ui_cfg.world_to_screen_scale),
        )
        self.engine = SimulationEngine(
            self.world,
            EngineConfig(physics_substeps=1),
            CombatSystem(PyfaBridge()),
        )

        self.canvas = BattleCanvas(
            self.engine,
            canvas_cfg,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            lambda *_args: None,
            self._controlled_squads,
            lambda _ship_id: True,
            lambda _squad_id: None,
            lambda *_args: None,
            self._language_getter,
            lambda: Team.BLUE,
            self._select_squad,
            lambda *_args: None,
            parent=self,
        )

        self.setWindowTitle(QCoreApplication.translate("eve_sim", "Battle Replay"))
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.btn_back_big = QPushButton("<< 10")
        self.btn_back = QPushButton("<")
        self.btn_play = QPushButton(QCoreApplication.translate("eve_sim", "Play"))
        self.btn_forward = QPushButton(">")
        self.btn_forward_big = QPushButton("10 >>")
        self.speed_combo = QComboBox(self)
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            self.speed_combo.addItem(f"{speed:g}x", speed)
        self.speed_combo.setCurrentIndex(2)
        self.lbl_position = QLabel("")
        controls.addWidget(self.btn_back_big)
        controls.addWidget(self.btn_back)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_forward)
        controls.addWidget(self.btn_forward_big)
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.lbl_position, 1)
        layout.addLayout(controls)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.player.snapshot_count - 1))
        layout.addWidget(self.slider)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.btn_back_big.clicked.connect(lambda: self.step(-10))
        self.btn_back.clicked.connect(lambda: self.step(-1))
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_forward.clicked.connect(lambda: self.step(1))
        self.btn_forward_big.clicked.connect(lambda: self.step(10))
        self.slider.valueChanged.connect(self.seek_to)

        if self.player.snapshot_count:
            self.seek_to(0)
            self.canvas.fit_to_map()
        else:
            self._set_controls_enabled(False)
            self.lbl_position.setText(QCoreApplication.translate("eve_sim", "No replay snapshots"))

    def _install_map_metadata(self) -> None:
        payload = self.player.metadata.get("map")
        if not isinstance(payload, dict):
            return
        try:
            map_definition = deserialize_map_definition(payload)
        except Exception:
            return
        self.world.map_definition = map_definition
        self.world.map_id = str(map_definition.map_id)
        self.world.map_name = str(map_definition.name)
        self.world.structures = instantiate_structures(map_definition)

    def _controlled_squads(self) -> list[str]:
        return sorted({ship.squad_id for ship in self.world.ships.values() if ship.team == Team.BLUE and ship.squad_id})

    def _select_squad(self, squad_id: str) -> None:
        self.canvas.selected_squad = str(squad_id)

    def _speed(self) -> float:
        return float(self.speed_combo.currentData() or 1.0)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.btn_back_big, self.btn_back, self.btn_play, self.btn_forward, self.btn_forward_big, self.speed_combo, self.slider):
            widget.setEnabled(enabled)

    def _set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if self._playing:
            self.btn_play.setText(QCoreApplication.translate("eve_sim", "Pause"))
            self.timer.start(33)
        else:
            self.btn_play.setText(QCoreApplication.translate("eve_sim", "Play"))
            self.timer.stop()

    def toggle_playback(self) -> None:
        if not self.player.snapshot_count:
            return
        if self._current_index >= self.player.snapshot_count - 1 and not self._playing:
            self.seek_to(0)
        self._set_playing(not self._playing)

    def seek_to(self, index: int) -> None:
        if not self.player.snapshot_count:
            return
        self._current_index = max(0, min(int(index), self.player.snapshot_count - 1))
        snapshot = self.player.snapshot_at_index(self._current_index)
        apply_snapshot_to_world(self.world, snapshot.snapshot)
        if not self.canvas.current_view_system_id:
            for ship in self.world.ships.values():
                self.canvas.current_view_system_id = str(getattr(ship.nav, "system_id", "") or "")
                break
        self.slider.blockSignals(True)
        self.slider.setValue(self._current_index)
        self.slider.blockSignals(False)
        self.lbl_position.setText(
            QCoreApplication.translate("eve_sim", "Frame {frame}/{total} | Tick {tick} | t={time:.2f}s | {speed}x").format(
                frame=self._current_index + 1,
                total=self.player.snapshot_count,
                tick=int(snapshot.tick),
                time=float(snapshot.at),
                speed=f"{self._speed():g}",
            )
        )
        self.canvas.update()

    def step(self, delta: int) -> None:
        self.seek_to(self._current_index + int(delta))

    def _on_timer(self) -> None:
        if not self._playing:
            return
        self._frame_accumulator += self._speed()
        frames = max(1, int(self._frame_accumulator))
        self._frame_accumulator = max(0.0, self._frame_accumulator - frames)
        self.seek_to(self._current_index + frames)
        if self._current_index >= self.player.snapshot_count - 1:
            self._set_playing(False)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
