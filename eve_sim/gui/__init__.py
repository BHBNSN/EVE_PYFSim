from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QMessageBox,
    QVBoxLayout,
)

from .fleet_setup_dialog import FleetSetupDialog
from .language_controls import language_icon_label
from .main_window import MainWindow
from ..agents import CommanderAgent
from ..config import UiConfig
from ..fleet_setup import build_world_from_manual_setup
from ..i18n import detect_system_language, install_language, language_options, normalize_language
from ..lan_session import ClientLanSession, HostLanSession
from ..models import Team
from ..pyfa_bridge import PyfaBridge
from ..simulation_engine import SimulationEngine
from ..systems import CombatSystem


class _StartupModeDialog(QDialog):
    def __init__(self, initial_language: str, parent=None) -> None:
        super().__init__(parent)
        self._lang = normalize_language(initial_language, "en_US")
        self.setModal(True)
        self.resize(360, 140)

        layout = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        self.lbl_mode = QLabel(self)
        self.mode_combo = QComboBox(self)
        mode_row.addWidget(self.lbl_mode)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        lang_row = QHBoxLayout()
        lang_row.addWidget(language_icon_label(self))
        self.lbl_language = QLabel("Language", self)
        self.lang_combo = QComboBox(self)
        lang_row.addWidget(self.lbl_language)
        lang_row.addWidget(self.lang_combo, 1)
        layout.addLayout(lang_row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        layout.addWidget(self.buttons)

        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._apply_language()

    def _refresh_language_combo(self, selected_lang: str | None = None) -> None:
        target_lang = normalize_language(selected_lang or self._lang, "en_US")
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        for label, lang_code in language_options():
            self.lang_combo.addItem(label, lang_code)
        idx = self.lang_combo.findData(target_lang)
        self.lang_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.lang_combo.blockSignals(False)

    def _refresh_mode_combo(self) -> None:
        current_mode = str(self.mode_combo.currentData() or "local")
        tr = lambda text: QCoreApplication.translate("eve_sim", text)
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItem(tr("Local"), "local")
        self.mode_combo.addItem(tr("Host LAN"), "host")
        self.mode_combo.addItem(tr("Join LAN"), "client")
        idx = self.mode_combo.findData(current_mode)
        self.mode_combo.setCurrentIndex(0 if idx < 0 else idx)
        self.mode_combo.blockSignals(False)

    def _apply_language(self) -> None:
        tr = lambda text: QCoreApplication.translate("eve_sim", text)
        self.setWindowTitle(tr("Battle Mode"))
        self.lbl_mode.setText(tr("Select mode"))
        self.lbl_language.setText("Language")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("OK"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("Cancel"))
        self._refresh_mode_combo()
        self._refresh_language_combo(self._lang)

    def _on_language_changed(self, _index: int) -> None:
        self._lang = install_language(str(self.lang_combo.currentData() or "en_US"))
        self._apply_language()

    def selected_mode(self) -> str:
        return str(self.mode_combo.currentData() or "local")

    def selected_language(self) -> str:
        return normalize_language(str(self.lang_combo.currentData() or self._lang), "en_US")


def run_gui() -> None:
    app = QApplication.instance() or QApplication([])
    initial_language = install_language(detect_system_language())

    tr = lambda text: QCoreApplication.translate("eve_sim", text)

    startup_dialog = _StartupModeDialog(initial_language)
    if startup_dialog.exec() != QDialog.DialogCode.Accepted:
        return
    selected_language = install_language(startup_dialog.selected_language())
    selected_mode = startup_dialog.selected_mode()

    network_mode = "local"
    controlled_team = Team.BLUE
    lan_server: HostLanSession | None = None
    lan_client: ClientLanSession | None = None

    if selected_mode == "host":
        port, ok = QInputDialog.getInt(None, tr("Host LAN"), tr("Port"), 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_server = HostLanSession(host="0.0.0.0", port=int(port))
        try:
            lan_server.start()
        except OSError as exc:
            QMessageBox.critical(
                None,
                tr("Host LAN"),
                tr("Failed to open LAN host: {error}").format(error=exc),
            )
            return
        network_mode = "host"
        controlled_team = Team.BLUE
    elif selected_mode == "client":
        host, ok = QInputDialog.getText(None, tr("Join LAN"), tr("Host IP"), text="127.0.0.1")
        if not ok or not host.strip():
            return
        port, ok = QInputDialog.getInt(None, tr("Join LAN"), tr("Port"), 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_client = ClientLanSession(host=host.strip(), port=int(port))
        if not lan_client.connect(timeout=5.0):
            QMessageBox.critical(None, tr("Join LAN"), tr("Failed to connect to host"))
            lan_client.close()
            return
        network_mode = "client"
        controlled_team = Team.RED

    setup_dialog = FleetSetupDialog(network_mode=network_mode, initial_language=selected_language)
    if setup_dialog.exec() != QDialog.DialogCode.Accepted:
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return

    selected_language = install_language(setup_dialog.selected_language())
    manual_setup = setup_dialog.to_manual_setup()
    if network_mode == "host":
        manual_setup = [row for row in manual_setup if row.team == Team.BLUE]
    elif network_mode == "client":
        manual_setup = [row for row in manual_setup if row.team == Team.RED]
    if not manual_setup:
        QMessageBox.critical(
            None,
            tr("Fleet Setup"),
            tr("No ships found for current side in this mode"),
        )
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return
    pyfa = PyfaBridge()
    cfg = setup_dialog.to_engine_config()
    selected_map = setup_dialog.selected_map_definition()
    world = build_world_from_manual_setup(manual_setup, map_definition=selected_map)
    engine = SimulationEngine(world=world, config=cfg, combat_system=CombatSystem(pyfa))

    blue_squads = sorted({s.squad_id for s in world.ships.values() if s.team == Team.BLUE})
    red_squads = sorted({s.squad_id for s in world.ships.values() if s.team == Team.RED})
    blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=blue_squads)
    red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=red_squads)
    engine.register_commander(blue_commander)
    engine.register_commander(red_commander)
    for ship_id in world.ships:
        engine.register_ship(ship_id)

    win = MainWindow(
        engine=engine,
        ui_cfg=UiConfig(),
        blue_commander=blue_commander,
        red_commander=red_commander,
        manual_setup=manual_setup,
        network_mode=network_mode,
        controlled_team=controlled_team,
        lan_server=lan_server,
        lan_client=lan_client,
        initial_language=selected_language,
    )
    win.show()
    app.exec()


__all__ = ["run_gui"]
