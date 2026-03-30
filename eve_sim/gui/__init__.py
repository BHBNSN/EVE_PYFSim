from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox

from .fleet_setup_dialog import FleetSetupDialog
from .main_window import MainWindow
from ..agents import CommanderAgent
from ..config import UiConfig
from ..fleet_setup import build_world_from_manual_setup
from ..i18n import install_language
from ..lan_session import ClientLanSession, HostLanSession
from ..models import Team
from ..pyfa_bridge import PyfaBridge
from ..simulation_engine import SimulationEngine
from ..systems import CombatSystem

def run_gui() -> None:
    app = QApplication.instance() or QApplication([])
    install_language("zh_CN")

    mode_options = ["Local", "Host LAN", "Join LAN"]
    mode, ok = QInputDialog.getItem(None, "Battle Mode", "Select mode", mode_options, 0, False)
    if not ok:
        return

    network_mode = "local"
    controlled_team = Team.BLUE
    lan_server: HostLanSession | None = None
    lan_client: ClientLanSession | None = None

    if mode == "Host LAN":
        port, ok = QInputDialog.getInt(None, "Host LAN", "Port", 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_server = HostLanSession(host="0.0.0.0", port=int(port))
        try:
            lan_server.start()
        except OSError as exc:
            QMessageBox.critical(None, "Host LAN", f"Failed to open LAN host: {exc}")
            return
        network_mode = "host"
        controlled_team = Team.BLUE
    elif mode == "Join LAN":
        host, ok = QInputDialog.getText(None, "Join LAN", "Host IP", text="127.0.0.1")
        if not ok or not host.strip():
            return
        port, ok = QInputDialog.getInt(None, "Join LAN", "Port", 50555, 1024, 65535, 1)
        if not ok:
            return
        lan_client = ClientLanSession(host=host.strip(), port=int(port))
        if not lan_client.connect(timeout=5.0):
            QMessageBox.critical(None, "Join LAN", "Failed to connect to host")
            lan_client.close()
            return
        network_mode = "client"
        controlled_team = Team.RED

    setup_dialog = FleetSetupDialog(network_mode=network_mode)
    if setup_dialog.exec() != QDialog.DialogCode.Accepted:
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return

    manual_setup = setup_dialog.to_manual_setup()
    if network_mode == "host":
        manual_setup = [row for row in manual_setup if row.team == Team.BLUE]
    elif network_mode == "client":
        manual_setup = [row for row in manual_setup if row.team == Team.RED]
    if not manual_setup:
        QMessageBox.critical(None, "Fleet Setup", "No ships found for current side in this mode")
        if lan_server is not None:
            lan_server.stop()
        if lan_client is not None:
            lan_client.close()
        return
    pyfa = PyfaBridge()
    world = build_world_from_manual_setup(manual_setup)
    cfg = setup_dialog.to_engine_config()
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
    )
    win.show()
    app.exec()


__all__ = ["run_gui"]
