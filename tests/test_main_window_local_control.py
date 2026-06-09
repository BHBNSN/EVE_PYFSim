from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

from eve_sim.agents import CommanderAgent
from eve_sim.config import EngineConfig, UiConfig
from eve_sim.fit_runtime import EffectClass, FitRuntime, HullProfile, ModuleEffect, ModuleRuntime, ModuleState, SkillProfile
from eve_sim.gui.main_window import MainWindow
from eve_sim.gui.models import UiPreferences
from eve_sim.lan_commands import (
    CMD_SET_MODULE_CHARGE_LOCK,
    CMD_SET_MODULE_MANUAL_MODE,
    CMD_SET_MODULE_TARGET_MODE,
    CMD_SYNC_MODULE_CONTROLS,
)
from eve_sim.math2d import Vector2
from eve_sim.models import CombatState, FitDescriptor, NavigationState, Order, QualityLevel, QualityState, ShipEntity, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem
from eve_sim.world import WorldState


def _make_ship(ship_id: str, team: Team, squad_id: str, position: Vector2) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name="Test Hull",
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_speed=150.0,
        max_cap=100.0,
        cap_recharge_time=1_000.0,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
    )
    profile = PyfaBridge().build_profile(fit)
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id=squad_id,
        fit=fit,
        profile=profile,
        nav=NavigationState(
            position=position,
            velocity=Vector2(0.0, 0.0),
            facing_deg=0.0,
            max_speed=profile.max_speed,
        ),
        combat=CombatState(),
        vital=VitalState(
            shield=profile.shield_hp,
            armor=profile.armor_hp,
            structure=profile.structure_hp,
            shield_max=profile.shield_hp,
            armor_max=profile.armor_hp,
            structure_max=profile.structure_hp,
            cap=profile.max_cap,
            cap_max=profile.max_cap,
            alive=True,
        ),
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
        runtime=None,
    )


def _make_runtime_ship(ship_id: str, team: Team, squad_id: str, position: Vector2, fit_key: str) -> ShipEntity:
    ship = _make_ship(ship_id, team, squad_id, position)
    ship.fit.fit_key = fit_key
    hull = HullProfile(
        ship_name=ship.fit.ship_name,
        role=ship.fit.role,
        base_dps=ship.fit.base_dps,
        volley=ship.fit.volley,
        optimal=ship.fit.optimal_range,
        falloff=ship.fit.falloff,
        tracking=ship.fit.tracking,
        sig_radius=ship.fit.signature_radius,
        scan_resolution=ship.fit.scan_resolution,
        max_target_range=ship.fit.max_target_range,
        max_speed=ship.fit.max_speed,
        cap_max=ship.fit.max_cap,
        cap_recharge_time=ship.fit.cap_recharge_time,
        shield_hp=ship.fit.shield_hp,
        armor_hp=ship.fit.armor_hp,
        structure_hp=ship.fit.structure_hp,
        rep_amount=0.0,
        rep_cycle=5.0,
    )
    ship.runtime = FitRuntime(
        fit_key=fit_key,
        hull=hull,
        skills=SkillProfile(),
        modules=[
            ModuleRuntime(
                module_id="mod-1",
                group="propulsion module",
                state=ModuleState.ONLINE,
                effects=[
                    ModuleEffect(
                        name="prop-a",
                        effect_class=EffectClass.LOCAL,
                        state_required=ModuleState.ACTIVE,
                        cycle_time=10.0,
                        local_mult={"speed": 1.5},
                    )
                ],
            )
        ],
        diagnostics={},
    )
    return ship


def _make_targeting_runtime_ship(ship_id: str, team: Team, squad_id: str, position: Vector2, fit_key: str) -> ShipEntity:
    ship = _make_ship(ship_id, team, squad_id, position)
    ship.fit.fit_key = fit_key
    hull = HullProfile(
        ship_name=ship.fit.ship_name,
        role=ship.fit.role,
        base_dps=ship.fit.base_dps,
        volley=ship.fit.volley,
        optimal=ship.fit.optimal_range,
        falloff=ship.fit.falloff,
        tracking=ship.fit.tracking,
        sig_radius=ship.fit.signature_radius,
        scan_resolution=ship.fit.scan_resolution,
        max_target_range=ship.fit.max_target_range,
        max_speed=ship.fit.max_speed,
        cap_max=ship.fit.max_cap,
        cap_recharge_time=ship.fit.cap_recharge_time,
        shield_hp=ship.fit.shield_hp,
        armor_hp=ship.fit.armor_hp,
        structure_hp=ship.fit.structure_hp,
        rep_amount=0.0,
        rep_cycle=5.0,
    )
    ship.runtime = FitRuntime(
        fit_key=fit_key,
        hull=hull,
        skills=SkillProfile(),
        modules=[
            ModuleRuntime(
                module_id="mod-1",
                group="sensor dampener",
                state=ModuleState.ONLINE,
                effects=[
                    ModuleEffect(
                        name="damp-a",
                        effect_class=EffectClass.PROJECTED,
                        state_required=ModuleState.ACTIVE,
                        cycle_time=10.0,
                        range_m=50_000.0,
                        projected_mult={"scan": 0.8, "range": 0.8},
                    )
                ],
                tags=("controlled", "hostile", "offensive_ewar", "projected"),
            )
        ],
        diagnostics={},
    )
    return ship


class MainWindowLocalControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_language_combo_uses_native_labels_and_has_safe_min_width(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "blue-fit")
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0), "red-fit")
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            self.assertEqual(window.lang_combo.itemText(0), "简体中文")
            self.assertEqual(window.lang_combo.itemText(1), "English")
            self.assertGreaterEqual(window.lang_combo.minimumWidth(), 140)
            window.retranslate_ui()
            self.assertEqual(window.lang_combo.itemText(0), "简体中文")
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    @patch("eve_sim.gui.main_window.OverviewOptionsDialog")
    def test_overview_filters_dialog_can_open(
        self,
        overview_dialog_cls,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0))
        world = WorldState(ships={blue_ship.ship_id: blue_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        engine.register_ship(blue_ship.ship_id)

        dialog_instance = overview_dialog_cls.return_value
        dialog_instance.exec.return_value = 0

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window.open_overview_options()
            overview_dialog_cls.assert_called_once()
            dialog_instance.exec.assert_called_once()
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="ALL", selected_squad="BLUE-ALPHA"),
    )
    def test_overview_selection_highlights_ship_on_canvas_and_survives_refresh(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "blue-fit")
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0), "red-fit")
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window.request_overview_refresh(force=True)
            target_row = next(
                row for row in range(window.overview_proxy.rowCount())
                if window.overview_proxy.get_row(row)["id"] == red_ship.ship_id
            )

            window.overview.selectRow(target_row)

            self.assertEqual(window.canvas.selected_ship_id, red_ship.ship_id)

            window.request_overview_refresh(force=True)

            self.assertEqual(window.canvas.selected_ship_id, red_ship.ship_id)
            current_row = window.overview.currentIndex().row()
            self.assertEqual(window.overview_proxy.get_row(current_row)["id"], red_ship.ship_id)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="ALL", selected_squad="BLUE-ALPHA"),
    )
    def test_overview_and_fleet_tables_show_ship_name_column(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "blue-fit")
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0), "red-fit")
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.discard(blue_ship.ship_id)
            window.engine.world.ships[blue_ship.ship_id].vital.alive = True
            window.refresh_blue_roster()
            window.request_overview_refresh(force=True)

            self.assertEqual(window.overview_model.headerData(1, Qt.Orientation.Horizontal), "Name")
            self.assertEqual(window.blue_roster_model.headerData(1, Qt.Orientation.Horizontal), "Ship Name")

            overview_row = next(
                row for row in (window.overview_proxy.get_row(idx) for idx in range(window.overview_proxy.rowCount()))
                if row is not None and row["id"] == red_ship.ship_id
            )
            self.assertEqual(overview_row["ship_name_display"], red_ship.fit.ship_name)

            blue_roster_row = window.blue_roster_model.get_row(0)
            self.assertIsNotNone(blue_roster_row)
            self.assertEqual(blue_roster_row["ship_name_display"], blue_ship.fit.ship_name)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_blue_roster_selection_highlights_ships_on_canvas_and_survives_refresh(
        self,
        _load,
        _save,
    ) -> None:
        blue_a = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "fit-a")
        blue_b = _make_runtime_ship("blue-2", Team.BLUE, "BLUE-ALPHA", Vector2(100.0, 0.0), "fit-b")
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0), "red-fit")
        world = WorldState(ships={blue_a.ship_id: blue_a, blue_b.ship_id: blue_b, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            for ship_id in (blue_a.ship_id, blue_b.ship_id):
                window._undeployed_ship_ids.discard(ship_id)
                window.engine.world.ships[ship_id].vital.alive = True
            window.refresh_blue_roster()

            selection_model = window.blue_roster.selectionModel()
            self.assertIsNotNone(selection_model)
            first_index = window.blue_roster_model.index(0, 0)
            second_index = window.blue_roster_model.index(1, 0)
            selection_model.select(
                first_index,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
            selection_model.select(
                second_index,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )

            self.assertEqual(window.canvas.highlighted_roster_ship_ids, {blue_a.ship_id, blue_b.ship_id})

            window.refresh_blue_roster()

            self.assertEqual(window.canvas.highlighted_roster_ship_ids, {blue_a.ship_id, blue_b.ship_id})
            selected_ids = {
                str(window.blue_roster_model.get_row(index.row())["ship_id"])
                for index in window.blue_roster.selectionModel().selectedRows()
            }
            self.assertEqual(selected_ids, {blue_a.ship_id, blue_b.ship_id})
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="ALL", selected_squad="BLUE-ALPHA"),
    )
    def test_common_ship_context_menu_matches_enemy_and_friendly_cases(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "blue-fit")
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0), "red-fit")
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.discard(blue_ship.ship_id)
            window.engine.world.ships[blue_ship.ship_id].vital.alive = True
            window.ui_state.selected_squad = "BLUE-ALPHA"
            window.canvas.selected_squad = "BLUE-ALPHA"
            focus_key = f"{Team.BLUE.value}:BLUE-ALPHA"
            window.engine.world.squad_focus_queues[focus_key] = [red_ship.ship_id]
            window.engine.world.squad_prelocked_targets[focus_key] = {"blue-1": {red_ship.ship_id}}

            friendly_menu = window._build_ship_context_menu(blue_ship.ship_id)
            enemy_menu = window._build_ship_context_menu(red_ship.ship_id)

            self.assertIsNotNone(enemy_menu)
            self.assertIsNotNone(friendly_menu)
            enemy_texts = [action.text() for action in enemy_menu.actions()]
            friendly_texts = [action.text() for action in friendly_menu.actions()]

            self.assertEqual(len(friendly_texts), 2)
            self.assertEqual(len(enemy_texts), 5)
            self.assertTrue(all(red_ship.ship_id in text for text in enemy_texts))
            self.assertTrue(all(blue_ship.ship_id in text for text in friendly_texts))
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="RED-ALPHA"),
    )
    def test_client_charge_lock_with_unloaded_option_sends_lan_command(
        self,
        _load,
        _save,
    ) -> None:
        class _FakeLanClient:
            def __init__(self) -> None:
                self.sent_commands: list[dict] = []

            def send_command(self, command: dict) -> None:
                self.sent_commands.append(dict(command))

            def close(self) -> None:
                return None

        red_ship = _make_ship("red-1", Team.RED, "RED-ALPHA", Vector2(0.0, 0.0))
        world = WorldState(ships={red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=[])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        lan_client = _FakeLanClient()
        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="client",
            controlled_team=Team.RED,
            lan_client=lan_client,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._ship_fit_texts[red_ship.ship_id] = "[Onyx, Test]\nWarp Disruption Field Generator I\n"

            ok, _message = window._set_ship_module_charge_lock(red_ship.ship_id, "mod-1", "")

            self.assertTrue(ok)
            self.assertEqual(
                lan_client.sent_commands[-1],
                {
                    "kind": CMD_SET_MODULE_CHARGE_LOCK,
                    "ship_id": red_ship.ship_id,
                    "module_id": "mod-1",
                    "charge_name": "",
                },
            )
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_host_mode_cannot_modify_remote_team_module_controls(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "blue-fit")
        red_ship = _make_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(100.0, 0.0), "red-fit")
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="host",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            ok, message = window._set_ship_module_manual_mode(red_ship.ship_id, "mod-1", "active")
            self.assertFalse(ok)
            self.assertTrue(message)
            self.assertNotIn("mod-1", red_ship.combat.module_manual_modes)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="RED-ALPHA"),
    )
    def test_client_module_control_changes_send_lan_commands(
        self,
        _load,
        _save,
    ) -> None:
        class _FakeLanClient:
            def __init__(self) -> None:
                self.sent_commands: list[dict] = []

            def send_command(self, command: dict) -> None:
                self.sent_commands.append(dict(command))

            def close(self) -> None:
                return None

        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(0.0, 0.0), "shared-fit")
        world = WorldState(ships={red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=[])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        engine.register_ship(red_ship.ship_id)

        lan_client = _FakeLanClient()
        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="client",
            controlled_team=Team.RED,
            lan_client=lan_client,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()

            ok_mode, _ = window._set_ship_module_manual_mode(red_ship.ship_id, "mod-1", "active")
            ok_target, _ = window._set_ship_module_target_mode(red_ship.ship_id, "mod-1", "enemy_nearest")
            ok_sync, _ = window._sync_ship_module_controls_to_matching_squad_fit(
                red_ship.ship_id,
                "mod-1",
                "online",
                "enemy_nearest",
            )

            self.assertTrue(ok_mode)
            self.assertTrue(ok_target)
            self.assertTrue(ok_sync)
            self.assertEqual(
                lan_client.sent_commands,
                [
                    {
                        "kind": CMD_SET_MODULE_MANUAL_MODE,
                        "ship_id": red_ship.ship_id,
                        "module_id": "mod-1",
                        "mode": "active",
                    },
                    {
                        "kind": CMD_SET_MODULE_TARGET_MODE,
                        "ship_id": red_ship.ship_id,
                        "module_id": "mod-1",
                        "mode": "enemy_nearest",
                    },
                    {
                        "kind": CMD_SYNC_MODULE_CONTROLS,
                        "ship_id": red_ship.ship_id,
                        "module_id": "mod-1",
                        "mode": "online",
                        "target_mode": "enemy_nearest",
                    },
                ],
            )
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="RED-ALPHA"),
    )
    def test_client_snapshot_applies_module_control_overrides(
        self,
        _load,
        _save,
    ) -> None:
        red_ship = _make_targeting_runtime_ship("red-1", Team.RED, "RED-ALPHA", Vector2(0.0, 0.0), "shared-fit")
        world = WorldState(ships={red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=[])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        engine.register_ship(red_ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="client",
            controlled_team=Team.RED,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._apply_remote_snapshot(
                {
                    "snapshot": {
                        "tick": 1,
                        "now": 1.0,
                        "ships": {
                            red_ship.ship_id: {
                                "team": Team.RED.value,
                                "squad_id": red_ship.squad_id,
                                "position": {"x": 0.0, "y": 0.0},
                                "velocity": {"x": 0.0, "y": 0.0},
                                "facing_deg": 0.0,
                                "shield": red_ship.vital.shield,
                                "armor": red_ship.vital.armor,
                                "structure": red_ship.vital.structure,
                                "shield_max": red_ship.vital.shield_max,
                                "armor_max": red_ship.vital.armor_max,
                                "structure_max": red_ship.vital.structure_max,
                                "cap": red_ship.vital.cap,
                                "cap_max": red_ship.vital.cap_max,
                                "alive": True,
                                "deployed": True,
                                "module_manual_modes": {"mod-1": "active"},
                                "module_target_modes": {"mod-1": "enemy_nearest"},
                            }
                        },
                    }
                }
            )

            self.assertEqual(window.engine.world.ships[red_ship.ship_id].combat.module_manual_modes.get("mod-1"), "active")
            self.assertEqual(window.engine.world.ships[red_ship.ship_id].combat.module_target_modes.get("mod-1"), "enemy_nearest")
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_local_mode_can_switch_controlled_team_and_refresh_overview(
        self,
        _load,
        _save,
    ) -> None:
        blue_ship = _make_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0))
        red_ship = _make_ship("red-1", Team.RED, "RED-ALPHA", Vector2(10_000.0, 0.0))
        world = WorldState(ships={blue_ship.ship_id: blue_ship, red_ship.ship_id: red_ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            blue_ship.vital.alive = True
            red_ship.vital.alive = True
            window._sync_blue_squads()
            window.refresh_blue_roster()
            window.overview_proxy.apply_preferences()
            window.request_overview_refresh(force=True)

            self.assertEqual(window.controlled_team, Team.BLUE)
            self.assertEqual(window.squad_combo.currentText(), "BLUE-ALPHA")
            self.assertEqual(window.lbl_controlled_team_value.text(), window._team_display_text(Team.BLUE))
            self.assertEqual(
                window.blue_roster_model.get_row(0),
                {
                    "ship_id": "blue-1",
                    "ship_name": "Test Hull",
                    "ship_name_display": "Test Hull",
                    "squad": "BLUE-ALPHA",
                    "role": "test",
                    "alive": True,
                    "hp": 100.0,
                },
            )
            self.assertEqual(window.overview_proxy.rowCount(), 1)
            self.assertEqual(window.overview_proxy.get_row(0)["team"], Team.BLUE.value)

            window.ui_state.selected_enemy_target = red_ship.ship_id
            window.canvas.selected_enemy_target = red_ship.ship_id
            window.toggle_local_controlled_team()

            self.assertEqual(window.controlled_team, Team.RED)
            self.assertEqual(window.squad_combo.currentText(), "RED-ALPHA")
            self.assertEqual(window.lbl_controlled_team_value.text(), window._team_display_text(Team.RED))
            self.assertEqual(
                window.blue_roster_model.get_row(0),
                {
                    "ship_id": "red-1",
                    "ship_name": "Test Hull",
                    "ship_name_display": "Test Hull",
                    "squad": "RED-ALPHA",
                    "role": "test",
                    "alive": True,
                    "hp": 100.0,
                },
            )
            self.assertEqual(window.overview_proxy.rowCount(), 1)
            self.assertEqual(window.overview_proxy.get_row(0)["team"], Team.RED.value)
            self.assertIsNone(window.ui_state.selected_enemy_target)
            self.assertIsNone(window.canvas.selected_enemy_target)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_module_control_sync_preserves_current_target_rule_when_only_mode_changes(
        self,
        _load,
        _save,
    ) -> None:
        blue_a = _make_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "shared-fit")
        blue_b = _make_runtime_ship("blue-2", Team.BLUE, "BLUE-ALPHA", Vector2(100.0, 0.0), "shared-fit")
        blue_c = _make_runtime_ship("blue-3", Team.BLUE, "BLUE-ALPHA", Vector2(200.0, 0.0), "other-fit")
        world = WorldState(ships={ship.ship_id: ship for ship in (blue_a, blue_b, blue_c)})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            blue_a.combat.module_target_modes["mod-1"] = "enemy_nearest"
            ok, message = window._sync_ship_module_controls_to_matching_squad_fit("blue-1", "mod-1", "active", "enemy_nearest")

            self.assertTrue(ok)
            self.assertIn("2", message)
            self.assertEqual(blue_a.combat.module_manual_modes.get("mod-1"), "active")
            self.assertEqual(blue_b.combat.module_manual_modes.get("mod-1"), "active")
            self.assertNotIn("mod-1", blue_c.combat.module_manual_modes)
            self.assertNotIn("mod-1", blue_a.combat.module_target_modes)
            self.assertNotIn("mod-1", blue_b.combat.module_target_modes)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_module_control_sync_applies_mode_and_target_rule_to_same_squad_same_initial_fit(
        self,
        _load,
        _save,
    ) -> None:
        blue_a = _make_targeting_runtime_ship("blue-1", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), "shared-fit")
        blue_b = _make_targeting_runtime_ship("blue-2", Team.BLUE, "BLUE-ALPHA", Vector2(100.0, 0.0), "shared-fit")
        blue_c = _make_targeting_runtime_ship("blue-3", Team.BLUE, "BLUE-ALPHA", Vector2(200.0, 0.0), "other-fit")
        world = WorldState(ships={ship.ship_id: ship for ship in (blue_a, blue_b, blue_c)})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            ok, message = window._sync_ship_module_controls_to_matching_squad_fit("blue-1", "mod-1", "active", "enemy_nearest")

            self.assertTrue(ok)
            self.assertIn("2", message)
            self.assertEqual(blue_a.combat.module_manual_modes.get("mod-1"), "active")
            self.assertEqual(blue_b.combat.module_manual_modes.get("mod-1"), "active")
            self.assertNotIn("mod-1", blue_c.combat.module_manual_modes)
            self.assertEqual(blue_a.combat.module_target_modes.get("mod-1"), "enemy_nearest")
            self.assertEqual(blue_b.combat.module_target_modes.get("mod-1"), "enemy_nearest")
            self.assertNotIn("mod-1", blue_c.combat.module_target_modes)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA"),
    )
    def test_issue_warp_only_queues_members_beyond_min_distance_and_replaces_move_attack_orders(
        self,
        _load,
        _save,
    ) -> None:
        blue_far = _make_ship("blue-far", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0))
        blue_near = _make_ship("blue-near", Team.BLUE, "BLUE-ALPHA", Vector2(140_000.0, 0.0))
        red_target = _make_ship("red-target", Team.RED, "RED-ALPHA", Vector2(250_000.0, 0.0))
        blue_far.order_queue = [
            Order(kind="MOVE", payload={"x": 1.0, "y": 2.0}, issue_time=0.0),
            Order(kind="ATTACK", payload={"target_id": red_target.ship_id}, issue_time=0.0),
        ]
        world = WorldState(ships={ship.ship_id: ship for ship in (blue_far, blue_near, red_target)})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-ALPHA"])
        engine.register_commander(blue_commander)
        engine.register_commander(red_commander)
        for ship_id in world.ships:
            engine.register_ship(ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
            network_mode="local",
            controlled_team=Team.BLUE,
        )

        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            blue_far.vital.alive = True
            blue_near.vital.alive = True
            red_target.vital.alive = True
            window._sync_blue_squads()
            window.refresh_blue_roster()
            scoped_key = window._focus_key(Team.BLUE, "BLUE-ALPHA")
            window._squad_approach_targets[scoped_key] = red_target.ship_id

            window.issue_warp_to_ship("BLUE-ALPHA", red_target.ship_id)
            window._flush_tick_ops()

            self.assertNotIn(scoped_key, window._squad_approach_targets)
            self.assertEqual(len(blue_far.order_queue), 1)
            self.assertEqual(blue_far.order_queue[0].kind, "WARP")
            self.assertEqual(blue_far.order_queue[0].payload.get("target_ship_id"), red_target.ship_id)
            self.assertEqual(blue_near.order_queue, [])
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
