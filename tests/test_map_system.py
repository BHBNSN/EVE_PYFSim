from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from eve_sim.agents import CommanderAgent
from eve_sim.config import EngineConfig, UiConfig
from eve_sim.fleet_setup import build_world_from_manual_setup
from eve_sim.gui.main_window import MainWindow
from eve_sim.gui.models import UiPreferences
from eve_sim.maps.models import MapDefinition, MapSystemDefinition
from eve_sim.maps import load_map_definition
from eve_sim.math2d import Vector2
from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem
from eve_sim.systems.movement import MovementSystem
from eve_sim.systems.perception import PerceptionSystem
from eve_sim.world import WorldState


def _make_ship(ship_id: str, team: Team, squad_id: str, position: Vector2, *, system_id: str) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name="Test Hull",
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_speed=200.0,
        max_cap=100.0,
        cap_recharge_time=1_000.0,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
        warp_speed_au_s=3.0,
        warp_capacitor_need=0.000001,
        mass=10_000_000.0,
        agility=0.5,
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
            system_id=system_id,
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


class MapSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_build_world_uses_selected_map_resources(self) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        self.assertEqual(world.map_id, "dual_system_crossroads")
        self.assertIn("alpha_gate_to_beta", world.structures)
        self.assertIn("beta_gate_to_alpha", world.structures)
        self.assertNotIn("gate-1", world.structures)

    def test_warp_to_stargate_does_not_auto_jump_ship(self) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        source_gate = world.structures["alpha_gate_to_beta"]
        ship = _make_ship("BLUE-TEST-001", Team.BLUE, "BLUE-ALPHA", Vector2(source_gate.position.x - 1_000.0, source_gate.position.y), system_id="alpha")
        ship.nav.warp.phase = "warp"
        ship.nav.warp.origin = Vector2(ship.nav.position.x, ship.nav.position.y)
        ship.nav.warp.destination = Vector2(source_gate.position.x, source_gate.position.y)
        ship.nav.warp.warp_distance_m = ship.nav.position.distance_to(source_gate.position)
        ship.nav.warp.warp_duration = 0.2
        ship.nav.warp.warp_elapsed = 0.19
        ship.nav.warp.target_beacon_id = "alpha_gate_to_beta"
        world.ships[ship.ship_id] = ship

        movement = MovementSystem()
        movement.run(world, 0.05)

        self.assertEqual(ship.nav.system_id, "alpha")
        self.assertAlmostEqual(ship.nav.position.distance_to(source_gate.position), 0.0, delta=0.01)

    def test_gate_use_applies_gate_cloak_and_follow_hold(self) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        source_gate = world.structures["alpha_gate_to_beta"]
        dest_gate = world.structures["beta_gate_to_alpha"]
        leader = _make_ship(
            "BLUE-ALPHA-001",
            Team.BLUE,
            "BLUE-ALPHA",
            Vector2(source_gate.position.x + float(source_gate.radius) + 2_000.0, source_gate.position.y),
            system_id="alpha",
        )
        follower = _make_ship(
            "BLUE-ALPHA-002",
            Team.BLUE,
            "BLUE-ALPHA",
            Vector2(source_gate.position.x + float(source_gate.radius) + 2_200.0, source_gate.position.y + 300.0),
            system_id="alpha",
        )
        leader.nav.gate.target_structure_id = source_gate.structure_id
        follower.nav.gate.target_structure_id = source_gate.structure_id
        world.ships = {leader.ship_id: leader, follower.ship_id: follower}
        world.squad_leaders["BLUE:BLUE-ALPHA"] = leader.ship_id

        movement = MovementSystem()
        movement.run(world, 0.05)

        for ship in (leader, follower):
            self.assertEqual(ship.nav.system_id, "beta")
            distance = ship.nav.position.distance_to(dest_gate.position)
            self.assertGreaterEqual(distance, 10_000.0 - 1.0)
            self.assertLessEqual(distance, 15_000.0 + 1.0)
            self.assertTrue(ship.nav.cloak.active)
            self.assertAlmostEqual(ship.nav.cloak.expires_at, world.now + 60.0, delta=0.01)
            self.assertIsNone(ship.nav.gate.target_structure_id)
        self.assertFalse(leader.nav.follow_hold_active)
        self.assertTrue(follower.nav.follow_hold_active)
        self.assertEqual(follower.nav.follow_hold_leader_id, leader.ship_id)

    def test_perception_ignores_gate_cloaked_ships(self) -> None:
        world = WorldState()
        visible = _make_ship("BLUE-001", Team.BLUE, "BLUE", Vector2(0.0, 0.0), system_id="alpha")
        cloaked = _make_ship("RED-001", Team.RED, "RED", Vector2(10_000.0, 0.0), system_id="alpha")
        cloaked.nav.cloak.active = True
        cloaked.nav.cloak.expires_at = 60.0
        cloaked.nav.cloak.source = "stargate"
        world.ships = {visible.ship_id: visible, cloaked.ship_id: cloaked}
        world.now = 1.0

        PerceptionSystem(sensor_range=50_000.0).run(world)

        self.assertNotIn(cloaked.ship_id, visible.perception)
        self.assertEqual(cloaked.perception, [])

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_overview_double_click_centers_camera_on_ship(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures
        ship = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(123_456.0, -78_900.0), system_id="alpha")
        world.ships[ship.ship_id] = ship
        world.squad_leaders["BLUE:BLUE-ALPHA"] = ship.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window._undeployed_ship_ids.clear()
            ship.vital.alive = True
            window.request_overview_refresh(force=True)
            index = window.overview_proxy.index(0, 0)
            self.assertTrue(index.isValid())
            window._on_overview_double_clicked(index)
            self.assertAlmostEqual(window.canvas.pan_world.x, ship.nav.position.x)
            self.assertAlmostEqual(window.canvas.pan_world.y, ship.nav.position.y)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_gate_cloaked_followers_are_hidden_but_leader_remains_visible(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures
        leader = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        follower = _make_ship("BLUE-ALPHA-002", Team.BLUE, "BLUE-ALPHA", Vector2(5_000.0, 0.0), system_id="alpha")
        leader.nav.cloak.active = True
        leader.nav.cloak.expires_at = 60.0
        leader.nav.cloak.source = "stargate"
        follower.nav.cloak.active = True
        follower.nav.cloak.expires_at = 60.0
        follower.nav.cloak.source = "stargate"
        world.ships = {leader.ship_id: leader, follower.ship_id: follower}
        world.squad_leaders["BLUE:BLUE-ALPHA"] = leader.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(leader.ship_id)
        engine.register_ship(follower.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            self.assertTrue(window._is_ship_visible(leader.ship_id))
            self.assertFalse(window._is_ship_visible(follower.ship_id))
            window.request_overview_refresh(force=True)
            row_ids = {
                str(window.overview_proxy.get_row(row).get("id", ""))
                for row in range(window.overview_proxy.rowCount())
            }
            self.assertIn(leader.ship_id, row_ids)
            self.assertNotIn(follower.ship_id, row_ids)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_overview_includes_structures_and_double_click_centers_camera_on_structure(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures
        ship = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        world.ships[ship.ship_id] = ship
        world.squad_leaders["BLUE:BLUE-ALPHA"] = ship.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            ship.vital.alive = True
            window.request_overview_refresh(force=True)
            structure_id = "alpha_gate_to_beta"
            target_row = next(
                row for row in range(window.overview_proxy.rowCount())
                if window.overview_proxy.get_row(row)["id"] == structure_id
            )
            index = window.overview_proxy.index(target_row, 0)
            self.assertTrue(index.isValid())
            window._on_overview_double_clicked(index)
            structure = world.structures[structure_id]
            self.assertAlmostEqual(window.canvas.pan_world.x, structure.position.x)
            self.assertAlmostEqual(window.canvas.pan_world.y, structure.position.y)
            self.assertEqual(window.canvas.selected_structure_id, structure_id)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_structure_context_menu_warp_action_targets_beacon(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures
        ship = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        world.ships[ship.ship_id] = ship
        world.squad_leaders["BLUE:BLUE-ALPHA"] = ship.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            with patch.object(window, "issue_warp_to_beacon") as warp_mock:
                menu = window._build_structure_context_menu("alpha_gate_to_beta")
                self.assertIsNotNone(menu)
                warp_action = next(
                    action
                    for action in menu.actions()
                    if ("Warp To" in action.text()) or ("跃迁至" in action.text())
                )
                warp_action.trigger()
                warp_mock.assert_called_once_with("BLUE-ALPHA", "alpha_gate_to_beta")
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_structure_context_menu_localizes_stargate_actions(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures
        ship = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        world.ships[ship.ship_id] = ship
        world.squad_leaders["BLUE:BLUE-ALPHA"] = ship.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window.lang_combo.setCurrentIndex(window.lang_combo.findData("zh_CN"))
            menu = window._build_structure_context_menu("alpha_gate_to_beta")
            self.assertIsNotNone(menu)
            texts = [action.text() for action in menu.actions()]
            self.assertTrue(any("跃迁至" in text for text in texts))
            self.assertTrue(any("使用星门" in text for text in texts))
            self.assertFalse(any("Warp To" in text for text in texts))
            self.assertFalse(any("Take Gate" in text for text in texts))
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_canvas_left_click_can_select_structure(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = WorldState(map_id=map_definition.map_id, map_name=map_definition.name, map_definition=map_definition)
        world.structures = build_world_from_manual_setup([], map_definition=map_definition).structures

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[])

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=blue_commander,
            red_commander=red_commander,
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            structure = world.structures["alpha_gate_to_beta"]
            window.canvas.focus_structure(structure.structure_id)
            screen_x, screen_y = window.canvas._to_screen(structure.position)
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(float(screen_x), float(screen_y)),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            window.canvas.mousePressEvent(event)
            self.assertEqual(window.canvas.selected_structure_id, structure.structure_id)
            self.assertIsNone(window.canvas.selected_ship_id)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_system_graph_derives_stargate_connections(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            self.assertEqual(window.system_graph_window.canvas.system_edges(), (("alpha", "beta"),))
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_system_graph_click_switches_main_view_system(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._set_view_system("alpha", center=False)
            canvas = window.system_graph_window.canvas
            canvas.resize(360, 320)
            positions = canvas._screen_positions()
            beta_pos = positions["beta"]
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                beta_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            canvas.mousePressEvent(event)
            self.assertEqual(window.canvas.current_view_system_id, "beta")
            self.assertAlmostEqual(window.canvas.pan_world.x, 0.0)
            self.assertAlmostEqual(window.canvas.pan_world.y, 0.0)
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_warp_and_gate_commands_clear_selected_squad_propulsion_state_immediately(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        ship = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        world.ships = {ship.ship_id: ship}
        world.squad_leaders["BLUE:BLUE-ALPHA"] = ship.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)
        engine.register_ship(ship.ship_id)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            window._set_team_propulsion_state(Team.BLUE, "BLUE-ALPHA", True)
            window.issue_warp_to_beacon("BLUE-ALPHA", "alpha_gate_to_beta")
            self.assertFalse(window._get_squad_propulsion_state("BLUE-ALPHA"))

            window._set_team_propulsion_state(Team.BLUE, "BLUE-ALPHA", True)
            window.issue_use_gate("BLUE-ALPHA", "alpha_gate_to_beta")
            self.assertFalse(window._get_squad_propulsion_state("BLUE-ALPHA"))
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="FRIENDLY", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_system_view_controls_center_camera_on_selected_system(self, _load_mock, _save_mock) -> None:
        map_definition = load_map_definition("dual_system_crossroads")
        world = build_world_from_manual_setup([], map_definition=map_definition)
        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        engine.register_commander(commander)

        window = MainWindow(
            engine=engine,
            ui_cfg=UiConfig(),
            blue_commander=commander,
            red_commander=CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=[]),
            manual_setup=[],
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            self.assertGreaterEqual(window.system_view_combo.count(), 2)
            idx = window.system_view_combo.findData("beta")
            self.assertGreaterEqual(idx, 0)
            window.system_view_combo.setCurrentIndex(idx)
            self.assertAlmostEqual(window.canvas.pan_world.x, 0.0)
            self.assertAlmostEqual(window.canvas.pan_world.y, 0.0)
            self.assertEqual(window.canvas.current_view_system_id, "beta")
        finally:
            window.close()

    @patch("eve_sim.gui.main_window.PreferencesStore.save", return_value=None)
    @patch(
        "eve_sim.gui.main_window.PreferencesStore.load",
        return_value=UiPreferences(filter_team="ALL", selected_squad="BLUE-ALPHA", selected_map_id="dual_system_crossroads"),
    )
    def test_overview_only_shows_current_system_and_formats_long_distance_in_au(self, _load_mock, _save_mock) -> None:
        large_map = MapDefinition(
            map_id="large-map",
            name="Large Map",
            systems=[
                MapSystemDefinition(system_id="alpha", name="Alpha", origin=Vector2(0.0, 0.0), radius_m=30.0 * 149_597_870_700.0),
                MapSystemDefinition(system_id="beta", name="Beta", origin=Vector2(80.0 * 149_597_870_700.0, 0.0), radius_m=30.0 * 149_597_870_700.0),
            ],
        )
        world = WorldState(map_id=large_map.map_id, map_name=large_map.name, map_definition=large_map)
        alpha_leader = _make_ship("BLUE-ALPHA-001", Team.BLUE, "BLUE-ALPHA", Vector2(0.0, 0.0), system_id="alpha")
        alpha_far = _make_ship("BLUE-ALPHA-002", Team.BLUE, "BLUE-ALPHA", Vector2(0.2 * 149_597_870_700.0, 0.0), system_id="alpha")
        beta_ship = _make_ship("RED-BETA-001", Team.RED, "RED-BETA", Vector2(80.0 * 149_597_870_700.0, 0.0), system_id="beta")
        world.ships = {
            alpha_leader.ship_id: alpha_leader,
            alpha_far.ship_id: alpha_far,
            beta_ship.ship_id: beta_ship,
        }
        world.squad_leaders["BLUE:BLUE-ALPHA"] = alpha_leader.ship_id

        engine = SimulationEngine(world=world, config=EngineConfig(), combat_system=CombatSystem(PyfaBridge()))
        blue_commander = CommanderAgent(agent_id="cmd-blue", team=Team.BLUE, squad_ids=["BLUE-ALPHA"])
        red_commander = CommanderAgent(agent_id="cmd-red", team=Team.RED, squad_ids=["RED-BETA"])
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
        )
        try:
            window.tick_timer.stop()
            window.render_timer.stop()
            window._undeployed_ship_ids.clear()
            window._set_view_system("alpha", center=False)
            window.request_overview_refresh(force=True)
            rows = [window.overview_proxy.get_row(row) for row in range(window.overview_proxy.rowCount())]
            self.assertTrue(all(str(row.get("system_id", "")) == "alpha" for row in rows if row is not None))
            far_row = next(row for row in rows if row and row["id"] == "BLUE-ALPHA-002")
            self.assertIn("AU", str(far_row.get("dist_display", "")))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
