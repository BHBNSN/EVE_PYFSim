from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from eve_sim.config import EngineConfig, UiConfig
from eve_sim.fit_runtime import EffectClass, FitRuntime, HullProfile, ModuleEffect, ModuleRuntime, ModuleState, RuntimeStatEngine, SkillProfile
from eve_sim.gui.battle_canvas import BattleCanvas
from eve_sim.gui.dialogs import ShipStatusDialog
from eve_sim.math2d import Vector2
from eve_sim.models import CombatState, FitDescriptor, NavigationState, QualityLevel, QualityState, ShipEntity, StructureEntity, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem
from eve_sim.world import WorldState


def _make_ship(ship_id: str) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name="Test Hull",
        role="skirmish",
        base_dps=120.0,
        volley=420.0,
        optimal_range=30_000.0,
        falloff=12_000.0,
        tracking=0.045,
        signature_radius=135.0,
        scan_resolution=240.0,
        max_target_range=95_000.0,
        max_speed=640.0,
        max_cap=2_400.0,
        cap_recharge_time=260.0,
        shield_hp=2_200.0,
        armor_hp=1_700.0,
        structure_hp=1_500.0,
        mass=12_500_000.0,
        agility=0.56,
    )
    hull = HullProfile(
        ship_name=fit.ship_name,
        role=fit.role,
        base_dps=fit.base_dps,
        volley=fit.volley,
        optimal=fit.optimal_range,
        falloff=fit.falloff,
        tracking=fit.tracking,
        sig_radius=fit.signature_radius,
        scan_resolution=fit.scan_resolution,
        max_target_range=fit.max_target_range,
        max_speed=fit.max_speed,
        cap_max=fit.max_cap,
        cap_recharge_time=fit.cap_recharge_time,
        shield_hp=fit.shield_hp,
        armor_hp=fit.armor_hp,
        structure_hp=fit.structure_hp,
        rep_amount=0.0,
        rep_cycle=5.0,
        mass=fit.mass,
        agility=fit.agility,
    )
    modules = [
        ModuleRuntime(
            module_id="mod-1",
            group="10mn afterburner",
            state=ModuleState.ACTIVE,
            effects=[
                ModuleEffect(
                    name="ab-local",
                    effect_class=EffectClass.LOCAL,
                    state_required=ModuleState.ACTIVE,
                    cycle_time=10.0,
                    local_mult={"speed": 1.8},
                )
            ],
        ),
        ModuleRuntime(
            module_id="mod-2",
            group="hybrid turret",
            state=ModuleState.ACTIVE,
            charge_capacity=40,
            charge_remaining=32,
            charge_reload_time=5.0,
            effects=[
                ModuleEffect(
                    name="gun-projected",
                    effect_class=EffectClass.PROJECTED,
                    state_required=ModuleState.ACTIVE,
                    cycle_time=4.0,
                    projected_add={"damage_em": 40.0},
                )
            ],
        ),
    ]
    runtime = FitRuntime(fit_key=fit.fit_key, hull=hull, skills=SkillProfile(), modules=modules, diagnostics={})
    profile = RuntimeStatEngine().compute_base_profile(runtime)
    profile = replace(
        profile,
        dps=145.0,
        volley=510.0,
        turret_dps=145.0,
        max_locked_targets=7,
        scan_strength=21.0,
        sensor_strength_gravimetric=21.0,
        max_speed=920.0,
        warp_stability=1.0,
        damage_em=30.0,
        damage_thermal=90.0,
        damage_kinetic=25.0,
        damage_explosive=0.0,
        optimal_sig=125.0,
        mass=fit.mass,
        agility=fit.agility,
    )
    runtime.diagnostics["pyfa_base_profile"] = replace(profile)
    runtime.diagnostics["pyfa_runtime_resolve_cache"] = "hit"
    runtime.diagnostics["pyfa_projected_target_fit_cache"] = "resolved_hit"

    ship = ShipEntity(
        ship_id=ship_id,
        team=Team.BLUE,
        squad_id="BLUE-ALPHA",
        fit=fit,
        profile=replace(profile),
        nav=NavigationState(
            position=Vector2(0.0, 0.0),
            velocity=Vector2(420.0, 0.0),
            facing_deg=35.0,
            max_speed=profile.max_speed,
        ),
        combat=CombatState(
            lock_targets={"red-1"},
            current_target="red-1",
            module_cycle_timers={"mod-2": 1.5},
        ),
        vital=VitalState(
            shield=1_800.0,
            armor=1_550.0,
            structure=1_500.0,
            shield_max=profile.shield_hp,
            armor_max=profile.armor_hp,
            structure_max=profile.structure_hp,
            cap=1_700.0,
            cap_max=profile.max_cap,
            alive=True,
        ),
        quality=QualityState(
            level=QualityLevel.REGULAR,
            reaction_delay=0.0,
            ignore_order_probability=0.0,
            formation_jitter=0.0,
        ),
        runtime=runtime,
    )
    return ship


class ShipStatusDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _make_dialog(
        self,
        *,
        module_target_modes: dict[str, str] | None = None,
        module_modes: dict[str, str] | None = None,
        fit_text: str = "",
        lock_charge_getter=None,
        module_target_mode_setter=None,
        module_mode_setter=None,
        module_control_sync_setter=None,
        other_ships: list[ShipEntity] | None = None,
    ) -> ShipStatusDialog:
        ship = _make_ship("blue-1")
        world_ships = {ship.ship_id: ship}
        for other_ship in other_ships or []:
            world_ships[other_ship.ship_id] = other_ship
        world = WorldState(ships=world_ships)
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        local_target_modes = module_target_modes if module_target_modes is not None else {}
        local_module_modes = module_modes if module_modes is not None else {}

        def default_module_target_mode_setter(ship_id: str, module_id: str, mode: str) -> tuple[bool, str]:
            del ship_id
            local_target_modes[module_id] = mode
            return True, mode

        def default_module_mode_setter(ship_id: str, module_id: str, mode: str) -> tuple[bool, str]:
            del ship_id
            local_module_modes[module_id] = mode
            return True, mode

        def default_module_control_sync_setter(
            ship_id: str,
            module_id: str,
            mode: str,
            target_mode: str,
        ) -> tuple[bool, str]:
            del ship_id
            local_module_modes[module_id] = mode
            local_target_modes[module_id] = target_mode
            return True, f"sync:{module_id}:{target_mode}:{mode}"

        dialog = ShipStatusDialog(
            engine=engine,
            ship_id=ship.ship_id,
            language_getter=lambda: "zh_CN",
            fit_text_getter=lambda _ship_id: fit_text,
            lock_charge_getter=lock_charge_getter or (lambda _ship_id, _module_id: None),
            lock_charge_setter=lambda _ship_id, _module_id, _ammo: (True, "ok"),
            lock_charge_clearer=lambda _ship_id, _module_id: (True, "ok"),
            module_target_mode_getter=lambda _ship_id, module_id: local_target_modes.get(module_id, "auto"),
            module_target_mode_setter=module_target_mode_setter or default_module_target_mode_setter,
            module_mode_getter=lambda _ship_id, module_id: local_module_modes.get(module_id, "auto"),
            module_mode_setter=module_mode_setter or default_module_mode_setter,
            module_control_sync_setter=module_control_sync_setter or default_module_control_sync_setter,
        )
        dialog.timer.stop()
        dialog._test_module_target_modes = local_target_modes
        dialog._test_module_modes = local_module_modes
        return dialog

    @staticmethod
    def _table_rows(dialog_table) -> dict[str, str]:
        rows: dict[str, str] = {}
        for row in range(dialog_table.rowCount()):
            key_item = dialog_table.item(row, 0)
            value_item = dialog_table.item(row, 1)
            if key_item is None or value_item is None:
                continue
            rows[key_item.text()] = value_item.text()
        return rows

    def test_status_dialog_builds_structured_tabs_and_overview_metrics(self) -> None:
        dialog = self._make_dialog()
        try:
            self.assertEqual(dialog.tabs.count(), 6)
            self.assertEqual(dialog.tabs.tabText(0), QCoreApplication.translate("eve_sim", 'Overview'))
            self.assertEqual(dialog.tabs.tabText(1), QCoreApplication.translate("eve_sim", 'Combat'))
            self.assertEqual(dialog.tabs.tabText(2), QCoreApplication.translate("eve_sim", 'Defense'))

            rows = self._table_rows(dialog.overview_table)
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Ship')], "blue-1")
            self.assertIn(QCoreApplication.translate("eve_sim", 'Backend'), rows)
            self.assertIn(QCoreApplication.translate("eve_sim", 'Locked Targets'), rows)
            self.assertIn("red-1", rows[QCoreApplication.translate("eve_sim", 'Target')])
        finally:
            dialog.close()

    def test_status_dialog_cap_target_tab_shows_lock_state_details(self) -> None:
        outgoing_locked = _make_ship("red-2")
        outgoing_locked.team = Team.RED
        outgoing_locking = _make_ship("red-3")
        outgoing_locking.team = Team.RED
        incoming_locked = _make_ship("enemy-locker")
        incoming_locked.team = Team.RED
        incoming_locking = _make_ship("enemy-locking")
        incoming_locking.team = Team.RED
        incoming_locked.combat.lock_targets.add("blue-1")
        incoming_locking.combat.lock_timers["blue-1"] = 1.5

        dialog = self._make_dialog(other_ships=[outgoing_locked, outgoing_locking, incoming_locked, incoming_locking])
        try:
            ship = dialog.engine.world.ships["blue-1"]
            ship.combat.lock_targets = {"red-1", "red-2"}
            ship.combat.lock_timers["red-3"] = 2.5
            dialog.tabs.setCurrentIndex(3)

            rows = self._table_rows(dialog.targeting_table)
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Current Target')], "red-1")
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Locked Target Details')], "red-1, red-2")
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Locking Targets')], "red-3 (2.5s)")
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Locked By')], "enemy-locker")
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Locking By')], "enemy-locking (1.5s)")
        finally:
            dialog.close()

    def test_status_dialog_uses_live_effective_profile_instead_of_cached_pyfa_base_profile(self) -> None:
        dialog = self._make_dialog()
        try:
            ship = dialog.engine.world.ships["blue-1"]
            assert ship.runtime is not None
            ship.runtime.diagnostics["pyfa_base_profile"] = replace(
                ship.runtime.diagnostics["pyfa_base_profile"],
                scan_resolution=240.0,
                max_target_range=95_000.0,
            )
            ship.profile = replace(
                ship.profile,
                scan_resolution=123.4,
                max_target_range=54_321.0,
            )

            dialog.tabs.setCurrentIndex(3)
            rows = self._table_rows(dialog.targeting_table)

            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Scan Resolution')], "123.4 mm")
            self.assertEqual(rows[QCoreApplication.translate("eve_sim", 'Target Range')], "54.3 km")
        finally:
            dialog.close()

    def test_status_dialog_lazy_loads_modules_and_debug_tabs(self) -> None:
        dialog = self._make_dialog()
        try:
            self.assertEqual(dialog.tabs.currentIndex(), 0)
            self.assertEqual(dialog.modules_table.rowCount(), 0)
            self.assertEqual(dialog.info.toPlainText(), "")

            dialog.tabs.setCurrentIndex(4)
            self.assertGreaterEqual(dialog.modules_table.rowCount(), 2)
            module_name = dialog.modules_table.item(0, 1).text()
            self.assertTrue(module_name)

            dialog.tabs.setCurrentIndex(5)
            self.assertIn("runtime: yes", dialog.info.toPlainText())
            self.assertIn("diagnostics:", dialog.info.toPlainText())
        finally:
            dialog.close()

    @patch("eve_sim.gui.dialogs.get_charge_option_values_for_module", return_value=["Focused Warp Disruption Script", "Focused Warp Scrambling Script"])
    @patch("eve_sim.gui.dialogs.module_supports_unloaded_charge", return_value=True)
    def test_status_dialog_lock_charge_combo_supports_unloaded_selection(
        self,
        _supports_unloaded,
        _charge_values,
    ) -> None:
        dialog = self._make_dialog(
            fit_text="[Onyx, Test]\nWarp Disruption Field Generator I\n",
            lock_charge_getter=lambda _ship_id, module_id: "" if module_id == "mod-1" else None,
        )
        try:
            dialog._refresh_lock_controls()
            self.assertEqual(dialog.lock_module_combo.count(), 1)
            self.assertEqual(dialog.lock_ammo_combo.itemData(0), "")
            self.assertEqual(dialog.lock_ammo_combo.itemText(0), QCoreApplication.translate("eve_sim", "None"))
            self.assertEqual(dialog.lock_ammo_combo.currentData(), "")
            self.assertTrue(dialog.btn_lock_clear.isEnabled())
        finally:
            dialog.close()

    def test_status_dialog_modules_tab_exposes_target_rule_mode_and_sync_controls(self) -> None:
        active_target_modes = {"mod-2": "enemy_nearest"}
        active_module_modes = {"mod-1": "online"}
        target_set_calls: list[tuple[str, str, str]] = []
        set_calls: list[tuple[str, str, str]] = []
        sync_calls: list[tuple[str, str, str, str]] = []

        def module_target_mode_setter(ship_id: str, module_id: str, mode: str) -> tuple[bool, str]:
            active_target_modes[module_id] = mode
            target_set_calls.append((ship_id, module_id, mode))
            return True, mode

        def module_mode_setter(ship_id: str, module_id: str, mode: str) -> tuple[bool, str]:
            active_module_modes[module_id] = mode
            set_calls.append((ship_id, module_id, mode))
            return True, mode

        def module_control_sync_setter(ship_id: str, module_id: str, mode: str, target_mode: str) -> tuple[bool, str]:
            sync_calls.append((ship_id, module_id, mode, target_mode))
            return True, "synced"

        dialog = self._make_dialog(
            module_target_modes=active_target_modes,
            module_modes=active_module_modes,
            module_target_mode_setter=module_target_mode_setter,
            module_mode_setter=module_mode_setter,
            module_control_sync_setter=module_control_sync_setter,
        )
        try:
            dialog.tabs.setCurrentIndex(4)
            target_combo = dialog.modules_table.cellWidget(1, 7)
            self.assertIsNotNone(target_combo)
            assert target_combo is not None
            self.assertEqual(target_combo.currentData(), "enemy_nearest")
            self.assertEqual(dialog.modules_table.item(1, 7).text(), "")
            self.assertEqual(target_combo.count(), 3)
            self.assertEqual(target_combo.findData("auto"), -1)

            mode_combo = dialog.modules_table.cellWidget(0, 8)
            self.assertIsNotNone(mode_combo)
            assert mode_combo is not None
            self.assertEqual(mode_combo.currentData(), "online")
            self.assertEqual(dialog.modules_table.item(0, 8).text(), "")

            sync_button = dialog.modules_table.cellWidget(1, 9)
            self.assertIsNotNone(sync_button)
            assert sync_button is not None

            random_index = target_combo.findData("enemy_random")
            self.assertGreaterEqual(random_index, 0)
            target_combo.setCurrentIndex(random_index)
            self.assertIn(("blue-1", "mod-2", "enemy_random"), target_set_calls)

            mode_combo = dialog.modules_table.cellWidget(0, 8)
            self.assertIsNotNone(mode_combo)
            assert mode_combo is not None
            active_index = mode_combo.findData("active")
            self.assertGreaterEqual(active_index, 0)
            mode_combo.setCurrentIndex(active_index)

            self.assertIn(("blue-1", "mod-1", "active"), set_calls)
            sync_button = dialog.modules_table.cellWidget(1, 9)
            self.assertIsNotNone(sync_button)
            assert sync_button is not None
            with patch("eve_sim.gui.dialogs.QMessageBox.information", return_value=None):
                sync_button.click()
            self.assertIn(("blue-1", "mod-2", "auto", "enemy_random"), sync_calls)
        finally:
            dialog.close()

    def test_status_dialog_skips_module_refresh_while_mode_popup_is_open(self) -> None:
        dialog = self._make_dialog(module_target_modes={"mod-2": "auto"}, module_modes={"mod-1": "auto"})
        try:
            dialog.tabs.setCurrentIndex(4)
            combo = dialog.modules_table.cellWidget(1, 7)
            self.assertIsNotNone(combo)
            assert combo is not None

            dialog._on_module_mode_popup_visibility_changed(True)
            original_combo = dialog.modules_table.cellWidget(1, 7)
            dialog._tab_signatures.pop("modules", None)
            dialog.refresh_status(force=False)

            self.assertIs(dialog.modules_table.cellWidget(1, 7), original_combo)
        finally:
            dialog.close()

    def test_battle_canvas_hides_burst_jammer_area_overlay_style(self) -> None:
        burst_jammer = ModuleRuntime(module_id="jam-1", group="Burst Jammer", state=ModuleState.ACTIVE, tags=("area_effect", "burst_jammer"))
        smart_bomb = ModuleRuntime(module_id="smart-1", group="Smart Bomb", state=ModuleState.ACTIVE, tags=("area_effect", "smart_bomb"))

        self.assertIsNone(BattleCanvas._module_area_style(burst_jammer))
        self.assertIsNotNone(BattleCanvas._module_area_style(smart_bomb))

    @patch("eve_sim.gui.battle_canvas.get_ship_icon_key", return_value="frigate")
    def test_battle_canvas_ship_icon_pixmap_is_cached_in_memory(self, _icon_key) -> None:
        BattleCanvas._SHIP_ICON_SOURCE_CACHE = None
        BattleCanvas._SHIP_ICON_PIXMAP_CACHE.clear()

        color = QColor(80, 180, 255)
        first = BattleCanvas._ship_icon_for_name("Rifter", color)
        second = BattleCanvas._ship_icon_for_name("Rifter", color)

        self.assertIsNotNone(first)
        assert first is not None
        self.assertFalse(first.isNull())
        self.assertIs(first, second)
        self.assertEqual(len(BattleCanvas._SHIP_ICON_PIXMAP_CACHE), 1)
        self.assertEqual(first.width(), BattleCanvas._SHIP_ICON_SIZE_PX)

    def test_battle_canvas_selected_squad_uses_purple_ship_color(self) -> None:
        ship = _make_ship("purple-ship")
        color = BattleCanvas._ship_draw_color(ship, Team.BLUE, "BLUE-ALPHA")
        self.assertEqual(color, BattleCanvas._SELECTED_SQUAD_COLOR)

        other_blue = _make_ship("blue-ship")
        other_blue.squad_id = "BLUE-BETA"
        self.assertEqual(
            BattleCanvas._ship_draw_color(other_blue, Team.BLUE, "BLUE-ALPHA"),
            BattleCanvas._TEAM_BLUE_COLOR,
        )

        red_ship = _make_ship("red-ship")
        red_ship.team = Team.RED
        red_ship.squad_id = "RED-ALPHA"
        self.assertEqual(
            BattleCanvas._ship_draw_color(red_ship, Team.BLUE, "BLUE-ALPHA"),
            BattleCanvas._TEAM_RED_COLOR,
        )

    def test_battle_canvas_selection_highlight_uses_yellow_outer_size(self) -> None:
        ship = _make_ship("highlight-ship")
        world = WorldState(ships={ship.ship_id: ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        canvas = BattleCanvas(
            engine=engine,
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )
        try:
            self.assertEqual(canvas._ship_selection_highlight_level(ship), 0)
            self.assertIsNone(canvas._ship_selection_highlight_size_px(ship, BattleCanvas._SHIP_ICON_SIZE_PX))
            canvas.highlighted_roster_ship_ids = {ship.ship_id}
            self.assertEqual(canvas._ship_selection_highlight_level(ship), 1)
            self.assertEqual(
                canvas._ship_selection_highlight_size_px(ship, BattleCanvas._SHIP_ICON_SIZE_PX),
                BattleCanvas._SHIP_ICON_SIZE_PX + 4,
            )
            canvas.selected_ship_id = ship.ship_id
            self.assertEqual(canvas._ship_selection_highlight_level(ship), 2)
            self.assertEqual(
                canvas._ship_selection_highlight_size_px(ship, BattleCanvas._SHIP_ICON_SIZE_PX),
                BattleCanvas._SHIP_ICON_SIZE_PX + 6,
            )
            canvas.selected_enemy_target = ship.ship_id
            self.assertEqual(canvas._ship_selection_highlight_level(ship), 4)
            self.assertEqual(
                canvas._ship_selection_highlight_size_px(ship, BattleCanvas._SHIP_ICON_SIZE_PX),
                BattleCanvas._SHIP_ICON_SIZE_PX + 10,
            )
        finally:
            canvas.close()

    def test_battle_canvas_selection_priority_draws_highlighted_ships_last(self) -> None:
        normal_ship = _make_ship("normal-ship")
        highlighted_ship = _make_ship("highlighted-ship")
        target_ship = _make_ship("target-ship")
        world = WorldState(
            ships={
                normal_ship.ship_id: normal_ship,
                highlighted_ship.ship_id: highlighted_ship,
                target_ship.ship_id: target_ship,
            }
        )
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        canvas = BattleCanvas(
            engine=engine,
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )
        try:
            canvas.highlighted_roster_ship_ids = {highlighted_ship.ship_id}
            canvas.selected_enemy_target = target_ship.ship_id
            ordered_ids = [ship.ship_id for ship in sorted(world.ships.values(), key=canvas._ship_draw_priority)]
            self.assertEqual(ordered_ids, [normal_ship.ship_id, highlighted_ship.ship_id, target_ship.ship_id])
        finally:
            canvas.close()

    def test_battle_canvas_zoom_uses_normalized_minimum(self) -> None:
        canvas = BattleCanvas(
            engine=SimulationEngine(WorldState(), EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge())),
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )
        try:
            canvas.resize(800, 600)
            self.assertEqual(canvas.zoom, 0.3)
            canvas._set_zoom_anchored(0.05, QPoint(400, 300))
            self.assertEqual(canvas.zoom, 0.3)
            canvas._set_zoom_anchored(0.5, QPoint(400, 300))
            self.assertEqual(canvas.zoom, 0.5)
        finally:
            canvas.close()

    def test_battle_canvas_zoom_keeps_mouse_anchor_world_position(self) -> None:
        ship = _make_ship("anchor-ship")
        ship.nav.position = Vector2(100_000.0, -50_000.0)
        world = WorldState(ships={ship.ship_id: ship})
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        canvas = BattleCanvas(
            engine=engine,
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )
        try:
            canvas.resize(1000, 800)
            anchor = QPoint(180, 220)
            before = canvas._to_world(anchor)
            canvas._set_zoom_anchored(canvas.zoom * 1.5, anchor)
            after = canvas._to_world(anchor)
            self.assertAlmostEqual(before.x, after.x, places=6)
            self.assertAlmostEqual(before.y, after.y, places=6)
        finally:
            canvas.close()

    def test_battle_canvas_command_burst_overlay_tracks_ship_position(self) -> None:
        ship = _make_ship("burst-ship")
        burst_module = ModuleRuntime(
            module_id="burst-1",
            group="Command Burst",
            state=ModuleState.ACTIVE,
            tags=("area_effect", "command_burst", "support"),
            effects=[
                ModuleEffect(
                    name="burst-1-effect",
                    effect_class=EffectClass.PROJECTED,
                    state_required=ModuleState.ACTIVE,
                    range_m=60_000.0,
                    cycle_time=6.0,
                )
            ],
        )
        runtime = ship.runtime
        assert runtime is not None
        runtime.modules = [burst_module]
        ship.profile = RuntimeStatEngine().compute_base_profile(runtime)
        ship.nav.position = Vector2(1_000.0, 2_000.0)
        ship.combat.module_cycle_timers[burst_module.module_id] = 4.0

        world = WorldState(ships={ship.ship_id: ship})
        world.now = 10.0
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        canvas = BattleCanvas(
            engine=engine,
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )

        try:
            overlays = canvas._iter_active_area_overlays()
            self.assertEqual(len(overlays), 1)
            self.assertEqual(overlays[0].center.x, 1_000.0)
            self.assertEqual(overlays[0].center.y, 2_000.0)
            self.assertGreater(overlays[0].expand_duration_sec, 0.0)
            started_at = overlays[0].started_at

            ship.nav.position = Vector2(3_000.0, 4_000.0)
            moved_overlays = canvas._iter_active_area_overlays()
            self.assertEqual(len(moved_overlays), 1)
            self.assertEqual(moved_overlays[0].center.x, 3_000.0)
            self.assertEqual(moved_overlays[0].center.y, 4_000.0)
            self.assertEqual(moved_overlays[0].started_at, started_at)
        finally:
            canvas.close()

    def test_battle_canvas_skips_far_au_scale_structure_without_overflow(self) -> None:
        ship = _make_ship("near-ship")
        world = WorldState(
            ships={ship.ship_id: ship},
            structures={
                "far-structure": StructureEntity(
                    structure_id="far-structure",
                    position=Vector2(598_391_482_800.0, 0.0),
                    radius=3_500.0,
                    interaction_range=0.0,
                    kind="STARGATE",
                    system_id="beta",
                    display_name="Far Stargate",
                )
            },
        )
        engine = SimulationEngine(world, EngineConfig(tick_rate=1, physics_substeps=1), CombatSystem(PyfaBridge()))
        canvas = BattleCanvas(
            engine=engine,
            ui_cfg=UiConfig(),
            on_issue_move=lambda *_args: None,
            on_issue_approach=lambda *_args: None,
            on_issue_warp_ship=lambda *_args: None,
            on_issue_warp_beacon=lambda *_args: None,
            on_issue_focus=lambda *_args: None,
            on_issue_prefocus=lambda *_args: None,
            on_cancel_prefocus=lambda *_args: None,
            on_show_ship_context_menu=lambda *_args: None,
            on_induce_squad_spawn=lambda *_args: None,
            on_induce_fleet_spawn=lambda *_args: None,
            controlled_squads_getter=lambda: ["SQ1"],
            ship_visible_getter=lambda _ship_id: True,
            squad_guidance_target_getter=lambda _squad_id: None,
            on_show_status=lambda _ship_id: None,
            language_getter=lambda: "en_US",
            controlled_team_getter=lambda: Team.BLUE,
            on_select_squad=lambda _squad_id: None,
            on_select_enemy=lambda _ship_id: None,
        )
        try:
            canvas.resize(800, 600)
            canvas.zoom = 0.05
            canvas.pan_world = Vector2(0.0, 0.0)
            self.assertIsNone(canvas._screen_circle_rect(world.structures["far-structure"].position, 10.0))
            canvas.paintEvent(None)
        finally:
            canvas.close()


if __name__ == "__main__":
    unittest.main()
