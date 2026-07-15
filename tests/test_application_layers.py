from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

from eve_sim.application import MatchApplication
from eve_sim.application.commands import (
    AdvanceTicks,
    AssignShipsToSquad,
    IssueSquadApproach,
    IssueSquadFocus,
    IssueSquadMove,
    InduceShips,
    SetShipModuleChargeLock,
)
from eve_sim.lan_command_adapter import LanCommandAdapter
from eve_sim.lan_session import _decode_packet, _encode_packet
from eve_sim.application.queries import OverviewQuery
from eve_sim.config import EngineConfig
from eve_sim.gui.adapters import ApplicationRuntimeView, GuiCommandAdapter
from eve_sim.gui.models import PreferencesStore, UiPreferences
from eve_sim.math2d import Vector2
from eve_sim.domain.squad_service import SquadLeadershipService
from eve_sim.models import CombatState, FitDescriptor, FleetIntent, NavigationState, QualityLevel, QualityState, ShipEntity, Team, VitalState
from eve_sim.pyfa_bridge import PyfaBridge
from eve_sim.serialization import SnapshotLoader
from eve_sim.simulation_engine import SimulationEngine
from eve_sim.systems import CombatSystem
from eve_sim.world import WorldState


def _ship(ship_id: str, team: Team) -> ShipEntity:
    fit = FitDescriptor(
        fit_key=ship_id,
        ship_name="Test Hull",
        role="test",
        base_dps=0.0,
        volley=0.0,
        optimal_range=0.0,
        falloff=0.0,
        tracking=0.0,
        max_target_range=100_000.0,
        max_cap=100.0,
        shield_hp=100.0,
        armor_hp=100.0,
        structure_hp=100.0,
    )
    profile = PyfaBridge().build_profile(fit)
    return ShipEntity(
        ship_id=ship_id,
        team=team,
        squad_id="A",
        fit=fit,
        profile=profile,
        nav=NavigationState(Vector2(0.0, 0.0), Vector2(0.0, 0.0), 0.0, profile.max_speed, system_id="alpha"),
        combat=CombatState(),
        vital=VitalState(100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        quality=QualityState(QualityLevel.REGULAR, 0.0, 0.0, 0.0),
    )


def _application() -> MatchApplication:
    blue = _ship("blue", Team.BLUE)
    red = _ship("red", Team.RED)
    engine = SimulationEngine(WorldState(ships={blue.ship_id: blue, red.ship_id: red}), EngineConfig(), CombatSystem(PyfaBridge()))
    return MatchApplication.from_engine(engine, match_id="application-test")


def test_tick_bound_command_is_queued_applied_once_and_deduplicated() -> None:
    application = _application()
    command = IssueSquadFocus(command_id="focus-1", team=Team.BLUE, squad_id="A", target_id="red")

    queued = application.execute(command)
    duplicate = application.execute(command)

    assert queued.accepted and queued.applied_tick is None
    assert not duplicate.accepted and duplicate.error_code == "duplicate_command"
    assert application.session.world.squad_focus_queues == {}

    result = application.step()[0]

    assert result.tick == 1
    assert application.session.world.squad_focus_queues["BLUE:A"] == ["red"]
    assert application.session.command_results[command.command_id].accepted


def test_session_command_ids_are_deduplicated_by_the_same_ledger() -> None:
    application = _application()
    command = AdvanceTicks(command_id="advance-once", count=1)

    first = application.execute(command)
    duplicate = application.execute(command)

    assert first.accepted and first.applied_tick == 1
    assert not duplicate.accepted and duplicate.error_code == "duplicate_command"
    assert application.session.world.tick == 1


def test_leadership_refresh_removes_all_orphaned_squad_state() -> None:
    world = WorldState()
    key = "BLUE:EMPTY"
    world.intents[key] = FleetIntent(squad_id="EMPTY")
    world.squad_propulsion_commands[key] = True
    world.squad_leader_speed_limits[key] = 1_000.0
    world.squad_focus_queues[key] = ["missing"]
    world.squad_focus_updated_at[key] = 10.0

    SquadLeadershipService().refresh(world)

    assert key not in world.intents
    assert key not in world.squad_propulsion_commands
    assert key not in world.squad_leader_speed_limits
    assert key not in world.squad_focus_queues
    assert key not in world.squad_focus_updated_at


def test_invalid_command_does_not_partially_mutate_world() -> None:
    application = _application()
    before = deepcopy(application.session.world)

    application.execute(IssueSquadFocus(team=Team.BLUE, squad_id="A", target_id="missing"))
    application.step()

    assert application.session.world.squad_focus_queues == before.squad_focus_queues
    assert application.session.world.intents == before.intents


def test_scenario_assignment_validates_the_full_batch_before_mutation() -> None:
    application = _application()
    application.execute(
        AssignShipsToSquad(
            team=Team.BLUE,
            ship_ids=("blue", "red"),
            squad_id="NEW",
        )
    )

    application.step()

    assert application.session.world.ships["blue"].squad_id == "A"
    assert application.session.world.ships["red"].squad_id == "A"


def test_scenario_induction_validates_the_full_batch_before_mutation() -> None:
    application = _application()
    blue = application.session.world.ships["blue"]
    red = application.session.world.ships["red"]
    blue.deployed = red.deployed = False
    blue.vital.alive = red.vital.alive = False
    before_blue = deepcopy(blue)
    before_red = deepcopy(red)

    application.execute(
        InduceShips(
            team=Team.BLUE,
            ship_ids=("blue", "red"),
            center=Vector2(5_000.0, 6_000.0),
            system_id="beta",
        )
    )
    application.step()

    assert blue == before_blue
    assert red == before_red


def test_gui_and_lan_adapters_reach_the_same_typed_command_handler() -> None:
    gui_application = _application()
    lan_application = _application()

    GuiCommandAdapter(gui_application).focus(Team.BLUE, "A", "red")
    lan_command = LanCommandAdapter().decode(
        {"kind": "SQUAD_ATTACK", "command_id": "lan-focus-1", "squad_id": "A", "target_id": "red"},
        team=Team.BLUE,
    )
    lan_application.execute(lan_command)
    gui_application.step()
    lan_application.step()

    assert gui_application.session.world.squad_focus_queues == lan_application.session.world.squad_focus_queues


def test_lan_command_round_trip_preserves_typed_command_contract() -> None:
    adapter = LanCommandAdapter()
    commands = (
        IssueSquadApproach(
            command_id="approach-1",
            team=Team.BLUE,
            squad_id="A",
            target_id="red",
            range_m=12_500.0,
        ),
        AssignShipsToSquad(
            command_id="assign-1",
            team=Team.BLUE,
            ship_ids=("blue",),
            squad_id="B",
        ),
        SetShipModuleChargeLock(
            command_id="charge-1",
            team=Team.BLUE,
            ship_id="blue",
            module_id="mod-1",
            charge_name="Focused Warp Disruption Script",
        ),
    )

    for command in commands:
        assert adapter.decode(adapter.encode(command), team=Team.BLUE) == command


def test_lan_wire_requires_the_current_explicit_protocol_contract() -> None:
    encoded = _encode_packet("COMMAND", {"kind": "PING"}, 1)
    assert _decode_packet(encoded.rstrip(b"\n")) == ("COMMAND", {"kind": "PING"})

    missing_protocol = json.dumps(
        {"version": 1, "kind": "COMMAND", "payload": {"kind": "PING"}}
    ).encode("utf-8")
    old_version = json.dumps(
        {"protocol": "EVE_SIM_LAN", "version": 0, "kind": "COMMAND", "payload": {"kind": "PING"}}
    ).encode("utf-8")

    assert _decode_packet(missing_protocol) is None
    assert _decode_packet(old_version) is None


def test_ui_preferences_reject_unsupported_versions_instead_of_migrating(tmp_path) -> None:
    store = PreferencesStore()
    store.path = tmp_path / "preferences.json"
    store.path.write_text(
        json.dumps({"config_version": store.CURRENT_VERSION - 1, "selected_squad": "OLD"}),
        encoding="utf-8",
    )

    assert store.load() == UiPreferences()


def test_charge_command_keeps_fit_rewrite_out_of_the_presentation_payload() -> None:
    application = _application()
    ship = application.session.world.ships["blue"]
    ship.fit_text = (
        "[Onyx, Test]\n"
        "Warp Disruption Field Generator I, Focused Warp Disruption Script\n"
    )
    command = SetShipModuleChargeLock(
        team=Team.BLUE,
        ship_id=ship.ship_id,
        module_id="mod-1",
        charge_name="",
    )

    application.execute(command)
    assert "Focused Warp Disruption Script" in ship.fit_text
    application.step()

    assert ship.fit_text == "[Onyx, Test]\nWarp Disruption Field Generator I"
    assert ship.locked_module_charges == {"mod-1": ""}


def test_queries_and_snapshot_builder_are_read_only() -> None:
    application = _application()
    before = deepcopy(application.session.world)

    overview = application.query_service.overview(OverviewQuery(alive_only=True))
    snapshot = application.snapshot()

    assert len(overview.ships) == 2
    assert snapshot["ships"]["blue"]["ship_id"] == "blue"
    assert application.session.world == before


def test_presentation_runtime_view_is_detached_from_authoritative_world() -> None:
    application = _application()
    view = ApplicationRuntimeView(application)

    view.world.ships["blue"].squad_id = "PRESENTATION-ONLY"
    assert application.session.world.ships["blue"].squad_id == "A"

    application.session.world.ships["blue"].nav.position = Vector2(12_000.0, 3_000.0)
    view.refresh()
    assert view.world.ships["blue"].nav.position == Vector2(12_000.0, 3_000.0)
    assert view.world.ships["blue"].fit.role == "test"


def test_core_packages_do_not_import_qt_or_presentation_layers() -> None:
    root = Path(__file__).resolve().parents[1] / "eve_sim"
    for package in ("application", "domain", "serialization"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            assert not any("PySide" in name or ".gui" in name or name.startswith("eve_sim.gui") for name in imports), path

    engine_tree = ast.parse((root / "simulation_engine.py").read_text(encoding="utf-8"))
    engine_imports = [node.module or "" for node in ast.walk(engine_tree) if isinstance(node, ast.ImportFrom)]
    assert not any("gui" in name or "lan" in name for name in engine_imports)


def test_dependency_edges_are_explicit_and_do_not_use_wildcard_aggregators() -> None:
    root = Path(__file__).resolve().parents[1] / "eve_sim"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ), path

    for path in (root / "domain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("application" in module for module in imported_modules), path

    for leaf in ("battle_canvas.py", "dialogs.py", "fleet_setup_dialog.py", "table_models.py"):
        source = (root / "gui" / leaf).read_text(encoding="utf-8-sig")
        assert "lan_commands" not in source
        assert "lan_session" not in source

    assert not (root / "systems" / "combat_common.py").exists()

    gui_source = "\n".join(path.read_text(encoding="utf-8-sig") for path in (root / "gui").rglob("*.py"))
    assert "from ..simulation_engine import" not in gui_source
    assert "from ...simulation_engine import" not in gui_source
    assert "CombatSystem(" not in gui_source


def test_layer_boundaries_do_not_regress_to_legacy_shortcuts() -> None:
    root = Path(__file__).resolve().parents[1] / "eve_sim"
    application_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "application").rglob("*.py")
    )
    handlers_source = (root / "application" / "command_handlers.py").read_text(encoding="utf-8")
    engine_source = (root / "simulation_engine.py").read_text(encoding="utf-8")
    builder_source = (root / "serialization" / "snapshot_builder.py").read_text(encoding="utf-8")
    agents_source = (root / "agents.py").read_text(encoding="utf-8")
    main_window_source = (root / "gui" / "main_window.py").read_text(encoding="utf-8")

    assert "lan_commands" not in application_source
    assert "session.engine.deployables" not in handlers_source
    assert "_snapshot_payload" not in engine_source
    assert "_snapshot_payload" not in builder_source
    assert "def snapshot(" not in engine_source
    assert "def _apply_squad_follow_state" not in agents_source
    assert "SquadFollowService.apply" in agents_source
    assert "application._apply_queued_commands" not in main_window_source
    assert "lan_client.send_command" not in main_window_source
    assert "_rewrite_fit_text_with_lock_rules" not in main_window_source
    assert "_squad_approach_targets" not in main_window_source
    assert "_engine_config_payload" not in main_window_source
    assert "engine.tidi_factor =" not in main_window_source
    assert "self.engine" not in main_window_source
    assert "application.session" not in main_window_source
    assert "CommanderAgent" not in main_window_source
    assert "manual_setup" not in main_window_source
    assert "self.squad_ids" not in agents_source
    assert "squad_ids: list[str]" not in agents_source
    assert "SimulationEngine" not in builder_source
    assert "build_from_engine" not in builder_source
    assert "engine._logger" not in application_source
    assert "engine._dt" not in application_source
    assert "engine.ship_agents" not in application_source
    assert "engine.combat" not in application_source
    assert "engine.deployables" not in application_source
    assert "_battle_recorder" not in main_window_source
    assert "_record_battle_snapshot" not in main_window_source
    assert "def _focus_key" not in "\n".join(
        path.read_text(encoding="utf-8-sig") for path in root.rglob("*.py")
    )
    assert not (root / "gui" / "adapters" / "snapshot_adapter.py").exists()
    assert not (root / "replay" / "snapshot_mapper.py").exists()


def test_snapshot_loader_preserves_leader_location_version_without_refreshing() -> None:
    application = _application()
    application.session.world.squad_leaders["BLUE:A"] = "blue"
    application.session.world.squad_leader_location_versions["BLUE:A"] = 7

    restored = SnapshotLoader().load_world(application.snapshot())

    assert restored.squad_leader_location_versions["BLUE:A"] == 7
    assert restored.squad_leader_locations["BLUE:A"].leader_id == "blue"
    assert restored.squad_leader_locations["BLUE:A"].system_id == "alpha"
    assert restored.squad_leader_locations["BLUE:A"].location_version == 7


def test_move_command_enters_world_only_at_tick_boundary() -> None:
    application = _application()
    application.execute(IssueSquadMove(team=Team.BLUE, squad_id="A", target=Vector2(12_000.0, 3_000.0)))

    assert application.session.world.intents == {}
    application.step()

    assert application.session.world.intents["BLUE:A"].target_position == Vector2(12_000.0, 3_000.0)
