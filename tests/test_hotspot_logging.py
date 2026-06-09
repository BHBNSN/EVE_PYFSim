from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
import uuid

from eve_sim.config import EngineConfig
from eve_sim.hotspot_log import aggregate_duration_by_tick, format_record_context, load_hotspot_records, summarize_records
from eve_sim.sim_logging import get_sim_logger, log_sim_event


@contextmanager
def _workspace_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".tmp" / "test-artifacts" / f"hotspot-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class HotspotLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_sim_logger(EngineConfig())

    def test_hotspot_logs_are_written_to_separate_file(self) -> None:
        with _workspace_temp_dir() as root:
            detail_log = root / "detail.log"
            hotspot_log = root / "hotspot.log"
            logger = get_sim_logger(
                EngineConfig(
                    detailed_logging=True,
                    hotspot_logging=True,
                    detail_log_file=str(detail_log),
                    hotspot_log_file=str(hotspot_log),
                )
            )

            log_sim_event(logger, "user_operation", action="drag_target", ship_id="BLUE-001")
            log_sim_event(
                logger,
                "hotspot",
                name="combat.pyfa_resolve_batch",
                duration_ms=12.5,
                tick=3,
                ship_ids=["BLUE-001", "BLUE-002"],
            )

            for handler in logger.handlers:
                handler.flush()

            detail_text = detail_log.read_text(encoding="utf-8")
            hotspot_text = hotspot_log.read_text(encoding="utf-8")

            self.assertIn("event=user_operation", detail_text)
            self.assertNotIn("event=hotspot", detail_text)
            self.assertIn("event=hotspot", hotspot_text)
            self.assertNotIn("event=user_operation", hotspot_text)

            get_sim_logger(EngineConfig())

    def test_hotspot_log_parser_summarizes_records(self) -> None:
        with _workspace_temp_dir() as root:
            log_path = root / "hotspot.log"
            log_path.write_text(
                "\n".join(
                    [
                        "2026-03-09 11:50:14 | INFO | event=hotspot duration_ms=170.6291 name=combat.pyfa_resolve_batch tick=1 batch_size=20 resolve_cache=miss ship_ids=RED-001,RED-002",
                        "2026-03-09 11:50:15 | INFO | event=hotspot duration_ms=6.3044 name=engine.combat tick=1 slice_index=9",
                        "2026-03-09 11:50:16 | INFO | event=hotspot duration_ms=8.0000 name=engine.combat tick=2 slice_index=0",
                        "2026-03-09 11:50:17 | INFO | event=user_operation action=zoom",
                    ]
                ),
                encoding="utf-8",
            )

            records = load_hotspot_records(log_path)
            summaries = summarize_records(records)
            per_tick = aggregate_duration_by_tick([record for record in records if record.name == "engine.combat"])

            self.assertEqual(len(records), 3)
            self.assertEqual(summaries[0].name, "combat.pyfa_resolve_batch")
            self.assertAlmostEqual(summaries[0].total_ms, 170.6291)
            self.assertEqual(records[0].fields.get("resolve_cache"), "miss")
            self.assertIn("resolve_cache=miss", format_record_context(records[0]))
            self.assertEqual(len(per_tick), 2)
            self.assertEqual(per_tick[0].tick, 1)
            self.assertAlmostEqual(per_tick[0].total_ms, 6.3044)
            self.assertEqual(per_tick[1].tick, 2)
            self.assertAlmostEqual(per_tick[1].total_ms, 8.0)



if __name__ == "__main__":
    unittest.main()
