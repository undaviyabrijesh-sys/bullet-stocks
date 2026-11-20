"""Tests for offline aggregation of privacy-minimized pipeline timing logs."""
import json
from io import StringIO
import unittest
from unittest.mock import patch

from stockmind_telegram_mvp.app.pipeline_telemetry import (
    aggregate_timing_lines,
    main,
    parse_timing_line,
)


class PipelineTelemetryTests(unittest.TestCase):
    def test_parses_request_timing_with_logger_prefix(self):
        event = parse_timing_line(
            "2026-11-20 INFO bot StockMind request timing "
            "mode=chart outcome=completed total_ms=125 "
            "stages=image_prepare=5ms analysis_execution=90ms overlay_render=10ms"
        )
        self.assertEqual(
            event,
            {
                "kind": "request",
                "mode": "chart",
                "outcome": "completed",
                "total_ms": 125,
                "stages": {
                    "image_prepare": 5,
                    "analysis_execution": 90,
                    "overlay_render": 10,
                },
            },
        )

    def test_parses_request_with_no_completed_stages(self):
        self.assertEqual(
            parse_timing_line(
                "StockMind request timing mode=text outcome=failed "
                "total_ms=0 stages=none"
            ),
            {
                "kind": "request",
                "mode": "text",
                "outcome": "failed",
                "total_ms": 0,
                "stages": {},
            },
        )

    def test_aggregates_median_nearest_rank_p95_and_stage_coverage(self):
        lines = []
        for duration in range(1, 21):
            stages = f"analysis_execution={duration * 2}ms"
            if duration == 1:
                stages += " overlay_render=3ms"
            lines.append(
                "StockMind request timing mode=chart outcome=completed "
                f"total_ms={duration * 10} stages={stages}"
            )

        report = aggregate_timing_lines(lines)
        self.assertEqual(report["parsed_request_events"], 20)
        self.assertEqual(report["rejected_timing_lines"], 0)
        group = report["request_groups"][0]
        self.assertEqual(group["request_count"], 20)
        self.assertEqual(group["total_ms"]["p50_ms"], 105)
        self.assertEqual(group["total_ms"]["p95_ms"], 190)
        self.assertEqual(
            group["stages"]["analysis_execution"]["p50_ms"], 21
        )
        self.assertEqual(
            group["stages"]["analysis_execution"]["p95_ms"], 38
        )
        self.assertEqual(
            group["stages"]["overlay_render"]["sample_count"], 1
        )
        self.assertEqual(
            group["stages"]["overlay_render"]["missing_count"], 19
        )

    def test_separates_mode_and_outcome_groups(self):
        report = aggregate_timing_lines(
            [
                "StockMind request timing mode=chart outcome=completed "
                "total_ms=100 stages=analysis_execution=70ms",
                "StockMind request timing mode=chart outcome=failed "
                "total_ms=200 stages=analysis_execution=150ms",
                "StockMind request timing mode=text outcome=completed "
                "total_ms=300 stages=analysis_execution=250ms",
            ]
        )
        keys = [
            (group["mode"], group["outcome"])
            for group in report["request_groups"]
        ]
        self.assertEqual(
            keys,
            [("chart", "completed"), ("chart", "failed"), ("text", "completed")],
        )

    def test_keeps_photo_and_document_handler_metrics_separate(self):
        report = aggregate_timing_lines(
            [
                "StockMind Telegram photo download_ms=10",
                "StockMind Telegram photo download_ms=30",
                "StockMind Telegram photo handler_total_ms=80",
                "StockMind Telegram document download_ms=20",
                "StockMind Telegram document handler_total_ms=100",
            ]
        )
        self.assertEqual(report["parsed_handler_events"], 5)
        self.assertEqual(
            report["telegram_handler_metrics"]["telegram_photo_download_ms"]["p50_ms"],
            20,
        )
        self.assertEqual(
            report["telegram_handler_metrics"]["telegram_photo_handler_total_ms"]["p95_ms"],
            80,
        )
        self.assertEqual(
            report["telegram_handler_metrics"]["telegram_document_download_ms"]["p50_ms"],
            20,
        )

    def test_rejects_duplicate_or_malformed_stage_fields(self):
        report = aggregate_timing_lines(
            [
                "StockMind request timing mode=chart outcome=completed "
                "total_ms=10 stages=analysis_execution=5ms analysis_execution=6ms",
                "StockMind request timing mode=chart outcome=completed "
                "total_ms=-1 stages=none",
                "StockMind request timing mode=chart outcome=completed "
                "total_ms=10 stages=analysis_execution=not-a-duration",
                "ordinary application log line",
            ]
        )
        self.assertEqual(report["parsed_request_events"], 0)
        self.assertEqual(report["rejected_timing_lines"], 3)
        self.assertEqual(report["request_groups"], [])

    def test_ignores_unrelated_log_context_in_exported_report(self):
        report = aggregate_timing_lines(
            [
                "user_id=123 prompt='Map BTC 1H' StockMind request timing "
                "mode=chart outcome=completed total_ms=10 stages=analysis_execution=5ms"
            ]
        )
        serialized = json.dumps(report)
        self.assertNotIn("123", serialized)
        self.assertNotIn("Map BTC", serialized)

    def test_empty_input_returns_explicit_empty_coverage(self):
        report = aggregate_timing_lines([])
        self.assertEqual(report["parsed_request_events"], 0)
        self.assertEqual(report["parsed_handler_events"], 0)
        self.assertEqual(report["request_groups"], [])
        self.assertEqual(report["telegram_handler_metrics"], {})

    def test_cli_reads_standard_input_and_emits_json(self):
        source = StringIO(
            "StockMind request timing mode=text outcome=completed "
            "total_ms=12 stages=analysis_execution=8ms\n"
        )
        output = StringIO()
        with patch("sys.stdin", source), patch("sys.stdout", output):
            exit_code = main(["-"])
        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["parsed_request_events"], 1)
        self.assertEqual(report["request_groups"][0]["total_ms"]["p50_ms"], 12)


if __name__ == "__main__":
    unittest.main()
