"""Outcome-specific Telegram upload latency aggregation tests."""
import unittest

from stockmind_telegram_mvp.app.pipeline_telemetry import aggregate_timing_lines


class PipelineOutcomeLatencyTests(unittest.TestCase):
    def test_summarizes_handler_and_correlated_download_latency_by_outcome(self):
        first = "a" * 32
        second = "b" * 32
        rejected = "c" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram photo request_id={first} download_ms=10",
                f"StockMind Telegram photo request_id={first} outcome=completed handler_total_ms=100",
                f"StockMind Telegram photo request_id={second} download_ms=30",
                f"StockMind Telegram photo request_id={second} outcome=completed handler_total_ms=300",
                f"StockMind Telegram photo request_id={rejected} outcome=rejected handler_total_ms=50",
            ]
        )

        groups = {
            (row["upload_kind"], row["outcome"]): row
            for row in report["telegram_handler_outcome_latency"]
        }
        completed = groups[("photo", "completed")]
        self.assertEqual(completed["handler_total_ms"]["sample_count"], 2)
        self.assertEqual(completed["handler_total_ms"]["p50_ms"], 200)
        self.assertEqual(completed["handler_total_ms"]["p95_ms"], 300)
        self.assertEqual(completed["download_ms"]["sample_count"], 2)
        self.assertEqual(completed["download_ms"]["p50_ms"], 20)
        self.assertEqual(completed["download_ms"]["p95_ms"], 30)
        self.assertEqual(completed["download_not_correlated_count"], 0)

        rejected_row = groups[("photo", "rejected")]
        self.assertEqual(rejected_row["handler_total_ms"]["p50_ms"], 50)
        self.assertNotIn("download_ms", rejected_row)
        self.assertEqual(rejected_row["download_not_correlated_count"], 1)

    def test_conflicted_request_is_not_used_for_download_outcome_pairing(self):
        request_id = "d" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram document request_id={request_id} download_ms=5",
                f"StockMind Telegram document request_id={request_id} download_ms=6",
                f"StockMind Telegram document request_id={request_id} outcome=completed handler_total_ms=60",
            ]
        )

        self.assertEqual(report["correlation_summary"]["conflicted_request_id_count"], 1)
        group = report["telegram_handler_outcome_latency"][0]
        self.assertEqual(group["upload_kind"], "document")
        self.assertEqual(group["outcome"], "completed")
        self.assertEqual(group["handler_total_ms"]["sample_count"], 1)
        self.assertNotIn("download_ms", group)
        self.assertEqual(group["correlated_download_sample_count"], 0)
        self.assertEqual(group["download_not_correlated_count"], 1)

    def test_legacy_handler_outcome_contributes_total_but_not_download_latency(self):
        report = aggregate_timing_lines(
            ["StockMind Telegram photo outcome=download_failed handler_total_ms=25"]
        )

        group = report["telegram_handler_outcome_latency"][0]
        self.assertEqual(group["handler_total_ms"]["p50_ms"], 25)
        self.assertNotIn("download_ms", group)
        self.assertEqual(group["download_not_correlated_count"], 1)

    def test_photo_and_document_outcome_groups_remain_separate(self):
        report = aggregate_timing_lines(
            [
                "StockMind Telegram photo outcome=completed handler_total_ms=10",
                "StockMind Telegram document outcome=completed handler_total_ms=20",
            ]
        )

        self.assertEqual(
            [(row["upload_kind"], row["outcome"]) for row in report["telegram_handler_outcome_latency"]],
            [("document", "completed"), ("photo", "completed")],
        )


    def test_duplicate_handler_completion_events_are_excluded_from_outcome_latency(self):
        request_id = "e" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram photo request_id={request_id} "
                "outcome=completed handler_total_ms=100",
                f"StockMind Telegram photo request_id={request_id} "
                "outcome=completed handler_total_ms=200",
            ]
        )

        self.assertEqual(report["correlation_summary"]["conflicted_request_id_count"], 1)
        self.assertEqual(
            report["correlation_summary"]["conflicted_handler_outcome_request_id_count"],
            1,
        )
        self.assertEqual(report["correlation_summary"]["handler_outcome_event_count"], 0)
        self.assertEqual(report["telegram_handler_outcomes"], {})
        self.assertEqual(report["telegram_handler_outcome_latency"], [])

    def test_duplicate_download_does_not_discard_unique_handler_outcome_latency(self):
        request_id = "f" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram photo request_id={request_id} download_ms=5",
                f"StockMind Telegram photo request_id={request_id} download_ms=6",
                f"StockMind Telegram photo request_id={request_id} "
                "outcome=completed handler_total_ms=60",
            ]
        )

        self.assertEqual(report["correlation_summary"]["conflicted_request_id_count"], 1)
        self.assertEqual(
            report["correlation_summary"]["conflicted_handler_outcome_request_id_count"],
            0,
        )
        self.assertEqual(report["correlation_summary"]["handler_outcome_event_count"], 1)
        group = report["telegram_handler_outcome_latency"][0]
        self.assertEqual(group["outcome"], "completed")
        self.assertEqual(group["handler_total_ms"]["sample_count"], 1)
        self.assertEqual(group["handler_total_ms"]["p50_ms"], 60)
        self.assertEqual(group["correlated_download_sample_count"], 0)
        self.assertEqual(group["download_not_correlated_count"], 1)
        self.assertNotIn("download_ms", group)

    def test_upload_kind_conflict_excludes_correlated_handler_outcome(self):
        request_id = "g" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram photo request_id={request_id} download_ms=5",
                f"StockMind Telegram document request_id={request_id} "
                "outcome=completed handler_total_ms=60",
            ]
        )

        self.assertEqual(report["correlation_summary"]["conflicted_request_id_count"], 1)
        self.assertEqual(
            report["correlation_summary"]["conflicted_handler_outcome_request_id_count"],
            1,
        )
        self.assertEqual(report["correlation_summary"]["handler_outcome_event_count"], 0)
        self.assertEqual(report["telegram_handler_outcomes"], {})
        self.assertEqual(report["telegram_handler_outcome_latency"], [])


    def test_upload_kind_conflict_without_outcome_does_not_count_as_outcome_conflict(self):
        request_id = "h" * 32
        report = aggregate_timing_lines(
            [
                f"StockMind Telegram photo request_id={request_id} download_ms=5",
                f"StockMind Telegram document request_id={request_id} download_ms=6",
            ]
        )

        self.assertEqual(report["correlation_summary"]["conflicted_request_id_count"], 1)
        self.assertEqual(
            report["correlation_summary"]["conflicted_handler_outcome_request_id_count"],
            0,
        )
        self.assertEqual(report["correlation_summary"]["handler_outcome_event_count"], 0)
        self.assertEqual(report["telegram_handler_outcome_latency"], [])


if __name__ == "__main__":
    unittest.main()
