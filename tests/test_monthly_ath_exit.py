import unittest

import pandas as pd

import monthly_ath as ath


class CapitalRecyclingExitPlanTests(unittest.TestCase):
    def setUp(self):
        self.opened = pd.Timestamp("2026-11-03 09:15:00", tz="Asia/Kolkata")
        self.checkpoint = self.opened + pd.Timedelta(minutes=1)
        self.bar_time = self.opened + pd.Timedelta(minutes=2)
        self.today = pd.Timestamp("2026-11-04")

    def make_position(self, strategy_version=3):
        pos = ath.make_position(
            entry=100.0,
            level=90.0,
            opened=self.opened.isoformat(),
            setup_month="Oct'26",
            monthly_sma20=70.0,
            monthly_sma10=90.0,
        )
        self.assertIsNotNone(pos)
        pos["strategy_version"] = strategy_version
        pos["last_processed_bar"] = self.checkpoint.isoformat()
        return pos

    def frame(self, high, low=90.0, close=None):
        return pd.DataFrame(
            [{
                "Open": 100.0,
                "High": float(high),
                "Low": float(low),
                "Close": float(close if close is not None else high),
            }],
            index=pd.DatetimeIndex([self.bar_time]),
        )

    def test_t1_books_25_percent_of_original_quantity(self):
        pos = self.make_position()
        events, closed = ath.evaluate_intraday_position(
            pos, self.frame(high=126.0, low=90.0), self.today, monthly_sma10=90.0
        )
        self.assertIsNone(closed)
        self.assertEqual([event["type"] for event in events], ["T1"])
        self.assertAlmostEqual(events[0]["sold_fraction"], 0.25)
        self.assertAlmostEqual(pos["remaining_fraction"], 0.75)

    def test_t2_activates_sma10_trailing_and_t3_books_25_percent_of_remainder(self):
        pos = self.make_position()
        events, closed = ath.evaluate_intraday_position(
            pos, self.frame(high=151.0, low=120.0), self.today, monthly_sma10=110.0
        )
        self.assertIsNone(closed)
        self.assertEqual([event["type"] for event in events], ["T1", "T2", "T3"])
        self.assertTrue(pos["trailing"])
        self.assertEqual(pos["trail_activation_date"], "2026-11-03")
        self.assertAlmostEqual(pos["sl"], 100.0)
        self.assertTrue(events[1]["trail_active"])
        self.assertAlmostEqual(events[0]["sold_fraction"], 0.25)
        self.assertAlmostEqual(events[2]["sold_fraction"], 0.1875)
        self.assertAlmostEqual(pos["remaining_fraction"], 0.5625)

    def test_monthly_sma10_trailing_stop_only_moves_up(self):
        pos = self.make_position()
        pos["t1_done"] = True
        pos["t2_done"] = True
        pos["trailing"] = True
        pos["sl"] = 105.0
        events, closed = ath.evaluate_intraday_position(
            pos, self.frame(high=115.0, low=112.0), self.today, monthly_sma10=110.0
        )
        self.assertIsNone(closed)
        self.assertEqual([event["type"] for event in events], ["TRAIL"])
        self.assertAlmostEqual(pos["sl"], 110.0)

    def test_new_strategy_does_not_wait_for_t4_target(self):
        pos = self.make_position()
        events, closed = ath.evaluate_intraday_position(
            pos, self.frame(high=205.0, low=120.0), self.today, monthly_sma10=90.0
        )
        self.assertIsNone(closed)
        self.assertNotIn("T4", [event["type"] for event in events])
        self.assertTrue(pos["trailing"])

    def test_legacy_positions_keep_their_original_t1_booking_rule(self):
        pos = self.make_position(strategy_version=2)
        events, closed = ath.evaluate_intraday_position(
            pos, self.frame(high=126.0, low=90.0), self.today, monthly_sma10=90.0
        )
        self.assertIsNone(closed)
        self.assertEqual([event["type"] for event in events], ["T1"])
        self.assertAlmostEqual(events[0]["sold_fraction"], 0.50)
        self.assertAlmostEqual(pos["remaining_fraction"], 0.50)


if __name__ == "__main__":
    unittest.main()
