import unittest
import pandas as pd
import monthly_ath as ath


class DeferredTrailAndExpiryTests(unittest.TestCase):
    def test_sma10_trail_applies_on_next_session(self):
        opened = pd.Timestamp("2026-11-03 09:15:00", tz="Asia/Kolkata")
        pos = ath.make_position(100.0, 90.0, opened.isoformat(), "Oct'26", 70.0, 90.0)
        pos.update({
            "last_processed_bar": (opened + pd.Timedelta(minutes=1)).isoformat(),
            "t1_done": True,
            "t2_done": True,
            "trailing": True,
            "trail_activation_date": "2026-11-03",
            "sl": 105.0,
        })
        bar = pd.DataFrame(
            [{"Open": 113.0, "High": 115.0, "Low": 112.0, "Close": 114.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2026-11-04 09:17:00", tz="Asia/Kolkata")]),
        )
        events, closed = ath.evaluate_intraday_position(
            pos, bar, pd.Timestamp("2026-11-04"), monthly_sma10=110.0
        )
        self.assertIsNone(closed)
        self.assertEqual([event["type"] for event in events], ["TRAIL"])
        self.assertEqual(pos["sl"], 110.0)

    def test_sma10_trail_does_not_apply_on_t2_session(self):
        opened = pd.Timestamp("2026-11-03 09:15:00", tz="Asia/Kolkata")
        pos = ath.make_position(100.0, 90.0, opened.isoformat(), "Oct'26", 70.0, 90.0)
        pos.update({
            "last_processed_bar": (opened + pd.Timedelta(minutes=1)).isoformat(),
            "t1_done": True,
            "t2_done": True,
            "trailing": True,
            "trail_activation_date": "2026-11-03",
            "sl": 100.0,
        })
        bar = pd.DataFrame(
            [{"Open": 106.0, "High": 115.0, "Low": 105.0, "Close": 114.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2026-11-03 09:17:00", tz="Asia/Kolkata")]),
        )
        events, closed = ath.evaluate_intraday_position(
            pos, bar, pd.Timestamp("2026-11-03"), monthly_sma10=110.0
        )
        self.assertIsNone(closed)
        self.assertEqual(events, [])
        self.assertEqual(pos["sl"], 100.0)

    def test_90th_trading_session_closes_position(self):
        opened = pd.Timestamp("2026-11-03 09:15:00", tz="Asia/Kolkata")
        pos = ath.make_position(100.0, 90.0, opened.isoformat(), "Oct'26", 70.0, 90.0)
        pos["last_processed_bar"] = (opened + pd.Timedelta(minutes=1)).isoformat()
        bar = pd.DataFrame(
            [{"Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0}],
            index=pd.DatetimeIndex([opened + pd.Timedelta(minutes=2)]),
        )
        events, closed = ath.evaluate_intraday_position(
            pos, bar, pd.Timestamp("2026-11-04"), monthly_sma10=90.0, trading_days_held=90
        )
        self.assertEqual(events, [])
        self.assertIsNotNone(closed)
        self.assertEqual(closed["result"], "EXPIRED")


if __name__ == "__main__":
    unittest.main()
