import unittest

import pandas as pd

import monthly_ath as ath


class MonthlySMA20UndercutFilterTests(unittest.TestCase):
    def build_frame(self, with_prior_close_undercut=False, wick_only=False):
        rows = []
        months = pd.period_range("2021-06", "2026-10", freq="M")
        for month_index, month in enumerate(months):
            dates = pd.bdate_range(month.start_time, month.end_time)[:20]
            close = 100.0 + 2.0 * month_index
            if with_prior_close_undercut and month_index == 40:
                # Close below the 20-month SMA, then later recover to a closing ATH.
                close -= 30.0
            for day_index, date in enumerate(dates):
                low = close - 1.0
                if wick_only and month_index == 40 and day_index == 10:
                    # Price wicks below the SMA, but the monthly candle still closes above it.
                    low = 50.0
                rows.append((date, close, low))

        return pd.DataFrame(
            {
                "Open": [close for _, close, _ in rows],
                "High": [close + 1.0 for _, close, _ in rows],
                "Low": [low for _, _, low in rows],
                "Close": [close for _, close, _ in rows],
                "Volume": [1_000_000 for _ in rows],
            },
            index=pd.DatetimeIndex([date for date, _, _ in rows]),
        )

    def test_rejects_ath_stock_without_any_prior_monthly_close_below_sma20(self):
        result = ath.analyse(
            self.build_frame(), nifty_6m=0.05, today=pd.Timestamp("2026-11-15")
        )
        self.assertIsNone(result)

    def test_accepts_ath_stock_after_a_prior_monthly_close_below_sma20(self):
        result = ath.analyse(
            self.build_frame(with_prior_close_undercut=True),
            nifty_6m=0.05,
            today=pd.Timestamp("2026-11-15"),
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["monthly_sma20_undercut"])
        self.assertEqual(result["monthly_sma20_undercut_month"], "Oct'24")

    def test_wick_only_below_sma20_does_not_qualify(self):
        result = ath.analyse(
            self.build_frame(wick_only=True),
            nifty_6m=0.05,
            today=pd.Timestamp("2026-11-15"),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
