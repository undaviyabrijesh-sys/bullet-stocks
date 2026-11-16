import unittest

import pandas as pd

import monthly_ath as ath


class MonthlySMA20UndercutFilterTests(unittest.TestCase):
    @staticmethod
    def make_history(with_prior_undercut=False, current_month_only=False):
        rows = []
        months = pd.period_range("2021-06", "2026-11" if current_month_only else "2026-10", freq="M")
        for month_index, month in enumerate(months):
            dates = pd.bdate_range(month.start_time, month.end_time)
            # The completed monthly closes form a steadily rising closing-ATH series.
            close = 100.0 + 2.0 * month_index
            for day_index, date in enumerate(dates[:20]):
                low = close - 1.0
                if with_prior_undercut and month_index == 40 and day_index == 10:
                    # A single historical monthly wick pierces the 20-month SMA.
                    low = 50.0
                if current_month_only and month_index == len(months) - 1 and day_index == 10:
                    # An unfinished current month must not satisfy the historical filter.
                    low = 50.0
                rows.append({
                    "Open": close,
                    "High": close + 1.0,
                    "Low": low,
                    "Close": close,
                    "Volume": 1_000_000,
                })
        return pd.DataFrame(
            rows,
            index=pd.DatetimeIndex([row_date for row_date in pd.bdate_range("2021-06-01", "2026-11-30")][:0])
            if False else None,
        )

    def build_frame(self, with_prior_undercut=False, current_month_only=False):
        rows = []
        months = pd.period_range("2021-06", "2026-11" if current_month_only else "2026-10", freq="M")
        for month_index, month in enumerate(months):
            dates = pd.bdate_range(month.start_time, month.end_time)[:20]
            close = 100.0 + 2.0 * month_index
            for day_index, date in enumerate(dates):
                low = close - 1.0
                if with_prior_undercut and month_index == 40 and day_index == 10:
                    low = 50.0
                if current_month_only and month_index == len(months) - 1 and day_index == 10:
                    low = 50.0
                rows.append((date, close, low))
        frame = pd.DataFrame(
            {
                "Open": [close for _, close, _ in rows],
                "High": [close + 1.0 for _, close, _ in rows],
                "Low": [low for _, _, low in rows],
                "Close": [close for _, close, _ in rows],
                "Volume": [1_000_000 for _ in rows],
            },
            index=pd.DatetimeIndex([date for date, _, _ in rows]),
        )
        return frame

    def test_rejects_ath_stock_without_any_prior_monthly_sma20_undercut(self):
        result = ath.analyse(
            self.build_frame(), nifty_6m=0.05, today=pd.Timestamp("2026-11-15")
        )
        self.assertIsNone(result)

    def test_accepts_ath_stock_after_a_prior_monthly_low_undercut_sma20(self):
        result = ath.analyse(
            self.build_frame(with_prior_undercut=True),
            nifty_6m=0.05,
            today=pd.Timestamp("2026-11-15"),
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["monthly_sma20_undercut"])
        self.assertEqual(result["monthly_sma20_undercut_month"], "Oct'24")

    def test_unfinished_current_month_cannot_satisfy_the_undercut_filter(self):
        result = ath.analyse(
            self.build_frame(current_month_only=True),
            nifty_6m=0.05,
            today=pd.Timestamp("2026-11-15"),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
