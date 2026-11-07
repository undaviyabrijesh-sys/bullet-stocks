# Bullet Stocks

Monthly ATH breakout scanner with Discord alerts and a Streamlit paper portfolio.

## StockMind Telegram MVP

The StockMind Telegram assistant supports multilingual Indian-equity and crypto research, chart screenshot DNS analysis, and offline human-reviewed zone benchmarking.

- [StockMind Telegram MVP setup and usage](stockmind_telegram_mvp/README.md)
- [DNS screenshot benchmark methodology](stockmind_telegram_mvp/ZONE_BENCHMARK.md)
- [Strict JSON input policy](stockmind_telegram_mvp/STRICT_JSON.md)

The chart benchmark is an offline evaluation aid. Synthetic unit tests do not establish real-market chart accuracy; only independently reviewed screenshots paired with actual model predictions are eligible for that evaluation.

## Cyber trading desk dashboard

The Streamlit app now uses a cyberpunk multi-workspace layout:

- **Command Center:** portfolio equity, cash, exposure, live open-position monitor and top signals.
- **Market Scanner:** LTP, day change, day high/low, trigger distance, monthly ATH level, monthly SMA20/SMA10, volume ratio, relative strength and setup score.
- **Positions:** entry, remaining quantity, MTM, active stop, and T1–T4 progress.
- **Portfolio Risk:** deployed capital, risk to active stop and configurable capital/allocation.
- **Activity Log:** timestamped paper entries, target bookings, stop exits and trailing-stop updates.

Quotes are batch-fetched from Yahoo Finance 1-minute candles and cached briefly to reduce repeated requests. LTP is the latest available Yahoo 1-minute close, not a guaranteed real-time exchange tick. The dashboard refreshes quote panels every 60 seconds; automatic paper processing remains on a five-minute cadence and can be disabled.

The dashboard is a simulation only. It does not connect to a broker or place real orders.

Default portfolio settings:
- Starting capital: ₹10,00,000
- Allocation per new entry: ₹50,000
- Auto paper processing: every 5 minutes (toggle in the Command Center)

The app reads `watchlist.json` produced by `monthly_ath.py`, checks intraday Yahoo Finance prices, and records paper positions in a local SQLite database named `paper_portfolio.db`.

### Run locally

```bash
pip install -r requirements.txt
python monthly_ath.py
streamlit run app.py
```

Keep the scanner running on its normal schedule to refresh the eligible-stock watchlist. The Streamlit app should be run on a machine that remains online if you want periodic paper signal processing.

### Paper trade management

- Entry: first observed 1-minute price at/above the watchlist trigger, with a 6% maximum-chase filter.
- Initial stop: higher of entry less 25% or the latest completed-month 20 SMA.
- T1 (+25%): sell 50% of original quantity.
- T2 (+40%): move stop to entry.
- T3 (+50%): sell 50% of remaining quantity.
- T4 (+100%): activate trailing stop using the latest completed-month 10 SMA.

Prices are sourced from Yahoo Finance and may be delayed, incomplete, or unavailable. This is a research/paper-trading tool, not an execution-grade trading system. The app requires internet access and local persistent storage for its SQLite database.
