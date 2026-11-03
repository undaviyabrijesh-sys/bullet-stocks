# Bullet Stocks

Monthly ATH breakout scanner with Discord alerts and a Streamlit paper portfolio.

## Paper portfolio dashboard

The dashboard is a simulation only. It does not connect to a broker or place real orders.

Default portfolio settings:
- Starting capital: ₹10,00,000
- Allocation per new entry: ₹50,000
- Auto paper processing: every 5 minutes (can be switched off in the sidebar)

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
