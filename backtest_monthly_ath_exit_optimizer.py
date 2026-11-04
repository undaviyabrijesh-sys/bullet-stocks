import json, time, warnings
from io import StringIO
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
START = "2016-01-01"
BATCH_SIZE = 40
MAX_HOLD_DAYS = 90
ENTRY_BUFFER = 0.02
STAKE = 10_000
T1_BOOK = [0.25, 0.50, 0.75]       # fraction of original quantity
T3_BOOK = [0.25, 0.50, 0.75]       # fraction of quantity remaining after T1
TRAIL_START = ["T2", "T3", "T4"]
TRAIL_SMA = [5, 10, 15, 20]

def get_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(StringIO(r.text))
    d.columns = [str(c).strip().upper() for c in d.columns]
    d["SERIES"] = d["SERIES"].astype(str).str.strip()
    return sorted(set(d.loc[d["SERIES"] == "EQ", "SYMBOL"].dropna().astype(str).str.strip() + ".NS"))

def download_batch(symbols):
    try:
        raw = yf.download(symbols, start=START, interval="1d", auto_adjust=False,
                          group_by="ticker", threads=True, progress=False, timeout=30)
        out = {}
        for sym in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym in raw.columns.get_level_values(0):
                        d = raw[sym].copy()
                    elif sym in raw.columns.get_level_values(1):
                        d = raw.xs(sym, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    if len(symbols) != 1: continue
                    d = raw.copy()
                d = d.dropna(subset=["High", "Low", "Close"])
                if len(d) > 250: out[sym] = d
            except Exception:
                pass
        return out
    except Exception as e:
        print("Download error:", str(e)[:160])
        return {}

def generate_signals(sym, d):
    d = d.copy()
    if d.index.tz is not None: d.index = d.index.tz_localize(None)
    m = d.resample("ME").agg({"High": "max", "Close": "last"})
    m["SMA20"] = m["Close"].rolling(20).mean()
    m["SMA5"] = m["Close"].rolling(5).mean()
    m["SMA10"] = m["Close"].rolling(10).mean()
    m["SMA15"] = m["Close"].rolling(15).mean()
    signals = []
    for i in range(48, len(m)-1):
        setup = m.iloc[i]
        prior = m["Close"].iloc[:i].dropna()
        if pd.isna(setup["Close"]) or setup["Close"] < prior.max() or pd.isna(setup["SMA20"]):
            continue
        trigger = float(setup["High"] * (1 + ENTRY_BUFFER))
        start = m.index[i] + pd.offsets.MonthBegin(1)
        end = m.index[i+1]
        window = d.loc[(d.index >= start) & (d.index <= end)]
        crossed = window[window["High"] >= trigger]
        if crossed.empty: continue
        eb = crossed.iloc[0]
        ed = crossed.index[0]
        entry = max(trigger, float(eb["Open"])) if pd.notna(eb["Open"]) else trigger
        initial_sl = max(entry * 0.75, float(setup["SMA20"]))
        if initial_sl >= entry: continue
        signals.append({"symbol": sym, "setup_month": m.index[i].strftime("%Y-%m"),
                        "entry_date": ed, "entry": entry, "initial_sl": initial_sl,
                        "targets": [entry*1.25, entry*1.40, entry*1.50, entry*2.00],
                        "daily": d, "monthly": m})
    return signals

def simulate_one(sig, t1_frac, t3_frac, trail_start, sma_n):
    d, m = sig["daily"], sig["monthly"]
    entry, targets = sig["entry"], sig["targets"]
    remaining, realised = 1.0, 0.0
    t1 = t2 = t3 = t4 = False
    active_sl = sig["initial_sl"]
    exit_date, exit_price, reason = None, None, "OPEN"
    bars = d.loc[d.index >= sig["entry_date"]].head(MAX_HOLD_DAYS)
    for j, (date, bar) in enumerate(bars.iterrows()):
        hi, lo = float(bar["High"]), float(bar["Low"])
        # Trailing stop calculated from the latest fully completed month;
        # it becomes effective on the following daily bar.
        if j > 0:
            activated = (trail_start == "T2" and t2) or (trail_start == "T3" and t3) or (trail_start == "T4" and t4)
            if activated:
                completed = m.loc[m.index < date.to_period("M").to_timestamp()]
                col = f"SMA{sma_n}"
                if len(completed) and pd.notna(completed[col].iloc[-1]):
                    active_sl = max(active_sl, float(completed[col].iloc[-1]))
        # Conservative daily OHLC assumption: existing stop is checked before targets.
        if lo <= active_sl:
            realised += remaining * (active_sl/entry - 1)
            remaining = 0.0
            exit_date, exit_price = date, active_sl
            reason = "INITIAL_SL" if not t1 else "TRAIL_OR_BE_SL"
            break
        if not t1 and hi >= targets[0]:
            sold = min(t1_frac, remaining)
            realised += sold * (targets[0]/entry - 1)
            remaining -= sold
            t1 = True
        if t1 and not t2 and hi >= targets[1]:
            t2 = True
            # Break-even stop is effective from next daily bar.
            active_sl = max(active_sl, entry)
        if t2 and not t3 and hi >= targets[2]:
            sold = min(remaining, remaining * t3_frac)
            realised += sold * (targets[2]/entry - 1)
            remaining -= sold
            t3 = True
        if t3 and not t4 and hi >= targets[3]:
            t4 = True
        if remaining <= 1e-9:
            exit_date, exit_price, reason = date, targets[2], "FULLY_BOOKED"
            break
    if remaining > 1e-9:
        exit_date = bars.index[-1]
        exit_price = float(bars["Close"].iloc[-1])
        realised += remaining * (exit_price/entry - 1)
        reason = "TIME_EXIT"
    return {"symbol": sig["symbol"], "setup_month": sig["setup_month"],
            "entry_date": sig["entry_date"], "exit_date": exit_date,
            "entry": round(entry, 2), "exit_price": round(float(exit_price), 2),
            "initial_sl": round(sig["initial_sl"], 2), "T1": t1, "T2": t2,
            "T3": t3, "T4": t4, "exit_reason": reason,
            "return_pct": realised*100, "t1_booking_pct": t1_frac*100,
            "t3_booking_pct_of_remaining": t3_frac*100,
            "trail_start": trail_start, "trail_sma": sma_n}

def enforce_one_position_per_stock(df):
    kept = []
    for sym, g in df.sort_values(["symbol", "entry_date", "exit_date"]).groupby("symbol"):
        last_exit = None
        for row in g.to_dict("records"):
            if last_exit is None or row["entry_date"] > last_exit:
                kept.append(row)
                last_exit = row["exit_date"]
    return pd.DataFrame(kept)

def portfolio_stats(df):
    if df.empty: return {}
    x = df.copy()
    x["entry_date"] = pd.to_datetime(x["entry_date"])
    x["exit_date"] = pd.to_datetime(x["exit_date"])
    # Release capital before allocating new positions on the same date.
    events = []
    for r in x.itertuples():
        events.append((r.entry_date, 1))
        events.append((r.exit_date, -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda z: (z[0], z[1])):
        active += delta
        peak = max(peak, active)
    pnl = x["return_pct"] * STAKE / 100
    exits = pd.DataFrame({"date": x["exit_date"], "pnl": pnl}).groupby("date", as_index=False)["pnl"].sum().sort_values("date")
    exits["equity"] = exits["pnl"].cumsum()
    exits["peak"] = exits["equity"].cummax().clip(lower=0)
    exits["drawdown"] = exits["equity"] - exits["peak"]
    wins = x[x["return_pct"] > 0]
    losses = x[x["return_pct"] <= 0]
    gross_profit = float(wins["return_pct"].sum())
    gross_loss = abs(float(losses["return_pct"].sum()))
    return {"trades": len(x), "wins": len(wins), "losses": len(losses),
            "win_rate_pct": round(len(wins)/len(x)*100, 2),
            "avg_return_pct": round(float(x["return_pct"].mean()), 2),
            "median_return_pct": round(float(x["return_pct"].median()), 2),
            "profit_factor": round(gross_profit/gross_loss, 3) if gross_loss else None,
            "total_pnl_fixed_10000_per_trade_inr": round(float(pnl.sum()), 2),
            "peak_concurrent_positions": int(peak),
            "minimum_capital_at_10000_each_inr": int(peak*STAKE),
            "realised_equity_max_drawdown_inr": round(float(exits["drawdown"].min()), 2),
            "best_trade_pct": round(float(x["return_pct"].max()), 2),
            "worst_trade_pct": round(float(x["return_pct"].min()), 2)}

def main():
    symbols = get_symbols()
    print("NSE EQ symbols:", len(symbols))
    signals, failed = [], []
    for k in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[k:k+BATCH_SIZE]
        data = download_batch(batch)
        for sym in batch:
            if sym not in data:
                failed.append(sym)
                continue
            signals.extend(generate_signals(sym, data[sym]))
        print(f"Prepared {min(k+BATCH_SIZE,len(symbols))}/{len(symbols)} | signals={len(signals)} | failed={len(failed)}")
        time.sleep(1)
    print("Applying one-open-position-per-stock rule and evaluating exit grid...")
    configs = [(a,b,c,s) for a in T1_BOOK for b in T3_BOOK for c in TRAIL_START for s in TRAIL_SMA]
    summaries, all_rows = [], []
    for n, (a,b,c,s) in enumerate(configs, 1):
        rows = [simulate_one(sig, a, b, c, s) for sig in signals]
        df = enforce_one_position_per_stock(pd.DataFrame(rows))
        stats = portfolio_stats(df)
        stats.update({"t1_booking_pct": a*100, "t3_booking_pct_of_remaining": b*100,
                      "trail_start": c, "trail_sma": s})
        summaries.append(stats)
        if (a,b,c,s) == (0.50,0.50,"T4",10):
            df.to_csv("monthly_ath_exit_optimizer_baseline_trades.csv", index=False)
        if n % 12 == 0: print(f"Completed {n}/{len(configs)} configurations")
    out = pd.DataFrame(summaries).sort_values(
        ["total_pnl_fixed_10000_per_trade_inr", "profit_factor"], ascending=False)
    out.to_csv("monthly_ath_exit_optimizer_results.csv", index=False)
    with open("monthly_ath_exit_optimizer_summary.json","w") as f:
        json.dump({"period": f"{START} to latest available daily bar",
                   "symbols_attempted": len(symbols), "symbols_with_data": len(symbols)-len(failed),
                   "signals_before_one_position_filter": len(signals),
                   "configs_tested": len(configs), "stake_per_trade_inr": STAKE,
                   "universe": "Current NSE EQ constituents; survivorship bias applies",
                   "method": "Each configuration independently replays daily OHLC. One open position per stock; capital released on exit and reusable. Same-bar stop checked before targets. Fixed ₹10,000 per trade, no compounding.",
                   "top_10": out.head(10).to_dict("records"),
                   "baseline": next((z for z in summaries if z.get("t1_booking_pct")==50 and z.get("t3_booking_pct_of_remaining")==50 and z.get("trail_start")=="T4" and z.get("trail_sma")==10), {}),
                   "caveats": ["No brokerage, taxes, slippage or dividends.", "Realised-equity drawdown uses exit-date P&L, not daily mark-to-market drawdown.", "Present-day constituents introduce survivorship bias.", "Trailing SMA uses completed monthly data and is applied from the next daily bar."]}, f, indent=2)
    print(out.head(10).to_string(index=False))
    print("Baseline:", next((z for z in summaries if z.get("t1_booking_pct")==50 and z.get("t3_booking_pct_of_remaining")==50 and z.get("trail_start")=="T4" and z.get("trail_sma")==10), {}))

if __name__ == "__main__":
    main()
