import json, time, warnings
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
START = "2016-01-01"
END = None
ENTRY_BUFFER = 0.02
STOP_FRACTION = 0.25
MAX_HOLD_DAYS = 90
BATCH_SIZE = 40

def get_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(StringIO(r.text))
    d.columns = [str(c).strip().upper() for c in d.columns]
    d["SERIES"] = d["SERIES"].astype(str).str.strip()
    return sorted(set(d.loc[d["SERIES"] == "EQ", "SYMBOL"].dropna().astype(str).str.strip() + ".NS"))

def download_batch(symbols):
    try:
        raw = yf.download(symbols, start=START, end=END, interval="1d",
                          auto_adjust=False, group_by="ticker", threads=True,
                          progress=False, timeout=30)
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
                d = d.dropna(subset=["High","Low","Close"])
                if len(d) > 250: out[sym] = d
            except Exception:
                pass
        return out
    except Exception as e:
        print("Batch download error:", str(e)[:160])
        return {}


def score_at_entry(d, setup_month_end, entry_date, trigger, nifty):
    # Reconstruct the production scanner score using data available at entry.
    x = d.loc[:entry_date].copy()
    if len(x) < 200:
        return None
    close, vol = x["Close"], x["Volume"]
    price = float(close.iloc[-1])
    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
    if pd.isna(dma50) or pd.isna(dma200) or dma200 <= 0 or price <= dma200 or dma50 <= dma200:
        return None
    ret6 = price / float(close.iloc[-126]) - 1 if len(close) >= 126 else 0.0
    n = nifty.loc[:entry_date].dropna()
    nifty_ret6 = float(n.iloc[-1] / n.iloc[-126] - 1) if len(n) >= 126 else 0.0
    rs = ret6 - nifty_ret6
    recent_vol = vol.tail(5).mean()
    base_vol = vol.iloc[-55:-5].mean()
    vol_ratio = float(recent_vol / base_vol) if base_vol > 0 else 0.0
    m = d.loc[:setup_month_end].resample("ME").agg({"High":"max","Close":"last"})
    pre = m.iloc[:-1].tail(6)
    base_range = float(pre["High"].max() / pre["Close"].min()) if len(pre) >= 3 and pre["Close"].min() > 0 else 1.6
    clip = lambda v: max(0.0, min(1.0, v))
    rs_pts = clip(rs / 0.25)
    vol_pts = clip((vol_ratio - 0.8) / 1.2)
    base_pts = clip((1.6 - base_range) / 0.5)
    trend_pts = clip((dma50 / dma200 - 1) / 0.15)
    static_score = 25 * rs_pts + 20 * vol_pts + 15 * base_pts + 10 * trend_pts
    gap = price / trigger - 1
    if gap < 0 or gap > 0.06:
        return None
    proximity = 1 - gap / 0.06
    return {"score": round(30 * proximity + static_score, 2),
            "rs_pct": round(rs * 100, 2), "volume_ratio": round(vol_ratio, 2)}

def simulate(sym, d, nifty):
    d = d.copy()
    if d.index.tz is not None: d.index = d.index.tz_localize(None)
    m = d.resample("ME").agg({"High":"max","Close":"last"})
    m["SMA20"] = m["Close"].rolling(20).mean()
    m["SMA10"] = m["Close"].rolling(10).mean()
    trades = []
    # A completed month qualifies only when its close is an ATH vs all earlier monthly closes.
    for i in range(48, len(m)-1):
        setup = m.iloc[i]
        prior_closes = m["Close"].iloc[:i].dropna()
        if pd.isna(setup["Close"]) or setup["Close"] < prior_closes.max(): continue
        if pd.isna(setup["SMA20"]): continue
        trigger = float(setup["High"] * (1 + ENTRY_BUFFER))
        next_start = m.index[i] + pd.offsets.MonthBegin(1)
        next_end = m.index[i+1]
        window = d.loc[(d.index >= next_start) & (d.index <= next_end)]
        if window.empty: continue
        # Find first daily bar that reaches the trigger. Gap entries fill at the open.
        crossed = window[window["High"] >= trigger]
        if crossed.empty: continue
        entry_bar = crossed.iloc[0]
        entry_date = crossed.index[0]
        entry = max(trigger, float(entry_bar["Open"])) if pd.notna(entry_bar["Open"]) else trigger
        sl = max(entry * (1-STOP_FRACTION), float(setup["SMA20"]))
        if sl >= entry: continue
        score = score_at_entry(d, m.index[i], entry_date, trigger, nifty)
        if score is None or score["score"] < 80:
            continue
        targets = [entry*1.25, entry*1.40, entry*1.50, entry*2.00]
        remaining, realised = 1.0, 0.0
        t1=t2=t3=t4=False
        active_sl = sl
        exit_date, exit_price, result = None, None, "OPEN"
        # Entry day is included; if SL and a target both trade on one daily bar, assume SL first.
        bars = d.loc[d.index >= entry_date].head(MAX_HOLD_DAYS)
        for j, (date, bar) in enumerate(bars.iterrows()):
            hi, lo = float(bar["High"]), float(bar["Low"])
            if j > 0 and t4:
                # Trail using the latest fully completed month's SMA10 only.
                completed = m.loc[m.index < date.to_period("M").to_timestamp()]
                if len(completed) and pd.notna(completed["SMA10"].iloc[-1]):
                    active_sl = max(active_sl, float(completed["SMA10"].iloc[-1]))
            if lo <= active_sl:
                realised += remaining * (active_sl/entry - 1)
                remaining = 0
                exit_date, exit_price, result = date, active_sl, "SL" if not t1 else "TRAIL/SL"
                break
            # On entry day, don't award targets if the bar opened below trigger then traded down first;
            # daily OHLC cannot reveal path, so conservative stop check above takes priority.
            if not t1 and hi >= targets[0]:
                realised += 0.50 * (targets[0]/entry - 1)
                remaining -= 0.50; t1=True
            if t1 and not t2 and hi >= targets[1]:
                t2=True
                active_sl = entry  # effective from the next bar; conservative daily-bar simulation
            if t2 and not t3 and hi >= targets[2]:
                sold = remaining * 0.50
                realised += sold * (targets[2]/entry - 1)
                remaining -= sold; t3=True
            if t3 and not t4 and hi >= targets[3]:
                t4=True
            if remaining <= 1e-9:
                exit_date, exit_price, result = date, targets[2], "COMPLETED"
                break
        if remaining > 0:
            last_date = bars.index[-1]
            last_close = float(bars["Close"].iloc[-1])
            realised += remaining * (last_close/entry - 1)
            exit_date, exit_price, result = last_date, last_close, "TIME/END"
        trades.append({
            "symbol":sym, "setup_month":m.index[i].strftime("%Y-%m"),
            "entry_date":entry_date.strftime("%Y-%m-%d"), "entry":round(entry,2),
            "initial_sl":round(sl,2), "T1":t1, "T2":t2, "T3":t3, "T4":t4,
            "exit_date":exit_date.strftime("%Y-%m-%d") if exit_date is not None else "",
            "exit_price":round(float(exit_price),2) if exit_price is not None else None,
            "exit_reason":result, "return_pct":round(realised*100,2), "score":score["score"], "rs_pct":score["rs_pct"], "volume_ratio":score["volume_ratio"]
        })
    return trades

def main():
    symbols = get_symbols()
    nd = yf.download("^NSEI", start=START, end=END, interval="1d", auto_adjust=False, progress=False)
    nifty = nd["Close"]
    if isinstance(nifty, pd.DataFrame): nifty = nifty.iloc[:, 0]
    nifty = nifty.dropna()
    if nifty.index.tz is not None: nifty.index = nifty.index.tz_localize(None)
    print("NSE EQ symbols:", len(symbols), "| Nifty bars:", len(nifty))
    all_trades, failed = [], []
    for k in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[k:k+BATCH_SIZE]
        data = download_batch(batch)
        for sym in batch:
            if sym not in data:
                failed.append(sym); continue
            all_trades.extend(simulate(sym, data[sym], nifty))
        print(f"Processed {min(k+BATCH_SIZE,len(symbols))}/{len(symbols)} | trades={len(all_trades)} | failed={len(failed)}")
        time.sleep(1)
    df = pd.DataFrame(all_trades)
    if df.empty:
        print("NO TRADES GENERATED")
        df.to_csv("monthly_ath_score80_trades.csv", index=False)
        return
    df.to_csv("monthly_ath_backtest_trades.csv", index=False)
    wins = int((df.return_pct > 0).sum())
    summary = {
        "period": f"{START} to latest available daily bar",
        "universe": "Current NSE EQ constituents (survivorship bias applies)",
        "symbols_attempted": len(symbols), "symbols_with_data": len(symbols)-len(failed),
        "score_threshold": 80, "trades": len(df), "wins": wins, "losses": int((df.return_pct <= 0).sum()),
        "win_rate_pct": round(wins/len(df)*100,2),
        "average_return_pct": round(float(df.return_pct.mean()),2),
        "median_return_pct": round(float(df.return_pct.median()),2),
        "total_compounded_return_pct": round(float(((1+df.return_pct/100).prod()-1)*100),2),
        "best_trade_pct": round(float(df.return_pct.max()),2),
        "worst_trade_pct": round(float(df.return_pct.min()),2),
        "failed_symbols": len(failed),
        "assumptions": [
            "Entry at 2% above previous completed monthly ATH high; gap fills at open.",
            "Initial SL is the higher of entry minus 25% or completed-month SMA20.",
            "T1 +25% sell 50%; T2 +40% move SL to entry; T3 +50% sell half remaining; T4 +100% trail remainder on completed-month SMA10.",
            "Daily OHLC simulation: if stop and target both touch on the same bar, stop is assumed first.",
            "90 trading-day maximum hold; remaining shares marked to final close.",
            "No brokerage, taxes, slippage or dividends included.",
            "Current NSE constituents only; delisted stocks and historical membership are not represented."
        ]
    }
    # One position per stock; calculate peak simultaneous trades and required capital.
    events = []
    for _, row in df.iterrows():
        events.append((pd.Timestamp(row["entry_date"]), 1))
        events.append((pd.Timestamp(row["exit_date"]), -1))
    events.sort(key=lambda item: (item[0], item[1]))  # exits before entries
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    summary["max_concurrent_trades"] = int(peak)
    summary["minimum_capital_at_10000_per_position"] = int(peak * 10000)
    summary["net_pnl_at_10000_per_position"] = round(float((df["return_pct"] * 100).sum()), 2)
    with open("monthly_ath_score80_summary.json","w") as f: json.dump(summary,f,indent=2)
    print("\nBACKTEST SUMMARY\n"+json.dumps(summary,indent=2))
if __name__ == "__main__": main()
