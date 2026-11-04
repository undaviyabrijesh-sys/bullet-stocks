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

def simulate(sym, d):
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
            "exit_reason":result, "return_pct":round(realised*100,2)
        })
    return trades

def main():
    symbols = get_symbols()
    print("NSE EQ symbols:", len(symbols))
    all_trades, failed = [], []
    for k in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[k:k+BATCH_SIZE]
        data = download_batch(batch)
        for sym in batch:
            if sym not in data:
                failed.append(sym); continue
            all_trades.extend(simulate(sym, data[sym]))
        print(f"Processed {min(k+BATCH_SIZE,len(symbols))}/{len(symbols)} | trades={len(all_trades)} | failed={len(failed)}")
        time.sleep(1)
    df = pd.DataFrame(all_trades)
    if df.empty:
        print("NO TRADES GENERATED")
        df.to_csv("monthly_ath_backtest_trades.csv", index=False)
        return
    df.to_csv("monthly_ath_backtest_trades.csv", index=False)
    wins = int((df.return_pct > 0).sum())
    summary = {
        "period": f"{START} to latest available daily bar",
        "universe": "Current NSE EQ constituents (survivorship bias applies)",
        "symbols_attempted": len(symbols), "symbols_with_data": len(symbols)-len(failed),
        "trades": len(df), "wins": wins, "losses": int((df.return_pct <= 0).sum()),
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
    with open("monthly_ath_backtest_summary.json","w") as f: json.dump(summary,f,indent=2)
    print("\nBACKTEST SUMMARY\n"+json.dumps(summary,indent=2))
if __name__ == "__main__": main()
