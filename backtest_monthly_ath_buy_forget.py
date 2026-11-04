import json, time, warnings
from io import StringIO
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
START = "2016-01-01"
BATCH_SIZE = 40
ENTRY_BUFFER = 0.02
STAKE = 10_000
HORIZONS_MONTHS = [6, 12, 24, 36, 60]

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
                    if sym in raw.columns.get_level_values(0): d = raw[sym].copy()
                    elif sym in raw.columns.get_level_values(1): d = raw.xs(sym, axis=1, level=1).copy()
                    else: continue
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

def make_signals(sym, d):
    d = d.copy()
    if d.index.tz is not None: d.index = d.index.tz_localize(None)
    m = d.resample("ME").agg({"High":"max", "Close":"last"})
    m["SMA20"] = m["Close"].rolling(20).mean()
    sigs = []
    for i in range(48, len(m)-1):
        setup = m.iloc[i]
        prior = m["Close"].iloc[:i].dropna()
        if pd.isna(setup["Close"]) or setup["Close"] < prior.max() or pd.isna(setup["SMA20"]): continue
        trigger = float(setup["High"] * (1+ENTRY_BUFFER))
        start = m.index[i] + pd.offsets.MonthBegin(1)
        end = m.index[i+1]
        window = d.loc[(d.index >= start) & (d.index <= end)]
        crossed = window[window["High"] >= trigger]
        if crossed.empty: continue
        bar = crossed.iloc[0]
        entry_date = crossed.index[0]
        entry = max(trigger, float(bar["Open"])) if pd.notna(bar["Open"]) else trigger
        sigs.append({"symbol":sym, "setup_month":m.index[i].strftime("%Y-%m"),
                     "entry_date":entry_date, "entry":entry, "daily":d})
    return sigs

def simulate(sig, months):
    d = sig["daily"]
    desired_exit = sig["entry_date"] + pd.DateOffset(months=months)
    future = d.loc[d.index >= desired_exit]
    if len(future):
        exit_date = future.index[0]
        exit_price = float(future.iloc[0]["Close"])
        exit_reason = f"HOLD_{months}M"
    else:
        exit_date = d.index[-1]
        exit_price = float(d["Close"].iloc[-1])
        exit_reason = "DATA_END"
    ret = (exit_price / sig["entry"] - 1) * 100
    return {"symbol":sig["symbol"], "setup_month":sig["setup_month"],
            "entry_date":sig["entry_date"], "exit_date":exit_date,
            "entry":round(sig["entry"],2), "exit_price":round(exit_price,2),
            "return_pct":ret, "holding_months_target":months,
            "exit_reason":exit_reason}

def filter_one_position(df):
    kept=[]
    for sym,g in df.sort_values(["symbol","entry_date","exit_date"]).groupby("symbol"):
        last_exit=None
        for row in g.to_dict("records"):
            if last_exit is None or row["entry_date"] > last_exit:
                kept.append(row); last_exit=row["exit_date"]
    return pd.DataFrame(kept)

def stats(df, months):
    if df.empty: return {"holding_months":months,"trades":0}
    x=df.copy()
    x["entry_date"]=pd.to_datetime(x["entry_date"]); x["exit_date"]=pd.to_datetime(x["exit_date"])
    events=[]
    for r in x.itertuples():
        events += [(r.entry_date,1),(r.exit_date,-1)]
    active=peak=0
    for _,delta in sorted(events,key=lambda z:(z[0],z[1])):
        active+=delta; peak=max(peak,active)
    pnl=x["return_pct"]*STAKE/100
    ex=pd.DataFrame({"date":x["exit_date"],"pnl":pnl}).groupby("date",as_index=False)["pnl"].sum().sort_values("date")
    ex["equity"]=ex["pnl"].cumsum(); ex["peak"]=ex["equity"].cummax().clip(lower=0)
    ex["dd"]=ex["equity"]-ex["peak"]
    wins=x[x.return_pct>0]; losses=x[x.return_pct<=0]
    gp=float(wins.return_pct.sum()); gl=abs(float(losses.return_pct.sum()))
    hold=(x["exit_date"]-x["entry_date"]).dt.days
    return {"holding_months":months,"trades":len(x),"wins":len(wins),"losses":len(losses),
            "win_rate_pct":round(len(wins)/len(x)*100,2),
            "avg_return_pct":round(float(x.return_pct.mean()),2),
            "median_return_pct":round(float(x.return_pct.median()),2),
            "profit_factor":round(gp/gl,3) if gl else None,
            "total_profit_at_10000_per_trade_inr":round(float(pnl.sum()),2),
            "peak_positions":int(peak),"minimum_capital_inr":int(peak*STAKE),
            "avg_holding_days":round(float(hold.mean()),1),
            "capital_productivity_inr_per_1lakh_capital_day":round(float(pnl.sum())/float((hold.clip(lower=1)*STAKE).sum())*100000,2),
            "realised_equity_drawdown_inr":round(float(ex.dd.min()),2),
            "best_trade_pct":round(float(x.return_pct.max()),2),
            "worst_trade_pct":round(float(x.return_pct.min()),2)}

def main():
    symbols=get_symbols(); all_sigs=[]; failed=[]
    print("NSE EQ symbols:",len(symbols))
    for k in range(0,len(symbols),BATCH_SIZE):
        batch=symbols[k:k+BATCH_SIZE]; data=download_batch(batch)
        for sym in batch:
            if sym not in data: failed.append(sym); continue
            all_sigs.extend(make_signals(sym,data[sym]))
        print(f"Prepared {min(k+BATCH_SIZE,len(symbols))}/{len(symbols)} | signals={len(all_sigs)} | failed={len(failed)}")
        time.sleep(1)
    results=[]; summaries=[]
    for months in HORIZONS_MONTHS:
        df=filter_one_position(pd.DataFrame([simulate(s,months) for s in all_sigs]))
        df.to_csv(f"monthly_ath_buy_forget_{months}m_trades.csv",index=False)
        summaries.append(stats(df,months))
        results.append(df)
    out=pd.DataFrame(summaries).sort_values(
        ["capital_productivity_inr_per_1lakh_capital_day","total_profit_at_10000_per_trade_inr"],
        ascending=False)
    out.to_csv("monthly_ath_buy_forget_summary.csv",index=False)
    with open("monthly_ath_buy_forget_summary.json","w") as f:
        json.dump({"period":f"{START} to latest available daily bar",
                   "universe":"Current NSE EQ constituents; survivorship bias applies",
                   "signals_before_position_filter":len(all_sigs),"symbols_attempted":len(symbols),
                   "symbols_with_data":len(symbols)-len(failed),"stake_per_trade_inr":STAKE,
                   "method":"Same monthly ATH trigger and entry as core strategy; no stop loss, no profit targets; hold fixed 6,12,24,36,60 calendar months; one open position per stock; reuse capital after exit; positions reaching data end are marked to latest close.",
                   "results":out.to_dict("records"),
                   "caveats":["No brokerage, taxes, slippage or dividends.","Current constituents only; survivorship bias applies.","Open positions at data end are marked to latest close and are not necessarily realised exits.","Realised-equity drawdown is based on exit-date P&L, not daily mark-to-market drawdown."]},f,indent=2)
    print(out.to_string(index=False))

if __name__=="__main__": main()
