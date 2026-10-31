import io, json, os, time, datetime as dt
import pandas as pd, requests, yfinance as yf

WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]
BUFFER = 0.02                       # entry = monthly high + 2%
TARGETS = {"T1": 1.25, "T2": 1.40, "T3": 1.50, "T4": 2.00}
SL_PCT = 0.25                       # stop = max(entry*0.75, monthly 20 SMA)
MIN_MONTHS = 48                     # 4 years of history
STATE = "state.json"


def discord(msg):
    for i in range(0, len(msg), 1900):
        requests.post(WEBHOOK, json={"content": msg[i:i + 1900]}, timeout=30)


def send_lines(title, lines):
    msg = title
    for ln in lines:
        if len(msg) + len(ln) + 1 > 1900:
            discord(msg)
            msg = ""
        msg += "\n" + ln
    if msg.strip():
        discord(msg)


def load():
    s = json.load(open(STATE))
    for k, v in (("last_scan", ""), ("last_daily", ""), ("watch", {}), ("pos", {})):
        s.setdefault(k, v)
    return s


def clean_monthly(df):
    df = df.dropna(subset=["Close"])
    if df.empty:
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    today = dt.date.today()
    last = df.index[-1]
    if (last.year, last.month) == (today.year, today.month):   # drop running month
        df = df.iloc[:-1]
    return df


def monthly(sym):
    df = yf.Ticker(sym + ".NS").history(period="max", interval="1mo", auto_adjust=False)
    return clean_monthly(df)


def get_universe():
    try:
        r = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        syms = df[df["SERIES"].str.strip() == "EQ"]["SYMBOL"].str.strip().tolist()
        if len(syms) > 1000:
            return syms
    except Exception as e:
        print("NSE list failed:", e)
    return [x.strip().upper() for x in open("universe.txt") if x.strip()]  # fallback


def monthly_bulk(symbols, chunk_size=100):
    for i in range(0, len(symbols), chunk_size):
        chunk = [s + ".NS" for s in symbols[i:i + chunk_size]]
        try:
            data = yf.download(chunk, period="max", interval="1mo",
                               group_by="ticker", auto_adjust=False,
                               threads=True, progress=False)
        except Exception as e:
            print("chunk failed", i, e)
            continue
        for t in chunk:
            try:
                df = clean_monthly(data[t])
                if not df.empty:
                    yield t[:-3], df
            except Exception:
                continue
        time.sleep(2)


def scan(state, symbols, ym):
    found = []
    start = (pd.Timestamp(ym + "-01") + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    expires = (pd.Timestamp(ym + "-01") + pd.DateOffset(months=2)).strftime("%Y-%m")
    for sym, df in monthly_bulk(symbols):
        try:
            if len(df) < MIN_MONTHS:                         # rule 2: 4 yrs old
                continue
            if df["Close"].iloc[-1] < df["Close"].max():     # rule 1: monthly ATH close
                continue
            high = float(df["High"].iloc[-1])                # rule 3
            entry = round(high * (1 + BUFFER), 2)
            sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
            sl = round(max(entry * (1 - SL_PCT), sma20), 2)  # rule 5
            state["watch"][sym] = {"high": high, "entry": entry, "sl": sl,
                                   "start": start, "expires": expires}
            found.append((sym, high, entry, sl))
        except Exception as e:
            print(sym, e)
    state["last_scan"] = ym
    if found:
        lines = [f"**{s}** | high {h:.2f} | BUY above **{e:.2f}** | SL {l:.2f}"
                 for s, h, e, l in sorted(found)]
        send_lines(f"📅 **Monthly ATH close list ({ym})** - {len(found)} stocks", lines)
    else:
        discord(f"📅 Monthly scan {ym}: no stock closed at ATH.")


def daily(state):
    since = state["last_daily"] or "1900-01-01"
    ym_now = dt.date.today().strftime("%Y-%m")
    for s in [k for k, v in state["watch"].items() if v["expires"] <= ym_now]:
        del state["watch"][s]

    for sym in set(state["watch"]) | set(state["pos"]):
        try:
            d = yf.Ticker(sym + ".NS").history(period="15d", interval="1d", auto_adjust=False)
            d.index = d.index.tz_localize(None) if d.index.tz else d.index
            d = d[d.index > pd.Timestamp(since)]
        except Exception as e:
            print(sym, e)
            continue
        for day, bar in d.iterrows():
            ds = day.strftime("%Y-%m-%d")
            # ENTRY (rule 3)
            if sym in state["watch"] and sym not in state["pos"]:
                w = state["watch"][sym]
                if ds >= w["start"] and bar["High"] >= w["entry"]:
                    e = w["entry"]
                    state["pos"][sym] = {"entry": e, "sl": w["sl"], "hit": [], "trail": False}
                    del state["watch"][sym]
                    tg = " | ".join(f"{k} {e * v:.2f}" for k, v in TARGETS.items())
                    discord(f"🟢 **ENTRY {sym}** @ {e:.2f} ({ds})\nSL {w['sl']:.2f}\n{tg}")
            # POSITION MANAGEMENT (rules 4 and 5)
            if sym in state["pos"]:
                p = state["pos"][sym]
                if not p["trail"] and bar["Low"] <= p["sl"]:
                    discord(f"🔴 **STOP LOSS {sym}** hit {p['sl']:.2f} ({ds})")
                    del state["pos"][sym]
                    continue
                for k, m in TARGETS.items():
                    if k not in p["hit"] and bar["High"] >= p["entry"] * m:
                        p["hit"].append(k)
                        discord(f"🎯 **{k} {sym}** hit {p['entry'] * m:.2f} ({ds})")
                        if k == "T4":
                            p["trail"] = True
                            discord(f"🔁 {sym}: now trailing with monthly 10 SMA")
                if p["trail"]:
                    sma10 = float(monthly(sym)["Close"].rolling(10).mean().iloc[-1])
                    if bar["Close"] < sma10:
                        discord(f"🏁 **EXIT {sym}**: close {bar['Close']:.2f} < monthly 10 SMA {sma10:.2f} ({ds})")
                        del state["pos"][sym]
    state["last_daily"] = dt.date.today().strftime("%Y-%m-%d")


if __name__ == "__main__":
    state = load()
    today = dt.date.today()
    ym = (today.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")  # last completed month
    if state["last_scan"] != ym:
        scan(state, get_universe(), ym)
    daily(state)
    json.dump(state, open(STATE, "w"), indent=1)
