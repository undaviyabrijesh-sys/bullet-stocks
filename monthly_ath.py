import os
import json
from io import StringIO
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = "state.json"
NIFTY_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

# ---------- Filter settings (tune these) ----------
MAX_ALERTS = 15          # send only the top N stocks
MIN_TRADED_VALUE = 5e7   # Rs 5 crore average daily traded value (60 days)
MAX_EXTENSION = 1.15     # close at most 15% above the old high
MIN_VOLUME_RATIO = 1.5   # month volume vs 20-month average
MAX_BASE_RANGE = 1.30    # prior 6 months range (high / low) at most 30%
BATCH = 50


def discord(msg):
    if not WEBHOOK:
        print("No DISCORD_WEBHOOK_URL set")
        print(msg)
        return
    for i in range(0, len(msg), 1900):
        requests.post(WEBHOOK, json={"content": msg[i:i + 1900]}, timeout=30)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def load_symbols():
    try:
        r = requests.get(NIFTY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        syms = [s.strip() + ".NS" for s in df["Symbol"].dropna()]
        print("Loaded", len(syms), "symbols from NSE")
        return syms
    except Exception as e:
        print("NSE list failed:", e)
    if os.path.exists("symbols.txt"):
        with open("symbols.txt") as f:
            syms = [x.strip() for x in f if x.strip()]
        syms = [s if s.endswith(".NS") else s + ".NS" for s in syms]
        print("Loaded", len(syms), "symbols from symbols.txt")
        return syms
    return []


def monthly(df):
    agg = {"High": "max", "Close": "last", "Volume": "sum"}
    try:
        return df.resample("ME").agg(agg)
    except ValueError:
        return df.resample("M").agg(agg)


def quality_check(df, nifty_6m):
    """Return a ranking score if the stock passes all filters, else None."""
    df = df.dropna(subset=["High", "Close", "Volume"])
    if len(df) < 250:
        return None

    m = monthly(df)
    if len(m) < 24:
        return None

    cur = m.iloc[-1]
    old_high = m["High"].iloc[:-1].max()

    # 1. Monthly close above the previous all-time high
    if cur["Close"] <= old_high:
        return None
    # 2. Not over-extended
    if cur["Close"] > old_high * MAX_EXTENSION:
        return None
    # 3. Liquidity
    if (df["Close"] * df["Volume"]).tail(60).mean() < MIN_TRADED_VALUE:
        return None
    # 4. Volume surge
    if cur["Volume"] < MIN_VOLUME_RATIO * m["Volume"].iloc[-21:-1].mean():
        return None
    # 5. Trend: price > 50 DMA > 200 DMA
    close = df["Close"]
    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
    if not (close.iloc[-1] > dma50 > dma200):
        return None
    # 6. Base: prior 6 months within a tight range
    base = m.iloc[-7:-1]
    if (base["High"].max() / base["Close"].min()) > MAX_BASE_RANGE:
        return None
    # 7. Relative strength: beat Nifty over 6 months
    ret6 = close.iloc[-1] / close.iloc[-126] - 1
    if nifty_6m is not None and ret6 <= nifty_6m:
        return None

    return {
        "score": float(ret6),
        "close": float(close.iloc[-1]),
        "old_high": float(old_high),
        "vol_ratio": float(cur["Volume"] / m["Volume"].iloc[-21:-1].mean()),
    }


def get_nifty_6m():
    try:
        n = yf.download("^NSEI", period="2y", interval="1d", progress=False, auto_adjust=False)
        c = n["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        return float(c.iloc[-1] / c.iloc[-126] - 1)
    except Exception as e:
        print("Nifty fetch failed:", e)
        return None


def main():
    symbols = load_symbols()
    if not symbols:
        discord("ATH scan: could not load stock list.")
        return

    month_key = datetime.utcnow().strftime("%Y-%m")
    state = load_state()
    already = set(state.get(month_key, []))
    nifty_6m = get_nifty_6m()

    found = []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            data = yf.download(
                batch, period="max", interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception as e:
            print("batch failed:", e)
            continue

        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                res = quality_check(df, nifty_6m)
            except Exception:
                continue
            if res and sym not in already:
                res["symbol"] = sym.replace(".NS", "")
                found.append(res)

    found.sort(key=lambda x: x["score"], reverse=True)
    top = found[:MAX_ALERTS]

    if top:
        lines = [f"**ATH Monthly Alert ({month_key})**: top {len(top)} of {len(found)} passed filters\n"]
        for r in top:
            lines.append(
                f"**{r['symbol']}**  Close {r['close']:.2f} | Old ATH {r['old_high']:.2f} | "
                f"Vol {r['vol_ratio']:.1f}x | 6M {r['score'] * 100:.0f}%"
            )
        discord("\n".join(lines))
        already.update(r["symbol"] + ".NS" for r in top)
    else:
        print("No new stocks passed the filters.")

    state = {month_key: sorted(already)}
    save_state(state)


if __name__ == "__main__":
    main()
