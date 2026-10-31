import os
import json
import time
from io import StringIO
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = "state.json"
NSE_ALL_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# ---------- Filter settings (tune these) ----------
MAX_ALERTS = 15          # send only the top N stocks
MIN_PRICE = 20           # skip penny stocks (Rs)
MIN_TRADED_VALUE = 5e7   # Rs 5 crore average daily traded value (60 days)
MAX_EXTENSION = 1.15     # close at most 15% above the old high
MIN_VOLUME_RATIO = 1.5   # month volume vs 20-month average
MAX_BASE_RANGE = 1.30    # prior 6 months range (high / low) at most 30%
BATCH = 50
ALLOWED_SERIES = {"EQ"}  # add "BE" if you also want trade-to-trade stocks

EMBED_COLOR = 0x2ECC71   # green


# ---------- Discord ----------
def discord_post(payload):
    if not WEBHOOK:
        print("No DISCORD_WEBHOOK_URL set")
        print(json.dumps(payload, indent=1, ensure_ascii=False))
        return
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=30)
        if r.status_code >= 300:
            print("Discord error:", r.status_code, r.text[:300])
    except Exception as e:
        print("Discord failed:", e)


def discord(msg):
    for i in range(0, len(msg), 1900):
        discord_post({"content": msg[i:i + 1900]})


def stock_entry(rank, r):
    link = f"https://www.tradingview.com/chart/?symbol={quote('NSE:' + r['symbol'], safe=':')}"
    return (
        f"**{rank}. [{r['symbol']}]({link})** · ₹{r['close']:,.2f}\n"
        f"Old ATH ₹{r['old_high']:,.2f} → **+{r['above'] * 100:.1f}%** above · "
        f"Volume **{r['vol_ratio']:.1f}x** · 6M **{r['score'] * 100:+.0f}%**\n"
    )


def send_alert(top, found_count, scanned, month_label):
    entries = [stock_entry(i, r) for i, r in enumerate(top, 1)]

    # Split into several embeds if the text gets too long (Discord limit is 4096)
    descriptions, current = [], ""
    for e in entries:
        if len(current) + len(e) + 1 > 3800:
            descriptions.append(current)
            current = ""
        current += e + "\n"
    if current:
        descriptions.append(current)

    embeds = []
    for idx, desc in enumerate(descriptions):
        embed = {"description": desc.strip(), "color": EMBED_COLOR}
        if idx == 0:
            embed["title"] = f"📈 ATH Monthly Alert · {month_label}"
            embed["description"] = (
                f"Top **{len(top)}** of **{found_count}** stocks that passed your filters\n\n"
                + desc.strip()
            )
        if idx == len(descriptions) - 1:
            embed["footer"] = {
                "text": f"Scanned {scanned:,} NSE stocks · Closed above old ATH with volume"
            }
            embed["timestamp"] = datetime.utcnow().isoformat()
        embeds.append(embed)

    for i in range(0, len(embeds), 10):  # Discord allows 10 embeds per message
        discord_post({"embeds": embeds[i:i + 10]})


# ---------- State ----------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


# ---------- Symbols ----------
def load_symbols():
    # 1. Full NSE equity list
    try:
        r = requests.get(NSE_ALL_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]
        df["SERIES"] = df["SERIES"].astype(str).str.strip()
        df = df[df["SERIES"].isin(ALLOWED_SERIES)]
        syms = [s.strip() + ".NS" for s in df["SYMBOL"].dropna()]
        print("Loaded", len(syms), "symbols from NSE")
        if len(syms) > 500:
            return syms
    except Exception as e:
        print("NSE list failed:", e)
    # 2. Fallback: symbols.txt in the repo root (one symbol per line)
    if os.path.exists("symbols.txt"):
        with open("symbols.txt") as f:
            syms = [x.strip() for x in f if x.strip()]
        syms = [s if s.endswith(".NS") else s + ".NS" for s in syms]
        print("Loaded", len(syms), "symbols from symbols.txt")
        return syms
    return []


# ---------- Filters ----------
def monthly(df):
    agg = {"High": "max", "Close": "last", "Volume": "sum"}
    try:
        return df.resample("ME").agg(agg)
    except ValueError:
        return df.resample("M").agg(agg)


def quality_check(df, nifty_6m):
    """Return a dict if the stock passes all filters, else None."""
    df = df.dropna(subset=["High", "Close", "Volume"])
    if len(df) < 250:
        return None

    close = df["Close"]
    if close.iloc[-1] < MIN_PRICE:
        return None

    # Skip stocks that have not traded recently (suspended / delisted)
    if (pd.Timestamp.now().normalize() - df.index[-1]).days > 10:
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
    if (close * df["Volume"]).tail(60).mean() < MIN_TRADED_VALUE:
        return None
    # 4. Volume surge
    avg_vol = m["Volume"].iloc[-21:-1].mean()
    if avg_vol <= 0 or cur["Volume"] < MIN_VOLUME_RATIO * avg_vol:
        return None
    # 5. Trend: price > 50 DMA > 200 DMA
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
        "above": float(cur["Close"] / old_high - 1),
        "vol_ratio": float(cur["Volume"] / avg_vol),
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


def download_batch(batch):
    for attempt in range(2):
        try:
            return yf.download(
                batch, period="max", interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception as e:
            print("batch failed, retrying:", e)
            time.sleep(5)
    return None


# ---------- Main ----------
def main():
    symbols = load_symbols()
    if not symbols:
        discord("⚠️ ATH scan: could not load the stock list.")
        return

    now = datetime.utcnow()
    month_key = now.strftime("%Y-%m")
    month_label = now.strftime("%B %Y")
    state = load_state()
    already = set(state.get(month_key, []))
    nifty_6m = get_nifty_6m()

    found = []
    scanned = 0
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        data = download_batch(batch)
        if data is None or data.empty:
            continue

        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                res = quality_check(df, nifty_6m)
                scanned += 1
            except Exception:
                continue
            if res and sym not in already:
                res["symbol"] = sym.replace(".NS", "")
                found.append(res)
        print(f"Progress: {min(i + BATCH, len(symbols))}/{len(symbols)}")

    print("Scanned", scanned, "| passed", len(found))

    found.sort(key=lambda x: x["score"], reverse=True)
    top = found[:MAX_ALERTS]

    if top:
        send_alert(top, len(found), scanned, month_label)
        already.update(r["symbol"] + ".NS" for r in top)
    else:
        print("No new stocks passed the filters.")

    state = {month_key: sorted(already)}
    save_state(state)


if __name__ == "__main__":
    main()
