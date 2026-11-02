"""
Monthly-ATH breakout scanner (NSE) with Discord alerts.

Modes (automatic):
  * First run of a trading day -> FULL scan (all NSE EQ stocks), saves watchlist.json
  * Every later run that day   -> LIVE refresh (watchlist + open positions only, fast)

Flags:
  --full   force a full scan now
  --force  ignore market-hours / weekday / holiday guard (manual testing)
"""
import os
import sys
import json
import time
from io import StringIO
from datetime import datetime, timezone, time as dtime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = "state.json"
WATCH_FILE = "watchlist.json"
NSE_ALL_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
IST = ZoneInfo("Asia/Kolkata")

FORCE = "--force" in sys.argv
FULL = "--full" in sys.argv

# ---------- Setup rules ----------
MIN_AGE_MONTHS = 48
SETUP_LOOKBACK = 3
ENTRY_BUFFER = 0.02       # trigger = monthly high + 2%
MAX_CHASE = 0.06          # triggered: skip if price is > 6% above the trigger
POTENTIAL_BAND = 0.05     # potential: price within 5% below the trigger
SL_BELOW_LEVEL = 0.02     # SL = monthly high - 2%
RR = 2.0
ENTRY_AT_CMP = True       # True = entry is the live price when triggered (honest P&L);
                          # False = entry is the trigger price (old behaviour)

# Watchlist is wider than the alert bands so live runs can catch moves during the day
WATCH_UP = 0.10
WATCH_DOWN = 0.10

# ---------- Quality filters ----------
MIN_PRICE = 50
MIN_TRADED_VALUE = 5e7
MAX_ZERO_VOL_DAYS = 2
MIN_SCORE = 35

# ---------- Position tracking ----------
MAX_HOLD_DAYS = 90
KEEP_CLOSED = 200
MAX_OPEN_SHOWN = 15

# ---------- Output ----------
MAX_TRIGGERED = 4
MAX_POTENTIAL = 4
DEDUPE_POTENTIAL_SAME_DAY = False   # False = potential names repeat every 30-min run
BATCH = 50
ALLOWED_SERIES = {"EQ"}

GREEN = 0x2ECC71
AMBER = 0xF1C40F
RED = 0xE74C3C
BLUE = 0x3498DB
GREY = 0x95A5A6


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


def fmt(x):
    return f"₹{x:,.2f}"


def tv_link(symbol):
    return f"https://www.tradingview.com/chart/?symbol={quote('NSE:' + symbol, safe=':')}"


def stock_entry(rank, r):
    link = tv_link(r["symbol"])
    if r["status"] == "TRIGGERED":
        gap = f"🔺 **{abs(r['gap']) * 100:.1f}%** above trigger"
    else:
        gap = f"⏳ **{abs(r['gap']) * 100:.1f}%** to go for trigger"
    return (
        f"**{rank}. [{r['symbol']}]({link})** · Score **{r['score']:.0f}**/100\n"
        f"💰 CMP **{fmt(r['price'])}** · {gap}\n"
        f"📌 Monthly ATH high {fmt(r['level'])} ({r['setup_month']}) → Trigger **{fmt(r['trigger'])}**\n"
        f"🎯 Entry {fmt(r['entry'])} · 🛑 SL {fmt(r['sl'])} · 🏁 Target {fmt(r['target'])} (1:{RR:g})\n"
        f"📊 Vol **{r['vol_ratio']:.1f}x** · RS **{r['rs'] * 100:+.0f}%** vs Nifty · "
        f"Base range {r['base_range'] * 100:.0f}%\n"
    )


def build_embed(title, color, items, empty_text):
    if items:
        desc = "\n".join(stock_entry(i, r) for i, r in enumerate(items, 1))
    else:
        desc = empty_text
    return {"title": title, "description": desc[:4000], "color": color}


def send_alert(trig, pot, scanned, now, n_trig_all, n_pot_all, mode):
    header = f"📊 **ATH Monthly Alert** · {now:%d %b %Y} · {now:%I:%M %p} IST · {mode}"
    e1 = build_embed(
        f"🟢 ENTRY TRIGGERED ({len(trig)} of {n_trig_all})",
        GREEN, trig,
        "No fresh breakout above trigger right now.",
    )
    e2 = build_embed(
        f"🟡 POTENTIAL ENTRY ({len(pot)} of {n_pot_all})",
        AMBER, pot,
        "No stock is close to its trigger right now.",
    )
    e2["footer"] = {
        "text": (
            f"Watchlist built from {scanned:,} NSE stocks · Monthly ATH close + 2% trigger · "
            f"Liquidity ≥ ₹{MIN_TRADED_VALUE / 1e7:.0f} Cr · For study, not advice"
        )
    }
    e2["timestamp"] = datetime.now(timezone.utc).isoformat()
    discord_post({"content": header, "embeds": [e1, e2]})


# ---------- Position alerts ----------
RESULT_STYLE = {
    "TARGET": ("🎯", "TARGET HIT", GREEN),
    "SL": ("🛑", "STOPLOSS HIT", RED),
    "EXPIRED": ("⏰", f"TIME EXIT ({MAX_HOLD_DAYS}d)", GREY),
}


def closed_embed(c):
    icon, label, color = RESULT_STYLE[c["result"]]
    opened = pd.Timestamp(c["opened"]).strftime("%d %b")
    closed = pd.Timestamp(c["closed"]).strftime("%d %b")
    desc = (
        f"**[{c['symbol']}]({tv_link(c['symbol'])})** · {fmt(c['entry'])} → **{fmt(c['exit'])}**\n"
        f"💵 P&L **{c['pnl'] * 100:+.1f}%** · **{c['r']:+.1f}R**\n"
        f"🛑 SL {fmt(c['sl'])} · 🏁 Target {fmt(c['target'])}\n"
        f"🗓️ Opened {opened} → Closed {closed} · {c['days']} day(s)"
    )
    if c.get("note"):
        desc += f"\n⚠️ {c['note']}"
    return {"title": f"{icon} {label} · {c['symbol']}", "description": desc, "color": color}


def open_embed(positions, prices, closed_hist, today):
    items = sorted(positions.items(), key=lambda kv: kv[1]["opened"])
    lines = []
    for sym, pos in items[:MAX_OPEN_SHOWN]:
        last = prices.get(sym)
        days = max((today - pd.Timestamp(pos["opened"])).days, 0)
        if last:
            pnl = last / pos["entry"] - 1
            dot = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"{dot} **[{sym}]({tv_link(sym)})** · CMP {fmt(last)} · "
                f"P&L **{pnl * 100:+.1f}%** · {days}d\n"
                f"↳ Entry {fmt(pos['entry'])} · 🛑 {fmt(pos['sl'])} ({(pos['sl'] / last - 1) * 100:+.1f}%) · "
                f"🏁 {fmt(pos['target'])} ({(pos['target'] / last - 1) * 100:+.1f}%)"
            )
        else:
            lines.append(f"⚪ **{sym}** · price unavailable · {days}d")
    if len(items) > MAX_OPEN_SHOWN:
        lines.append(f"…and {len(items) - MAX_OPEN_SHOWN} more")
    wins = sum(1 for c in closed_hist if c["pnl"] > 0)
    losses = len(closed_hist) - wins
    if closed_hist:
        stats = f"Closed so far: {wins}W · {losses}L · Win rate {wins / len(closed_hist) * 100:.0f}%"
    else:
        stats = "No closed trades yet"
    return {
        "title": f"📂 ACTIVE ENTRIES ({len(items)})",
        "description": "\n".join(lines)[:4000],
        "color": BLUE,
        "footer": {"text": stats + " · % in brackets = distance from CMP"},
    }


# ---------- State / watchlist ----------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ---------- Market guard ----------
def market_hours_now(now):
    if now.weekday() >= 5:
        return False
    return dtime(9, 0) <= now.time() <= dtime(15, 45)


def nifty_traded_today(today):
    """False on exchange holidays (latest Nifty daily bar is not today)."""
    try:
        n = yf.download("^NSEI", period="5d", interval="1d", progress=False, auto_adjust=False)
        if n is None or n.empty:
            return True
        last = pd.Timestamp(n.index[-1])
        if last.tzinfo is not None:
            last = last.tz_localize(None)
        return last.normalize() == today
    except Exception as e:
        print("Holiday check failed (continuing):", e)
        return True


# ---------- Symbols ----------
def load_symbols():
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
    if os.path.exists("symbols.txt"):
        with open("symbols.txt") as f:
            syms = [x.strip() for x in f if x.strip()]
        syms = [s if s.endswith(".NS") else s + ".NS" for s in syms]
        print("Loaded", len(syms), "symbols from symbols.txt")
        return syms
    return []


# ---------- Analysis ----------
def monthly(df):
    agg = {"High": "max", "Close": "last", "Volume": "sum"}
    try:
        return df.resample("ME").agg(agg)
    except ValueError:
        return df.resample("M").agg(agg)


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def pick(data, sym):
    """Pull one ticker's OHLCV frame out of a yfinance download."""
    if isinstance(data.columns, pd.MultiIndex):
        if sym in data.columns.get_level_values(0):
            return data[sym]
        if sym in data.columns.get_level_values(1):
            return data.xs(sym, axis=1, level=1)
        raise KeyError(sym)
    return data


def analyse(df, nifty_6m, today):
    """Full-scan step: return a watchlist item if the stock has a valid setup, else None.
    Static fields (level, trigger, sl, static score) are stored; live runs add price-based parts."""
    if getattr(df.index, "tz", None) is not None:
        df = df.tz_localize(None)
    df = df.dropna(subset=["High", "Close", "Volume"])
    if len(df) < 250:
        return None
    close = df["Close"]
    vol = df["Volume"]
    price = float(close.iloc[-1])

    if price < MIN_PRICE:
        return None
    if (today - df.index[-1].normalize()).days > 7:
        return None
    if (close * vol).tail(60).mean() < MIN_TRADED_VALUE:
        return None
    if (vol.tail(20) == 0).sum() > MAX_ZERO_VOL_DAYS:
        return None

    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
    if not (price > dma200 and dma50 > dma200):
        return None

    m = monthly(df)
    partial = df.index[-1].to_period("M") == today.to_period("M")
    hist = m.iloc[:-1] if partial else m
    if len(hist) < MIN_AGE_MONTHS:
        return None

    closes = hist["Close"]
    is_ath_close = closes >= closes.cummax()
    recent = is_ath_close.iloc[-SETUP_LOOKBACK:]
    if not recent.any():
        return None
    setup_idx = recent[recent].index[-1]
    level = float(hist.loc[setup_idx:, "High"].max())
    trigger = level * (1 + ENTRY_BUFFER)

    gap = price / trigger - 1
    if gap > WATCH_UP or -gap > WATCH_DOWN:
        return None

    ret6 = price / float(close.iloc[-126]) - 1
    rs = (ret6 - nifty_6m) if nifty_6m is not None else 0.0
    recent_vol = vol.tail(5).mean()
    base_vol = vol.iloc[-55:-5].mean()
    vol_ratio = float(recent_vol / base_vol) if base_vol > 0 else 0.0
    pre = hist.loc[:setup_idx].iloc[:-1].tail(6)
    base_range = float(pre["High"].max() / pre["Close"].min()) if len(pre) >= 3 else 1.6

    rs_pts = clip(rs / 0.25) if nifty_6m is not None else 0.5
    vol_pts = clip((vol_ratio - 0.8) / 1.2)
    base_pts = clip((1.6 - base_range) / 0.5)
    trend_pts = clip((dma50 / dma200 - 1) / 0.15)
    static_score = 25 * rs_pts + 20 * vol_pts + 15 * base_pts + 10 * trend_pts

    return {
        "level": level,
        "trigger": float(trigger),
        "sl": float(level * (1 - SL_BELOW_LEVEL)),
        "static_score": float(static_score),
        "vol_ratio": vol_ratio,
        "rs": float(rs),
        "base_range": base_range - 1,
        "setup_month": setup_idx.strftime("%b'%y"),
    }


def classify(item, price):
    """Live step: turn a watchlist item + current price into TRIGGERED / POTENTIAL (or None)."""
    trigger = item["trigger"]
    gap = price / trigger - 1
    if gap >= 0:
        if gap > MAX_CHASE:
            return None
        status = "TRIGGERED"
        prox = 1 - gap / MAX_CHASE
    else:
        if -gap > POTENTIAL_BAND:
            return None
        status = "POTENTIAL"
        prox = 1 - (-gap) / POTENTIAL_BAND
    score = 30 * prox + item["static_score"]
    sl = item["sl"]
    entry = price if (ENTRY_AT_CMP and status == "TRIGGERED") else trigger
    target = entry + RR * (entry - sl)
    return {
        "symbol": item["symbol"],
        "status": status,
        "score": float(score),
        "price": float(price),
        "level": item["level"],
        "trigger": float(trigger),
        "entry": float(entry),
        "sl": float(sl),
        "target": float(target),
        "gap": float(gap),
        "vol_ratio": item["vol_ratio"],
        "rs": item["rs"],
        "base_range": item["base_range"],
        "setup_month": item["setup_month"],
    }


# ---------- Position tracking ----------
def fetch_daily(symbols, period="6mo", chunk=100):
    """Daily OHLC for a list of plain symbols (no .NS). Last row is today's live bar in market hours."""
    out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        tickers = [s + ".NS" for s in part]
        try:
            data = yf.download(
                tickers, period=period, interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception as e:
            print("Daily data failed:", e)
            continue
        if data is None or data.empty:
            continue
        for s, t in zip(part, tickers):
            try:
                df = pick(data, t)
                if getattr(df.index, "tz", None) is not None:
                    df = df.tz_localize(None)
                df = df.dropna(subset=["High", "Low", "Close"])
                if not df.empty:
                    out[s] = df
            except Exception:
                continue
    return out


def evaluate_position(pos, df, today):
    sl, target = pos["sl"], pos["target"]
    opened = pd.Timestamp(pos["opened"])
    last = float(df["Close"].iloc[-1])
    last_date = df.index[-1]

    for ts, row in df[df.index > opened].iterrows():
        o = row.get("Open")
        o = float(o) if pd.notna(o) else None
        hi, lo = float(row["High"]), float(row["Low"])
        if lo <= sl:
            exit_p = min(sl, o) if o is not None else sl
            note = "SL and target both touched this day; SL assumed first" if hi >= target else ""
            if o is not None and o < sl:
                note = (note + "; " if note else "") + "gapped below SL, exit at open"
            return {"result": "SL", "exit": exit_p, "date": ts, "note": note}
        if hi >= target:
            exit_p = max(target, o) if o is not None else target
            return {"result": "TARGET", "exit": exit_p, "date": ts, "note": ""}

    if last <= sl:
        return {"result": "SL", "exit": sl, "date": last_date, "note": ""}
    if last >= target:
        return {"result": "TARGET", "exit": target, "date": last_date, "note": ""}
    if (today - opened).days >= MAX_HOLD_DAYS:
        return {"result": "EXPIRED", "exit": last, "date": last_date, "note": ""}
    return {"result": "OPEN"}


def close_position(sym, pos, ev):
    entry, sl = pos["entry"], pos["sl"]
    exit_p = float(ev["exit"])
    risk = entry - sl
    opened = pd.Timestamp(pos["opened"])
    return {
        "symbol": sym,
        "result": ev["result"],
        "entry": float(entry),
        "exit": exit_p,
        "sl": float(sl),
        "target": float(pos["target"]),
        "pnl": float(exit_p / entry - 1),
        "r": float((exit_p - entry) / risk) if risk > 0 else 0.0,
        "opened": pos["opened"],
        "closed": ev["date"].strftime("%Y-%m-%d"),
        "days": int(max((ev["date"] - opened).days, 0)),
        "note": ev.get("note", ""),
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


# ---------- Full scan (once per trading day) ----------
def full_scan(symbols, today):
    nifty_6m = get_nifty_6m()
    items, scanned = [], 0
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        data = download_batch(batch)
        if data is None or data.empty:
            continue
        for sym in batch:
            try:
                df = pick(data, sym)
                res = analyse(df, nifty_6m, today)
                scanned += 1
            except Exception:
                continue
            if res:
                res["symbol"] = sym.replace(".NS", "")
                items.append(res)
        print(f"Progress: {min(i + BATCH, len(symbols))}/{len(symbols)}")
    print("Scanned", scanned, "| watchlist", len(items))
    return items, scanned


# ---------- Main ----------
def main():
    now = datetime.now(IST)
    today = pd.Timestamp(now.date())
    month_key = now.strftime("%Y-%m")
    day_key = now.strftime("%Y-%m-%d")

    if not FORCE:
        if not market_hours_now(now):
            print("Outside market hours / weekend - skipping (use --force to override)")
            return
        if not nifty_traded_today(today):
            print("Market holiday - skipping")
            return

    state = load_json(STATE_FILE)
    wl = load_json(WATCH_FILE)

    # ---- 1. watchlist: full scan on the first run of the day ----
    mode = "Live check"
    if FULL or wl.get("date") != day_key:
        symbols = load_symbols()
        if not symbols:
            discord("⚠️ ATH scan: could not load the stock list.")
            return
        items, scanned = full_scan(symbols, today)
        if scanned == 0:
            discord("⚠️ ATH scan: no price data returned, skipping this run.")
            return
        wl = {"date": day_key, "scanned": scanned, "items": items}
        save_json(WATCH_FILE, wl)
        mode = "Full scan"
    scanned = wl.get("scanned", 0)

    trig_done = set(state.get("triggered", {}).get(month_key, []))
    pot_done = set(state.get("potential", {}).get(day_key, [])) if DEDUPE_POTENTIAL_SAME_DAY else set()
    positions = state.get("positions", {})
    closed_hist = state.get("closed", [])

    # ---- 2. one price fetch for watchlist + open positions ----
    syms = sorted(set(positions) | {i["symbol"] for i in wl.get("items", [])})
    data = fetch_daily(syms)

    # ---- 3. update open positions (SL / target) ----
    prices = {}
    newly_closed = []
    for sym in list(positions):
        df = data.get(sym)
        if df is None:
            continue
        prices[sym] = float(df["Close"].iloc[-1])
        ev = evaluate_position(positions[sym], df, today)
        if ev["result"] != "OPEN":
            newly_closed.append(close_position(sym, positions.pop(sym), ev))
    closed_hist = (closed_hist + newly_closed)[-KEEP_CLOSED:]
    print("Open positions:", len(positions), "| closed this run:", len(newly_closed))

    # ---- 4. classify watchlist against live prices ----
    found = []
    for it in wl.get("items", []):
        df = data.get(it["symbol"])
        if df is None:
            continue
        if (today - df.index[-1].normalize()).days > 4:   # stale / suspended
            continue
        r = classify(it, float(df["Close"].iloc[-1]))
        if r:
            found.append(r)

    trig_all = sorted(
        [r for r in found
         if r["status"] == "TRIGGERED" and r["score"] >= MIN_SCORE
         and r["symbol"] not in trig_done and r["symbol"] not in positions],
        key=lambda x: x["score"], reverse=True,
    )
    pot_all = sorted(
        [r for r in found
         if r["status"] == "POTENTIAL" and r["score"] >= MIN_SCORE
         and r["symbol"] not in trig_done and r["symbol"] not in pot_done
         and r["symbol"] not in positions],
        key=lambda x: x["score"], reverse=True,
    )
    trig = trig_all[:MAX_TRIGGERED]
    pot = pot_all[:MAX_POTENTIAL]

    # ---- 5. open a tracked position for every new triggered entry ----
    for r in trig:
        positions[r["symbol"]] = {
            "entry": r["entry"],
            "sl": r["sl"],
            "target": r["target"],
            "level": r["level"],
            "opened": day_key,
            "setup_month": r["setup_month"],
        }
        prices[r["symbol"]] = r["price"]

    # ---- 6. Discord ----
    if newly_closed:
        discord_post({
            "content": f"🔔 **Position Update** · {now:%d %b %Y} · {now:%I:%M %p} IST",
            "embeds": [closed_embed(c) for c in newly_closed[:10]],
        })
    send_alert(trig, pot, scanned, now, len(trig_all), len(pot_all), mode)
    if positions:
        discord_post({"embeds": [open_embed(positions, prices, closed_hist, today)]})

    # ---- 7. save state ----
    trig_done.update(r["symbol"] for r in trig)
    pot_done.update(r["symbol"] for r in pot)
    save_json(STATE_FILE, {
        "triggered": {month_key: sorted(trig_done)},
        "potential": {day_key: sorted(pot_done)},
        "positions": positions,
        "closed": closed_hist,
    })


if __name__ == "__main__":
    main()
