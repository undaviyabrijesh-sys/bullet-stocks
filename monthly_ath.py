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
ENTRY_AT_CMP = True       # Enter at the first observed live price at/above trigger.
STOP_FRACTION = 0.25
T1_RETURN = 0.25
T2_RETURN = 0.40
T3_RETURN = 0.50
T4_RETURN = 1.00

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
DEDUPE_POTENTIAL_SAME_DAY = True    # Legacy setting; notifications are now deduped permanently per stock
BATCH = 50
ALLOWED_SERIES = {"EQ"}

GREEN = 0x2ECC71
AMBER = 0xF1C40F
RED = 0xE74C3C
BLUE = 0x3498DB
GREY = 0x95A5A6


# ---------- Discord ----------
def telegram_post(payload):
    """Mirror stock-event alerts to the configured Telegram group."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram not configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return

    lines = []
    content = payload.get("content")
    if content:
        lines.append(str(content))
    for embed in payload.get("embeds", []):
        if embed.get("title"):
            lines.append(str(embed["title"]))
        if embed.get("description"):
            lines.append(str(embed["description"]))
        for field in embed.get("fields", []):
            name, value = field.get("name"), field.get("value")
            if name or value:
                lines.append(f"{name or ''}: {value or ''}")
        footer = embed.get("footer", {}).get("text")
        if footer:
            lines.append(str(footer))
    message = "\n".join(lines).strip() or "Bullet Stocks event"
    # Telegram Bot API text limit is 4096 characters; leave room for chunking.
    chunks = [message[i:i + 3900] for i in range(0, len(message), 3900)]
    for chunk in chunks:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
                timeout=20,
            )
            if response.status_code >= 300:
                print("Telegram error:", response.status_code, response.text[:300])
        except Exception as exc:
            print("Telegram failed:", exc)


def discord_post(payload):
    if not WEBHOOK:
        print("No DISCORD_WEBHOOK_URL set")
        print(json.dumps(payload, indent=1, ensure_ascii=False))
    else:
        try:
            r = requests.post(WEBHOOK, json=payload, timeout=30)
            if r.status_code >= 300:
                print("Discord error:", r.status_code, r.text[:300])
        except Exception as e:
            print("Discord failed:", e)
    # Keep Telegram independent: a missing Discord webhook must not block it.
    telegram_post(payload)


def discord(msg):
    for i in range(0, len(msg), 1900):
        discord_post({"content": msg[i:i + 1900]})


def fmt(x):
    return f"₹{x:,.2f}"


def tv_link(symbol):
    return f"https://www.tradingview.com/chart/?symbol={quote('NSE:' + symbol, safe=':')}"


def stock_entry_field(r):
    """One compact stock card inside a Discord embed."""
    link = tv_link(r["symbol"])
    status_line = (
        "🟢 Trigger crossed — entry observed at current scan price"
        if r["status"] == "TRIGGERED"
        else f"🟡 Watchlist — {abs(r['gap']) * 100:.1f}% below trigger"
    )
    risk_pct = (r["entry"] - r["sl"]) / r["entry"] * 100 if r["entry"] else 0.0
    value = (
        f"{status_line}\n"
        f"**CMP:** {fmt(r['price'])}  ·  **Trigger:** {fmt(r['trigger'])}\n"
        f"**Planned entry:** {fmt(r['entry'])}  ·  **Initial SL:** {fmt(r['sl'])} ({risk_pct:.1f}% risk)\n"
        f"**Monthly ATH high:** {fmt(r['level'])} ({r['setup_month']})\n"
        f"**Score:** {r['score']:.0f}/100  ·  **Volume:** {r['vol_ratio']:.1f}x  ·  **RS vs Nifty:** {r['rs'] * 100:+.0f}%\n"
        f"**Exit plan:** T1 +25% (sell 50%) · T2 +40% (SL to cost) · "
        f"T3 +50% (sell half remaining) · T4 +100% (monthly SMA10 trail)\n"
        f"[Open chart]({link})"
    )
    return {"name": f"{r['symbol']}  |  Score {r['score']:.0f}/100", "value": value, "inline": False}


def build_embed(title, color, items, empty_text, intro):
    fields = [stock_entry_field(r) for r in items[:MAX_TRIGGERED]]
    return {
        "title": title,
        "description": intro if items else empty_text,
        "color": color,
        "fields": fields,
    }


def send_alert(trig, pot, scanned, now, n_trig_all, n_pot_all, mode):
    """Send a clean signal digest; called only for a fresh scan or actual candidates."""
    stamp = now.strftime("%d %b %Y · %I:%M %p IST")
    e1 = build_embed(
        f"🟢 BREAKOUT TRIGGERED · {len(trig)}",
        GREEN, trig,
        "No new breakout entries on this scan.",
        f"**New entry candidates:** {n_trig_all} · Showing the top {len(trig)} by score.",
    )
    e2 = build_embed(
        f"🟡 APPROACHING TRIGGER · {len(pot)}",
        AMBER, pot,
        "No watchlist stocks are currently within the trigger zone.",
        f"**Potential candidates:** {n_pot_all} · Showing the top {len(pot)} by score.",
    )
    footer = (
        f"{mode} · {scanned:,} NSE stocks scanned · Completed-month ATH close + 2% trigger · "
        f"Liquidity ≥ ₹{MIN_TRADED_VALUE / 1e7:.0f} Cr"
    )
    e1["footer"] = {"text": footer}
    e2["footer"] = {"text": footer + " · Educational tracking, not investment advice"}
    e1["timestamp"] = now.astimezone(timezone.utc).isoformat()
    e2["timestamp"] = now.astimezone(timezone.utc).isoformat()
    discord_post({
        "content": f"📡 **NSE MONTHLY ATH DESK**  ·  {stamp}",
        "embeds": [e1, e2],
    })


# ---------- Position alerts ----------
RESULT_STYLE = {
    "SL": ("🛑", "STOP-LOSS EXIT", RED),
    "EXPIRED": ("⏰", f"TIME EXIT · {MAX_HOLD_DAYS} DAYS", GREY),
    "COMPLETED": ("🏁", "POSITION COMPLETED", GREEN),
}


def closed_embed(c):
    icon, label, color = RESULT_STYLE.get(c["result"], ("📌", c["result"], GREY))
    opened = pd.Timestamp(c["opened"]).strftime("%d %b %Y")
    closed = pd.Timestamp(c["closed"]).strftime("%d %b %Y")
    desc = (
        f"**[{c['symbol']}]({tv_link(c['symbol'])})**\n"
        f"Trade opened **{opened}** and closed **{closed}** after **{c['days']} day(s)**."
    )
    if c.get("note"):
        desc += f"\n\n⚠️ {c['note']}"
    return {
        "title": f"{icon} {label} · {c['symbol']}",
        "description": desc,
        "color": color,
        "fields": [
            {"name": "Entry → Exit", "value": f"{fmt(c['entry'])} → **{fmt(c['exit'])}**", "inline": True},
            {"name": "Realized return", "value": f"**{c['pnl'] * 100:+.2f}%** of original position · **{c['r']:+.2f}R**", "inline": True},
            {"name": "Final stop / T4", "value": f"{fmt(c['sl'])} / {fmt(c['target'])}", "inline": True},
        ],
    }


EVENT_STYLE = {
    "T1": ("💰", "T1 REACHED · +25%", GREEN),
    "T2": ("🛡️", "T2 REACHED · STOP TO COST", GREEN),
    "T3": ("💵", "T3 REACHED · +50%", GREEN),
    "T4": ("🚀", "T4 REACHED · TRAILING ACTIVATED", BLUE),
    "TRAIL": ("🔒", "TRAILING STOP RAISED", BLUE),
}


def position_event_embed(symbol, ev, entry):
    icon, label, color = EVENT_STYLE.get(ev["type"], ("📌", ev["type"], GREY))
    event_date = pd.Timestamp(ev["date"])
    if event_date.tzinfo is None:
        event_date = event_date.tz_localize(IST)
    else:
        event_date = event_date.tz_convert(IST)

    remaining = float(ev.get("remaining_fraction", 1.0))
    sold = float(ev.get("sold_fraction", 0.0))
    active_sl = float(ev.get("active_sl", ev.get("new_sl", entry - 0.25 * entry)))
    move_pct = (float(ev["price"]) / entry - 1) * 100 if entry else 0.0
    fields = [
        {"name": "Entry", "value": fmt(entry), "inline": True},
        {"name": "Event price", "value": f"**{fmt(ev['price'])}** ({move_pct:+.1f}%)", "inline": True},
        {"name": "Active stop", "value": fmt(active_sl), "inline": True},
    ]
    if sold > 0:
        fields.append({"name": "Quantity sold", "value": f"{sold * 100:.0f}% of original quantity", "inline": True})
    fields.append({"name": "Quantity remaining", "value": f"**{remaining * 100:.0f}%**", "inline": True})
    if ev.get("ambiguous"):
        fields.append({"name": "Execution note", "value": "Stop and target were both inside the same 1-minute candle. Exact order is unknown; conservative stop-first handling was used.", "inline": False})

    return {
        "title": f"{icon} {label} · {symbol}",
        "description": f"**[{symbol}]({tv_link(symbol)})** · Position management update",
        "color": color,
        "fields": fields,
        "timestamp": event_date.tz_convert(timezone.utc).isoformat(),
        "footer": {"text": "Prices are based on 1-minute candles; a candle cannot reveal tick-by-tick order."},
    }


def open_embed(positions, prices, closed_hist, today):
    items = sorted(positions.items(), key=lambda kv: kv[1]["opened"])
    fields = []
    for sym, pos in items[:MAX_OPEN_SHOWN]:
        last = prices.get(sym)
        entry = float(pos["entry"])
        remaining = float(pos.get("remaining_fraction", 1.0))
        booked = float(pos.get("realised_pnl_fraction", 0.0))
        days = max((today - pd.Timestamp(pos["opened"]).tz_localize(None).normalize()).days, 0) if pd.Timestamp(pos["opened"]).tzinfo is not None else max((today - pd.Timestamp(pos["opened"]).normalize()).days, 0)
        stage = (
            "T4 trailing" if pos.get("trailing") else
            "T3 booked" if pos.get("t3_done") else
            "T2 · stop at cost" if pos.get("t2_done") else
            "T1 booked" if pos.get("t1_done") else "Initial"
        )
        if last is not None:
            open_contribution = remaining * (float(last) / entry - 1)
            total_contribution = booked + open_contribution
            price_move = (float(last) / entry - 1) * 100
            detail = (
                f"**CMP:** {fmt(last)} ({price_move:+.1f}% vs entry)\n"
                f"**Entry:** {fmt(entry)} · **Active SL:** {fmt(pos['sl'])}\n"
                f"**Remaining:** {remaining * 100:.0f}% · **Booked contribution:** {booked * 100:+.2f}%\n"
                f"**Total position contribution:** {total_contribution * 100:+.2f}% · **Stage:** {stage} · **Age:** {days}d"
            )
        else:
            detail = f"Price unavailable · **Entry:** {fmt(entry)} · **Active SL:** {fmt(pos['sl'])} · **Remaining:** {remaining * 100:.0f}%"
        fields.append({"name": f"{sym} · {stage}", "value": detail + f"\n[Open chart]({tv_link(sym)})", "inline": False})

    if len(items) > MAX_OPEN_SHOWN:
        fields.append({"name": "Additional positions", "value": f"{len(items) - MAX_OPEN_SHOWN} more position(s) not shown.", "inline": False})

    wins = sum(1 for c in closed_hist if c["pnl"] > 0)
    losses = len(closed_hist) - wins
    if closed_hist:
        stats = f"Closed trades: {wins}W · {losses}L · Win rate {wins / len(closed_hist) * 100:.0f}%"
    else:
        stats = "No closed trades yet"
    return {
        "title": f"📂 OPEN POSITIONS · {len(items)}",
        "description": "Live position monitor · values in the cards are updated from the latest available price.",
        "color": BLUE,
        "fields": fields,
        "footer": {"text": stats},
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    closes = hist["Close"].dropna()
    # Only the latest completed monthly candle can qualify as the current ATH setup.
    if len(closes) < MIN_AGE_MONTHS or closes.iloc[-1] < closes.iloc[:-1].max():
        return None

    setup_idx = hist.index[-1]
    # Confirmed rule: breakout is 2% above the previous completed month's HIGH.
    level = float(hist["High"].iloc[-1])
    trigger = level * (1 + ENTRY_BUFFER)
    sma20 = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None
    sma10 = float(closes.rolling(10).mean().iloc[-1]) if len(closes) >= 10 else None

    gap = price / trigger - 1
    if gap > WATCH_UP or -gap > WATCH_DOWN:
        return None

    ret6 = price / float(close.iloc[-126]) - 1
    rs = (ret6 - nifty_6m) if nifty_6m is not None else 0.0
    recent_vol = vol.tail(5).mean()
    base_vol = vol.iloc[-55:-5].mean()
    vol_ratio = float(recent_vol / base_vol) if base_vol > 0 else 0.0
    pre = hist.iloc[:-1].tail(6)
    base_range = float(pre["High"].max() / pre["Close"].min()) if len(pre) >= 3 else 1.6

    rs_pts = clip(rs / 0.25) if nifty_6m is not None else 0.5
    vol_pts = clip((vol_ratio - 0.8) / 1.2)
    base_pts = clip((1.6 - base_range) / 0.5)
    trend_pts = clip((dma50 / dma200 - 1) / 0.15)
    static_score = 25 * rs_pts + 20 * vol_pts + 15 * base_pts + 10 * trend_pts

    return {
        "level": level,
        "trigger": float(trigger),
        "monthly_sma20": sma20,
        "monthly_sma10": sma10,
        "static_score": float(static_score),
        "vol_ratio": vol_ratio,
        "rs": float(rs),
        "base_range": base_range - 1,
        "setup_month": setup_idx.strftime("%b'%y"),
    }


def classify(item, price):
    """Live step: classify price vs trigger and calculate the agreed initial stop."""
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
    entry = price if (ENTRY_AT_CMP and status == "TRIGGERED") else trigger
    sma20 = item.get("monthly_sma20")
    if sma20 is None or pd.isna(sma20):
        return None
    # Higher of 25% below entry and completed-month monthly SMA20.
    sl = max(entry * (1 - STOP_FRACTION), float(sma20))
    if sl >= entry:
        return None
    return {
        "symbol": item["symbol"],
        "status": status,
        "score": float(score),
        "price": float(price),
        "level": item["level"],
        "trigger": float(trigger),
        "entry": float(entry),
        "sl": float(sl),
        "monthly_sma20": float(sma20),
        "monthly_sma10": item.get("monthly_sma10"),
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


def fetch_intraday(symbols, period="5d", chunk=20, now=None):
    """Fetch completed 1-minute candles for open positions; ignore the newest two minutes."""
    out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        tickers = [s + ".NS" for s in part]
        try:
            data = yf.download(
                tickers, period=period, interval="1m", group_by="ticker",
                auto_adjust=False, prepost=False, threads=True, progress=False,
            )
        except Exception as e:
            print("1-minute data failed:", e)
            continue
        if data is None or data.empty:
            continue
        for symbol, ticker in zip(part, tickers):
            try:
                frame = pick(data, ticker)
                frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
                if frame.empty:
                    continue
                if frame.index.tz is None:
                    frame.index = frame.index.tz_localize(IST)
                else:
                    frame.index = frame.index.tz_convert(IST)
                asof = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=IST)
                if asof.tzinfo is None:
                    asof = asof.tz_localize(IST)
                else:
                    asof = asof.tz_convert(IST)
                cutoff = asof.floor("min") - pd.Timedelta(minutes=2)
                frame = frame[frame.index <= cutoff]
                if not frame.empty:
                    out[symbol] = frame.sort_index()
            except Exception as e:
                print(f"Could not prepare 1-minute data for {symbol}:", e)
    return out


def make_position(entry, level, opened, setup_month, monthly_sma20, monthly_sma10):
    """Create a position using the agreed staged exit model."""
    sl = max(entry * (1 - STOP_FRACTION), float(monthly_sma20))
    if sl >= entry:
        return None
    return {
        "entry": float(entry),
        "initial_sl": float(sl),
        "sl": float(sl),
        "level": float(level),
        "opened": opened,
        "setup_month": setup_month,
        "monthly_sma20": float(monthly_sma20),
        "monthly_sma10": float(monthly_sma10) if monthly_sma10 is not None else None,
        "t1": float(entry * (1 + T1_RETURN)),
        "t2": float(entry * (1 + T2_RETURN)),
        "t3": float(entry * (1 + T3_RETURN)),
        "t4": float(entry * (1 + T4_RETURN)),
        "t1_done": False,
        "t2_done": False,
        "t3_done": False,
        "trailing": False,
        "remaining_fraction": 1.0,
        "realised_pnl_fraction": 0.0,
        "last_processed_bar": opened if "T" in str(opened) else None,
    }


def migrate_position(pos, monthly_sma20, monthly_sma10):
    """Upgrade legacy single-target state records without losing their entry/history."""
    if "t1" in pos and "remaining_fraction" in pos:
        if "last_processed_bar" not in pos:
            pos["last_processed_bar"] = None
        if monthly_sma10 is not None:
            pos["monthly_sma10"] = float(monthly_sma10)
        return pos
    entry = float(pos["entry"])
    sma20 = float(monthly_sma20) if monthly_sma20 is not None and monthly_sma20 < entry else entry * (1 - STOP_FRACTION)
    upgraded = make_position(
        entry, float(pos.get("level", entry)), pos.get("opened", datetime.now(IST).strftime("%Y-%m-%d")),
        pos.get("setup_month", ""), sma20, monthly_sma10,
    )
    if upgraded is None:
        return pos
    upgraded["migration_note"] = "Migrated from legacy single-target state; staged targets start from original entry."
    return upgraded


def fetch_monthly_indicators(symbols, today, chunk=50):
    """Fetch completed monthly closes and calculate non-repainting SMA10/SMA20."""
    out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        tickers = [sym + ".NS" for sym in part]
        try:
            data = yf.download(tickers, period="5y", interval="1mo", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print("Monthly indicator fetch failed:", e)
            continue
        if data is None or data.empty:
            continue
        for sym, ticker in zip(part, tickers):
            try:
                frame = pick(data, ticker)
                close = frame["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                if len(close) and pd.Timestamp(close.index[-1]).to_period("M") == today.to_period("M"):
                    close = close.iloc[:-1]
                out[sym] = {
                    "sma20": float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None,
                    "sma10": float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else None,
                }
            except Exception:
                continue
    return out


def evaluate_intraday_position(pos, df, today, monthly_sma10=None):
    """Process only new 1-minute bars; never replay candles already evaluated."""
    entry = float(pos["entry"])
    sl = float(pos["sl"])
    remaining = float(pos.get("remaining_fraction", 1.0))
    realised = float(pos.get("realised_pnl_fraction", 0.0))
    events = []
    closed = None

    if df is None or df.empty:
        return events, None

    df = df.sort_index()
    opened = pd.Timestamp(pos["opened"])
    if opened.tzinfo is None:
        opened = opened.tz_localize(IST)
    else:
        opened = opened.tz_convert(IST)

    last_processed = pos.get("last_processed_bar")
    if last_processed:
        last_processed = pd.Timestamp(last_processed)
        if last_processed.tzinfo is None:
            last_processed = last_processed.tz_localize(IST)
        else:
            last_processed = last_processed.tz_convert(IST)
    else:
        # First run after enabling intraday tracking: anchor at the newest bar.
        # This avoids replaying historical candles against already-updated stop/target flags.
        pos["last_processed_bar"] = df.index[-1].isoformat()
        return events, None

    # A completed-month SMA10 step-up is effective before processing new bars.
    if pos.get("trailing") and monthly_sma10 is not None:
        new_sl = max(sl, float(monthly_sma10))
        if new_sl > sl:
            sl = new_sl
            events.append({
                "type": "TRAIL", "price": sl, "new_sl": sl, "active_sl": sl,
                "remaining_fraction": remaining, "date": df.index[-1],
            })

    for ts, row in df[df.index > last_processed].iterrows():
        if ts <= opened:
            pos["last_processed_bar"] = ts.isoformat()
            continue

        o = float(row["Open"]) if pd.notna(row.get("Open")) else None
        hi, lo = float(row["High"]), float(row["Low"])
        target_hits = []
        for key, done_key in (("t1", "t1_done"), ("t2", "t2_done"), ("t3", "t3_done"), ("t4", "trailing")):
            if not pos.get(done_key) and hi >= float(pos[key]):
                target_hits.append(key)

        stop_hit = lo <= sl
        if stop_hit and target_hits:
            exit_p = min(sl, o) if o is not None else sl
            if remaining > 0:
                realised += remaining * (exit_p / entry - 1)
                events.append({
                    "type": "SL", "price": exit_p, "date": ts, "remaining": remaining,
                    "remaining_fraction": 0.0, "active_sl": sl, "ambiguous": True,
                })
            remaining = 0.0
            closed = {
                "result": "SL", "exit": exit_p, "date": ts, "pnl": realised,
                "note": "The same 1-minute candle touched the active stop and at least one target; exact order is unknown, so stop-first handling was used.",
            }
            pos["last_processed_bar"] = ts.isoformat()
            break

        if stop_hit:
            exit_p = min(sl, o) if o is not None else sl
            if remaining > 0:
                realised += remaining * (exit_p / entry - 1)
                events.append({
                    "type": "SL", "price": exit_p, "date": ts, "remaining": remaining,
                    "remaining_fraction": 0.0, "active_sl": sl,
                })
            remaining = 0.0
            closed = {
                "result": "SL", "exit": exit_p, "date": ts, "pnl": realised,
                "note": "Active stop-loss touched by a 1-minute candle.",
            }
            pos["last_processed_bar"] = ts.isoformat()
            break

        # When several ascending targets are crossed in one candle, process them in rule order.
        newly_active_stop = None
        if "t1" in target_hits:
            sold = min(remaining, 0.50)
            realised += sold * (float(pos["t1"]) / entry - 1)
            remaining -= sold
            pos["t1_done"] = True
            events.append({
                "type": "T1", "price": float(pos["t1"]), "sold_fraction": sold,
                "remaining_fraction": remaining, "active_sl": sl,
                "realised_pnl_fraction": realised, "date": ts,
            })
        if "t2" in target_hits:
            sl = max(sl, entry)
            newly_active_stop = sl
            pos["t2_done"] = True
            events.append({
                "type": "T2", "price": float(pos["t2"]), "new_sl": sl, "active_sl": sl,
                "remaining_fraction": remaining, "realised_pnl_fraction": realised, "date": ts,
            })
        if "t3" in target_hits and remaining > 0:
            sold = min(remaining, remaining * 0.50)
            realised += sold * (float(pos["t3"]) / entry - 1)
            remaining -= sold
            pos["t3_done"] = True
            events.append({
                "type": "T3", "price": float(pos["t3"]), "sold_fraction": sold,
                "remaining_fraction": remaining, "active_sl": sl,
                "realised_pnl_fraction": realised, "date": ts,
            })
        if "t4" in target_hits:
            pos["trailing"] = True
            if monthly_sma10 is not None:
                sl = max(sl, float(monthly_sma10))
            newly_active_stop = sl
            events.append({
                "type": "T4", "price": float(pos["t4"]), "new_sl": sl, "active_sl": sl,
                "remaining_fraction": remaining, "date": ts,
            })

        if newly_active_stop is not None and remaining > 0 and lo <= newly_active_stop:
            exit_p = newly_active_stop
            realised += remaining * (exit_p / entry - 1)
            events.append({
                "type": "SL", "price": exit_p, "date": ts, "remaining": remaining,
                "remaining_fraction": 0.0, "active_sl": newly_active_stop, "ambiguous": True,
            })
            remaining = 0.0
            closed = {
                "result": "SL", "exit": exit_p, "date": ts, "pnl": realised,
                "note": "A target raised the stop inside this 1-minute candle and the candle also crossed that new stop. The exact sequence is unknown; conservative stop handling was used.",
            }
            pos["last_processed_bar"] = ts.isoformat()
            break

        pos["last_processed_bar"] = ts.isoformat()
        if remaining <= 0:
            closed = {
                "result": "COMPLETED", "exit": float(row["Close"]), "date": ts,
                "pnl": realised, "note": "All planned quantities booked.",
            }
            break

    pos["sl"] = float(sl)
    pos["remaining_fraction"] = max(0.0, remaining)
    pos["realised_pnl_fraction"] = float(realised)

    if closed is None and remaining > 0 and (today.date() - opened.date()).days >= MAX_HOLD_DAYS:
        last = float(df["Close"].iloc[-1])
        last_date = df.index[-1]
        realised += remaining * (last / entry - 1)
        closed = {
            "result": "EXPIRED", "exit": last, "date": last_date,
            "pnl": realised, "note": f"Position exceeded the {MAX_HOLD_DAYS}-day holding window.",
        }
        remaining = 0.0

    pos["remaining_fraction"] = max(0.0, remaining)
    pos["realised_pnl_fraction"] = float(realised)
    return events, closed


def close_position(sym, pos, ev):
    entry = float(pos["entry"])
    risk = entry - float(pos.get("initial_sl", pos["sl"]))
    opened = pd.Timestamp(pos["opened"])
    event_date = pd.Timestamp(ev["date"])
    if opened.tzinfo is None:
        opened = opened.tz_localize(IST)
    else:
        opened = opened.tz_convert(IST)
    if event_date.tzinfo is None:
        event_date = event_date.tz_localize(IST)
    else:
        event_date = event_date.tz_convert(IST)
    return {
        "symbol": sym,
        "result": ev["result"],
        "entry": entry,
        "exit": float(ev["exit"]),
        "sl": float(pos["sl"]),
        "target": float(pos["t4"]),
        "pnl": float(ev.get("pnl", pos.get("realised_pnl_fraction", 0.0))),
        "r": float(ev.get("pnl", 0.0) * entry / risk) if risk > 0 else 0.0,
        "opened": pos["opened"],
        "closed": event_date.strftime("%Y-%m-%d"),
        "days": int(max((event_date - opened).days, 0)),
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
    if FULL or wl.get("date") != day_key or wl.get("strategy_version") != 2:
        symbols = load_symbols()
        if not symbols:
            discord("⚠️ ATH scan: could not load the stock list.")
            return
        items, scanned = full_scan(symbols, today)
        if scanned == 0:
            discord("⚠️ ATH scan: no price data returned, skipping this run.")
            return
        wl = {"date": day_key, "strategy_version": 2, "scanned": scanned, "items": items}
        save_json(WATCH_FILE, wl)
        mode = "Full scan"
    scanned = wl.get("scanned", 0)

    trig_done = set(state.get("triggered", {}).get(month_key, []))
    # Remember potential/watchlist notifications permanently so the same stock does not ping again every day.
    legacy_potential = state.get("potential", {})
    legacy_seen = {sym for values in legacy_potential.values() for sym in values} if isinstance(legacy_potential, dict) else set()
    pot_done = set(state.get("potential_notified", [])) | legacy_seen
    positions = state.get("positions", {})
    closed_hist = state.get("closed", [])

    # ---- 2. one price fetch for watchlist + open positions ----
    syms = sorted(set(positions) | {i["symbol"] for i in wl.get("items", [])})
    data = fetch_daily(syms)

    # ---- 3. manage open positions from 1-minute candles ----
    prices = {}
    newly_closed = []
    milestone_alerts = []
    monthly_values = fetch_monthly_indicators(list(positions), today)
    intraday_values = fetch_intraday(list(positions), period="5d", now=now)

    for sym in list(positions):
        daily_df = data.get(sym)
        minute_df = intraday_values.get(sym)
        if daily_df is not None and not daily_df.empty:
            prices[sym] = float(daily_df["Close"].iloc[-1])
        if minute_df is None or minute_df.empty:
            print(f"1-minute price data unavailable for {sym}; keeping position open without daily-OHLC exit guesses.")
            continue

        prices[sym] = float(minute_df["Close"].iloc[-1])
        smas = monthly_values.get(sym, {})
        old = positions[sym]
        if "t1" not in old:
            positions[sym] = migrate_position(old, smas.get("sma20"), smas.get("sma10"))

        # Existing staged positions are anchored at the newest minute on first adoption.
        # Newly created positions have an exact opened timestamp and process later bars.
        if not positions[sym].get("last_processed_bar"):
            positions[sym]["last_processed_bar"] = minute_df.index[-1].isoformat()
            print(f"{sym}: initialized 1-minute checkpoint at {minute_df.index[-1]}")
            continue

        evs, ev = evaluate_intraday_position(positions[sym], minute_df, today, smas.get("sma10"))
        for item in evs:
            if item["type"] != "SL":
                milestone_alerts.append((sym, item, float(positions[sym]["entry"])))
        if ev is not None:
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
        positions[r["symbol"]] = make_position(
            r["entry"], r["level"], now.isoformat(timespec="seconds"), r["setup_month"],
            r["monthly_sma20"], r.get("monthly_sma10"),
        )
        prices[r["symbol"]] = r["price"]

    # ---- 6. Discord ----
    for sym, ev, event_entry in milestone_alerts:
        discord_post({"embeds": [position_event_embed(sym, ev, event_entry)]})

    if newly_closed:
        discord_post({
            "content": f"📣 **POSITION EXIT UPDATE** · {now:%d %b %Y · %I:%M %p} IST",
            "embeds": [closed_embed(c) for c in newly_closed[:10]],
        })

    # Send only genuine new events. No empty scan reports and no routine portfolio snapshots.
    if trig or pot:
        send_alert(trig, pot, scanned, now, len(trig_all), len(pot_all), mode)

    # ---- 7. save state ----
    trig_done.update(r["symbol"] for r in trig)
    pot_done.update(r["symbol"] for r in pot)
    save_json(STATE_FILE, {
        "triggered": {month_key: sorted(trig_done)},
        "potential_notified": sorted(pot_done),
        "positions": positions,
        "closed": closed_hist,
    })


if __name__ == "__main__":
    main()
