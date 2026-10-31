import os
import json
import requests

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")


import pandas as pd

def quality_check(df):
    """Return a score if the stock passes all filters, else None."""
    df = df.dropna()
    if len(df) < 250:
        return None

    try:
        m = df.resample("ME").agg({"High": "max", "Close": "last", "Volume": "sum"})
    except ValueError:
        m = df.resample("M").agg({"High": "max", "Close": "last", "Volume": "sum"})
    if len(m) < 24:
        return None

    cur = m.iloc[-1]
    old_high = m["High"].iloc[:-1].max()

    # 1. Monthly CLOSE above the previous all-time high
    if cur["Close"] <= old_high:
        return None
    # 2. Not over-extended (max 15% above old high)
    if cur["Close"] > old_high * 1.15:
        return None
    # 3. Liquidity: average daily traded value over 60 days >= Rs 5 crore
    if (df["Close"] * df["Volume"]).tail(60).mean() < 5e7:
        return None
    # 4. Volume surge: month volume >= 1.5x the 20-month average
    if cur["Volume"] < 1.5 * m["Volume"].iloc[-21:-1].mean():
        return None
    # 5. Trend: price > 50 DMA > 200 DMA
    dma50 = df["Close"].rolling(50).mean().iloc[-1]
    dma200 = df["Close"].rolling(200).mean().iloc[-1]
    if not (df["Close"].iloc[-1] > dma50 > dma200):
        return None
    # 6. Base: prior 6 months stayed within a 30% range
    base = m.iloc[-7:-1]
    if (base["High"].max() / base["Close"].min()) > 1.30:
        return None

    # Score for ranking: 6-month return
    return df["Close"].iloc[-1] / df["Close"].iloc[-126] - 1
