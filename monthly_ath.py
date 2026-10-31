import os
import json
import requests

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")


def telegram(msg):
    if not (TG_TOKEN and TG_CHAT):
        return
    plain = msg.replace("**", "")
    for i in range(0, len(plain), 4000):
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": plain[i:i + 4000]},
                timeout=30,
            )
        except Exception as e:
            print("telegram failed:", e)


def discord(msg):
    for i in range(0, len(msg), 1900):
        requests.post(WEBHOOK, json={"content": msg[i:i + 1900]}, timeout=30)
    telegram(msg)
