import asyncio
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .ai import ResearchAI
from .config import (
    BETA_USER_LIMIT,
    DAILY_QUERY_LIMIT,
    MAX_HISTORY_MESSAGES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TAPETIDE_TOKEN,
    TELEGRAM_BOT_TOKEN,
)
from .tapetide import TapetideClient

IST = ZoneInfo("Asia/Kolkata")
market = TapetideClient(TAPETIDE_TOKEN)
research = ResearchAI(OPENAI_API_KEY, OPENAI_MODEL, market)

# Prototype-only in-memory state. Persist this before external beta testing.
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
usage = defaultdict(int)
usage_day = {}
beta_users = set()

bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

WELCOME = (
    "👋 Welcome to StockMind AI Beta!\n\n"
    "Ask about Indian stocks in English, Hinglish, Hindi, Gujarati or your own words.\n\n"
    "Examples:\n"
    "• Analyse Reliance fundamentals\n"
    "• Bhai HDFC Bank ka kya scene hai?\n"
    "• Find stocks near 52-week lows\n"
    "• Compare TCS and Infosys\n\n"
    "I provide research, not trade execution or guaranteed buy/sell calls.\n"
    "Beta limit: 10 research questions per day."
)


@dp.message(CommandStart())
async def start(message: Message):
    user_id = message.from_user.id
    if user_id not in beta_users and len(beta_users) >= BETA_USER_LIMIT:
        await message.answer("The 10-person beta is currently full. Please try again later.")
        return
    beta_users.add(user_id)
    await message.answer(WELCOME)


@dp.message(Command("reset"))
async def reset(message: Message):
    histories[message.from_user.id].clear()
    await message.answer("Conversation context cleared. Ask a new research question.")


@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    if user_id not in beta_users:
        if len(beta_users) >= BETA_USER_LIMIT:
            await message.answer("The 10-person beta is currently full. Please try again later.")
            return
        beta_users.add(user_id)

    today = datetime.now(IST).date().isoformat()
    if usage_day.get(user_id) != today:
        usage_day[user_id] = today
        usage[user_id] = 0

    if usage[user_id] >= DAILY_QUERY_LIMIT:
        await message.answer("You've reached today's beta research limit. It resets at midnight IST.")
        return

    usage[user_id] += 1
    status = await message.answer("🔎 Understanding your question and checking market data…")
    try:
        answer = await research.answer(list(histories[user_id]), message.text)
        histories[user_id].append({"role": "user", "content": message.text})
        histories[user_id].append({"role": "assistant", "content": answer})
        await status.edit_text(answer[:4000])
    except Exception:
        await status.edit_text(
            "I hit a temporary issue while researching. Please try again shortly. "
            "No market figure has been guessed."
        )


@dp.message()
async def unsupported(message: Message):
    await message.answer("For this beta, please send a text question. Voice support can be added next.")


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await market.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
