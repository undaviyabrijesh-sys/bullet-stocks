import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .ai import ResearchAI
from .crypto import BinanceMarketData
from .overlay_engine import render_overlay, validate_chart_image
from .image_upload import BoundedBytesIO, UploadTooLargeError
from .telegram_delivery import send_analysis_text, send_overlay_photo
from .config import (
    BETA_USER_LIMIT,
    DAILY_QUERY_LIMIT,
    MAX_CHART_IMAGE_BYTES,
    MAX_HISTORY_MESSAGES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    TAPETIDE_TOKEN,
    TELEGRAM_BOT_TOKEN,
)
from .tapetide import TapetideClient

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

market = TapetideClient(TAPETIDE_TOKEN)
crypto_market = BinanceMarketData()
research = ResearchAI(
    OPENAI_API_KEY, OPENAI_MODEL, market, crypto_market, vision_model=OPENAI_VISION_MODEL
)

# Prototype-only in-memory state. Persist this before external beta testing.
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
usage = defaultdict(int)
usage_day = {}
beta_users = set()

bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

WELCOME = (
    "👋 Welcome to StockMind AI Beta!\n\n"
    "Ask about Indian stocks and crypto in English, Hinglish, Hindi, Gujarati or mixed language.\n\n"
    "You can also upload a chart screenshot for Demand & Supply (DNS), fresh/mitigated zones, "
    "BOS/CHOCH and liquidity analysis. When a zone is clear, I return an annotated copy with "
    "red supply and green demand bands.\n\n"
    "Examples:\n"
    "• Analyse Reliance fundamentals\n"
    "• Bhai HDFC Bank ka kya scene hai?\n"
    "• BTC 1H chart par demand supply draw karo\n"
    "• Is this demand zone fresh or mitigated?\n"
    "• Compare TCS and Infosys\n\n"
    "For chart screenshots, keep the price scale and timeframe visible. Add symbol, exchange "
    "and timeframe in your caption when they are not shown. Visual price levels may be estimates.\n\n"
    "I provide research, not trade execution or guaranteed buy/sell calls.\n"
    "Beta limit: 10 research questions per day."
)


async def run_research(
    message: Message,
    user_text: str,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/jpeg",
):
    user = message.from_user
    if user is None:
        return

    if image_bytes is not None:
        try:
            # Inspect the actual file bytes; do not trust the Telegram MIME header.
            image_mime_type = validate_chart_image(image_bytes)
        except ValueError as exc:
            await message.answer(str(exc))
            return

    user_id = user.id
    if user_id not in beta_users and len(beta_users) >= BETA_USER_LIMIT:
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
    try:
        status = await message.answer(
            "🖼️ Reading the chart and mapping visible structure…"
            if image_bytes is not None
            else "🔎 Understanding your question and checking market data…"
        )
    except Exception:
        # A progress-message failure should not prevent the research itself.
        logger.exception("Could not send StockMind progress status")
        status = None
    try:
        overlay_image = None
        if image_bytes is not None:
            chart_result = await research.analyze_chart_image(
                list(histories[user_id]), user_text, image_bytes, image_mime_type
            )
            answer = chart_result.get("analysis") or "I could not read the chart reliably."
            try:
                overlay_image, drawn_count = render_overlay(
                image_bytes,
                chart_result.get("zones", []),
                validated=True,
                plot_bounds=chart_result.get("plot_bounds"),
                require_plot_bounds=True,
            )
            except Exception:
                overlay_image, drawn_count = None, 0
            if not drawn_count:
                answer += "\n\nI did not draw an overlay because no zone location was clear enough."
        else:
            answer = await research.answer(list(histories[user_id]), user_text)

        history_text = user_text or "Please analyse the attached chart screenshot for DNS zones."
        if image_bytes is not None:
            history_text += "\n[Chart screenshot attached; visual analysis was performed.]"
        histories[user_id].append({"role": "user", "content": history_text})
        histories[user_id].append({"role": "assistant", "content": answer})
    except Exception:
        logger.exception("StockMind research request failed")
        failure_message = (
            "I couldn't complete this analysis right now. For screenshots, check that "
            "OPENAI_VISION_MODEL is set to a vision-capable model and try again. "
            "No missing price level or market figure has been guessed."
        )
        if not await send_analysis_text(message, status, failure_message):
            logger.warning("StockMind research failed and Telegram could not deliver the error message")
        return

    if not await send_analysis_text(message, status, answer[:4000]):
        logger.warning("StockMind analysis succeeded, but Telegram text delivery failed")

    # Deliver the overlay after the written analysis has been committed to chat.
    # A Telegram photo-send failure must not turn a successful AI analysis into
    # a failed research response or overwrite the user's readable answer.
    if overlay_image is not None:
        delivered = await send_overlay_photo(
            message,
            overlay_image,
            caption=(
                "StockMind AI visual zone overlay — estimated from this screenshot; "
                "verify against the chart price scale."
            ),
        )
        if not delivered:
            logger.warning("Chart analysis succeeded, but Telegram overlay delivery failed")


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


@dp.message(F.photo)
async def handle_chart_photo(message: Message):
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > MAX_CHART_IMAGE_BYTES:
        await message.answer("That screenshot is too large. Please send an image under 5 MB.")
        return

    try:
        telegram_file = await bot.get_file(photo.file_id)
        if not telegram_file.file_path:
            raise RuntimeError("Telegram did not return a file path.")
        buffer = BoundedBytesIO(MAX_CHART_IMAGE_BYTES)
        await bot.download_file(telegram_file.file_path, destination=buffer)
        image_bytes = buffer.getvalue()
        if len(image_bytes) > MAX_CHART_IMAGE_BYTES:
            await message.answer("That screenshot is too large. Please send an image under 5 MB.")
            return
    except UploadTooLargeError:
        await message.answer("That screenshot is too large. Please send an image under 5 MB.")
        return
    except Exception:
        await message.answer("I couldn't download that screenshot. Please send it again.")
        return

    caption = message.caption or (
        "Please analyse this chart screenshot for Demand and Supply zones, structure and "
        "conditional setups. Tell me if the symbol or timeframe is not visible."
    )
    await run_research(message, caption, image_bytes, "image/jpeg")


@dp.message(F.document)
async def handle_chart_document(message: Message):
    document = message.document
    mime_type = document.mime_type or ""
    if mime_type not in ALLOWED_IMAGE_TYPES:
        await message.answer("Please attach a chart as a photo or as a JPEG, PNG or WEBP image file.")
        return
    if document.file_size and document.file_size > MAX_CHART_IMAGE_BYTES:
        await message.answer("That image is too large. Please send an image under 5 MB.")
        return

    try:
        telegram_file = await bot.get_file(document.file_id)
        if not telegram_file.file_path:
            raise RuntimeError("Telegram did not return a file path.")
        buffer = BoundedBytesIO(MAX_CHART_IMAGE_BYTES)
        await bot.download_file(telegram_file.file_path, destination=buffer)
        image_bytes = buffer.getvalue()
        if len(image_bytes) > MAX_CHART_IMAGE_BYTES:
            await message.answer("That image is too large. Please send an image under 5 MB.")
            return
    except UploadTooLargeError:
        await message.answer("That image is too large. Please send an image under 5 MB.")
        return
    except Exception:
        await message.answer("I couldn't download that image. Please send it again.")
        return

    caption = message.caption or (
        "Please analyse this chart screenshot for Demand and Supply zones, structure and "
        "conditional setups. Tell me if the symbol or timeframe is not visible."
    )
    await run_research(message, caption, image_bytes, mime_type)


@dp.message(F.text)
async def handle_text(message: Message):
    await run_research(message, message.text)


@dp.message()
async def unsupported(message: Message):
    await message.answer(
        "Please send a text question or upload a chart screenshot as a photo/image. "
        "Voice support can be added separately."
    )


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await market.close()
        await crypto_market.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
