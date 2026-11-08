import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .ai import ResearchAI, ResearchTurnDeadlineExceeded
from .crypto import BinanceMarketData
from .overlay_engine import prepare_chart_image, _render_overlay_prepared
from .image_upload import BoundedBytesIO, UploadTooLargeError
from .telegram_delivery import send_analysis_text, send_overlay_photo
from .user_locks import UserLockRegistry
from .config import (
    BETA_USER_LIMIT,
    DAILY_QUERY_LIMIT,
    MAX_CHART_IMAGE_BYTES,
    MAX_HISTORY_MESSAGES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    RESEARCH_TURN_TIMEOUT_SECONDS,
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
    OPENAI_API_KEY,
    OPENAI_MODEL,
    market,
    crypto_market,
    vision_model=OPENAI_VISION_MODEL,
    turn_timeout_seconds=RESEARCH_TURN_TIMEOUT_SECONDS,
)

# Prototype-only in-memory state. Persist this before external beta testing.
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
# Serialize requests per user so concurrent turns cannot race on shared history.
user_locks = UserLockRegistry()
usage = defaultdict(int)
usage_day = {}
beta_users = set()


def _admit_beta_user(user_id: int) -> bool:
    """Admit a user once without exceeding the process-local beta cap.

    Keep the check and set mutation synchronous: with one asyncio event loop,
    there is no task switch between these operations. Multi-worker deployments
    need shared atomic storage instead of this in-memory guard.
    """
    if user_id in beta_users:
        return True
    if len(beta_users) >= BETA_USER_LIMIT:
        return False
    beta_users.add(user_id)
    return True


def _consume_daily_query(user_id: int, today: str) -> bool:
    """Consume one admitted request slot for the supplied Asia/Kolkata date."""
    if usage_day.get(user_id) != today:
        usage_day[user_id] = today
        usage[user_id] = 0
    if usage[user_id] >= DAILY_QUERY_LIMIT:
        return False
    usage[user_id] += 1
    return True


def _research_admission_error(user_id: int) -> str | None:
    """Return a non-consuming beta/quota rejection before expensive image work.

    This is only an early preflight: _admit_beta_user and
    _consume_daily_query remain the authoritative checks after image validation.
    """
    if user_id not in beta_users and len(beta_users) >= BETA_USER_LIMIT:
        return "The 10-person beta is currently full. Please try again later."

    today = datetime.now(IST).date().isoformat()
    if usage_day.get(user_id) == today and usage.get(user_id, 0) >= DAILY_QUERY_LIMIT:
        return "You've reached today's beta research limit. It resets at midnight IST."
    return None


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

    async with user_locks.hold(user.id):
        await _run_research_locked(message, user_text, image_bytes, image_mime_type)


async def _run_research_locked(
    message: Message,
    user_text: str,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/jpeg",
):
    user = message.from_user
    if user is None:
        return

    user_id = user.id
    admission_error = _research_admission_error(user_id)
    if admission_error:
        await message.answer(admission_error)
        return

    if image_bytes is not None:
        try:
            # Inspect actual bytes and normalize EXIF orientation once. The exact
            # same canonical bytes go to vision, hashing and Pillow overlay rendering.
            image_bytes, image_mime_type = prepare_chart_image(
                image_bytes, max_output_bytes=MAX_CHART_IMAGE_BYTES
            )
        except ValueError as exc:
            await message.answer(str(exc))
            return

    if not _admit_beta_user(user_id):
        await message.answer("The 10-person beta is currently full. Please try again later.")
        return

    today = datetime.now(IST).date().isoformat()
    if not _consume_daily_query(user_id, today):
        await message.answer("You've reached today's beta research limit. It resets at midnight IST.")
        return
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
            overlay_render_failed = False
            try:
                overlay_image, drawn_count = _render_overlay_prepared(
                    image_bytes,
                    chart_result.get("zones", []),
                    plot_bounds=chart_result.get("plot_bounds"),
                    require_plot_bounds=True,
                )
            except Exception:
                logger.exception("Chart analysis completed, but overlay rendering failed")
                overlay_image, drawn_count = None, 0
                overlay_render_failed = True

            if overlay_render_failed:
                answer += (
                    "\n\nThe chart analysis completed, but I couldn't render the overlay. "
                    "Please verify the described zones against the original screenshot."
                )
            elif not drawn_count:
                answer += "\n\nI did not draw an overlay because no zone location was clear enough."
        else:
            answer = await research.answer(list(histories[user_id]), user_text)

        history_text = user_text or "Please analyse the attached chart screenshot for DNS zones."
        if image_bytes is not None:
            history_text += "\n[Chart screenshot attached; visual analysis was performed.]"
        histories[user_id].append({"role": "user", "content": history_text})
        histories[user_id].append({"role": "assistant", "content": answer})
    except ResearchTurnDeadlineExceeded as exc:
        if image_bytes is not None:
            failure_message = (
                "I stopped this chart analysis after its "
                f"{exc.timeout_seconds:g}-second time budget expired. "
                "No completed zone analysis or overlay was produced. "
                "Please retry; this request was not treated as a successful analysis."
            )
        else:
            failure_message = (
                "I stopped this research turn after its "
                f"{exc.timeout_seconds:g}-second time budget expired. "
                "No completed research answer was produced. A provider request may already "
                "have been received; no further model or market-data calls were started. "
                "Please retry with a narrower question."
            )
        if not await send_analysis_text(message, status, failure_message):
            logger.warning("StockMind research deadline expired and Telegram could not deliver the timeout message")
        return
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
    if not _admit_beta_user(user_id):
        await message.answer("The 10-person beta is currently full. Please try again later.")
        return
    await message.answer(WELCOME)


@dp.message(Command("reset"))
async def reset(message: Message):
    user_id = message.from_user.id
    # Wait for an active turn to finish before clearing its conversation context.
    async with user_locks.hold(user_id):
        history = histories.get(user_id)
        if history is not None:
            history.clear()
    await message.answer("Conversation context cleared. Ask a new research question.")


@dp.message(F.photo)
async def handle_chart_photo(message: Message):
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > MAX_CHART_IMAGE_BYTES:
        await message.answer("That screenshot is too large. Please send an image under 5 MB.")
        return

    if message.from_user is not None:
        admission_error = _research_admission_error(message.from_user.id)
        if admission_error:
            await message.answer(admission_error)
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

    if message.from_user is not None:
        admission_error = _research_admission_error(message.from_user.id)
        if admission_error:
            await message.answer(admission_error)
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
