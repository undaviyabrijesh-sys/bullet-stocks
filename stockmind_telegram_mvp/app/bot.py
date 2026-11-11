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
from .localization import LANGUAGE_NAMES, normalize_language, resolve_language, translate
from .user_preferences import LanguagePreferenceStore
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
    USER_PREFERENCES_DB,
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
# Cache preferences used during this process; SQLite is authoritative across restarts.
user_languages = {}
try:
    language_preferences = LanguagePreferenceStore(USER_PREFERENCES_DB)
except Exception:
    logger.exception("Persistent StockMind language preferences are unavailable")
    language_preferences = None


def _overlay_unavailable_note(chart_result: dict, drawn_count: int, language: str = "en") -> str:
    """Explain why an otherwise successful chart analysis has no overlay."""
    if drawn_count:
        return ""

    plot_bounds = chart_result.get("plot_bounds")
    zones = chart_result.get("zones")
    if not isinstance(plot_bounds, dict):
        return translate(language, "overlay_plot_missing")

    if not isinstance(zones, (list, tuple)) or not zones:
        return translate(language, "overlay_no_zone")

    zone_objects = [zone for zone in zones if isinstance(zone, dict)]
    if not zone_objects:
        return translate(language, "overlay_invalid_zone_objects")

    drawable_candidates = [
        zone for zone in zone_objects if zone.get("draw_overlay") is True
    ]
    if not drawable_candidates:
        return translate(language, "overlay_not_drawable")

    return translate(language, "overlay_renderer_rejected")


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


def _research_admission_error(user_id: int, language: str = "en") -> str | None:
    """Return a non-consuming beta/quota rejection before expensive image work.

    This is only an early preflight: _admit_beta_user and
    _consume_daily_query remain the authoritative checks after image validation.
    """
    if user_id not in beta_users and len(beta_users) >= BETA_USER_LIMIT:
        return translate(language, "beta_full")

    today = datetime.now(IST).date().isoformat()
    if usage_day.get(user_id) == today and usage.get(user_id, 0) >= DAILY_QUERY_LIMIT:
        return translate(language, "daily_limit")
    return None


bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

WELCOME = translate("en", "welcome")


def _language_for(message: Message) -> str:
    user = getattr(message, "from_user", None)
    if user is None:
        return "en"

    explicit = user_languages.get(user.id)
    if explicit is None and language_preferences is not None:
        try:
            explicit = language_preferences.get(user.id)
        except Exception:
            logger.exception("Could not load a saved StockMind language preference")
        if explicit is not None:
            user_languages[user.id] = explicit

    return resolve_language(
        explicit,
        getattr(user, "language_code", None),
    )


def _ui(message: Message, key: str, **values) -> str:
    return translate(_language_for(message), key, **values)


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
    language = _language_for(message)
    admission_error = _research_admission_error(user_id, language)
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
        await message.answer(translate(language, "beta_full"))
        return

    today = datetime.now(IST).date().isoformat()
    if not _consume_daily_query(user_id, today):
        await message.answer(translate(language, "daily_limit"))
        return
    try:
        status = await message.answer(
            translate(language, "progress_chart")
            if image_bytes is not None
            else translate(language, "progress_research")
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
            answer = chart_result.get("analysis") or translate(language, "chart_analysis_fallback")
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
                    translate(language, "overlay_render_failed")
                )
            elif not drawn_count:
                answer += _overlay_unavailable_note(chart_result, drawn_count, language)
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
                translate(language, "chart_timeout", seconds=exc.timeout_seconds)
            )
        else:
            failure_message = (
                translate(language, "research_timeout", seconds=exc.timeout_seconds)
            )
        if not await send_analysis_text(message, status, failure_message):
            logger.warning("StockMind research deadline expired and Telegram could not deliver the timeout message")
        return
    except Exception:
        logger.exception("StockMind research request failed")
        failure_message = translate(language, "research_failed")
        if not await send_analysis_text(message, status, failure_message):
            logger.warning("StockMind research failed and Telegram could not deliver the error message")
        return

    if not await send_analysis_text(message, status, answer):
        logger.warning("StockMind analysis succeeded, but Telegram text delivery failed")

    # Deliver the overlay after the written analysis has been committed to chat.
    # A Telegram photo-send failure must not turn a successful AI analysis into
    # a failed research response or overwrite the user's readable answer.
    if overlay_image is not None:
        delivered = await send_overlay_photo(
            message,
            overlay_image,
            caption=(
                translate(language, "overlay_caption")
            ),
        )
        if not delivered:
            logger.warning("Chart analysis succeeded, but Telegram overlay delivery failed")


@dp.message(CommandStart())
async def start(message: Message):
    user_id = message.from_user.id
    language = _language_for(message)
    if not _admit_beta_user(user_id):
        await message.answer(translate(language, "beta_full"))
        return
    await message.answer(translate(language, "welcome"))



@dp.message(Command("language", "lang"))
async def set_language(message: Message):
    user = message.from_user
    if user is None:
        return
    current = _language_for(message)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        await message.answer(
            translate(
                current,
                "language_help",
                language_name=LANGUAGE_NAMES[current],
            )
        )
        return

    requested = normalize_language(parts[1])
    if requested is None:
        await message.answer(translate(current, "language_invalid"))
        return

    # Apply the choice immediately, even if the local preference database is
    # temporarily unavailable. Never claim persistence unless SQLite confirms it.
    user_languages[user.id] = requested
    persisted = False
    if language_preferences is not None:
        try:
            language_preferences.set(user.id, requested)
            persisted = True
        except Exception:
            logger.exception("Could not save a StockMind language preference")

    response_key = "language_changed" if persisted else "language_changed_not_saved"
    await message.answer(
        translate(
            requested,
            response_key,
            language_name=LANGUAGE_NAMES[requested],
        )
    )


@dp.message(Command("reset"))
async def reset(message: Message):
    user_id = message.from_user.id
    # Wait for an active turn to finish before clearing its conversation context.
    async with user_locks.hold(user_id):
        history = histories.get(user_id)
        if history is not None:
            history.clear()
    await message.answer(_ui(message, "reset_done"))


@dp.message(F.photo)
async def handle_chart_photo(message: Message):
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > MAX_CHART_IMAGE_BYTES:
        await message.answer(_ui(message, "upload_too_large"))
        return

    if message.from_user is not None:
        admission_error = _research_admission_error(message.from_user.id, _language_for(message))
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
            await message.answer(_ui(message, "upload_too_large"))
            return
    except UploadTooLargeError:
        await message.answer("That screenshot is too large. Please send an image under 5 MB.")
        return
    except Exception:
        await message.answer(_ui(message, "upload_download_failed"))
        return

    caption = message.caption or _ui(message, "chart_caption_default")
    await run_research(message, caption, image_bytes, "image/jpeg")


@dp.message(F.document)
async def handle_chart_document(message: Message):
    document = message.document
    mime_type = document.mime_type or ""
    if mime_type not in ALLOWED_IMAGE_TYPES:
        await message.answer(_ui(message, "unsupported_document"))
        return
    if document.file_size and document.file_size > MAX_CHART_IMAGE_BYTES:
        await message.answer(_ui(message, "upload_too_large"))
        return

    if message.from_user is not None:
        admission_error = _research_admission_error(message.from_user.id, _language_for(message))
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
            await message.answer(_ui(message, "upload_too_large"))
            return
    except UploadTooLargeError:
        await message.answer(_ui(message, "upload_too_large"))
        return
    except Exception:
        await message.answer(_ui(message, "upload_download_failed"))
        return

    caption = message.caption or _ui(message, "chart_caption_default")
    await run_research(message, caption, image_bytes, mime_type)


@dp.message(F.text)
async def handle_text(message: Message):
    await run_research(message, message.text)


@dp.message()
async def unsupported(message: Message):
    await message.answer(_ui(message, "unsupported_message"))


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        if language_preferences is not None:
            language_preferences.close()
        await market.close()
        await crypto_market.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
