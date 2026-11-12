import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .ai import ResearchAI, ResearchTurnDeadlineExceeded, ResearchResponseIncomplete
from .crypto import BinanceMarketData
from .overlay_engine import prepare_chart_image, _render_overlay_prepared
from .image_upload import BoundedBytesIO, UploadTooLargeError
from .telegram_delivery import send_analysis_text, send_overlay_photo
from .user_locks import (
    UserLockBusy,
    UserLockUnavailable,
    create_user_lock_registry,
    is_lease_lost_cancellation,
)
from .localization import LANGUAGE_NAMES, normalize_language, resolve_language, translate
from .user_preferences import LanguagePreferenceStore
from .usage_store import AdmissionStore
from .conversation_history import ConversationHistoryStore, bound_history
from .config import (
    BETA_USER_LIMIT,
    DAILY_QUERY_LIMIT,
    MAX_CHART_IMAGE_BYTES,
    MAX_HISTORY_MESSAGES,
    MAX_HISTORY_BYTES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    RESEARCH_TURN_TIMEOUT_SECONDS,
    USER_LOCK_WAIT_TIMEOUT_SECONDS,
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

# The deque is a process-local cache; SQLite is authoritative when available.
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
# Serialize same-user requests across processes sharing the persistent local DB.
# The lease exceeds the model-turn deadline and is renewed while a request runs.
try:
    user_locks = create_user_lock_registry(
        USER_PREFERENCES_DB,
        lease_seconds=max(180.0, RESEARCH_TURN_TIMEOUT_SECONDS + 60.0),
        wait_timeout_seconds=USER_LOCK_WAIT_TIMEOUT_SECONDS,
    )
except Exception:
    logger.exception("Persistent StockMind per-user request locking is unavailable")
    user_locks = None
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

try:
    admission_store = AdmissionStore(USER_PREFERENCES_DB)
except Exception:
    logger.exception("Persistent StockMind beta admission and quota storage is unavailable")
    admission_store = None

try:
    conversation_history_store = ConversationHistoryStore(USER_PREFERENCES_DB)
except Exception:
    logger.exception("Persistent StockMind conversation history is unavailable")
    conversation_history_store = None

# A failed per-user history operation falls back to the in-memory cache for the
# rest of this process, rather than silently reloading stale persisted context.
history_storage_unavailable_users = set()


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


def _admit_beta_user(user_id: int) -> bool | None:
    """Persist beta membership; return None when durable storage is unavailable."""
    if admission_store is None:
        return None
    if user_id in beta_users:
        return True
    try:
        admitted = admission_store.admit(user_id, BETA_USER_LIMIT)
    except Exception:
        logger.exception("Could not persist StockMind beta admission")
        return None
    if admitted:
        beta_users.add(user_id)
    return admitted


def _consume_daily_query(user_id: int, today: str) -> bool:
    """Atomically consume one daily slot and refresh the process-local mirror."""
    if admission_store is None:
        return False
    try:
        consumed = admission_store.consume_daily_query(
            user_id, today, DAILY_QUERY_LIMIT
        )
    except Exception:
        logger.exception("Could not persist StockMind daily quota usage")
        return False
    if consumed:
        if usage_day.get(user_id) != today:
            usage_day[user_id] = today
            usage[user_id] = 0
        usage[user_id] += 1
    return consumed


def _research_admission_error(user_id: int, language: str = "en") -> str | None:
    """Return a non-consuming beta/quota/storage rejection before image work."""
    if admission_store is None:
        return translate(language, "quota_storage_unavailable")

    # Local mirrors avoid a database read for users already known to be blocked.
    if user_id not in beta_users and len(beta_users) >= BETA_USER_LIMIT:
        return translate(language, "beta_full")

    today = datetime.now(IST).date().isoformat()
    if usage_day.get(user_id) == today and usage.get(user_id, 0) >= DAILY_QUERY_LIMIT:
        return translate(language, "daily_limit")

    try:
        status = admission_store.check_status(
            user_id, today, BETA_USER_LIMIT, DAILY_QUERY_LIMIT
        )
    except Exception:
        logger.exception("Could not check StockMind beta admission or daily quota")
        return translate(language, "quota_storage_unavailable")
    return translate(language, status) if status else None


def _reserve_research_request(user_id: int, today: str) -> str:
    """Atomically admit the user and consume one slot after image validation."""
    if admission_store is None:
        return "quota_storage_unavailable"
    try:
        status = admission_store.admit_and_consume(
            user_id, today, BETA_USER_LIMIT, DAILY_QUERY_LIMIT
        )
    except Exception:
        logger.exception("Could not reserve a StockMind research request")
        return "quota_storage_unavailable"
    if status == "ok":
        beta_users.add(user_id)
        if usage_day.get(user_id) != today:
            usage_day[user_id] = today
            usage[user_id] = 0
        usage[user_id] += 1
    return status


def _load_conversation_history(user_id: int) -> list[dict]:
    """Load the latest bounded conversation context, preferring durable storage."""
    history = histories[user_id]
    if (
        conversation_history_store is None
        or user_id in history_storage_unavailable_users
    ):
        return list(history)
    try:
        saved = conversation_history_store.get(
            user_id, MAX_HISTORY_MESSAGES, MAX_HISTORY_BYTES
        )
    except Exception:
        logger.exception("Could not load StockMind conversation history")
        history_storage_unavailable_users.add(user_id)
        return list(history)
    history.clear()
    history.extend(saved)
    return list(history)


def _record_conversation_turn(user_id: int, user_text: str, answer: str) -> None:
    """Persist a completed text-only turn and refresh the bounded local cache."""
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    history = histories[user_id]
    bounded = bound_history(
        list(history) + messages,
        MAX_HISTORY_MESSAGES,
        MAX_HISTORY_BYTES,
    )
    if (
        conversation_history_store is not None
        and user_id not in history_storage_unavailable_users
    ):
        try:
            conversation_history_store.append(
                user_id, messages, MAX_HISTORY_MESSAGES, MAX_HISTORY_BYTES
            )
        except Exception:
            logger.exception("Could not persist StockMind conversation history")
            history_storage_unavailable_users.add(user_id)
    history.clear()
    history.extend(bounded)


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

    if user_locks is None:
        await message.answer(translate(_language_for(message), "quota_storage_unavailable"))
        return

    try:
        async with user_locks.hold(user.id):
            await _run_research_locked(message, user_text, image_bytes, image_mime_type)
    except UserLockBusy:
        await message.answer(translate(_language_for(message), "request_busy"))
    except UserLockUnavailable:
        logger.exception("StockMind could not acquire the shared per-user request lock")
        await message.answer(translate(_language_for(message), "quota_storage_unavailable"))
    except asyncio.CancelledError as exc:
        if not is_lease_lost_cancellation(exc):
            raise
        logger.error("StockMind stopped a request after its shared per-user lock lease was lost")
        await message.answer(translate(_language_for(message), "quota_storage_unavailable"))


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

    today = datetime.now(IST).date().isoformat()
    reservation = _reserve_research_request(user_id, today)
    if reservation != "ok":
        await message.answer(translate(language, reservation))
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
    history = _load_conversation_history(user_id)
    try:
        overlay_image = None
        if image_bytes is not None:
            chart_result = await research.analyze_chart_image(
                history, user_text, image_bytes, image_mime_type
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
            answer = await research.answer(history, user_text)

        history_text = user_text or "Please analyse the attached chart screenshot for DNS zones."
        if image_bytes is not None:
            history_text += "\n[Chart screenshot attached; visual analysis was performed.]"
        _record_conversation_turn(user_id, history_text, answer)
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
    except ResearchResponseIncomplete:
        failure_message = translate(language, "research_incomplete")
        if not await send_analysis_text(message, status, failure_message):
            logger.warning("StockMind received an incomplete model response and Telegram could not deliver the warning")
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
    admission = _admit_beta_user(user_id)
    if admission is None:
        await message.answer(translate(language, "quota_storage_unavailable"))
        return
    if not admission:
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
    language = _language_for(message)
    # Wait for an active turn to finish before clearing its conversation context.
    if user_locks is None:
        await message.answer(translate(language, "reset_lock_unavailable"))
        return
    try:
        async with user_locks.hold(user_id):
            history = histories.get(user_id)
            if history is not None:
                history.clear()
            persisted = False
            if conversation_history_store is not None:
                try:
                    conversation_history_store.reset(user_id)
                    history_storage_unavailable_users.discard(user_id)
                    persisted = True
                except Exception:
                    logger.exception("Could not delete saved StockMind conversation history")
                    history_storage_unavailable_users.add(user_id)
            else:
                history_storage_unavailable_users.add(user_id)
    except UserLockBusy:
        await message.answer(translate(language, "reset_lock_busy"))
        return
    except UserLockUnavailable:
        logger.exception("StockMind could not acquire the per-user lock for /reset")
        await message.answer(translate(language, "reset_lock_unavailable"))
        return
    except asyncio.CancelledError as exc:
        if not is_lease_lost_cancellation(exc):
            raise
        logger.error("StockMind stopped /reset after its per-user lock lease was lost")
        await message.answer(translate(language, "reset_lock_unavailable"))
        return
    await message.answer(
        translate(language, "reset_done" if persisted else "reset_storage_failed")
    )


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
        await message.answer(_ui(message, "upload_too_large"))
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
        if admission_store is not None:
            admission_store.close()
        if conversation_history_store is not None:
            conversation_history_store.close()
        if user_locks is not None:
            user_locks.close()
        await market.close()
        await crypto_market.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
