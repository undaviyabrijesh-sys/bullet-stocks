"""Best-effort Telegram delivery for StockMind text and chart overlays."""

import warnings

TELEGRAM_TEXT_CHUNK_LIMIT = 4000

# Keep a margin below Telegram's documented 10 MB photo ceiling. Oversized
# or unusually shaped overlays are sent as documents instead of being resized.
TELEGRAM_PHOTO_MAX_BYTES = 9_000_000
TELEGRAM_DOCUMENT_MAX_BYTES = 45_000_000
# Match the screenshot validator's decoded-pixel ceiling, including direct callers.
TELEGRAM_OVERLAY_MAX_PIXELS = 25_000_000
TELEGRAM_PHOTO_MAX_DIMENSION_SUM = 10_000
TELEGRAM_PHOTO_MAX_ASPECT_RATIO = 20


def _telegram_text_units(text: str) -> int:
    """Count UTF-16 code units, the coordinate unit used by Telegram entities."""
    return sum(2 if ord(character) > 0xFFFF else 1 for character in text)


def _prefix_end_within_limit(text: str, limit: int) -> int:
    """Return a character index whose prefix fits the UTF-16 unit budget."""
    units = 0
    for index, character in enumerate(text):
        units += 2 if ord(character) > 0xFFFF else 1
        if units > limit:
            return index
    return len(text)


def split_telegram_text(text: str, *, limit: int = TELEGRAM_TEXT_CHUNK_LIMIT) -> list[str]:
    """Split text below Telegram's message limit without dropping any characters.

    Telegram entity offsets use UTF-16 code units, so supplementary Unicode
    characters (including many emoji) count as two. Prefer paragraph, line and
    word boundaries. Long unbroken tokens fall back to a hard split without
    cutting a Unicode code point. A 4,000-unit default leaves headroom below
    Telegram's 4,096-character text limit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    remaining_units = _telegram_text_units(text)
    if remaining_units <= limit:
        return [text]

    chunks = []
    remaining = text
    while remaining_units > limit:
        max_end = _prefix_end_within_limit(remaining, limit)
        if max_end == 0:
            raise ValueError("limit is too small for the next Unicode character")
        preferred_floor = _prefix_end_within_limit(remaining, max(1, limit // 2))

        boundary = remaining.rfind("\n\n", 0, max_end)
        separator_length = 2
        if boundary < preferred_floor:
            boundary = remaining.rfind("\n", 0, max_end)
            separator_length = 1
        if boundary < preferred_floor:
            boundary = remaining.rfind(" ", 0, max_end)
            separator_length = 1
        if boundary < preferred_floor:
            boundary = max_end
            separator_length = 0

        end = boundary + separator_length
        chunk = remaining[:end]
        # Keep a running UTF-16 budget. Recounting the entire remaining suffix
        # on every iteration makes many-chunk responses quadratic in length.
        remaining_units -= _telegram_text_units(chunk)
        chunks.append(chunk)
        remaining = remaining[end:]

    if remaining:
        chunks.append(remaining)
    return chunks

async def send_analysis_text(message, status, text: str) -> bool:
    """Deliver a completed analysis without confusing transport errors with AI errors.

    Prefer editing the progress message. Fall back to a new reply only when
    Telegram explicitly rejects that edit as impossible. For ambiguous network
    failures, do not send a second copy: Telegram may already have applied the edit.
    """
    # Telegram rejects empty sendMessage text. Treat an empty model result as
    # an undelivered response without issuing a doomed request or editing status.
    if text == "":
        return False

    chunks = split_telegram_text(text)
    first_chunk = chunks[0]

    if status is None:
        try:
            await message.answer(first_chunk)
        except Exception:
            return False
    else:
        try:
            await status.edit_text(first_chunk)
        except Exception as exc:
            reason = str(exc).casefold()
            if "message is not modified" in reason:
                # Telegram confirms the requested final text is already present.
                pass
            else:
                recoverable_edit_errors = (
                    "message to edit not found",
                    "message can't be edited",
                    "message identifier is not specified",
                    "message_id_invalid",
                )
                if not any(marker in reason for marker in recoverable_edit_errors):
                    # The request may have reached Telegram before the connection failed.
                    # Avoid a duplicate final answer when delivery state is ambiguous.
                    return False

                try:
                    await message.answer(first_chunk)
                except Exception:
                    return False

    for chunk in chunks[1:]:
        try:
            await message.answer(chunk)
        except Exception:
            # Do not retry: Telegram may have accepted a chunk before a transport
            # error was raised. The caller keeps the complete answer in history.
            return False
    return True


def _fits_telegram_photo_limits(width: int, height: int, byte_count: int) -> bool:
    """Return whether an image satisfies Telegram's sendPhoto constraints."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (width, height, byte_count)
    ):
        return False
    if width < 1 or height < 1 or byte_count < 1:
        return False
    if byte_count > TELEGRAM_PHOTO_MAX_BYTES:
        return False
    if width + height > TELEGRAM_PHOTO_MAX_DIMENSION_SUM:
        return False
    return max(width / height, height / width) <= TELEGRAM_PHOTO_MAX_ASPECT_RATIO


async def send_overlay_photo(message, image_bytes: bytes, *, caption: str) -> bool:
    """Send an already-rendered overlay without affecting the written analysis.

    Telegram photo delivery is a secondary presentation step. A transport/API
    failure is reported to the caller as False and must not invalidate the
    chart analysis that has already been sent as text.
    """
    from io import BytesIO

    from aiogram.types import BufferedInputFile
    from PIL import Image

    if (
        not isinstance(image_bytes, bytes)
        or not image_bytes
        or len(image_bytes) > TELEGRAM_DOCUMENT_MAX_BYTES
    ):
        return False

    try:
        # Verify the complete decoded image, not only its container header.
        # Pillow's verify() alone can accept truncated JPEG pixel data. Decode
        # one frame first, then verify the container on a fresh decoder.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                frame_count = getattr(image, "n_frames", 1)
                if image_format not in {"PNG", "JPEG"} or frame_count != 1:
                    return False
                if (
                    width < 1
                    or height < 1
                    or width * height > TELEGRAM_OVERLAY_MAX_PIXELS
                ):
                    return False
                image.load()

            with Image.open(BytesIO(image_bytes)) as verification_image:
                verification_image.verify()
    except Exception:
        return False

    filename = (
        "stockmind-chart-zones.png"
        if image_format == "PNG"
        else "stockmind-chart-zones.jpg"
    )
    upload = BufferedInputFile(image_bytes, filename=filename)
    try:
        if _fits_telegram_photo_limits(width, height, len(image_bytes)):
            await message.answer_photo(upload, caption=caption)
        else:
            # A document preserves the exact rendered pixels. Do not retry as a
            # document after a photo transport error: the photo may have arrived.
            await message.answer_document(upload, caption=caption)
    except Exception:
        return False
    return True
