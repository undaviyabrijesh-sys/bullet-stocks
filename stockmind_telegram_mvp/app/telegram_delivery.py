"""Best-effort Telegram delivery for StockMind text and chart overlays."""

TELEGRAM_TEXT_CHUNK_LIMIT = 4000


def split_telegram_text(text: str, *, limit: int = TELEGRAM_TEXT_CHUNK_LIMIT) -> list[str]:
    """Split text below Telegram's message limit without dropping any characters.

    Prefer paragraph, line and word boundaries. Long unbroken tokens fall back
    to a hard split. A conservative 4,000-character default leaves headroom
    below Telegram's 4,096-character text limit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        boundary = remaining.rfind("\n\n", 0, limit)
        separator_length = 2
        if boundary < limit // 2:
            boundary = remaining.rfind("\n", 0, limit)
            separator_length = 1
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit)
            separator_length = 1
        if boundary < limit // 2:
            boundary = limit
            separator_length = 0

        end = boundary + separator_length
        chunks.append(remaining[:end])
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


async def send_overlay_photo(message, image_bytes: bytes, *, caption: str) -> bool:
    """Send an already-rendered overlay without affecting the written analysis.

    Telegram photo delivery is a secondary presentation step. A transport/API
    failure is reported to the caller as False and must not invalidate the
    chart analysis that has already been sent as text.
    """
    from aiogram.types import BufferedInputFile

    try:
        await message.answer_photo(
            BufferedInputFile(image_bytes, filename="stockmind-chart-zones.jpg"),
            caption=caption,
        )
    except Exception:
        return False
    return True
