"""Best-effort Telegram delivery for StockMind text and chart overlays."""


async def send_analysis_text(message, status, text: str) -> bool:
    """Deliver a completed analysis without confusing transport errors with AI errors.

    Prefer editing the progress message. Fall back to a new reply only when
    Telegram explicitly rejects that edit as impossible. For ambiguous network
    failures, do not send a second copy: Telegram may already have applied the edit.
    """
    if status is None:
        try:
            await message.answer(text)
        except Exception:
            return False
        return True

    try:
        await status.edit_text(text)
    except Exception as exc:
        reason = str(exc).casefold()
        if "message is not modified" in reason:
            # Telegram confirms the requested final text is already present.
            return True

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
            await message.answer(text)
        except Exception:
            return False
        return True

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
