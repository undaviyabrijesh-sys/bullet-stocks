"""Best-effort Telegram delivery for optional chart overlay images."""

from aiogram.types import BufferedInputFile


async def send_overlay_photo(message, image_bytes: bytes, *, caption: str) -> bool:
    """Send an already-rendered overlay without affecting the written analysis.

    Telegram photo delivery is a secondary presentation step. A transport/API
    failure is reported to the caller as False and must not invalidate the
    chart analysis that has already been sent as text.
    """
    try:
        await message.answer_photo(
            BufferedInputFile(image_bytes, filename="stockmind-chart-zones.jpg"),
            caption=caption,
        )
    except Exception:
        return False
    return True
