import math
import warnings
from io import BytesIO

from PIL import Image, ImageDraw, UnidentifiedImageError


MAX_CHART_IMAGE_PIXELS = 25_000_000
SUPPORTED_IMAGE_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def validate_chart_image(image_bytes: bytes) -> str:
    """Validate an uploaded static chart image and return its detected MIME type.

    The declared Telegram MIME type is not trusted: the encoded file format is
    inspected before any bytes are sent to the vision provider.
    """
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Please send a non-empty JPEG, PNG or WEBP chart image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                frame_count = getattr(image, "n_frames", 1)
                if image_format not in SUPPORTED_IMAGE_MIME_TYPES:
                    raise ValueError("Please send a JPEG, PNG or WEBP chart image.")
                if width < 1 or height < 1 or width * height > MAX_CHART_IMAGE_PIXELS:
                    raise ValueError(
                        "That image has too many pixels. Please resize the chart screenshot and try again."
                    )
                if frame_count != 1:
                    raise ValueError("Animated images are not supported. Please send a single chart frame.")
                image.verify()
    except ValueError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("The upload is not a valid JPEG, PNG or WEBP image. Please resend the chart.") from exc

    return SUPPORTED_IMAGE_MIME_TYPES[image_format]


def _zone_overlay_label(zone: dict, zone_type: str) -> str:
    fallback = "SUPPLY" if zone_type == "supply" else "DEMAND"
    label = zone.get("label")
    if not isinstance(label, str):
        return fallback
    # Pillow's default bitmap font is not Unicode-complete. Keep overlay labels
    # short, printable and English-safe; the full multilingual explanation stays in chat.
    label = " ".join(label.split()).encode("ascii", "ignore").decode("ascii")
    label = "".join(character for character in label if character.isprintable())
    return label[:32].strip().upper() or fallback


def normalized_zone_to_pixel_box(
    zone: dict,
    width: int,
    height: int,
    *,
    require_draw_overlay: bool = True,
    min_width: int = 3,
    min_height: int = 2,
) -> tuple[int, int, int, int] | None:
    """Return the renderer's rounded pixel bounds, or None for undrawable geometry.

    The defaults are the exact production overlay rules. Evaluators may lower
    the minimum size for manual reference annotations, but predictions must
    use these defaults so benchmark scores reflect rectangles actually drawn.
    """
    if not isinstance(zone, dict):
        return None
    if zone.get("zone_type") not in {"supply", "demand"}:
        return None
    if require_draw_overlay and zone.get("draw_overlay") is not True:
        return None
    if (
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        or isinstance(height, bool) or not isinstance(height, int) or height <= 0
        or isinstance(min_width, bool) or not isinstance(min_width, int) or min_width < 1
        or isinstance(min_height, bool) or not isinstance(min_height, int) or min_height < 1
    ):
        return None

    raw_coordinates = [
        zone.get(key) for key in ("x_start", "x_end", "y_top", "y_bottom")
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_coordinates
    ):
        return None
    try:
        x0, x1, y0, y1 = (float(value) for value in raw_coordinates)
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x0, x1, y0, y1)):
        return None
    if x1 <= x0 or y1 <= y0 or y1 - y0 > 0.45:
        return None

    left, right = round(x0 * width), round(x1 * width)
    top, bottom = round(y0 * height), round(y1 * height)
    if right - left < min_width or bottom - top < min_height:
        return None
    return left, top, right, bottom


def render_overlay(
    image_bytes: bytes, zones: list[dict], *, validated: bool = False
) -> tuple[bytes | None, int]:
    """Draw validated, normalized rectangles onto a copy of the uploaded chart.

    validated=True is reserved for callers that have just run validate_chart_image
    on these exact bytes before sending them to AI.
    """
    if not validated:
        validate_chart_image(image_bytes)
    if not isinstance(zones, (list, tuple)):
        return None, 0

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    count = 0

    for zone in zones[:8]:
        if not isinstance(zone, dict) or zone.get("draw_overlay") is not True:
            continue
        zone_type = zone.get("zone_type")
        pixel_box = normalized_zone_to_pixel_box(zone, width, height)
        if pixel_box is None:
            continue
        left, top, right, bottom = pixel_box

        color = (220, 65, 65, 60) if zone_type == "supply" else (30, 170, 100, 60)
        edge = (235, 65, 65, 230) if zone_type == "supply" else (25, 185, 110, 230)
        draw.rectangle((left, top, right, bottom), fill=color, outline=edge, width=max(2, width // 600))
        title = _zone_overlay_label(zone, zone_type)
        draw.text((left + 5, max(0, top - 18)), title, fill=edge[:3] + (255,))
        count += 1

    if not count:
        return None, 0
    result = Image.alpha_composite(image, layer).convert("RGB")
    out = BytesIO()
    result.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue(), count
