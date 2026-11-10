import math
import warnings
from io import BytesIO

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


MAX_CHART_IMAGE_PIXELS = 25_000_000
MAX_OVERLAY_ZONES = 8
SUPPORTED_IMAGE_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def validate_chart_image(image_bytes: bytes) -> str:
    """Fully decode and integrity-check a static chart image before any AI request.

    The declared Telegram MIME type is not trusted: the encoded file format is
    inspected, every pixel is decoded, and the container integrity check is run
    on a fresh decoder before the bytes are sent to the vision provider.
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
                # Image.verify() checks file structure but does not fully decode
                # pixel data. In particular, a truncated JPEG can pass verify().
                # Force the decoder to read every pixel before spending vision tokens.
                image.load()

            # verify() must run on a fresh Image object (Pillow invalidates the
            # decoder after load()). Keep both checks so pixel truncation and
            # malformed container structure are rejected before the API call.
            with Image.open(BytesIO(image_bytes)) as verification_image:
                verification_image.verify()
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


def prepare_chart_image(
    image_bytes: bytes,
    *,
    max_output_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Validate and normalize EXIF orientation before vision and overlay use.

    The returned bytes are the canonical screenshot used by both the model and
    renderer. Images without a rotation/flip tag are returned byte-for-byte
    unchanged. Oriented images are re-encoded without EXIF orientation metadata,
    so provider rendering, Pillow overlays and manual intake share one frame.
    """
    if max_output_bytes is not None and (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 1
    ):
        raise ValueError("max_output_bytes must be a positive integer or None")

    mime_type = validate_chart_image(image_bytes)
    if max_output_bytes is not None and len(image_bytes) > max_output_bytes:
        raise ValueError(
            f"Chart image exceeds the {max_output_bytes}-byte processing limit."
        )

    with Image.open(BytesIO(image_bytes)) as image:
        orientation = image.getexif().get(274, 1)
        if (
            isinstance(orientation, bool)
            or not isinstance(orientation, int)
            or orientation not in range(1, 9)
        ):
            raise ValueError("The chart image contains invalid EXIF orientation metadata.")
        if orientation == 1:
            return image_bytes, mime_type

        image_format = image.format
        oriented = ImageOps.exif_transpose(image)
        save_options = {}
        icc_profile = image.info.get("icc_profile")
        if icc_profile:
            save_options["icc_profile"] = icc_profile

        if image_format == "JPEG":
            if oriented.mode not in {"L", "RGB", "CMYK"}:
                oriented = oriented.convert("RGB")
            save_options.update({"quality": 95, "optimize": True})
        elif image_format == "WEBP":
            if oriented.mode not in {"L", "RGB", "RGBA"}:
                oriented = oriented.convert("RGB")
            save_options.update({"quality": 95, "method": 6})
        elif image_format == "PNG":
            save_options["optimize"] = True
        else:
            raise ValueError("Please send a JPEG, PNG or WEBP chart image.")

        output = BytesIO()
        oriented.save(output, format=image_format, **save_options)
        normalized_bytes = output.getvalue()

    if max_output_bytes is not None and len(normalized_bytes) > max_output_bytes:
        raise ValueError(
            "Orientation-corrected chart exceeds the "
            f"{max_output_bytes}-byte processing limit. Resize or re-export the screenshot."
        )

    normalized_mime_type = validate_chart_image(normalized_bytes)
    return normalized_bytes, normalized_mime_type


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
    zone_type = zone.get("zone_type")
    if not isinstance(zone_type, str) or zone_type not in {"supply", "demand"}:
        return None
    if require_draw_overlay and zone.get("draw_overlay") is not True:
        return None
    if require_draw_overlay:
        # New chart responses carry qualitative confidence. Low or malformed
        # confidence must never reach the renderer; missing confidence remains
        # accepted here only for legacy/offline geometry callers.
        confidence = zone.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, str) or confidence not in {"high", "medium"}
        ):
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

    try:
        left, right = round(x0 * width), round(x1 * width)
        top, bottom = round(y0 * height), round(y1 * height)
    except (OverflowError, ValueError):
        # Public geometry helpers also receive manifest dimensions. Reject
        # unrepresentable scaling rather than leaking an arithmetic exception.
        return None
    if right - left < min_width or bottom - top < min_height:
        return None
    return left, top, right, bottom


def normalized_plot_bounds_to_pixel_box(
    bounds: dict,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Convert a normalized main-price-panel rectangle to half-open pixel bounds."""
    if not isinstance(bounds, dict):
        return None
    if (
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        or isinstance(height, bool) or not isinstance(height, int) or height <= 0
    ):
        return None
    values = [bounds.get(key) for key in ("x_start", "x_end", "y_top", "y_bottom")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    try:
        x0, x1, y0, y1 = (float(value) for value in values)
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x0, x1, y0, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        left, right = round(x0 * width), round(x1 * width)
        top, bottom = round(y0 * height), round(y1 * height)
    except (OverflowError, ValueError):
        # Public geometry helpers also receive manifest dimensions. Reject
        # unrepresentable scaling rather than leaking an arithmetic exception.
        return None
    if right - left < 3 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _pillow_rectangle_bounds(pixel_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert half-open pixel bounds to Pillow's inclusive rectangle endpoints.

    normalized_zone_to_pixel_box returns [left, right) × [top, bottom)
    geometry, matching the evaluator's width/height calculations. Pillow's
    ImageDraw.rectangle includes both endpoints, so right and bottom must be
    decremented to draw exactly the same pixel extent.
    """
    left, top, right, bottom = pixel_box
    return left, top, right - 1, bottom - 1


def render_overlay(
    image_bytes: bytes,
    zones: list[dict],
    *,
    plot_bounds: dict | None = None,
    require_plot_bounds: bool = False,
) -> tuple[bytes | None, int]:
    """Safely prepare an uploaded image, then render its normalized zones.

    This public entry point always validates and normalizes image bytes. Callers
    cannot disable those checks with a boolean flag.
    """
    canonical_image_bytes, _ = prepare_chart_image(image_bytes)
    return _render_overlay_prepared(
        canonical_image_bytes,
        zones,
        plot_bounds=plot_bounds,
        require_plot_bounds=require_plot_bounds,
    )


def _render_overlay_prepared(
    image_bytes: bytes,
    zones: list[dict],
    *,
    plot_bounds: dict | None = None,
    require_plot_bounds: bool = False,
) -> tuple[bytes | None, int]:
    """Render bytes already prepared by prepare_chart_image in this call flow.

    Internal fast path for the Telegram handler: it avoids decoding the same
    screenshot twice after that handler has prepared the exact bytes submitted
    to the vision model. External/direct callers should use render_overlay().
    At most MAX_OVERLAY_ZONES drawable rectangles are emitted; rejected
    candidates do not consume that limit.
    """
    if not isinstance(zones, (list, tuple)):
        return None, 0

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    width, height = image.size
    plot_box = (
        normalized_plot_bounds_to_pixel_box(plot_bounds, width, height)
        if plot_bounds is not None
        else None
    )
    if require_plot_bounds and plot_box is None:
        return None, 0
    if plot_bounds is not None and plot_box is None:
        return None, 0
    if plot_box is None:
        plot_box = (0, 0, width, height)
    plot_left, plot_top, plot_right, plot_bottom = plot_box

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    count = 0

    for zone in zones:
        # The cap applies to successfully drawn overlays, not malformed or
        # out-of-plot candidates that must be skipped.
        if count >= MAX_OVERLAY_ZONES:
            break
        if not isinstance(zone, dict) or zone.get("draw_overlay") is not True:
            continue
        # Treat the renderer as a final trust boundary: direct callers must
        # provide the same explicit confidence required by the v6 sanitizer.
        if zone.get("confidence") not in {"high", "medium"}:
            continue
        zone_type = zone.get("zone_type")
        pixel_box = normalized_zone_to_pixel_box(zone, width, height)
        if pixel_box is None:
            continue
        left, top, right, bottom = pixel_box
        if left < plot_left or top < plot_top or right > plot_right or bottom > plot_bottom:
            continue

        color = (220, 65, 65, 60) if zone_type == "supply" else (30, 170, 100, 60)
        edge = (235, 65, 65, 230) if zone_type == "supply" else (25, 185, 110, 230)
        draw.rectangle(_pillow_rectangle_bounds((left, top, right, bottom)), fill=color, outline=edge, width=max(2, width // 600))
        title = _zone_overlay_label(zone, zone_type)
        text_box = draw.textbbox((0, 0), title)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        if (
            text_width <= plot_right - plot_left - 4
            and text_height <= plot_bottom - plot_top - 4
        ):
            label_x = min(left + 5, max(plot_left + 2, plot_right - text_width - 2))
            label_y = min(max(plot_top, top - 18), plot_bottom - text_height - 2)
            draw.text((label_x, label_y), title, fill=edge[:3] + (255,))
        count += 1

    if not count:
        return None, 0
    result = Image.alpha_composite(image, layer).convert("RGB")
    out = BytesIO()
    result.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue(), count
