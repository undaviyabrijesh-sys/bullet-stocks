import math
import warnings
from io import BytesIO

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


MAX_CHART_IMAGE_PIXELS = 25_000_000
MAX_OVERLAY_ZONES = 8
MAX_OVERLAY_PNG_BYTES = 9_000_000
SUPPORTED_IMAGE_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ChartImageError(ValueError):
    """A safe, user-facing chart-image validation failure with a stable code."""

    CODES = {"invalid", "unsupported", "too_many_pixels", "animated", "too_large"}

    def __init__(self, code: str, message: str):
        self.code = code if isinstance(code, str) and code in self.CODES else "invalid"
        super().__init__(message)


_ICC_COLOR_SPACE_BY_MODE = {
    "1": b"GRAY",
    "L": b"GRAY",
    "LA": b"GRAY",
    "P": b"RGB ",
    "RGB": b"RGB ",
    "RGBA": b"RGB ",
    "CMYK": b"CMYK",
}


def _compatible_icc_profile(profile: bytes | None, mode: str) -> bytes | None:
    """Return a structurally valid ICC profile only when it matches pixel mode.

    ICC color-space signatures describe channel data, not the container format.
    A profile for grayscale or CMYK samples must not be copied onto RGB pixels
    after Pillow converts the image mode. This is a conservative header check,
    not a color-managed transform or a complete ICC profile parser.
    """
    expected_color_space = _ICC_COLOR_SPACE_BY_MODE.get(mode)
    if expected_color_space is None or not isinstance(profile, bytes) or len(profile) < 128:
        return None
    declared_size = int.from_bytes(profile[:4], "big")
    if (
        declared_size < 128
        or declared_size > len(profile)
        or profile[16:20] != expected_color_space
        or profile[36:40] != b"acsp"
    ):
        return None
    # Ignore bytes beyond the profile's declared extent rather than passing
    # unrelated trailing data into an encoder.
    return profile[:declared_size]


def validate_chart_image(image_bytes: bytes) -> str:
    """Fully decode and integrity-check a static chart image before any AI request.

    The declared Telegram MIME type is not trusted: the encoded file format is
    inspected, every pixel is decoded, and the container integrity check is run
    on a fresh decoder before the bytes are sent to the vision provider.
    """
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ChartImageError("invalid", "Please send a non-empty JPEG, PNG or WEBP chart image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                frame_count = getattr(image, "n_frames", 1)
                if image_format not in SUPPORTED_IMAGE_MIME_TYPES:
                    raise ChartImageError("unsupported", "Please send a JPEG, PNG or WEBP chart image.")
                if width < 1 or height < 1 or width * height > MAX_CHART_IMAGE_PIXELS:
                    raise ChartImageError(
                        "too_many_pixels",
                        "That image has too many pixels. Please resize the chart screenshot and try again.",
                    )
                if frame_count != 1:
                    raise ChartImageError("animated", "Animated images are not supported. Please send a single chart frame.")
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
        raise ChartImageError("invalid", "The upload is not a valid JPEG, PNG or WEBP image. Please resend the chart.") from exc

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

    # Enforce the raw byte budget before Pillow fully decodes the image.
    # This keeps oversized direct callers from spending CPU/memory on a file
    # that will be rejected anyway. Invalid/empty inputs still flow through
    # validate_chart_image so they retain its specific user-facing error.
    if (
        max_output_bytes is not None
        and isinstance(image_bytes, bytes)
        and image_bytes
        and len(image_bytes) > max_output_bytes
    ):
        raise ChartImageError(
            "too_large",
            f"Chart image exceeds the {max_output_bytes}-byte processing limit.",
        )

    mime_type = validate_chart_image(image_bytes)

    with Image.open(BytesIO(image_bytes)) as image:
        orientation = image.getexif().get(274, 1)
        if (
            isinstance(orientation, bool)
            or not isinstance(orientation, int)
            or orientation not in range(1, 9)
        ):
            raise ChartImageError("invalid", "The chart image contains invalid EXIF orientation metadata.")
        if orientation == 1:
            return image_bytes, mime_type

        image_format = image.format
        oriented = ImageOps.exif_transpose(image)
        save_options = {}
        source_profile = image.info.get("icc_profile")

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
            raise ChartImageError("unsupported", "Please send a JPEG, PNG or WEBP chart image.")

        # Check against the final encoded pixel mode, after any conversion.
        # For example, a grayscale/CMYK WebP may be expanded to RGB by Pillow;
        # its original non-RGB profile must not be attached to those RGB bytes.
        icc_profile = _compatible_icc_profile(source_profile, oriented.mode)
        if icc_profile:
            save_options["icc_profile"] = icc_profile

        output = BytesIO()
        oriented.save(output, format=image_format, **save_options)
        normalized_bytes = output.getvalue()

    if max_output_bytes is not None and len(normalized_bytes) > max_output_bytes:
        raise ChartImageError(
            "too_large",
            "Orientation-corrected chart exceeds the "
            f"{max_output_bytes}-byte processing limit. Resize or re-export the screenshot.",
        )

    normalized_mime_type = validate_chart_image(normalized_bytes)
    return normalized_bytes, normalized_mime_type


def _zone_overlay_label(zone: dict, zone_type: str) -> str:
    """Derive a fixed overlay label from validated provenance, never model text.

    The structured chart response deliberately has no free-form label field.
    Keep that invariant at the renderer boundary too: direct callers cannot
    make an inferred zone look user-marked by supplying a forged label.
    Missing or invalid provenance is conservatively treated as AI-inferred.
    """
    safe_zone_type = (
        zone_type.upper()
        if isinstance(zone_type, str) and zone_type in {"supply", "demand"}
        else "DEMAND"
    )
    origin = zone.get("origin") if isinstance(zone, dict) else None
    prefix = "USER" if origin == "user_marked" else "AI"
    return f"{prefix} {safe_zone_type}"


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
        # The production renderer requires explicit high/medium confidence.
        # Legacy/manual geometry-only callers must opt out with
        # require_draw_overlay=False; missing confidence is not drawable.
        confidence = zone.get("confidence")
        if not isinstance(confidence, str) or confidence not in {"high", "medium"}:
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

    with Image.open(BytesIO(image_bytes)) as source:
        # Preserve the source profile on the returned JPEG. The pixel values
        # are composited without a color transform; carrying the profile keeps
        # color-managed viewers from interpreting them in an unrelated space.
        source_format = source.format
        source_profile = source.info.get("icc_profile")
        source_has_alpha = "A" in source.getbands() or "transparency" in source.info
        # The output canvas is RGB/RGBA. Never attach a grayscale/CMYK profile
        # to RGB pixels: that would misdescribe the encoded colors. Preserve
        # only a structurally recognizable RGB profile from an RGB-like source.
        icc_profile = (
            _compatible_icc_profile(source_profile, "RGB")
            if source.mode in {"RGB", "RGBA", "P", "PA"}
            else None
        )
        image = source.convert("RGBA")
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
        confidence = zone.get("confidence")
        if not isinstance(confidence, str) or confidence not in {"high", "medium"}:
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
    composited = Image.alpha_composite(image, layer)

    # Preserve transparency for PNG screenshots. Converting RGBA directly to
    # RGB discards alpha rather than blending it: translucent overlay fills can
    # become fully saturated, and transparent chart backgrounds can turn dark.
    # Keep the original alpha channel in PNG output; JPEG fallback must instead
    # flatten the composited result against an explicit white background.
    if source_format == "PNG":
        png_result = composited if source_has_alpha else composited.convert("RGB")
        # Image operations may carry source info forward. Clear inherited
        # metadata so only the explicitly validated RGB ICC profile is emitted.
        png_result.info.clear()
        png_out = BytesIO()
        png_options = {"format": "PNG", "optimize": True}
        if isinstance(icc_profile, bytes) and icc_profile:
            png_options["icc_profile"] = icc_profile
        png_result.save(png_out, **png_options)
        png_bytes = png_out.getvalue()
        if len(png_bytes) <= MAX_OVERLAY_PNG_BYTES:
            return png_bytes, count

    if source_has_alpha:
        white_background = Image.new("RGBA", composited.size, (255, 255, 255, 255))
        jpeg_result = Image.alpha_composite(white_background, composited).convert("RGB")
    else:
        jpeg_result = composited.convert("RGB")
    jpeg_result.info.clear()

    # JPEG cannot represent transparency. Flatten it deliberately rather than
    # silently dropping alpha and changing the screenshot's visible colors.
    out = BytesIO()
    save_options = {"format": "JPEG", "quality": 88, "optimize": True}
    if isinstance(icc_profile, bytes) and icc_profile:
        save_options["icc_profile"] = icc_profile
    jpeg_result.save(out, **save_options)
    return out.getvalue(), count
