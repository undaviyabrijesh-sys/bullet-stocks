"""Create privacy-conscious, non-benchmark-ready drafts for manual chart annotation.

The intake utility validates screenshots, normalizes EXIF orientation and removes
embedded descriptive metadata before storing them when such metadata is present.
It records a SHA-256 and dimensions for those exact canonical bytes and never
invents zones, predictions, prices, identities or approval timestamps.
"""

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from .overlay_engine import prepare_chart_image

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_GUIDELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CONTEXT_FIELDS = ("asset_class", "symbol", "exchange", "timeframe")
_CONTEXT_LIMITS = {"asset_class": 32, "symbol": 64, "exchange": 64, "timeframe": 16}
_IMAGE_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


# Keep color/format information needed to render the screenshot, but do not carry
# descriptive metadata (EXIF, XMP, IPTC/Photoshop blocks, comments or PNG text)
# into a manually curated research dataset.
_SAFE_IMAGE_INFO_KEYS = {
    "icc_profile", "dpi", "gamma", "chromaticity", "srgb", "transparency",
    "jfif", "jfif_version", "jfif_unit", "jfif_density", "progressive",
    "progression", "adobe", "adobe_transform", "loop", "background",
    "timestamp", "duration",
}


def _has_embedded_descriptive_metadata(image):
    """Return whether Pillow exposes metadata that should not enter the fixture set."""
    try:
        if image.getexif():
            return True
    except (OSError, SyntaxError, TypeError, ValueError) as exc:
        raise ValueError("could not safely inspect screenshot metadata") from exc
    return any(
        str(key).strip().casefold() not in _SAFE_IMAGE_INFO_KEYS
        for key in image.info
    )


def _strip_annotation_metadata(image_bytes, max_image_bytes):
    """Remove descriptive metadata while preserving the canonical visible image.

    Images without descriptive metadata remain byte-for-byte unchanged. Metadata-
    bearing JPEGs are re-encoded at maximum quality with 4:4:4 chroma; WEBP is
    re-encoded losslessly; PNG remains lossless. The resulting bytes—not the
    original upload—are the hash-bound image that must be used for prediction.
    """
    with Image.open(BytesIO(image_bytes)) as image:
        if not _has_embedded_descriptive_metadata(image):
            return image_bytes

        image.load()
        image_format = image.format
        save_options = {}
        icc_profile = image.info.get("icc_profile")
        if icc_profile:
            save_options["icc_profile"] = icc_profile

        if image_format == "JPEG":
            if image.mode not in {"L", "RGB", "CMYK"}:
                image = image.convert("RGB")
            save_options.update({
                "quality": 100,
                "subsampling": 0,
                "optimize": True,
            })
        elif image_format == "WEBP":
            if image.mode not in {"L", "RGB", "RGBA"}:
                image = image.convert("RGB")
            save_options.update({"lossless": True, "method": 6})
        elif image_format == "PNG":
            if image.mode == "P" and "transparency" in image.info:
                save_options["transparency"] = image.info["transparency"]
            save_options["optimize"] = True
        else:
            raise ValueError("Please use a JPEG, PNG or WEBP chart screenshot.")

        # Some Pillow format plugins copy metadata directly from image.info
        # even when it is not passed as a save option (notably WEBP EXIF/XMP).
        for key in tuple(image.info):
            if str(key).strip().casefold() not in _SAFE_IMAGE_INFO_KEYS:
                image.info.pop(key, None)

        output = BytesIO()
        image.save(output, format=image_format, **save_options)
        sanitized_bytes = output.getvalue()

    if len(sanitized_bytes) > max_image_bytes:
        raise ValueError(
            "Metadata-cleaned chart image exceeds max_image_bytes "
            f"({max_image_bytes}); resize or re-export the screenshot."
        )

    with Image.open(BytesIO(sanitized_bytes)) as sanitized:
        sanitized.load()
        if _has_embedded_descriptive_metadata(sanitized):
            raise ValueError("could not remove embedded screenshot metadata safely")

    return sanitized_bytes


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _normalize_market_context(context):
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("market_context must be an object")
    if any(not isinstance(key, str) for key in context):
        raise ValueError("market_context keys must be strings")
    extra = sorted(set(context) - set(_CONTEXT_FIELDS))
    if extra:
        raise ValueError(f"market_context contains unsupported fields: {', '.join(extra)}")

    normalized = {}
    for field in _CONTEXT_FIELDS:
        value = context.get(field)
        if value is None:
            normalized[field] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"market_context.{field} must be a non-empty string or null")
        cleaned = value.strip()
        if len(cleaned) > _CONTEXT_LIMITS[field]:
            raise ValueError(f"market_context.{field} cannot exceed {_CONTEXT_LIMITS[field]} characters")
        if any(not character.isprintable() for character in cleaned):
            raise ValueError(f"market_context.{field} cannot contain control characters")
        normalized[field] = cleaned.upper() if field == "symbol" else cleaned.casefold()
    return normalized


def prepare_annotation_draft(
    image_path,
    output_dir,
    fixture_id,
    *,
    market_context=None,
    guideline_version="dns-manual-v1",
    max_image_bytes=DEFAULT_MAX_IMAGE_BYTES,
):
    """Store one validated, orientation-normalized screenshot and pending draft.

    Images without EXIF rotation are preserved byte-for-byte. When EXIF requires
    a rotation or flip, the visual image is normalized before hashing and saving.
    Existing outputs are never overwritten. If either output write fails, any
    newly created partial files are removed. The draft is not scoreable until it
    is manually annotated, independently reviewed and paired with an AI result.
    """
    max_image_bytes = _positive_integer(max_image_bytes, "max_image_bytes")
    if not isinstance(fixture_id, str) or not _FIXTURE_ID_RE.fullmatch(fixture_id):
        raise ValueError("fixture_id must be a 1-64 character alphanumeric token using . _ or -")
    if not isinstance(guideline_version, str) or not _GUIDELINE_VERSION_RE.fullmatch(guideline_version):
        raise ValueError("guideline_version must be a non-empty version token")
    context = _normalize_market_context(market_context)

    source = Path(image_path).expanduser()
    if not source.is_file():
        raise ValueError("source image does not exist or is not a regular file")
    with source.open("rb") as stream:
        image_bytes = stream.read(max_image_bytes + 1)
    if len(image_bytes) > max_image_bytes:
        raise ValueError(f"source image exceeds max_image_bytes ({max_image_bytes})")
    image_bytes, mime_type = prepare_chart_image(
        image_bytes, max_output_bytes=max_image_bytes
    )
    image_bytes = _strip_annotation_metadata(image_bytes, max_image_bytes)

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size

    output_root = Path(output_dir).expanduser().resolve()
    images_dir = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    resolved_images_dir = images_dir.resolve()
    try:
        resolved_images_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("output images directory must not resolve outside output_dir") from exc

    extension = _IMAGE_EXTENSIONS[mime_type]
    image_name = f"{fixture_id}{extension}"
    destination_image = images_dir / image_name
    destination_draft = output_root / f"{fixture_id}.draft.json"
    if source.resolve() == destination_image.resolve(strict=False):
        raise ValueError("source image already occupies the destination path")
    if destination_image.exists() or destination_draft.exists():
        raise FileExistsError(f"fixture output already exists for {fixture_id!r}; refusing to overwrite")

    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    draft = {
        "draft_schema": "stockmind.annotation-draft.v1",
        "benchmark_ready": False,
        "status": "pending_annotation",
        "fixture_id": fixture_id,
        "image_file": f"images/{image_name}",
        "image_sha256": image_sha256,
        "image_width": width,
        "image_height": height,
        "market_context": context,
        "reference_plot_bounds": None,
        "reference_zones": [],
        "price_axis_anchors": [],
        "price_axis_scale": None,
        "price_axis_plot_bounds": None,
        "annotation_metadata": {
            "source": "manual",
            "annotator_id": None,
            "reviewer_id": None,
            "guideline_version": guideline_version,
            "review_status": "pending",
            "annotated_at": None,
            "reviewed_at": None,
        },
        "prediction_status": "not_captured",
        "predicted_plot_bounds": None,
        "predicted_zones": None,
        "prediction_metadata": None,
    }

    image_created = False
    draft_created = False
    try:
        image_stream = destination_image.open("xb")
        image_created = True
        with image_stream:
            image_stream.write(image_bytes)
        draft_stream = destination_draft.open("x", encoding="utf-8", newline="\n")
        draft_created = True
        with draft_stream:
            json.dump(draft, draft_stream, ensure_ascii=False, indent=2, allow_nan=False)
            draft_stream.write("\n")
    except Exception:
        if draft_created:
            destination_draft.unlink(missing_ok=True)
        if image_created:
            destination_image.unlink(missing_ok=True)
        raise

    return {
        "fixture_id": fixture_id,
        "image_file": f"images/{image_name}",
        "draft_file": destination_draft.name,
        "image_sha256": image_sha256,
        "image_width": width,
        "image_height": height,
        "benchmark_ready": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare an unchanged chart screenshot and pending manual-annotation draft."
    )
    parser.add_argument("image", type=Path, help="source JPEG, PNG or WEBP screenshot")
    parser.add_argument("--output-dir", type=Path, required=True, help="annotation inbox directory")
    parser.add_argument("--fixture-id", required=True, help="pseudonymous fixture ID; do not use a person's name")
    parser.add_argument("--asset-class")
    parser.add_argument("--symbol")
    parser.add_argument("--exchange")
    parser.add_argument("--timeframe")
    parser.add_argument("--guideline-version", default="dns-manual-v1")
    parser.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    args = parser.parse_args(argv)

    context = {
        "asset_class": args.asset_class,
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
    }
    try:
        result = prepare_annotation_draft(
            args.image,
            args.output_dir,
            args.fixture_id,
            market_context=context,
            guideline_version=args.guideline_version,
            max_image_bytes=args.max_image_bytes,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
