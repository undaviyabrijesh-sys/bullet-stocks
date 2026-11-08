"""Offline desktop editor for independent manual DNS screenshot annotations.

The editor never calls a model, market-data provider, Telegram, or network
service. Annotation and review are separate state transitions and always write
a new JSON file rather than modifying the intake draft in place. Failed writes
remove a partially created output, and existing files are never overwritten.
"""

import argparse
import copy
import json
import math
import re
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .overlay_engine import normalized_zone_to_pixel_box
from .price_axis import MAX_AXIS_ANCHORS, calibrate_price_axis
from .strict_json import loads as strict_json_loads

MAX_REFERENCE_ZONES = 8
_ANNOTATOR_ID_RE = re.compile(r"^ann-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_REVIEWER_ID_RE = re.compile(r"^rev-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_GUIDELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BOX_FIELDS = {"x_start", "x_end", "y_top", "y_bottom"}



def fit_editor_image_size(image_width, image_height, screen_width, screen_height):
    """Fit the annotation preview inside the usable desktop area.

    The editor window reserves 40 px horizontally and 220 px vertically for
    window margins, controls and status text. The source image is never
    modified; only its on-screen preview is scaled.
    """
    for field, value in (
        ("image_width", image_width),
        ("image_height", image_height),
        ("screen_width", screen_width),
        ("screen_height", screen_height),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")

    available_width = max(1, screen_width - 80)
    available_height = max(1, screen_height - 220)
    scale = min(
        1.0,
        1200 / image_width,
        760 / image_height,
        available_width / image_width,
        available_height / image_height,
    )
    return (
        max(1, round(image_width * scale)),
        max(1, round(image_height * scale)),
    )


def editor_control_rows(mode, display_width):
    """Return responsive control rows for the offline editor.

    Narrow previews stack the drawing tool and axis controls separately. A
    dedicated view row keeps zoom controls available in both annotation and
    review modes without crowding the save/decision controls.
    """
    if isinstance(display_width, bool) or not isinstance(display_width, int) or display_width <= 0:
        raise ValueError("display_width must be a positive integer")
    if mode == "annotate":
        if display_width < 420:
            return ("identity", "drawing", "axis", "view", "actions")
        return ("identity", "drawing", "view", "actions")
    if mode == "review":
        return ("identity", "view", "actions")
    raise ValueError("mode must be 'annotate' or 'review'")


EDITOR_ZOOM_LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0)


def zoomed_editor_image_size(image_width, image_height, zoom):
    """Return the rendered image size for a bounded, source-relative zoom."""
    for field, value in (("image_width", image_width), ("image_height", image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if isinstance(zoom, bool) or not isinstance(zoom, (int, float)):
        raise ValueError("zoom must be a finite number between 1 and 4")
    try:
        zoom = float(zoom)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("zoom must be a finite number between 1 and 4") from exc
    if not math.isfinite(zoom) or not 1.0 <= zoom <= 4.0:
        raise ValueError("zoom must be a finite number between 1 and 4")
    return max(1, round(image_width * zoom)), max(1, round(image_height * zoom))


def editor_canvas_point_to_normalized(x, y, render_width, render_height):
    """Map scrolled/zoomed canvas coordinates to source-image normalized space."""
    for field, value in (("render_width", render_width), ("render_height", render_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    normalized = []
    for field, value, dimension in (("x", x, render_width), ("y", y, render_height)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a finite number")
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
        normalized.append(min(1.0, max(0.0, value / dimension)))
    return tuple(normalized)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an ISO-8601 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _validate_box(box, field):
    if not isinstance(box, dict) or set(box) != _BOX_FIELDS:
        raise ValueError(f"{field} must contain exactly x_start, x_end, y_top and y_bottom")
    values = {}
    for key in _BOX_FIELDS:
        value = box[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field}.{key} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field}.{key} must be between 0 and 1")
        values[key] = value
    if values["x_start"] >= values["x_end"]:
        raise ValueError(f"{field} must have positive width")
    if values["y_top"] >= values["y_bottom"]:
        raise ValueError(f"{field} must have positive height")
    return values


def validate_manual_geometry(plot_bounds, zones):
    """Validate normalized human-drawn bounds and zones against the screenshot."""
    plot = _validate_box(plot_bounds, "reference_plot_bounds")
    if not isinstance(zones, list):
        raise ValueError("reference_zones must be a list")
    if len(zones) > MAX_REFERENCE_ZONES:
        raise ValueError(f"reference_zones cannot exceed {MAX_REFERENCE_ZONES} entries")

    normalized_zones = []
    for index, zone in enumerate(zones):
        field = f"reference_zones[{index}]"
        if not isinstance(zone, dict) or set(zone) != _BOX_FIELDS | {"zone_type"}:
            raise ValueError(f"{field} must contain zone_type and exactly four normalized coordinates")
        zone_type = zone["zone_type"]
        if zone_type not in {"supply", "demand"}:
            raise ValueError(f"{field}.zone_type must be 'supply' or 'demand'")
        bounds = _validate_box({key: zone[key] for key in _BOX_FIELDS}, field)
        if (
            bounds["x_start"] < plot["x_start"]
            or bounds["x_end"] > plot["x_end"]
            or bounds["y_top"] < plot["y_top"]
            or bounds["y_bottom"] > plot["y_bottom"]
        ):
            raise ValueError(f"{field} must be fully inside reference_plot_bounds")
        normalized_zones.append({"zone_type": zone_type, **bounds})
    return plot, normalized_zones


def validate_reference_zone_pixel_geometry(zones, image_width, image_height):
    """Reject manual boxes that the benchmark cannot score at source resolution."""
    for dimension, value in (("image_width", image_width), ("image_height", image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{dimension} must be a positive integer")

    for index, zone in enumerate(zones):
        pixel_box = normalized_zone_to_pixel_box(
            zone,
            image_width,
            image_height,
            require_draw_overlay=False,
            min_width=1,
            min_height=1,
        )
        if pixel_box is None:
            raise ValueError(
                f"reference_zones[{index}] is not scoreable at the original image resolution; "
                "each zone must span at least one pixel in both dimensions and be no taller "
                "than 45% of the screenshot"
            )


def validate_price_axis_anchor_points(plot_bounds, anchors):
    """Validate manually read price labels without fitting until save time."""
    plot = _validate_box(plot_bounds, "price_axis_plot_bounds")
    if not isinstance(anchors, list):
        raise ValueError("price_axis_anchors must be a list")
    if len(anchors) > MAX_AXIS_ANCHORS:
        raise ValueError(f"price_axis_anchors cannot exceed {MAX_AXIS_ANCHORS} entries")

    normalized = []
    for index, anchor in enumerate(anchors):
        field = f"price_axis_anchors[{index}]"
        if not isinstance(anchor, dict) or set(anchor) != {"y", "price"}:
            raise ValueError(f"{field} must contain exactly y and price")
        point = {}
        for key in ("y", "price"):
            value = anchor[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field}.{key} must be a finite number")
            try:
                value = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{field}.{key} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{field}.{key} must be a finite number")
            point[key] = value
        if not plot["y_top"] <= point["y"] <= plot["y_bottom"]:
            raise ValueError(f"{field}.y must fall inside the marked price plot")
        if point["price"] <= 0:
            raise ValueError(f"{field}.price must be positive")
        normalized.append(point)

    normalized.sort(key=lambda point: point["y"])
    for previous, current in zip(normalized, normalized[1:]):
        if current["y"] - previous["y"] <= 1e-9:
            raise ValueError("price-axis anchors must have distinct y coordinates")
        if current["price"] >= previous["price"]:
            raise ValueError("price must decrease from top to bottom of the chart")
    return normalized


def validate_price_axis_annotations(plot_bounds, anchors, scale):
    """Validate 3..20 manual anchors and reject a distorted linear/log fit."""
    normalized = validate_price_axis_anchor_points(plot_bounds, anchors)
    if not normalized:
        return [], None
    if len(normalized) < 3:
        raise ValueError("price-axis calibration requires at least 3 anchors")
    if not isinstance(scale, str) or scale not in {"linear", "log"}:
        raise ValueError("price_axis_scale must be 'linear' or 'log'")
    calibrate_price_axis(normalized, scale=scale)
    return normalized, scale


def _validate_draft(draft, expected_status):
    if not isinstance(draft, dict):
        raise ValueError("annotation draft must be a JSON object")
    if draft.get("draft_schema") != "stockmind.annotation-draft.v1":
        raise ValueError("unsupported annotation draft schema")
    if draft.get("benchmark_ready") is not False:
        raise ValueError("annotation drafts must remain benchmark_ready=false")
    fixture_id = draft.get("fixture_id")
    if not isinstance(fixture_id, str) or not _FIXTURE_ID_RE.fullmatch(fixture_id):
        raise ValueError("draft fixture_id is invalid")
    image_width = draft.get("image_width")
    image_height = draft.get("image_height")
    if (
        isinstance(image_width, bool) or not isinstance(image_width, int) or image_width <= 0
        or isinstance(image_height, bool) or not isinstance(image_height, int) or image_height <= 0
    ):
        raise ValueError("draft image_width and image_height must be positive integers")
    if draft.get("status") != expected_status:
        raise ValueError(f"draft status must be {expected_status!r}")
    metadata = draft.get("annotation_metadata")
    required = {
        "source", "annotator_id", "reviewer_id", "guideline_version",
        "review_status", "annotated_at", "reviewed_at",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("annotation_metadata does not match the draft schema")
    if metadata.get("source") != "manual":
        raise ValueError("annotation_metadata.source must be 'manual'")
    guideline = metadata.get("guideline_version")
    if not isinstance(guideline, str) or not _GUIDELINE_VERSION_RE.fullmatch(guideline):
        raise ValueError("annotation_metadata.guideline_version is invalid")
    if metadata.get("review_status") != "pending":
        raise ValueError("annotation draft review_status must be 'pending'")
    if (
        draft.get("prediction_status") != "not_captured"
        or draft.get("predicted_plot_bounds") is not None
        or draft.get("predicted_zones") is not None
        or draft.get("prediction_metadata") is not None
    ):
        raise ValueError("annotation editor accepts drafts without AI prediction data only")
    if expected_status == "pending_annotation":
        if any(metadata.get(key) is not None for key in ("annotator_id", "reviewer_id", "annotated_at", "reviewed_at")):
            raise ValueError("unannotated draft must not contain annotation or review provenance")
        if draft.get("reference_plot_bounds") is not None or draft.get("reference_zones") != []:
            raise ValueError("unannotated draft must not contain manual geometry")
        if (
            draft.get("price_axis_anchors", []) not in (None, [])
            or draft.get("price_axis_scale") is not None
            or draft.get("price_axis_plot_bounds") is not None
        ):
            raise ValueError("unannotated draft must not contain price-axis calibration")
    else:
        annotator_id = metadata.get("annotator_id")
        if not isinstance(annotator_id, str) or not _ANNOTATOR_ID_RE.fullmatch(annotator_id):
            raise ValueError("annotation_metadata.annotator_id is invalid")
        _parse_timestamp(metadata.get("annotated_at"), "annotation_metadata.annotated_at")
        if metadata.get("reviewer_id") is not None or metadata.get("reviewed_at") is not None:
            raise ValueError("pending review draft must not contain reviewer provenance")
        plot, normalized_zones = validate_manual_geometry(
            draft.get("reference_plot_bounds"), draft.get("reference_zones")
        )
        validate_reference_zone_pixel_geometry(normalized_zones, image_width, image_height)
        anchors = draft.get("price_axis_anchors", [])
        if anchors is None:
            anchors = []
        if not isinstance(anchors, list):
            raise ValueError("price_axis_anchors must be a list")
        axis_scale = draft.get("price_axis_scale")
        axis_plot = draft.get("price_axis_plot_bounds")
        if anchors:
            validate_price_axis_annotations(plot, anchors, axis_scale)
            if axis_plot is None or _validate_box(axis_plot, "price_axis_plot_bounds") != plot:
                raise ValueError("price_axis_plot_bounds must match the manually marked reference plot")
        elif axis_scale is not None or axis_plot is not None:
            raise ValueError("price-axis scale and plot bounds require at least 3 anchors")
    return metadata


def apply_manual_annotations(
    draft,
    plot_bounds,
    zones,
    annotator_id,
    *,
    annotated_at=None,
    price_axis_anchors=None,
    price_axis_scale=None,
):
    """Return a pending-review copy of an intake draft with validated geometry."""
    _validate_draft(draft, "pending_annotation")
    if not isinstance(annotator_id, str) or not _ANNOTATOR_ID_RE.fullmatch(annotator_id):
        raise ValueError("annotator_id must be an anonymized ID like 'ann-001'")
    plot, normalized_zones = validate_manual_geometry(plot_bounds, zones)
    validate_reference_zone_pixel_geometry(
        normalized_zones, draft["image_width"], draft["image_height"]
    )
    axis_anchors, axis_scale = validate_price_axis_annotations(
        plot, price_axis_anchors if price_axis_anchors is not None else [], price_axis_scale
    )
    timestamp = annotated_at or _utc_now()
    _parse_timestamp(timestamp, "annotated_at")

    result = copy.deepcopy(draft)
    result["reference_plot_bounds"] = plot
    result["reference_zones"] = normalized_zones
    result["price_axis_anchors"] = axis_anchors
    result["price_axis_scale"] = axis_scale
    result["price_axis_plot_bounds"] = copy.deepcopy(plot) if axis_anchors else None
    result["status"] = "pending_review"
    result["benchmark_ready"] = False
    result["annotation_metadata"].update({
        "annotator_id": annotator_id,
        "reviewer_id": None,
        "review_status": "pending",
        "annotated_at": timestamp,
        "reviewed_at": None,
    })
    return result


def review_annotation_draft(draft, reviewer_id, *, approve, reviewed_at=None):
    """Return a separately reviewed draft; annotator and reviewer roles differ."""
    metadata = _validate_draft(draft, "pending_review")
    if not isinstance(reviewer_id, str) or not _REVIEWER_ID_RE.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must be an anonymized ID like 'rev-001'")
    if not isinstance(approve, bool):
        raise ValueError("approve must be a boolean")
    timestamp = reviewed_at or _utc_now()
    reviewed_time = _parse_timestamp(timestamp, "reviewed_at")
    annotated_time = _parse_timestamp(metadata["annotated_at"], "annotated_at")
    if reviewed_time < annotated_time:
        raise ValueError("reviewed_at cannot be earlier than annotated_at")

    result = copy.deepcopy(draft)
    result["status"] = "approved_pending_prediction" if approve else "rejected_annotation"
    result["benchmark_ready"] = False
    result["annotation_metadata"].update({
        "reviewer_id": reviewer_id,
        "review_status": "approved" if approve else "rejected",
        "reviewed_at": timestamp,
    })
    return result


def load_draft_and_image(draft_path, image_path=None):
    """Load a draft and verify its selected screenshot hash and dimensions."""
    draft_path = Path(draft_path).expanduser().resolve()
    with draft_path.open("rb") as stream:
        draft_bytes = stream.read(1024 * 1024 + 1)
    if len(draft_bytes) > 1024 * 1024:
        raise ValueError("annotation draft exceeds the 1 MiB editor limit")
    draft = strict_json_loads(draft_bytes.decode("utf-8"))
    if not isinstance(draft, dict) or draft.get("draft_schema") != "stockmind.annotation-draft.v1":
        raise ValueError("unsupported annotation draft schema")

    if image_path is None:
        relative = PurePosixPath(draft.get("image_file", ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("draft image_file must be a safe relative path")
        image_path = draft_path.parent.joinpath(*relative.parts)
        try:
            image_path.resolve().relative_to(draft_path.parent)
        except ValueError as exc:
            raise ValueError("draft image_file must remain inside the draft directory") from exc
    image_path = Path(image_path).expanduser()
    with image_path.open("rb") as stream:
        image_bytes = stream.read(5 * 1024 * 1024 + 1)
    if len(image_bytes) > 5 * 1024 * 1024:
        raise ValueError("chart screenshot exceeds the 5 MiB annotation editor limit")

    from .overlay_engine import validate_chart_image
    validate_chart_image(image_bytes)
    import hashlib
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest != draft.get("image_sha256"):
        raise ValueError("selected screenshot SHA-256 does not match the annotation draft")
    # Decode the exact bytes whose hash was checked above. Reopening the path
    # here would allow a file replacement between verification and decoding.
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        image_copy = image.copy()
    if width != draft.get("image_width") or height != draft.get("image_height"):
        raise ValueError("selected screenshot dimensions do not match the annotation draft")
    return draft, image_copy


def _write_new_json(path, value):
    """Create a JSON output without leaving a partial file after write failure."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path


class _DesktopEditor:
    """Tkinter UI, imported lazily so validation helpers work on headless hosts."""

    def __init__(self, root, draft, image, mode, output_path):
        import tkinter as tk
        from tkinter import messagebox, simpledialog
        from PIL import ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.root = root
        self.draft = draft
        self.mode = mode
        self.output_path = output_path
        # Only a successful save initiated by this editor session counts.
        # File existence alone is insufficient: another process could create
        # the output after main() performs its initial no-overwrite check.
        self.saved = False
        self.image_width, self.image_height = image.size
        self.plot_bounds = draft.get("reference_plot_bounds")
        self.zones = copy.deepcopy(draft.get("reference_zones", []))
        self.axis_anchors = copy.deepcopy(draft.get("price_axis_anchors", []))
        self.axis_scale = tk.StringVar(value=draft.get("price_axis_scale") or "linear")
        self.tool = tk.StringVar(value="plot" if self.plot_bounds is None else "demand")
        self.identity = tk.StringVar()
        self.drag_start = None
        self.preview_id = None
        self.undo_stack = []
        self.source_image = image.copy()
        self.zoom_index = 0
        self.zoom_factor = EDITOR_ZOOM_LEVELS[self.zoom_index]
        self.zoom_label = tk.StringVar(value="100%")

        self.display_w, self.display_h = fit_editor_image_size(
            image.width,
            image.height,
            root.winfo_screenwidth(),
            root.winfo_screenheight(),
        )
        self.render_w, self.render_h = self.display_w, self.display_h
        display_image = self.source_image.convert("RGB").resize(
            (self.render_w, self.render_h), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(display_image)

        root.title(f"StockMind annotation — {draft['fixture_id']} — {mode}")
        root.geometry(f"{self.display_w + 40}x{self.display_h + 220}")
        controls = tk.Frame(root, padx=8, pady=8)
        controls.pack(fill="x")
        rows = {}
        for index, row_name in enumerate(editor_control_rows(mode, self.display_w)):
            row = tk.Frame(controls)
            row.pack(fill="x", pady=(0, 4) if index else 0)
            rows[row_name] = row

        identity_row = rows["identity"]
        tk.Label(identity_row, text="Anonymized ID:").pack(side="left")
        tk.Entry(identity_row, textvariable=self.identity, width=18).pack(side="left", padx=(4, 12))

        if mode == "annotate":
            drawing_row = rows["drawing"]
            axis_row = rows.get("axis", drawing_row)
            action_row = rows["actions"]
            tk.Label(drawing_row, text="Draw:").pack(side="left")
            tk.OptionMenu(drawing_row, self.tool, "plot", "demand", "supply", "axis_anchor").pack(side="left")
            tk.Label(axis_row, text="Axis scale:").pack(side="left", padx=(8, 2))
            tk.OptionMenu(axis_row, self.axis_scale, "linear", "log").pack(side="left")
            tk.Button(action_row, text="Undo", command=self.undo).pack(side="left", padx=6)
            tk.Button(action_row, text="Save annotation draft", command=self.save_annotation).pack(side="left", padx=6)
        else:
            action_row = rows["actions"]
            if self.axis_anchors:
                tk.Label(action_row, text=f"Price-axis calibration: {self.axis_scale.get()} ({len(self.axis_anchors)} anchors)").pack(side="left", padx=6)
            tk.Button(action_row, text="Approve", command=lambda: self.save_review(True)).pack(side="left", padx=6)
            tk.Button(action_row, text="Reject", command=lambda: self.save_review(False)).pack(side="left", padx=6)

        view_row = rows["view"]
        tk.Label(view_row, text="Zoom:").pack(side="left", padx=(0, 4))
        tk.Button(view_row, text="−", command=self.zoom_out, width=3).pack(side="left", padx=2)
        tk.Label(view_row, textvariable=self.zoom_label, width=6, anchor="center").pack(side="left", padx=2)
        tk.Button(view_row, text="+", command=self.zoom_in, width=3).pack(side="left", padx=2)
        tk.Button(view_row, text="Reset", command=self.reset_zoom).pack(side="left", padx=4)

        self.status = tk.StringVar(value=self._instructions())
        tk.Label(root, textvariable=self.status, anchor="w", padx=8).pack(fill="x")
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.display_w,
            height=self.display_h,
            highlightthickness=0,
            cursor="crosshair",
        )
        vertical_scrollbar = tk.Scrollbar(
            canvas_frame, orient="vertical", command=self.canvas.yview
        )
        horizontal_scrollbar = tk.Scrollbar(
            canvas_frame, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(
            xscrollcommand=horizontal_scrollbar.set,
            yscrollcommand=vertical_scrollbar.set,
            scrollregion=(0, 0, self.render_w, self.render_h),
        )
        vertical_scrollbar.pack(side="right", fill="y")
        horizontal_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self._redraw()
        if mode == "annotate":
            self.canvas.bind("<ButtonPress-1>", self._start_draw)
            self.canvas.bind("<B1-Motion>", self._move_draw)
            self.canvas.bind("<ButtonRelease-1>", self._finish_draw)
        else:
            self.canvas.configure(cursor="arrow")

    def _instructions(self):
        if self.mode == "annotate":
            return (
                "Draw the price-plot boundary, then supply/demand bands. "
                "Optional: choose axis_anchor, click 3..20 price levels inside the plot, "
                "and enter their exact visible labels. Use zoom and scrollbars for precise edges."
            )
        return (
            "Review the screenshot and manual price anchors. Use zoom and scrollbars "
            "to inspect boundaries, and a separate reviewer ID."
        )

    def _coords(self, box):
        return (box["x_start"] * self.render_w, box["y_top"] * self.render_h,
                box["x_end"] * self.render_w, box["y_bottom"] * self.render_h)

    def _redraw(self):
        self.canvas.delete("overlay")
        if self.plot_bounds:
            self.canvas.create_rectangle(*self._coords(self.plot_bounds), outline="#3B82F6", width=2, dash=(7, 4), tags="overlay")
        for zone in self.zones:
            color = "#DC2626" if zone["zone_type"] == "supply" else "#16A34A"
            self.canvas.create_rectangle(*self._coords(zone), outline=color, width=2, stipple="gray25", tags="overlay")
            x1, y1, _, _ = self._coords(zone)
            self.canvas.create_text(x1 + 5, y1 + 5, text=zone["zone_type"].upper(), fill=color, anchor="nw", tags="overlay")
        if self.plot_bounds:
            plot_x1, _, plot_x2, _ = self._coords(self.plot_bounds)
            for anchor in self.axis_anchors:
                y = anchor["y"] * self.render_h
                self.canvas.create_line(plot_x1, y, plot_x2, y, fill="#F59E0B", dash=(3, 3), width=1, tags="overlay")
                self.canvas.create_oval(plot_x2 - 4, y - 4, plot_x2 + 4, y + 4, outline="#F59E0B", fill="#F59E0B", tags="overlay")
                self.canvas.create_text(plot_x2 - 6, y - 5, text=f"{anchor['price']:g}", fill="#F59E0B", anchor="se", tags="overlay")

    def _canvas_point(self, event):
        """Convert viewport event coordinates into the scrolled image canvas."""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return (
            max(0.0, min(float(self.render_w), float(x))),
            max(0.0, min(float(self.render_h), float(y))),
        )

    def _set_zoom(self, index):
        if self.drag_start is not None:
            self.status.set("Finish or cancel the current drawing before changing zoom.")
            return
        index = max(0, min(len(EDITOR_ZOOM_LEVELS) - 1, index))
        if index == self.zoom_index:
            return

        viewport_width = max(1, self.canvas.winfo_width())
        viewport_height = max(1, self.canvas.winfo_height())
        center_x = self.canvas.canvasx(viewport_width / 2) / self.render_w
        center_y = self.canvas.canvasy(viewport_height / 2) / self.render_h

        self.zoom_index = index
        self.zoom_factor = EDITOR_ZOOM_LEVELS[index]
        self.render_w, self.render_h = zoomed_editor_image_size(
            self.display_w, self.display_h, self.zoom_factor
        )
        resized = self.source_image.convert("RGB").resize(
            (self.render_w, self.render_h), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, self.render_w, self.render_h))
        self.zoom_label.set(f"{self.zoom_factor:.0%}")

        self.canvas.xview_moveto(
            max(0.0, min(1.0, center_x - viewport_width / (2 * self.render_w)))
        )
        self.canvas.yview_moveto(
            max(0.0, min(1.0, center_y - viewport_height / (2 * self.render_h)))
        )
        self._redraw()

    def zoom_in(self):
        self._set_zoom(self.zoom_index + 1)

    def zoom_out(self):
        self._set_zoom(self.zoom_index - 1)

    def reset_zoom(self):
        self._set_zoom(0)

    def _start_draw(self, event):
        self.drag_start = self._canvas_point(event)
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
        if self.tool.get() == "axis_anchor":
            y = self.drag_start[1]
            self.preview_id = self.canvas.create_line(0, y, self.render_w, y, fill="#F59E0B", dash=(3, 3))
        else:
            self.preview_id = self.canvas.create_rectangle(*self.drag_start, *self.drag_start, outline="#F59E0B", width=2, dash=(3, 3))

    def _move_draw(self, event):
        if self.drag_start is None or self.preview_id is None:
            return
        x, y = self._canvas_point(event)
        if self.tool.get() == "axis_anchor":
            self.canvas.coords(self.preview_id, 0, y, self.render_w, y)
        else:
            self.canvas.coords(self.preview_id, self.drag_start[0], self.drag_start[1], x, y)

    def _finish_draw(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._canvas_point(event)
        self.drag_start = None
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None
        if self.tool.get() == "axis_anchor":
            if self.plot_bounds is None:
                self.status.set("Draw the price-plot boundary before adding price-axis anchors.")
                return
            _, y = editor_canvas_point_to_normalized(
                0, y1, self.render_w, self.render_h
            )
            if not self.plot_bounds["y_top"] <= y <= self.plot_bounds["y_bottom"]:
                self.status.set("Price-axis anchors must be placed inside the main price plot.")
                return
            raw_price = self.simpledialog.askstring(
                "Manual price-axis anchor",
                "Enter the exact visible price label aligned with this horizontal level (no commas):",
                parent=self.root,
            )
            if raw_price is None:
                self.status.set(self._instructions())
                return
            try:
                price = float(raw_price.strip())
                anchor = {"y": y, "price": price}
                normalized = validate_price_axis_anchor_points(
                    self.plot_bounds, self.axis_anchors + [anchor]
                )
            except (AttributeError, ValueError) as exc:
                self.status.set(str(exc))
                return
            self.axis_anchors = normalized
            self.undo_stack.append(("axis_anchor", anchor))
            self.status.set(
                f"Added manual price anchor {price:g}; {len(self.axis_anchors)} recorded (3 minimum for scoring)."
            )
            self._redraw()
            return

        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 2 or bottom - top < 2:
            self.status.set("Rectangle too small; drag a larger area.")
            return
        x_start, y_top = editor_canvas_point_to_normalized(
            left, top, self.render_w, self.render_h
        )
        x_end, y_bottom = editor_canvas_point_to_normalized(
            right, bottom, self.render_w, self.render_h
        )
        box = {
            "x_start": x_start,
            "x_end": x_end,
            "y_top": y_top,
            "y_bottom": y_bottom,
        }
        if self.tool.get() == "plot":
            try:
                validate_manual_geometry(box, self.zones)
                validate_price_axis_anchor_points(box, self.axis_anchors)
            except ValueError as exc:
                self.status.set(str(exc))
                return
            self.undo_stack.append(("plot", copy.deepcopy(self.plot_bounds)))
            self.plot_bounds = box
        else:
            if self.plot_bounds is None:
                self.status.set("Draw the price-plot boundary before adding zones.")
                return
            zone = {"zone_type": self.tool.get(), **box}
            try:
                _, candidate_zones = validate_manual_geometry(
                    self.plot_bounds, self.zones + [zone]
                )
                validate_reference_zone_pixel_geometry(
                    candidate_zones, self.image_width, self.image_height
                )
            except ValueError as exc:
                self.status.set(str(exc))
                return
            self.zones.append(zone)
            self.undo_stack.append(("zone", zone))
        self.status.set(self._instructions())
        self._redraw()

    def undo(self):
        if not self.undo_stack:
            self.status.set("No new drawing action to undo.")
            return
        action, value = self.undo_stack.pop()
        if action == "zone":
            self.zones.remove(value)
        elif action == "axis_anchor":
            self.axis_anchors.remove(value)
        elif action == "plot":
            self.plot_bounds = value
        self._redraw()
        self.status.set(self._instructions())

    def save_annotation(self):
        try:
            updated = apply_manual_annotations(
                self.draft,
                self.plot_bounds,
                self.zones,
                self.identity.get().strip(),
                price_axis_anchors=self.axis_anchors,
                price_axis_scale=self.axis_scale.get(),
            )
            _write_new_json(self.output_path, updated)
            self.saved = True
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Cannot save annotation", str(exc))
            return
        self.messagebox.showinfo("Annotation saved", f"Pending-review draft saved to:\n{self.output_path}")
        self.root.destroy()

    def save_review(self, approve):
        try:
            updated = review_annotation_draft(
                self.draft, self.identity.get().strip(), approve=approve
            )
            _write_new_json(self.output_path, updated)
            self.saved = True
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Cannot save review", str(exc))
            return
        decision = "approved" if approve else "rejected"
        self.messagebox.showinfo("Review saved", f"Annotation {decision}.\nSaved to:\n{self.output_path}")
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open an offline StockMind screenshot annotation/review editor.")
    parser.add_argument("draft", type=Path, help="annotation draft JSON")
    parser.add_argument("--mode", choices=("annotate", "review"), required=True)
    parser.add_argument("--image", type=Path, help="optional screenshot path; defaults to draft.image_file")
    parser.add_argument("--output", type=Path, help="new output JSON path; input is never overwritten")
    args = parser.parse_args(argv)

    try:
        draft, image = load_draft_and_image(args.draft, args.image)
        expected = "pending_annotation" if args.mode == "annotate" else "pending_review"
        _validate_draft(draft, expected)
        suffix = ".annotated.json" if args.mode == "annotate" else ".reviewed.json"
        output = args.output or args.draft.with_name(args.draft.stem + suffix)
        if output.exists():
            raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
        import tkinter as tk
        root = tk.Tk()
        editor = _DesktopEditor(root, draft, image, args.mode, output)
        root.mainloop()
        if not editor.saved:
            print(
                "Annotation editor closed without saving; this session did not create an output.",
                file=sys.stderr,
            )
            return 1
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"could not start desktop editor ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
