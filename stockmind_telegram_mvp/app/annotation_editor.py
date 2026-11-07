"""Offline desktop editor for independent manual DNS screenshot annotations.

The editor never calls a model, market-data provider, Telegram, or network
service. Annotation and review are separate state transitions and always write
a new JSON file rather than modifying the intake draft in place.
"""

import argparse
import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

MAX_REFERENCE_ZONES = 8
_ANNOTATOR_ID_RE = re.compile(r"^ann-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_REVIEWER_ID_RE = re.compile(r"^rev-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_GUIDELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BOX_FIELDS = {"x_start", "x_end", "y_top", "y_bottom"}


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
    if expected_status == "pending_annotation":
        if any(metadata.get(key) is not None for key in ("annotator_id", "reviewer_id", "annotated_at", "reviewed_at")):
            raise ValueError("unannotated draft must not contain annotation or review provenance")
    else:
        annotator_id = metadata.get("annotator_id")
        if not isinstance(annotator_id, str) or not _ANNOTATOR_ID_RE.fullmatch(annotator_id):
            raise ValueError("annotation_metadata.annotator_id is invalid")
        _parse_timestamp(metadata.get("annotated_at"), "annotation_metadata.annotated_at")
        if metadata.get("reviewer_id") is not None or metadata.get("reviewed_at") is not None:
            raise ValueError("pending review draft must not contain reviewer provenance")
        validate_manual_geometry(draft.get("reference_plot_bounds"), draft.get("reference_zones"))
    return metadata


def apply_manual_annotations(draft, plot_bounds, zones, annotator_id, *, annotated_at=None):
    """Return a pending-review copy of an intake draft with validated geometry."""
    _validate_draft(draft, "pending_annotation")
    if not isinstance(annotator_id, str) or not _ANNOTATOR_ID_RE.fullmatch(annotator_id):
        raise ValueError("annotator_id must be an anonymized ID like 'ann-001'")
    plot, normalized_zones = validate_manual_geometry(plot_bounds, zones)
    timestamp = annotated_at or _utc_now()
    _parse_timestamp(timestamp, "annotated_at")

    result = copy.deepcopy(draft)
    result["reference_plot_bounds"] = plot
    result["reference_zones"] = normalized_zones
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
    draft = json.loads(draft_bytes.decode("utf-8"))
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
    with Image.open(image_path) as image:
        width, height = image.size
        image_copy = image.copy()
    if width != draft.get("image_width") or height != draft.get("image_height"):
        raise ValueError("selected screenshot dimensions do not match the annotation draft")
    return draft, image_copy


def _write_new_json(path, value):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


class _DesktopEditor:
    """Tkinter UI, imported lazily so validation helpers work on headless hosts."""

    def __init__(self, root, draft, image, mode, output_path):
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.root = root
        self.draft = draft
        self.mode = mode
        self.output_path = output_path
        self.plot_bounds = draft.get("reference_plot_bounds")
        self.zones = copy.deepcopy(draft.get("reference_zones", []))
        self.tool = tk.StringVar(value="plot" if self.plot_bounds is None else "demand")
        self.identity = tk.StringVar()
        self.drag_start = None
        self.preview_id = None

        screen_w = max(800, root.winfo_screenwidth() - 100)
        screen_h = max(600, root.winfo_screenheight() - 180)
        scale = min(1.0, 1200 / image.width, 760 / image.height, screen_w / image.width, screen_h / image.height)
        self.display_w = max(1, round(image.width * scale))
        self.display_h = max(1, round(image.height * scale))
        display_image = image.convert("RGB").resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_image)

        root.title(f"StockMind annotation — {draft['fixture_id']} — {mode}")
        root.geometry(f"{self.display_w + 40}x{min(self.display_h + 180, screen_h + 180)}")
        controls = tk.Frame(root, padx=8, pady=8)
        controls.pack(fill="x")
        tk.Label(controls, text="Anonymized ID:").pack(side="left")
        tk.Entry(controls, textvariable=self.identity, width=18).pack(side="left", padx=(4, 12))

        if mode == "annotate":
            tk.Label(controls, text="Draw:").pack(side="left")
            tk.OptionMenu(controls, self.tool, "plot", "demand", "supply").pack(side="left")
            tk.Button(controls, text="Undo", command=self.undo).pack(side="left", padx=6)
            tk.Button(controls, text="Save annotation draft", command=self.save_annotation).pack(side="left", padx=6)
        else:
            tk.Button(controls, text="Approve", command=lambda: self.save_review(True)).pack(side="left", padx=6)
            tk.Button(controls, text="Reject", command=lambda: self.save_review(False)).pack(side="left", padx=6)

        self.status = tk.StringVar(value=self._instructions())
        tk.Label(root, textvariable=self.status, anchor="w", padx=8).pack(fill="x")
        self.canvas = tk.Canvas(root, width=self.display_w, height=self.display_h, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self._redraw()
        if mode == "annotate":
            self.canvas.bind("<ButtonPress-1>", self._start_draw)
            self.canvas.bind("<B1-Motion>", self._move_draw)
            self.canvas.bind("<ButtonRelease-1>", self._finish_draw)
        else:
            self.canvas.configure(cursor="arrow")

    def _instructions(self):
        if self.mode == "annotate":
            return "Draw the price-plot boundary first; then draw each visible supply or demand band. Empty zone lists are allowed."
        return "Review the original screenshot and manual rectangles. Approval requires a separate reviewer ID."

    def _coords(self, box):
        return (box["x_start"] * self.display_w, box["y_top"] * self.display_h,
                box["x_end"] * self.display_w, box["y_bottom"] * self.display_h)

    def _redraw(self):
        self.canvas.delete("overlay")
        if self.plot_bounds:
            self.canvas.create_rectangle(*self._coords(self.plot_bounds), outline="#3B82F6", width=2, dash=(7, 4), tags="overlay")
        for zone in self.zones:
            color = "#DC2626" if zone["zone_type"] == "supply" else "#16A34A"
            self.canvas.create_rectangle(*self._coords(zone), outline=color, width=2, stipple="gray25", tags="overlay")
            x1, y1, _, _ = self._coords(zone)
            self.canvas.create_text(x1 + 5, y1 + 5, text=zone["zone_type"].upper(), fill=color, anchor="nw", tags="overlay")

    def _start_draw(self, event):
        self.drag_start = (max(0, min(self.display_w, event.x)), max(0, min(self.display_h, event.y)))
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
        self.preview_id = self.canvas.create_rectangle(*self.drag_start, *self.drag_start, outline="#F59E0B", width=2, dash=(3, 3))

    def _move_draw(self, event):
        if self.drag_start is None or self.preview_id is None:
            return
        x = max(0, min(self.display_w, event.x))
        y = max(0, min(self.display_h, event.y))
        self.canvas.coords(self.preview_id, self.drag_start[0], self.drag_start[1], x, y)

    def _finish_draw(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1 = max(0, min(self.display_w, event.x))
        y1 = max(0, min(self.display_h, event.y))
        self.drag_start = None
        if self.preview_id is not None:
            self.canvas.delete(self.preview_id)
            self.preview_id = None
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 2 or bottom - top < 2:
            self.status.set("Rectangle too small; drag a larger area.")
            return
        box = {
            "x_start": left / self.display_w,
            "x_end": right / self.display_w,
            "y_top": top / self.display_h,
            "y_bottom": bottom / self.display_h,
        }
        if self.tool.get() == "plot":
            try:
                validate_manual_geometry(box, self.zones)
            except ValueError as exc:
                self.status.set(str(exc))
                return
            self.plot_bounds = box
        else:
            if self.plot_bounds is None:
                self.status.set("Draw the price-plot boundary before adding zones.")
                return
            zone = {"zone_type": self.tool.get(), **box}
            try:
                validate_manual_geometry(self.plot_bounds, self.zones + [zone])
            except ValueError as exc:
                self.status.set(str(exc))
                return
            self.zones.append(zone)
        self.status.set(self._instructions())
        self._redraw()

    def undo(self):
        if self.zones:
            self.zones.pop()
        elif self.plot_bounds is not None:
            self.plot_bounds = None
        self._redraw()

    def save_annotation(self):
        try:
            updated = apply_manual_annotations(
                self.draft, self.plot_bounds, self.zones, self.identity.get().strip()
            )
            _write_new_json(self.output_path, updated)
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
        _DesktopEditor(root, draft, image, args.mode, output)
        root.mainloop()
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"could not start desktop editor ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
