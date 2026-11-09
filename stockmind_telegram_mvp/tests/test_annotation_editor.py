import copy
import hashlib
from contextlib import redirect_stderr
import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app.annotation_editor import (
    apply_manual_annotations,
    load_draft_and_image,
    review_annotation_draft,
    validate_manual_geometry,
    validate_price_axis_anchor_points,
    validate_price_axis_annotations,
    _write_new_json,
    fit_editor_image_size,
    editor_control_rows,
    zoomed_editor_image_size,
    editor_canvas_point_to_normalized,
    editor_canvas_point_to_source_normalized,
    editor_box_to_canvas_coordinates,
    editor_canvas_point_inside_image,
    editor_canvas_point_inside_plot,
    main as editor_main,
)


PLOT = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
ZONE = {
    "zone_type": "demand",
    "x_start": 0.2,
    "x_end": 0.8,
    "y_top": 0.6,
    "y_bottom": 0.8,
}


def _draft():
    return {
        "draft_schema": "stockmind.annotation-draft.v1",
        "benchmark_ready": False,
        "status": "pending_annotation",
        "fixture_id": "btc-1h-001",
        "image_file": "images/btc-1h-001.png",
        "image_sha256": "a" * 64,
        "image_width": 1600,
        "image_height": 900,
        "market_context": {"asset_class": "crypto", "symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1h"},
        "reference_plot_bounds": None,
        "reference_zones": [],
        "price_axis_anchors": [],
        "price_axis_scale": None,
        "price_axis_plot_bounds": None,
        "annotation_metadata": {
            "source": "manual",
            "annotator_id": None,
            "reviewer_id": None,
            "guideline_version": "dns-manual-v1",
            "review_status": "pending",
            "annotated_at": None,
            "reviewed_at": None,
        },
        "prediction_status": "not_captured",
        "predicted_plot_bounds": None,
        "predicted_zones": None,
        "prediction_metadata": None,
    }




class AnnotationEditorSizingTests(unittest.TestCase):
    def test_large_chart_preview_respects_content_caps_on_desktop(self):
        self.assertEqual(
            fit_editor_image_size(3840, 2160, 1920, 1080),
            (1200, 675),
        )

    def test_large_chart_preview_fits_small_remote_desktop(self):
        display_width, display_height = fit_editor_image_size(
            3840, 2160, 800, 600
        )
        self.assertEqual((display_width, display_height), (604, 340))
        self.assertLessEqual(display_width + 40, 800 - 40)
        self.assertLessEqual(display_height + 220, 600 - 40)

    def test_portrait_chart_preview_fits_available_height(self):
        display_width, display_height = fit_editor_image_size(
            900, 1600, 800, 600
        )
        self.assertLessEqual(display_width + 40, 800 - 40)
        self.assertLessEqual(display_height + 220, 600 - 40)

    def test_rejects_non_positive_or_non_integer_dimensions(self):
        invalid = (
            (0, 100, 800, 600),
            (100, True, 800, 600),
            (100, 100, 0, 600),
            (100, 100, 800, 600.0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_editor_image_size(*values)


class AnnotationEditorControlLayoutTests(unittest.TestCase):
    def test_wide_annotation_preview_keeps_controls_in_three_rows(self):
        self.assertEqual(
            editor_control_rows("annotate", 676),
            ("identity", "drawing", "view", "actions"),
        )

    def test_narrow_annotation_preview_stacks_axis_controls(self):
        self.assertEqual(
            editor_control_rows("annotate", 214),
            ("identity", "drawing", "axis", "view", "actions"),
        )

    def test_review_controls_use_compact_rows(self):
        self.assertEqual(
            editor_control_rows("review", 214),
            ("identity", "view", "actions"),
        )

    def test_rejects_invalid_mode_or_display_width(self):
        for mode, width in (("unknown", 500), ("annotate", 0), ("review", True), ("review", 500.0)):
            with self.subTest(mode=mode, width=width), self.assertRaises(ValueError):
                editor_control_rows(mode, width)




class AnnotationEditorZoomTests(unittest.TestCase):
    def test_zoomed_image_size_scales_from_the_same_preview_dimensions(self):
        self.assertEqual(zoomed_editor_image_size(1200, 675, 1.0), (1200, 675))
        self.assertEqual(zoomed_editor_image_size(1200, 675, 2.0), (2400, 1350))
        self.assertEqual(zoomed_editor_image_size(1200, 675, 4.0), (4800, 2700))

    def test_zoom_is_capped_at_original_source_dimensions(self):
        self.assertEqual(
            zoomed_editor_image_size(
                1200, 675, 4.0, max_width=1600, max_height=900
            ),
            (1600, 900),
        )
        self.assertEqual(
            zoomed_editor_image_size(
                720, 405, 4.0, max_width=800, max_height=450
            ),
            (800, 450),
        )

    def test_source_dimension_caps_reject_invalid_values(self):
        for max_width, max_height in (
            (0, 900),
            (1600, True),
            (-1, 900),
            ("1600", 900),
        ):
            with self.subTest(max_width=max_width, max_height=max_height), self.assertRaises(ValueError):
                zoomed_editor_image_size(
                    1200, 675, 2.0, max_width=max_width, max_height=max_height
                )

    def test_zoom_rejects_invalid_dimensions_and_levels(self):
        for width, height, zoom in (
            (0, 100, 2.0),
            (100, True, 2.0),
            (100, 100, True),
            (100, 100, 0.99),
            (100, 100, 4.01),
            (100, 100, float("nan")),
            (100, 100, float("inf")),
            (100, 100, "2"),
        ):
            with self.subTest(width=width, height=height, zoom=zoom), self.assertRaises(ValueError):
                zoomed_editor_image_size(width, height, zoom)

    def test_same_source_point_maps_to_same_normalized_coordinates_at_any_zoom(self):
        self.assertEqual(
            editor_canvas_point_to_normalized(600, 337.5, 1200, 675),
            (0.5, 0.5),
        )
        self.assertEqual(
            editor_canvas_point_to_normalized(1200, 675, 2400, 1350),
            (0.5, 0.5),
        )

    def test_scrolled_canvas_coordinates_are_clamped_to_image_edges(self):
        self.assertEqual(
            editor_canvas_point_to_normalized(-20, 1400, 1200, 1350),
            (0.0, 1.0),
        )
        self.assertEqual(
            editor_canvas_point_to_normalized(1300, -10, 1200, 1350),
            (1.0, 0.0),
        )

    def test_canvas_mapping_rejects_boolean_and_non_finite_coordinates(self):
        for x, y in ((True, 1), (1, False), (float("nan"), 1), (1, float("inf")), ("1", 1)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                editor_canvas_point_to_normalized(x, y, 1200, 675)





class AnnotationSourcePixelAlignmentTests(unittest.TestCase):
    IMAGE_WIDTH = 4033
    IMAGE_HEIGHT = 2269
    RENDER_SIZES = (
        (1200, 675),
        (1800, 1012),
        (2400, 1350),
        (3600, 2025),
        (4033, 2269),
    )

    def test_source_pixel_point_round_trips_at_every_zoom_level(self):
        source_x, source_y = 1235, 713
        for render_width, render_height in self.RENDER_SIZES:
            with self.subTest(render_size=(render_width, render_height)):
                canvas_x = source_x * render_width / self.IMAGE_WIDTH
                canvas_y = source_y * render_height / self.IMAGE_HEIGHT
                self.assertEqual(
                    editor_canvas_point_to_source_normalized(
                        canvas_x, canvas_y,
                        render_width, render_height,
                        self.IMAGE_WIDTH, self.IMAGE_HEIGHT,
                    ),
                    (source_x / self.IMAGE_WIDTH, source_y / self.IMAGE_HEIGHT),
                )

    def test_box_preview_uses_the_same_rounded_source_pixels_at_every_zoom(self):
        box = {
            "x_start": 1235.49 / self.IMAGE_WIDTH,
            "x_end": 3001.51 / self.IMAGE_WIDTH,
            "y_top": 713.49 / self.IMAGE_HEIGHT,
            "y_bottom": 1401.51 / self.IMAGE_HEIGHT,
        }
        source_edges = (1235, 713, 3002, 1402)
        for render_width, render_height in self.RENDER_SIZES:
            with self.subTest(render_size=(render_width, render_height)):
                actual = editor_box_to_canvas_coordinates(
                    box,
                    self.IMAGE_WIDTH,
                    self.IMAGE_HEIGHT,
                    render_width,
                    render_height,
                )
                expected = (
                    source_edges[0] * render_width / self.IMAGE_WIDTH,
                    source_edges[1] * render_height / self.IMAGE_HEIGHT,
                    source_edges[2] * render_width / self.IMAGE_WIDTH,
                    source_edges[3] * render_height / self.IMAGE_HEIGHT,
                )
                for actual_value, expected_value in zip(actual, expected):
                    self.assertAlmostEqual(actual_value, expected_value, places=10)

    def test_editor_box_coordinates_match_renderer_pixel_rounding(self):
        from stockmind_telegram_mvp.app.annotation_editor import _DesktopEditor

        editor = object.__new__(_DesktopEditor)
        editor.image_width = self.IMAGE_WIDTH
        editor.image_height = self.IMAGE_HEIGHT
        editor.render_w, editor.render_h = 1800, 1012
        box = {
            "x_start": 1235.49 / self.IMAGE_WIDTH,
            "x_end": 3001.51 / self.IMAGE_WIDTH,
            "y_top": 713.49 / self.IMAGE_HEIGHT,
            "y_bottom": 1401.51 / self.IMAGE_HEIGHT,
        }
        self.assertEqual(
            editor._coords(box),
            editor_box_to_canvas_coordinates(
                box, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1800, 1012
            ),
        )

    def test_source_pixel_mapping_rejects_invalid_dimensions(self):
        for image_width, image_height in (
            (0, 900), (1600, True), (1.5, 900),
        ):
            with self.subTest(image_width=image_width, image_height=image_height):
                with self.assertRaises(ValueError):
                    editor_canvas_point_to_source_normalized(
                        10, 10, 100, 100, image_width, image_height
                    )
                with self.assertRaises(ValueError):
                    editor_box_to_canvas_coordinates(
                        {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9},
                        image_width, image_height, 100, 100,
                    )


class AnnotationEditorAxisAnchorGeometryTests(unittest.TestCase):
    def test_raw_canvas_point_must_be_inside_image_without_edge_clamping(self):
        self.assertTrue(editor_canvas_point_inside_image(0, 0, 1000, 500))
        self.assertTrue(editor_canvas_point_inside_image(1000, 500, 1000, 500))
        self.assertFalse(editor_canvas_point_inside_image(-0.01, 250, 1000, 500))
        self.assertFalse(editor_canvas_point_inside_image(1000.01, 250, 1000, 500))
        self.assertFalse(editor_canvas_point_inside_image(500, -0.01, 1000, 500))
        self.assertFalse(editor_canvas_point_inside_image(500, 500.01, 1000, 500))

    def test_raw_canvas_point_rejects_invalid_values(self):
        for point in ((True, 1), (1, False), (float("nan"), 1), (1, float("inf"))):
            with self.subTest(point=point), self.assertRaises(ValueError):
                editor_canvas_point_inside_image(*point, 1000, 500)

    def test_axis_anchor_point_must_be_inside_both_plot_dimensions(self):
        self.assertTrue(
            editor_canvas_point_inside_plot(500, 250, 1000, 500, PLOT)
        )
        self.assertFalse(
            editor_canvas_point_inside_plot(950, 250, 1000, 500, PLOT)
        )
        self.assertFalse(
            editor_canvas_point_inside_plot(500, 480, 1000, 500, PLOT)
        )

    def test_axis_anchor_point_on_plot_edges_is_allowed(self):
        self.assertTrue(
            editor_canvas_point_inside_plot(100, 50, 1000, 500, PLOT)
        )
        self.assertTrue(
            editor_canvas_point_inside_plot(900, 450, 1000, 500, PLOT)
        )

    def test_axis_anchor_point_rejects_invalid_plot_geometry(self):
        invalid_plot = {**PLOT, "x_start": 0.95}
        with self.assertRaisesRegex(ValueError, "positive width"):
            editor_canvas_point_inside_plot(500, 250, 1000, 500, invalid_plot)


class AnnotationEditorAxisAnchorInteractionTests(unittest.TestCase):
    def _editor(self, x, y, *, plot=PLOT, start_inside=True):
        from stockmind_telegram_mvp.app.annotation_editor import _DesktopEditor

        editor = object.__new__(_DesktopEditor)
        editor.drag_start = (x, y)
        editor.drag_start_inside_image = start_inside
        editor.preview_id = None
        editor._canvas_event_point = lambda event: (event.x, event.y)
        editor._canvas_point = lambda event: (
            max(0, min(1000, event.x)), max(0, min(500, event.y))
        )
        editor.tool = types.SimpleNamespace(get=lambda: "axis_anchor")
        editor.plot_bounds = plot
        editor.render_w = 1000
        editor.render_h = 500
        editor.image_width = 1000
        editor.image_height = 500
        editor.axis_anchors = []
        editor.undo_stack = []
        editor.canvas = types.SimpleNamespace(delete=lambda _item: None)
        editor.status = types.SimpleNamespace(set=lambda value: setattr(editor, "status_text", value))
        editor.root = object()
        editor.prompt_calls = []
        editor.simpledialog = types.SimpleNamespace(
            askstring=lambda *args, **kwargs: (
                editor.prompt_calls.append((args, kwargs)) or "100"
            )
        )
        editor._redraw = lambda: None
        return editor

    def test_click_in_price_label_margin_is_rejected_before_prompt(self):
        editor = self._editor(950, 250)

        editor._finish_draw(types.SimpleNamespace(x=950, y=250))

        self.assertEqual(editor.prompt_calls, [])
        self.assertEqual(editor.axis_anchors, [])
        self.assertIn("inside the main price plot", editor.status_text)

    def test_click_inside_plot_can_add_price_anchor(self):
        editor = self._editor(500, 250)

        editor._finish_draw(types.SimpleNamespace(x=500, y=250))

        self.assertEqual(len(editor.prompt_calls), 1)
        self.assertEqual(editor.axis_anchors, [{"y": 0.5, "price": 100.0}])
        self.assertEqual(editor.undo_stack[0][0], "axis_anchor")

    def test_axis_anchor_is_quantized_to_original_screenshot_row(self):
        editor = self._editor(500, 251.2)
        editor.image_width = 1600
        editor.image_height = 900

        editor._finish_draw(types.SimpleNamespace(x=500, y=251.2))

        self.assertEqual(
            editor.axis_anchors,
            [{"y": 452 / 900, "price": 100.0}],
        )

    def test_release_in_blank_canvas_is_not_clamped_into_edge_plot(self):
        edge_plot = {"x_start": 0.1, "x_end": 1.0, "y_top": 0.1, "y_bottom": 1.0}
        editor = self._editor(500, 250, plot=edge_plot)

        editor._finish_draw(types.SimpleNamespace(x=1001, y=250))

        self.assertEqual(editor.prompt_calls, [])
        self.assertEqual(editor.axis_anchors, [])
        self.assertIn("within the screenshot image", editor.status_text)

    def test_press_in_blank_canvas_cannot_become_edge_anchor(self):
        edge_plot = {"x_start": 0.1, "x_end": 1.0, "y_top": 0.1, "y_bottom": 1.0}
        editor = self._editor(1001, 250, plot=edge_plot, start_inside=False)

        editor._finish_draw(types.SimpleNamespace(x=1001, y=250))

        self.assertEqual(editor.prompt_calls, [])
        self.assertEqual(editor.axis_anchors, [])
        self.assertIn("within the screenshot image", editor.status_text)


class AnnotationEditorZoomInteractionTests(unittest.TestCase):
    class FakeCanvas:
        def __init__(self):
            self.configured = {}
            self.x_position = None
            self.y_position = None
            self.image = None

        def winfo_width(self):
            return 600

        def winfo_height(self):
            return 400

        def canvasx(self, value):
            return value + 200

        def canvasy(self, value):
            return value + 100

        def itemconfigure(self, item, **kwargs):
            self.image = kwargs.get("image")

        def configure(self, **kwargs):
            self.configured.update(kwargs)

        def xview_moveto(self, value):
            self.x_position = value

        def yview_moveto(self, value):
            self.y_position = value

        def delete(self, tag):
            pass

    def _editor(self):
        from stockmind_telegram_mvp.app.annotation_editor import _DesktopEditor

        editor = object.__new__(_DesktopEditor)
        editor.drag_start = None
        editor.zoom_index = 0
        editor.zoom_factor = 1.0
        editor.zoom_label = types.SimpleNamespace(set=lambda value: setattr(editor, "zoom_text", value))
        editor.display_w = 1200
        editor.display_h = 675
        editor.image_width = 2400
        editor.image_height = 1350
        editor.render_w = 1200
        editor.render_h = 675
        editor.source_image = Image.new("RGB", (2400, 1350), "white")
        editor.ImageTk = types.SimpleNamespace(PhotoImage=lambda image: image.copy())
        editor.photo = None
        editor.canvas = self.FakeCanvas()
        editor.image_item = 1
        editor.plot_bounds = None
        editor.zones = []
        editor.axis_anchors = []
        editor.status = types.SimpleNamespace(set=lambda value: setattr(editor, "status_text", value))
        return editor

    def test_zoom_in_renders_from_source_and_preserves_view_center(self):
        editor = self._editor()

        editor.zoom_in()

        self.assertEqual(editor.zoom_factor, 1.5)
        self.assertEqual((editor.render_w, editor.render_h), (1800, 1012))
        self.assertEqual(editor.photo.size, (1800, 1012))
        self.assertEqual(editor.canvas.configured["scrollregion"], (0, 0, 1800, 1012))
        self.assertEqual(editor.zoom_text, "150%")
        self.assertAlmostEqual(editor.canvas.x_position, 0.25)
        self.assertAlmostEqual(editor.canvas.y_position, 0.2468, places=3)

    def test_zoom_stops_at_native_resolution_and_reports_the_effective_scale(self):
        editor = self._editor()
        editor.image_width = 1600
        editor.image_height = 900
        editor.source_image = Image.new("RGB", (1600, 900), "white")

        editor.zoom_in()

        self.assertAlmostEqual(editor.zoom_factor, 1600 / 1200)
        self.assertEqual((editor.render_w, editor.render_h), (1600, 900))
        self.assertEqual(editor.zoom_text, "133%")

        editor.zoom_in()

        self.assertEqual(editor.zoom_index, 1)
        self.assertIn("native resolution", editor.status_text.lower())
        self.assertEqual((editor.render_w, editor.render_h), (1600, 900))

        editor.zoom_out()

        self.assertEqual(editor.zoom_index, 0)
        self.assertEqual((editor.render_w, editor.render_h), (1200, 675))

    def test_canvas_event_coordinates_include_scroll_offsets(self):
        editor = self._editor()

        self.assertEqual(
            editor._canvas_point(types.SimpleNamespace(x=100, y=50)),
            (300.0, 150.0),
        )

    def test_zoom_change_is_blocked_during_an_active_drag(self):
        editor = self._editor()
        editor.drag_start = (100.0, 100.0)

        editor.zoom_in()

        self.assertEqual(editor.zoom_index, 0)
        self.assertEqual((editor.render_w, editor.render_h), (1200, 675))
        self.assertIn("current drawing", editor.status_text)


class AnnotationGeometryTests(unittest.TestCase):
    def test_accepts_plot_with_no_visible_zones(self):
        plot, zones = validate_manual_geometry(PLOT, [])
        self.assertEqual(plot, PLOT)
        self.assertEqual(zones, [])

    def test_normalizes_numeric_coordinates_to_floats(self):
        plot, zones = validate_manual_geometry(
            {"x_start": 0, "x_end": 1, "y_top": 0, "y_bottom": 1},
            [{"zone_type": "supply", "x_start": 0, "x_end": 1, "y_top": 0, "y_bottom": 1}],
        )
        self.assertEqual(plot["x_start"], 0.0)
        self.assertEqual(zones[0]["y_bottom"], 1.0)

    def test_rejects_boolean_nan_infinite_and_out_of_range_coordinates(self):
        invalid_values = (True, float("nan"), float("inf"), -0.01, 1.01, "0.5")
        for value in invalid_values:
            with self.subTest(value=value):
                bad = dict(PLOT, x_start=value)
                with self.assertRaises(ValueError):
                    validate_manual_geometry(bad, [])

    def test_rejects_zero_or_negative_width_and_height(self):
        with self.assertRaisesRegex(ValueError, "positive width"):
            validate_manual_geometry(dict(PLOT, x_end=PLOT["x_start"]), [])
        with self.assertRaisesRegex(ValueError, "positive height"):
            validate_manual_geometry(dict(PLOT, y_bottom=PLOT["y_top"]), [])

    def test_rejects_zone_outside_plot(self):
        outside = dict(ZONE, x_start=0.05)
        with self.assertRaisesRegex(ValueError, "fully inside"):
            validate_manual_geometry(PLOT, [outside])

    def test_rejects_unknown_zone_type_and_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "zone_type"):
            validate_manual_geometry(PLOT, [dict(ZONE, zone_type="neutral")])
        with self.assertRaisesRegex(ValueError, "exactly four"):
            validate_manual_geometry(PLOT, [dict(ZONE, note="guess")])

    def test_rejects_non_list_or_more_than_eight_zones(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_manual_geometry(PLOT, ())
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_manual_geometry(PLOT, [ZONE] * 9)


class PriceAxisAnnotationTests(unittest.TestCase):
    LINEAR_ANCHORS = [
        {"y": 0.2, "price": 120.0},
        {"y": 0.5, "price": 90.0},
        {"y": 0.8, "price": 60.0},
    ]

    def test_normalizes_anchor_order_and_calibrates_linear_scale(self):
        reversed_order = list(reversed(self.LINEAR_ANCHORS))
        anchors, scale = validate_price_axis_annotations(PLOT, reversed_order, "linear")
        self.assertEqual([point["y"] for point in anchors], [0.2, 0.5, 0.8])
        self.assertEqual(scale, "linear")

    def test_calibrates_log_scale_from_manually_supplied_anchors(self):
        anchors = [
            {"y": 0.2, "price": 100.0},
            {"y": 0.5, "price": 70.0},
            {"y": 0.8, "price": 49.0},
        ]
        normalized, scale = validate_price_axis_annotations(PLOT, anchors, "log")
        self.assertEqual(len(normalized), 3)
        self.assertEqual(scale, "log")

    def test_geometry_only_annotation_does_not_invent_a_scale(self):
        anchors, scale = validate_price_axis_annotations(PLOT, [], "linear")
        self.assertEqual(anchors, [])
        self.assertIsNone(scale)

    def test_rejects_one_or_two_anchors_for_price_scoring(self):
        for anchors in (self.LINEAR_ANCHORS[:1], self.LINEAR_ANCHORS[:2]):
            with self.subTest(count=len(anchors)), self.assertRaisesRegex(ValueError, "at least 3"):
                validate_price_axis_annotations(PLOT, anchors, "linear")

    def test_rejects_anchor_outside_plot_or_duplicate_y(self):
        outside = [dict(self.LINEAR_ANCHORS[0], y=0.05), *self.LINEAR_ANCHORS[1:]]
        duplicate = [*self.LINEAR_ANCHORS[:2], dict(self.LINEAR_ANCHORS[2], y=0.5)]
        with self.assertRaisesRegex(ValueError, "inside the marked price plot"):
            validate_price_axis_anchor_points(PLOT, outside)
        with self.assertRaisesRegex(ValueError, "distinct y"):
            validate_price_axis_anchor_points(PLOT, duplicate)

    def test_rejects_reversed_or_non_positive_prices(self):
        reversed_prices = [
            {"y": 0.2, "price": 60.0},
            {"y": 0.5, "price": 90.0},
            {"y": 0.8, "price": 120.0},
        ]
        with self.assertRaisesRegex(ValueError, "decrease"):
            validate_price_axis_anchor_points(PLOT, reversed_prices)
        invalid = [dict(self.LINEAR_ANCHORS[0], price=0), *self.LINEAR_ANCHORS[1:]]
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_price_axis_anchor_points(PLOT, invalid)

    def test_rejects_boolean_non_finite_and_extra_anchor_fields(self):
        for value in (True, float("nan"), float("inf")):
            bad = [dict(self.LINEAR_ANCHORS[0], price=value), *self.LINEAR_ANCHORS[1:]]
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_price_axis_anchor_points(PLOT, bad)
        extra = [dict(self.LINEAR_ANCHORS[0], note="estimated"), *self.LINEAR_ANCHORS[1:]]
        with self.assertRaisesRegex(ValueError, "exactly y and price"):
            validate_price_axis_anchor_points(PLOT, extra)

    def test_rejects_distorted_fit_and_invalid_scale(self):
        distorted = [
            {"y": 0.2, "price": 120.0},
            {"y": 0.5, "price": 100.0},
            {"y": 0.8, "price": 50.0},
        ]
        with self.assertRaisesRegex(ValueError, "distorted or inconsistent"):
            validate_price_axis_annotations(PLOT, distorted, "linear")
        with self.assertRaisesRegex(ValueError, "linear' or 'log"):
            validate_price_axis_annotations(PLOT, self.LINEAR_ANCHORS, "unknown")


class AnnotationWorkflowTests(unittest.TestCase):
    def test_annotation_creates_pending_review_without_mutating_input(self):
        draft = _draft()
        original = copy.deepcopy(draft)
        result = apply_manual_annotations(
            draft, PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z"
        )
        self.assertEqual(draft, original)
        self.assertEqual(result["status"], "pending_review")
        self.assertFalse(result["benchmark_ready"])
        self.assertEqual(result["reference_zones"], [ZONE])
        self.assertEqual(result["annotation_metadata"]["annotator_id"], "ann-001")
        self.assertIsNone(result["annotation_metadata"]["reviewer_id"])
        self.assertEqual(result["annotation_metadata"]["review_status"], "pending")

    def test_allows_no_zone_annotation_when_plot_is_valid(self):
        result = apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        self.assertEqual(result["reference_zones"], [])
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual(result["price_axis_anchors"], [])
        self.assertIsNone(result["price_axis_scale"])
        self.assertIsNone(result["price_axis_plot_bounds"])

    def test_annotation_persists_manual_axis_anchors_and_plot_bounds(self):
        anchors = [
            {"y": 0.8, "price": 60},
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
        ]
        result = apply_manual_annotations(
            _draft(), PLOT, [ZONE], "ann-001",
            annotated_at="2026-11-07T12:00:00Z",
            price_axis_anchors=anchors,
            price_axis_scale="linear",
        )
        self.assertEqual(result["price_axis_anchors"], [
            {"y": 0.2, "price": 120.0},
            {"y": 0.5, "price": 90.0},
            {"y": 0.8, "price": 60.0},
        ])
        self.assertEqual(result["price_axis_scale"], "linear")
        self.assertEqual(result["price_axis_plot_bounds"], PLOT)

    def test_reviewer_revalidates_and_preserves_price_axis_calibration(self):
        annotated = apply_manual_annotations(
            _draft(), PLOT, [], "ann-001",
            annotated_at="2026-11-07T12:00:00Z",
            price_axis_anchors=[
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
        )
        approved = review_annotation_draft(
            annotated, "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z"
        )
        self.assertEqual(approved["price_axis_scale"], "linear")
        self.assertEqual(len(approved["price_axis_anchors"]), 3)
        self.assertEqual(approved["price_axis_plot_bounds"], PLOT)

    def test_rejects_review_when_axis_plot_does_not_match_reference_plot(self):
        annotated = apply_manual_annotations(
            _draft(), PLOT, [], "ann-001",
            annotated_at="2026-11-07T12:00:00Z",
            price_axis_anchors=[
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
        )
        annotated["price_axis_plot_bounds"] = dict(PLOT, x_start=0.2)
        with self.assertRaisesRegex(ValueError, "must match"):
            review_annotation_draft(
                annotated, "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z"
            )

    def test_rejects_invalid_annotator_id(self):
        for value in ("ann-", "person@example.com", "../ann-1", "rev-001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_manual_annotations(_draft(), PLOT, [], value, annotated_at="2026-11-07T12:00:00Z")

    def test_rejects_unannotated_draft_with_existing_provenance(self):
        draft = _draft()
        draft["annotation_metadata"]["annotator_id"] = "ann-001"
        with self.assertRaisesRegex(ValueError, "must not contain"):
            apply_manual_annotations(draft, PLOT, [], "ann-002")

    def test_rejects_intake_draft_with_preexisting_geometry_or_prediction(self):
        draft_with_geometry = _draft()
        draft_with_geometry["reference_plot_bounds"] = PLOT
        with self.assertRaisesRegex(ValueError, "manual geometry"):
            apply_manual_annotations(draft_with_geometry, PLOT, [], "ann-001")

        draft_with_prediction = _draft()
        draft_with_prediction["prediction_status"] = "captured"
        draft_with_prediction["predicted_zones"] = []
        with self.assertRaisesRegex(ValueError, "AI prediction"):
            apply_manual_annotations(draft_with_prediction, PLOT, [], "ann-001")

    def test_requires_timezone_aware_annotation_timestamp(self):
        for value in ("2026-11-07T12:00:00", "not-a-time"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at=value)

    def test_review_requires_reviewer_role_and_valid_time_order(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "reviewer_id"):
            review_annotation_draft(annotated, "ann-001", approve=True, reviewed_at="2026-11-07T12:01:00Z")
        with self.assertRaisesRegex(ValueError, "earlier"):
            review_annotation_draft(annotated, "rev-001", approve=True, reviewed_at="2026-11-07T11:59:00Z")

    def test_approval_is_a_separate_non_benchmark_ready_transition(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        approved = review_annotation_draft(annotated, "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z")
        self.assertEqual(approved["status"], "approved_pending_prediction")
        self.assertEqual(approved["annotation_metadata"]["review_status"], "approved")
        self.assertEqual(approved["annotation_metadata"]["reviewer_id"], "rev-002")
        self.assertFalse(approved["benchmark_ready"])
        self.assertEqual(annotated["status"], "pending_review")

    def test_rejection_is_recorded_and_cannot_be_mistaken_for_approval(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        rejected = review_annotation_draft(annotated, "rev-002", approve=False, reviewed_at="2026-11-07T12:05:00Z")
        self.assertEqual(rejected["status"], "rejected_annotation")
        self.assertEqual(rejected["annotation_metadata"]["review_status"], "rejected")
        self.assertFalse(rejected["benchmark_ready"])

    def test_reviewer_cannot_approve_an_unreviewed_intake_draft(self):
        with self.assertRaisesRegex(ValueError, "status"):
            review_annotation_draft(_draft(), "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z")

    def test_approve_flag_must_be_a_real_boolean(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "boolean"):
            review_annotation_draft(annotated, "rev-002", approve=1, reviewed_at="2026-11-07T12:05:00Z")


class AnnotationFilePersistenceTests(unittest.TestCase):
    def test_partial_output_is_removed_when_json_write_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "annotation.json"

            def fail_after_partial_write(value, stream, **kwargs):
                stream.write("{")
                raise OSError("simulated disk full")

            with patch(
                "stockmind_telegram_mvp.app.annotation_editor.json.dump",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "simulated disk full"):
                    _write_new_json(output, {"status": "pending_review"})

            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "annotation.json"
            output.write_text("existing annotation\\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _write_new_json(output, {"status": "pending_review"})

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "existing annotation\\n",
            )

    def test_successful_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "annotation.json"
            result = _write_new_json(output, {"status": "pending_review"})

            self.assertEqual(result, output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "pending_review"},
            )


class DraftImageVerificationTests(unittest.TestCase):
    def _png(self, size=(32, 24)):
        output = BytesIO()
        Image.new("RGB", size, "white").save(output, format="PNG")
        return output.getvalue()

    def _draft_file(self, root, image_bytes, image_file="images/chart.png"):
        image_path = root / image_file
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        draft = _draft()
        draft["image_file"] = image_file
        draft["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(BytesIO(image_bytes)) as image:
            draft["image_width"], draft["image_height"] = image.size
        draft_path = root / "chart.draft.json"
        draft_path.write_text(json.dumps(draft), encoding="utf-8")
        return draft_path, image_path

    def test_loads_image_only_after_hash_and_dimensions_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self._png()
            draft_path, _ = self._draft_file(root, data)
            stub = types.ModuleType("stockmind_telegram_mvp.app.overlay_engine")
            stub.prepare_chart_image = lambda image_bytes, **kwargs: (image_bytes, "image/png")
            with patch.dict(sys.modules, {stub.__name__: stub}):
                draft, image = load_draft_and_image(draft_path)
            self.assertEqual(draft["fixture_id"], "btc-1h-001")
            self.assertEqual(image.size, (32, 24))

    def test_rejects_noncanonical_exif_oriented_screenshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            encoded = BytesIO()
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (32, 24), "white").save(
                encoded, format="JPEG", quality=95, exif=exif
            )
            draft_path, _ = self._draft_file(root, encoded.getvalue())

            with self.assertRaisesRegex(ValueError, "not EXIF-normalized"):
                load_draft_and_image(draft_path)

    def test_accepts_canonical_screenshot_without_reencoding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft_path, _ = self._draft_file(root, self._png())

            draft, image = load_draft_and_image(draft_path)

            self.assertEqual(draft["fixture_id"], "btc-1h-001")
            self.assertEqual(image.size, (32, 24))

    def test_decodes_the_exact_bytes_prepared_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = self._png()
            draft_path, image_path = self._draft_file(root, original)
            stub = types.ModuleType("stockmind_telegram_mvp.app.overlay_engine")

            def replace_path_after_preparation(image_bytes, **kwargs):
                self.assertEqual(image_bytes, original)
                replacement = BytesIO()
                Image.new("RGB", (32, 24), "red").save(replacement, format="PNG")
                image_path.write_bytes(replacement.getvalue())
                return image_bytes, "image/png"

            stub.prepare_chart_image = replace_path_after_preparation
            with patch.dict(sys.modules, {stub.__name__: stub}):
                _, image = load_draft_and_image(draft_path)

            self.assertEqual(image.size, (32, 24))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_rejects_screenshot_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft_path, image_path = self._draft_file(root, self._png())
            image_path.write_bytes(self._png((33, 24)))
            stub = types.ModuleType("stockmind_telegram_mvp.app.overlay_engine")
            stub.prepare_chart_image = lambda image_bytes, **kwargs: (image_bytes, "image/png")
            with patch.dict(sys.modules, {stub.__name__: stub}), self.assertRaisesRegex(ValueError, "SHA-256"):
                load_draft_and_image(draft_path)

    def test_rejects_unsafe_relative_image_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = _draft()
            draft["image_file"] = "../outside.png"
            draft_path = root / "chart.draft.json"
            draft_path.write_text(json.dumps(draft), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                load_draft_and_image(draft_path)


class AnnotationEditorCliExitTests(unittest.TestCase):
    def _run_editor(self, mode, *, save_output, external_output=False):
        with tempfile.TemporaryDirectory() as temp:
            draft_path = Path(temp) / "chart.draft.json"
            draft_path.write_text("{}", encoding="utf-8")
            draft = _draft()
            if mode == "review":
                draft = apply_manual_annotations(
                    draft,
                    PLOT,
                    [ZONE],
                    "ann-001",
                    annotated_at="2026-11-08T09:00:00Z",
                )

            root = types.SimpleNamespace(mainloop=lambda: None)
            tkinter_stub = types.ModuleType("tkinter")
            tkinter_stub.Tk = lambda: root

            def fake_editor(_root, _draft_value, _image, _mode, output_path):
                editor = types.SimpleNamespace(saved=False)
                if save_output:
                    output_path.write_text("{}", encoding="utf-8")
                    editor.saved = True
                elif external_output:
                    # Simulate another process creating the path after the
                    # editor has started; that is not this session's save.
                    output_path.write_text("{}", encoding="utf-8")
                return editor

            stderr = __import__("io").StringIO()
            with (
                patch(
                    "stockmind_telegram_mvp.app.annotation_editor.load_draft_and_image",
                    return_value=(draft, object()),
                ),
                patch(
                    "stockmind_telegram_mvp.app.annotation_editor._DesktopEditor",
                    side_effect=fake_editor,
                ),
                patch.dict(sys.modules, {"tkinter": tkinter_stub}),
                redirect_stderr(stderr),
            ):
                exit_code = editor_main([str(draft_path), "--mode", mode])
            return exit_code, stderr.getvalue()

    def test_closing_annotation_or_review_without_saving_returns_nonzero(self):
        for mode in ("annotate", "review"):
            with self.subTest(mode=mode):
                exit_code, stderr = self._run_editor(mode, save_output=False)
                self.assertEqual(exit_code, 1)
                self.assertIn("closed without saving", stderr)

    def test_saving_annotation_or_review_returns_zero(self):
        for mode in ("annotate", "review"):
            with self.subTest(mode=mode):
                exit_code, stderr = self._run_editor(mode, save_output=True)
                self.assertEqual(exit_code, 0)
                self.assertEqual(stderr, "")

    def test_external_output_created_after_launch_does_not_count_as_editor_save(self):
        for mode in ("annotate", "review"):
            with self.subTest(mode=mode):
                exit_code, stderr = self._run_editor(
                    mode, save_output=False, external_output=True
                )
                self.assertEqual(exit_code, 1)
                self.assertIn("this session did not create an output", stderr)



class AnnotationEditorDrawingStartInteractionTests(unittest.TestCase):
    class FakeCanvas:
        def __init__(self):
            self.created = []
            self.deleted = []

        def create_rectangle(self, *coords, **kwargs):
            self.created.append(("rectangle", coords, kwargs))
            return len(self.created)

        def create_line(self, *coords, **kwargs):
            self.created.append(("line", coords, kwargs))
            return len(self.created)

        def delete(self, item):
            self.deleted.append(item)

    def _editor(self, tool):
        from stockmind_telegram_mvp.app.annotation_editor import _DesktopEditor

        editor = object.__new__(_DesktopEditor)
        editor.render_w = 1000
        editor.render_h = 500
        editor.image_width = 1600
        editor.image_height = 900
        editor.drag_start = None
        editor.drag_start_inside_image = False
        editor.preview_id = None
        editor.tool = types.SimpleNamespace(get=lambda: tool)
        editor.canvas = self.FakeCanvas()
        editor.status = types.SimpleNamespace(
            set=lambda value: setattr(editor, "status_text", value)
        )
        editor._canvas_event_point = lambda event: (float(event.x), float(event.y))
        editor._canvas_point = lambda event: (
            max(0.0, min(float(editor.render_w), float(event.x))),
            max(0.0, min(float(editor.render_h), float(event.y))),
        )
        editor.plot_bounds = None
        editor.zones = []
        editor.axis_anchors = []
        editor.undo_stack = []
        editor.mode = "annotate"
        editor._instructions = lambda: "Draw the price-plot boundary."
        editor._redraw = lambda: None
        return editor

    def test_blank_canvas_press_cannot_start_any_drawing_tool(self):
        for tool in ("plot", "supply", "demand", "axis_anchor"):
            with self.subTest(tool=tool):
                editor = self._editor(tool)

                editor._start_draw(types.SimpleNamespace(x=1001, y=250))

                self.assertIsNone(editor.drag_start)
                self.assertFalse(editor.drag_start_inside_image)
                self.assertIsNone(editor.preview_id)
                self.assertEqual(editor.canvas.created, [])
                self.assertIn("Start drawing inside", editor.status_text)

    def test_rectangle_drag_can_still_finish_outside_and_clip_to_image_edge(self):
        editor = self._editor("plot")

        editor._start_draw(types.SimpleNamespace(x=500, y=250))
        editor._finish_draw(types.SimpleNamespace(x=1100, y=300))

        self.assertEqual(editor.plot_bounds["x_start"], 0.5)
        self.assertEqual(editor.plot_bounds["x_end"], 1.0)
        self.assertEqual(editor.plot_bounds["y_top"], 0.5)
        self.assertEqual(editor.plot_bounds["y_bottom"], 0.6)
        self.assertEqual(editor.undo_stack[-1][0], "plot")

    def test_rectangle_drag_saves_identical_source_pixel_edges_at_every_zoom(self):
        source_width, source_height = 4033, 2269
        source_edges = (1235, 713, 3002, 1402)
        render_sizes = (
            (1200, 675), (1800, 1012), (2400, 1350),
            (3600, 2025), (4033, 2269),
        )
        for render_width, render_height in render_sizes:
            with self.subTest(render_size=(render_width, render_height)):
                editor = self._editor("plot")
                editor.image_width = source_width
                editor.image_height = source_height
                editor.render_w = render_width
                editor.render_h = render_height
                start = types.SimpleNamespace(
                    x=source_edges[0] * render_width / source_width,
                    y=source_edges[1] * render_height / source_height,
                )
                finish = types.SimpleNamespace(
                    x=source_edges[2] * render_width / source_width,
                    y=source_edges[3] * render_height / source_height,
                )

                editor._start_draw(start)
                editor._finish_draw(finish)

                self.assertEqual(editor.plot_bounds, {
                    "x_start": source_edges[0] / source_width,
                    "x_end": source_edges[2] / source_width,
                    "y_top": source_edges[1] / source_height,
                    "y_bottom": source_edges[3] / source_height,
                })

if __name__ == "__main__":
    unittest.main()
