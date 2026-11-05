"""Opt-in calibration from manually read chart-axis labels to screenshot prices.

This module does not perform OCR or guess axis labels. Callers must provide at
least three manually verified (normalized y, price) anchors from the same
screenshot. A fit that is non-monotonic or too distorted is rejected.
"""
from dataclasses import dataclass
import math

MAX_AXIS_ANCHORS = 20
DEFAULT_MAX_RELATIVE_ERROR = 0.02
MAX_ALLOWED_RELATIVE_ERROR = 0.10
_Y_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PriceAxisCalibration:
    scale: str
    slope: float
    intercept: float
    min_y: float
    max_y: float
    anchor_count: int
    max_anchor_relative_error: float

    def price_at_y(self, y: float) -> float:
        """Convert a normalized screenshot y-coordinate to price without extrapolation."""
        y = _finite_number(y, "y")
        if not 0 <= y <= 1:
            raise ValueError("y must be a finite normalized coordinate")
        if y < self.min_y - _Y_TOLERANCE or y > self.max_y + _Y_TOLERANCE:
            raise ValueError("y is outside the calibrated price-axis range")

        fitted = self.intercept + self.slope * y
        try:
            price = math.exp(fitted) if self.scale == "log" else fitted
        except OverflowError as exc:
            raise ValueError("calibrated price is outside the supported numeric range") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("calibrated price is not positive and finite")
        return price


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def calibrate_price_axis(
    anchors,
    *,
    scale="linear",
    max_relative_error=DEFAULT_MAX_RELATIVE_ERROR,
):
    """Fit a linear or logarithmic price axis from 3..20 manual anchors.

    Each anchor is {"y": normalized_full_image_y, "price": positive_price}.
    The chart's conventional orientation is required: higher prices must appear
    above lower prices. At least three anchors are required so curvature or a
    broken/non-uniform scale can be detected instead of being fit perfectly by
    two points.
    """
    if not isinstance(scale, str) or scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'")
    if not isinstance(anchors, (list, tuple)) or not 3 <= len(anchors) <= MAX_AXIS_ANCHORS:
        raise ValueError(f"anchors must contain between 3 and {MAX_AXIS_ANCHORS} entries")
    tolerance = _finite_number(max_relative_error, "max_relative_error")
    if not 0 < tolerance <= MAX_ALLOWED_RELATIVE_ERROR:
        raise ValueError(
            f"max_relative_error must be greater than 0 and at most {MAX_ALLOWED_RELATIVE_ERROR}"
        )

    points = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            raise ValueError(f"Invalid price-axis anchor at index {index}")
        y = _finite_number(anchor.get("y"), f"anchor {index} y")
        price = _finite_number(anchor.get("price"), f"anchor {index} price")
        if not 0 <= y <= 1:
            raise ValueError(f"anchor {index} y must be between 0 and 1")
        if price <= 0:
            raise ValueError(f"anchor {index} price must be positive")
        points.append((y, price))

    points.sort(key=lambda point: point[0])
    for previous, current in zip(points, points[1:]):
        if current[0] - previous[0] <= _Y_TOLERANCE:
            raise ValueError("price-axis anchors must have distinct y coordinates")
        if current[1] >= previous[1]:
            raise ValueError("price must decrease as screenshot y increases")

    ys = [point[0] for point in points]
    values = [math.log(point[1]) if scale == "log" else point[1] for point in points]
    mean_y = sum(ys) / len(ys)
    mean_value = sum(values) / len(values)
    denominator = sum((y - mean_y) ** 2 for y in ys)
    if denominator <= 0:
        raise ValueError("price-axis anchors do not span a usable y range")

    slope = sum(
        (y - mean_y) * (value - mean_value)
        for y, value in zip(ys, values)
    ) / denominator
    intercept = mean_value - slope * mean_y
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope >= 0:
        raise ValueError("price-axis fit is not a descending price scale")

    calibration = PriceAxisCalibration(
        scale=scale,
        slope=slope,
        intercept=intercept,
        min_y=ys[0],
        max_y=ys[-1],
        anchor_count=len(points),
        max_anchor_relative_error=0.0,
    )
    relative_errors = []
    for y, observed_price in points:
        fitted_price = calibration.price_at_y(y)
        relative_errors.append(abs(fitted_price - observed_price) / observed_price)
    max_error = max(relative_errors)
    if max_error > tolerance:
        raise ValueError(
            "price axis appears distorted or inconsistent: "
            f"maximum anchor error {max_error:.4%} exceeds {tolerance:.4%}"
        )

    return PriceAxisCalibration(
        scale=scale,
        slope=slope,
        intercept=intercept,
        min_y=ys[0],
        max_y=ys[-1],
        anchor_count=len(points),
        max_anchor_relative_error=max_error,
    )


def zone_price_bounds(zone, calibration):
    """Return high/low price bounds for a zone using calibrated screenshot y values."""
    if not isinstance(zone, dict):
        raise ValueError("zone must be an object")
    if not isinstance(calibration, PriceAxisCalibration):
        raise ValueError("calibration must be a PriceAxisCalibration")
    y_top = _finite_number(zone.get("y_top"), "zone y_top")
    y_bottom = _finite_number(zone.get("y_bottom"), "zone y_bottom")
    if not 0 <= y_top < y_bottom <= 1:
        raise ValueError("zone y coordinates must be ordered within 0..1")
    price_high = calibration.price_at_y(y_top)
    price_low = calibration.price_at_y(y_bottom)
    if price_high <= price_low:
        raise ValueError("calibrated zone bounds are not ordered")
    return {"price_low": price_low, "price_high": price_high}
