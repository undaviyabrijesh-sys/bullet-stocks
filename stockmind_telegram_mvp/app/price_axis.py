"""Opt-in calibration from manually read chart-axis labels to screenshot prices.

This module does not perform OCR or guess axis labels. Callers must provide at
least three manually verified (normalized y, price) anchors from the same
screenshot. A fit that is non-monotonic or too distorted is rejected.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re

MAX_AXIS_ANCHORS = 20
DEFAULT_MAX_RELATIVE_ERROR = 0.02
MAX_ALLOWED_RELATIVE_ERROR = 0.10
_Y_TOLERANCE = 1e-9
MAX_REPORTED_PRICE_TEXT_LENGTH = 32
_REPORTED_PRICE_PATTERN = re.compile(r"[0-9]+(?:[.][0-9]+)?")


def parse_zone_price_bounds(zone):
    """Parse a complete, positive, ordered pair of plain decimal price strings.

    This validates label ordering only. A manually calibrated screenshot axis
    is still required to verify that the values correspond to normalized y.
    """
    if not isinstance(zone, dict):
        raise ValueError("zone must be an object")

    parsed = []
    for field in ("price_low", "price_high"):
        value = zone.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.strip()) > MAX_REPORTED_PRICE_TEXT_LENGTH
        ):
            raise ValueError(f"{field} must be a positive decimal string")
        text = value.strip()
        if _REPORTED_PRICE_PATTERN.fullmatch(text) is None:
            raise ValueError(f"{field} must be a plain decimal string")
        try:
            price = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"{field} must be a valid decimal price") from exc
        if not price.is_finite() or price <= 0:
            raise ValueError(f"{field} must be positive and finite")
        parsed.append(price)

    price_low, price_high = parsed
    if price_low >= price_high:
        raise ValueError("price_low must be lower than price_high")
    return price_low, price_high


@dataclass(frozen=True)
class PriceAxisCalibration:
    scale: str
    slope: float
    intercept: float | None
    min_y: float
    max_y: float
    anchor_count: int
    max_anchor_relative_error: float
    # New calibrations are evaluated around the anchor centroid instead of
    # extrapolating the fitted line to screenshot coordinate y=0. These
    # optional fields preserve compatibility with older direct constructors.
    reference_y: float | None = None
    reference_value: float | None = None

    def price_at_y(self, y: float) -> float:
        """Convert a normalized screenshot y-coordinate to price without extrapolation."""
        y = _finite_number(y, "y")
        if not 0 <= y <= 1:
            raise ValueError("y must be a finite normalized coordinate")
        if not isinstance(self.scale, str) or self.scale not in {"linear", "log"}:
            raise ValueError("calibration scale must be 'linear' or 'log'")
        min_y = _finite_number(self.min_y, "calibration min_y")
        max_y = _finite_number(self.max_y, "calibration max_y")
        if not 0 <= min_y < max_y <= 1:
            raise ValueError("calibration y range must be ordered within 0..1")
        slope = _finite_number(self.slope, "calibration slope")
        if slope >= 0:
            raise ValueError("calibration slope must be negative")
        intercept = (
            _finite_number(self.intercept, "calibration intercept")
            if self.intercept is not None else None
        )

        has_reference_y = self.reference_y is not None
        has_reference_value = self.reference_value is not None
        if has_reference_y != has_reference_value:
            raise ValueError(
                "centered calibration requires both reference_y and reference_value"
            )
        if has_reference_y:
            reference_y = _finite_number(self.reference_y, "calibration reference_y")
            reference_value = _finite_number(
                self.reference_value, "calibration reference_value"
            )
            if not min_y <= reference_y <= max_y:
                raise ValueError("calibration reference_y must be inside its anchor range")
            if self.scale == "linear" and reference_value <= 0:
                raise ValueError("linear calibration reference_value must be positive")
        else:
            reference_y = None
            reference_value = None
            # Compatibility path for calibrations constructed with the
            # original seven positional fields.
            if intercept is None:
                raise ValueError(
                    "calibration has no finite intercept or centered reference"
                )

        # Tolerate only tiny binary-rounding drift at an anchor edge. Clamp
        # that drift to the verified boundary instead of extrapolating.
        if y < min_y:
            if min_y - y <= _Y_TOLERANCE:
                y = min_y
            else:
                raise ValueError("y is outside the calibrated price-axis range")
        elif y > max_y:
            if y - max_y <= _Y_TOLERANCE:
                y = max_y
            else:
                raise ValueError("y is outside the calibrated price-axis range")

        try:
            if has_reference_y:
                # Evaluate locally around the anchor centroid. The legacy
                # intercept at y=0 may be outside float range even when every
                # price inside the calibrated screenshot region is representable.
                fitted = math.fsum((
                    reference_value,
                    slope * (y - reference_y),
                ))
            else:
                # Compatibility path for calibrations constructed with the
                # original seven positional fields.
                fitted = math.fsum((intercept, slope * y))
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "calibrated price is outside the supported numeric range"
            ) from exc

        if not math.isfinite(fitted):
            raise ValueError("calibrated price is outside the supported numeric range")
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

    # Linear prices can be close to the largest finite float. Normalize them
    # before least-squares arithmetic so summing several valid anchors does
    # not overflow. Log prices are already bounded by the float exponent range
    # and are left unscaled to preserve their available precision.
    value_scale = max(abs(value) for value in values) if scale == "linear" else 1.0
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("price-axis values cannot be normalized")
    fit_values = [value / value_scale for value in values]

    mean_y = math.fsum(ys) / len(ys)
    mean_value = math.fsum(fit_values) / len(fit_values)
    denominator = math.fsum((y - mean_y) ** 2 for y in ys)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("price-axis anchors do not span a usable y range")

    slope_scaled = math.fsum(
        (y - mean_y) * (value - mean_value)
        for y, value in zip(ys, fit_values)
    ) / denominator
    slope = slope_scaled * value_scale
    reference_value = mean_value * value_scale if scale == "linear" else mean_value
    if (
        not math.isfinite(slope)
        or not math.isfinite(reference_value)
        or (scale == "linear" and reference_value <= 0)
    ):
        raise ValueError("price-axis fit exceeds the supported numeric range")
    if slope >= 0:
        raise ValueError("price-axis fit is not a descending price scale")

    # Keep the old intercept available for ordinary ranges, but do not reject
    # a valid calibrated region merely because extrapolating to y=0 overflows.
    try:
        intercept = math.fsum((reference_value, -(slope * mean_y)))
    except (OverflowError, ValueError):
        intercept = None
    if intercept is not None and not math.isfinite(intercept):
        intercept = None

    calibration = PriceAxisCalibration(
        scale=scale,
        slope=slope,
        intercept=intercept,
        min_y=ys[0],
        max_y=ys[-1],
        anchor_count=len(points),
        max_anchor_relative_error=0.0,
        reference_y=mean_y,
        reference_value=reference_value,
    )
    relative_errors = []
    for y, observed_price in points:
        fitted_price = calibration.price_at_y(y)
        relative_error = abs(fitted_price - observed_price) / observed_price
        if not math.isfinite(relative_error):
            raise ValueError("price-axis calibration residual exceeds the supported numeric range")
        relative_errors.append(relative_error)
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
        reference_y=mean_y,
        reference_value=reference_value,
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
