"""Strict JSON decoding for untrusted model and benchmark inputs.

Python's default JSON decoder silently keeps the last duplicate object key,
accepts NaN/Infinity constants, and converts sufficiently large exponent-form
numbers (for example 1e999) to float infinity. Reject these ambiguous or
non-finite values so model requests and benchmark evidence have one parse.
"""

import json
import math


class StrictJSONError(ValueError):
    """Raised when JSON uses ambiguous or unsupported numeric values."""


def _reject_constant(token):
    raise StrictJSONError(f"non-standard JSON constant {token!r} is not permitted")


def _finite_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise StrictJSONError(
            f"JSON number {token!r} is outside the supported finite numeric range"
        )
    # float() can also silently underflow a non-zero JSON number to 0.0.
    # That is dangerous for normalized chart coordinates and numeric tool
    # arguments because a tiny but intentional value becomes an exact boundary.
    if value == 0.0:
        mantissa = token.lower().split("e", 1)[0]
        if any(digit in "123456789" for digit in mantissa):
            raise StrictJSONError(
                f"JSON number {token!r} underflows the supported finite numeric range"
            )
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def loads(value):
    """Decode strict JSON and normalize excessive decoder nesting."""
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except RecursionError as exc:
        raise StrictJSONError(
            "JSON nesting exceeds the supported parser depth"
        ) from exc
