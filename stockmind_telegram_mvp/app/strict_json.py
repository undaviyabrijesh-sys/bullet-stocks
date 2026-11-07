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
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def loads(value):
    """Decode JSON, rejecting duplicate keys and non-finite numeric values."""
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
