"""Strict JSON decoding for untrusted model and benchmark inputs.

Python's default JSON decoder silently keeps the last duplicate object key and
accepts NaN/Infinity, neither of which is valid interoperable JSON. Reject both
so screenshots, predictions and benchmark evidence have one unambiguous parse.
"""

import json


class StrictJSONError(ValueError):
    """Raised when JSON uses ambiguous or non-standard values."""


def _reject_constant(token):
    raise StrictJSONError(f"non-standard JSON constant {token!r} is not permitted")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def loads(value):
    """Decode JSON while rejecting duplicate keys and NaN/Infinity constants."""
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
