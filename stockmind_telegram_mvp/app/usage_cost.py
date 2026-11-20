"""Explicit, date-bound currency estimates for provider-reported chart token usage."""
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re


_SCHEMA = "stockmind.usage-pricing.v1"
_RATE_FIELDS = {
    "input_usd_per_million_tokens",
    "cached_input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
}
_RATE_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?$")
_MONEY_QUANTUM = Decimal("0.000000000000000001")


def _parse_date(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must use YYYY-MM-DD format")
    return parsed


def _parse_rate(value, label):
    if not isinstance(value, str) or not _RATE_RE.fullmatch(value):
        raise ValueError(f"{label} must be a non-negative decimal string with at most 12 places")
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not a valid decimal rate") from exc
    if not rate.is_finite() or rate < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return rate


def validate_pricing_card(card):
    """Validate a user-supplied rate card; never embeds or fetches provider prices."""
    required = {
        "schema", "version", "currency", "effective_from", "effective_until",
        "source", "models",
    }
    if not isinstance(card, dict) or set(card) != required:
        raise ValueError("pricing card must contain exactly the documented fields")
    if card["schema"] != _SCHEMA:
        raise ValueError(f"pricing card schema must be {_SCHEMA}")
    version = card["version"]
    if not isinstance(version, str) or not version.strip() or len(version) > 80:
        raise ValueError("pricing card version must be a non-empty string of at most 80 characters")
    if card["currency"] != "USD":
        raise ValueError("pricing card currency must be USD")
    effective_from = _parse_date(card["effective_from"], "effective_from")
    effective_until = (
        _parse_date(card["effective_until"], "effective_until")
        if card["effective_until"] is not None else None
    )
    if effective_until is not None and effective_until < effective_from:
        raise ValueError("effective_until cannot precede effective_from")
    source = card["source"]
    if not isinstance(source, str) or not source.strip() or len(source) > 500:
        raise ValueError("pricing card source must be a non-empty string of at most 500 characters")
    raw_models = card["models"]
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("pricing card models must be a non-empty object")

    models = {}
    for model, raw_rates in raw_models.items():
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError("pricing card model keys must be non-empty exact model identifiers")
        if not isinstance(raw_rates, dict):
            raise ValueError(f"pricing rates for {model!r} must be an object")
        if set(raw_rates) - _RATE_FIELDS:
            raise ValueError(f"pricing rates for {model!r} contain unsupported fields")
        required_rates = {"input_usd_per_million_tokens", "output_usd_per_million_tokens"}
        if not required_rates.issubset(raw_rates):
            raise ValueError(f"pricing rates for {model!r} require input and output rates")
        models[model] = {
            key: _parse_rate(value, f"models.{model}.{key}")
            for key, value in raw_rates.items()
        }

    return {
        "schema": _SCHEMA,
        "version": version,
        "currency": "USD",
        "effective_from": effective_from,
        "effective_until": effective_until,
        "source": source.strip(),
        "models": models,
    }


def _unpriced(reason, model=None):
    return {"status": "unpriced", "reason": reason, "model": model}


def _money(value):
    rounded = value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    return format(rounded, "f")


def estimate_fixture_cost(fixture, pricing_card):
    """Estimate one fixture only when exact model, date and required usage are known."""
    metadata = fixture.get("prediction_metadata")
    if not isinstance(metadata, dict):
        return _unpriced("prediction_metadata_missing")
    model = metadata.get("model")
    if not isinstance(model, str) or not model:
        return _unpriced("provider_model_missing")
    rates = pricing_card["models"].get(model)
    if rates is None:
        return _unpriced("model_not_in_rate_card", model)

    generated_at = metadata.get("generated_at")
    if not isinstance(generated_at, str):
        return _unpriced("provider_timestamp_missing", model)
    try:
        timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return _unpriced("provider_timestamp_invalid", model)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return _unpriced("provider_timestamp_missing_timezone", model)
    generated_date = timestamp.date()
    if generated_date < pricing_card["effective_from"] or (
        pricing_card["effective_until"] is not None
        and generated_date > pricing_card["effective_until"]
    ):
        return _unpriced("outside_pricing_effective_period", model)

    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return _unpriced("provider_usage_missing", model)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    for value in (prompt, completion, total):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return _unpriced("provider_usage_incomplete", model)
    if total != prompt + completion:
        return _unpriced("provider_usage_inconsistent", model)

    cached_tokens = usage.get("cached_prompt_tokens")
    if cached_tokens is not None and (
        isinstance(cached_tokens, bool)
        or not isinstance(cached_tokens, int)
        or cached_tokens < 0
        or cached_tokens > prompt
    ):
        return _unpriced("cached_token_usage_invalid", model)

    input_rate = rates["input_usd_per_million_tokens"]
    cached_rate = rates.get("cached_input_usd_per_million_tokens")
    split_cached = cached_rate is not None and cached_rate != input_rate
    if split_cached and cached_tokens is None:
        return _unpriced("cached_token_usage_missing", model)

    if split_cached:
        standard_input_tokens = prompt - cached_tokens
        input_cost = Decimal(standard_input_tokens) * input_rate / Decimal(1_000_000)
        cached_cost = Decimal(cached_tokens) * cached_rate / Decimal(1_000_000)
    else:
        standard_input_tokens = prompt
        input_cost = Decimal(prompt) * input_rate / Decimal(1_000_000)
        cached_cost = Decimal(0)

    output_cost = Decimal(completion) * rates["output_usd_per_million_tokens"] / Decimal(1_000_000)
    total_cost = input_cost + cached_cost + output_cost
    return {
        "status": "priced",
        "model": model,
        "currency": "USD",
        "pricing_version": pricing_card["version"],
        "generated_date": generated_date.isoformat(),
        "standard_input_tokens": standard_input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": completion,
        "input_cost_usd": _money(input_cost),
        "cached_input_cost_usd": _money(cached_cost),
        "output_cost_usd": _money(output_cost),
        "total_cost_usd": _money(total_cost),
        "image_tokens_included_in_input": usage.get("image_tokens"),
        "reasoning_tokens_included_in_output": usage.get("reasoning_tokens"),
    }


def _aggregate_cost(estimates, fixture_count):
    priced = [row for row in estimates if row["status"] == "priced"]
    reasons = {}
    for row in estimates:
        if row["status"] != "priced":
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    total = sum((Decimal(row["total_cost_usd"]) for row in priced), Decimal(0))
    mean = total / len(priced) if priced else None
    return {
        "fixture_count": fixture_count,
        "priced_fixture_count": len(priced),
        "unpriced_fixture_count": fixture_count - len(priced),
        "unpriced_reasons": dict(sorted(reasons.items())),
        "total_cost_usd": _money(total) if priced else None,
        "mean_cost_usd_per_priced_fixture": _money(mean) if mean is not None else None,
    }


def _paired_cost_delta(baseline, candidate):
    deltas = []
    for fixture_id in sorted(baseline):
        left = baseline[fixture_id]
        right = candidate[fixture_id]
        if left["status"] == "priced" and right["status"] == "priced":
            deltas.append(
                Decimal(right["total_cost_usd"]) - Decimal(left["total_cost_usd"])
            )
    if not deltas:
        return {
            "paired_fixture_count": 0,
            "missing_pair_count": len(baseline),
            "total_delta_usd": None,
            "mean_delta_usd": None,
            "median_delta_usd": None,
            "min_delta_usd": None,
            "max_delta_usd": None,
        }
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median = (
        ordered[middle] if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / Decimal(2)
    )
    total = sum(deltas, Decimal(0))
    return {
        "paired_fixture_count": len(deltas),
        "missing_pair_count": len(baseline) - len(deltas),
        "total_delta_usd": _money(total),
        "mean_delta_usd": _money(total / len(deltas)),
        "median_delta_usd": _money(median),
        "min_delta_usd": _money(min(deltas)),
        "max_delta_usd": _money(max(deltas)),
    }


def compare_fixture_costs(baseline_fixtures, candidate_fixtures, raw_pricing_card):
    """Return date-bound cost summaries and per-fixture estimates for a paired run."""
    card = validate_pricing_card(raw_pricing_card)
    baseline = {
        fixture["fixture_id"]: estimate_fixture_cost(fixture, card)
        for fixture in baseline_fixtures
    }
    candidate = {
        fixture["fixture_id"]: estimate_fixture_cost(fixture, card)
        for fixture in candidate_fixtures
    }
    fixture_details = {}
    for fixture_id in sorted(baseline):
        left, right = baseline[fixture_id], candidate[fixture_id]
        delta = None
        if left["status"] == "priced" and right["status"] == "priced":
            delta = _money(
                Decimal(right["total_cost_usd"]) - Decimal(left["total_cost_usd"])
            )
        fixture_details[fixture_id] = {
            "baseline": left,
            "candidate": right,
            "candidate_minus_baseline_usd": delta,
        }

    summary = {
        "pricing_card": {
            "schema": card["schema"],
            "version": card["version"],
            "currency": card["currency"],
            "effective_from": card["effective_from"].isoformat(),
            "effective_until": (
                card["effective_until"].isoformat()
                if card["effective_until"] is not None else None
            ),
            "source": card["source"],
        },
        "baseline": _aggregate_cost(list(baseline.values()), len(baseline)),
        "candidate": _aggregate_cost(list(candidate.values()), len(candidate)),
        "paired_deltas": _paired_cost_delta(baseline, candidate),
    }
    return summary, fixture_details
