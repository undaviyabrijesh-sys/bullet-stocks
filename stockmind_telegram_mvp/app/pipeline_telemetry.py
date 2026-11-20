"""Offline aggregation for StockMind's privacy-minimized Telegram timing logs.

The parser deliberately extracts only known timing fields. It never copies raw
log lines into the report, so unrelated logger context is not exported.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from statistics import median
from typing import Iterable

_MAX_DURATION_MS = (1 << 63) - 1
_REQUEST_MARKER = "StockMind request timing "
_HANDLER_MARKER = "StockMind Telegram "
_HANDLER_OUTCOMES = frozenset({
    "completed", "delivery_failed", "timeout", "incomplete_response",
    "invalid_image", "research_failed", "rejected", "download_failed",
    "request_busy", "capacity_timeout", "storage_unavailable",
    "handler_error", "unknown",
})

_REQUEST_RE = re.compile(
    r"StockMind request timing "
    r"(?:request_id=(?P<request_id>[0-9a-f]{32}) )?"
    r"mode=(?P<mode>[a-z][a-z0-9_]*) "
    r"outcome=(?P<outcome>[a-z][a-z0-9_]*) "
    r"total_ms=(?P<total>[0-9]+) stages=(?P<stages>.*)$"
)
_STAGE_RE = re.compile(r"(?P<name>[a-z][a-z0-9_]*)=(?P<duration>[0-9]+)ms")
_HANDLER_RE = re.compile(
    r"StockMind Telegram (?P<kind>photo|document) "
    r"(?:request_id=(?P<request_id>[0-9a-f]{32}) )?"
    r"(?:outcome=(?P<outcome>[a-z][a-z0-9_]*) )?"
    r"(?P<metric>download_ms|handler_total_ms)=(?P<duration>[0-9]+)\s*$"
)


def _duration(value: str) -> int | None:
    """Parse a bounded non-negative millisecond count."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0 or parsed > _MAX_DURATION_MS:
        return None
    return parsed


def _parse_stages(raw: str) -> dict[str, int] | None:
    if raw == "none":
        return {}
    if not raw:
        return None

    stages: dict[str, int] = {}
    for token in raw.split():
        match = _STAGE_RE.fullmatch(token)
        if match is None:
            return None
        name = match.group("name")
        duration = _duration(match.group("duration"))
        if duration is None or name in stages:
            return None
        stages[name] = duration
    return stages


def parse_timing_line(line: str) -> dict | None:
    """Extract one recognized timing event, or return None for other/malformed lines."""
    if not isinstance(line, str):
        return None

    request_match = _REQUEST_RE.search(line.rstrip("\r\n"))
    if request_match is not None:
        total_ms = _duration(request_match.group("total"))
        stages = _parse_stages(request_match.group("stages"))
        if total_ms is None or stages is None:
            return None
        event = {
            "kind": "request",
            "mode": request_match.group("mode"),
            "outcome": request_match.group("outcome"),
            "total_ms": total_ms,
            "stages": stages,
        }
        request_id = request_match.group("request_id")
        if request_id is not None:
            event["request_id"] = request_id
        return event

    handler_match = _HANDLER_RE.search(line.rstrip("\r\n"))
    if handler_match is not None:
        duration_ms = _duration(handler_match.group("duration"))
        metric = handler_match.group("metric")
        outcome = handler_match.group("outcome")
        if duration_ms is None:
            return None
        if outcome is not None and (
            metric != "handler_total_ms" or outcome not in _HANDLER_OUTCOMES
        ):
            return None
        event = {
            "kind": "handler",
            "event": f"telegram_{handler_match.group('kind')}_{metric}",
            "duration_ms": duration_ms,
            "upload_kind": handler_match.group("kind"),
            "metric": metric,
        }
        request_id = handler_match.group("request_id")
        if request_id is not None:
            event["request_id"] = request_id
        if outcome is not None:
            event["outcome"] = outcome
        return event
    return None


def _summarize(values: list[int]) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("cannot summarize an empty sample")

    # p50 is the conventional median; p95 uses the nearest-rank method.
    middle = count // 2
    p50 = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    p95_rank = (95 * count + 99) // 100
    return {
        "sample_count": count,
        "p50_ms": p50,
        "p95_ms": ordered[p95_rank - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def aggregate_timing_lines(lines: Iterable[str]) -> dict:
    """Aggregate timings and correlate upload/request events by opaque request ID."""
    request_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    handler_samples: dict[str, list[int]] = defaultdict(list)
    correlations: dict[str, dict] = {}
    conflicted_request_ids: set[str] = set()
    parsed_requests = 0
    parsed_handlers = 0
    legacy_request_events = 0
    legacy_handler_events = 0
    legacy_handler_outcome_events = 0
    handler_outcome_counts: dict[str, int] = defaultdict(int)
    handler_outcome_total_samples: dict[tuple[str, str], list[int]] = defaultdict(list)
    handler_outcome_download_samples: dict[tuple[str, str], list[int]] = defaultdict(list)
    rejected_timing_lines = 0

    for line in lines:
        event = parse_timing_line(line)
        if event is None:
            if isinstance(line, str) and (
                _REQUEST_MARKER in line
                or (
                    _HANDLER_MARKER in line
                    and ("download_ms=" in line or "handler_total_ms=" in line)
                )
            ):
                rejected_timing_lines += 1
            continue

        request_id = event.get("request_id")
        if event["kind"] == "request":
            request_groups[(event["mode"], event["outcome"])].append(event)
            parsed_requests += 1
            if request_id is None:
                legacy_request_events += 1
            else:
                record = correlations.setdefault(
                    request_id,
                    {"request": None, "upload_kind": None, "handler_metrics": {}, "outcome": None},
                )
                if record["request"] is not None:
                    conflicted_request_ids.add(request_id)
                else:
                    record["request"] = event
        else:
            handler_samples[event["event"]].append(event["duration_ms"])
            parsed_handlers += 1
            metric = event["metric"]
            outcome = event.get("outcome")
            if metric == "handler_total_ms":
                if outcome is None:
                    legacy_handler_outcome_events += 1
                else:
                    handler_outcome_counts[outcome] += 1
                    handler_outcome_total_samples[
                        (event["upload_kind"], outcome)
                    ].append(event["duration_ms"])
            if request_id is None:
                legacy_handler_events += 1
            else:
                record = correlations.setdefault(
                    request_id,
                    {"request": None, "upload_kind": None, "handler_metrics": {}, "outcome": None},
                )
                if (
                    record["upload_kind"] is not None
                    and record["upload_kind"] != event["upload_kind"]
                ):
                    conflicted_request_ids.add(request_id)
                record["upload_kind"] = event["upload_kind"]
                if outcome is not None:
                    if record["outcome"] is not None:
                        conflicted_request_ids.add(request_id)
                    else:
                        record["outcome"] = outcome
                if metric in record["handler_metrics"]:
                    conflicted_request_ids.add(request_id)
                else:
                    record["handler_metrics"][metric] = event["duration_ms"]

    groups = []
    for (mode, outcome), events in sorted(request_groups.items()):
        totals = [event["total_ms"] for event in events]
        stage_names = sorted(
            {name for event in events for name in event["stages"]}
        )
        stages = {}
        for name in stage_names:
            values = [
                event["stages"][name]
                for event in events
                if name in event["stages"]
            ]
            summary = _summarize(values)
            summary["missing_count"] = len(events) - len(values)
            stages[name] = summary
        groups.append(
            {
                "mode": mode,
                "outcome": outcome,
                "request_count": len(events),
                "total_ms": _summarize(totals),
                "stages": stages,
            }
        )

    correlated_requests = []
    complete_upload_requests = 0
    for request_id, record in sorted(correlations.items()):
        if request_id in conflicted_request_ids:
            continue
        request = record["request"]
        handler_metrics = record["handler_metrics"]
        if (
            request is not None
            and request["mode"] == "chart"
            and record["upload_kind"] is not None
            and {"download_ms", "handler_total_ms"} <= set(handler_metrics)
        ):
            complete_upload_requests += 1
        outcome = record["outcome"]
        if (
            record["upload_kind"] is not None
            and outcome is not None
            and "handler_total_ms" in handler_metrics
            and "download_ms" in handler_metrics
        ):
            handler_outcome_download_samples[
                (record["upload_kind"], outcome)
            ].append(handler_metrics["download_ms"])
        handler_summary = None
        if handler_metrics:
            handler_summary = {
                "kind": record["upload_kind"],
                **handler_metrics,
            }
            if record["outcome"] is not None:
                handler_summary["outcome"] = record["outcome"]
        correlated_requests.append(
            {
                "request_id": request_id,
                "request": (
                    None
                    if request is None
                    else {
                        "mode": request["mode"],
                        "outcome": request["outcome"],
                        "total_ms": request["total_ms"],
                        "stages": request["stages"],
                    }
                ),
                "telegram_handler": handler_summary,
            }
        )

    outcome_latency_groups = []
    outcome_keys = sorted(
        set(handler_outcome_total_samples) | set(handler_outcome_download_samples)
    )
    for upload_kind, outcome in outcome_keys:
        total_values = handler_outcome_total_samples.get((upload_kind, outcome), [])
        download_values = handler_outcome_download_samples.get((upload_kind, outcome), [])
        group = {
            "upload_kind": upload_kind,
            "outcome": outcome,
            "handler_total_ms": _summarize(total_values),
            "correlated_download_sample_count": len(download_values),
            "download_not_correlated_count": max(
                0, len(total_values) - len(download_values)
            ),
        }
        if download_values:
            group["download_ms"] = _summarize(download_values)
        outcome_latency_groups.append(group)

    return {
        "schema_version": 3,
        "percentile_method": {
            "p50": "median",
            "p95": "nearest_rank",
        },
        "parsed_request_events": parsed_requests,
        "parsed_handler_events": parsed_handlers,
        "rejected_timing_lines": rejected_timing_lines,
        "request_groups": groups,
        "telegram_handler_metrics": {
            name: _summarize(values)
            for name, values in sorted(handler_samples.items())
        },
        "correlation_summary": {
            "request_ids_seen": len(correlations),
            "correlated_request_count": len(correlated_requests),
            "complete_upload_request_count": complete_upload_requests,
            "conflicted_request_id_count": len(conflicted_request_ids),
            "legacy_request_event_count": legacy_request_events,
            "legacy_handler_event_count": legacy_handler_events,
            "handler_outcome_event_count": sum(handler_outcome_counts.values()),
            "legacy_handler_outcome_event_count": legacy_handler_outcome_events,
        },
        "telegram_handler_outcomes": dict(sorted(handler_outcome_counts.items())),
        "telegram_handler_outcome_latency": outcome_latency_groups,
        "correlated_requests": correlated_requests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate privacy-minimized StockMind Telegram timing logs."
    )
    parser.add_argument(
        "log_file",
        help="application log file to read, or '-' to read standard input",
    )
    args = parser.parse_args(argv)

    if args.log_file == "-":
        report = aggregate_timing_lines(sys.stdin)
    else:
        try:
            with open(args.log_file, "r", encoding="utf-8", errors="replace") as handle:
                report = aggregate_timing_lines(handle)
        except OSError as exc:
            parser.exit(2, f"error: cannot read timing log: {exc}\n")

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
