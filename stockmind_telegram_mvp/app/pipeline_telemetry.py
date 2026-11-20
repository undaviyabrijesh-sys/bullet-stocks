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

_REQUEST_RE = re.compile(
    r"StockMind request timing mode=(?P<mode>[a-z][a-z0-9_]*) "
    r"outcome=(?P<outcome>[a-z][a-z0-9_]*) "
    r"total_ms=(?P<total>[0-9]+) stages=(?P<stages>.*)$"
)
_STAGE_RE = re.compile(r"(?P<name>[a-z][a-z0-9_]*)=(?P<duration>[0-9]+)ms")
_HANDLER_RE = re.compile(
    r"StockMind Telegram (?P<kind>photo|document) "
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
        return {
            "kind": "request",
            "mode": request_match.group("mode"),
            "outcome": request_match.group("outcome"),
            "total_ms": total_ms,
            "stages": stages,
        }

    handler_match = _HANDLER_RE.search(line.rstrip("\r\n"))
    if handler_match is not None:
        duration_ms = _duration(handler_match.group("duration"))
        if duration_ms is None:
            return None
        return {
            "kind": "handler",
            "event": (
                f"telegram_{handler_match.group('kind')}_"
                f"{handler_match.group('metric')}"
            ),
            "duration_ms": duration_ms,
        }
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
    """Aggregate request and upload-handler timing events without retaining raw lines."""
    request_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    handler_samples: dict[str, list[int]] = defaultdict(list)
    parsed_requests = 0
    parsed_handlers = 0
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

        if event["kind"] == "request":
            request_groups[(event["mode"], event["outcome"])].append(event)
            parsed_requests += 1
        else:
            handler_samples[event["event"]].append(event["duration_ms"])
            parsed_handlers += 1

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

    return {
        "schema_version": 1,
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
