# Telegram upload latency by outcome

The current report schema is version 4. Consumers should check `schema_version`
when reading generated JSON because outcome-conflict accounting is explicit in
this version.

Timing values are accepted up to `2**52` milliseconds. The report computes
p50 as the arithmetic median; this bound keeps the sum of any two accepted
values at or below `2**53`, so integer and half-millisecond medians remain
exact in common IEEE-754 JSON consumers. Larger values are rejected and
counted in `rejected_timing_lines`. This is a reporting-precision guard, not
an operational timeout.

The offline pipeline report includes `telegram_handler_outcome_latency`,
grouped by upload kind (`photo` or `document`) and the bounded final handler
outcome. Each group reports p50, nearest-rank p95, minimum, maximum and sample
count for `handler_total_ms`.

When a download event can be safely joined to the same request ID, the group
also reports a separate `download_ms` distribution. Download timing events do
not carry an outcome at emission time; the final outcome is attached only
after correlation by opaque request ID. Events are never paired by log-line
proximity.

`download_not_correlated_count` is the difference between parsed handler
completion samples and cleanly correlated download samples for that
kind/outcome group. It can include early rejections that never attempted a
download, legacy logs without request IDs, and requests excluded because of
duplicate or conflicting timing events. It is not a count of failed downloads.

The handler-total distribution includes each legacy completion event
with an explicit supported outcome and one unambiguous completion per
correlated request ID. A request ID is excluded from outcome counts and
handler-total distributions when it contains duplicate handler-total/outcome
events or inconsistent upload kinds that affect a recorded outcome. The report
exposes this count as
`conflicted_handler_outcome_request_id_count`; it is narrower than
`conflicted_request_id_count`, which also includes conflicts that
affect only request or download correlation.

A duplicate or conflicting download event makes download pairing ambiguous,
but does not discard a unique handler-total/outcome sample. Download
distributions include only request IDs with one unambiguous upload kind, one
final handler outcome, one handler-total event and exactly one parsed download
event. This deliberately favors trustworthy pairing over sample coverage.

Run the report with:

```bash
python -m stockmind_telegram_mvp.app.pipeline_telemetry /path/to/stockmind.log > latency-report.json
```

Outcome-specific distributions are descriptive only. Small samples, especially
p95, are not latency guarantees. The feature does not infer missing stage
durations, does not make stage durations additive, and does not claim a
production improvement until representative application logs are analyzed.
