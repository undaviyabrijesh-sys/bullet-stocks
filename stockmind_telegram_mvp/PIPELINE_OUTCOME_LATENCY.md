# Telegram upload latency by outcome

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

The handler-total distribution includes all parsed handler completion events
with an explicit supported outcome, including legacy records without request
IDs. Download distributions include only non-conflicted request IDs that have
both a final handler outcome and exactly one parsed download event. This
deliberately favors trustworthy pairing over sample coverage.

Run the report with:

```bash
python -m stockmind_telegram_mvp.app.pipeline_telemetry /path/to/stockmind.log > latency-report.json
```

Outcome-specific distributions are descriptive only. Small samples, especially
p95, are not latency guarantees. The feature does not infer missing stage
durations, does not make stage durations additive, and does not claim a
production improvement until representative application logs are analyzed.
