# Strict JSON policy

StockMind uses a shared strict JSON decoder for untrusted model output, model
tool arguments, annotation drafts, prediction captures and benchmark reports.

The decoder rejects:

- Duplicate keys in any object, including nested objects.
- The non-standard constants `NaN`, `Infinity` and `-Infinity`.
- Exponent-form JSON numbers that Python would otherwise convert to a
  non-finite float (for example, `1e999`).
- Non-zero JSON numbers that Python would silently underflow to `0.0`
  (for example, `1e-9999`), which could collapse a chart coordinate or tool value.
- JSON nesting that exceeds the decoder's supported recursion depth,
  normalized to the same controlled `StrictJSONError` used for other invalid input.

The last case matters because `1e999` is valid JSON number syntax, but Python's
default float conversion produces positive infinity. Without an explicit
finite-value check, an argument such as `{"limit":1e999}` could pass the
strict parser and reach a market-data provider as an infinite numeric value.

The reverse range boundary matters too: a non-zero value such as `1e-9999`
is valid JSON, but Python's float conversion silently rounds it to `0.0`.
For normalized chart coordinates or numeric tool arguments, that can turn a
small non-zero value into an exact edge value. StockMind rejects only non-zero
tokens that underflow all the way to zero; exact zero and representable
subnormal floats remain supported.

Finite exponent-form values remain supported. Ordinary strings containing
words such as `"NaN"` or `"Infinity"` are also valid.

For model tool calls, the raw JSON argument payload is limited to 8 KiB
of UTF-8 before JSON decoding. Oversized or invalid-Unicode payloads are rejected
without invoking the decoder, limiting parser work and preventing unnecessarily
large model-generated object graphs from reaching schema validation. The limit is
generous relative to the flat tool schemas and their individual field limits.

Parsing and local schema validation happen before dispatch. StockMind checks
required fields, rejects undeclared fields, validates
primitive types (including rejecting booleans where an integer is required),
enforces enum choices, numeric bounds and schema-defined string length limits, then applies
provider defaults. Search queries are limited to 120 characters, sector/industry filters
to 80 characters, stock/crypto symbols to 32 characters, and intraday date strings
to 10 characters. Overlong values are rejected rather than silently truncated.
Provider-side function schemas are guidance rather than the sole trust
boundary. Invalid arguments are returned to the model as an
`InvalidToolArguments` result and do not invoke a market-data provider.

For `get_intraday_stock_bars`, local validation also enforces the cross-field
date contract before dispatch: supply either one `date` or both `from_date`
and `to_date`, never both modes; dates must be exact valid `YYYY-MM-DD`
calendar dates; the range must be ordered and may end at most seven calendar
days after its start. This prevents incomplete, ambiguous or oversized
historical requests from consuming provider calls.

Outbound market-data results are also serialized with non-finite floats disabled.
Python's default JSON encoder otherwise emits non-standard `NaN` and
`Infinity` tokens. If a provider response contains a non-finite numeric value,
StockMind withholds that result and returns a bounded
`ToolResultSerializationError` payload instead of passing ambiguous numeric
data into model context. Finite large values remain valid JSON numbers.

Run the focused tests from the repository root:

```bash
python -m unittest stockmind_telegram_mvp.tests.test_strict_json -v
```

The decoder does not authenticate data provenance or validate provider
semantics; it only ensures that JSON has an unambiguous structure and that
parsed floating-point values remain finite.
