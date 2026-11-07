# Strict JSON policy

StockMind uses a shared strict JSON decoder for untrusted model output, model
tool arguments, annotation drafts, prediction captures and benchmark reports.

The decoder rejects:

- Duplicate keys in any object, including nested objects.
- The non-standard constants `NaN`, `Infinity` and `-Infinity`.
- Exponent-form JSON numbers that Python would otherwise convert to a
  non-finite float (for example, `1e999`).

The last case matters because `1e999` is valid JSON number syntax, but Python's
default float conversion produces positive infinity. Without an explicit
finite-value check, an argument such as `{"limit":1e999}` could pass the
strict parser and reach a market-data provider as an infinite numeric value.

Finite exponent-form values remain supported. Ordinary strings containing
words such as `"NaN"` or `"Infinity"` are also valid.

For model tool calls, parsing happens before dispatch. Invalid arguments are
returned to the model as an `InvalidToolArguments` result and do not invoke a
market-data provider.

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
