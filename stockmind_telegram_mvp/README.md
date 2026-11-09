# StockMind AI — Telegram MVP

A multilingual Telegram research assistant for Indian equities and crypto, including intraday chart analysis. Users can ask in English, Hinglish, Hindi, Gujarati or mixed/broken language. The AI selects the appropriate Tapetide or Binance read-only data tools and responds in the user's language.

## Included
- Telegram bot (aiogram 3)
- OpenAI tool-calling for intent understanding and research planning (default model: GPT-6 Luna)
- Tapetide REST adapter: stock search, latest/session-live quote, financial statements, daily history and historical 1/5/15-minute intraday bars
- Binance public REST adapter: crypto spot and USD-M futures 24-hour ticker and OHLCV candles
- Per-user conversational context
- Prototype cap of 10 beta users and 10 questions per user per day
- Intraday analysis for Indian cash equities (historical 1/5/15-minute candles) and crypto (Binance spot/USD-M futures)
- Telegram chart screenshot analysis using a vision-capable OpenAI model (photo uploads and JPEG/PNG/WEBP image documents)
- Visual DNS zone assessment: distinct supply/demand bands, user-drawn zone review, freshness/mitigation, structure and conditional scenarios
- No order placement or trade execution

## Setup

The runtime requires Python 3.11 or newer.

1. Create a bot with Telegram's @BotFather.
2. Create an OpenAI API key.
3. Create a Tapetide token and confirm that your plan permits the intended beta data display and caching.
4. Enter the project directory and copy .env.example to .env.
5. Install dependencies and run:

```bash
cd stockmind_telegram_mvp
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.bot
```

Windows activation: `.venv\\Scripts\\activate`.

## Configuration
- TELEGRAM_BOT_TOKEN
- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-6-luna; Chat Completions function calling is configured with reasoning_effort=none)
- TAPETIDE_TOKEN
- DAILY_QUERY_LIMIT (default: 10)
- BETA_USER_LIMIT (default: 10)
- MAX_HISTORY_MESSAGES (default: 12)
- OPENAI_VISION_MODEL (defaults to OPENAI_MODEL; must support image input and structured outputs; default GPT-6 Luna)
- MAX_CHART_IMAGE_BYTES (default: 5 MB)
- RESEARCH_TURN_TIMEOUT_SECONDS (default: 120 seconds; finite positive number)

## Provider request bounds

The OpenAI client uses an explicit 90-second HTTPX timeout for each request,
with a 10-second connection timeout and at most one SDK retry. In addition,
each text research turn has one overall deadline controlled by
`RESEARCH_TURN_TIMEOUT_SECONDS` (default 120 seconds). The same deadline
covers every model round and every awaited market-data provider call; it is
not reset between steps. When it expires, StockMind cancels the in-flight
coroutine and does not start further model or market-data calls. The caller
receives a clear timeout response rather than a partial answer presented as
complete. A provider may already have received an in-flight request, so
cancellation does not guarantee that upstream processing or billing stopped.
A provider's own timeout is preserved as a provider failure rather than being
misreported as the overall turn deadline. Chart-image analysis also applies the
same overall deadline around its vision request, in addition to the per-request
OpenAI timeout. Whichever local deadline expires first stops the coroutine;
a chart timeout is reported as an incomplete analysis and is not saved as a
successful assistant turn. Text-turn deadline exceptions are also propagated
to Telegram and are not appended to conversation history as completed answers.
Cancellation cannot guarantee that an upstream
provider stopped processing or billing a request it had already received.

## Chart screenshot analysis

Send a chart screenshot as a Telegram photo or JPEG/PNG/WEBP image document, with a caption such as "BTC 1H — mark demand and supply" or "Is my demand zone still fresh?". Keep the price scale, timeframe and enough candles visible; include the symbol, exchange and timeframe in the caption when they are not shown.

Before any vision request, the bot checks the uploaded file's actual encoded format, fully decodes its pixel data, performs a separate container-integrity check, and enforces a 25-megapixel decoded-image ceiling rather than trusting its declared MIME type. Telegram downloads are streamed into a bounded in-memory buffer capped at `MAX_CHART_IMAGE_BYTES`, so missing or inaccurate Telegram file-size metadata cannot cause an unbounded image download before the limit is enforced. Only static JPEG, PNG and WEBP images are accepted; uploads that fail full pixel decoding or the container-integrity check, animated images and oversized uploads are rejected before they consume an AI request. The overlay renderer draws at most eight successfully validated zones; malformed or out-of-plot candidates do not consume that limit. It also ignores non-finite, boolean, out-of-range and sub-pixel box coordinates so malformed geometry is not presented as a precise zone. The structured response schema also bounds every normalized coordinate to 0–1 and caps the response at eight zones. It requires the model to identify the main price-plot rectangle separately from the price scale, toolbars and adjacent indicator/volume panels. Every drawable zone must fit wholly inside that rectangle; if the plot boundary is missing or uncertain, overlays are withheld. The local response guard independently validates the plot frame, coordinate ordering and band height, removes malformed zone objects, and disables drawing (with zeroed coordinates) whenever geometry is invalid. The renderer rechecks the same plot boundary in pixel space before drawing, so a zone cannot spill into the price scale or a neighbouring panel due to coordinate rounding. Readable analysis text is retained when only the overlay geometry fails.

The vision model assesses visible chart structure and returns distinct supply/demand price bands, likely mitigation state, visible BOS/CHOCH or liquidity events, and conditional scenarios with invalidation. Existing user-drawn zones are assessed first and kept distinct from additional inferred zones. Numeric bounds are only reported when the chart scale is readable; otherwise the model must say that the level cannot be read reliably.

The bot now returns a second image with translucent supply (red) and demand (green) bands when the vision model can locate the zones confidently. Short English labels preserve the distinction between clearly user-marked zones and AI-inferred zones (USER SUPPLY/DEMAND versus AI SUPPLY/DEMAND); unclear origins must not be presented as user-drawn. It also returns the written zone map. The bot edits its progress message to deliver the completed analysis; if Telegram explicitly rejects that edit (for example, because the progress message is no longer editable), it falls back to a new reply. For ambiguous network failures it does not send a second copy, because Telegram may already have applied the edit. Failure to deliver an optional overlay remains independent of the written analysis. Each rectangle is positioned from normalized screenshot coordinates estimated by the vision model; it is not pixel-perfect calibration to exchange OHLC data, so users must verify its position against the visible price scale. When no overlay is drawn, the Telegram response distinguishes a missing main-plot boundary, no distinct zone candidate, model-declined geometry and candidates rejected by final pixel/plot checks. This does not claim that no market zone exists; it explains why no image was drawn. The bot intentionally withholds the overlay rather than drawing a guessed box. This is visual screenshot research, not a live chart connection. The screenshot may be delayed or cropped, freshness may be indeterminate if prior candles are not visible, and chart-vision models can misread small labels or chart graphics. Verify levels against an OHLC/live market feed before relying on any alert or trading decision. Image analysis uses more input tokens than text-only questions. GPT-6 Luna is selected for its low token cost and support for image input, Chat Completions function calling (with reasoning effort set to none), and structured outputs.

## Intraday data boundaries
- Tapetide's Indian-equity intraday bars are historical and update after the session closes; they are not a live candle feed. A separate session-live quote route is integrated for supported stocks; availability/coverage and provider timestamp must be checked on every request.
- Binance public ticker/candle endpoints provide crypto market data without API credentials. The latest candle may still be forming; the assistant must label it accordingly. Server-time samples are cached independently for spot and USD-M futures for 30 seconds. If a time request fails or returns malformed data, the adapter briefly caches the unverified state for 5 seconds to avoid repeating the failing time request on every kline call; candles remain unverified throughout that cooldown, and the adapter retries after it expires. It never falls back to the machine wall clock.
- This version does not provide a verified live Indian index/stock-options feed, broker execution, order routing, or exchange-grade streaming. Do not present delayed or end-of-session candles as real-time.
- Indian cash-market session reference: 09:15–15:30 IST. Crypto markets trade 24/7.

## Important prototype limitations
- User membership, daily usage and chat history are in memory. They reset on restart and are not suitable for a public beta until moved to persistent storage. Beta admission and daily-slot consumption are synchronous check-and-update operations on the bot's single asyncio event loop; research turns from the same user are serialized with a per-user lock. Photo/document handlers perform a non-consuming beta-cap and daily-quota preflight before downloading chart bytes, and the research handler repeats that preflight before image decoding. The checks are advisory only: final beta admission and daily-slot consumption are rechecked after image validation, so concurrent requests cannot bypass the process-local cap. Invalid images do not consume a daily slot; a request consumes one once it passes validation and admission, before provider work begins, so provider failures also count against the cap and repeated retries cannot bypass the spend guard. The lock registry counts active holders and queued waiters, then removes an entry after the last request exits; `/reset` uses the same lock and does not create history for an unknown user. These protections are process-local only—multiple bot workers require shared persistence, atomic quota/admission updates and a distributed per-user lock.
- Tapetide's free quota is limited and shared across REST and MCP. Each research turn is now bounded to at most 6 model-requested market-data tool calls, 12 KiB per serialized tool result and 36 KiB of serialized tool results total. Oversized time-series arrays are sampled from the end of the provider's returned order (stock search keeps the first matches) and carry explicit truncation metadata; oversized non-list payloads are withheld rather than passed as partial data. Within one research turn, duplicate requests are deduplicated by tool name and effective arguments after dispatcher defaults are applied. Symbol values are trimmed and uppercased where the provider does the same, and stock-search text is trimmed without case folding. Later duplicate calls receive a compact reference to the first tool result. Duplicate model calls still count against the six-call ceiling, and this cache is not shared across turns or users. These are per-turn context/call guards, not a provider-wide daily quota, request cache or guarantee that the upstream free plan will not be exhausted.
- Do not assume a free API plan permits redistribution. Obtain written provider permission for any external beta or paid service.
- Paid stock research/recommendations can raise Indian regulatory obligations. Get qualified compliance advice before monetisation.
- Keep .env out of Git. Never commit tokens or API keys.

## Data integrity

### Binance numeric payload validation

The read-only Binance client validates market-data values before passing them to the research model. Kline timestamps and trade counts must be non-negative integers, candle open times must be strictly increasing, OHLC values must be positive finite decimal strings with a consistent high/low envelope, and base/quote volumes must be finite and non-negative. Each kline response includes a conservative UTC `as_of_ms` estimate plus `clock_source`, `clock_uncertainty_ms` and per-candle `is_closed`/`candle_status` fields. Candle confirmation uses the matching Binance server-time endpoint (spot `/api/v3/time`, USD-M `/fapi/v1/time`) rather than the machine wall clock. The server-time sample is cached for up to 30 seconds per market; the estimate subtracts the observed full request round-trip time, one millisecond for timestamp resolution and a modeled 100 ppm drift allowance while cached. A candle is confirmed closed only when its `close_time` is strictly earlier than that conservative estimate. If server-time retrieval fails or returns malformed data, candles remain available but are marked `unverified` with `is_closed=false`; the model must not call them confirmed. The 24-hour ticker must identify the requested symbol and contain a positive finite last price. Malformed successful HTTP responses are treated as provider failures rather than plausible market data. This validates response structure and numeric consistency; it does not independently verify exchange accuracy, clock synchronisation or market freshness.
At the ResearchAI tool boundary, Binance candle closure is independently recomputed from a positive integer `as_of_ms` and each row's integer `close_time`; only `close_time < as_of_ms` is marked closed. If the clock source or timestamp is missing or invalid, every candle is forced to `is_closed=false` and `candle_status=unverified`, even if an upstream adapter supplied a conflicting closed flag. This defense-in-depth check prevents an adapter regression or malformed test/provider result from turning an unverified candle into confirmed BOS/CHOCH evidence. It does not prove that the model will always follow the resulting evidence, so final analysis still needs review.\n\nMalformed or non-object model tool arguments are contained as explicit tool errors and returned to the model for recovery; they do not terminate the user's research turn before a response can be produced. Tapetide responses must contain a JSON object with a `data` field and object-shaped metadata; a non-JSON success page, JSON array, missing data envelope or malformed metadata is treated as a provider error rather than an empty successful result. Binance candle responses must be arrays of structurally valid kline rows, and ticker responses must identify the requested symbol and include a last price.

Provider failures sent back to the model are normalized: timeout, connection and HTTP-status errors do not forward raw request URLs, and unexpected exceptions are not copied verbatim into model context. Tapetide error codes and local argument-validation messages remain available for recovery. If a result contains truncation metadata, the model is instructed to treat it as a partial sample and not infer omitted rows. These guards improve response integrity but do not validate the correctness or freshness of provider market data.

Answers must show provider timestamps when available, distinguish missing/stale data from valid values, and never invent prices, fundamentals or technical indicators. This product is for research/education, not order execution or guaranteed-return claims.

## Automated tests

Run the StockMind checks from the repository root:

```bash
python -m compileall -q stockmind_telegram_mvp/app
python -m unittest discover -s stockmind_telegram_mvp/tests -v
```

The research-to-Telegram flow tests import the bot with disposable dummy configuration and mock the research provider and Telegram message methods. They cover successful text/chart delivery, progress-message failure, ambiguous final-message delivery, optional overlay-photo failure, and provider failure. Provider-client tests use mocked HTTP responses to cover Tapetide envelope validation, malformed/non-JSON success responses, provider error bodies, Binance kline/ticker shape checks and safe error normalization. AI tool-budget tests verify the six-call ceiling, per-result and per-turn byte limits, explicit time-series/search sampling metadata, and withholding oversized non-list payloads. These tests do not contact OpenAI, Tapetide, Binance or Telegram and do not validate live service credentials. Additional concurrency tests verify that same-user requests run in order, later turns receive the preceding completed context, `/reset` waits for an active turn, idle per-user locks are reclaimed after holders and waiters exit, cancelled waiters release their references, different users can proceed concurrently, resetting an unknown user does not allocate history, concurrent `/start` requests respect the beta-user cap, `/start` and research share one admission policy, same-user concurrent requests cannot exceed the daily limit, and the daily counter resets on a new IST date.

### Offline DNS benchmark and price-axis calibration

The optional offline benchmark in `app/zone_evaluation.py` scores AI overlay geometry against manually annotated rectangles. It can also report price-boundary error when a fixture supplies at least three manually verified chart-axis anchors (`y` normalized to the full screenshot plus a positive numeric `price`). The calibration supports linear and logarithmic axes, rejects inconsistent/non-monotonic anchors, and refuses to extrapolate beyond the anchor range. This is manual calibration—not automatic OCR—and it does not change the Telegram bot's chart output. See [ZONE_BENCHMARK.md](ZONE_BENCHMARK.md) for the fixture format and scoring rules.


## Offline annotation-to-benchmark assembly

## Desktop annotation editor display sizing

The offline editor scales only the on-screen preview, never the source screenshot
or its normalized annotation coordinates. The preview is capped at 1200 × 760
pixels and is further fitted to the detected display size, reserving space for
the window controls and status text. Annotation controls are arranged in
separate identity, drawing, view and action rows; narrow previews stack axis
controls on their own row.

The annotation and review windows include 100%, 150%, 200%, 300% and 400% zoom
presets plus horizontal and vertical scrollbars. Zoom is rendered from the
original in-memory screenshot, not from the already reduced preview, and is
capped at the source image's native pixel dimensions. The displayed percentage
shows the effective magnification; lower-resolution screenshots therefore stop
at their actual pixel resolution instead of being enlarged to suggest detail
that is not present. Drawing coordinates are converted from the scrolled
canvas back to normalized full-screenshot coordinates, so a box drawn after
zooming or panning is saved in the same coordinate frame as the original image
and AI prediction. Drawing must begin inside the screenshot; rectangle drags may
finish outside the image and are clipped to its edge. Price-axis anchors are
stricter: both the press and release must remain inside the image and the marked
main price plot. Changing zoom during an active drag is blocked to avoid mixing
coordinate frames.

After an annotation has been independently approved, save the actual
ResearchAI.analyze_chart_image result as JSON. The result includes
source_image_sha256, which binds it to the exact screenshot bytes used by the
vision request. The offline assembler refuses missing provider-returned model
or timestamp metadata, a different screenshot hash, unapproved annotations,
tampered geometry, and output overwrites.

~~~bash
python -m stockmind_telegram_mvp.app.annotation_fixture capture \
  benchmark/inbox/btc-1h-001.reviewed.json path/to/actual-chart-result.json

python -m stockmind_telegram_mvp.app.annotation_fixture assemble \
  benchmark/inbox/btc-1h-001.reviewed.json \
  benchmark/inbox/btc-1h-001.prediction.json

python -m stockmind_telegram_mvp.app.zone_benchmark \
  benchmark/inbox/btc-1h-001.fixture.jsonl
~~~

Annotation intake uses exclusive file creation and removes any newly created
partial image or draft when a write fails; pre-existing outputs are never
overwritten. The current Telegram bot does not automatically persist user
screenshots or model results. The capture/assembly utility is intended for an
explicitly controlled local research harness. It does not call providers or
store the original image. A structurally valid fixture is not evidence of
model accuracy; real screenshots still require independent manual review.


## Telegram response delivery

Research answers are retained in full in the conversation history. Before
delivery, long answers are split into Telegram-safe messages (4,000 UTF-16
code units per chunk, below Telegram's 4,096-character limit). The first chunk edits the
progress message when possible; remaining chunks are sent as replies. Splitting
prefers paragraph, line and word boundaries and never drops characters. If a
later chunk fails, delivery is reported as partial rather than silently
truncating the answer. Ambiguous Telegram transport failures are not retried,
because the platform may already have accepted the message.


The 4,000-character chunk budget is measured in UTF-16 code units so
supplementary Unicode characters (including emoji) cannot exceed Telegram's
entity-length accounting or be split in the middle of a code point.

