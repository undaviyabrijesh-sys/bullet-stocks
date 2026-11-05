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

## Chart screenshot analysis

Send a chart screenshot as a Telegram photo or JPEG/PNG/WEBP image document, with a caption such as "BTC 1H — mark demand and supply" or "Is my demand zone still fresh?". Keep the price scale, timeframe and enough candles visible; include the symbol, exchange and timeframe in the caption when they are not shown.

The vision model assesses visible chart structure and returns distinct supply/demand price bands, likely mitigation state, visible BOS/CHOCH or liquidity events, and conditional scenarios with invalidation. Existing user-drawn zones are assessed first and kept distinct from additional inferred zones. Numeric bounds are only reported when the chart scale is readable; otherwise the model must say that the level cannot be read reliably.

The bot now returns a second image with translucent supply (red) and demand (green) bands when the vision model can locate the zones confidently. It also returns the written zone map. Each rectangle is positioned from normalized screenshot coordinates estimated by the vision model; it is not pixel-perfect calibration to exchange OHLC data, so users must verify its position against the visible price scale. When the location is unclear, the bot intentionally withholds the overlay rather than drawing a guessed box. This is visual screenshot research, not a live chart connection. The screenshot may be delayed or cropped, freshness may be indeterminate if prior candles are not visible, and chart-vision models can misread small labels or chart graphics. Verify levels against an OHLC/live market feed before relying on any alert or trading decision. Image analysis uses more input tokens than text-only questions. GPT-6 Luna is selected for its low token cost and support for image input, Chat Completions function calling (with reasoning effort set to none), and structured outputs.

## Intraday data boundaries
- Tapetide's Indian-equity intraday bars are historical and update after the session closes; they are not a live candle feed. A separate session-live quote route is integrated for supported stocks; availability/coverage and provider timestamp must be checked on every request.
- Binance public ticker/candle endpoints provide crypto market data without API credentials. The latest candle may still be forming; the assistant must label it accordingly.
- This version does not provide a verified live Indian index/stock-options feed, broker execution, order routing, or exchange-grade streaming. Do not present delayed or end-of-session candles as real-time.
- Indian cash-market session reference: 09:15–15:30 IST. Crypto markets trade 24/7.

## Important prototype limitations
- User membership, daily usage and chat history are in memory. They reset on restart and are not suitable for a public beta until moved to persistent storage.
- Tapetide's free quota is limited and shared across REST and MCP. One AI query may trigger multiple data requests; add quota-aware budgeting before onboarding testers.
- Do not assume a free API plan permits redistribution. Obtain written provider permission for any external beta or paid service.
- Paid stock research/recommendations can raise Indian regulatory obligations. Get qualified compliance advice before monetisation.
- Keep .env out of Git. Never commit tokens or API keys.

## Data integrity
Answers must show provider timestamps when available, distinguish missing/stale data from valid values, and never invent prices, fundamentals or technical indicators. This product is for research/education, not order execution or guaranteed-return claims.
