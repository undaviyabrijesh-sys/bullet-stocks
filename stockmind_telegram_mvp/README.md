# StockMind AI — Telegram MVP

A multilingual Telegram research assistant for Indian equities. Users can ask in English, Hinglish, Hindi, Gujarati or mixed/broken language. The AI decides which Tapetide read-only endpoints to call and responds in the user's language.

## Included
- Telegram bot (aiogram 3)
- OpenAI tool-calling for intent understanding and research planning
- Tapetide REST adapter: stock search, latest quote, financial statements and price history
- Per-user conversational context
- Prototype cap of 10 beta users and 10 questions per user per day
- No order placement or trade execution

## Setup
1. Create a bot with Telegram's @BotFather.
2. Create an OpenAI API key.
3. Create a Tapetide token and confirm that your plan permits the intended beta data display and caching.
4. Copy .env.example to .env and set the credentials.
5. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.bot
```

Windows activation: `.venv\\Scripts\\activate`.

## Configuration
- TELEGRAM_BOT_TOKEN
- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-4.1-mini)
- TAPETIDE_TOKEN
- DAILY_QUERY_LIMIT (default: 10)
- BETA_USER_LIMIT (default: 10)
- MAX_HISTORY_MESSAGES (default: 12)

## Important prototype limitations
- User membership, daily usage and chat history are in memory. They reset on restart and are not suitable for a public beta until moved to persistent storage.
- Tapetide's free quota is limited and shared across REST and MCP. One AI query may trigger multiple data requests; add quota-aware budgeting before onboarding testers.
- Do not assume a free API plan permits redistribution. Obtain written provider permission for any external beta or paid service.
- Paid stock research/recommendations can raise Indian regulatory obligations. Get qualified compliance advice before monetisation.
- Keep .env out of Git. Never commit tokens or API keys.

## Data integrity
Answers must show provider timestamps when available, distinguish missing/stale data from valid values, and never invent prices, fundamentals or technical indicators. This product is for research/education, not order execution or guaranteed-return claims.
