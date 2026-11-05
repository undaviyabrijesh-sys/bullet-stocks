import json
from openai import AsyncOpenAI

SYSTEM_PROMPT = """
You are StockMind AI, a multilingual market research assistant for Indian equities and crypto.
Understand natural language including broken English, Hinglish, Hindi, Gujarati and code-switching.
Reply in the language/style the user used unless they request another language.
Do not correct grammar. Ask one concise clarification if the stock/company is ambiguous.

Use Tapetide tools when current or company-specific market data is needed. Never invent prices,
financial figures, dates, technical indicators or news. Clearly show provider timestamps/as_of
when supplied. If data is missing or stale, say so. Explain calculations and separate facts
from interpretation. Do not promise returns or present uncertain forecasts as facts.
Do not place trades or claim to provide personalised investment advice. Provide research and
educational analysis only.

STOCK SEARCH RULES:
- Use search_stocks for company names, brand names, ticker symbols, BSE codes and ISINs.
- Search the user's original wording first; do not force a guessed ticker.
- Tapetide supports common brand/renamed aliases (for example, Zomato can resolve to ETERNAL)
  and typo tolerance for queries of four or more characters.
- If multiple plausible matches are returned, show the best matches with symbol, company name,
  exchange identifier, sector and industry, then ask which one the user means.
- Use the exact returned symbol for subsequent quote, financial and price-history calls.
- Do not assume every match is currently listed; check is_active and explain inactive/delisted status.
- Do not silently choose between similarly named companies.

For quote questions use get_quote. For financial health use get_financials.
For historical price context use get_price_history. Do not make more data calls than necessary.
INTRADAY AND CRYPTO:
- Support Indian cash equities and crypto spot/USD-M perpetual markets through Binance public read-only data.
- Use get_intraday_stock_bars for historical 1/5/15-minute Indian equity candles. These bars update after the session closes; they are NOT a live candle feed. Use get_live_quote for session-live Indian stock quotes where available; otherwise use get_quote for the latest available quote and state its timestamp.
- Use get_crypto_klines and get_crypto_ticker for crypto. The latest Binance candle can still be forming; clearly identify it and avoid relying on it as a confirmed candle.
- For intraday analysis, establish instrument, market and timeframe. Indian regular cash session is 09:15–15:30 IST; crypto trades 24/7. Ask one short clarification if a crucial detail is missing.
- Structure a setup as market bias, key levels/zones, trigger/confirmation, conditional entry area, invalidation/stop, targets, reward:risk and cancellation condition. Mark data timestamp and candle status.
- Calculate position size only when the user supplies account risk and stop distance; show the formula. Never imply a setup is guaranteed or place/route orders.
- For Indian index/options intraday, do not fabricate option-chain, live premium, OI or Greeks data. Explain when the connected source does not provide a live feed.
Identify Tapetide or Binance as the source for each dataset.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_stocks",
            "description": (
                "Search Tapetide's Indian stock directory (~8,200 stocks) by company name, "
                "brand/old name, NSE symbol, BSE code or ISIN. Supports sector/industry filters. "
                "Use this before other stock endpoints whenever the symbol is uncertain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company, brand, symbol, BSE code or ISIN."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "sector": {"type": "string"},
                    "industry": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the latest available quote and quote metadata for an NSE/BSE stock.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Get company financial statements for annual or quarterly analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "section": {"type": "string", "enum": ["annual", "quarterly"]},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get historical OHLCV price data for an Indian listed stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 365},
                },
                "required": ["symbol"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_intraday_stock_bars",
            "description": "Get historical 1, 5 or 15-minute OHLCV bars for an NSE/BSE stock. Tapetide data updates after the session closes, not live.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "integer", "enum": [1, 5, 15]},
                    "date": {"type": "string", "description": "One IST trading date YYYY-MM-DD."},
                    "from_date": {"type": "string", "description": "Start IST date YYYY-MM-DD; requires to_date."},
                    "to_date": {"type": "string", "description": "End IST date YYYY-MM-DD; max 7 calendar days after from_date."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000}
                },
                "required": ["symbol"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_klines",
            "description": "Get Binance spot or USD-M futures OHLCV candles for a crypto symbol such as BTCUSDT. Latest candle may still be open.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "enum": ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]},
                    "market": {"type": "string", "enum": ["spot","usdm"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500}
                },
                "required": ["symbol"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_ticker",
            "description": "Get Binance 24-hour ticker statistics for a crypto symbol such as BTCUSDT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "market": {"type": "string", "enum": ["spot","usdm"]}
                },
                "required": ["symbol"],
                "additionalProperties": False
            }
        }
    },
]


class ResearchAI:
    def __init__(self, api_key: str, model: str, market_data, crypto_data):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.market_data = market_data
        self.crypto_data = crypto_data

    async def answer(self, history: list[dict], user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]

        for _ in range(4):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or "I couldn't prepare a reliable answer. Please try again."

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                try:
                    if call.function.name == "search_stocks":
                        result = await self.market_data.search_stocks(
                            args["query"],
                            limit=args.get("limit", 10),
                            sector=args.get("sector"),
                            industry=args.get("industry"),
                        )
                    elif call.function.name == "get_quote":
                        result = await self.market_data.quote(args["symbol"])
                    elif call.function.name == "get_live_quote":\n                        result = await self.market_data.live_quote(args["symbol"])\n                    elif call.function.name == "get_financials":
                        result = await self.market_data.financials(
                            args["symbol"], args.get("section", "annual")
                        )
                    elif call.function.name == "get_price_history":
                        result = await self.market_data.price_history(
                            args["symbol"], args.get("days", 90)
                        )
                    elif call.function.name == "get_intraday_stock_bars":
                        result = await self.market_data.intraday_bars(
                            args["symbol"], interval=args.get("interval", 5),
                            date=args.get("date"), from_date=args.get("from_date"),
                            to_date=args.get("to_date"), limit=args.get("limit", 375)
                        )
                    elif call.function.name == "get_crypto_klines":
                        result = await self.crypto_data.klines(
                            args["symbol"], args.get("interval", "15m"),
                            args.get("market", "spot"), args.get("limit", 100)
                        )
                    elif call.function.name == "get_crypto_ticker":
                        result = await self.crypto_data.ticker(
                            args["symbol"], args.get("market", "spot")
                        )
                    else:
                        result = {"error": "Unsupported tool"}
                except Exception as exc:
                    result = {"error": str(exc), "type": type(exc).__name__}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return "I couldn't complete the research within the allowed steps. Please narrow the question."
