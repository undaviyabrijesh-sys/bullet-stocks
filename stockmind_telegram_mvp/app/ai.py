import json
from openai import AsyncOpenAI

SYSTEM_PROMPT = """
You are StockMind AI, a multilingual Indian stock-market research assistant.
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
Identify Tapetide as the data source.
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
]


class ResearchAI:
    def __init__(self, api_key: str, model: str, market_data):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.market_data = market_data

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

            messages.append(message)
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
                    elif call.function.name == "get_financials":
                        result = await self.market_data.financials(
                            args["symbol"], args.get("section", "annual")
                        )
                    elif call.function.name == "get_price_history":
                        result = await self.market_data.price_history(
                            args["symbol"], args.get("days", 90)
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
