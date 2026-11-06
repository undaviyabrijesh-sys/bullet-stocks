import re
import httpx

SPOT = "https://api.binance.com"
FUTURES = "https://fapi.binance.com"
INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}


class BinanceMarketData:
    """Public, read-only Binance spot and USD-M futures market data."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)

    @staticmethod
    def _symbol(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Use a valid Binance symbol such as BTCUSDT or ETHUSDT")
        symbol = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{5,24}", symbol):
            raise ValueError("Use a valid Binance symbol such as BTCUSDT or ETHUSDT")
        return symbol

    def _base(self, market: str) -> str:
        if market == "spot":
            return SPOT
        if market == "usdm":
            return FUTURES
        raise ValueError("market must be 'spot' or 'usdm'")

    async def _get_json(self, url: str, params: dict):
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Binance returned a non-JSON response.") from exc

    async def klines(self, symbol: str, interval: str = "15m", market: str = "spot", limit: int = 100):
        symbol = self._symbol(symbol)
        if interval not in INTERVALS:
            raise ValueError(f"Unsupported interval. Choose from: {', '.join(sorted(INTERVALS))}")
        limit = max(1, min(int(limit), 500))
        base = self._base(market)
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
        rows = await self._get_json(
            base + path,
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not isinstance(rows, list):
            raise RuntimeError("Binance returned an unexpected kline response.")
        if any(not isinstance(row, (list, tuple)) or len(row) < 9 for row in rows):
            raise RuntimeError("Binance returned a malformed kline row.")

        return {
            "source": "Binance",
            "market": market,
            "symbol": symbol,
            "interval": interval,
            "timezone": "UTC timestamps (display in IST when useful)",
            "data": [
                {"open_time": row[0], "open": row[1], "high": row[2], "low": row[3],
                 "close": row[4], "volume": row[5], "close_time": row[6],
                 "quote_volume": row[7], "trades": row[8]}
                for row in rows
            ],
            "note": "The final candle may still be forming; do not treat it as closed until close_time has passed."
        }

    async def ticker(self, symbol: str, market: str = "spot"):
        symbol = self._symbol(symbol)
        base = self._base(market)
        path = "/api/v3/ticker/24hr" if market == "spot" else "/fapi/v1/ticker/24hr"
        payload = await self._get_json(base + path, {"symbol": symbol})
        if (
            not isinstance(payload, dict)
            or payload.get("symbol") != symbol
            or not isinstance(payload.get("lastPrice"), (str, int, float))
            or isinstance(payload.get("lastPrice"), bool)
        ):
            raise RuntimeError("Binance returned an unexpected ticker response.")
        return {"source": "Binance", "market": market, "symbol": symbol, "data": payload}

    async def close(self):
        await self.client.aclose()
