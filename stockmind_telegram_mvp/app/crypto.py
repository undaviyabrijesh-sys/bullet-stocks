import re
import time
from decimal import Decimal, InvalidOperation

import httpx

_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")

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

    @staticmethod
    def _decimal_field(value, *, allow_zero: bool, error_message: str) -> Decimal:
        """Parse a finite, plain decimal string from a Binance market-data payload."""
        if (
            not isinstance(value, str)
            or len(value) > 64
            or _DECIMAL_TEXT.fullmatch(value) is None
        ):
            raise RuntimeError(error_message)
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise RuntimeError(error_message) from exc
        if not number.is_finite() or (number < 0 if allow_zero else number <= 0):
            raise RuntimeError(error_message)
        return number

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
        # Capture one UTC wall-clock timestamp for the entire response so candle
        # closure flags are internally consistent, even across a minute boundary.
        as_of_ms = time.time_ns() // 1_000_000
        if any(not isinstance(row, (list, tuple)) or len(row) < 9 for row in rows):
            raise RuntimeError("Binance returned a malformed kline row.")

        validated_rows = []
        previous_open_time = None
        for row in rows:
            open_time, close_time, trades = row[0], row[6], row[8]
            if (
                isinstance(open_time, bool) or not isinstance(open_time, int) or open_time < 0
                or isinstance(close_time, bool) or not isinstance(close_time, int) or close_time < open_time
                or isinstance(trades, bool) or not isinstance(trades, int) or trades < 0
            ):
                raise RuntimeError("Binance returned a malformed kline timestamp or trade count.")
            if previous_open_time is not None and open_time <= previous_open_time:
                raise RuntimeError("Binance returned non-chronological kline rows.")
            previous_open_time = open_time

            open_price = self._decimal_field(
                row[1], allow_zero=False, error_message="Binance returned a malformed kline numeric field."
            )
            high_price = self._decimal_field(
                row[2], allow_zero=False, error_message="Binance returned a malformed kline numeric field."
            )
            low_price = self._decimal_field(
                row[3], allow_zero=False, error_message="Binance returned a malformed kline numeric field."
            )
            close_price = self._decimal_field(
                row[4], allow_zero=False, error_message="Binance returned a malformed kline numeric field."
            )
            self._decimal_field(
                row[5], allow_zero=True, error_message="Binance returned a malformed kline numeric field."
            )
            self._decimal_field(
                row[7], allow_zero=True, error_message="Binance returned a malformed kline numeric field."
            )
            if (
                high_price < max(open_price, low_price, close_price)
                or low_price > min(open_price, high_price, close_price)
            ):
                raise RuntimeError("Binance returned inconsistent OHLC values.")
            validated_rows.append(row)

        return {
            "source": "Binance",
            "market": market,
            "symbol": symbol,
            "interval": interval,
            "timezone": "UTC timestamps (display in IST when useful)",
            "data": [
                {"open_time": row[0], "open": row[1], "high": row[2], "low": row[3],
                 "close": row[4], "volume": row[5], "close_time": row[6],
                 "quote_volume": row[7], "trades": row[8],
                 "is_closed": row[6] < as_of_ms,
                 "candle_status": "closed" if row[6] < as_of_ms else "forming"}
                for row in validated_rows
            ],
            "as_of_ms": as_of_ms,
            "note": "Each candle's is_closed flag is evaluated against this response's as_of_ms timestamp."
        }

    async def ticker(self, symbol: str, market: str = "spot"):
        symbol = self._symbol(symbol)
        base = self._base(market)
        path = "/api/v3/ticker/24hr" if market == "spot" else "/fapi/v1/ticker/24hr"
        payload = await self._get_json(base + path, {"symbol": symbol})
        if not isinstance(payload, dict) or payload.get("symbol") != symbol:
            raise RuntimeError("Binance returned an unexpected ticker response.")
        try:
            self._decimal_field(
                payload.get("lastPrice"),
                allow_zero=False,
                error_message="Binance returned an unexpected ticker response.",
            )
        except RuntimeError:
            raise RuntimeError("Binance returned an unexpected ticker response.") from None
        return {"source": "Binance", "market": market, "symbol": symbol, "data": payload}

    async def close(self):
        await self.client.aclose()
