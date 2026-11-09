import asyncio
import math
import re
import time
from decimal import Decimal, InvalidOperation

import httpx

_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")

SPOT = "https://api.binance.com"
FUTURES = "https://fapi.binance.com"
INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
SERVER_TIME_CACHE_TTL_NS = 30_000_000_000
SERVER_CLOCK_DRIFT_PPM = 100


class BinanceMarketData:
    """Public, read-only Binance spot and USD-M futures market data."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._server_clock_cache = {}
        self._server_clock_lock = asyncio.Lock()

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

    async def _server_clock(self, market: str) -> dict:
        """Return a conservative Binance-time estimate, cached per market.

        The uncertainty margin includes the full observed round-trip time,
        one millisecond for server timestamp resolution, and a 100 ppm drift
        allowance while the monotonic-clock sample is cached.
        """
        base = self._base(market)
        path = "/api/v3/time" if market == "spot" else "/fapi/v1/time"
        cache = getattr(self, "_server_clock_cache", None)
        if cache is None:
            cache = {}
            self._server_clock_cache = cache
        lock = getattr(self, "_server_clock_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._server_clock_lock = lock

        async with lock:
            now_ns = time.monotonic_ns()
            cached = cache.get(market)
            if cached is not None:
                elapsed_ns = now_ns - cached["received_monotonic_ns"]
                if 0 <= elapsed_ns <= SERVER_TIME_CACHE_TTL_NS:
                    elapsed_ms = elapsed_ns / 1_000_000
                    drift_allowance_ms = math.ceil(
                        elapsed_ms * SERVER_CLOCK_DRIFT_PPM / 1_000_000
                    )
                    uncertainty_ms = cached["uncertainty_ms"] + drift_allowance_ms
                    as_of_ms = math.floor(
                        cached["server_time_ms"] + elapsed_ms - uncertainty_ms
                    )
                    if as_of_ms > 0:
                        return {
                            "as_of_ms": as_of_ms,
                            "clock_source": "binance_server_time",
                            "clock_uncertainty_ms": uncertainty_ms,
                        }

            started_ns = now_ns
            try:
                payload = await self._get_json(base + path, {})
            except (httpx.HTTPError, RuntimeError, ValueError):
                return {
                    "as_of_ms": None,
                    "clock_source": "unverified",
                    "clock_uncertainty_ms": None,
                }

            received_ns = time.monotonic_ns()
            server_time_ms = payload.get("serverTime") if isinstance(payload, dict) else None
            if (
                isinstance(server_time_ms, bool)
                or not isinstance(server_time_ms, int)
                or server_time_ms <= 0
            ):
                return {
                    "as_of_ms": None,
                    "clock_source": "unverified",
                    "clock_uncertainty_ms": None,
                }

            round_trip_ms = math.ceil(max(0, received_ns - started_ns) / 1_000_000)
            uncertainty_ms = round_trip_ms + 1
            as_of_ms = server_time_ms - uncertainty_ms
            if as_of_ms <= 0:
                return {
                    "as_of_ms": None,
                    "clock_source": "unverified",
                    "clock_uncertainty_ms": None,
                }

            cache[market] = {
                "server_time_ms": server_time_ms,
                "received_monotonic_ns": received_ns,
                "uncertainty_ms": uncertainty_ms,
            }
            return {
                "as_of_ms": as_of_ms,
                "clock_source": "binance_server_time",
                "clock_uncertainty_ms": uncertainty_ms,
            }

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

        clock = await self._server_clock(market)
        clock_verified = clock["clock_source"] == "binance_server_time"
        as_of_ms = clock["as_of_ms"]
        data = []
        for row in validated_rows:
            is_closed = bool(clock_verified and row[6] < as_of_ms)
            candle_status = (
                "closed" if is_closed else "forming" if clock_verified else "unverified"
            )
            data.append({
                "open_time": row[0], "open": row[1], "high": row[2], "low": row[3],
                "close": row[4], "volume": row[5], "close_time": row[6],
                "quote_volume": row[7], "trades": row[8],
                "is_closed": is_closed,
                "candle_status": candle_status,
            })

        return {
            "source": "Binance",
            "market": market,
            "symbol": symbol,
            "interval": interval,
            "timezone": "UTC timestamps (display in IST when useful)",
            "data": data,
            **clock,
            "note": (
                "as_of_ms is a conservative lower-bound estimate of Binance server time; "
                "candle closure is unverified when clock_source is unverified."
            ),
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
