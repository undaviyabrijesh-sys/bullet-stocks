import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from stockmind_telegram_mvp.app.crypto import BinanceMarketData


class BinanceClockCacheBoundaryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _response(payload, *, status=200, path="/test"):
        return httpx.Response(
            status,
            json=payload,
            request=httpx.Request("GET", f"https://api.binance.com{path}"),
        )

    @staticmethod
    def _client(responses):
        client = object.__new__(BinanceMarketData)
        client.client = SimpleNamespace(get=AsyncMock(side_effect=responses))
        return client

    async def test_cache_ttl_boundaries_never_reuse_an_expired_successful_clock(self):
        row = [
            1700000000000,
            "1",
            "2",
            "0.5",
            "1.5",
            "10",
            1700000899999,
            "15",
            100,
        ]
        client = self._client(
            [
                self._response([row], path="/api/v3/klines"),
                self._response(
                    {"serverTime": 1700001000000},
                    path="/api/v3/time",
                ),
                self._response([row], path="/api/v3/klines"),
                self._response([row], path="/api/v3/klines"),
                self._response(
                    {"message": "temporarily unavailable"},
                    status=503,
                    path="/api/v3/time",
                ),
                self._response([row], path="/api/v3/klines"),
            ]
        )

        # The first successful sample is received at 1 ms. The second request
        # lands exactly 30 seconds later and may reuse it. One nanosecond later,
        # the sample is expired and a failed refresh must make the clock
        # unverified rather than falling back to the stale successful sample.
        # The final request is exactly 5 seconds after that failure and must
        # remain unverified without retrying the time endpoint yet.
        monotonic_values = [
            0,
            1_000_000,
            30_001_000_000,
            30_001_000_001,
            30_001_000_002,
            35_001_000_002,
        ]
        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=monotonic_values,
        ) as monotonic:
            first = await client.klines("BTCUSDT", interval="1m")
            at_success_ttl = await client.klines("BTCUSDT", interval="1m")
            after_success_ttl = await client.klines("BTCUSDT", interval="1m")
            at_failure_ttl = await client.klines("BTCUSDT", interval="1m")

        self.assertEqual(first["clock_source"], "binance_server_time")
        self.assertEqual(at_success_ttl["clock_source"], "binance_server_time")
        self.assertEqual(after_success_ttl["clock_source"], "unverified")
        self.assertEqual(at_failure_ttl["clock_source"], "unverified")
        self.assertFalse(after_success_ttl["data"][0]["is_closed"])
        self.assertEqual(after_success_ttl["data"][0]["candle_status"], "unverified")
        self.assertFalse(at_failure_ttl["data"][0]["is_closed"])
        self.assertEqual(at_failure_ttl["data"][0]["candle_status"], "unverified")

        requested_urls = [call.args[0] for call in client.client.get.await_args_list]
        self.assertEqual(len(requested_urls), 6)
        self.assertEqual(
            [url for url in requested_urls if url.endswith("/api/v3/time")],
            [
                "https://api.binance.com/api/v3/time",
                "https://api.binance.com/api/v3/time",
            ],
        )
        self.assertEqual(monotonic.call_count, len(monotonic_values))


if __name__ == "__main__":
    unittest.main()
