import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from stockmind_telegram_mvp.app.ai import _tool_error_payload
from stockmind_telegram_mvp.app.crypto import BinanceMarketData
from stockmind_telegram_mvp.app.tapetide import TapetideAPIError, TapetideClient


class TapetideResponseValidationTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, response):
        client = object.__new__(TapetideClient)
        client.client = SimpleNamespace(get=AsyncMock(return_value=response))
        return client

    async def test_preserves_valid_data_and_metadata_envelope(self):
        response = httpx.Response(
            200,
            json={"data": [{"symbol": "RELIANCE"}], "meta": {"as_of": "2026-11-07T00:00:00Z"}},
        )
        client = self.make_client(response)

        result = await client._get("/stocks")

        self.assertEqual(result["data"], [{"symbol": "RELIANCE"}])
        self.assertEqual(result["meta"]["as_of"], "2026-11-07T00:00:00Z")

    async def test_rejects_non_json_success_instead_of_returning_empty_data(self):
        client = self.make_client(httpx.Response(200, text="<html>gateway page</html>"))

        with self.assertRaises(TapetideAPIError) as caught:
            await client._get("/stocks")

        self.assertEqual(caught.exception.code, "invalid_response")
        self.assertEqual(caught.exception.status_code, 200)

    async def test_rejects_json_array_success_payload(self):
        client = self.make_client(httpx.Response(200, json=[{"symbol": "RELIANCE"}]))

        with self.assertRaisesRegex(TapetideAPIError, "unexpected response format"):
            await client._get("/stocks")

    async def test_rejects_success_payload_without_data_field(self):
        client = self.make_client(httpx.Response(200, json={"meta": {}}))

        with self.assertRaisesRegex(TapetideAPIError, "missing the data field"):
            await client._get("/stocks")

    async def test_rejects_non_object_metadata(self):
        client = self.make_client(httpx.Response(200, json={"data": [], "meta": []}))

        with self.assertRaisesRegex(TapetideAPIError, "metadata has an unexpected format"):
            await client._get("/stocks")

    async def test_normalizes_malformed_http_error_body(self):
        response = httpx.Response(
            503,
            json={"error": "temporary upstream failure"},
            request=httpx.Request("GET", "https://api.tapetide.com/v1/stocks"),
        )
        client = self.make_client(response)

        with self.assertRaises(TapetideAPIError) as caught:
            await client._get("/stocks")

        self.assertEqual(caught.exception.code, "http_503")
        self.assertEqual(caught.exception.status_code, 503)

    async def test_preserves_stable_provider_error_code(self):
        response = httpx.Response(
            429,
            json={"error": {"code": "quota_exceeded", "message": "Daily quota reached."}},
            request=httpx.Request("GET", "https://api.tapetide.com/v1/stocks"),
        )
        client = self.make_client(response)

        with self.assertRaises(TapetideAPIError) as caught:
            await client._get("/stocks")

        self.assertEqual(caught.exception.code, "quota_exceeded")
        self.assertEqual(str(caught.exception), "Daily quota reached.")
        self.assertEqual(caught.exception.status_code, 429)

    async def test_rejects_invalid_financial_section_before_request(self):
        client = self.make_client(httpx.Response(200, json={"data": {}, "meta": {}}))

        with self.assertRaisesRegex(ValueError, "Financial section"):
            await client.financials("RELIANCE", "monthly")

        client.client.get.assert_not_awaited()


class BinanceResponseValidationTests(unittest.IsolatedAsyncioTestCase):
    def prepare_response(self, response):
        # AsyncClient attaches the originating request before raise_for_status().
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=httpx.Request("GET", "https://api.binance.com/test"),
        )

    def make_client(self, response, time_response=None):
        response = self.prepare_response(response)
        if time_response is None:
            time_response = httpx.Response(200, json={"serverTime": 1700001000000})
        time_response = self.prepare_response(time_response)
        client = object.__new__(BinanceMarketData)
        client.client = SimpleNamespace(get=AsyncMock(side_effect=[response, time_response]))
        return client

    async def test_maps_valid_kline_rows_without_changing_provider_values(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        client = self.make_client(httpx.Response(200, json=[row]))

        result = await client.klines("BTCUSDT", interval="1h", market="usdm")

        self.assertEqual(result["data"][0]["open"], "1")
        self.assertEqual(result["data"][0]["close"], "1.5")
        self.assertEqual(result["data"][0]["trades"], 100)
        self.assertEqual(result["market"], "usdm")

    async def test_marks_closed_and_forming_klines_against_conservative_binance_time(self):
        first = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        boundary = [1700000900000, "1", "2", "0.5", "1.5", "10", 1700000900000, "15", 100]
        as_of_ms = 1700000900000
        time_response = httpx.Response(200, json={"serverTime": as_of_ms + 2})
        client = self.make_client(
            httpx.Response(200, json=[first, boundary]),
            time_response=time_response,
        )

        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=[0, 1_000_000],
        ) as clock:
            result = await client.klines("BTCUSDT", interval="1m")

        self.assertEqual(result["as_of_ms"], as_of_ms)
        self.assertEqual(result["clock_source"], "binance_server_time")
        self.assertEqual(result["clock_uncertainty_ms"], 2)
        self.assertEqual([row["is_closed"] for row in result["data"]], [True, False])
        self.assertEqual(
            [row["candle_status"] for row in result["data"]],
            ["closed", "forming"],
        )
        self.assertEqual(clock.call_count, 2)
        self.assertIn("/api/v3/time", client.client.get.await_args_list[1].args[0])

    async def test_unavailable_or_malformed_server_time_never_confirms_candles(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        invalid_time_responses = (
            httpx.Response(
                503,
                json={"message": "temporarily unavailable"},
                request=httpx.Request("GET", "https://api.binance.com/api/v3/time"),
            ),
            httpx.Response(200, json={"serverTime": "1700001000000"}),
            httpx.Response(200, json={"serverTime": True}),
            httpx.Response(200, json={"serverTime": -1}),
        )
        for time_response in invalid_time_responses:
            with self.subTest(time_response=time_response.status_code):
                client = self.make_client(
                    httpx.Response(200, json=[row]),
                    time_response=time_response,
                )
                result = await client.klines("BTCUSDT")

                self.assertEqual(result["clock_source"], "unverified")
                self.assertIsNone(result["as_of_ms"])
                self.assertIsNone(result["clock_uncertainty_ms"])
                self.assertFalse(result["data"][0]["is_closed"])
                self.assertEqual(result["data"][0]["candle_status"], "unverified")

    async def test_reuses_server_time_sync_within_market_cache_ttl(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        client = self.make_client(
            httpx.Response(200, json=[row]),
            time_response=httpx.Response(200, json={"serverTime": 1700001000000}),
        )
        queued_responses = list(client.client.get.side_effect)
        second_kline_response = self.prepare_response(httpx.Response(200, json=[row]))
        client.client.get.side_effect = queued_responses + [second_kline_response]

        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=[0, 1_000_000, 2_000_000],
        ) as clock:
            first = await client.klines("BTCUSDT", interval="1h")
            second = await client.klines("BTCUSDT", interval="1h")

        self.assertEqual(first["clock_source"], "binance_server_time")
        self.assertEqual(second["clock_source"], "binance_server_time")
        self.assertEqual(client.client.get.await_count, 3)
        self.assertEqual(clock.call_count, 3)

    async def test_rejects_non_finite_non_positive_or_malformed_kline_values(self):
        base_row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        invalid_fields = (
            (1, "NaN"),
            (2, "Infinity"),
            (3, "-0.1"),
            (4, "0"),
            (5, "NaN"),
            (7, "-1"),
            (1, "1e3"),
            (2, "9" * 65),
        )
        for index, value in invalid_fields:
            with self.subTest(index=index, value=value):
                row = list(base_row)
                row[index] = value
                client = self.make_client(httpx.Response(200, json=[row]))
                with self.assertRaisesRegex(RuntimeError, "malformed kline numeric field"):
                    await client.klines("BTCUSDT")

    async def test_rejects_inconsistent_ohlc_relationships(self):
        base_row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        invalid_rows = []
        high_below_open = list(base_row)
        high_below_open[2] = "0.9"
        invalid_rows.append(high_below_open)
        low_above_close = list(base_row)
        low_above_close[3] = "1.6"
        invalid_rows.append(low_above_close)
        low_above_high = list(base_row)
        low_above_high[3] = "2.1"
        invalid_rows.append(low_above_high)

        for row in invalid_rows:
            with self.subTest(row=row):
                client = self.make_client(httpx.Response(200, json=[row]))
                with self.assertRaisesRegex(RuntimeError, "inconsistent OHLC"):
                    await client.klines("BTCUSDT")

    async def test_rejects_invalid_timestamps_and_trade_counts(self):
        base_row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        invalid_fields = ((0, True), (0, -1), (6, 1), (8, True), (8, 1.5), (8, -1))
        for index, value in invalid_fields:
            with self.subTest(index=index, value=value):
                row = list(base_row)
                row[index] = value
                client = self.make_client(httpx.Response(200, json=[row]))
                with self.assertRaisesRegex(RuntimeError, "malformed kline timestamp or trade count"):
                    await client.klines("BTCUSDT")

    async def test_rejects_duplicate_or_out_of_order_kline_timestamps(self):
        first = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        duplicate = list(first)
        later = list(first)
        later[0] += 60000
        later[6] += 60000
        for rows in ([first, duplicate], [later, first]):
            client = self.make_client(httpx.Response(200, json=rows))
            with self.assertRaisesRegex(RuntimeError, "non-chronological"):
                await client.klines("BTCUSDT")

    async def test_rejects_object_error_payload_from_kline_endpoint(self):
        client = self.make_client(httpx.Response(200, json={"code": -1121, "msg": "Invalid symbol."}))

        with self.assertRaisesRegex(RuntimeError, "unexpected kline response"):
            await client.klines("BTCUSDT")

    async def test_rejects_malformed_kline_row(self):
        client = self.make_client(httpx.Response(200, json=[[1700000000000, "1", "2"]]))

        with self.assertRaisesRegex(RuntimeError, "malformed kline row"):
            await client.klines("BTCUSDT")

    async def test_rejects_non_json_kline_response(self):
        client = self.make_client(httpx.Response(200, text="upstream proxy error"))

        with self.assertRaisesRegex(RuntimeError, "non-JSON response"):
            await client.klines("BTCUSDT")

    async def test_accepts_ticker_only_when_symbol_and_last_price_are_present(self):
        payload = {"symbol": "BTCUSDT", "lastPrice": "68000.10"}
        client = self.make_client(httpx.Response(200, json=payload))

        result = await client.ticker("BTCUSDT", market="spot")

        self.assertEqual(result["data"], payload)
        self.assertEqual(result["symbol"], "BTCUSDT")

    async def test_rejects_non_positive_or_malformed_ticker_last_price(self):
        invalid_prices = ("NaN", "Infinity", "0", "-1", "not-a-price", "", True, 0)
        for last_price in invalid_prices:
            with self.subTest(last_price=repr(last_price)):
                client = self.make_client(
                    httpx.Response(200, json={"symbol": "BTCUSDT", "lastPrice": last_price})
                )
                with self.assertRaisesRegex(RuntimeError, "unexpected ticker response"):
                    await client.ticker("BTCUSDT")

    async def test_rejects_ticker_payload_for_a_different_symbol(self):
        client = self.make_client(
            httpx.Response(200, json={"symbol": "ETHUSDT", "lastPrice": "2000"})
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected ticker response"):
            await client.ticker("BTCUSDT")

    async def test_rejects_ticker_error_object(self):
        client = self.make_client(
            httpx.Response(200, json={"code": -1121, "msg": "Invalid symbol."})
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected ticker response"):
            await client.ticker("BTCUSDT")


    async def test_caches_unverified_clock_failure_briefly_to_avoid_repeated_time_requests(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        failure = httpx.Response(
            503,
            json={"message": "temporarily unavailable"},
            request=httpx.Request("GET", "https://api.binance.com/api/v3/time"),
        )
        client = self.make_client(
            httpx.Response(200, json=[row]),
            time_response=failure,
        )
        queued_responses = list(client.client.get.side_effect)
        second_kline = self.prepare_response(httpx.Response(200, json=[row]))
        client.client.get.side_effect = queued_responses + [second_kline]

        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=[0, 1_000_000, 2_000_000],
        ) as clock:
            first = await client.klines("BTCUSDT", interval="1m")
            second = await client.klines("BTCUSDT", interval="1m")

        self.assertEqual(first["clock_source"], "unverified")
        self.assertEqual(second["clock_source"], "unverified")
        self.assertEqual(client.client.get.await_count, 3)
        self.assertEqual(clock.call_count, 3)

    async def test_retries_server_time_after_failure_cooldown_and_recovers(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        failure = httpx.Response(
            503,
            json={"message": "temporarily unavailable"},
            request=httpx.Request("GET", "https://api.binance.com/api/v3/time"),
        )
        client = self.make_client(
            httpx.Response(200, json=[row]),
            time_response=failure,
        )
        queued_responses = list(client.client.get.side_effect)
        second_kline = self.prepare_response(httpx.Response(200, json=[row]))
        recovered_time = self.prepare_response(
            httpx.Response(200, json={"serverTime": 1700001000000})
        )
        client.client.get.side_effect = queued_responses + [second_kline, recovered_time]

        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=[0, 1_000_000, 5_001_000_001, 5_002_000_001],
        ) as clock:
            first = await client.klines("BTCUSDT", interval="1m")
            recovered = await client.klines("BTCUSDT", interval="1m")

        self.assertEqual(first["clock_source"], "unverified")
        self.assertEqual(recovered["clock_source"], "binance_server_time")
        self.assertEqual(client.client.get.await_count, 4)
        self.assertEqual(clock.call_count, 4)

    async def test_spot_and_usdm_server_time_caches_are_independent(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        responses = [
            self.prepare_response(httpx.Response(200, json=[row])),
            self.prepare_response(httpx.Response(200, json={"serverTime": 1700001000000})),
            self.prepare_response(httpx.Response(200, json=[row])),
            self.prepare_response(httpx.Response(200, json={"serverTime": 1700001000000})),
        ]
        client = object.__new__(BinanceMarketData)
        client.client = SimpleNamespace(get=AsyncMock(side_effect=responses))

        with patch(
            "stockmind_telegram_mvp.app.crypto.time.monotonic_ns",
            side_effect=[0, 1_000_000, 2_000_000, 3_000_000],
        ):
            spot = await client.klines("BTCUSDT", interval="1h", market="spot")
            usdm = await client.klines("BTCUSDT", interval="1h", market="usdm")

        self.assertEqual(spot["clock_source"], "binance_server_time")
        self.assertEqual(usdm["clock_source"], "binance_server_time")
        self.assertEqual(client.client.get.await_count, 4)
        requested_urls = [call.args[0] for call in client.client.get.await_args_list]
        self.assertIn("/api/v3/time", requested_urls[1])
        self.assertIn("/fapi/v1/time", requested_urls[3])



class SafeToolErrorTests(unittest.TestCase):
    def test_timeout_error_does_not_expose_request_url_or_query_parameters(self):
        request = httpx.Request(
            "GET",
            "https://api.example.test/quote?token=do-not-forward",
        )
        error = httpx.ReadTimeout("timed out", request=request)

        result = _tool_error_payload(error)

        self.assertEqual(result["type"], "ProviderTimeout")
        self.assertNotIn("do-not-forward", str(result))
        self.assertNotIn("api.example.test", str(result))

    def test_tapetide_error_exposes_code_but_not_provider_message(self):
        error = TapetideAPIError("quota_exceeded", "sensitive provider detail", 429)

        result = _tool_error_payload(error)

        self.assertEqual(result["type"], "TapetideAPIError")
        self.assertEqual(result["code"], "quota_exceeded")
        self.assertEqual(result["status_code"], 429)
        self.assertNotIn("sensitive provider detail", str(result))

    def test_http_status_error_does_not_expose_request_url(self):
        request = httpx.Request("GET", "https://api.example.test/private?key=hidden")
        response = httpx.Response(502, request=request)
        error = httpx.HTTPStatusError("upstream error", request=request, response=response)

        result = _tool_error_payload(error)

        self.assertEqual(result["type"], "ProviderHTTPError")
        self.assertEqual(result["status_code"], 502)
        self.assertNotIn("private", str(result))
        self.assertNotIn("hidden", str(result))

    def test_unexpected_exception_is_not_forwarded_verbatim(self):
        result = _tool_error_payload(RuntimeError("internal secret detail"))

        self.assertEqual(result["type"], "ToolExecutionError")
        self.assertNotIn("internal secret detail", str(result))

    def test_local_argument_errors_remain_actionable(self):
        result = _tool_error_payload(ValueError("Unsupported interval."))

        self.assertEqual(result, {"error": "Unsupported interval.", "type": "ValueError"})


if __name__ == "__main__":
    unittest.main()
