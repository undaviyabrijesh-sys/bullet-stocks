import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    def make_client(self, response):
        # AsyncClient attaches the originating request before raise_for_status().
        # Rebuild mocked responses with that same required HTTPX invariant.
        response = httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=httpx.Request("GET", "https://api.binance.com/test"),
        )
        client = object.__new__(BinanceMarketData)
        client.client = SimpleNamespace(get=AsyncMock(return_value=response))
        return client

    async def test_maps_valid_kline_rows_without_changing_provider_values(self):
        row = [1700000000000, "1", "2", "0.5", "1.5", "10", 1700000899999, "15", 100]
        client = self.make_client(httpx.Response(200, json=[row]))

        result = await client.klines("BTCUSDT", interval="1h", market="usdm")

        self.assertEqual(result["data"][0]["open"], "1")
        self.assertEqual(result["data"][0]["close"], "1.5")
        self.assertEqual(result["data"][0]["trades"], 100)
        self.assertEqual(result["market"], "usdm")

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
