import httpx
from urllib.parse import quote

BASE_URL = "https://api.tapetide.com/v1"


class TapetideAPIError(RuntimeError):
    """A documented Tapetide API error, including its stable error code."""

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.code = str(code)[:64]
        self.status_code = status_code


class TapetideClient:
    """Read-only client for Tapetide's documented Indian-equity REST API."""

    def __init__(self, token: str):
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )

    async def _get(self, path: str, params: dict | None = None):
        response = await self.client.get(path, params=params)
        try:
            body = response.json()
        except ValueError:
            body = None

        if response.is_error or (isinstance(body, dict) and "error" in body):
            error = body.get("error") if isinstance(body, dict) else None
            if not isinstance(error, dict):
                error = {}
            code = error.get("code")
            if not isinstance(code, (str, int)) or isinstance(code, bool) or not str(code).strip():
                code = f"http_{response.status_code}" if response.is_error else "api_error"
            message = error.get("message")
            if not isinstance(message, str) or not message.strip():
                message = response.reason_phrase or "Tapetide API error"
            raise TapetideAPIError(
                code=str(code),
                message=message.strip()[:240],
                status_code=response.status_code,
            )

        # A successful HTTP status is not enough: never turn an HTML error page,
        # JSON array, or missing data envelope into a plausible empty result.
        if not isinstance(body, dict):
            raise TapetideAPIError(
                "invalid_response",
                "Tapetide returned an unexpected response format.",
                status_code=response.status_code,
            )
        if "data" not in body:
            raise TapetideAPIError(
                "invalid_response",
                "Tapetide response is missing the data field.",
                status_code=response.status_code,
            )
        meta = body.get("meta", {})
        if not isinstance(meta, dict):
            raise TapetideAPIError(
                "invalid_response",
                "Tapetide response metadata has an unexpected format.",
                status_code=response.status_code,
            )

        # Preserve Tapetide's data/meta envelope, including as_of and route metadata.
        return {"data": body["data"], "meta": meta}

    async def search_stocks(
        self,
        query: str,
        *,
        limit: int = 10,
        sector: str | None = None,
        industry: str | None = None,
    ):
        """Search ~8,200 Indian stocks by name, symbol, BSE code or ISIN.

        Tapetide also matches common brand names/renamed aliases and tolerates
        typos in queries of four or more characters.
        """
        if not isinstance(query, str):
            raise ValueError("Stock search query must be text")
        query = query.strip()
        if not query:
            raise ValueError("Stock search query cannot be empty")

        params = {"q": query, "limit": max(1, min(int(limit), 50))}
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        return await self._get("/stocks", params)

    async def quote(self, symbol: str):
        symbol = quote(symbol.strip().upper(), safe="")
        return await self._get(f"/stocks/{symbol}/quote")

    async def live_quote(self, symbol: str):
        """Fetch the provider's session-live quote route; availability depends on coverage/session."""
        symbol = quote(symbol.strip().upper(), safe="")
        return await self._get(f"/stocks/{symbol}/quote/live")

    async def financials(self, symbol: str, section: str = "annual"):
        symbol = quote(symbol.strip().upper(), safe="")
        if section not in {"annual", "quarterly"}:
            raise ValueError("Financial section must be 'annual' or 'quarterly'")
        return await self._get(
            f"/stocks/{symbol}/financials", {"section": section}
        )

    async def price_history(self, symbol: str, days: int = 90):
        symbol = quote(symbol.strip().upper(), safe="")
        days = max(1, min(int(days), 365))
        return await self._get(f"/stocks/{symbol}/price", {"days": days})

    async def intraday_bars(
        self,
        symbol: str,
        *,
        interval: int = 5,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 375,
    ):
        """Fetch historical 1/5/15-minute NSE/BSE bars; Tapetide updates after session close."""
        symbol = quote(symbol.strip().upper(), safe="")
        if interval not in (1, 5, 15):
            raise ValueError("Intraday interval must be 1, 5 or 15 minutes")
        if date and (from_date or to_date):
            raise ValueError("Use date or from_date/to_date, not both")
        if bool(from_date) != bool(to_date):
            raise ValueError("from_date and to_date must be supplied together")
        params = {"interval": interval, "limit": max(1, min(int(limit), 2000))}
        if date:
            params["date"] = date
        elif from_date and to_date:
            params.update({"from": from_date, "to": to_date})
        return await self._get(f"/stocks/{symbol}/prices/intraday", params)

    async def close(self):
        await self.client.aclose()
