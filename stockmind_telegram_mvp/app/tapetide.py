import httpx

BASE_URL = "https://api.tapetide.com/v1"


class TapetideClient:
    """Small read-only client for the documented Tapetide REST endpoints."""

    def __init__(self, token: str):
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )

    async def _get(self, path: str, params: dict | None = None):
        response = await self.client.get(path, params=params)
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            error = body["error"]
            raise RuntimeError(error.get("message", "Tapetide API error"))
        return {"data": body.get("data"), "meta": body.get("meta", {})}

    async def search_stocks(self, query: str):
        return await self._get("/stocks", {"q": query})

    async def quote(self, symbol: str):
        return await self._get(f"/stocks/{symbol}/quote")

    async def financials(self, symbol: str, section: str = "annual"):
        return await self._get(
            f"/stocks/{symbol}/financials", {"section": section}
        )

    async def price_history(self, symbol: str, days: int = 90):
        days = max(1, min(int(days), 365))
        return await self._get(
            f"/stocks/{symbol}/price", {"days": days}
        )

    async def close(self):
        await self.client.aclose()
