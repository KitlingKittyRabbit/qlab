"""Read-only Binance USD-M production market-data client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

import requests


PRODUCTION_MARKET_BASE_URL = "https://fapi.binance.com"


@dataclass(frozen=True)
class ObservedPayload:
    payload: Any
    observed_ts: str


@dataclass(frozen=True)
class RawObservedPayload:
    path: str
    raw_payload: bytes
    request_ts: str
    response_ts: str

    def parse_json(self) -> Any:
        return json.loads(self.raw_payload)


class BinanceUmProductionMarketClient:
    """Public-data-only client fixed to the Binance USD-M production host."""

    def __init__(
        self,
        *,
        base_url: str = PRODUCTION_MARKET_BASE_URL,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        if self.base_url != PRODUCTION_MARKET_BASE_URL:
            raise ValueError("production market client only permits fapi.binance.com")

    def request_raw(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> RawObservedPayload:
        request_ts = datetime.now(timezone.utc).isoformat()
        response = self.session.get(
            f"{self.base_url}{path}",
            params={
                key: value
                for key, value in (params or {}).items()
                if value is not None
            },
            timeout=self.timeout,
        )
        response_ts = datetime.now(timezone.utc).isoformat()
        if response.status_code >= 400:
            raise RuntimeError(
                f"Binance production market HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        raw_payload = bytes(response.content)
        if not raw_payload:
            raise RuntimeError("Binance production market returned an empty payload")
        return RawObservedPayload(
            path=path,
            raw_payload=raw_payload,
            request_ts=request_ts,
            response_ts=response_ts,
        )

    def _get(self, path: str) -> ObservedPayload:
        """Convenience parser for non-authoritative connectivity preflight only."""
        raw = self.request_raw(path)
        return ObservedPayload(raw.parse_json(), raw.response_ts)

    def server_time(self) -> ObservedPayload:
        return self._get("/fapi/v1/time")

    def exchange_info(self) -> ObservedPayload:
        return self._get("/fapi/v1/exchangeInfo")

    def book_tickers(self) -> ObservedPayload:
        return self._get("/fapi/v1/ticker/bookTicker")


__all__ = [
    "BinanceUmProductionMarketClient",
    "ObservedPayload",
    "PRODUCTION_MARKET_BASE_URL",
    "RawObservedPayload",
]
