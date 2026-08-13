from __future__ import annotations

import json

import pytest

from qlab.data.crypto.binance_um_market import BinanceUmProductionMarketClient


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload
        self.content = json.dumps(payload).encode("utf-8")

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("bookTicker"):
            return _Response([{"symbol": "BTCUSDT", "bidPrice": "99", "askPrice": "101"}])
        return _Response({"serverTime": 1})


def test_production_market_client_is_read_only_and_host_locked() -> None:
    with pytest.raises(ValueError, match="only permits"):
        BinanceUmProductionMarketClient(base_url="https://demo-fapi.binance.com")
    session = _Session()
    client = BinanceUmProductionMarketClient(session=session)
    result = client.book_tickers()
    assert result.payload[0]["bidPrice"] == "99"
    assert result.observed_ts.endswith("+00:00")
    raw = client.request_raw("/fapi/v1/ticker/bookTicker")
    assert raw.parse_json()[0]["askPrice"] == "101"
    assert raw.request_ts <= raw.response_ts
    assert session.calls[0][0] == "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    client.request_raw(
        "/futures/data/globalLongShortAccountRatio",
        params={"symbol": "BTCUSDT", "period": "12h", "limit": 1},
    )
    assert session.calls[-1][1]["params"] == {
        "symbol": "BTCUSDT",
        "period": "12h",
        "limit": 1,
    }
    assert not hasattr(client, "new_order")
