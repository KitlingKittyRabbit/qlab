"""Canonical realtime adapters for the frozen KSV4 TRUE OOS source contract.

The functions in this module only normalize as-received public market payloads
into the raw columns consumed by ``keystore_coinglass_panel``.  They do not
score candidates, select positions, or submit orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

from qlab.data.crypto.binance_um_market import BinanceUmProductionMarketClient
from qlab.data.crypto.keystore_coinglass_client import KeystoreCoinglassClient


SHADOW_SOURCE_CONTRACT_VERSION = "ksv4_shadow_sources_v4"
SHADOW_SOURCE_DEADLINE_SECONDS = 210
SHADOW_PUBLIC_WORKERS = 16
SHADOW_PUBLIC_RETRIES = 3
SHADOW_REQUEST_TIMEOUT_SECONDS = 20.0
SHADOW_ORDERBOOK_HISTORY_LIMIT = 10
SHADOW_EXPECTED_SOURCE_COUNTS = {
    "keystore": 26,
    "binance_public": 58,
    "bybit_public": 17,
    "okx_public": 17,
}
KEYSTORE_REALTIME_PATHS = {
    "coins_markets": "/api/futures/coins-markets",
    "funding_exchange_list": "/api/futures/funding-rate/exchange-list",
    "pairs_markets": "/api/futures/pairs-markets",
    "net_position_v2": "/api/futures/v2/net-position/history",
}


def pairs_markets_params(symbol: str) -> dict[str, str]:
    """Build the CoinGlass pairs-markets query from a base-asset symbol."""
    symbol_upper = str(symbol).strip().upper()
    if (
        not symbol_upper
        or not symbol_upper.isalnum()
        or symbol_upper.endswith(("USDT", "USDC"))
    ):
        raise ValueError("pairs-markets requires a base-asset symbol such as ADA")
    return {"symbol": symbol_upper}


@dataclass(frozen=True)
class PublicRawPayload:
    venue: str
    path: str
    request_params: dict[str, Any]
    raw_payload: bytes
    request_ts: str
    response_ts: str

    def parse_json(self) -> Any:
        return json.loads(self.raw_payload)


class PublicGetClient:
    """Host-locked, unauthenticated GET-only market-data client."""

    _HOSTS = {
        "okx": "https://www.okx.com",
        "bybit": "https://api.bybit.com",
    }

    def __init__(
        self,
        venue: str,
        *,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        venue_name = str(venue).strip().lower()
        if venue_name not in self._HOSTS:
            raise ValueError("public market venue must be okx or bybit")
        self.venue = venue_name
        self.base_url = self._HOSTS[venue_name]
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    def request_raw(
        self, path: str, *, params: Mapping[str, object] | None = None
    ) -> PublicRawPayload:
        if not str(path).startswith("/"):
            raise ValueError("public market path must be absolute")
        clean_params = {
            str(key): value
            for key, value in (params or {}).items()
            if value is not None
        }
        request_ts = datetime.now(timezone.utc).isoformat()
        response = self.session.get(
            f"{self.base_url}{path}",
            params=clean_params,
            headers={"User-Agent": "qlab-true-oos/1.0", "Accept": "application/json"},
            timeout=self.timeout,
        )
        response_ts = datetime.now(timezone.utc).isoformat()
        if response.status_code >= 400:
            raise RuntimeError(
                f"{self.venue} public market HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        raw_payload = bytes(response.content)
        if not raw_payload:
            raise RuntimeError(f"{self.venue} public market returned an empty payload")
        return PublicRawPayload(
            venue=self.venue,
            path=path,
            request_params=clean_params,
            raw_payload=raw_payload,
            request_ts=request_ts,
            response_ts=response_ts,
        )


@dataclass(frozen=True)
class ShadowSourceResponse:
    request_id: str
    request_order: int
    source: str
    route: str
    symbol: str
    signal_timeframe: str
    request_path: str
    request_params: dict[str, object]
    raw_payload: bytes
    request_ts: str
    response_ts: str


def build_shadow_source_contract(source_plan: pd.DataFrame) -> pd.DataFrame:
    """Turn the repaired 118-row logical plan into exact HTTP requests."""
    required = {
        "request_id", "request_order", "source", "route", "symbol",
        "signal_timeframe", "serialized",
    }
    missing = sorted(required.difference(source_plan.columns))
    if missing:
        raise ValueError("source plan missing columns: " + ", ".join(missing))
    if source_plan["request_id"].astype(str).duplicated().any():
        raise ValueError("source plan request_id must be unique")
    counts = source_plan.groupby("source").size().astype(int).to_dict()
    if counts != SHADOW_EXPECTED_SOURCE_COUNTS or len(source_plan) != 118:
        raise ValueError(
            f"shadow source counts mismatch: expected={SHADOW_EXPECTED_SOURCE_COUNTS}, "
            f"actual={counts}"
        )

    def request_for(row: object) -> tuple[str, dict[str, object]]:
        source = str(row.source)
        route = str(row.route)
        symbol = str(row.symbol)
        timeframe = str(row.signal_timeframe)
        if source == "keystore":
            if route == "coins-markets":
                return KEYSTORE_REALTIME_PATHS["coins_markets"], {
                    "exchange_list": "Binance,OKX,Bybit", "per_page": 200, "page": 1,
                }
            if route == "funding-rate/exchange-list":
                return KEYSTORE_REALTIME_PATHS["funding_exchange_list"], {}
            if route == "futures/v2/net-position/history":
                return KEYSTORE_REALTIME_PATHS["net_position_v2"], {
                    "exchange": "Binance", "symbol": f"{symbol}USDT",
                    "interval": "1h", "limit": 2,
                }
            if route == "orderbook/ask-bids-history":
                return "/api/futures/orderbook/ask-bids-history", {
                    "exchange": "Binance", "symbol": f"{symbol}USDT",
                    "interval": "1m", "limit": SHADOW_ORDERBOOK_HISTORY_LIMIT,
                    "range": "1",
                }
            if route == "orderbook/aggregated-ask-bids-history":
                return "/api/futures/orderbook/aggregated-ask-bids-history", {
                    "exchange_list": "Binance,OKX,Bybit", "symbol": symbol,
                    "interval": "1m", "limit": SHADOW_ORDERBOOK_HISTORY_LIMIT,
                    "range": "1",
                }
        if source == "binance_public" and route == "top-position-ratio":
            return "/futures/data/topLongShortPositionRatio", {
                "symbol": f"{symbol}USDT", "period": timeframe, "limit": 2,
            }
        if source == "binance_public" and route == "orderbook":
            return "/fapi/v1/depth", {"symbol": f"{symbol}USDT", "limit": 1000}
        if source == "okx_public" and route == "orderbook":
            return "/api/v5/market/books", {
                "instId": f"{symbol}-USDT-SWAP", "sz": 400,
            }
        if source == "bybit_public" and route == "orderbook":
            return "/v5/market/orderbook", {
                "category": "linear", "symbol": f"{symbol}USDT", "limit": 500,
            }
        raise ValueError(f"unsupported shadow source request: {source}/{route}")

    rows = []
    for row in source_plan.sort_values("request_order").itertuples(index=False):
        path, params = request_for(row)
        rows.append(
            {
                **row._asdict(),
                "request_path": path,
                "request_params_json": json.dumps(
                    params, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ),
                "source_contract_version": SHADOW_SOURCE_CONTRACT_VERSION,
                "request_timeout_seconds": SHADOW_REQUEST_TIMEOUT_SECONDS,
                "max_attempts": (
                    4 if str(row.source) == "keystore" else SHADOW_PUBLIC_RETRIES
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_shadow_source_responses(
    contract: pd.DataFrame,
    *,
    deadline_ts: str | pd.Timestamp,
    keystore_start_interval_seconds: float = 6.2,
    public_workers: int = SHADOW_PUBLIC_WORKERS,
) -> list[ShadowSourceResponse]:
    """Collect the frozen route concurrently without parsing response bytes."""
    required = {
        "request_id", "request_order", "source", "route", "symbol",
        "signal_timeframe", "request_path", "request_params_json",
        "source_contract_version",
    }
    missing = sorted(required.difference(contract.columns))
    if missing:
        raise ValueError("shadow source contract missing columns: " + ", ".join(missing))
    counts = contract.groupby("source").size().astype(int).to_dict()
    if counts != SHADOW_EXPECTED_SOURCE_COUNTS or len(contract) != 118:
        raise ValueError("shadow source contract no longer contains the frozen 118 requests")
    if not contract["source_contract_version"].eq(SHADOW_SOURCE_CONTRACT_VERSION).all():
        raise ValueError("shadow source contract version mismatch")
    deadline = pd.Timestamp(deadline_ts)
    deadline = deadline.tz_localize("UTC") if deadline.tz is None else deadline.tz_convert("UTC")

    def ensure_before_deadline(request_id: str) -> None:
        if pd.Timestamp.now(tz="UTC") > deadline:
            raise TimeoutError(f"shadow source deadline exceeded: {request_id}")

    def response_record(row: object, observed: object) -> ShadowSourceResponse:
        ensure_before_deadline(str(row.request_id))
        raw_payload = bytes(observed.raw_payload)
        if not raw_payload:
            raise RuntimeError(f"shadow source returned empty payload: {row.request_id}")
        return ShadowSourceResponse(
            request_id=str(row.request_id), request_order=int(row.request_order),
            source=str(row.source), route=str(row.route), symbol=str(row.symbol),
            signal_timeframe=str(row.signal_timeframe), request_path=str(row.request_path),
            request_params=json.loads(str(row.request_params_json)),
            raw_payload=raw_payload, request_ts=str(observed.request_ts),
            response_ts=str(observed.response_ts),
        )

    def collect_keystore(rows: pd.DataFrame) -> list[ShadowSourceResponse]:
        client = KeystoreCoinglassClient(
            timeout=SHADOW_REQUEST_TIMEOUT_SECONDS,
            rate_limit_sleep=keystore_start_interval_seconds,
        )
        output = []
        previous_start: float | None = None
        for row in rows.sort_values("request_order").itertuples(index=False):
            ensure_before_deadline(str(row.request_id))
            if previous_start is not None:
                wait = keystore_start_interval_seconds - (time.monotonic() - previous_start)
                if wait > 0:
                    time.sleep(wait)
            previous_start = time.monotonic()
            observed = client.request_raw(
                str(row.request_path), params=json.loads(str(row.request_params_json))
            )
            output.append(response_record(row, observed))
        return output

    def collect_public_row(row: object) -> ShadowSourceResponse:
        last_error: Exception | None = None
        for attempt in range(SHADOW_PUBLIC_RETRIES):
            ensure_before_deadline(str(row.request_id))
            try:
                params = json.loads(str(row.request_params_json))
                if str(row.source) == "binance_public":
                    observed = BinanceUmProductionMarketClient(
                        timeout=SHADOW_REQUEST_TIMEOUT_SECONDS
                    ).request_raw(str(row.request_path), params=params)
                else:
                    venue = "okx" if str(row.source) == "okx_public" else "bybit"
                    observed = PublicGetClient(
                        venue, timeout=SHADOW_REQUEST_TIMEOUT_SECONDS
                    ).request_raw(str(row.request_path), params=params)
                return response_record(row, observed)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < SHADOW_PUBLIC_RETRIES:
                    time.sleep(float(attempt + 1))
        raise RuntimeError(f"shadow public request failed: {row.request_id}: {last_error}")

    keystore_rows = contract.loc[contract["source"].eq("keystore")]
    public_rows = contract.loc[~contract["source"].eq("keystore")]
    with ThreadPoolExecutor(max_workers=2) as routes:
        keystore_future = routes.submit(collect_keystore, keystore_rows)
        def collect_public() -> list[ShadowSourceResponse]:
            output = []
            with ThreadPoolExecutor(max_workers=public_workers) as executor:
                futures = [
                    executor.submit(collect_public_row, row)
                    for row in public_rows.itertuples(index=False)
                ]
                for future in as_completed(futures):
                    output.append(future.result())
            return output
        public_future = routes.submit(collect_public)
        responses = [*keystore_future.result(), *public_future.result()]
    if len(responses) != 118 or len({row.request_id for row in responses}) != 118:
        raise RuntimeError("shadow source collection did not return all 118 responses")
    return sorted(responses, key=lambda row: row.request_order)


def _rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("market payload must be an object or row list")
    code = str(payload.get("code", "0"))
    if code not in {"0", "", "200", "00000"}:
        raise ValueError(f"market payload returned code={code}")
    data = payload.get("data", payload.get("result"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping):
        nested = data.get("list", data.get("rows"))
        if isinstance(nested, list):
            return [dict(row) for row in nested if isinstance(row, Mapping)]
    raise ValueError("market payload does not contain row data")


def index_coins_markets(payload: object) -> dict[str, dict[str, object]]:
    result = {}
    for row in _rows(payload):
        symbol = str(row.get("symbol", "")).upper()
        if symbol:
            result[symbol] = row
    return result


def index_funding_exchange_list(payload: object) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in _rows(payload):
        symbol = str(row.get("symbol", "")).upper()
        stable = row.get("stablecoin_margin_list", [])
        if not symbol or not isinstance(stable, list):
            continue
        matches = [
            item
            for item in stable
            if isinstance(item, Mapping)
            and str(item.get("exchange", "")).lower() == "binance"
            and item.get("funding_rate") is not None
        ]
        if len(matches) == 1:
            result[symbol] = float(matches[0]["funding_rate"])
    return result


def select_binance_usdt_pair(payload: object, symbol: str) -> dict[str, object]:
    symbol_upper = str(symbol).upper()
    matches = [
        row
        for row in _rows(payload)
        if str(row.get("exchange_name", "")).lower() == "binance"
        and str(row.get("instrument_id", "")).upper() == f"{symbol_upper}USDT"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Binance {symbol_upper}USDT pairs-markets row, got {len(matches)}"
        )
    return matches[0]


def close_basis_percent(pair_row: Mapping[str, object]) -> float:
    current = float(pair_row["current_price"])
    index = float(pair_row["index_price"])
    if index <= 0.0 or current <= 0.0:
        raise ValueError("basis prices must be positive")
    return (index - current) / index * 100.0


def _market_timestamp(value: object) -> pd.Timestamp:
    numeric = float(value)
    unit = "ms" if abs(numeric) >= 10_000_000_000 else "s"
    return pd.Timestamp(numeric, unit=unit, tz="UTC")


def latest_net_position_observation(payload: object) -> tuple[float, pd.Timestamp]:
    rows = _rows(payload)
    if not rows:
        raise ValueError("net-position payload is empty")
    def timestamp(row: Mapping[str, object]) -> float:
        for key in ("time", "timestamp", "t"):
            if row.get(key) is not None:
                return float(row[key])
        return 0.0
    latest = max(rows, key=timestamp)
    for key in (
        "net_position_change_cum",
        "netPositionChangeCum",
        "net_position_cum",
    ):
        if latest.get(key) is not None:
            return float(latest[key]), _market_timestamp(timestamp(latest))
    raise ValueError("net-position payload lacks cumulative net-position field")


def net_position_observation_at(
    payload: object, target_label_ts: str | pd.Timestamp
) -> tuple[float, pd.Timestamp]:
    target = pd.Timestamp(target_label_ts)
    target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    matches = []
    for row in _rows(payload):
        raw_ts = next(
            (
                row.get(key)
                for key in ("time", "timestamp", "t")
                if row.get(key) is not None
            ),
            None,
        )
        if raw_ts is not None and _market_timestamp(raw_ts) == target:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one net-position row at {target.isoformat()}, got {len(matches)}"
        )
    return latest_net_position_observation(matches)


def latest_net_position_value(payload: object) -> float:
    return latest_net_position_observation(payload)[0]


def latest_ratio_observation(payload: object) -> tuple[float, pd.Timestamp]:
    rows = _rows(payload)
    if not rows:
        raise ValueError("long-short ratio payload is empty")
    latest = max(rows, key=lambda row: float(row.get("timestamp", 0)))
    if latest.get("longShortRatio") is None:
        raise ValueError("ratio payload lacks longShortRatio")
    return (
        float(latest["longShortRatio"]),
        _market_timestamp(latest.get("timestamp", 0)),
    )


def ratio_observation_at(
    payload: object, target_label_ts: str | pd.Timestamp
) -> tuple[float, pd.Timestamp]:
    target = pd.Timestamp(target_label_ts)
    target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    matching = [
        row
        for row in _rows(payload)
        if row.get("timestamp") is not None
        and _market_timestamp(row["timestamp"]) == target
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one ratio row at {target.isoformat()}, got {len(matching)}"
        )
    return latest_ratio_observation(matching)


def latest_ratio_value(payload: object) -> float:
    return latest_ratio_observation(payload)[0]


def latest_orderbook_history_imbalance(
    payload: object,
    *,
    bid_key: str,
    ask_key: str,
    allow_incomplete_latest: bool = False,
) -> tuple[float, pd.Timestamp]:
    """Select a positive fixed-band history row and return its imbalance."""
    rows = _rows(payload)
    if not rows:
        raise ValueError("orderbook history payload is empty")

    def timestamp(row: Mapping[str, object]) -> float:
        for key in ("time", "timestamp", "t"):
            if row.get(key) is not None:
                return float(row[key])
        raise ValueError("orderbook history row lacks timestamp")

    ordered = sorted(rows, key=timestamp)
    latest = ordered[-1]
    candidates = ordered if allow_incomplete_latest else [latest]
    positive = [
        row for row in candidates
        if row.get(bid_key) is not None
        and row.get(ask_key) is not None
        and float(row[bid_key]) > 0.0
        and float(row[ask_key]) > 0.0
    ]
    if not positive:
        raise ValueError("orderbook history has no eligible positive depth row")
    selected = positive[-1]
    bid = float(selected[bid_key])
    ask = float(selected[ask_key])
    return (bid - ask) / (bid + ask), _market_timestamp(timestamp(selected))


def orderbook_history_imbalance_at(
    payload: object,
    *,
    target_label_ts: str | pd.Timestamp,
    bid_key: str,
    ask_key: str,
) -> tuple[float, pd.Timestamp]:
    target = pd.Timestamp(target_label_ts)
    target = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    matching = []
    for row in _rows(payload):
        raw_ts = next(
            (
                row.get(key)
                for key in ("time", "timestamp", "t")
                if row.get(key) is not None
            ),
            None,
        )
        if raw_ts is not None and _market_timestamp(raw_ts) == target:
            matching.append(row)
    if len(matching) != 1:
        raise ValueError(
            f"expected one orderbook row at {target.isoformat()}, got {len(matching)}"
        )
    return latest_orderbook_history_imbalance(
        matching,
        bid_key=bid_key,
        ask_key=ask_key,
    )


def usd_depth_within_band(
    bids: Sequence[Sequence[object]],
    asks: Sequence[Sequence[object]],
    *,
    midpoint: float | None = None,
    band_fraction: float = 0.01,
    quantity_multiplier: float = 1.0,
) -> tuple[float, float]:
    bid_rows = [(float(row[0]), float(row[1])) for row in bids]
    ask_rows = [(float(row[0]), float(row[1])) for row in asks]
    if not bid_rows or not ask_rows:
        raise ValueError("orderbook must contain bids and asks")
    mid = midpoint or (max(price for price, _ in bid_rows) + min(price for price, _ in ask_rows)) / 2.0
    if mid <= 0.0 or not 0.0 < band_fraction < 1.0 or quantity_multiplier <= 0.0:
        raise ValueError("invalid orderbook normalization inputs")
    lower = mid * (1.0 - band_fraction)
    upper = mid * (1.0 + band_fraction)
    bid_usd = sum(
        price * quantity * quantity_multiplier
        for price, quantity in bid_rows
        if lower <= price <= mid
    )
    ask_usd = sum(
        price * quantity * quantity_multiplier
        for price, quantity in ask_rows
        if mid <= price <= upper
    )
    if bid_usd <= 0.0 or ask_usd <= 0.0:
        raise ValueError("orderbook has no positive depth inside the fixed band")
    return bid_usd, ask_usd


def depth_covers_band(
    bids: Sequence[Sequence[object]],
    asks: Sequence[Sequence[object]],
    *,
    midpoint: float | None = None,
    band_fraction: float = 0.01,
) -> bool:
    """Return whether a finite order-book snapshot spans the full price band."""
    bid_prices = [float(row[0]) for row in bids]
    ask_prices = [float(row[0]) for row in asks]
    if not bid_prices or not ask_prices:
        return False
    mid = midpoint or (max(bid_prices) + min(ask_prices)) / 2.0
    if mid <= 0.0 or not 0.0 < band_fraction < 1.0:
        raise ValueError("invalid orderbook coverage inputs")
    return min(bid_prices) <= mid * (1.0 - band_fraction) and max(ask_prices) >= mid * (1.0 + band_fraction)


def binance_depth_usd(payload: object) -> tuple[float, float]:
    if not isinstance(payload, Mapping):
        raise ValueError("Binance depth payload must be an object")
    return usd_depth_within_band(payload.get("bids", []), payload.get("asks", []))


def bybit_depth_usd(payload: object) -> tuple[float, float]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
        raise ValueError("Bybit depth payload is malformed")
    result = payload["result"]
    return usd_depth_within_band(result.get("b", []), result.get("a", []))


def okx_contract_multiplier(instrument_payload: object) -> float:
    rows = _rows(instrument_payload)
    if len(rows) != 1:
        raise ValueError("OKX instrument payload must identify exactly one swap")
    row = rows[0]
    ct_val = float(row["ctVal"])
    ct_val_ccy = str(row.get("ctValCcy", "")).upper()
    settle_ccy = str(row.get("settleCcy", "")).upper()
    if ct_val <= 0.0 or not ct_val_ccy or ct_val_ccy == settle_ccy:
        raise ValueError("OKX swap must expose a positive base-coin contract value")
    return ct_val


def okx_depth_usd(
    book_payload: object, instrument_payload: object
) -> tuple[float, float]:
    rows = _rows(book_payload)
    if len(rows) != 1:
        raise ValueError("OKX depth payload must contain one book")
    row = rows[0]
    return usd_depth_within_band(
        row.get("bids", []),
        row.get("asks", []),
        quantity_multiplier=okx_contract_multiplier(instrument_payload),
    )


def aggregate_depth_usd(
    binance: tuple[float, float],
    okx: tuple[float, float],
    bybit: tuple[float, float],
) -> tuple[float, float]:
    return (
        float(binance[0] + okx[0] + bybit[0]),
        float(binance[1] + okx[1] + bybit[1]),
    )


def aggregate_available_depth_usd(
    venue_depths: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    """Sum USD depth across the explicitly frozen available venue set."""
    if not venue_depths:
        raise ValueError("aggregated orderbook requires at least one available venue")
    bids = sum(float(depth[0]) for depth in venue_depths.values())
    asks = sum(float(depth[1]) for depth in venue_depths.values())
    if bids <= 0.0 or asks <= 0.0:
        raise ValueError("available venue depth must be positive")
    return bids, asks


def realtime_raw_values(
    *,
    symbol: str,
    coins_row: Mapping[str, object],
    funding_rate: float,
    pair_row: Mapping[str, object],
    net_position_value: float,
    global_ratio: float,
    top_account_ratio: float,
    top_position_ratio: float,
    pair_depth: tuple[float, float],
    aggregate_depth: tuple[float, float],
) -> dict[str, dict[str, float]]:
    """Return endpoint-keyed raw columns expected by the frozen registry."""
    required = (
        "avg_funding_rate_by_oi",
        "avg_funding_rate_by_vol",
        "open_interest_usd",
        "long_liquidation_usd_24h",
        "short_liquidation_usd_24h",
    )
    missing = [key for key in required if coins_row.get(key) is None]
    if missing:
        raise ValueError(
            f"coins-markets row for {str(symbol).upper()} missing: {', '.join(missing)}"
        )
    return {
        "basis": {"close_basis": close_basis_percent(pair_row)},
        "fr": {"fr_close": float(funding_rate)},
        "fr_oi_weight": {"close": float(coins_row["avg_funding_rate_by_oi"])},
        "fr_vol_weight": {"close": float(coins_row["avg_funding_rate_by_vol"])},
        "oi": {"oi_close": float(coins_row["open_interest_usd"])},
        "liq": {
            "long_liq": float(coins_row["long_liquidation_usd_24h"]),
            "short_liq": float(coins_row["short_liquidation_usd_24h"]),
        },
        "global_ls": {"global_ls_ratio": float(global_ratio)},
        "top_acct": {"top_acct_ls_ratio": float(top_account_ratio)},
        "top_pos": {"top_pos_ls_ratio": float(top_position_ratio)},
        "futures_net_pos_v2": {
            "net_position_change_cum": float(net_position_value)
        },
        "ob_pair": {"bids_usd": float(pair_depth[0]), "asks_usd": float(pair_depth[1])},
        "ob_agg": {
            "aggregated_bids_usd": float(aggregate_depth[0]),
            "aggregated_asks_usd": float(aggregate_depth[1]),
        },
    }


def repaired_realtime_raw_values(
    *,
    coins_row: Mapping[str, object],
    funding_rate: float,
    net_position_value: float,
    top_position_ratio_1h: float,
    top_position_ratio_12h: float,
    pair_depth: tuple[float, float],
    aggregate_depth: tuple[float, float],
) -> dict[str, dict[str, float]]:
    """Return only the eight endpoints used by the repaired 100-candidate line."""
    required = (
        "avg_funding_rate_by_oi", "avg_funding_rate_by_vol", "open_interest_usd",
    )
    missing = [key for key in required if coins_row.get(key) is None]
    if missing:
        raise ValueError("coins-markets row missing: " + ", ".join(missing))
    common = {
        "fr": {"fr_close": float(funding_rate)},
        "fr_oi_weight": {"close": float(coins_row["avg_funding_rate_by_oi"])},
        "fr_vol_weight": {"close": float(coins_row["avg_funding_rate_by_vol"])},
        "oi": {"oi_close": float(coins_row["open_interest_usd"])},
        "futures_net_pos_v2": {
            "net_position_change_cum": float(net_position_value)
        },
        "ob_pair": {"bids_usd": float(pair_depth[0]), "asks_usd": float(pair_depth[1])},
        "ob_agg": {
            "aggregated_bids_usd": float(aggregate_depth[0]),
            "aggregated_asks_usd": float(aggregate_depth[1]),
        },
    }
    common["ksv4_1h:top_pos"] = {"top_pos_ls_ratio": float(top_position_ratio_1h)}
    common["ksv4_12h:top_pos"] = {"top_pos_ls_ratio": float(top_position_ratio_12h)}
    return common


def realtime_projection_identity(
    registry_frame: pd.DataFrame,
    *,
    source_scope: str,
    endpoint: str,
    decision_ts: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    rows = registry_frame.loc[
        registry_frame["source_scope"].astype(str).eq(str(source_scope))
        & registry_frame["endpoint"].astype(str).eq(str(endpoint))
    ]
    if rows.empty:
        raise ValueError(f"registry identity is missing: {source_scope}/{endpoint}")
    timeframes = rows["signal_timeframe"].astype(str).unique()
    timestamp_kinds = rows["timestamp_kind"].astype(str).unique()
    if len(timeframes) != 1 or len(timestamp_kinds) != 1:
        raise ValueError(f"registry identity is ambiguous: {source_scope}/{endpoint}")
    decision = pd.Timestamp(decision_ts)
    decision = decision.tz_localize("UTC") if decision.tz is None else decision.tz_convert("UTC")
    timeframe = str(timeframes[0])
    duration = pd.Timedelta(days=1) if timeframe == "1d" else pd.Timedelta(timeframe)
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    if (decision - epoch) % duration != pd.Timedelta(0):
        raise ValueError(
            f"decision is not aligned to signal timeframe: {decision.isoformat()}/{timeframe}"
        )
    timestamp_kind = str(timestamp_kinds[0])
    if timestamp_kind == "bar_start":
        return decision - duration, decision
    if timestamp_kind == "bar_end":
        return decision, decision
    raise ValueError(f"registry has unsupported timestamp semantics: {timestamp_kind}")


def latest_completed_native_identity(
    *,
    signal_timeframe: str,
    timestamp_kind: str,
    as_of_ts: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Identify the latest native period completed no later than ``as_of_ts``."""
    as_of = pd.Timestamp(as_of_ts)
    as_of = as_of.tz_localize("UTC") if as_of.tz is None else as_of.tz_convert("UTC")
    timeframe = str(signal_timeframe)
    duration = pd.Timedelta(days=1) if timeframe == "1d" else pd.Timedelta(timeframe)
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    completed_end = as_of - ((as_of - epoch) % duration)
    kind = str(timestamp_kind)
    if kind == "bar_start":
        return completed_end - duration, completed_end
    if kind == "bar_end":
        return completed_end, completed_end
    raise ValueError(f"unsupported native timestamp semantics: {kind}")


def shadow_response_native_identity(
    response: ShadowSourceResponse,
    payload: object,
    *,
    decision_ts: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Bind an acquired shadow response to its latest completed native period."""
    decision = pd.Timestamp(decision_ts)
    decision = (
        decision.tz_localize("UTC")
        if decision.tz is None
        else decision.tz_convert("UTC")
    )
    route = str(response.route)
    if route == "futures/v2/net-position/history":
        label, native_end = latest_completed_native_identity(
            signal_timeframe="1h", timestamp_kind="bar_start", as_of_ts=decision
        )
        _, observed = net_position_observation_at(payload, label)
        if observed != label:
            raise ValueError("net-position response native identity mismatch")
        return observed, native_end
    if route == "top-position-ratio":
        label, native_end = latest_completed_native_identity(
            signal_timeframe=str(response.signal_timeframe),
            timestamp_kind="bar_start",
            as_of_ts=decision,
        )
        _, observed = ratio_observation_at(payload, label)
        if observed != label:
            raise ValueError("ratio response native identity mismatch")
        return observed, native_end
    if route in {
        "orderbook/ask-bids-history",
        "orderbook/aggregated-ask-bids-history",
    }:
        if route == "orderbook/ask-bids-history":
            bid_key, ask_key = "bids_usd", "asks_usd"
        else:
            bid_key, ask_key = "aggregated_bids_usd", "aggregated_asks_usd"
        _, observed = orderbook_history_imbalance_at(
            payload,
            target_label_ts=decision,
            bid_key=bid_key,
            ask_key=ask_key,
        )
        return observed, observed
    observed = pd.Timestamp(response.response_ts)
    observed = (
        observed.tz_localize("UTC")
        if observed.tz is None
        else observed.tz_convert("UTC")
    )
    return observed, observed


def build_realtime_cache_payloads(
    *,
    symbols: Sequence[str],
    registry_frame: pd.DataFrame,
    decision_ts: str | pd.Timestamp,
    values_by_symbol: Mapping[str, Mapping[str, Mapping[str, float]]],
    historical_payloads: Mapping[str, Mapping[str, pd.DataFrame]],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Overlay one observed decision row on prior cache rows for delta transforms."""
    decision = pd.Timestamp(decision_ts)
    decision = decision.tz_localize("UTC") if decision.tz is None else decision.tz_convert("UTC")
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for scope, rows in registry_frame.groupby("source_scope", sort=True):
        timeframe = str(rows["signal_timeframe"].iloc[0])
        result[str(scope)] = {}
        for symbol in symbols:
            symbol_upper = str(symbol).upper()
            for endpoint in sorted(rows["endpoint"].astype(str).unique()):
                label, _ = realtime_projection_identity(
                    registry_frame,
                    source_scope=str(scope),
                    endpoint=endpoint,
                    decision_ts=decision,
                )
                cache_key = f"{symbol_upper}_{endpoint}"
                scoped_endpoint = f"{scope}:{endpoint}"
                symbol_values = values_by_symbol.get(symbol_upper, {})
                value_key = scoped_endpoint if scoped_endpoint in symbol_values else endpoint
                if symbol_upper not in values_by_symbol or value_key not in symbol_values:
                    raise ValueError(f"missing realtime raw values: {symbol_upper}/{endpoint}")
                prior = historical_payloads.get(str(scope), {}).get(cache_key)
                if prior is None or prior.empty:
                    raise ValueError(f"missing historical seed frame: {scope}/{cache_key}")
                seeded = prior.loc[
                    pd.to_datetime(prior.index, utc=True) < label
                ].sort_index().tail(2).copy()
                if seeded.empty:
                    raise ValueError(
                        f"historical seed has no row before realtime label: "
                        f"{scope}/{cache_key}/{label.isoformat()}"
                    )
                current = pd.DataFrame(
                    [dict(symbol_values[value_key])],
                    index=pd.DatetimeIndex([label], name="ts"),
                )
                combined = pd.concat([seeded, current]).sort_index()
                result[str(scope)][cache_key] = combined[
                    ~combined.index.duplicated(keep="last")
                ]
    return result


__all__ = [
    "KEYSTORE_REALTIME_PATHS",
    "PublicGetClient",
    "PublicRawPayload",
    "SHADOW_EXPECTED_SOURCE_COUNTS",
    "SHADOW_PUBLIC_WORKERS",
    "SHADOW_REQUEST_TIMEOUT_SECONDS",
    "SHADOW_SOURCE_CONTRACT_VERSION",
    "SHADOW_SOURCE_DEADLINE_SECONDS",
    "ShadowSourceResponse",
    "aggregate_depth_usd",
    "aggregate_available_depth_usd",
    "binance_depth_usd",
    "build_realtime_cache_payloads",
    "build_shadow_source_contract",
    "bybit_depth_usd",
    "close_basis_percent",
    "collect_shadow_source_responses",
    "depth_covers_band",
    "index_coins_markets",
    "index_funding_exchange_list",
    "latest_net_position_value",
    "latest_net_position_observation",
    "net_position_observation_at",
    "latest_orderbook_history_imbalance",
    "orderbook_history_imbalance_at",
    "latest_ratio_observation",
    "ratio_observation_at",
    "latest_ratio_value",
    "latest_completed_native_identity",
    "okx_contract_multiplier",
    "okx_depth_usd",
    "pairs_markets_params",
    "realtime_raw_values",
    "realtime_projection_identity",
    "shadow_response_native_identity",
    "repaired_realtime_raw_values",
    "select_binance_usdt_pair",
    "usd_depth_within_band",
]
