from __future__ import annotations

"""Lifecycle: candidate.

Endpoint registry for the KeyStore/CoinGlass v4 replacement route. Keep endpoint
metadata here instead of scattering path and parameter guesses across research
scripts. Promote only after coverage/overlap validation passes.
"""

from dataclasses import dataclass, field
from typing import Any


KEYSTORE_NATIVE_INTERVALS = ("1h", "2h", "4h", "6h", "8h", "12h", "1d")
DEFAULT_EXCHANGE = "Binance"
DEFAULT_EXCHANGE_LIST = "Binance,OKX,Bybit"


@dataclass(frozen=True)
class KeystoreEndpointSpec:
    name: str
    path: str
    family: str
    params_kind: str
    parser: str
    cache_prefix: str
    research_role: str
    native_intervals: tuple[str, ...] = KEYSTORE_NATIVE_INTERVALS
    limit: int = 1000
    static_params: dict[str, str] = field(default_factory=dict)
    interval_limits: dict[str, int] = field(default_factory=dict)
    supported_symbols: tuple[str, ...] | None = None
    migration_type: str = ""
    notes: str = ""
    pagination_kind: str = "end_time_backward"
    required_columns: tuple[str, ...] = ()

    def supports_interval(self, interval: str) -> bool:
        return interval in self.native_intervals


def pair_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return normalized if normalized.endswith("USDT") else f"{normalized}USDT"


def coin_symbol(symbol: str) -> str:
    return symbol.strip().upper().removesuffix("USDT")


def build_history_params(
    spec: KeystoreEndpointSpec,
    *,
    symbol: str,
    interval: str,
    limit: int | None = None,
    end_time_ms: int | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    exchange_list: str = DEFAULT_EXCHANGE_LIST,
) -> dict[str, Any]:
    if not spec.supports_interval(interval):
        raise ValueError(f"{spec.name} does not support interval={interval}")

    effective_limit = limit or spec.interval_limits.get(interval, spec.limit)
    params: dict[str, Any] = {"interval": interval, "limit": str(effective_limit)}
    if spec.params_kind == "pair_exchange_interval":
        params.update({"exchange": exchange, "symbol": pair_symbol(symbol)})
    elif spec.params_kind == "coin_interval":
        params.update({"symbol": coin_symbol(symbol)})
    elif spec.params_kind == "coin_exchange_list_interval":
        params.update({"exchange_list": exchange_list, "symbol": coin_symbol(symbol)})
    elif spec.params_kind == "none_interval":
        pass
    else:
        raise ValueError(f"Unknown KeyStore params_kind: {spec.params_kind}")

    params.update(spec.static_params)
    if end_time_ms is not None:
        params["end_time"] = str(end_time_ms)
    return params


BASE_REPLACEMENT_ENDPOINTS = [
    KeystoreEndpointSpec(
        name="oi",
        path="/api/futures/open-interest/aggregated-history",
        family="open_interest",
        params_kind="coin_interval",
        parser="oi_ohlc",
        cache_prefix="oi",
        research_role="base_replacement",
        limit=4500,
        static_params={"unit": "usd"},
        migration_type="v3_to_v4_path_change",
        notes="Candidate replacement for v3 aggregated OI.",
        required_columns=("oi_open", "oi_high", "oi_low", "oi_close"),
    ),
    KeystoreEndpointSpec(
        name="liq",
        path="/api/futures/liquidation/aggregated-history",
        family="liquidation",
        params_kind="coin_exchange_list_interval",
        parser="liquidation",
        cache_prefix="liq",
        research_role="base_replacement",
        limit=4500,
        migration_type="v3_to_v4_path_change",
        required_columns=("long_liq", "short_liq", "net_liq", "total_liq"),
    ),
    KeystoreEndpointSpec(
        name="global_ls",
        path="/api/futures/global-long-short-account-ratio/history",
        family="long_short_ratio",
        params_kind="pair_exchange_interval",
        parser="global_ls",
        cache_prefix="global_ls",
        research_role="base_replacement",
        limit=4500,
        migration_type="v3_to_v4_path_change",
        required_columns=("global_ls_ratio",),
    ),
    KeystoreEndpointSpec(
        name="top_acct",
        path="/api/futures/top-long-short-account-ratio/history",
        family="long_short_ratio",
        params_kind="pair_exchange_interval",
        parser="top_acct",
        cache_prefix="top_acct",
        research_role="base_replacement",
        limit=4500,
        migration_type="v3_to_v4_path_change",
        required_columns=("top_acct_ls_ratio",),
    ),
    KeystoreEndpointSpec(
        name="top_pos",
        path="/api/futures/top-long-short-position-ratio/history",
        family="long_short_ratio",
        params_kind="pair_exchange_interval",
        parser="top_pos",
        cache_prefix="top_pos",
        research_role="base_replacement",
        limit=4500,
        migration_type="v3_to_v4_path_change",
        required_columns=("top_pos_ls_ratio",),
    ),
    KeystoreEndpointSpec(
        name="fr",
        path="/api/futures/funding-rate/history",
        family="funding",
        params_kind="pair_exchange_interval",
        parser="fr_ohlc",
        cache_prefix="fr",
        research_role="base_replacement",
        limit=4500,
        migration_type="v3_to_v4_path_change",
        required_columns=("fr_close",),
    ),
    KeystoreEndpointSpec(
        name="taker_pair",
        path="/api/futures/v2/taker-buy-sell-volume/history",
        family="taker_flow",
        params_kind="pair_exchange_interval",
        parser="taker_pair",
        cache_prefix="taker_pair",
        research_role="base_replacement",
        migration_type="v4_source_replacement",
        limit=4500,
        required_columns=("buy", "sell"),
    ),
    KeystoreEndpointSpec(
        name="taker_agg",
        path="/api/futures/aggregated-taker-buy-sell-volume/history",
        family="taker_flow",
        params_kind="coin_exchange_list_interval",
        parser="taker_agg",
        cache_prefix="taker_agg",
        research_role="base_replacement",
        migration_type="v4_source_replacement",
        limit=4500,
        static_params={"unit": "usd"},
        required_columns=("buy", "sell"),
    ),
    KeystoreEndpointSpec(
        name="basis",
        path="/api/futures/basis/history",
        family="basis",
        params_kind="pair_exchange_interval",
        parser="basis",
        cache_prefix="basis",
        research_role="base_replacement",
        limit=4500,
        interval_limits={"1d": 1000},
        migration_type="v4_source_replacement",
        required_columns=("close_basis",),
    ),
    KeystoreEndpointSpec(
        name="fr_oi_weight",
        path="/api/futures/funding-rate/oi-weight-history",
        family="funding_weighted",
        params_kind="coin_interval",
        parser="ohlc",
        cache_prefix="fr_oi_weight",
        research_role="base_replacement",
        limit=4500,
        migration_type="v4_source_replacement_symbol_shape_change",
        notes="Verified sample: BTC returns rows, BTCUSDT returns empty.",
        required_columns=("close",),
    ),
    KeystoreEndpointSpec(
        name="fr_vol_weight",
        path="/api/futures/funding-rate/vol-weight-history",
        family="funding_weighted",
        params_kind="coin_interval",
        parser="ohlc",
        cache_prefix="fr_vol_weight",
        research_role="base_replacement",
        limit=4500,
        migration_type="v4_source_replacement_symbol_shape_change",
        required_columns=("close",),
    ),
    KeystoreEndpointSpec(
        name="oi_stablecoin",
        path="/api/futures/open-interest/aggregated-stablecoin-history",
        family="open_interest",
        params_kind="coin_exchange_list_interval",
        parser="ohlc",
        cache_prefix="oi_stablecoin",
        research_role="base_replacement",
        static_params={"unit": "usd"},
        migration_type="v4_source_replacement",
        required_columns=("close",),
    ),
    KeystoreEndpointSpec(
        name="oi_coin_margin",
        path="/api/futures/open-interest/aggregated-coin-margin-history",
        family="open_interest",
        params_kind="coin_exchange_list_interval",
        parser="ohlc",
        cache_prefix="oi_coin_margin",
        research_role="candidate_factor",
        static_params={"unit": "usd"},
        migration_type="v4_source_replacement",
        required_columns=("close",),
    ),
    KeystoreEndpointSpec(
        name="bitfinex_margin",
        path="/api/bitfinex-margin-long-short",
        family="margin",
        params_kind="coin_interval",
        parser="bitfinex_margin",
        cache_prefix="bitfinex_margin",
        research_role="candidate_factor",
        supported_symbols=("BTC", "ETH"),
        migration_type="v4_source_replacement",
    ),
]


CANDIDATE_ENDPOINTS = [
    KeystoreEndpointSpec(
        name="futures_cvd",
        path="/api/futures/cvd/history",
        family="cvd",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_cvd",
        research_role="candidate_factor",
        limit=4500,
        required_columns=("cum_vol_delta",),
    ),
    KeystoreEndpointSpec(
        name="futures_cvd_agg",
        path="/api/futures/aggregated-cvd/history",
        family="cvd",
        params_kind="coin_exchange_list_interval",
        parser="generic_numeric",
        cache_prefix="futures_cvd_agg",
        research_role="candidate_factor",
        limit=4500,
        static_params={"unit": "usd"},
        required_columns=("cum_vol_delta",),
    ),
    KeystoreEndpointSpec(
        name="futures_net_pos_v2",
        path="/api/futures/v2/net-position/history",
        family="net_position",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_net_pos_v2",
        research_role="candidate_factor",
        required_columns=("net_position_change_cum",),
    ),
    KeystoreEndpointSpec(
        name="futures_net_pos",
        path="/api/futures/net-position/history",
        family="net_position",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_net_pos",
        research_role="candidate_factor",
        required_columns=("net_position_change_cum",),
    ),
    KeystoreEndpointSpec(
        name="ob_pair",
        path="/api/futures/orderbook/ask-bids-history",
        family="orderbook",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="ob_pair",
        research_role="candidate_factor",
        static_params={"range": "1"},
        required_columns=("bids_usd", "asks_usd"),
    ),
    KeystoreEndpointSpec(
        name="ob_agg",
        path="/api/futures/orderbook/aggregated-ask-bids-history",
        family="orderbook",
        params_kind="coin_exchange_list_interval",
        parser="generic_numeric",
        cache_prefix="ob_agg",
        research_role="candidate_factor",
        static_params={"range": "1"},
        required_columns=("aggregated_bids_usd", "aggregated_asks_usd"),
    ),
    KeystoreEndpointSpec(
        name="spot_taker_pair",
        path="/api/spot/taker-buy-sell-volume/history",
        family="spot_taker_flow",
        params_kind="pair_exchange_interval",
        parser="taker_pair",
        cache_prefix="spot_taker_pair",
        research_role="candidate_factor",
        required_columns=("buy", "sell"),
    ),
    KeystoreEndpointSpec(
        name="spot_cvd",
        path="/api/spot/cvd/history",
        family="spot_cvd",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="spot_cvd",
        research_role="candidate_factor",
        required_columns=("cum_vol_delta",),
    ),
    KeystoreEndpointSpec(
        name="spot_cvd_agg",
        path="/api/spot/aggregated-cvd/history",
        family="spot_cvd",
        params_kind="coin_exchange_list_interval",
        parser="generic_numeric",
        cache_prefix="spot_cvd_agg",
        research_role="candidate_factor",
        required_columns=("cum_vol_delta",),
    ),
    KeystoreEndpointSpec(
        name="futures_rsi",
        path="/api/futures/indicators/rsi",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_rsi",
        research_role="candidate_factor",
        static_params={"window": "14", "series_type": "close"},
    ),
    KeystoreEndpointSpec(
        name="futures_ma",
        path="/api/futures/indicators/ma",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_ma",
        research_role="candidate_factor",
        static_params={"window": "20", "series_type": "close"},
    ),
    KeystoreEndpointSpec(
        name="futures_ema",
        path="/api/futures/indicators/ema",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_ema",
        research_role="candidate_factor",
        static_params={"window": "20", "series_type": "close"},
    ),
    KeystoreEndpointSpec(
        name="futures_boll",
        path="/api/futures/indicators/boll",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_boll",
        research_role="candidate_factor",
        static_params={"window": "20", "series_type": "close"},
    ),
    KeystoreEndpointSpec(
        name="futures_macd",
        path="/api/futures/indicators/macd",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_macd",
        research_role="candidate_factor",
    ),
    KeystoreEndpointSpec(
        name="futures_atr",
        path="/api/futures/indicators/avg-true-range",
        family="technical",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_atr",
        research_role="candidate_factor",
        static_params={"window": "14"},
    ),
    KeystoreEndpointSpec(
        name="futures_whale_index",
        path="/api/futures/whale-index/history",
        family="whale_index",
        params_kind="pair_exchange_interval",
        parser="generic_numeric",
        cache_prefix="futures_whale_index",
        research_role="candidate_factor",
        required_columns=("whale_index_value",),
    ),
]


ALL_ENDPOINTS = tuple(BASE_REPLACEMENT_ENDPOINTS + CANDIDATE_ENDPOINTS)
ENDPOINTS_BY_NAME = {endpoint.name: endpoint for endpoint in ALL_ENDPOINTS}


def select_endpoints(names: list[str] | None = None, roles: tuple[str, ...] = ("base_replacement",)) -> list[KeystoreEndpointSpec]:
    if names:
        missing = [name for name in names if name not in ENDPOINTS_BY_NAME]
        if missing:
            raise KeyError(f"Unknown KeyStore endpoint names: {missing}")
        return [ENDPOINTS_BY_NAME[name] for name in names]
    return [endpoint for endpoint in ALL_ENDPOINTS if endpoint.research_role in roles]
