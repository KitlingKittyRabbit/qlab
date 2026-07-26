from __future__ import annotations

"""KeyStore/CoinGlass v4 factor registry for the crypto 1h panel route.

This module is qlab data infrastructure. It defines factor eligibility and
panel-transform metadata only; it does not run research gates or approve any
strategy conclusion.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from .keystore_coinglass_endpoints import ENDPOINTS_BY_NAME


PANEL_FREQUENCIES = ("1h",)
SIGNAL_SOURCE_FREQUENCIES_BY_PANEL_FREQUENCY = {
    "1h": ("1h", "2h", "4h", "6h", "8h", "12h", "1d"),
}
SOURCE_SCOPES_BY_PANEL_FREQUENCY = {
    panel_frequency: {
        f"ksv4_{signal_timeframe}"
        for signal_timeframe in signal_timeframes
    }
    for panel_frequency, signal_timeframes in SIGNAL_SOURCE_FREQUENCIES_BY_PANEL_FREQUENCY.items()
}
PANEL_TRANSFORMS = (
    "raw_column",
    "delta1_raw_column",
    "log_ratio",
    "log1p_ratio",
    "buy_minus_sell",
    "buy_sell_imbalance",
)
STANDARDIZATION_POLICIES = ("none", "rank_to_minus1_1")
REQUIRED_COLUMNS = {
    "feature_name",
    "family",
    "endpoint",
    "source_scope",
    "frequency",
    "signal_timeframe",
    "role_class",
    "design_role",
    "base_panel_allowed",
    "panel_transform",
    "cross_section_standardization",
    "required_columns",
    "timestamp_kind",
    "value_status",
    "earliest_safe_decision_rule",
    "notes",
}


@dataclass(frozen=True)
class FactorEligibilitySpec:
    feature_name: str
    family: str
    endpoint: str
    source_scope: str
    frequency: str
    signal_timeframe: str
    role_class: str
    design_role: str
    base_panel_allowed: str
    panel_transform: str
    cross_section_standardization: str
    required_columns: str
    timestamp_kind: str
    value_status: str
    earliest_safe_decision_rule: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["base_panel_allowed"] = str(self.base_panel_allowed).lower()
        return payload


def decision_rule(signal_timeframe: str) -> str:
    return (
        f"use completed {signal_timeframe} KeyStore v4 bars only on their native UTC-aligned "
        f"decision timestamps after label + {signal_timeframe} + 1m; do not carry forward "
        "to intermediate 1h grid rows"
    )


BASE_REPLACEMENT_TEMPLATES = (
    {
        "feature_name": "oi_close_delta1",
        "family": "oi",
        "endpoint": "oi",
        "panel_transform": "delta1_raw_column",
        "required_columns": "oi_close",
        "notes": "Open-interest close first difference; KeyStore v4 replacement.",
    },
    {
        "feature_name": "liq_log_ratio",
        "family": "liquidation",
        "endpoint": "liq",
        "panel_transform": "log1p_ratio",
        "required_columns": "long_liq,short_liq",
        "notes": "Directional liquidation pressure using KeyStore long/short fields.",
    },
    {
        "feature_name": "global_ls_ratio",
        "family": "ls_ratio",
        "endpoint": "global_ls",
        "panel_transform": "raw_column",
        "required_columns": "global_ls_ratio",
        "notes": "Global account long-short ratio level.",
    },
    {
        "feature_name": "global_ls_ratio_delta1",
        "family": "ls_ratio",
        "endpoint": "global_ls",
        "panel_transform": "delta1_raw_column",
        "required_columns": "global_ls_ratio",
        "notes": "Global account long-short ratio first difference.",
    },
    {
        "feature_name": "top_acct_ls_ratio",
        "family": "ls_ratio",
        "endpoint": "top_acct",
        "panel_transform": "raw_column",
        "required_columns": "top_acct_ls_ratio",
        "notes": "Top-account long-short ratio level.",
    },
    {
        "feature_name": "top_acct_ls_ratio_delta1",
        "family": "ls_ratio",
        "endpoint": "top_acct",
        "panel_transform": "delta1_raw_column",
        "required_columns": "top_acct_ls_ratio",
        "notes": "Top-account long-short ratio first difference.",
    },
    {
        "feature_name": "top_pos_ls_ratio",
        "family": "ls_ratio",
        "endpoint": "top_pos",
        "panel_transform": "raw_column",
        "required_columns": "top_pos_ls_ratio",
        "notes": "Top-position long-short ratio level.",
    },
    {
        "feature_name": "top_pos_ls_ratio_delta1",
        "family": "ls_ratio",
        "endpoint": "top_pos",
        "panel_transform": "delta1_raw_column",
        "required_columns": "top_pos_ls_ratio",
        "notes": "Top-position long-short ratio first difference.",
    },
    {
        "feature_name": "funding_close",
        "family": "funding",
        "endpoint": "fr",
        "panel_transform": "raw_column",
        "required_columns": "fr_close",
        "notes": "Funding close level.",
    },
    {
        "feature_name": "taker_pair_log_ratio",
        "family": "taker",
        "endpoint": "taker_pair",
        "panel_transform": "log_ratio",
        "required_columns": "buy,sell",
        "notes": "Futures pair taker buy/sell log ratio.",
    },
    {
        "feature_name": "taker_agg_net",
        "family": "taker",
        "endpoint": "taker_agg",
        "panel_transform": "buy_minus_sell",
        "required_columns": "buy,sell",
        "notes": "Aggregated futures taker buy minus sell.",
    },
    {
        "feature_name": "taker_agg_imbalance",
        "family": "taker",
        "endpoint": "taker_agg",
        "panel_transform": "buy_sell_imbalance",
        "required_columns": "buy,sell",
        "notes": "Aggregated futures taker imbalance.",
    },
    {
        "feature_name": "basis_close_basis",
        "family": "basis",
        "endpoint": "basis",
        "panel_transform": "raw_column",
        "required_columns": "close_basis",
        "notes": "Close basis level; old-source 12h overlap has known convention risk.",
    },
    {
        "feature_name": "funding_oi_weight_close",
        "family": "funding_weighted",
        "endpoint": "fr_oi_weight",
        "panel_transform": "raw_column",
        "required_columns": "close",
        "notes": "Open-interest weighted funding close.",
    },
    {
        "feature_name": "funding_vol_weight_close",
        "family": "funding_weighted",
        "endpoint": "fr_vol_weight",
        "panel_transform": "raw_column",
        "required_columns": "close",
        "notes": "Volume-weighted funding close.",
    },
    {
        "feature_name": "oi_stablecoin_close_delta1",
        "family": "oi_split",
        "endpoint": "oi_stablecoin",
        "panel_transform": "delta1_raw_column",
        "required_columns": "close",
        "notes": "Stablecoin-margined OI close first difference.",
    },
)


MAIN_CANDIDATE_TEMPLATES = (
    {
        "feature_name": "futures_cvd_agg_delta1",
        "family": "cvd",
        "endpoint": "futures_cvd_agg",
        "panel_transform": "delta1_raw_column",
        "required_columns": "cum_vol_delta",
        "notes": "Aggregated futures CVD first difference.",
    },
    {
        "feature_name": "spot_cvd_delta1",
        "family": "spot_cvd",
        "endpoint": "spot_cvd",
        "panel_transform": "delta1_raw_column",
        "required_columns": "cum_vol_delta",
        "notes": "Spot pair CVD first difference.",
    },
    {
        "feature_name": "spot_cvd_agg_delta1",
        "family": "spot_cvd",
        "endpoint": "spot_cvd_agg",
        "panel_transform": "delta1_raw_column",
        "required_columns": "cum_vol_delta",
        "notes": "Aggregated spot CVD first difference.",
    },
    {
        "feature_name": "spot_taker_imbalance",
        "family": "spot_taker",
        "endpoint": "spot_taker_pair",
        "panel_transform": "buy_sell_imbalance",
        "required_columns": "buy,sell",
        "notes": "Spot taker buy/sell imbalance.",
    },
    {
        "feature_name": "ob_pair_imbalance",
        "family": "orderbook",
        "endpoint": "ob_pair",
        "panel_transform": "buy_sell_imbalance",
        "required_columns": "bids_usd,asks_usd",
        "notes": "Pair orderbook imbalance: positive means bid depth is thicker.",
    },
    {
        "feature_name": "ob_agg_imbalance",
        "family": "orderbook",
        "endpoint": "ob_agg",
        "panel_transform": "buy_sell_imbalance",
        "required_columns": "aggregated_bids_usd,aggregated_asks_usd",
        "notes": "Aggregated orderbook imbalance: positive means bid depth is thicker.",
    },
    {
        "feature_name": "net_pos_delta1",
        "family": "net_position",
        "endpoint": "futures_net_pos_v2",
        "panel_transform": "delta1_raw_column",
        "required_columns": "net_position_change_cum",
        "notes": "Net position cumulative change first difference; v2 endpoint preferred.",
    },
    {
        "feature_name": "whale_index_raw",
        "family": "whale_index",
        "endpoint": "futures_whale_index",
        "panel_transform": "raw_column",
        "required_columns": "whale_index_value",
        "notes": "Whale index level.",
    },
)


BASE_PANEL_TEMPLATES = BASE_REPLACEMENT_TEMPLATES + MAIN_CANDIDATE_TEMPLATES


def canonical_feature_name(feature_name: str, signal_timeframe: str) -> str:
    return f"{feature_name}__{signal_timeframe}"


def infer_signal_timeframe(source_scope: str) -> str:
    for signal_timeframe in SIGNAL_SOURCE_FREQUENCIES_BY_PANEL_FREQUENCY["1h"]:
        if source_scope.endswith(f"_{signal_timeframe}"):
            return signal_timeframe
    return ""


def build_base_panel_specs() -> tuple[FactorEligibilitySpec, ...]:
    specs: list[FactorEligibilitySpec] = []
    for signal_timeframe in SIGNAL_SOURCE_FREQUENCIES_BY_PANEL_FREQUENCY["1h"]:
        for template in BASE_PANEL_TEMPLATES:
            specs.append(
                FactorEligibilitySpec(
                    feature_name=canonical_feature_name(template["feature_name"], signal_timeframe),
                    family=template["family"],
                    endpoint=template["endpoint"],
                    source_scope=f"ksv4_{signal_timeframe}",
                    frequency="1h",
                    signal_timeframe=signal_timeframe,
                    role_class="coin_level",
                    design_role="base_score",
                    base_panel_allowed="yes",
                    panel_transform=template["panel_transform"],
                    cross_section_standardization="rank_to_minus1_1",
                    required_columns=template["required_columns"],
                    timestamp_kind="bar_start",
                    value_status="final_historical_aggregate",
                    earliest_safe_decision_rule=decision_rule(signal_timeframe),
                    notes=template["notes"],
                )
            )
    return tuple(specs)


FACTOR_SPECS = build_base_panel_specs()


def build_factor_eligibility_registry() -> pd.DataFrame:
    frame = pd.DataFrame([spec.to_dict() for spec in FACTOR_SPECS])
    return validate_factor_eligibility_registry(frame)


def validate_factor_eligibility_registry(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("ksv4 registry missing required columns: " + ", ".join(sorted(missing)))

    registry = frame.copy()
    for column in REQUIRED_COLUMNS:
        registry[column] = registry[column].astype(str)
    registry["base_panel_allowed"] = registry["base_panel_allowed"].str.lower()

    duplicates = registry.loc[registry.duplicated(subset=["frequency", "feature_name"]), ["frequency", "feature_name"]]
    if not duplicates.empty:
        labels = duplicates.apply(lambda row: f"{row['frequency']}:{row['feature_name']}", axis=1).tolist()
        raise ValueError("duplicate ksv4 frequency/feature pairs: " + ", ".join(sorted(set(labels))))

    unknown_endpoints = sorted(set(registry["endpoint"]).difference(ENDPOINTS_BY_NAME))
    if unknown_endpoints:
        raise ValueError("ksv4 registry references unknown endpoints: " + ", ".join(unknown_endpoints))

    base_panel = registry[registry["base_panel_allowed"] == "yes"]
    if base_panel.empty:
        raise ValueError("ksv4 registry exposes no base-panel factors")
    if set(base_panel["frequency"]) != {"1h"}:
        raise ValueError("ksv4 base-panel factors must use frequency=1h")
    if not (base_panel["role_class"] == "coin_level").all():
        raise ValueError("ksv4 base-panel factors must be coin_level")
    if not (base_panel["design_role"] == "base_score").all():
        raise ValueError("ksv4 base-panel factors must have design_role=base_score")
    if not (base_panel["cross_section_standardization"] == "rank_to_minus1_1").all():
        raise ValueError("ksv4 base-panel factors must use rank_to_minus1_1")

    invalid_transforms = sorted(set(registry.loc[~registry["panel_transform"].isin(PANEL_TRANSFORMS), "panel_transform"]))
    if invalid_transforms:
        raise ValueError("invalid ksv4 panel_transform values: " + ", ".join(invalid_transforms))

    invalid_standardization = sorted(
        set(registry.loc[~registry["cross_section_standardization"].isin(STANDARDIZATION_POLICIES), "cross_section_standardization"])
    )
    if invalid_standardization:
        raise ValueError("invalid ksv4 standardization values: " + ", ".join(invalid_standardization))

    supported_scopes = SOURCE_SCOPES_BY_PANEL_FREQUENCY["1h"]
    unsupported_scopes = base_panel.loc[~base_panel["source_scope"].isin(supported_scopes), "feature_name"].tolist()
    if unsupported_scopes:
        raise ValueError("ksv4 base-panel factors use unsupported source scopes: " + ", ".join(unsupported_scopes))

    inferred = base_panel["source_scope"].map(infer_signal_timeframe)
    mismatched = base_panel.loc[inferred != base_panel["signal_timeframe"], "feature_name"].tolist()
    if mismatched:
        raise ValueError("ksv4 signal_timeframe must match source_scope: " + ", ".join(mismatched))

    missing_semantics = base_panel.loc[
        (base_panel["timestamp_kind"].str.strip() == "")
        | (base_panel["value_status"].str.strip() == "")
        | (base_panel["earliest_safe_decision_rule"].str.strip() == "")
    ]
    if not missing_semantics.empty:
        raise ValueError("ksv4 registry contains factors without explicit semantics")

    disallowed = base_panel.loc[base_panel["endpoint"].isin({"futures_cvd", "futures_net_pos"}), "feature_name"].tolist()
    if disallowed:
        raise ValueError("ksv4 base-panel includes conditional/side endpoints: " + ", ".join(disallowed))

    return registry.sort_values(["frequency", "signal_timeframe", "family", "feature_name"]).reset_index(drop=True)


def base_panel_registry(frequency: str = "1h") -> pd.DataFrame:
    if frequency != "1h":
        raise ValueError("unsupported ksv4 panel frequency: " + frequency)
    frame = build_factor_eligibility_registry()
    return frame.loc[frame["base_panel_allowed"] == "yes"].reset_index(drop=True)


def feature_registry_for_panel(feature_names: list[str] | tuple[str, ...], frequency: str = "1h") -> pd.DataFrame:
    requested = tuple(feature_names)
    if not requested:
        raise ValueError("feature_names must not be empty")
    frame = base_panel_registry(frequency)
    order = {feature_name: idx for idx, feature_name in enumerate(requested)}
    selected = frame.loc[frame["feature_name"].isin(order)].copy()
    missing = [feature_name for feature_name in requested if feature_name not in set(selected["feature_name"])]
    if missing:
        raise ValueError("missing ksv4 base-panel features: " + ", ".join(missing))
    selected["__feature_order"] = selected["feature_name"].map(order)
    return selected.sort_values("__feature_order").drop(columns="__feature_order").reset_index(drop=True)


__all__ = [
    "PANEL_FREQUENCIES",
    "SIGNAL_SOURCE_FREQUENCIES_BY_PANEL_FREQUENCY",
    "SOURCE_SCOPES_BY_PANEL_FREQUENCY",
    "PANEL_TRANSFORMS",
    "STANDARDIZATION_POLICIES",
    "FactorEligibilitySpec",
    "build_factor_eligibility_registry",
    "validate_factor_eligibility_registry",
    "base_panel_registry",
    "feature_registry_for_panel",
]
