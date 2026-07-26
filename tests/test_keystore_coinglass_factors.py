from __future__ import annotations

import pytest

from qlab.data.crypto import keystore_coinglass_factors as registry


def test_ksv4_registry_is_independent_1h_surface_with_native_signal_suffixes():
    frame = registry.build_factor_eligibility_registry()
    base = registry.base_panel_registry("1h")

    assert not frame.empty
    assert set(base["frequency"]) == {"1h"}
    assert set(base["signal_timeframe"]) == {"1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    assert base["feature_name"].str.contains("__", regex=False).all()
    assert base["source_scope"].str.startswith("ksv4_").all()
    assert base["earliest_safe_decision_rule"].str.contains("do not carry forward", regex=False).all()


def test_ksv4_registry_includes_only_approved_main_candidates():
    base = registry.base_panel_registry("1h")
    features = set(base["feature_name"])

    assert "futures_cvd_agg_delta1__1h" in features
    assert "spot_cvd_delta1__1h" in features
    assert "ob_pair_imbalance__1h" in features
    assert "net_pos_delta1__1h" in features
    assert "whale_index_raw__1h" in features
    assert not any(feature.startswith("futures_cvd_delta1__") for feature in features)
    assert not any(feature.startswith("futures_rsi") for feature in features)
    assert not any(feature.startswith("ob_pair_log_ratio__") for feature in features)


def test_ksv4_registry_uses_verified_ob_columns():
    base = registry.base_panel_registry("1h").set_index("feature_name")

    assert base.loc["ob_pair_imbalance__1h", "required_columns"] == "bids_usd,asks_usd"
    assert base.loc["ob_agg_imbalance__1h", "required_columns"] == "aggregated_bids_usd,aggregated_asks_usd"


def test_ksv4_registry_fails_closed_on_conditional_pair_cvd():
    frame = registry.build_factor_eligibility_registry()
    broken = frame.iloc[[0]].copy()
    broken.loc[:, "feature_name"] = "futures_cvd_delta1__1h"
    broken.loc[:, "endpoint"] = "futures_cvd"

    with pytest.raises(ValueError, match="conditional"):
        registry.validate_factor_eligibility_registry(broken)


def test_ksv4_registry_fails_closed_on_unknown_endpoint():
    frame = registry.build_factor_eligibility_registry()
    broken = frame.iloc[[0]].copy()
    broken.loc[:, "endpoint"] = "missing_endpoint"

    with pytest.raises(ValueError, match="unknown endpoints"):
        registry.validate_factor_eligibility_registry(broken)
