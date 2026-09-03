from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qlab.data.crypto import keystore_coinglass_factors as factor_registry
from qlab.full_pipeline_simulation import (
    DecisionWindow,
    ObservedEffectCandidate,
    ObservedEffectScaleContract,
    ObservedEffectScaleInput,
    _candidate_specs_content_sha256,
    _decision_windows_content_sha256,
    _horizon_contract_content_sha256,
    _registry_content_sha256,
    _signal_identity_equal,
    _symbols_content_sha256,
    _validate_contract,
    _validate_frozen_source_manifest_identity,
    estimate_l0_l4_observed_effect_scale_v1,
)


HORIZON_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}
DECISION = pd.Timestamp("2026-01-01T00:00:00Z")


def _registry() -> pd.DataFrame:
    base = factor_registry.feature_registry_for_panel(("funding_close__1h",)).iloc[0].copy()
    rows = []
    for feature_name, column in (
        ("positive__1h", "positive"),
        ("negative__1h", "negative"),
        ("positive_alias__1h", "positive"),
    ):
        row = base.copy()
        row["feature_name"] = feature_name
        row["required_columns"] = column
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _cache_payloads() -> dict[str, dict[str, pd.DataFrame]]:
    source_index = pd.date_range(
        "2025-12-31T23:00:00Z", periods=27, freq="1h", name="ts"
    )
    payloads: dict[str, dict[str, pd.DataFrame]] = {"ksv4_1h": {}}
    for symbol, positive, negative in (("AAA", 0.0, 0.0), ("BBB", 1.0, -1.0)):
        payloads["ksv4_1h"][f"{symbol}_fr"] = pd.DataFrame(
            {"positive": positive, "negative": negative}, index=source_index
        )
    return payloads


def _minute_klines() -> dict[str, pd.DataFrame]:
    timestamps = pd.date_range(
        DECISION + pd.Timedelta(minutes=4),
        DECISION + pd.Timedelta(days=2, minutes=4),
        freq="1min",
    )
    result: dict[str, pd.DataFrame] = {}
    for symbol in ("AAA", "BBB"):
        opens = np.full(len(timestamps), 100.0)
        if symbol == "BBB":
            for decision in (DECISION, DECISION + pd.Timedelta(days=1)):
                for horizon in ("4h", "8h", "12h", "1d"):
                    target = decision + HORIZON_DELTAS[horizon] + pd.Timedelta(minutes=4)
                    position = timestamps.get_loc(target)
                    opens[position] = 120.0
        result[symbol] = pd.DataFrame({"open_time": timestamps, "open": opens})
    return result


def _specs() -> tuple[ObservedEffectCandidate, ...]:
    return (
        ObservedEffectCandidate("positive_4h", "positive__1h", "4h"),
        ObservedEffectCandidate(
            "negative_sign_alias_4h",
            "negative__1h",
            "4h",
            canonical_orientation=-1,
            declared_alias_of="positive_4h",
        ),
        ObservedEffectCandidate(
            "positive_exact_alias_4h",
            "positive_alias__1h",
            "4h",
            declared_alias_of="positive_4h",
        ),
        ObservedEffectCandidate("positive_8h", "positive__1h", "8h"),
        ObservedEffectCandidate("positive_alias_8h", "positive_alias__1h", "8h"),
        ObservedEffectCandidate("positive_12h", "positive__1h", "12h"),
        ObservedEffectCandidate("positive_alias_12h", "positive_alias__1h", "12h"),
        ObservedEffectCandidate("positive_1d", "positive__1h", "1d"),
        ObservedEffectCandidate("positive_alias_1d", "positive_alias__1h", "1d"),
        ObservedEffectCandidate("negative_8h", "negative__1h", "8h"),
        ObservedEffectCandidate("negative_12h", "negative__1h", "12h"),
        ObservedEffectCandidate("negative_1d", "negative__1h", "1d"),
    )


def _windows() -> dict[str, DecisionWindow]:
    return {horizon: DecisionWindow(start=DECISION, end=DECISION) for horizon in ("4h", "8h", "12h", "1d")}


def _fixture_contract_kwargs(
    *,
    decision_windows: dict[str, DecisionWindow] | None = None,
    candidate_specs: tuple[ObservedEffectCandidate, ...] | None = None,
    minimum_support_rows: int = 2,
    min_common_panel_rows: int = 1,
) -> dict[str, object]:
    registry = factor_registry.validate_factor_eligibility_registry(_registry())
    specs = _specs() if candidate_specs is None else candidate_specs
    symbols = ("AAA", "BBB")
    windows = _windows() if decision_windows is None else decision_windows
    return {
        "coverage_contract_identity": "test_fixture:two-asset-v1",
        "registry_feature_count": 3,
        "candidate_pair_count": 12,
        "admitted_symbol_count": 2,
        "decision_window_identity": "test_fixture:single-decision-v1",
        "registry_content_sha256": _registry_content_sha256(registry),
        "candidate_specs_sha256": _candidate_specs_content_sha256(specs),
        "admitted_symbols_sha256": _symbols_content_sha256(symbols),
        "decision_windows_sha256": _decision_windows_content_sha256(windows),
        "horizon_contract_sha256": _horizon_contract_content_sha256(
            HORIZON_DELTAS,
            execution_delay_minutes=4,
            require_complete_cross_sections=True,
            minimum_support_rows=minimum_support_rows,
            min_common_panel_rows=min_common_panel_rows,
        ),
    }


def _formal_candidate_specs(registry: pd.DataFrame) -> tuple[ObservedEffectCandidate, ...]:
    specs = []
    for row in registry.itertuples(index=False):
        signal_delta = HORIZON_DELTAS[str(row.signal_timeframe)]
        for horizon in ("4h", "8h", "12h", "1d"):
            horizon_delta = HORIZON_DELTAS[horizon]
            if signal_delta <= horizon_delta and horizon_delta % signal_delta == pd.Timedelta(0):
                specs.append(
                    ObservedEffectCandidate(
                        f"{row.feature_name}::{horizon}",
                        str(row.feature_name),
                        horizon,
                    )
                )
    return tuple(specs)


def _reality_contract(
    registry: pd.DataFrame,
    specs: tuple[ObservedEffectCandidate, ...],
    symbols: tuple[str, ...],
    windows: dict[str, DecisionWindow],
) -> ObservedEffectScaleContract:
    windows_sha256 = _decision_windows_content_sha256(windows)
    candidate_specs_sha256 = _candidate_specs_content_sha256(specs)
    horizon_contract_sha256 = _horizon_contract_content_sha256(
        HORIZON_DELTAS,
        execution_delay_minutes=4,
        require_complete_cross_sections=True,
        minimum_support_rows=2,
        min_common_panel_rows=1,
    )
    return ObservedEffectScaleContract(
        admitted_symbols=symbols,
        candidate_specs=specs,
        decision_windows=windows,
        horizon_deltas=HORIZON_DELTAS,
        minimum_support_rows=2,
        min_common_panel_rows=1,
        registry_identity="reality-effect-registry-v1",
        candidate_set_identity=f"reality_effect_scale_v1:{candidate_specs_sha256}",
        coverage_contract_identity=f"reality_effect_scale_v1:{horizon_contract_sha256}",
        registry_feature_count=len(registry),
        candidate_pair_count=len(specs),
        admitted_symbol_count=len(symbols),
        decision_window_identity=f"reality_effect_scale_v1:{windows_sha256}",
        registry_content_sha256=_registry_content_sha256(registry.reset_index(drop=True)),
        candidate_specs_sha256=candidate_specs_sha256,
        admitted_symbols_sha256=_symbols_content_sha256(symbols),
        decision_windows_sha256=windows_sha256,
        horizon_contract_sha256=horizon_contract_sha256,
    )


def _production_windows() -> dict[str, DecisionWindow]:
    return {
        "4h": DecisionWindow(
            start=pd.Timestamp("2024-12-21T12:00:00Z"),
            end=pd.Timestamp("2026-04-30T08:00:00Z"),
        ),
        "8h": DecisionWindow(
            start=pd.Timestamp("2024-12-21T16:00:00Z"),
            end=pd.Timestamp("2026-04-30T08:00:00Z"),
        ),
        "12h": DecisionWindow(
            start=pd.Timestamp("2024-12-21T12:00:00Z"),
            end=pd.Timestamp("2026-04-30T00:00:00Z"),
        ),
        "1d": DecisionWindow(
            start=pd.Timestamp("2024-12-22T00:00:00Z"),
            end=pd.Timestamp("2026-04-30T00:00:00Z"),
        ),
    }


def _production_input(input_case: str) -> ObservedEffectScaleInput:
    source_manifest_sha256 = ("a" if input_case == "B" else "b") * 64
    provisional = ObservedEffectScaleInput(
        _cache_payloads(),
        _minute_klines(),
        "provisional-production-identity",
        source_manifest_sha256,
    )
    identity = (
        f"reality_effect_scale_v1:{input_case}:{source_manifest_sha256}:"
        f"{provisional.cache_sha256}:{provisional.minute_klines_sha256}"
    )
    return replace(provisional, input_identity=identity)


def _run(
    *,
    minimum_support_rows: int = 2,
    decision_windows=None,
    candidate_specs=None,
    payloads_b=None,
    payloads_c=None,
):
    windows = _windows() if decision_windows is None else decision_windows
    specs = _specs() if candidate_specs is None else candidate_specs
    contract = ObservedEffectScaleContract(
        admitted_symbols=("AAA", "BBB"),
        candidate_specs=specs,
        decision_windows=windows,
        horizon_deltas=HORIZON_DELTAS,
        minimum_support_rows=minimum_support_rows,
            min_common_panel_rows=1,
            registry_identity="synthetic-registry-v1",
            candidate_set_identity="synthetic-unfiltered-grid-v1",
            **_fixture_contract_kwargs(
                decision_windows=windows,
                candidate_specs=specs,
                minimum_support_rows=minimum_support_rows,
            ),
        )
    return estimate_l0_l4_observed_effect_scale_v1(
        registry_frame=_registry(),
        contract=contract,
        input_cases={
            "B": ObservedEffectScaleInput(
                _cache_payloads() if payloads_b is None else payloads_b,
                _minute_klines(),
                "cache-B-v1",
            ),
            "C": ObservedEffectScaleInput(
                _cache_payloads() if payloads_c is None else payloads_c,
                _minute_klines(),
                "cache-C-v1",
            ),
        },
    )


def test_observed_effect_scale_uses_formal_rank_and_four_minute_executable_return():
    artifacts = _run()
    estimates = artifacts.candidate_estimates.query("input_case == 'B'").set_index("candidate_id")

    # Hand calculation at the one frozen decision: Z=(-1, 1), Y=(0, .2).
    # alpha=.1, beta=.1, population sigma_Y=.1, delta=1.
    for candidate_id in ("positive_4h", "positive_8h", "positive_12h", "positive_1d"):
        row = estimates.loc[candidate_id]
        assert row["status"] == "ok"
        assert row["support_rows"] == 2
        assert row["support_asset_count"] == 2
        assert row["support_decision_count"] == 1
        assert row["alpha_obs"] == pytest.approx(0.1)
        assert row["beta_obs"] == pytest.approx(0.1)
        assert row["sigma_y"] == pytest.approx(0.1)
        assert row["delta_obs"] == pytest.approx(1.0)

    negative = estimates.loc["negative_sign_alias_4h"]
    assert negative["beta_obs"] == pytest.approx(-0.1)
    assert negative["delta_obs"] == pytest.approx(-1.0)

    duplicates = artifacts.duplicate_mapping.query("input_case == 'B'").set_index("candidate_id")
    assert duplicates.loc["positive_4h", "canonical_candidate_id"] == "positive_4h"
    assert duplicates.loc["negative_sign_alias_4h", "canonical_candidate_id"] == "positive_4h"
    assert bool(duplicates.loc["negative_sign_alias_4h", "is_exact_duplicate"])
    assert duplicates.loc["positive_exact_alias_4h", "canonical_candidate_id"] == "positive_4h"
    assert bool(duplicates.loc["positive_exact_alias_4h", "is_exact_duplicate"])

    signed_4h = artifacts.distribution_summary.query(
        "input_case == 'B' and return_horizon == '4h' and distribution == 'signed'"
    ).iloc[0]
    assert signed_4h["candidate_count"] == 1
    assert signed_4h[["p10", "p50", "p90"]].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert set(artifacts.input_case_comparison["status"]) == {"equal"}


def test_observed_effect_scale_keeps_two_continuous_native_decisions():
    windows = {
        horizon: DecisionWindow(start=DECISION, end=DECISION + pd.Timedelta(days=1))
        for horizon in ("4h", "8h", "12h", "1d")
    }
    artifacts = _run(decision_windows=windows)
    estimates = artifacts.candidate_estimates.query("input_case == 'B'").set_index("candidate_id")
    assert estimates.loc["positive_4h", "support_decision_count"] >= 2
    assert estimates.loc["positive_1d", "support_decision_count"] == 2


def test_observed_effect_scale_requires_two_identical_frozen_input_cases():
    b_payloads = _cache_payloads()
    c_klines = _minute_klines()
    c_klines["BBB"].loc[
        c_klines["BBB"]["open_time"] == DECISION + pd.Timedelta(hours=4, minutes=4), "open"
    ] = 130.0

    with pytest.raises(ValueError, match="frozen B/C inputs do not produce identical"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(b_payloads, _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(deepcopy(b_payloads), c_klines, "cache-C-v1"),
            },
        )


def test_public_entry_rejects_reusing_one_frozen_input_case_for_both_labels():
    shared = ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1")
    with pytest.raises(ValueError, match="distinct objects"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={"B": shared, "C": shared},
        )


def test_public_entry_rejects_selected_candidate_subset_and_partial_cross_sections():
    subset_specs = (_specs()[0],)
    subset_windows = {"4h": _windows()["4h"]}
    with pytest.raises(ValueError, match="complete unfiltered registry grid"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=(_specs()[0],),
                decision_windows={"4h": _windows()["4h"]},
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                    **_fixture_contract_kwargs(
                        decision_windows=subset_windows,
                        candidate_specs=subset_specs,
                    ),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )

    with pytest.raises(ValueError, match="complete cross-sections"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
                require_complete_cross_sections=False,
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


def test_public_entry_rejects_duplicate_feature_horizon_identity():
    duplicate_specs = list(_specs())
    duplicate_specs.append(
        ObservedEffectCandidate("duplicate_positive_4h", "positive__1h", "4h")
    )
    with pytest.raises(ValueError, match="feature/horizon identities must be unique"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=tuple(duplicate_specs),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                    **_fixture_contract_kwargs(candidate_specs=tuple(duplicate_specs)),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


def test_production_contract_cannot_silently_accept_a_synthetic_coverage_shape():
    production_kwargs = _fixture_contract_kwargs()
    production_kwargs.update(
        {
            "coverage_contract_identity": (
                "reality_effect_scale_v1:" + production_kwargs["horizon_contract_sha256"]
            ),
            "registry_feature_count": 68,
            "candidate_pair_count": 159,
            "admitted_symbol_count": 20,
            "decision_window_identity": "reality_effect_scale_v1:full-frozen-window",
        }
    )
    with pytest.raises(ValueError, match="formal qlab registry"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **production_kwargs,
            ),
            input_cases={
                "B": _production_input("B"),
                "C": _production_input("C"),
            },
        )


@pytest.mark.parametrize("identity_case", ("registry", "symbols"))
def test_production_contract_rejects_same_size_forged_registry_or_universe(identity_case):
    registry = factor_registry.base_panel_registry("1h")
    symbols = tuple(
        (
            f"FORGED{number:02d}"
            if identity_case == "symbols"
            else symbol
        )
        for number, symbol in enumerate(
            ("ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
             "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP")
        )
    )
    if identity_case == "registry":
        registry = registry.copy()
        registry.loc[0, "feature_name"] = "forged_feature__1h"
    specs = _formal_candidate_specs(registry)
    windows = {
        horizon: DecisionWindow(start=DECISION, end=DECISION)
        for horizon in ("4h", "8h", "12h", "1d")
    }
    contract = _reality_contract(registry, specs, symbols, windows)
    with pytest.raises(ValueError, match=("formal qlab registry" if identity_case == "registry" else "canonical universe")):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=registry,
            contract=contract,
            input_cases={
                "B": _production_input("B"),
                "C": _production_input("C"),
            },
        )


def test_production_contract_rejects_changed_window_after_contract_freeze():
    registry = factor_registry.base_panel_registry("1h")
    symbols = (
        "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
        "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
    )
    specs = _formal_candidate_specs(registry)
    windows = _production_windows()
    contract = _reality_contract(registry, specs, symbols, windows)
    windows["4h"] = DecisionWindow(
        start=DECISION + pd.Timedelta(hours=4),
        end=DECISION + pd.Timedelta(hours=4),
    )
    with pytest.raises(ValueError, match="contract content changed"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=registry,
            contract=contract,
            input_cases={
                "B": _production_input("B"),
                "C": _production_input("C"),
            },
        )


def test_production_identity_binds_registry_order_windows_and_candidate_metadata():
    registry = factor_registry.base_panel_registry("1h")
    symbols = (
        "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
        "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
    )
    specs = _formal_candidate_specs(registry)
    contract = _reality_contract(registry, specs, symbols, _production_windows())

    assert len(_validate_contract(contract, registry)) == 159

    with pytest.raises(ValueError, match="registry content/order"):
        _validate_contract(contract, registry.iloc[::-1].reset_index(drop=True))

    renamed = (replace(specs[0], candidate_id="renamed-candidate::4h"),) + specs[1:]
    renamed_contract = replace(
        contract,
        candidate_specs=renamed,
        candidate_specs_sha256=_candidate_specs_content_sha256(renamed),
    )
    with pytest.raises(ValueError, match="candidate_set_identity"):
        _validate_contract(renamed_contract, registry)

    changed_windows = dict(_production_windows())
    changed_windows["4h"] = DecisionWindow(
        start=changed_windows["4h"].start + pd.Timedelta(minutes=1),
        end=changed_windows["4h"].end,
    )
    changed_window_contract = replace(contract, decision_windows=changed_windows)
    with pytest.raises(ValueError, match="contract content changed"):
        _validate_contract(changed_window_contract, registry)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (("minimum_support_rows", 3), ("min_common_panel_rows", 2)),
)
def test_production_coverage_identity_binds_support_thresholds(
    changed_field, changed_value
):
    registry = factor_registry.base_panel_registry("1h")
    symbols = (
        "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
        "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
    )
    specs = _formal_candidate_specs(registry)
    contract = _reality_contract(registry, specs, symbols, _production_windows())
    threshold_values = {
        "minimum_support_rows": contract.minimum_support_rows,
        "min_common_panel_rows": contract.min_common_panel_rows,
    }
    threshold_values[changed_field] = changed_value
    changed_horizon_sha256 = _horizon_contract_content_sha256(
        HORIZON_DELTAS,
        execution_delay_minutes=4,
        require_complete_cross_sections=True,
        minimum_support_rows=threshold_values["minimum_support_rows"],
        min_common_panel_rows=threshold_values["min_common_panel_rows"],
    )
    changed_threshold_contract = replace(
        contract,
        **{changed_field: changed_value},
        horizon_contract_sha256=changed_horizon_sha256,
    )
    with pytest.raises(ValueError, match="coverage_contract_identity must bind"):
        _validate_contract(changed_threshold_contract, registry)


@pytest.mark.parametrize("whitespace", ("leading", "trailing"))
def test_production_coverage_identity_rejects_surrounding_whitespace(whitespace):
    registry = factor_registry.base_panel_registry("1h")
    symbols = (
        "ADA", "APT", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
        "FET", "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "TRX", "UNI", "XRP",
    )
    specs = _formal_candidate_specs(registry)
    contract = _reality_contract(registry, specs, symbols, _production_windows())
    identity = contract.coverage_contract_identity
    malformed_identity = f" {identity}" if whitespace == "leading" else f"{identity} "
    with pytest.raises(ValueError, match="must not contain surrounding whitespace"):
        _validate_contract(
            replace(contract, coverage_contract_identity=malformed_identity), registry
        )


def test_production_input_identity_binds_external_manifest_and_content_digests():
    source_manifest_sha256 = "a" * 64
    coverage_identity = "reality_effect_scale_v1:" + "0" * 64
    input_data = ObservedEffectScaleInput(
        _cache_payloads(),
        _minute_klines(),
        "wrong-production-input-identity",
        source_manifest_sha256,
    )
    with pytest.raises(ValueError, match="must bind source manifest"):
        _validate_frozen_source_manifest_identity(
            "B", input_data, coverage_identity
        )

    valid_identity = (
        f"reality_effect_scale_v1:B:{source_manifest_sha256}:"
        f"{input_data.cache_sha256}:{input_data.minute_klines_sha256}"
    )
    valid_input = replace(input_data, input_identity=valid_identity)
    _validate_frozen_source_manifest_identity(
        "B", valid_input, coverage_identity
    )


def test_input_identity_manifest_contains_content_digests_and_mutation_fails_closed():
    artifacts = _run()
    identity_manifest = artifacts.input_identity_manifest
    assert list(identity_manifest.columns) == [
        "input_case",
        "input_identity",
        "source_manifest_sha256",
        "cache_sha256",
        "minute_klines_sha256",
        "coverage_contract_identity",
        "registry_identity",
        "registry_content_sha256",
        "candidate_set_identity",
        "candidate_specs_sha256",
        "admitted_symbol_count",
        "admitted_symbols_sha256",
        "decision_window_identity",
        "decision_windows_sha256",
        "horizon_contract_sha256",
        "execution_delay_minutes",
        "decision_coverage_policy",
    ]
    assert set(identity_manifest["input_case"]) == {"B", "C"}
    assert identity_manifest["cache_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert identity_manifest["minute_klines_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for column in (
        "registry_content_sha256",
        "candidate_specs_sha256",
        "admitted_symbols_sha256",
        "decision_windows_sha256",
        "horizon_contract_sha256",
    ):
        assert identity_manifest[column].str.fullmatch(r"[0-9a-f]{64}").all()

    payloads = _cache_payloads()
    b_input = ObservedEffectScaleInput(payloads, _minute_klines(), "cache-B-v1")
    payloads["ksv4_1h"]["AAA_fr"].loc[:, "positive"] = 9.0
    with pytest.raises(ValueError, match="frozen B cache input changed"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": b_input,
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


@pytest.mark.parametrize("input_case", ("B", "C"))
def test_postconstruction_minute_price_mutation_fails_closed(input_case):
    b_input = ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1")
    c_input = ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1")
    inputs = {"B": b_input, "C": c_input}
    inputs[input_case].minute_klines_by_symbol["BBB"].loc[0, "open"] = 101.0

    with pytest.raises(ValueError, match=f"frozen {input_case} minute price input changed"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases=inputs,
        )


def test_declared_sign_alias_must_be_a_canonical_exact_duplicate():
    invalid_specs = list(_specs())
    invalid_specs[1] = ObservedEffectCandidate(
        "negative_sign_alias_4h",
        "negative__1h",
        "4h",
        canonical_orientation=1,
        declared_alias_of="positive_4h",
    )
    with pytest.raises(ValueError, match="declared sign alias is not an exact canonical signal"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=tuple(invalid_specs),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                    **_fixture_contract_kwargs(candidate_specs=tuple(invalid_specs)),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


def test_near_alias_is_not_collapsed_by_the_formal_panel_path():
    source_index = pd.date_range(
        "2025-12-31T23:00:00Z", periods=27, freq="1h", name="ts"
    )
    base = factor_registry.feature_registry_for_panel(("funding_close__1h",)).iloc[0].copy()
    registry_rows = []
    for feature_name, column in (("near_a__1h", "a"), ("near_b__1h", "b")):
        row = base.copy()
        row["feature_name"] = feature_name
        row["required_columns"] = column
        registry_rows.append(row)
    registry = pd.DataFrame(registry_rows).reset_index(drop=True)
    payloads = {"ksv4_1h": {}}
    for symbol in ("AAA", "BBB"):
        b_values = np.ones(len(source_index))
        if symbol == "BBB":
            b_values[0] = np.nextafter(1.0, 2.0)
        payloads["ksv4_1h"][f"{symbol}_fr"] = pd.DataFrame(
            {"a": np.ones(len(source_index)), "b": b_values}, index=source_index
        )
    specs = _formal_candidate_specs(registry)
    windows = _windows()
    validated = factor_registry.validate_factor_eligibility_registry(registry)
    contract = ObservedEffectScaleContract(
        admitted_symbols=("AAA", "BBB"),
        candidate_specs=specs,
        decision_windows=windows,
        horizon_deltas=HORIZON_DELTAS,
        minimum_support_rows=2,
        min_common_panel_rows=1,
        registry_identity="synthetic-near-alias-registry-v1",
        candidate_set_identity="synthetic-near-alias-candidates-v1",
        coverage_contract_identity="test_fixture:near-alias-v1",
        registry_feature_count=2,
        candidate_pair_count=8,
        admitted_symbol_count=2,
        decision_window_identity="test_fixture:near-alias-window-v1",
        registry_content_sha256=_registry_content_sha256(validated),
        candidate_specs_sha256=_candidate_specs_content_sha256(specs),
        admitted_symbols_sha256=_symbols_content_sha256(("AAA", "BBB")),
        decision_windows_sha256=_decision_windows_content_sha256(windows),
        horizon_contract_sha256=_horizon_contract_content_sha256(
            HORIZON_DELTAS,
            execution_delay_minutes=4,
            require_complete_cross_sections=True,
            minimum_support_rows=2,
            min_common_panel_rows=1,
        ),
    )
    artifacts = estimate_l0_l4_observed_effect_scale_v1(
        registry_frame=registry,
        contract=contract,
        input_cases={
            "B": ObservedEffectScaleInput(payloads, _minute_klines(), "near-B-v1"),
            "C": ObservedEffectScaleInput(
                deepcopy(payloads), _minute_klines(), "near-C-v1"
            ),
        },
    )
    mapping = artifacts.duplicate_mapping.query(
        "input_case == 'B' and return_horizon == '4h'"
    ).set_index("candidate_id")
    assert mapping.loc["near_a__1h::4h", "status"] == "unique"
    assert mapping.loc["near_b__1h::4h", "status"] == "unique"
    assert mapping.loc["near_a__1h::4h", "duplicate_group_id"] != mapping.loc[
        "near_b__1h::4h", "duplicate_group_id"
    ]


def test_missing_admitted_price_symbol_fails_closed():
    price = _minute_klines()
    del price["BBB"]
    with pytest.raises(ValueError, match="minute price input missing admitted symbol: BBB"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), price, "cache-B-v1"),
                "C": ObservedEffectScaleInput(
                    _cache_payloads(), _minute_klines(), "cache-C-v1"
                ),
            },
        )


def test_raw_duplicate_or_infinite_required_values_fail_closed_before_normalization():
    duplicated = _cache_payloads()
    duplicated["ksv4_1h"]["AAA_fr"] = pd.concat(
        [duplicated["ksv4_1h"]["AAA_fr"], duplicated["ksv4_1h"]["AAA_fr"].iloc[[-1]]]
    )
    with pytest.raises(ValueError, match="duplicate source timestamps"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(duplicated, _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )

    price_infinite = _minute_klines()
    price_infinite["AAA"].loc[0, "open"] = np.inf
    with pytest.raises(ValueError, match="minute price input contains non-finite"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), price_infinite, "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )

    infinite = _cache_payloads()
    infinite["ksv4_1h"]["AAA_fr"].loc[:, "positive"] = np.inf
    with pytest.raises(ValueError, match="invalid required value"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(infinite, _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


@pytest.mark.parametrize("invalid_open", (0.0, -1.0))
def test_non_positive_minute_prices_fail_closed(invalid_open):
    price = _minute_klines()
    price["AAA"].loc[0, "open"] = invalid_open
    with pytest.raises(ValueError, match="minute price input contains non-positive"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), price, "cache-B-v1"),
                "C": ObservedEffectScaleInput(
                    _cache_payloads(), _minute_klines(), "cache-C-v1"
                ),
            },
        )


def test_missing_decision_is_jointly_removed_for_one_candidate_and_horizon():
    missing = _cache_payloads()
    # The fixture is a bar_start source: DECISION consumes the prior 1h row.
    missing["ksv4_1h"]["AAA_fr"].loc[
        DECISION - pd.Timedelta(hours=1), "positive"
    ] = np.nan
    specs = tuple(
        replace(spec, canonical_orientation=1, declared_alias_of=None)
        for spec in _specs()
    )
    artifacts = _run(
        decision_windows={
            "4h": DecisionWindow(start=DECISION, end=DECISION + pd.Timedelta(hours=4)),
            "8h": DecisionWindow(start=DECISION, end=DECISION),
            "12h": DecisionWindow(start=DECISION, end=DECISION),
            "1d": DecisionWindow(start=DECISION, end=DECISION),
        },
        candidate_specs=specs,
        payloads_b=missing,
        payloads_c=deepcopy(missing),
    )
    estimates = artifacts.candidate_estimates.query("input_case == 'B'").set_index("candidate_id")
    positive = estimates.loc["positive_4h"]
    assert positive["status"] == "ok"
    assert positive["support_decision_count"] == 1
    assert positive["support_rows"] == 2
    assert positive["beta_obs"] == pytest.approx(0.0)
    assert estimates.loc["negative_sign_alias_4h", "status"] == "ok"
    assert set(artifacts.input_case_comparison["status"]) == {"equal"}


def test_all_nan_decisions_are_recorded_without_allowing_a_partial_cross_section():
    missing = _cache_payloads()
    for symbol in ("AAA", "BBB"):
        missing["ksv4_1h"][f"{symbol}_fr"].loc[
            DECISION - pd.Timedelta(hours=1), "positive"
        ] = np.nan

    specs = tuple(
        replace(spec, canonical_orientation=1, declared_alias_of=None)
        for spec in _specs()
    )
    artifacts = _run(
        candidate_specs=specs,
        payloads_b=missing,
        payloads_c=deepcopy(missing),
    )
    estimates = artifacts.candidate_estimates.query("input_case == 'B'").set_index("candidate_id")
    assert estimates.loc["positive_4h", "status"] == "insufficient_support"
    assert estimates.loc["positive_4h", "failure_reason"] == "no_complete_cross_section_decisions"
    assert estimates.loc["positive_4h", "support_decision_count"] == 0
    assert estimates.loc["negative_sign_alias_4h", "status"] == "ok"


def test_common_slope_uses_the_entire_twenty_asset_cross_section():
    symbols = tuple(f"S{number:02d}" for number in range(20))
    source_index = pd.date_range("2025-12-31T23:00:00Z", periods=27, freq="1h")
    payloads = {"ksv4_1h": {}}
    klines = {}
    for number, symbol in enumerate(symbols):
        payloads["ksv4_1h"][f"{symbol}_fr"] = pd.DataFrame(
            {"positive": float(number), "negative": float(-number)}, index=source_index
        )
        execution_times = [DECISION + pd.Timedelta(minutes=4)] + [
            DECISION + HORIZON_DELTAS[horizon] + pd.Timedelta(minutes=4)
            for horizon in ("4h", "8h", "12h", "1d")
        ]
        klines[symbol] = pd.DataFrame(
            {
                "open_time": execution_times,
                "open": [
                    100.0,
                    *[100.0 * (1.0 + 0.001 * number * (offset + 1)) for offset in range(4)],
                ],
            }
        )
    specs = tuple(
        ObservedEffectCandidate(f"positive_{horizon}", "positive__1h", horizon)
        for horizon in ("4h", "8h", "12h", "1d")
    )
    windows = {
        horizon: DecisionWindow(start=DECISION, end=DECISION)
        for horizon in ("4h", "8h", "12h", "1d")
    }
    registry = factor_registry.validate_factor_eligibility_registry(
        _registry().iloc[[0]].reset_index(drop=True)
    )
    contract = ObservedEffectScaleContract(
        admitted_symbols=symbols,
        candidate_specs=specs,
        decision_windows=windows,
        horizon_deltas=HORIZON_DELTAS,
        minimum_support_rows=20,
        min_common_panel_rows=1,
        registry_identity="synthetic-registry-v1",
        candidate_set_identity="synthetic-unfiltered-grid-v1",
        coverage_contract_identity="test_fixture:twenty-asset-v1",
        registry_feature_count=1,
        candidate_pair_count=4,
        admitted_symbol_count=20,
        decision_window_identity="test_fixture:single-decision-20-assets-v1",
        registry_content_sha256=_registry_content_sha256(registry),
        candidate_specs_sha256=_candidate_specs_content_sha256(specs),
        admitted_symbols_sha256=_symbols_content_sha256(symbols),
        decision_windows_sha256=_decision_windows_content_sha256(windows),
        horizon_contract_sha256=_horizon_contract_content_sha256(
            HORIZON_DELTAS,
            execution_delay_minutes=4,
            require_complete_cross_sections=True,
            minimum_support_rows=20,
            min_common_panel_rows=1,
        ),
    )
    artifacts = estimate_l0_l4_observed_effect_scale_v1(
        registry_frame=_registry().iloc[[0]].reset_index(drop=True),
        contract=contract,
        input_cases={
            "B": ObservedEffectScaleInput(payloads, klines, "cache-B-v1"),
            "C": ObservedEffectScaleInput(deepcopy(payloads), deepcopy(klines), "cache-C-v1"),
        },
    )
    estimate = artifacts.candidate_estimates.query("input_case == 'B'").iloc[0]
    assert estimate["status"] == "ok"
    assert estimate["support_rows"] == 20
    assert estimate["support_asset_count"] == 20
    assert estimate["beta_obs"] > 0.0


def test_observed_effect_scale_preserves_unfiltered_insufficient_support_receipt():
    artifacts = _run(minimum_support_rows=3)

    assert set(artifacts.candidate_estimates["candidate_id"]) == {
        "positive_4h",
        "negative_sign_alias_4h",
        "positive_exact_alias_4h",
        "positive_8h",
        "positive_alias_8h",
        "positive_12h",
        "positive_alias_12h",
        "positive_1d",
        "positive_alias_1d",
        "negative_8h",
        "negative_12h",
        "negative_1d",
    }
    assert set(artifacts.candidate_estimates["status"]) == {"insufficient_support"}
    assert set(artifacts.candidate_estimates["failure_reason"]) == {"support_rows_below_minimum"}
    assert set(artifacts.distribution_summary["status"]) == {"no_eligible_nonduplicate_candidates"}


def test_observed_effect_scale_fails_closed_for_misaligned_decision_window():
    windows = _windows()
    windows["4h"] = DecisionWindow(
        start=DECISION + pd.Timedelta(hours=1),
        end=DECISION + pd.Timedelta(hours=1),
    )
    with pytest.raises(ValueError, match="UTC phase"):
        estimate_l0_l4_observed_effect_scale_v1(
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=windows,
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(decision_windows=windows),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
        )


def test_exact_duplicate_requires_canonical_float64_equality_not_tolerance():
    index = pd.MultiIndex.from_product(
        [[DECISION], ["AAA", "BBB"]], names=["decision_ts", "symbol"]
    )
    exact = pd.Series([-1.0, 1.0], index=index)
    almost = pd.Series([-1.0, np.nextafter(1.0, 2.0)], index=index)

    assert _signal_identity_equal(exact, exact)
    assert not _signal_identity_equal(exact, almost)


def test_public_entry_has_no_selection_or_significance_input():
    parameter_names = set(inspect.signature(estimate_l0_l4_observed_effect_scale_v1).parameters)
    forbidden = {"p_values", "significance", "l2_gate", "l3_selection", "l4_selection", "discovery_results"}
    assert parameter_names.isdisjoint(forbidden)
    with pytest.raises(TypeError):
        estimate_l0_l4_observed_effect_scale_v1(  # type: ignore[call-arg]
            registry_frame=_registry(),
            contract=ObservedEffectScaleContract(
                admitted_symbols=("AAA", "BBB"),
                candidate_specs=_specs(),
                decision_windows=_windows(),
                horizon_deltas=HORIZON_DELTAS,
                minimum_support_rows=2,
                min_common_panel_rows=1,
                registry_identity="synthetic-registry-v1",
                candidate_set_identity="synthetic-unfiltered-grid-v1",
                **_fixture_contract_kwargs(),
            ),
            input_cases={
                "B": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-B-v1"),
                "C": ObservedEffectScaleInput(_cache_payloads(), _minute_klines(), "cache-C-v1"),
            },
            l2_gate=True,
        )
