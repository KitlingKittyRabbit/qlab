from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qlab.full_pipeline_simulation import (
    ObservedEffectScaleArtifacts,
    map_observed_effect_scale_to_beta_total_v1,
)


HORIZONS = ("4h", "8h", "12h", "1d")


def _beta_total_fixture() -> ObservedEffectScaleArtifacts:
    """A small hand-calculable B/C inventory for the signal-level mapping."""

    estimates: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []

    signals = {
        "direct__1h": (0.1, -0.4, 0.2, 0.3),
        # This is a signed exact alias: orientation=-1 converts its beta back
        # to the canonical direct signal before any signal-level selection.
        "direct_alias__1h": (-0.1, 0.4, -0.2, -0.3),
        # It is deliberately close to, but not exactly equal to, direct.
        "near_alias__1h": (0.09, -0.39, 0.19, 0.29),
        "negative__1h": (-0.2, -0.1, -0.15, -0.18),
        "missing__1h": (0.05, np.nan, np.nan, np.nan),
        "empty__1h": (np.nan, np.nan, np.nan, np.nan),
        "tie__1h": (0.5, -0.5, 0.2, 0.1),
    }

    for input_case, p_value in (("B", 0.001), ("C", 0.999)):
        for feature_name, betas in signals.items():
            for horizon, beta in zip(HORIZONS, betas):
                candidate_id = f"{feature_name}::{horizon}"
                is_direct_alias = feature_name == "direct_alias__1h"
                is_empty = feature_name == "empty__1h"
                is_missing = feature_name == "missing__1h"
                status = "ok"
                failure_reason: object = None
                if is_empty or (is_missing and horizon != "4h"):
                    status = "insufficient_support"
                    failure_reason = "support_rows_below_minimum"
                estimates.append(
                    {
                        "input_case": input_case,
                        "candidate_id": candidate_id,
                        "feature_name": feature_name,
                        "return_horizon": horizon,
                        "status": status,
                        "failure_reason": failure_reason,
                        "beta_obs": beta,
                        # This field deliberately differs between B and C.
                        # The formal mapping is not allowed to read it.
                        "p_value": p_value,
                    }
                )
                duplicate_group_id = (
                    f"direct-exact::{horizon}"
                    if feature_name in {"direct__1h", "direct_alias__1h"}
                    else f"singleton::{candidate_id}"
                )
                duplicates.append(
                    {
                        "input_case": input_case,
                        "candidate_id": candidate_id,
                        "return_horizon": horizon,
                        "duplicate_group_id": duplicate_group_id,
                        "canonical_candidate_id": (
                            f"direct__1h::{horizon}"
                            if feature_name in {"direct__1h", "direct_alias__1h"}
                            else candidate_id
                        ),
                        "declared_alias_of": None,
                        "canonical_orientation": -1 if is_direct_alias else 1,
                        "is_exact_duplicate": is_direct_alias,
                        "status": (
                            "exact_duplicate"
                            if feature_name in {"direct__1h", "direct_alias__1h"}
                            else "unique"
                        ),
                    }
                )

    upstream_comparison = pd.DataFrame(
        [
            {
                "candidate_id": row["candidate_id"],
                "return_horizon": row["return_horizon"],
                "signal_equal": True,
                "return_equal": True,
                "estimate_equal": True,
                "status": "equal",
            }
            for row in estimates
            if row["input_case"] == "B"
        ]
    )

    return ObservedEffectScaleArtifacts(
        candidate_estimates=pd.DataFrame(estimates),
        duplicate_mapping=pd.DataFrame(duplicates),
        distribution_summary=pd.DataFrame(),
        input_case_comparison=upstream_comparison,
        input_identity_manifest=pd.DataFrame(),
    )


def test_beta_total_mapping_is_signal_level_and_hand_calculable() -> None:
    artifacts = map_observed_effect_scale_to_beta_total_v1(_beta_total_fixture())

    b = artifacts.signal_level_scales.query("input_case == 'B'").set_index(
        "canonical_signal_id"
    )
    # direct/direct_alias is one signal; near_alias remains a separate signal.
    assert len(b) == 6
    assert b.loc["direct__1h", "selected_horizon"] == "8h"
    assert b.loc["direct__1h", "selected_signed_beta_obs"] == pytest.approx(-0.4)
    assert b.loc["direct__1h", "selected_abs_beta_total_scale"] == pytest.approx(0.4)
    assert "near_alias__1h" in b.index
    direct_alias = artifacts.canonical_signal_mapping.query(
        "input_case == 'B' and feature_name == 'direct_alias__1h'"
    ).iloc[0]
    assert direct_alias["canonical_signal_id"] == "direct__1h"
    assert bool(direct_alias["is_exact_duplicate_signal"])
    assert direct_alias["canonical_orientation"] == -1
    assert b.loc["negative__1h", "selected_signed_beta_obs"] == pytest.approx(-0.2)
    assert b.loc["missing__1h", "selected_horizon"] == "4h"
    assert b.loc["missing__1h", "excluded_horizons"] == '["8h","12h","1d"]'
    assert b.loc["empty__1h", "status"] == "no_valid_horizon"
    assert json.loads(b.loc["tie__1h", "tie_candidate_ids"]) == [
        "tie__1h::4h",
        "tie__1h::8h",
    ]

    # The five valid signal-level absolute scales are .40, .39, .20, .05,
    # and .50.  Linear quantiles therefore give .11, .39, and .46.
    absolute = artifacts.distribution_summary.query(
        "input_case == 'B' and distribution == 'absolute'"
    ).iloc[0]
    assert absolute["signal_count"] == 5
    assert absolute["p10"] == pytest.approx(0.11)
    assert absolute["p50"] == pytest.approx(0.39)
    assert absolute["p90"] == pytest.approx(0.46)

    # B and C contain different significance values, but the formal output
    # and the frozen comparison are identical because significance is ignored.
    assert set(artifacts.input_case_comparison["status"]) == {"equal"}
    assert artifacts.input_case_comparison["signal_equal"].all()
    assert artifacts.input_case_comparison["scale_equal"].all()


def test_beta_total_mapping_rejects_a_b_c_scale_mismatch() -> None:
    fixture = _beta_total_fixture()
    fixture.candidate_estimates.loc[
        (fixture.candidate_estimates["input_case"] == "C")
        & fixture.candidate_estimates["candidate_id"].isin(
            {"direct__1h::8h", "direct_alias__1h::8h"}
        ),
        "beta_obs",
    ] = [-0.41, 0.41]

    with pytest.raises(ValueError, match="frozen B/C observed beta-total mapping differs"):
        map_observed_effect_scale_to_beta_total_v1(fixture)


def test_beta_total_mapping_rejects_nonselected_b_c_horizon_mismatch() -> None:
    fixture = _beta_total_fixture()
    fixture.candidate_estimates.loc[
        (fixture.candidate_estimates["input_case"] == "C")
        & (fixture.candidate_estimates["candidate_id"] == "near_alias__1h::4h"),
        "beta_obs",
    ] = 0.10

    with pytest.raises(ValueError, match="upstream candidate/horizon row"):
        map_observed_effect_scale_to_beta_total_v1(fixture)


def test_beta_total_mapping_fails_closed_for_ambiguous_exact_group() -> None:
    fixture = _beta_total_fixture()
    mask = (
        (fixture.duplicate_mapping["input_case"] == "B")
        & (fixture.duplicate_mapping["duplicate_group_id"] == "direct-exact::8h")
    )
    fixture.duplicate_mapping.loc[mask, "canonical_candidate_id"] = "direct_alias__1h::8h"
    fixture.duplicate_mapping.loc[mask, "is_exact_duplicate"] = [True, False]

    with pytest.raises(ValueError):
        map_observed_effect_scale_to_beta_total_v1(fixture)
