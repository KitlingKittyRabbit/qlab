from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
LIVE_SCRIPT_ENV = "QLAB_COINGLASS_LIVE_VALIDATION_SCRIPT"


def resolve_live_script() -> Path:
    raw_value = os.environ.get(LIVE_SCRIPT_ENV, "").strip()
    if not raw_value:
        pytest.skip(
            f"set {LIVE_SCRIPT_ENV} to run private live validation integration tests")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        pytest.skip(f"private live validation script not found: {candidate}")
    return candidate


def load_live_module():
    live_script = resolve_live_script()
    spec = importlib.util.spec_from_file_location(
        "coinglass_small_live_validation_test_module",
        live_script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubTradeExchange:

    def __init__(self, trades):
        self.has = {"fetchMyTrades": True}
        self._trades = trades

    def market(self, symbol):
        base = symbol.split("/")[0]
        return {"symbol": symbol, "id": f"{base}USDT"}

    def fetch_my_trades(self, symbol, since=None, limit=None):
        return list(self._trades)


class StubOhlcvExchange:

    def __init__(self, rows):
        self.has = {"fetchOHLCV": True}
        self._rows = rows

    def fetch_ohlcv(self, symbol, timeframe="15m", since=None, limit=None):
        return list(self._rows)


def test_expired_entry_reason_reports_lateness_after_grace_window():
    live = load_live_module()

    reason = live.expired_entry_reason(
        entry_time=pd.Timestamp("2026-05-15T12:15:00+00:00"),
        now=datetime(2026, 5, 15, 14, 40, 57, tzinfo=UTC),
        max_entry_lateness_seconds=180,
    )

    assert reason == "entry window expired; late by 2h25m57s (max 3m00s)"


def test_evaluate_candidate_skips_expired_entry_instead_of_opening(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)
    state = state_map[config.candidate_id]
    state.selector_active = True
    state.selector_reason = "selected for test"
    state.calibration_train_start = "2026-01-16T00:00:00+00:00"
    state.calibration_train_end = "2026-04-15T00:00:00+00:00"
    state.calibration_mean = 0.0
    state.calibration_std = 1.0
    state.train_ic = 0.7
    state.direction = 1

    research_frame = pd.DataFrame(
        {
            "factor_raw": [-10.0],
            "fwd_12h": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-15T00:00:00+00:00")]),
    )
    monkeypatch.setattr(live, "build_research_event_frame",
                        lambda *_args, **_kwargs: research_frame)

    open_15m = pd.Series(
        [9.744, 9.56],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-05-15T12:15:00+00:00"),
                pd.Timestamp("2026-05-15T14:30:00+00:00"),
            ]
        ),
    )

    updated_state, cycle_row, trade_rows = live.evaluate_candidate(
        config=config,
        state=state,
        candidate_lookup={live.lookup_key(config): object()},
        open_prices={"AVAX": open_15m},
        now=datetime(2026, 5, 15, 14, 40, 57, tzinfo=UTC),
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=False,
        exchange=None,
        symbol_blockers={},
        max_entry_lateness_seconds=180,
    )

    assert updated_state.current_position == 0
    assert updated_state.last_consumed_signal_bar == "2026-05-15T00:00:00+00:00"
    assert updated_state.last_action == "skip"
    assert updated_state.last_cycle_status == "missed-entry"
    assert updated_state.last_action_reason == "entry window expired; late by 2h25m57s (max 3m00s)"
    assert cycle_row["last_action"] == "skip"
    assert trade_rows == []


def test_evaluate_candidate_allows_scheduled_close_even_when_symbol_is_blocked(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)
    state = state_map[config.candidate_id]
    state.selector_active = True
    state.selector_reason = "selected for test"
    state.calibration_train_start = "2026-01-16T00:00:00+00:00"
    state.calibration_train_end = "2026-04-15T00:00:00+00:00"
    state.calibration_mean = 0.0
    state.calibration_std = 1.0
    state.train_ic = 0.7
    state.direction = 1
    state.current_position = -1
    state.position_quantity = 102.0
    state.entry_fill_price = 9.479
    state.entry_reference_price = 9.744
    state.scheduled_entry_time = "2026-05-15T12:15:00+00:00"
    state.scheduled_exit_time = "2026-05-16T00:15:00+00:00"
    state.last_signal_bar = "2026-05-15T00:00:00+00:00"
    state.last_consumed_signal_bar = "2026-05-15T00:00:00+00:00"

    research_frame = pd.DataFrame(
        {
            "factor_raw": [-10.0],
            "fwd_12h": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-15T00:00:00+00:00")]),
    )
    monkeypatch.setattr(live, "build_research_event_frame",
                        lambda *_args, **_kwargs: research_frame)

    open_15m = pd.Series(
        [9.744, 9.56, 9.52],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-05-15T12:15:00+00:00"),
                pd.Timestamp("2026-05-16T00:15:00+00:00"),
                pd.Timestamp("2026-05-16T00:30:00+00:00"),
            ]
        ),
    )

    updated_state, cycle_row, trade_rows = live.evaluate_candidate(
        config=config,
        state=state,
        candidate_lookup={live.lookup_key(config): object()},
        open_prices={"AVAX": open_15m},
        now=datetime(2026, 5, 16, 0, 30, 0, tzinfo=UTC),
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=False,
        exchange=None,
        symbol_blockers={
            "AVAX": "exchange reconcile mismatch expected_qty=-102.000000 actual_qty=-101.000000"},
        max_entry_lateness_seconds=180,
    )

    assert updated_state.current_position == 0
    assert updated_state.last_action == "close"
    assert updated_state.last_cycle_status == "ok"
    assert "scheduled exit reached" in updated_state.last_action_reason
    assert cycle_row["last_action"] == "close"
    assert len(trade_rows) == 1
    assert trade_rows[0]["action"] == "close"


def test_evaluate_candidate_live_mode_fetches_missing_entry_reference_price_from_exchange(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__liquidation__liq__imbalance__ETH__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="ETH",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)
    state = state_map[config.candidate_id]
    state.selector_active = True
    state.selector_reason = "selected for test"
    state.calibration_train_start = "2026-01-16T00:00:00+00:00"
    state.calibration_train_end = "2026-04-15T00:00:00+00:00"
    state.calibration_mean = 0.0
    state.calibration_std = 1.0
    state.train_ic = 0.7
    state.direction = 1

    research_frame = pd.DataFrame(
        {
            "factor_raw": [-10.0],
            "fwd_12h": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-15T12:00:00+00:00")]),
    )
    monkeypatch.setattr(live, "build_research_event_frame",
                        lambda *_args, **_kwargs: research_frame)
    monkeypatch.setattr(live, "normalize_order_quantity",
                        lambda _exchange, _market_symbol, quantity: quantity)
    monkeypatch.setattr(
        live,
        "place_market_order",
        lambda *_args, **_kwargs: {"id": "12345",
                                   "average": 2210.5, "filled": 0.45},
    )
    monkeypatch.setattr(
        live,
        "reconcile_open_position_with_exchange",
        lambda **kwargs: kwargs["expected_qty"],
    )

    open_15m = pd.Series(
        [2223.6],
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-16T00:00:00+00:00")]),
    )
    exchange = StubOhlcvExchange(
        rows=[
            [
                int(pd.Timestamp("2026-05-16T00:15:00+00:00").timestamp() * 1000),
                2210.5,
                2218.0,
                2201.0,
                2208.0,
                1000.0,
            ]
        ]
    )

    updated_state, cycle_row, trade_rows = live.evaluate_candidate(
        config=config,
        state=state,
        candidate_lookup={live.lookup_key(config): object()},
        open_prices={"ETH": open_15m},
        now=datetime(2026, 5, 16, 0, 17, 56, tzinfo=UTC),
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=True,
        exchange=exchange,
        symbol_blockers={},
        max_entry_lateness_seconds=180,
    )

    assert updated_state.current_position == -1
    assert updated_state.last_action == "open"
    assert updated_state.entry_reference_price == 2210.5
    assert cycle_row["last_action"] == "open"
    assert len(trade_rows) == 1
    assert trade_rows[0]["action"] == "open"


def test_recover_orphan_exchange_position_rebuilds_local_state_from_trade_time():
    live = load_live_module()
    entry_time = datetime(2026, 5, 15, 12, 15, 0, tzinfo=UTC)

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)
    state = state_map[config.candidate_id]
    state.selector_active = True
    state.last_signal_bar = "2026-05-15T00:00:00+00:00"

    open_15m = pd.Series(
        [9.744, 9.56],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-05-15T12:15:00+00:00"),
                pd.Timestamp("2026-05-15T12:30:00+00:00"),
            ]
        ),
    )
    exchange = StubTradeExchange(
        trades=[
            {
                "timestamp": int(entry_time.timestamp() * 1000),
                "side": "sell",
                "amount": 102.0,
                "info": {"symbol": "AVAXUSDT", "positionSide": "BOTH"},
            }
        ]
    )

    messages = live.recover_orphan_exchange_positions(
        configs=[config],
        state_map=state_map,
        actual_symbol_qty={"AVAX": -102.0},
        open_prices={"AVAX": open_15m},
        now=datetime(2026, 5, 15, 12, 30, 0, tzinfo=UTC),
        exchange=exchange,
        execution_mode=live.EXECUTION_MODE_LIVE,
    )

    assert len(messages) == 1
    assert "recovered orphan exchange position" in messages[0]
    assert state.current_position == -1
    assert state.position_quantity == 102.0
    assert state.scheduled_entry_time == entry_time.isoformat()
    assert state.scheduled_exit_time == (
        entry_time + pd.Timedelta("12h")).isoformat()
    assert state.entry_fill_price == 9.744
    assert state.last_action == "recovered"


def test_recover_orphan_exchange_position_forces_immediate_exit_without_trade_time():
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)

    now = datetime(2026, 5, 15, 12, 30, 0, tzinfo=UTC)
    messages = live.recover_orphan_exchange_positions(
        configs=[config],
        state_map=state_map,
        actual_symbol_qty={"AVAX": -102.0},
        open_prices={"AVAX": pd.Series(dtype=float)},
        now=now,
        exchange=StubTradeExchange(trades=[]),
        execution_mode=live.EXECUTION_MODE_LIVE,
    )

    state = state_map[config.candidate_id]
    assert len(messages) == 1
    assert "force immediate exit" in messages[0]
    assert state.current_position == -1
    assert state.scheduled_entry_time == now.isoformat()
    assert state.scheduled_exit_time == now.isoformat()
    assert state.entry_fill_price == 0.0


def test_run_once_background_refresh_schedules_async_refresh(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state = live.CandidateState(
        candidate_id=config.candidate_id, symbol=config.symbol)
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "ensure_runtime_dir", lambda: None)
    monkeypatch.setattr(live, "load_state_map", lambda _mode: {})
    monkeypatch.setattr(live, "load_candidate_configs", lambda: [config])
    monkeypatch.setattr(
        live,
        "sync_state_map",
        lambda _configs, _initial_state_map, position_usd: {
            config.candidate_id: state},
    )
    monkeypatch.setattr(
        live,
        "load_open_prices",
        lambda: {
            "AVAX": pd.Series(
                [9.744],
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-05-15T12:15:00+00:00")]),
            )
        },
    )
    monkeypatch.setattr(live, "build_candidate_lookup",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        live,
        "latest_data_timestamp",
        lambda _open_prices: pd.Timestamp("2026-05-15T12:15:00+00:00"),
    )
    monkeypatch.setattr(live, "refresh_selector_state", lambda **_kwargs: None)
    monkeypatch.setattr(
        live,
        "evaluate_candidate",
        lambda **kwargs: (
            kwargs["state"],
            {
                "timestamp": "2026-05-15T12:15:00+00:00",
                "candidate_id": config.candidate_id,
                "symbol": config.symbol,
                "last_action": "idle",
                "last_action_reason": "test",
                "cycle_status": "ok",
            },
            [],
        ),
    )
    monkeypatch.setattr(live, "append_csv_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live, "save_state_map", lambda **_kwargs: None)
    monkeypatch.setattr(
        live,
        "build_status_payload",
        lambda **_kwargs: {"generated_at": "2026-05-15T12:15:00+00:00",
                           "candidates": [], "symbol_summaries": []},
    )
    monkeypatch.setattr(live, "write_status_snapshot", lambda _payload: None)
    monkeypatch.setattr(live, "render_status_markdown", lambda _payload: "")
    monkeypatch.setattr(live, "render_terminal_dashboard",
                        lambda _payload: "ok")
    monkeypatch.setattr(
        live,
        "maybe_refresh_live_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
            "background refresh should not block run_once")),
    )

    def capture_schedule(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(live, "schedule_live_inputs_refresh", capture_schedule)

    live.run_once(
        poll_seconds=30,
        position_usd=1000.0,
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=False,
        exchange=None,
        background_refresh=True,
    )

    assert captured["allow_factor_refresh"] is True
    assert captured["auto_refresh_market_data"] is True
    assert captured["auto_refresh_selector_pool"] is True


def test_apply_frozen_live_candidate_state_uses_config_metadata():
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__liquidation__liq__imbalance__ETH__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="ETH",
        horizon="12h",
        entry_rule="下一根 open",
        execution_window_start="2026-04-16T00:20:01+00:00",
        execution_window_end="2026-05-16T00:20:01+00:00",
        calibration_train_start="2026-01-15T00:20:01+00:00",
        calibration_train_end="2026-04-14T00:20:01+00:00",
        calibration_mean=-0.3,
        calibration_std=1.4,
        train_ic=0.8,
        direction=1,
    )
    state_map = live.sync_state_map([config], {}, position_usd=1000.0)

    live.apply_frozen_live_candidate_state([config], state_map)

    state = state_map[config.candidate_id]
    assert state.selector_active is True
    assert state.selector_reason == "selected from current live candidates file"
    assert state.selector_window_start == "2026-04-16T00:20:01+00:00"
    assert state.calibration_mean == -0.3
    assert state.calibration_std == 1.4
    assert state.train_ic == 0.8
    assert state.direction == 1


def test_sync_state_map_preserves_only_open_legacy_positions():
    live = load_live_module()

    current_config = live.CandidateConfig(
        candidate_id="12h__liquidation__liq__imbalance__ETH__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="ETH",
        horizon="12h",
        entry_rule="下一根 open",
    )
    open_legacy_state = live.CandidateState(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
        current_position=-1,
        position_quantity=100.0,
    )
    flat_legacy_state = live.CandidateState(
        candidate_id="12h__liquidation__liq__imbalance__DOGE__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="DOGE",
        horizon="12h",
        entry_rule="下一根 open",
        current_position=0,
    )

    state_map = live.sync_state_map(
        [current_config],
        {
            open_legacy_state.candidate_id: open_legacy_state,
            flat_legacy_state.candidate_id: flat_legacy_state,
        },
        position_usd=1000.0,
    )

    assert current_config.candidate_id in state_map
    assert open_legacy_state.candidate_id in state_map
    assert flat_legacy_state.candidate_id not in state_map


def test_load_runtime_candidate_configs_falls_back_to_open_legacy_positions(monkeypatch):
    live = load_live_module()

    legacy_state = live.CandidateState(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
        current_position=-1,
        position_quantity=100.0,
    )

    monkeypatch.setattr(
        live,
        "load_candidate_configs",
        lambda: (_ for _ in ()).throw(
            SystemExit("Live candidates file is empty")),
    )

    configs = live.load_runtime_candidate_configs(
        {legacy_state.candidate_id: legacy_state})

    assert configs == []


def test_evaluate_candidate_selector_inactive_does_not_force_close_open_position(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
    )
    state = live.CandidateState(
        candidate_id=config.candidate_id,
        signal_timeframe=config.signal_timeframe,
        family=config.family,
        source_name=config.source_name,
        transform_name=config.transform_name,
        symbol=config.symbol,
        horizon=config.horizon,
        entry_rule=config.entry_rule,
        current_position=-1,
        position_quantity=102.0,
        entry_fill_price=9.479,
        entry_reference_price=9.744,
        scheduled_entry_time="2026-05-15T12:15:00+00:00",
        scheduled_exit_time="2026-05-16T12:15:00+00:00",
        selector_active=False,
        selector_reason="legacy open position from prior live candidate window",
        calibration_train_start="2026-01-16T00:00:00+00:00",
        calibration_train_end="2026-04-15T00:00:00+00:00",
        calibration_mean=0.0,
        calibration_std=1.0,
        train_ic=0.7,
        direction=1,
        last_signal_bar="2026-05-15T00:00:00+00:00",
        last_consumed_signal_bar="2026-05-15T00:00:00+00:00",
    )

    research_frame = pd.DataFrame(
        {
            "factor_raw": [-10.0],
            "fwd_12h": [0.01],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-15T00:00:00+00:00")]),
    )
    monkeypatch.setattr(live, "build_research_event_frame",
                        lambda *_args, **_kwargs: research_frame)

    open_15m = pd.Series(
        [9.744, 9.56],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-05-15T12:15:00+00:00"),
                pd.Timestamp("2026-05-16T00:15:00+00:00"),
            ]
        ),
    )

    updated_state, cycle_row, trade_rows = live.evaluate_candidate(
        config=config,
        state=state,
        candidate_lookup={live.lookup_key(config): object()},
        open_prices={"AVAX": open_15m},
        now=datetime(2026, 5, 16, 0, 30, 0, tzinfo=UTC),
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=False,
        exchange=None,
        symbol_blockers={},
        max_entry_lateness_seconds=180,
    )

    assert updated_state.current_position == -1
    assert updated_state.last_action == "hold"
    assert updated_state.last_cycle_status == "ok"
    assert updated_state.last_action_reason == "legacy open position from prior live candidate window"
    assert cycle_row["last_action"] == "hold"
    assert trade_rows == []


def test_run_once_live_direct_inputs_include_legacy_open_positions(monkeypatch):
    live = load_live_module()

    current_config = live.CandidateConfig(
        candidate_id="12h__liquidation__liq__imbalance__ETH__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="ETH",
        horizon="12h",
        entry_rule="下一根 open",
        calibration_train_start="2026-01-15T00:20:01+00:00",
        calibration_train_end="2026-04-14T00:20:01+00:00",
        calibration_mean=-0.3,
        calibration_std=1.4,
        train_ic=0.8,
        direction=1,
    )
    legacy_state = live.CandidateState(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
        current_position=-1,
        position_quantity=100.0,
        scheduled_entry_time="2026-05-15T12:15:00+00:00",
        scheduled_exit_time="2026-05-16T12:15:00+00:00",
        calibration_train_start="2026-01-15T00:20:01+00:00",
        calibration_train_end="2026-04-14T00:20:01+00:00",
        calibration_mean=0.2,
        calibration_std=1.2,
        train_ic=0.7,
        direction=1,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "ensure_runtime_dir", lambda: None)
    monkeypatch.setattr(
        live,
        "load_state_map",
        lambda _mode: {legacy_state.candidate_id: legacy_state},
    )
    monkeypatch.setattr(live, "load_candidate_configs",
                        lambda: [current_config])

    def capture_direct_loader(configs, exchange):
        captured["loader_ids"] = [config.candidate_id for config in configs]
        return (
            {
                "ETH": pd.Series(
                    [2210.5],
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2026-05-16T00:15:00+00:00")]),
                ),
                "AVAX": pd.Series(
                    [9.56],
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2026-05-16T00:15:00+00:00")]),
                ),
            },
            {
                live.lookup_key(current_config): object(),
                live.lookup_key(live.candidate_config_from_state(legacy_state)): object(),
            },
        )

    monkeypatch.setattr(
        live, "load_live_direct_candidate_inputs", capture_direct_loader)
    monkeypatch.setattr(
        live,
        "fetch_exchange_positions_map",
        lambda *_args, **_kwargs: {"ETH/USDT:USDT": 0.0,
                                   "AVAX/USDT:USDT": -100.0},
    )
    monkeypatch.setattr(
        live, "reconcile_state_map_with_exchange", lambda **_kwargs: [])
    monkeypatch.setattr(
        live, "recover_orphan_exchange_positions", lambda **_kwargs: [])
    monkeypatch.setattr(live, "build_symbol_summaries", lambda **_kwargs: [])

    def capture_schedule(**kwargs):
        captured["allow_factor_refresh"] = kwargs["allow_factor_refresh"]
        return True

    monkeypatch.setattr(live, "schedule_live_inputs_refresh", capture_schedule)

    def capture_evaluate(**kwargs):
        captured.setdefault("evaluated_ids", []).append(
            kwargs["config"].candidate_id)
        return (
            kwargs["state"],
            {
                "timestamp": "2026-05-16T00:17:56+00:00",
                "candidate_id": kwargs["config"].candidate_id,
                "symbol": kwargs["config"].symbol,
                "last_action": "idle",
                "last_action_reason": "test",
                "cycle_status": "ok",
            },
            [],
        )

    monkeypatch.setattr(live, "evaluate_candidate", capture_evaluate)
    monkeypatch.setattr(live, "append_csv_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live, "save_state_map", lambda **_kwargs: None)
    monkeypatch.setattr(
        live,
        "build_status_payload",
        lambda **_kwargs: {"generated_at": "2026-05-16T00:17:56+00:00",
                           "candidates": [], "symbol_summaries": []},
    )
    monkeypatch.setattr(live, "write_status_snapshot", lambda _payload: None)
    monkeypatch.setattr(live, "render_status_markdown", lambda _payload: "")
    monkeypatch.setattr(live, "render_terminal_dashboard",
                        lambda _payload: "ok")

    live.run_once(
        poll_seconds=30,
        position_usd=1000.0,
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=True,
        exchange=object(),
        direct_live_api_inputs=True,
        background_refresh=True,
    )

    assert captured["loader_ids"] == [
        current_config.candidate_id, legacy_state.candidate_id]
    assert captured["evaluated_ids"] == [
        current_config.candidate_id, legacy_state.candidate_id]
    assert captured["allow_factor_refresh"] is True


def test_run_once_live_direct_inputs_fallback_to_legacy_positions_when_candidates_missing(monkeypatch):
    live = load_live_module()

    legacy_state = live.CandidateState(
        candidate_id="12h__open_interest__oi__delta1__AVAX__12h",
        signal_timeframe="12h",
        family="open_interest",
        source_name="oi",
        transform_name="delta1",
        symbol="AVAX",
        horizon="12h",
        entry_rule="下一根 open",
        current_position=-1,
        position_quantity=100.0,
        scheduled_entry_time="2026-05-15T12:15:00+00:00",
        scheduled_exit_time="2026-05-16T12:15:00+00:00",
        calibration_train_start="2026-01-15T00:20:01+00:00",
        calibration_train_end="2026-04-14T00:20:01+00:00",
        calibration_mean=0.2,
        calibration_std=1.2,
        train_ic=0.7,
        direction=1,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "ensure_runtime_dir", lambda: None)
    monkeypatch.setattr(
        live,
        "load_state_map",
        lambda _mode: {legacy_state.candidate_id: legacy_state},
    )
    monkeypatch.setattr(
        live,
        "load_candidate_configs",
        lambda: (_ for _ in ()).throw(SystemExit(
            "Missing current selector pool live candidates file")),
    )

    def capture_direct_loader(configs, exchange):
        captured["loader_ids"] = [config.candidate_id for config in configs]
        return (
            {
                "AVAX": pd.Series(
                    [9.56],
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2026-05-16T00:15:00+00:00")]),
                ),
            },
            {
                live.lookup_key(live.candidate_config_from_state(legacy_state)): object(),
            },
        )

    monkeypatch.setattr(
        live, "load_live_direct_candidate_inputs", capture_direct_loader)
    monkeypatch.setattr(
        live,
        "fetch_exchange_positions_map",
        lambda *_args, **_kwargs: {"AVAX/USDT:USDT": -100.0},
    )
    monkeypatch.setattr(
        live, "reconcile_state_map_with_exchange", lambda **_kwargs: [])
    monkeypatch.setattr(
        live, "recover_orphan_exchange_positions", lambda **_kwargs: [])
    monkeypatch.setattr(live, "build_symbol_summaries", lambda **_kwargs: [])
    monkeypatch.setattr(
        live,
        "schedule_live_inputs_refresh",
        lambda **kwargs: captured.setdefault(
            "allow_factor_refresh", kwargs["allow_factor_refresh"]) or True,
    )

    def capture_evaluate(**kwargs):
        captured.setdefault("evaluated_ids", []).append(
            kwargs["config"].candidate_id)
        return (
            kwargs["state"],
            {
                "timestamp": "2026-05-16T00:17:56+00:00",
                "candidate_id": kwargs["config"].candidate_id,
                "symbol": kwargs["config"].symbol,
                "last_action": "idle",
                "last_action_reason": "test",
                "cycle_status": "ok",
            },
            [],
        )

    monkeypatch.setattr(live, "evaluate_candidate", capture_evaluate)
    monkeypatch.setattr(live, "append_csv_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live, "save_state_map", lambda **_kwargs: None)
    monkeypatch.setattr(
        live,
        "build_status_payload",
        lambda **_kwargs: {"generated_at": "2026-05-16T00:17:56+00:00",
                           "candidates": [], "symbol_summaries": []},
    )
    monkeypatch.setattr(live, "write_status_snapshot", lambda _payload: None)
    monkeypatch.setattr(live, "render_status_markdown", lambda _payload: "")
    monkeypatch.setattr(live, "render_terminal_dashboard",
                        lambda _payload: "ok")

    live.run_once(
        poll_seconds=30,
        position_usd=1000.0,
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=True,
        exchange=object(),
        direct_live_api_inputs=True,
        background_refresh=True,
    )

    assert captured["loader_ids"] == [legacy_state.candidate_id]
    assert captured["evaluated_ids"] == [legacy_state.candidate_id]


def test_main_prefers_process_environment_over_env_file(monkeypatch):
    live = load_live_module()

    class Args:
        once = True
        render_status = False
        skip_cache_refresh = False
        force_cache_refresh = False
        skip_selector_pool_refresh = False
        force_selector_pool_refresh = False

    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "parse_args", lambda: Args())
    monkeypatch.setenv("EXECUTION_MODE", "research-dry-run")
    monkeypatch.setenv("AUTO_REFRESH_MARKET_DATA", "false")
    monkeypatch.setenv("AUTO_REFRESH_SELECTOR_POOL", "false")
    monkeypatch.setenv("DIRECT_LIVE_API_INPUTS", "false")
    monkeypatch.setattr(live, "render_saved_status", lambda: None)

    def capture_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(live, "run_once", capture_run_once)

    live.main()

    assert captured["execution_mode"] == live.EXECUTION_MODE_DRY
    assert captured["live_mode"] is False
    assert captured["auto_refresh_market_data"] is False
    assert captured["auto_refresh_selector_pool"] is False
    assert captured["direct_live_api_inputs"] is False


def test_run_once_live_direct_inputs_use_direct_loader(monkeypatch):
    live = load_live_module()

    config = live.CandidateConfig(
        candidate_id="12h__liquidation__liq__imbalance__ETH__12h",
        signal_timeframe="12h",
        family="liquidation",
        source_name="liq",
        transform_name="imbalance",
        symbol="ETH",
        horizon="12h",
        entry_rule="下一根 open",
        calibration_train_start="2026-01-15T00:20:01+00:00",
        calibration_train_end="2026-04-14T00:20:01+00:00",
        calibration_mean=-0.3,
        calibration_std=1.4,
        train_ic=0.8,
        direction=1,
    )
    state = live.CandidateState(
        candidate_id=config.candidate_id, symbol=config.symbol)
    captured: dict[str, object] = {}

    monkeypatch.setattr(live, "ensure_runtime_dir", lambda: None)
    monkeypatch.setattr(live, "load_state_map", lambda _mode: {})
    monkeypatch.setattr(live, "load_candidate_configs", lambda: [config])
    monkeypatch.setattr(
        live,
        "sync_state_map",
        lambda _configs, _initial_state_map, position_usd: {
            config.candidate_id: state},
    )
    monkeypatch.setattr(
        live,
        "load_live_direct_candidate_inputs",
        lambda configs, exchange: (
            {
                "ETH": pd.Series(
                    [2210.5],
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2026-05-16T00:15:00+00:00")]),
                )
            },
            {live.lookup_key(config): object()},
        ),
    )

    def capture_frozen(configs, state_map):
        captured["frozen_called"] = True

    monkeypatch.setattr(
        live, "apply_frozen_live_candidate_state", capture_frozen)
    monkeypatch.setattr(
        live,
        "refresh_selector_state",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
            "live direct inputs should bypass refresh_selector_state")),
    )
    monkeypatch.setattr(live, "fetch_exchange_positions_map",
                        lambda *_args, **_kwargs: {"ETH/USDT:USDT": 0.0})
    monkeypatch.setattr(
        live, "reconcile_state_map_with_exchange", lambda **_kwargs: [])
    monkeypatch.setattr(
        live, "recover_orphan_exchange_positions", lambda **_kwargs: [])
    monkeypatch.setattr(live, "build_symbol_blockers", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "evaluate_candidate",
        lambda **kwargs: (
            kwargs["state"],
            {
                "timestamp": "2026-05-16T00:17:56+00:00",
                "candidate_id": config.candidate_id,
                "symbol": config.symbol,
                "last_action": "idle",
                "last_action_reason": "test",
                "cycle_status": "ok",
            },
            [],
        ),
    )
    monkeypatch.setattr(live, "append_csv_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live, "save_state_map", lambda **_kwargs: None)
    monkeypatch.setattr(live, "build_symbol_summaries", lambda **_kwargs: [])
    monkeypatch.setattr(
        live,
        "build_status_payload",
        lambda **_kwargs: {"generated_at": "2026-05-16T00:17:56+00:00",
                           "candidates": [], "symbol_summaries": []},
    )
    monkeypatch.setattr(live, "write_status_snapshot", lambda _payload: None)
    monkeypatch.setattr(live, "render_status_markdown", lambda _payload: "")
    monkeypatch.setattr(live, "render_terminal_dashboard",
                        lambda _payload: "ok")

    live.run_once(
        poll_seconds=30,
        position_usd=1000.0,
        execution_mode=live.EXECUTION_MODE_LIVE,
        live_mode=True,
        exchange=object(),
        direct_live_api_inputs=True,
        background_refresh=False,
    )

    assert captured["frozen_called"] is True
