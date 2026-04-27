from __future__ import annotations

import importlib.util
import os
import sys
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LIVE_SCRIPT_ENV = "QLAB_LIVE_TIMING_SCRIPT"


def resolve_live_script() -> Path:
    raw_value = os.environ.get(LIVE_SCRIPT_ENV, "").strip()
    if not raw_value:
        pytest.skip(f"set {LIVE_SCRIPT_ENV} to run live timing integration tests")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        pytest.skip(f"live timing integration script not found: {candidate}")
    return candidate


def load_live_module():
    live_script = resolve_live_script()
    spec = importlib.util.spec_from_file_location("live_top_pos_module", live_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubExchange:

    def __init__(self, positions, trades):
        self.has = {"fetchPositions": True, "fetchMyTrades": True}
        self._positions = positions
        self._trades = trades

    def market(self, symbol):
        return {"symbol": symbol, "id": "BTCUSDT"}

    def fetch_positions(self, symbols):
        return list(self._positions)

    def fetch_my_trades(self, symbol, since=None, limit=None):
        return list(self._trades)


class StubReduceOnlyRejectExchange:

    def __init__(self):
        self.calls = []

    def market(self, symbol):
        return {"symbol": symbol, "id": "BTCUSDT"}

    def create_order(self, symbol, order_type, side, quantity, params=None):
        payload = dict(params or {})
        self.calls.append(payload)
        if payload.get("reduceOnly"):
            raise RuntimeError(
                'binanceusdm {"code":-1106,"msg":"Parameter \'reduceonly\' sent when not required."}'
            )
        return {"average": 78000.0, "status": "closed"}


def test_infer_position_entry_time_from_trades_uses_real_open_time():
    live = load_live_module()
    entry_time = datetime(2026, 4, 21, 6, 33, 56, tzinfo=UTC)
    trades = [
        {
            "timestamp": int(entry_time.timestamp() * 1000),
            "side": "buy",
            "amount": 0.001,
            "info": {"symbol": "BTCUSDT", "positionSide": "BOTH"},
        }
    ]

    inferred = live.infer_position_entry_time_from_trades(
        trades=trades,
        position_direction=1,
        position_quantity=0.001,
    )

    assert inferred == entry_time


def test_reconcile_position_state_rebuilds_missing_schedule_from_trade_time():
    live = load_live_module()
    entry_time = datetime(2026, 4, 21, 6, 33, 56, tzinfo=UTC)
    exchange = StubExchange(
        positions=[
            {
                "symbol": live.EXCHANGE_SYMBOL,
                "positionAmt": "0.001",
                "info": {
                    "symbol": live.EXCHANGE_SYMBOL,
                    "positionAmt": "0.001",
                    "positionSide": "BOTH",
                },
            }
        ],
        trades=[
            {
                "timestamp": int(entry_time.timestamp() * 1000),
                "side": "buy",
                "amount": 0.001,
                "info": {"symbol": live.EXCHANGE_SYMBOL, "positionSide": "BOTH"},
            }
        ],
    )
    state = live.LiveState(current_position=1, quantity=0.001)

    live.reconcile_position_state(session=object(), state=state, exchange=exchange)

    assert state.entry_fill_time == entry_time.isoformat()
    assert state.scheduled_exit_time == (entry_time + timedelta(hours=live.HOLD_HOURS)).isoformat()


def test_place_market_order_retries_without_reduce_only_when_exchange_rejects():
    live = load_live_module()
    exchange = StubReduceOnlyRejectExchange()

    order = live.place_market_order(
        exchange=exchange,
        side="sell",
        quantity=0.001,
        reduce_only=True,
    )

    assert order["status"] == "closed"
    assert exchange.calls == [{"reduceOnly": True}, {}]


def test_load_trade_log_frame_repairs_mixed_legacy_and_new_width_rows(tmp_path):
    live = load_live_module()
    trade_log_path = tmp_path / "trades.csv"
    live.TRADE_LOG_FILE = trade_log_path

    legacy_row = [
        "2026-04-21T06:33:56.568190+00:00",
        "open",
        "BTC",
        "top_pos",
        "4h",
        "24h",
        "1",
        "2026-04-21T04:00:00+00:00",
        "2026-04-21T04:15:00+00:00",
        "2026-04-22T04:15:00+00:00",
        "2026-04-21T06:33:56.568177+00:00",
        "75569.0",
        "",
        "75950.8",
        "",
        "0.001",
        "",
        "",
        "",
        "",
        "",
        "1.2636948973484727",
        "live",
    ]
    new_close_row = [
        "2026-04-22T09:12:43.707274+00:00",
        "close",
        "BTC",
        "top_pos",
        "4h",
        "24h",
        "1",
        "2026-04-21T00:00:00+00:00",
        "",
        "2026-04-22T06:33:57.632000+00:00",
        "2026-04-22T09:12:43.707290+00:00",
        "2026-01-21T00:00:00+00:00",
        "2026-04-21T00:00:00+00:00",
        "2026-05-21T00:00:00+00:00",
        "1.2413518518518518",
        "0.38090828162861184",
        "0.89",
        "-0.9224053894276372",
        "77949.9",
        "position_reconcile_mark_price",
        "77935.4",
        "kline_open",
        "77949.9",
        "77994.4",
        "0.001",
        "0.0005708795008074752",
        "-0.00042912049919252484",
        "-0.00018601691599351433",
        "-0.0011860169159935144",
        "-0.03344990000000739",
        "0.9224053894276372",
        "live",
        "992497088451",
        "closed",
        "0.001",
        "77994.4",
        "77.9944",
        "",
        "",
        "True",
        "LONG",
    ]

    with trade_log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(live.LEGACY_TRADE_LOG_COLUMNS)
        writer.writerow(legacy_row)
        writer.writerow(new_close_row)

    frame = live.load_trade_log_frame()

    assert list(frame.columns) == live.TRADE_LOG_COLUMNS
    assert len(frame) == 2
    assert frame.iloc[0]["actual_entry_time"] == "2026-04-21T06:33:56.568177+00:00"
    assert frame.iloc[1]["actual_exit_time"] == "2026-04-22T09:12:43.707290+00:00"

    with trade_log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == live.TRADE_LOG_COLUMNS
    assert all(len(row) == len(live.TRADE_LOG_COLUMNS) for row in rows[1:])