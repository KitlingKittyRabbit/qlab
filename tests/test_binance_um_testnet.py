from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode

import pytest

from qlab.data.crypto.binance_um_testnet import (
    BinanceUmTestnetClient,
    BinanceUmTestnetApiError,
    non_flat_account_positions,
    recover_and_close_execution_canary,
    resting_buy_price,
    run_flat_aggregate_execution_canary,
    run_open_execution_canary,
)


def test_testnet_client_module_import_does_not_require_crypto_data_root(
    monkeypatch,
) -> None:
    monkeypatch.delenv("QLAB_CRYPTO_DATA_DIR", raising=False)
    monkeypatch.delenv("COINGLASS_DATA_DIR", raising=False)
    assert BinanceUmTestnetClient.__name__ == "BinanceUmTestnetClient"


class _Response:
    def __init__(self, payload, *, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/positionSide/dual"):
            return _Response({"dualSidePosition": False})
        if url.endswith("/ticker/bookTicker"):
            return _Response({"symbol": "BTCUSDT", "bidPrice": "999", "askPrice": "1001"})
        if url.endswith("/leverage"):
            return _Response({"symbol": "BTCUSDT", "leverage": 5})
        if url.endswith("/positionRisk"):
            return _Response([{"symbol": "BTCUSDT", "positionAmt": "0.001"}])
        if url.endswith("/userTrades"):
            return _Response([{"symbol": "BTCUSDT", "orderId": 1, "qty": "0.001"}])
        if url.endswith("/order"):
            return _Response(
                {
                    "status": "CANCELED" if method == "DELETE" else "FILLED",
                    "orderId": 1,
                }
            )
        return _Response({"serverTime": 1})


def test_client_refuses_production_and_missing_credentials() -> None:
    with pytest.raises(RuntimeError, match="credentials"):
        BinanceUmTestnetClient(api_key="", api_secret="")
    with pytest.raises(ValueError, match="approved Binance testnet"):
        BinanceUmTestnetClient(
            api_key="key", api_secret="secret", base_url="https://fapi.binance.com"
        )


def test_client_exposes_structured_binance_error() -> None:
    class _ErrorSession:
        def request(self, *args, **kwargs):
            return _Response(
                {"code": -2013, "msg": "Order does not exist."},
                status_code=400,
            )

    client = BinanceUmTestnetClient(
        api_key="key", api_secret="secret", session=_ErrorSession()
    )
    with pytest.raises(BinanceUmTestnetApiError) as caught:
        client.query_order(symbol="BTCUSDT", client_order_id="missing")
    assert caught.value.status_code == 400
    assert caught.value.code == -2013


def test_non_flat_account_positions_checks_the_whole_account() -> None:
    class _Client:
        def all_position_risk(self):
            return [
                {"symbol": "BTCUSDT", "positionAmt": "0"},
                {"symbol": "ETHUSDT", "positionAmt": "-0.25"},
            ]

    assert non_flat_account_positions(_Client()) == {"ETHUSDT": -0.25}


def test_signed_position_mode_request_and_market_order(monkeypatch) -> None:
    monkeypatch.setattr("time.time", lambda: 1.0)
    session = _Session()
    client = BinanceUmTestnetClient(
        api_key="key", api_secret="secret", session=session
    )
    assert client.position_mode() is False
    _, _, kwargs = session.calls[0]
    params = dict(kwargs["params"])
    signature = params.pop("signature")
    expected = hmac.new(
        b"secret", urlencode(params).encode(), hashlib.sha256
    ).hexdigest()
    assert signature == expected
    assert kwargs["headers"] == {"X-MBX-APIKEY": "key"}

    result = client.new_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        client_order_id="ksv4-test",
    )
    assert result["status"] == "FILLED"
    order_params = session.calls[-1][2]["params"]
    assert order_params["positionSide"] == "BOTH"
    assert order_params["newOrderRespType"] == "RESULT"
    assert order_params["newClientOrderId"] == "ksv4-test"
    with pytest.raises(ValueError, match="client_order_id"):
        client.new_market_order(symbol="BTCUSDT", side="BUY", quantity=0.001)


def test_limit_cancel_fill_position_and_leverage_contract() -> None:
    session = _Session()
    client = BinanceUmTestnetClient(
        api_key="key", api_secret="secret", session=session
    )
    assert client.book_ticker("BTCUSDT")["askPrice"] == "1001"
    assert client.change_initial_leverage(symbol="BTCUSDT", leverage=5)["leverage"] == 5
    assert client.position_risk(symbol="BTCUSDT")[0]["positionAmt"] == "0.001"
    assert client.all_position_risk()[0]["positionAmt"] == "0.001"

    order = client.new_limit_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        price=900.0,
        client_order_id="ksv4-limit",
    )
    assert order["orderId"] == 1
    limit_params = session.calls[-1][2]["params"]
    assert limit_params["type"] == "LIMIT"
    assert limit_params["timeInForce"] == "GTC"
    assert limit_params["newOrderRespType"] == "ACK"
    assert limit_params["newClientOrderId"] == "ksv4-limit"

    canceled = client.cancel_order(symbol="BTCUSDT", client_order_id="ksv4-limit")
    assert canceled["status"] == "CANCELED"
    assert session.calls[-1][0] == "DELETE"
    assert client.account_trades(symbol="BTCUSDT", order_id=1)[0]["orderId"] == 1

    with pytest.raises(ValueError, match="order_id or client_order_id"):
        client.cancel_order(symbol="BTCUSDT")
    with pytest.raises(ValueError, match="positive"):
        client.change_initial_leverage(symbol="BTCUSDT", leverage=0)
    with pytest.raises(ValueError, match="positive"):
        client.new_limit_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=0.0,
            client_order_id="bad",
        )


class _CanaryClient:
    def __init__(self) -> None:
        self.position = 0.0
        self.orders = {}
        self.next_order_id = 1

    def position_mode(self):
        return False

    def position_risk(self, *, symbol):
        return [{"symbol": symbol, "positionAmt": str(self.position)}]

    def all_position_risk(self):
        return [{"symbol": "BTCUSDT", "positionAmt": str(self.position)}]

    def change_initial_leverage(self, *, symbol, leverage):
        return {"symbol": symbol, "leverage": leverage}

    def new_limit_order(self, **kwargs):
        stored = {
            "symbol": kwargs["symbol"],
            "clientOrderId": kwargs["client_order_id"],
            "orderId": self.next_order_id,
            "status": "NEW",
        }
        self.next_order_id += 1
        self.orders[kwargs["client_order_id"]] = stored
        return {
            "symbol": kwargs["symbol"],
            "clientOrderId": kwargs["client_order_id"],
            "orderId": stored["orderId"],
        }

    def query_order(self, *, symbol, client_order_id=None, order_id=None):
        assert symbol == "BTCUSDT"
        return dict(self.orders[client_order_id])

    def cancel_order(self, *, symbol, client_order_id=None, order_id=None):
        assert symbol == "BTCUSDT"
        self.orders[client_order_id]["status"] = "CANCELED"
        return dict(self.orders[client_order_id])

    def new_market_order(self, **kwargs):
        signed = kwargs["quantity"] if kwargs["side"] == "BUY" else -kwargs["quantity"]
        if kwargs.get("reduce_only"):
            assert self.position * signed < 0
        self.position += signed
        result = {
            "symbol": kwargs["symbol"],
            "clientOrderId": kwargs["client_order_id"],
            "orderId": self.next_order_id,
            "status": "FILLED",
            "executedQty": str(kwargs["quantity"]),
        }
        self.next_order_id += 1
        self.orders[kwargs["client_order_id"]] = result
        return result

    def account_trades(self, *, symbol, order_id=None):
        return [{"symbol": symbol, "orderId": order_id, "qty": "0.001"}]


def test_open_canary_then_new_client_view_recovers_and_closes() -> None:
    exchange = _CanaryClient()
    opened = run_open_execution_canary(
        exchange,
        symbol="BTCUSDT",
        quantity=0.001,
        resting_price=900.0,
        client_order_prefix="ksv4-123",
        leverage=5,
    )
    assert opened["position_after_open"] == pytest.approx(0.001)
    assert opened["queried_canceled"]["status"] == "CANCELED"

    # A separately constructed client would see the same exchange-side state.
    restarted_client = exchange
    closed = recover_and_close_execution_canary(
        restarted_client,
        opened,
        close_client_order_id="ksv4-123-close",
    )
    assert closed["recovered_position"] == pytest.approx(0.001)
    assert closed["final_position"] == 0.0
    assert closed["final_account_flat"] is True
    assert exchange.position == 0.0


def test_canary_recovery_rejects_unrelated_account_position() -> None:
    class _UnrelatedPositionClient(_CanaryClient):
        def all_position_risk(self):
            return [
                {"symbol": "BTCUSDT", "positionAmt": str(self.position)},
                {"symbol": "ETHUSDT", "positionAmt": "1.0"},
            ]

    exchange = _UnrelatedPositionClient()
    opened = run_open_execution_canary(
        exchange,
        symbol="BTCUSDT",
        quantity=0.001,
        resting_price=900.0,
        client_order_prefix="ksv4-unrelated",
        leverage=5,
    )
    with pytest.raises(RuntimeError, match="account did not finish flat"):
        recover_and_close_execution_canary(
            exchange,
            opened,
            close_client_order_id="ksv4-unrelated-close",
        )


def test_resting_buy_price_is_tick_aligned_and_inside_percent_band() -> None:
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "100.0",
                        "tickSize": "0.1",
                    },
                    {
                        "filterType": "PERCENT_PRICE",
                        "multiplierDown": "0.9500",
                    },
                ],
            }
        ]
    }
    price = resting_buy_price(
        exchange_info,
        symbol="BTCUSDT",
        current_bid=63_800.13,
    )
    assert price == 62_205.1
    assert 63_800.13 * 0.95 < price < 63_800.13
    assert round(price * 10) == price * 10


def test_canary_trade_lookup_retries_bounded_eventual_consistency() -> None:
    class _DelayedTradesClient(_CanaryClient):
        def __init__(self) -> None:
            super().__init__()
            self.trade_attempts = {}

        def account_trades(self, *, symbol, order_id=None):
            count = self.trade_attempts.get(order_id, 0) + 1
            self.trade_attempts[order_id] = count
            return [] if count == 1 else super().account_trades(
                symbol=symbol, order_id=order_id
            )

    exchange = _DelayedTradesClient()
    opened = run_open_execution_canary(
        exchange,
        symbol="BTCUSDT",
        quantity=0.001,
        resting_price=900.0,
        client_order_prefix="ksv4-delay",
        trade_lookup_attempts=2,
        trade_lookup_delay_seconds=0.0,
    )
    closed = recover_and_close_execution_canary(
        exchange,
        opened,
        close_client_order_id="ksv4-delay-close",
        trade_lookup_attempts=2,
        trade_lookup_delay_seconds=0.0,
    )
    assert closed["final_position"] == 0.0
    assert exchange.trade_attempts[opened["queried_market"]["orderId"]] == 3
    assert exchange.trade_attempts[closed["queried_close"]["orderId"]] == 2


class _AggregateCanaryClient:
    def __init__(self, *, fail_open_symbol=None) -> None:
        self.positions = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
        self.orders = {}
        self.next_order_id = 1
        self.fail_open_symbol = fail_open_symbol

    def position_mode(self):
        return False

    def exchange_info(self):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "filters": [
                        {
                            "filterType": "MARKET_LOT_SIZE",
                            "minQty": "0.001",
                            "maxQty": "1000",
                            "stepSize": "0.001",
                        },
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
                for symbol in self.positions
            ]
        }

    def book_ticker(self, symbol):
        return {"symbol": symbol, "bidPrice": "999", "askPrice": "1001"}

    def position_risk(self, *, symbol):
        return [{"symbol": symbol, "positionAmt": str(self.positions[symbol])}]

    def all_position_risk(self):
        return [
            {"symbol": symbol, "positionAmt": str(amount)}
            for symbol, amount in self.positions.items()
        ]

    def change_initial_leverage(self, *, symbol, leverage):
        return {"symbol": symbol, "leverage": leverage}

    def new_market_order(self, **kwargs):
        if (
            kwargs["symbol"] == self.fail_open_symbol
            and not kwargs.get("reduce_only")
        ):
            raise RuntimeError("deliberate opening failure")
        signed = kwargs["quantity"] if kwargs["side"] == "BUY" else -kwargs["quantity"]
        if kwargs.get("reduce_only"):
            assert self.positions[kwargs["symbol"]] * signed < 0
        self.positions[kwargs["symbol"]] += signed
        result = {
            "symbol": kwargs["symbol"],
            "clientOrderId": kwargs["client_order_id"],
            "orderId": self.next_order_id,
            "status": "FILLED",
            "executedQty": str(kwargs["quantity"]),
        }
        self.next_order_id += 1
        self.orders[kwargs["client_order_id"]] = result
        return result

    def query_order(self, *, symbol, client_order_id=None, order_id=None):
        if client_order_id not in self.orders:
            raise BinanceUmTestnetApiError(
                status_code=400, code=-2013, message="Order does not exist."
            )
        return dict(self.orders[client_order_id])

    def account_trades(self, *, symbol, order_id=None):
        return [{"symbol": symbol, "orderId": order_id, "qty": "1"}]


def test_aggregate_canary_opens_reconciles_and_finishes_flat() -> None:
    exchange = _AggregateCanaryClient()
    result = run_flat_aggregate_execution_canary(
        exchange,
        [
            {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01},
            {"symbol": "ETHUSDT", "side": "SELL", "quantity": 0.1},
        ],
        client_order_prefix="ksv4-a1",
    )
    assert result["requested_order_count"] == 2
    assert len(result["opening_results"]) == 2
    assert len(result["closing_results"]) == 2
    assert result["final_flat"] is True
    assert exchange.positions == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}


def test_aggregate_canary_recovers_partial_open_before_failing() -> None:
    exchange = _AggregateCanaryClient(fail_open_symbol="ETHUSDT")
    with pytest.raises(RuntimeError, match="deliberate opening failure"):
        run_flat_aggregate_execution_canary(
            exchange,
            [
                {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01},
                {"symbol": "ETHUSDT", "side": "SELL", "quantity": 0.1},
            ],
            client_order_prefix="ksv4-a2",
        )
    assert exchange.positions == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}


def test_aggregate_canary_recovers_order_accepted_before_timeout() -> None:
    class _AcceptedThenTimeout(_AggregateCanaryClient):
        def __init__(self):
            super().__init__()
            self.timed_out = False

        def new_market_order(self, **kwargs):
            result = super().new_market_order(**kwargs)
            if not kwargs.get("reduce_only") and not self.timed_out:
                self.timed_out = True
                raise TimeoutError("response lost after exchange acceptance")
            return result

    exchange = _AcceptedThenTimeout()
    result = run_flat_aggregate_execution_canary(
        exchange,
        [{"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01}],
        client_order_prefix="ksv4-a3",
    )
    assert result["final_flat"] is True
    assert len(result["opening_results"]) == 1
    assert exchange.positions["BTCUSDT"] == 0.0


def test_aggregate_canary_rejects_testnet_filter_or_gross_mismatch() -> None:
    exchange = _AggregateCanaryClient()
    with pytest.raises(RuntimeError, match="quantity violates"):
        run_flat_aggregate_execution_canary(
            exchange,
            [{"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.0105}],
            client_order_prefix="ksv4-a4",
        )
    with pytest.raises(RuntimeError, match="gross exceeds"):
        run_flat_aggregate_execution_canary(
            exchange,
            [{"symbol": "BTCUSDT", "side": "BUY", "quantity": 1.0}],
            client_order_prefix="ksv4-a5",
            maximum_gross_notional=600.0,
        )


def test_aggregate_canary_recovers_after_process_dies_with_open_position() -> None:
    exchange = _AggregateCanaryClient()
    exchange.new_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01,
        client_order_id="ksv4-a6-o00",
    )
    assert exchange.positions["BTCUSDT"] == pytest.approx(0.01)
    result = run_flat_aggregate_execution_canary(
        exchange,
        [{"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01}],
        client_order_prefix="ksv4-a6",
    )
    assert result["final_flat"] is True
    assert exchange.positions["BTCUSDT"] == 0.0


def test_aggregate_canary_rejects_empty_probe() -> None:
    with pytest.raises(ValueError, match="at least one order"):
        run_flat_aggregate_execution_canary(
            _AggregateCanaryClient(),
            [],
            client_order_prefix="ksv4-empty",
        )


def test_aggregate_canary_rejects_unrelated_account_position() -> None:
    exchange = _AggregateCanaryClient()
    exchange.positions["ETHUSDT"] = 0.1
    with pytest.raises(RuntimeError, match="unrelated open positions"):
        run_flat_aggregate_execution_canary(
            exchange,
            [{"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01}],
            client_order_prefix="ksv4-other",
        )
