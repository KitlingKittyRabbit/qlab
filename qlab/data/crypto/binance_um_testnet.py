"""Minimal signed Binance USD-M testnet client for execution canaries."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Mapping
from urllib.parse import urlencode
from collections.abc import Sequence

import requests


DEFAULT_TESTNET_BASE_URL = "https://demo-fapi.binance.com"


class BinanceUmTestnetApiError(RuntimeError):
    """Structured testnet error used for fail-closed idempotent recovery."""

    def __init__(self, *, status_code: int, code: int | None, message: str) -> None:
        self.status_code = int(status_code)
        self.code = code
        self.message = str(message)
        super().__init__(
            f"Binance testnet HTTP {self.status_code}"
            + (f" code {self.code}" if self.code is not None else "")
            + f": {self.message}"
        )


class BinanceUmTestnetClient:
    """Fail-closed HMAC client. It never defaults to a production trade URL."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        base_url: str = DEFAULT_TESTNET_BASE_URL,
        timeout: float = 15.0,
        recv_window: int = 5_000,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("BINANCE_UM_TESTNET_API_KEY", "")).strip()
        self.api_secret = (
            api_secret or os.environ.get("BINANCE_UM_TESTNET_API_SECRET", "")
        ).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.recv_window = int(recv_window)
        self.session = session or requests.Session()
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Binance USD-M testnet API credentials are missing")
        if self.base_url != DEFAULT_TESTNET_BASE_URL:
            raise ValueError("testnet client only permits the approved Binance testnet URL")

    def _signed_params(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = {
            key: value for key, value in (params or {}).items() if value is not None
        }
        result.setdefault("recvWindow", self.recv_window)
        result.setdefault("timestamp", int(time.time() * 1000))
        query = urlencode(result, doseq=True)
        result["signature"] = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return result

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        request_params = self._signed_params(params) if signed else dict(params or {})
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            params=request_params,
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {}
            raise BinanceUmTestnetApiError(
                status_code=response.status_code,
                code=(
                    int(error_payload["code"])
                    if isinstance(error_payload, Mapping)
                    and error_payload.get("code") is not None
                    else None
                ),
                message=(
                    str(error_payload.get("msg", response.text[:500]))
                    if isinstance(error_payload, Mapping)
                    else response.text[:500]
                ),
            )
        return response.json()

    def server_time(self) -> int:
        payload = self._request("GET", "/fapi/v1/time")
        return int(payload["serverTime"])

    def exchange_info(self) -> dict[str, Any]:
        payload = self._request("GET", "/fapi/v1/exchangeInfo")
        if not isinstance(payload, dict):
            raise RuntimeError("testnet exchangeInfo is not an object")
        return payload

    def position_mode(self) -> bool:
        payload = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return bool(payload["dualSidePosition"])

    def account(self) -> dict[str, Any]:
        payload = self._request("GET", "/fapi/v3/account", signed=True)
        if not isinstance(payload, dict):
            raise RuntimeError("testnet account is not an object")
        return payload

    def book_ticker(self, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "GET", "/fapi/v1/ticker/bookTicker", params={"symbol": symbol}
        )
        if not isinstance(payload, dict):
            raise RuntimeError("testnet bookTicker is not an object")
        return payload

    def change_initial_leverage(self, *, symbol: str, leverage: int) -> dict[str, Any]:
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        payload = self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            signed=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("testnet leverage response is not an object")
        return payload

    def position_risk(self, *, symbol: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/fapi/v2/positionRisk",
            params={"symbol": symbol},
            signed=True,
        )
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise RuntimeError("testnet positionRisk is not a list of objects")
        return payload

    def all_position_risk(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise RuntimeError("testnet positionRisk is not a list of objects")
        return payload

    def new_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not client_order_id:
            raise ValueError("client_order_id is required for idempotent canary orders")
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": format(quantity, ".15g"),
            "positionSide": "BOTH",
            "reduceOnly": "true" if reduce_only else "false",
            "newOrderRespType": "RESULT",
        }
        params["newClientOrderId"] = client_order_id
        payload = self._request("POST", "/fapi/v1/order", params=params, signed=True)
        if not isinstance(payload, dict):
            raise RuntimeError("testnet order response is not an object")
        return payload

    def new_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        client_order_id: str,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        if not client_order_id:
            raise ValueError("client_order_id is required for idempotent canary orders")
        if time_in_force not in {"GTC", "IOC", "FOK", "GTX"}:
            raise ValueError("unsupported time_in_force")
        payload = self._request(
            "POST",
            "/fapi/v1/order",
            params={
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": time_in_force,
                "quantity": format(quantity, ".15g"),
                "price": format(price, ".15g"),
                "positionSide": "BOTH",
                "reduceOnly": "true" if reduce_only else "false",
                "newOrderRespType": "ACK",
                "newClientOrderId": client_order_id,
            },
            signed=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("testnet limit order response is not an object")
        return payload

    def query_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not client_order_id:
            raise ValueError("order_id or client_order_id is required")
        payload = self._request(
            "GET",
            "/fapi/v1/order",
            params={
                "symbol": symbol,
                "orderId": order_id,
                "origClientOrderId": client_order_id,
            },
            signed=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("testnet query order response is not an object")
        return payload

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not client_order_id:
            raise ValueError("order_id or client_order_id is required")
        payload = self._request(
            "DELETE",
            "/fapi/v1/order",
            params={
                "symbol": symbol,
                "orderId": order_id,
                "origClientOrderId": client_order_id,
            },
            signed=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("testnet cancel order response is not an object")
        return payload

    def account_trades(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/fapi/v1/userTrades",
            params={"symbol": symbol, "orderId": order_id},
            signed=True,
        )
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise RuntimeError("testnet userTrades is not a list of objects")
        return payload


def _single_position_amount(rows: list[dict[str, Any]], *, symbol: str) -> float:
    matching = [row for row in rows if str(row.get("symbol")) == symbol]
    if len(matching) != 1:
        raise RuntimeError(f"expected one {symbol} position row, found {len(matching)}")
    return float(matching[0]["positionAmt"])


def non_flat_account_positions(
    client: BinanceUmTestnetClient,
) -> dict[str, float]:
    """Return every non-flat USD-M Testnet position in the account."""
    return {
        str(row["symbol"]): float(row.get("positionAmt", 0.0))
        for row in client.all_position_risk()
        if float(row.get("positionAmt", 0.0)) != 0.0
    }


def resting_buy_price(
    exchange_info: Mapping[str, Any],
    *,
    symbol: str,
    current_bid: float,
) -> float:
    """Choose a tick-aligned bid far from the book but inside percent-price limits."""
    if current_bid <= 0:
        raise ValueError("current_bid must be positive")
    rows = exchange_info.get("symbols")
    if not isinstance(rows, list):
        raise ValueError("exchangeInfo payload has no symbols list")
    matching = [
        row for row in rows if isinstance(row, Mapping) and row.get("symbol") == symbol
    ]
    if len(matching) != 1:
        raise ValueError(f"exchangeInfo must contain exactly one {symbol} row")
    filters = {
        str(item.get("filterType")): item
        for item in matching[0].get("filters", [])
        if isinstance(item, Mapping)
    }
    price_filter = filters.get("PRICE_FILTER")
    percent_filter = filters.get("PERCENT_PRICE")
    if not isinstance(price_filter, Mapping) or not isinstance(percent_filter, Mapping):
        raise ValueError(f"{symbol} price filters are incomplete")
    tick = Decimal(str(price_filter["tickSize"]))
    minimum = Decimal(str(price_filter["minPrice"]))
    multiplier_down = Decimal(str(percent_filter["multiplierDown"]))
    if tick <= 0 or not Decimal("0") < multiplier_down < Decimal("1"):
        raise ValueError(f"{symbol} price filters are invalid")
    # The midpoint between the lower permitted multiplier and the live bid gives
    # several percent of resting distance on Binance without violating the band.
    target = Decimal(str(current_bid)) * (
        (Decimal("1") + multiplier_down) / Decimal("2")
    )
    rounded = (target / tick).to_integral_value(rounding=ROUND_DOWN) * tick
    if rounded < minimum:
        raise ValueError(f"{symbol} resting price falls below minPrice")
    return float(rounded)


def _account_trades_with_retry(
    client: BinanceUmTestnetClient,
    *,
    symbol: str,
    order_id: int,
    attempts: int,
    delay_seconds: float,
) -> list[dict[str, Any]]:
    if attempts <= 0 or delay_seconds < 0:
        raise ValueError("trade lookup retry settings are invalid")
    for attempt in range(attempts):
        trades = client.account_trades(symbol=symbol, order_id=order_id)
        if trades:
            return trades
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"testnet order {order_id} has no account trade records after {attempts} attempts"
    )


def _query_order_optional(
    client: BinanceUmTestnetClient,
    *,
    symbol: str,
    client_order_id: str,
) -> dict[str, Any] | None:
    try:
        return client.query_order(
            symbol=symbol, client_order_id=client_order_id
        )
    except BinanceUmTestnetApiError as exc:
        if exc.code == -2013:
            return None
        raise


def _reconciled_market_order(
    client: BinanceUmTestnetClient,
    *,
    symbol: str,
    side: str,
    quantity: float,
    reduce_only: bool,
    client_order_id: str,
) -> dict[str, Any]:
    """Return an existing deterministic order or reconcile an uncertain POST."""
    existing = _query_order_optional(
        client, symbol=symbol, client_order_id=client_order_id
    )
    if existing is not None:
        return existing
    try:
        return client.new_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )
    except Exception as submit_error:
        recovered = _query_order_optional(
            client, symbol=symbol, client_order_id=client_order_id
        )
        if recovered is not None:
            return recovered
        raise submit_error


def _validate_aggregate_orders_against_testnet(
    client: BinanceUmTestnetClient,
    orders: Sequence[Mapping[str, Any]],
    *,
    maximum_gross_notional: float,
) -> dict[str, float]:
    if maximum_gross_notional <= 0:
        raise ValueError("maximum_gross_notional must be positive")
    exchange_info = client.exchange_info()
    symbols = {
        str(row.get("symbol")): row
        for row in exchange_info.get("symbols", [])
        if isinstance(row, Mapping)
    }
    notionals: dict[str, float] = {}
    for row in orders:
        symbol = str(row["symbol"])
        info = symbols.get(symbol)
        if not isinstance(info, Mapping) or str(info.get("status")) != "TRADING":
            raise RuntimeError(f"testnet symbol is not TRADING: {symbol}")
        filters = {
            str(item.get("filterType")): item
            for item in info.get("filters", [])
            if isinstance(item, Mapping)
        }
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")
        if not isinstance(lot, Mapping) or not isinstance(
            notional_filter, Mapping
        ):
            raise RuntimeError(f"testnet trading filters are incomplete: {symbol}")
        quantity = Decimal(str(row["quantity"]))
        step = Decimal(str(lot["stepSize"]))
        minimum_quantity = Decimal(str(lot["minQty"]))
        maximum_quantity = Decimal(str(lot["maxQty"]))
        if (
            quantity < minimum_quantity
            or quantity > maximum_quantity
            or step <= 0
            or quantity % step != 0
        ):
            raise RuntimeError(f"testnet quantity violates filters: {symbol}")
        ticker = client.book_ticker(symbol)
        price = float(
            ticker["askPrice"] if row["side"] == "BUY" else ticker["bidPrice"]
        )
        notional = float(quantity) * price
        minimum_notional = float(
            notional_filter.get(
                "notional", notional_filter.get("minNotional", 0.0)
            )
        )
        if notional < minimum_notional:
            raise RuntimeError(f"testnet order is below minimum notional: {symbol}")
        notionals[symbol] = notional
    gross = sum(notionals.values())
    if gross > maximum_gross_notional + 1e-9:
        raise RuntimeError(
            f"aggregate testnet gross exceeds limit: {gross} > {maximum_gross_notional}"
        )
    return notionals


def run_open_execution_canary(
    client: BinanceUmTestnetClient,
    *,
    symbol: str,
    quantity: float,
    resting_price: float,
    client_order_prefix: str,
    leverage: int = 5,
    trade_lookup_attempts: int = 5,
    trade_lookup_delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Exercise resting-order cancellation and leave one filled canary to recover."""
    if quantity <= 0 or resting_price <= 0:
        raise ValueError("quantity and resting_price must be positive")
    if not client_order_prefix or len(client_order_prefix) > 24:
        raise ValueError("client_order_prefix must contain 1 to 24 characters")
    if client.position_mode():
        raise RuntimeError("testnet canary requires one-way position mode")
    initial_position = _single_position_amount(
        client.position_risk(symbol=symbol), symbol=symbol
    )
    if initial_position != 0.0:
        raise RuntimeError("testnet canary requires a flat initial position")
    leverage_result = client.change_initial_leverage(
        symbol=symbol, leverage=leverage
    )
    if int(leverage_result.get("leverage", 0)) != leverage:
        raise RuntimeError("testnet leverage response does not match the requested value")

    limit_client_id = f"{client_order_prefix}-limit"
    market_client_id = f"{client_order_prefix}-open"
    limit_order = client.new_limit_order(
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        price=resting_price,
        time_in_force="GTX",
        client_order_id=limit_client_id,
    )
    if str(limit_order.get("clientOrderId")) != limit_client_id:
        raise RuntimeError("testnet resting canary ACK has the wrong client order ID")
    queried_limit = client.query_order(
        symbol=symbol, client_order_id=limit_client_id
    )
    if str(queried_limit.get("status")) != "NEW":
        raise RuntimeError("testnet resting canary order was not queryable as NEW")
    canceled_limit = client.cancel_order(
        symbol=symbol, client_order_id=limit_client_id
    )
    if str(canceled_limit.get("status")) != "CANCELED":
        raise RuntimeError("testnet resting canary order was not canceled")
    queried_canceled = client.query_order(
        symbol=symbol, client_order_id=limit_client_id
    )
    if str(queried_canceled.get("status")) != "CANCELED":
        raise RuntimeError("testnet canceled canary order was not queryable as CANCELED")

    market_order = client.new_market_order(
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        client_order_id=market_client_id,
    )
    if str(market_order.get("status")) != "FILLED":
        raise RuntimeError("testnet market canary order was not filled")
    queried_market = client.query_order(
        symbol=symbol, client_order_id=market_client_id
    )
    if str(queried_market.get("status")) != "FILLED":
        raise RuntimeError("testnet market canary order was not queryable as FILLED")
    order_id = int(queried_market["orderId"])
    trades = _account_trades_with_retry(
        client,
        symbol=symbol,
        order_id=order_id,
        attempts=trade_lookup_attempts,
        delay_seconds=trade_lookup_delay_seconds,
    )
    position_after_open = _single_position_amount(
        client.position_risk(symbol=symbol), symbol=symbol
    )
    if position_after_open <= 0:
        raise RuntimeError("testnet market canary did not create a long position")
    return {
        "symbol": symbol,
        "requested_quantity": quantity,
        "leverage": leverage,
        "limit_client_order_id": limit_client_id,
        "market_client_order_id": market_client_id,
        "limit_order": limit_order,
        "queried_limit": queried_limit,
        "canceled_limit": canceled_limit,
        "queried_canceled": queried_canceled,
        "market_order": market_order,
        "queried_market": queried_market,
        "market_trades": trades,
        "position_after_open": position_after_open,
    }


def recover_and_close_execution_canary(
    client: BinanceUmTestnetClient,
    open_result: Mapping[str, Any],
    *,
    close_client_order_id: str,
    trade_lookup_attempts: int = 5,
    trade_lookup_delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Recover an open canary from exchange state, close it, and prove flatness."""
    symbol = str(open_result.get("symbol", ""))
    market_client_order_id = str(open_result.get("market_client_order_id", ""))
    if not symbol or not market_client_order_id or not close_client_order_id:
        raise ValueError("canary recovery identifiers are incomplete")
    if client.position_mode():
        raise RuntimeError("testnet canary recovery requires one-way position mode")
    recovered_open = client.query_order(
        symbol=symbol, client_order_id=market_client_order_id
    )
    if str(recovered_open.get("status")) != "FILLED":
        raise RuntimeError("recovered opening canary order is not FILLED")
    open_trades = _account_trades_with_retry(
        client,
        symbol=symbol,
        order_id=int(recovered_open["orderId"]),
        attempts=trade_lookup_attempts,
        delay_seconds=trade_lookup_delay_seconds,
    )
    recovered_position = _single_position_amount(
        client.position_risk(symbol=symbol), symbol=symbol
    )
    if recovered_position <= 0:
        raise RuntimeError("recovered canary position is not long")
    close_order = client.new_market_order(
        symbol=symbol,
        side="SELL",
        quantity=abs(recovered_position),
        reduce_only=True,
        client_order_id=close_client_order_id,
    )
    if str(close_order.get("status")) != "FILLED":
        raise RuntimeError("testnet close canary order was not filled")
    queried_close = client.query_order(
        symbol=symbol, client_order_id=close_client_order_id
    )
    if str(queried_close.get("status")) != "FILLED":
        raise RuntimeError("testnet close canary order was not queryable as FILLED")
    close_trades = _account_trades_with_retry(
        client,
        symbol=symbol,
        order_id=int(queried_close["orderId"]),
        attempts=trade_lookup_attempts,
        delay_seconds=trade_lookup_delay_seconds,
    )
    final_position = _single_position_amount(
        client.position_risk(symbol=symbol), symbol=symbol
    )
    if final_position != 0.0:
        raise RuntimeError("testnet canary did not finish flat")
    final_account_positions = non_flat_account_positions(client)
    if final_account_positions:
        raise RuntimeError(
            "testnet canary account did not finish flat: "
            f"{final_account_positions}"
        )
    return {
        "symbol": symbol,
        "market_client_order_id": market_client_order_id,
        "close_client_order_id": close_client_order_id,
        "recovered_open": recovered_open,
        "open_trades": open_trades,
        "recovered_position": recovered_position,
        "close_order": close_order,
        "queried_close": queried_close,
        "close_trades": close_trades,
        "final_position": final_position,
        "final_account_positions": final_account_positions,
        "final_account_flat": True,
    }


def run_flat_aggregate_execution_canary(
    client: BinanceUmTestnetClient,
    orders: Sequence[Mapping[str, Any]],
    *,
    client_order_prefix: str,
    leverage: int = 5,
    trade_lookup_attempts: int = 5,
    trade_lookup_delay_seconds: float = 0.5,
    maximum_gross_notional: float = 605.0,
) -> dict[str, Any]:
    """Submit an aggregate Testnet probe, reconcile fills, and finish flat.

    This execution-only probe is deliberately isolated from candidate virtual
    ledgers. Every opening order is followed by a reduce-only close, including
    fail-closed recovery after a partial opening batch.
    """
    if not client_order_prefix or len(client_order_prefix) > 20:
        raise ValueError("client_order_prefix must contain 1 to 20 characters")
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    normalized: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in orders:
        symbol = str(row.get("symbol", "")).strip()
        side = str(row.get("side", "")).strip()
        quantity = float(row.get("quantity", 0.0))
        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0:
            raise ValueError("aggregate canary order is invalid")
        if symbol in seen_symbols:
            raise ValueError(f"aggregate canary has duplicate symbol: {symbol}")
        seen_symbols.add(symbol)
        normalized.append({"symbol": symbol, "side": side, "quantity": quantity})
    if not normalized:
        raise ValueError("aggregate testnet canary requires at least one order")
    if client.position_mode():
        raise RuntimeError("aggregate testnet canary requires one-way position mode")
    validated_notionals = _validate_aggregate_orders_against_testnet(
        client,
        normalized,
        maximum_gross_notional=maximum_gross_notional,
    )

    account_positions = non_flat_account_positions(client)
    unexpected_positions = {
        symbol: amount
        for symbol, amount in account_positions.items()
        if symbol not in seen_symbols
    }
    if unexpected_positions:
        raise RuntimeError(
            "aggregate testnet account has unrelated open positions: "
            f"{unexpected_positions}"
        )
    initial_positions: dict[str, float] = {}
    for index, row in enumerate(normalized):
        symbol = row["symbol"]
        amount = _single_position_amount(
            client.position_risk(symbol=symbol), symbol=symbol
        )
        if amount != 0.0:
            existing_open = _query_order_optional(
                client,
                symbol=symbol,
                client_order_id=f"{client_order_prefix}-o{index:02d}",
            )
            expected_sign = 1.0 if row["side"] == "BUY" else -1.0
            if (
                existing_open is None
                or str(existing_open.get("status")) != "FILLED"
                or amount * expected_sign <= 0
            ):
                raise RuntimeError(
                    f"aggregate testnet position has no matching opening order: {symbol}"
                )
        initial_positions[symbol] = amount

    opening_results: list[dict[str, Any]] = []
    closing_results: list[dict[str, Any]] = []
    recovery_errors: list[str] = []
    try:
        for index, row in enumerate(normalized):
            symbol = row["symbol"]
            leverage_result = client.change_initial_leverage(
                symbol=symbol, leverage=leverage
            )
            if int(leverage_result.get("leverage", 0)) != leverage:
                raise RuntimeError(
                    f"testnet leverage response mismatch for {symbol}"
                )
            client_id = f"{client_order_prefix}-o{index:02d}"
            order = _reconciled_market_order(
                client,
                symbol=symbol,
                side=row["side"],
                quantity=row["quantity"],
                reduce_only=False,
                client_order_id=client_id,
            )
            if str(order.get("status")) != "FILLED":
                raise RuntimeError(f"testnet opening order not FILLED: {symbol}")
            queried = client.query_order(
                symbol=symbol, client_order_id=client_id
            )
            if str(queried.get("status")) != "FILLED":
                raise RuntimeError(
                    f"testnet opening order not queryable as FILLED: {symbol}"
                )
            trades = _account_trades_with_retry(
                client,
                symbol=symbol,
                order_id=int(queried["orderId"]),
                attempts=trade_lookup_attempts,
                delay_seconds=trade_lookup_delay_seconds,
            )
            opening_results.append(
                {
                    "request": row,
                    "client_order_id": client_id,
                    "order": order,
                    "queried_order": queried,
                    "trades": trades,
                }
            )
    finally:
        for index, row in enumerate(normalized):
            symbol = row["symbol"]
            try:
                position = _single_position_amount(
                    client.position_risk(symbol=symbol), symbol=symbol
                )
                if position == 0.0:
                    continue
                close_side = "SELL" if position > 0 else "BUY"
                client_id = f"{client_order_prefix}-c{index:02d}"
                order = _reconciled_market_order(
                    client,
                    symbol=symbol,
                    side=close_side,
                    quantity=abs(position),
                    reduce_only=True,
                    client_order_id=client_id,
                )
                if str(order.get("status")) != "FILLED":
                    raise RuntimeError(
                        f"testnet closing order not FILLED: {symbol}"
                    )
                queried = client.query_order(
                    symbol=symbol, client_order_id=client_id
                )
                if str(queried.get("status")) != "FILLED":
                    raise RuntimeError(
                        f"testnet closing order not queryable as FILLED: {symbol}"
                    )
                trades = _account_trades_with_retry(
                    client,
                    symbol=symbol,
                    order_id=int(queried["orderId"]),
                    attempts=trade_lookup_attempts,
                    delay_seconds=trade_lookup_delay_seconds,
                )
                closing_results.append(
                    {
                        "symbol": symbol,
                        "position_before_close": position,
                        "client_order_id": client_id,
                        "order": order,
                        "queried_order": queried,
                        "trades": trades,
                    }
                )
            except Exception as exc:  # preserve every recovery failure for the caller
                recovery_errors.append(f"{symbol}: {exc}")

    final_positions = non_flat_account_positions(client)
    if recovery_errors or final_positions:
        raise RuntimeError(
            "aggregate testnet canary recovery failed: "
            f"errors={recovery_errors}, non_flat={final_positions}"
        )
    if len(opening_results) != len(normalized):
        raise RuntimeError(
            "aggregate testnet canary opening batch failed before completion"
        )
    return {
        "environment": "Binance USD-M Demo/Testnet",
        "one_way_mode": True,
        "leverage": leverage,
        "requested_order_count": len(normalized),
        "opening_results": opening_results,
        "closing_results": closing_results,
        "initial_positions": initial_positions,
        "validated_open_notionals": validated_notionals,
        "maximum_gross_notional": maximum_gross_notional,
        "final_positions": {},
        "final_flat": True,
    }


__all__ = [
    "BinanceUmTestnetClient",
    "BinanceUmTestnetApiError",
    "DEFAULT_TESTNET_BASE_URL",
    "recover_and_close_execution_canary",
    "resting_buy_price",
    "run_open_execution_canary",
]
