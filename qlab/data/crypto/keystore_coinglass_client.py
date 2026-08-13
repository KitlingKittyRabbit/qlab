from __future__ import annotations

"""Lifecycle: candidate.

Reusable KeyStore/CoinGlass v4 proxy client. Promote to active only after
pagination, schema, coverage, and overlap validation pass for the research
universe. Archive if KeyStore is no longer the replacement data source.
"""

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc
import json
from pathlib import Path
from typing import Any, Iterable

import requests


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_URL = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass/v4"
DEFAULT_RATE_LIMIT_SLEEP = 6.2


def serialized_request_wait_seconds(
    previous_request_ts: str | datetime | None,
    current_ts: str | datetime,
    *,
    min_start_interval_seconds: float = DEFAULT_RATE_LIMIT_SLEEP,
) -> float:
    """Return the remaining wait needed between serialized request starts."""
    if min_start_interval_seconds < 0:
        raise ValueError("min_start_interval_seconds must be non-negative")
    if previous_request_ts is None:
        return 0.0

    def parse(value: str | datetime) -> datetime:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            raise ValueError("serialized request timestamps must be timezone-aware")
        return parsed.astimezone(UTC)

    previous = parse(previous_request_ts)
    current = parse(current_ts)
    if current < previous:
        raise ValueError("current request time precedes the previous request start")
    elapsed = (current - previous).total_seconds()
    return max(0.0, float(min_start_interval_seconds) - elapsed)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def default_env_paths() -> list[Path]:
    paths: list[Path] = []
    for key in ("KEYSTORE_ENV_PATH", "QLAB_TRADE_ENV_PATH", "QLAB_CRYPTO_ENV_PATH"):
        raw_value = os.environ.get(key, "").strip()
        if raw_value:
            candidate = Path(raw_value).expanduser()
            paths.append(candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate)
    paths.extend(
        [
            WORKSPACE_ROOT / "keystore_coinglass_probe" / ".env.local",
            WORKSPACE_ROOT / "trade" / "crypto_signal" / ".env",
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def get_env_value(key: str, env_paths: Iterable[Path] | None = None) -> str:
    raw_value = os.environ.get(key, "").strip()
    if raw_value:
        return raw_value
    for path in env_paths or default_env_paths():
        value = load_env_file(path).get(key, "").strip()
        if value:
            return value
    return ""


def load_api_key(env_paths: Iterable[Path] | None = None) -> str:
    api_key = get_env_value("KEYSTORE_API_KEY", env_paths=env_paths)
    if api_key:
        return api_key
    searched = ", ".join(str(path) for path in (env_paths or default_env_paths()))
    raise RuntimeError(f"KEYSTORE_API_KEY not found. Searched: {searched}")


def get_rate_limit_sleep() -> float:
    raw_value = os.environ.get("KEYSTORE_RATE_LIMIT_SLEEP", "").strip()
    if not raw_value:
        return DEFAULT_RATE_LIMIT_SLEEP
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_RATE_LIMIT_SLEEP


def parse_timestamp_ms(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    if isinstance(value, (int, float)):
        raw = int(value)
        return raw * 1000 if abs(raw) < 10_000_000_000 else raw
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        raw = int(text)
        return raw * 1000 if abs(raw) < 10_000_000_000 else raw
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def format_timestamp_ms(value: int | None) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()


def find_data_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("data", "result", "rows", "list", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = find_data_rows(value)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and value:
            return value
    return []


def extract_row_timestamp_ms(row: Any, time_keys: Iterable[str] = ("time", "timestamp", "t", "date")) -> int | None:
    if not isinstance(row, dict):
        return None
    for key in time_keys:
        if key not in row:
            continue
        parsed = parse_timestamp_ms(row.get(key))
        if parsed is not None:
            return parsed
    return None


@dataclass(frozen=True)
class HistoryPage:
    page_index: int
    request_params: dict[str, Any]
    row_count: int
    earliest_ms: int | None
    latest_ms: int | None
    rows: list[Any]

    @property
    def earliest(self) -> str:
        return format_timestamp_ms(self.earliest_ms)

    @property
    def latest(self) -> str:
        return format_timestamp_ms(self.latest_ms)


@dataclass(frozen=True)
class RawKeystoreResponse:
    path: str
    request_params: dict[str, Any]
    request_ts: str
    response_ts: str
    raw_payload: bytes

    def json_payload(self) -> Any:
        try:
            return json.loads(self.raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"KeyStore response is not valid JSON: {exc}") from exc


class KeystoreCoinglassClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 35.0,
        rate_limit_sleep: float | None = None,
    ) -> None:
        self.api_key = api_key or load_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_sleep = get_rate_limit_sleep() if rate_limit_sleep is None else max(0.0, rate_limit_sleep)
        self._request_start_lock = threading.Lock()
        self._last_request_start_monotonic: float | None = None

    def _wait_for_request_start_slot(self) -> None:
        with self._request_start_lock:
            now = time.monotonic()
            if self._last_request_start_monotonic is not None:
                remaining = self.rate_limit_sleep - (
                    now - self._last_request_start_monotonic
                )
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_start_monotonic = time.monotonic()

    def request_raw(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> RawKeystoreResponse:
        clean_params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"accept": "application/json", "X-Api-Key": self.api_key}
        last_error: Exception | None = None

        for attempt in range(retries):
            self._wait_for_request_start_slot()
            request_ts = datetime.now(UTC)
            try:
                response = requests.get(url, headers=headers, params=clean_params, timeout=self.timeout)
                response_ts = datetime.now(UTC)
                raw_payload = bytes(response.content)
                try:
                    payload: Any = json.loads(raw_payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    last_error = RuntimeError(f"JSON decode failed: {exc}")
                    time.sleep(3 * (attempt + 1))
                    continue
            except Exception as exc:
                last_error = exc
                time.sleep(3 * (attempt + 1))
                continue

            code = ""
            if isinstance(payload, dict):
                code = str(payload.get("code", payload.get("status", "")))
            if response.status_code == 429 or code in {"429", "40003"}:
                last_error = RuntimeError(
                    f"HTTP={response.status_code}, code={code}, rate limit"
                )
                time.sleep(max(10.0, self.rate_limit_sleep))
                continue
            if response.status_code >= 500 or code in {"500", "50000"}:
                last_error = RuntimeError(
                    f"HTTP={response.status_code}, code={code}, transient server error"
                )
                if attempt + 1 < retries:
                    time.sleep(3 * (attempt + 1))
                    continue
                break
            if response.status_code == 200 and code in {"", "0", "success"}:
                return RawKeystoreResponse(
                    path=path,
                    request_params=dict(clean_params),
                    request_ts=request_ts.isoformat(),
                    response_ts=response_ts.isoformat(),
                    raw_payload=raw_payload,
                )

            message = ""
            if isinstance(payload, dict):
                message = str(payload.get("msg", payload.get("message", payload.get("error", ""))))
            last_error = RuntimeError(f"HTTP={response.status_code}, code={code}, msg={message}")
            break

        raise RuntimeError(f"KeyStore CoinGlass request failed for {path} params={clean_params}: {last_error}")

    def request_json(self, path: str, params: dict[str, Any] | None = None, retries: int = 4) -> Any:
        return self.request_raw(path, params=params, retries=retries).json_payload()

    def fetch_rows(self, path: str, params: dict[str, Any] | None = None) -> list[Any]:
        return find_data_rows(self.request_json(path, params=params))

    def iter_history_pages(
        self,
        path: str,
        params: dict[str, Any],
        *,
        target_start_ms: int | None = None,
        initial_end_time_ms: int | None = None,
        max_pages: int = 1,
        time_keys: Iterable[str] = ("time", "timestamp", "t", "date"),
    ) -> list[HistoryPage]:
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")

        request_params = dict(params)
        request_params.pop("start_time", None)
        if initial_end_time_ms is not None:
            request_params["end_time"] = str(initial_end_time_ms)

        pages: list[HistoryPage] = []
        previous_end: int | None = None
        for page_index in range(max_pages):
            if page_index > 0 and self.rate_limit_sleep:
                time.sleep(self.rate_limit_sleep)
            try:
                rows = self.fetch_rows(path, request_params)
            except Exception:
                if pages:
                    break
                raise
            timestamps = [ts for ts in (extract_row_timestamp_ms(row, time_keys=time_keys) for row in rows) if ts is not None]
            earliest_ms = min(timestamps) if timestamps else None
            latest_ms = max(timestamps) if timestamps else None
            pages.append(
                HistoryPage(
                    page_index=page_index,
                    request_params=dict(request_params),
                    row_count=len(rows),
                    earliest_ms=earliest_ms,
                    latest_ms=latest_ms,
                    rows=rows,
                )
            )
            if not rows or earliest_ms is None:
                break
            if target_start_ms is not None and earliest_ms <= target_start_ms:
                break
            next_end = earliest_ms - 1
            if previous_end == next_end:
                break
            previous_end = next_end
            request_params["end_time"] = str(next_end)

        return pages
