"""Shared helpers for Polymarket Data API requests."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

DEFAULT_DATA_API_TIMEOUT_SECONDS = 30
DEFAULT_DATA_API_RETRY_ATTEMPTS = 3
DATA_API_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530})
DATA_API_429_BACKOFF_BASE_SECONDS = 10.0
DATA_API_MAX_RETRY_WAIT_SECONDS = 60.0
DATA_API_RETRY_JITTER_MAX_SECONDS = 1.0


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(
    response: requests.Response | None,
    *,
    now: datetime | None = None,
) -> float | None:
    if response is None:
        return None
    retry_after = str(getattr(response, "headers", {}).get("Retry-After", "") or "").strip()
    if not retry_after:
        return None
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max((retry_at - current).total_seconds(), 0.0)


def _retry_wait_seconds(
    *,
    attempt: int,
    response: requests.Response | None = None,
    jitter_fn: Callable[[float, float], float] | None = None,
) -> float:
    retry_after = _retry_after_seconds(response)
    if getattr(response, "status_code", None) == 429:
        backoff_floor = min(
            DATA_API_429_BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1)),
            DATA_API_MAX_RETRY_WAIT_SECONDS,
        )
        wait_seconds = max(backoff_floor, retry_after or 0.0)
        jitter_ceiling = min(
            DATA_API_RETRY_JITTER_MAX_SECONDS,
            max(DATA_API_MAX_RETRY_WAIT_SECONDS - wait_seconds, 0.0),
        )
        jitter = 0.0
        if jitter_ceiling > 0:
            choose_jitter = jitter_fn or random.uniform
            jitter = min(
                max(float(choose_jitter(0.0, jitter_ceiling)), 0.0),
                jitter_ceiling,
            )
        return float(min(wait_seconds + jitter, DATA_API_MAX_RETRY_WAIT_SECONDS))
    if retry_after is not None:
        return min(retry_after, DATA_API_MAX_RETRY_WAIT_SECONDS)
    return float(min(0.5 * (2 ** max(0, attempt - 1)), 8.0))


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    status_code = _http_status_code(exc)
    return status_code in DATA_API_RETRY_STATUSES


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int | float = DEFAULT_DATA_API_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_DATA_API_RETRY_ATTEMPTS,
) -> Any:
    """GET JSON from Polymarket's Data API with transient-failure retries."""
    attempts = max(int(attempts), 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        response: requests.Response | None = None
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in DATA_API_RETRY_STATUSES and attempt < attempts:
                wait_seconds = _retry_wait_seconds(attempt=attempt, response=response)
                logger.debug(
                    "Polymarket Data API returned %s for %s; retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    url,
                    wait_seconds,
                    attempt + 1,
                    attempts,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not _should_retry(exc):
                raise
            wait_seconds = _retry_wait_seconds(attempt=attempt, response=response)
            logger.debug(
                "Polymarket Data API request failed for %s; retrying in %.1fs (attempt %d/%d): %s",
                url,
                wait_seconds,
                attempt + 1,
                attempts,
                exc,
            )
            time.sleep(wait_seconds)
    raise last_exc or RuntimeError(f"Polymarket Data API request failed for {url}")
