"""Shared helpers for Polymarket Data API requests."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_DATA_API_TIMEOUT_SECONDS = 30
DEFAULT_DATA_API_RETRY_ATTEMPTS = 3
DATA_API_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530})


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    retry_after = str(getattr(response, "headers", {}).get("Retry-After", "") or "").strip()
    if not retry_after:
        return None
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return None


def _retry_wait_seconds(*, attempt: int, response: requests.Response | None = None) -> float:
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return min(retry_after, 60.0)
    if getattr(response, "status_code", None) == 429:
        return float(min(10 * attempt, 60))
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
