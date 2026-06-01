"""HTTP helpers for UFCStats pages.

UFCStats sometimes serves a small JavaScript proof-of-work page before the
normal static HTML. The scraper runs without a browser, so solve that challenge
in the requests session and retry the original URL.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

logger = logging.getLogger(__name__)

DEFAULT_UFCSTATS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_NONCE_RE = re.compile(r'var\s+nonce\s*=\s*"([^"]+)"')
_TARGET_RE = re.compile(r"target\s*=\s*new\s+Array\s*\(\s*(\d+)\s*\+\s*1\s*\)")
_UFCSTATS_HOSTS = {"ufcstats.com", "www.ufcstats.com"}
_UFCSTATS_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_UFCSTATS_DEFAULT_REQUEST_ATTEMPTS = 3


class UFCStatsChallengeError(RuntimeError):
    """Raised when a UFCStats JavaScript gate cannot be solved."""


def normalize_ufcstats_url(url: str) -> str:
    """Canonicalize real UFCStats URLs to HTTP.

    UFCStats no longer serves TLS, but stale profile/fight URLs can still enter
    through cached data or old scrape artifacts.
    """
    text = str(url or "").strip()
    parsed = urlsplit(text)
    if parsed.netloc.lower() not in _UFCSTATS_HOSTS:
        return text
    if parsed.scheme not in {"http", "https"}:
        return text
    return urlunsplit(("http", "ufcstats.com", parsed.path, parsed.query, parsed.fragment))


def looks_like_ufcstats_challenge(html: str | None) -> bool:
    """Return true when the response is the UFCStats browser-check page."""
    text = (html or "")[:6000].lower()
    if not text:
        return False
    return (
        "checking your browser" in text
        and "var nonce" in text
        and "/__c" in text
    ) or (
        "<title>loading" in text
        and "requires javascript" in text
        and "/__c" in text
    )


def _solve_challenge_nonce(
    *,
    nonce: str,
    target_length: int,
    max_iterations: int = 5_000_000,
) -> int:
    target = "0" * target_length
    for n in range(max_iterations):
        digest = hashlib.sha256(f"{nonce}:{n}".encode("utf-8")).hexdigest()
        if digest.startswith(target):
            return n
    raise UFCStatsChallengeError(
        f"Could not solve UFCStats challenge after {max_iterations} attempts"
    )


def _extract_challenge_solution(html: str) -> tuple[str, int, int]:
    nonce_match = _NONCE_RE.search(html or "")
    target_match = _TARGET_RE.search(html or "")
    if nonce_match is None or target_match is None:
        raise UFCStatsChallengeError("UFCStats challenge page did not expose nonce metadata")

    nonce = nonce_match.group(1)
    target_length = int(target_match.group(1))
    return nonce, target_length, _solve_challenge_nonce(
        nonce=nonce,
        target_length=target_length,
    )


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_wait_seconds(attempt: int) -> float:
    return float(min(0.5 * (2 ** max(0, attempt - 1)), 8.0))


def _should_retry_request_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    status_code = _http_status_code(exc)
    return status_code in _UFCSTATS_RETRY_STATUSES


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    timeout: int | float,
    attempts: int,
) -> requests.Response:
    attempts = max(int(attempts), 1)
    for attempt in range(1, attempts + 1):
        response: requests.Response | None = None
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if (
                getattr(response, "status_code", None) in _UFCSTATS_RETRY_STATUSES
                and attempt < attempts
            ):
                wait_seconds = _retry_wait_seconds(attempt)
                logger.debug(
                    "UFCStats returned %s for %s; retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    url,
                    wait_seconds,
                    attempt + 1,
                    attempts,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            if attempt >= attempts or not _should_retry_request_exception(exc):
                raise
            wait_seconds = _retry_wait_seconds(attempt)
            logger.debug(
                "UFCStats request failed for %s; retrying in %.1fs (attempt %d/%d): %s",
                url,
                wait_seconds,
                attempt + 1,
                attempts,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"UFCStats request failed for {url}")


def request_ufcstats(
    url: str,
    *,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | float = 30,
    max_request_attempts: int = _UFCSTATS_DEFAULT_REQUEST_ATTEMPTS,
    max_challenge_attempts: int = 2,
) -> requests.Response:
    """Fetch a UFCStats URL, solving the browser-check gate when present."""
    url = normalize_ufcstats_url(url)
    active_session = session or requests.Session()
    request_headers = dict(DEFAULT_UFCSTATS_HEADERS)
    if headers:
        request_headers.update(headers)

    response = _get_with_retries(
        active_session,
        url,
        headers=request_headers,
        timeout=timeout,
        attempts=max_request_attempts,
    )

    for _ in range(max(0, max_challenge_attempts)):
        if not looks_like_ufcstats_challenge(response.text):
            return response

        nonce, target_length, solution = _extract_challenge_solution(response.text)
        logger.info(
            "Solving UFCStats browser-check challenge for %s (target_prefix_zeros=%d)",
            url,
            target_length,
        )
        challenge_url = urljoin(response.url or url, "/__c")
        challenge_response = active_session.post(
            challenge_url,
            headers={
                **request_headers,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"nonce": nonce, "n": str(solution)},
            timeout=timeout,
        )
        challenge_response.raise_for_status()

        response = _get_with_retries(
            active_session,
            url,
            headers=request_headers,
            timeout=timeout,
            attempts=max_request_attempts,
        )

    if looks_like_ufcstats_challenge(response.text):
        raise UFCStatsChallengeError(
            f"UFCStats still returned browser-check page for {url}"
        )
    return response
