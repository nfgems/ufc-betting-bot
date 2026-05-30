"""HTTP helpers for UFCStats pages.

UFCStats sometimes serves a small JavaScript proof-of-work page before the
normal static HTML. The scraper runs without a browser, so solve that challenge
in the requests session and retry the original URL.
"""

from __future__ import annotations

import hashlib
import logging
import re
from urllib.parse import urljoin

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


class UFCStatsChallengeError(RuntimeError):
    """Raised when a UFCStats JavaScript gate cannot be solved."""


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


def request_ufcstats(
    url: str,
    *,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | float = 30,
    max_challenge_attempts: int = 2,
) -> requests.Response:
    """Fetch a UFCStats URL, solving the browser-check gate when present."""
    active_session = session or requests.Session()
    request_headers = dict(DEFAULT_UFCSTATS_HEADERS)
    if headers:
        request_headers.update(headers)

    response = active_session.get(url, headers=request_headers, timeout=timeout)
    response.raise_for_status()

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

        response = active_session.get(url, headers=request_headers, timeout=timeout)
        response.raise_for_status()

    if looks_like_ufcstats_challenge(response.text):
        raise UFCStatsChallengeError(
            f"UFCStats still returned browser-check page for {url}"
        )
    return response
