"""
Fallback fighter data scrapers — Sherdog and Tapology.

Used when UFCStats.com has no data for a fighter (e.g., regional/Contender
Series fighters). Provides partial feature data (physical attributes, record,
fight history with methods) but not per-fight striking/grappling stats.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag

try:
    import cloudscraper
except ImportError:  # pragma: no cover - optional dependency
    cloudscraper = None

from src.config import (
    BRAVE_SEARCH_API_KEY,
    BRAVE_SEARCH_API_URL,
    BRAVE_SEARCH_HTML_FALLBACK_ENABLED,
    BRAVE_SEARCH_TIMEOUT_SECONDS,
    DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED,
    DUCKDUCKGO_SEARCH_HTML_URL,
    FIGHTDX_BASE_URL,
    FIGHTDX_FAILURE_COOLDOWN_SECONDS,
    FIGHTDX_REQUEST_TIMEOUT_SECONDS,
    MARTIALBOT_BASE_URL,
    MARTIALBOT_SEARCH_URL,
    SHERDOG_BASE_URL,
    SHERDOG_SEARCH_URL,
    TAPOLOGY_BASE_URL,
    TAPOLOGY_BROWSER_BINARY,
    TAPOLOGY_BROWSER_FALLBACK_ENABLED,
    TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS,
    TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS,
    TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS,
    TAPOLOGY_CHROMEDRIVER_BINARY,
    TAPOLOGY_PROXY_URL,
    TAPOLOGY_READER_BASE_URL,
    TAPOLOGY_READER_FALLBACK_ENABLED,
    TAPOLOGY_READER_TIMEOUT_SECONDS,
    TAPOLOGY_SEARCH_URL,
    TAPOLOGY_XVFB_BINARY,
)
from src.data.name_utils import (
    normalize_cross_source_name,
    normalize_person_name,
    same_person_name,
)

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_DELAY = 1.5  # Slightly longer than UFCStats to be polite
TAPOLOGY_REQUEST_DELAY = 3.0
TAPOLOGY_TIMEOUT_SECONDS = 45
TAPOLOGY_MAX_RETRIES = 4
TAPOLOGY_READER_MAX_RETRIES = 3
MARTIALBOT_REQUEST_DELAY = 1.5
BRAVE_SEARCH_HTML_URL = "https://search.brave.com/search"
FIGHTDX_SITE_BASE_URL = FIGHTDX_BASE_URL.rsplit("/person", 1)[0]
FIGHTDX_SITEMAP_INDEX_URL = f"{FIGHTDX_SITE_BASE_URL.rstrip('/')}/sitemap.xml"
FIGHTDX_SEARCH_URL = f"{FIGHTDX_SITE_BASE_URL.rstrip('/')}/search/"
FIGHTDX_SITEMAP_REQUEST_DELAY = 0.1
ESPN_SEARCH_URL = "https://site.api.espn.com/apis/common/v3/search"
ESPN_CORE_ATHLETE_API_URL = "https://sports.core.api.espn.com/v2/sports/mma/athletes/{athlete_id}"

# Session caches
_sherdog_url_cache: dict[str, str] = {}
_tapology_url_cache: dict[str, str] = {}
_martialbot_url_cache: dict[str, str] = {}
_fightdx_url_cache: dict[str, str] = {}
_espn_url_cache: dict[str, str] = {}
_fightdx_person_urls_cache: list[str] | None = None
_fightdx_unavailable_until = 0.0
_tapology_scraper = None
_tapology_scraper_profile_index = 0
_last_tapology_request_at = 0.0
_last_tapology_browser_request_at = 0.0
_tapology_blocked: bool | None = None  # None = not yet tested
_tapology_search_blocked = False
_tapology_browser_unavailable = False
_tapology_browser_cloudflare_blocked = False
_tapology_browser_html_cache: dict[str, str] = {}
_tapology_reader_markdown_cache: dict[str, str] = {}
_tapology_reader_unavailable = False
_tapology_browser_profile_dir: str | None = None
_tapology_browser_lock = threading.Lock()
_site_search_disabled = False
_duckduckgo_site_search_disabled = False
_external_source_alert_keys: set[tuple[str, str]] = set()

_MANUAL_SEARCH_ALIASES: dict[str, list[str]] = {
    "abdulrakhman yakhyaev": ["Abdul Rakhman Yakhyaev"],
    "dmitrii smoliakov": ["Dmitry Smoliakov", "Dmitry Smolyakov"],
    "nursultan ruziboev": ["Nursulton Ruziboev"],
    "rafael cerquiera": ["Rafael Cerqueira"],
    "seokhyeon ko": ["Seok Hyeon Ko", "Seok-hyeon Ko"],
    "tsuyoshi kohsaka": ["Tsuyoshi Kosaka"],
}
_FALLBACK_PROFILE_MATCH_MIN_SCORE = 20

# MartialBot labels switch-stance fighters "switcher"; map it to the canonical
# vocabulary the feature encoder expects (see src/features/stance_utils.py) so
# the stance is not dropped to NaN downstream.
_MARTIALBOT_STANCE_ALIASES = {"switcher": "switch"}
_CANONICAL_STANCE_LABELS = {"orthodox", "southpaw", "switch", "open stance", "sideways"}
_PROFILE_MERGE_GROUPS = {
    "record": ("record", "wins", "losses", "draws"),
    "height": ("height_raw", "height"),
    "reach": ("reach_raw", "reach"),
    "weight": ("weight_raw", "weight"),
    "stance": ("stance",),
    "dob": ("dob", "age"),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _proxy_target(proxy_url: str) -> str:
    return proxy_url.rsplit("@", 1)[-1] if proxy_url else ""


def _tapology_proxies() -> dict[str, str] | None:
    if not TAPOLOGY_PROXY_URL:
        return None
    return {"http": TAPOLOGY_PROXY_URL, "https": TAPOLOGY_PROXY_URL}


def _is_cloudflare_challenge(resp: object) -> bool:
    text = str(getattr(resp, "text", "") or "")
    headers = getattr(resp, "headers", {}) or {}
    server = str(headers.get("server", "") or headers.get("Server", "") or "")
    status_code = int(getattr(resp, "status_code", 0) or 0)
    challenge_markers = (
        "Just a moment" in text
        or "__cf_chl" in text
        or (
            "security verification" in text
            and "not a bot" in text
        )
    )
    if challenge_markers:
        return True
    return status_code == 403 and (
        "cloudflare" in server.lower() or "cloudflare" in text.lower()
    )


def _tapology_error_detail(resp: object) -> str:
    if _is_cloudflare_challenge(resp):
        return "Cloudflare challenge"
    return ""


def _tapology_fetch_url(url: str, params: dict | None = None) -> str:
    if not params:
        return url
    prepared = requests.Request("GET", url, params=params).prepare()
    return str(prepared.url or url)


def _tapology_html_is_cloudflare_challenge(html: str) -> bool:
    text = str(html or "")
    lower = text.lower()
    return (
        "just a moment" in lower
        or "__cf_chl" in text
        or "challenge-platform" in lower
        or ("security verification" in lower and "not a bot" in lower)
    )


def _tapology_browser_page_ready(fetch_url: str, html: str) -> bool:
    if not str(html or "").strip() or _tapology_html_is_cloudflare_challenge(html):
        return False
    lower = html.lower()
    url_lower = fetch_url.lower()
    if "/fightcenter/fighters/" in url_lower:
        return (
            "pro mma record" in lower
            or "amateur mma record" in lower
            or "data-bout-id" in lower
        )
    if "/rankings/" in url_lower:
        return "/fightcenter/fighters/" in lower and "rankings" in lower
    if "/search" in url_lower:
        return "tapology" in lower and (
            "/fightcenter/fighters/" in lower
            or "search" in lower
            or "no results" in lower
        )
    return "tapology" in lower or "/fightcenter/" in lower


def _tapology_browser_dependency_paths() -> tuple[str, str, str]:
    browser = (
        TAPOLOGY_BROWSER_BINARY
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or ""
    )
    chromedriver = TAPOLOGY_CHROMEDRIVER_BINARY or shutil.which("chromedriver") or ""
    xvfb = TAPOLOGY_XVFB_BINARY or shutil.which("Xvfb") or ""
    return browser, chromedriver, xvfb


def _tapology_browser_profile_path() -> str:
    global _tapology_browser_profile_dir
    if _tapology_browser_profile_dir and os.path.isdir(_tapology_browser_profile_dir):
        return _tapology_browser_profile_dir
    _tapology_browser_profile_dir = tempfile.mkdtemp(prefix="tapology-browser-profile-")
    return _tapology_browser_profile_dir


def _prepare_tapology_browser_profile(profile_dir: str) -> None:
    os.makedirs(profile_dir, exist_ok=True)
    for lock_name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        try:
            os.remove(os.path.join(profile_dir, lock_name))
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove stale Chromium profile lock %s", lock_name, exc_info=True)


def _tapology_browser_proxy_server() -> str:
    proxy_url = str(TAPOLOGY_PROXY_URL or "").strip()
    if not proxy_url:
        return ""
    parsed = urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    scheme = parsed.scheme or "http"
    host = parsed.hostname
    if not host:
        return proxy_url
    port = f":{parsed.port}" if parsed.port is not None else ""
    if parsed.username or parsed.password:
        _log_external_source_error_once(
            "Tapology",
            "browser proxy authentication may be required",
            (
                "Chromium is being configured with the proxy host, but "
                "authenticated proxies may still require request-path recovery"
            ),
            level=logging.WARNING,
        )
    return f"{scheme}://{host}{port}"


def _tapology_browser_fallback_available() -> bool:
    if (
        not TAPOLOGY_BROWSER_FALLBACK_ENABLED
        or _tapology_browser_unavailable
        or _tapology_browser_cloudflare_blocked
    ):
        return False
    browser, _chromedriver, xvfb = _tapology_browser_dependency_paths()
    return bool(browser and (os.getenv("DISPLAY") or xvfb))


def _next_xvfb_display() -> str:
    base = 90 + (os.getpid() % 100)
    for offset in range(100):
        display_num = base + offset
        socket_path = f"/tmp/.X11-unix/X{display_num}"
        lock_path = f"/tmp/.X{display_num}-lock"
        if not os.path.exists(socket_path) and not os.path.exists(lock_path):
            return f":{display_num}"
    return f":{base}"


@contextmanager
def _tapology_virtual_display():
    """Provide a display for headed Chromium when the runtime has no DISPLAY."""
    if os.getenv("DISPLAY"):
        yield
        return

    _browser, _chromedriver, xvfb = _tapology_browser_dependency_paths()
    if not xvfb:
        raise RuntimeError("Tapology browser fallback requires Xvfb when DISPLAY is unset")

    display = _next_xvfb_display()
    proc = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1400x1000x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    old_display = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = display
    try:
        time.sleep(0.5)
        yield
    finally:
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@contextmanager
def _tapology_browser_environment(profile_dir: str):
    """Give Chromium writable per-session HOME/XDG dirs under locked execution."""
    env_paths = {
        "HOME": os.path.join(profile_dir, "home"),
        "XDG_CONFIG_HOME": os.path.join(profile_dir, "config"),
        "XDG_CACHE_HOME": os.path.join(profile_dir, "cache"),
        "XDG_RUNTIME_DIR": os.path.join(profile_dir, "runtime"),
    }
    old_values = {key: os.environ.get(key) for key in env_paths}
    for key, path in env_paths.items():
        os.makedirs(path, exist_ok=True)
        if key == "XDG_RUNTIME_DIR":
            os.chmod(path, 0o700)
        os.environ[key] = path
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _get_tapology_html_with_browser(fetch_url: str) -> str:
    """Fetch Tapology HTML through a real headed Chromium session under Xvfb."""
    global _last_tapology_browser_request_at
    global _tapology_browser_unavailable, _tapology_browser_cloudflare_blocked

    if not TAPOLOGY_BROWSER_FALLBACK_ENABLED:
        raise TapologyRequestError(fetch_url, status_code=403, detail="browser fallback disabled")
    if _tapology_browser_unavailable:
        raise TapologyRequestError(fetch_url, status_code=403, detail="browser fallback unavailable")
    if _tapology_browser_cloudflare_blocked:
        raise TapologyRequestError(
            fetch_url,
            status_code=403,
            detail="browser fallback blocked by Cloudflare",
        )

    browser, chromedriver, _xvfb = _tapology_browser_dependency_paths()
    if not browser:
        _tapology_browser_unavailable = True
        raise TapologyRequestError(fetch_url, status_code=403, detail="browser binary unavailable")

    try:
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except Exception as exc:  # pragma: no cover - depends on runtime package set
        _tapology_browser_unavailable = True
        raise TapologyRequestError(fetch_url, status_code=403, detail="selenium unavailable") from exc

    with _tapology_browser_lock:
        cached_html = _tapology_browser_html_cache.get(fetch_url)
        if cached_html:
            return cached_html

        sleep_for = TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS - (
            time.monotonic() - _last_tapology_browser_request_at
        )
        if sleep_for > 0:
            time.sleep(sleep_for)

        profile_dir = _tapology_browser_profile_path()
        _prepare_tapology_browser_profile(profile_dir)
        driver = None
        try:
            with _tapology_virtual_display(), _tapology_browser_environment(profile_dir):
                options = Options()
                options.binary_location = browser
                options.page_load_strategy = "eager"
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-background-networking")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--no-first-run")
                options.add_argument("--no-default-browser-check")
                options.add_argument("--lang=en-US")
                options.add_argument("--window-size=1400,1000")
                options.add_argument(f"--user-agent={_tapology_user_agent(_tapology_browser_profiles()[0])}")
                options.add_argument(f"--user-data-dir={profile_dir}")
                proxy_server = _tapology_browser_proxy_server()
                if proxy_server:
                    options.add_argument(f"--proxy-server={proxy_server}")
                    logger.info("Tapology browser proxy enabled: %s", _proxy_target(proxy_server))
                try:
                    options.add_experimental_option("excludeSwitches", ["enable-automation"])
                    options.add_experimental_option("useAutomationExtension", False)
                except Exception:
                    logger.debug("Could not set Chromium automation-reduction options", exc_info=True)
                service = Service(chromedriver) if chromedriver else Service()
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS)
                try:
                    driver.execute_cdp_cmd(
                        "Page.addScriptToEvaluateOnNewDocument",
                        {
                            "source": (
                                "Object.defineProperty(navigator, 'webdriver', "
                                "{get: () => undefined});"
                            )
                        },
                    )
                except Exception:
                    logger.debug("Could not install Tapology browser pre-navigation script", exc_info=True)
                try:
                    driver.get(fetch_url)
                except TimeoutException:
                    logger.warning(
                        "Tapology browser navigation timed out for %s after %.1fs; reading partial page",
                        fetch_url,
                        TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS,
                    )
                    try:
                        driver.execute_cdp_cmd("Page.stopLoading", {})
                    except Exception:
                        logger.debug("Could not stop Tapology browser navigation through CDP", exc_info=True)
                    try:
                        driver.execute_script("window.stop();")
                    except Exception:
                        logger.debug("Could not stop timed-out Tapology browser navigation", exc_info=True)
                deadline = time.monotonic() + max(1.0, TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS)
                html = driver.page_source or ""
                while time.monotonic() < deadline:
                    if _tapology_browser_page_ready(fetch_url, html):
                        break
                    time.sleep(1)
                    html = driver.page_source or ""

                _last_tapology_browser_request_at = time.monotonic()
                if not html.strip():
                    raise TapologyRequestError(fetch_url, detail="browser returned empty response")
                if _tapology_html_is_cloudflare_challenge(html):
                    _tapology_browser_cloudflare_blocked = True
                    raise TapologyRequestError(
                        fetch_url,
                        status_code=403,
                        detail="Cloudflare challenge from browser fallback",
                    )
                if not _tapology_browser_page_ready(fetch_url, html):
                    raise TapologyRequestError(
                        fetch_url,
                        detail="browser returned incomplete Tapology response",
                    )

                logger.info("Tapology browser fallback fetched %s", fetch_url)
                _tapology_browser_html_cache[fetch_url] = html
                return html
        except TapologyRequestError:
            raise
        except Exception as exc:
            if driver is None:
                _tapology_browser_unavailable = True
            raise TapologyRequestError(fetch_url, detail=f"browser fallback failed: {exc}") from exc
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass


def _get_tapology_soup_with_browser(url: str, params: dict | None = None) -> BeautifulSoup:
    fetch_url = _tapology_fetch_url(url, params)
    html = _get_tapology_html_with_browser(fetch_url)
    return BeautifulSoup(html, "lxml")


def _tapology_reader_available() -> bool:
    return (
        TAPOLOGY_READER_FALLBACK_ENABLED
        and bool(TAPOLOGY_READER_BASE_URL)
        and not _tapology_reader_unavailable
    )


def _tapology_reader_url(target_url: str) -> str:
    target = str(target_url or "").strip()
    if target.startswith("http://www.tapology.com/"):
        target = "https://" + target.removeprefix("http://")
    if target.startswith("//"):
        target = f"https:{target}"
    if target.startswith("/"):
        target = urljoin(TAPOLOGY_BASE_URL, target)
    return f"{TAPOLOGY_READER_BASE_URL.rstrip('/')}/{target}"


def _tapology_reader_markdown_has_mma_record(markdown: str) -> bool:
    lower = str(markdown or "").lower()
    return "pro mma record" in lower or "amateur mma record" in lower


def _tapology_reader_markdown_is_cloudflare(markdown: str) -> bool:
    text = str(markdown or "")
    lower = text.lower()
    return (
        "title: just a moment" in lower
        or "requiring captcha" in lower
        or ("cloudflare" in lower and not _tapology_reader_markdown_has_mma_record(markdown))
    )


def _tapology_reader_markdown_is_profile(markdown: str) -> bool:
    text = str(markdown or "")
    lower = text.lower()
    return (
        "mma fighter page | tapology" in lower
        and _tapology_reader_markdown_has_mma_record(markdown)
        and "fighter details" in lower
    )


def _get_tapology_markdown_with_reader(fighter_url: str) -> str:
    """Fetch a Tapology fighter page through a markdown reader service."""
    global _tapology_reader_unavailable
    if not _tapology_reader_available():
        raise TapologyRequestError(fighter_url, detail="reader fallback disabled")

    normalized_url = _tapology_reader_url(fighter_url).removeprefix(
        f"{TAPOLOGY_READER_BASE_URL.rstrip('/')}/"
    )
    cached = _tapology_reader_markdown_cache.get(normalized_url)
    if cached:
        return cached

    reader_url = _tapology_reader_url(normalized_url)
    retry_statuses = {403, 429, 500, 502, 503, 504}
    last_error: TapologyRequestError | None = None
    for attempt in range(1, TAPOLOGY_READER_MAX_RETRIES + 1):
        try:
            # The reader service can reject browser-shaped Tapology headers even
            # when the target profile is available. Use requests' default
            # headers here; HEADERS is for origin/source requests.
            response = requests.get(
                reader_url,
                timeout=TAPOLOGY_READER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            last_error = TapologyRequestError(
                fighter_url,
                detail=f"reader fallback request failed: {exc}",
            )
            if attempt < TAPOLOGY_READER_MAX_RETRIES:
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error from exc

        try:
            response.raise_for_status()
        except Exception as exc:
            if response.status_code == 401:
                _tapology_reader_unavailable = True
            last_error = TapologyRequestError(
                fighter_url,
                status_code=response.status_code,
                detail="reader fallback unavailable",
            )
            if response.status_code in retry_statuses and attempt < TAPOLOGY_READER_MAX_RETRIES:
                logger.info(
                    "Tapology reader fallback returned status %s for %s "
                    "(attempt %d/%d); retrying",
                    response.status_code,
                    normalized_url,
                    attempt,
                    TAPOLOGY_READER_MAX_RETRIES,
                )
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error from exc

        markdown = response.text or ""
        invalid_detail = ""
        invalid_status: int | None = None
        if not markdown.strip():
            invalid_detail = "reader fallback returned empty response"
        elif _tapology_reader_markdown_is_cloudflare(markdown):
            invalid_detail = "reader fallback returned Cloudflare challenge"
            invalid_status = 403
        elif not _tapology_reader_markdown_is_profile(markdown):
            invalid_detail = "reader fallback returned non-profile Tapology content"

        if invalid_detail:
            last_error = TapologyRequestError(
                fighter_url,
                status_code=invalid_status,
                detail=invalid_detail,
            )
            if attempt < TAPOLOGY_READER_MAX_RETRIES:
                logger.info(
                    "Tapology reader fallback returned invalid markdown for %s: %s "
                    "(attempt %d/%d); retrying",
                    normalized_url,
                    invalid_detail,
                    attempt,
                    TAPOLOGY_READER_MAX_RETRIES,
                )
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error

        logger.info("Tapology reader fallback fetched %s", normalized_url)
        _tapology_reader_markdown_cache[normalized_url] = markdown
        return markdown

    raise last_error or TapologyRequestError(
        fighter_url,
        detail="reader fallback unavailable",
    )


def _tapology_reader_markdown_is_search(markdown: str) -> bool:
    lower = str(markdown or "").lower()
    return (
        "search fighters, bouts & events | tapology" in lower
        and "search results" in lower
    )


def _get_tapology_search_markdown_with_reader(search_url: str) -> str:
    """Fetch a Tapology search page through the configured markdown reader."""
    global _tapology_reader_unavailable
    if not _tapology_reader_available():
        raise TapologyRequestError(search_url, detail="reader fallback disabled")

    normalized_url = _tapology_reader_url(search_url).removeprefix(
        f"{TAPOLOGY_READER_BASE_URL.rstrip('/')}/"
    )
    cached = _tapology_reader_markdown_cache.get(normalized_url)
    if cached:
        return cached

    reader_url = _tapology_reader_url(normalized_url)
    retry_statuses = {403, 429, 500, 502, 503, 504}
    last_error: TapologyRequestError | None = None
    for attempt in range(1, TAPOLOGY_READER_MAX_RETRIES + 1):
        try:
            response = requests.get(
                reader_url,
                timeout=TAPOLOGY_READER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            last_error = TapologyRequestError(
                search_url,
                detail=f"reader search request failed: {exc}",
            )
            if attempt < TAPOLOGY_READER_MAX_RETRIES:
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error from exc

        try:
            response.raise_for_status()
        except Exception as exc:
            if response.status_code == 401:
                _tapology_reader_unavailable = True
            last_error = TapologyRequestError(
                search_url,
                status_code=response.status_code,
                detail="reader search unavailable",
            )
            if response.status_code in retry_statuses and attempt < TAPOLOGY_READER_MAX_RETRIES:
                logger.info(
                    "Tapology reader search returned status %s for %s "
                    "(attempt %d/%d); retrying",
                    response.status_code,
                    normalized_url,
                    attempt,
                    TAPOLOGY_READER_MAX_RETRIES,
                )
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error from exc

        markdown = response.text or ""
        invalid_detail = ""
        invalid_status: int | None = None
        if not markdown.strip():
            invalid_detail = "reader search returned empty response"
        elif _tapology_reader_markdown_is_cloudflare(markdown):
            invalid_detail = "reader search returned Cloudflare challenge"
            invalid_status = 403
        elif not _tapology_reader_markdown_is_search(markdown):
            invalid_detail = "reader search returned non-search Tapology content"

        if invalid_detail:
            last_error = TapologyRequestError(
                search_url,
                status_code=invalid_status,
                detail=invalid_detail,
            )
            if attempt < TAPOLOGY_READER_MAX_RETRIES:
                logger.info(
                    "Tapology reader search returned invalid markdown for %s: %s "
                    "(attempt %d/%d); retrying",
                    normalized_url,
                    invalid_detail,
                    attempt,
                    TAPOLOGY_READER_MAX_RETRIES,
                )
                time.sleep(min(REQUEST_DELAY * attempt, 5.0))
                continue
            raise last_error

        logger.info("Tapology reader search fetched %s", normalized_url)
        _tapology_reader_markdown_cache[normalized_url] = markdown
        return markdown

    raise last_error or TapologyRequestError(
        search_url,
        detail="reader search unavailable",
    )


def _search_tapology_candidates_with_reader(
    fighter_name: str,
    query: str,
    *,
    scored_urls: dict[str, int],
) -> bool:
    """Use the existing Tapology reader fallback to discover search-result URLs."""
    if not _tapology_reader_available():
        return False

    search_url = _tapology_fetch_url(TAPOLOGY_SEARCH_URL, {"term": query})
    try:
        markdown = _get_tapology_search_markdown_with_reader(search_url)
    except TapologyRequestError as exc:
        _log_external_source_error_once(
            "Tapology",
            "reader search failed",
            f"{search_url}: {exc}",
            level=logging.WARNING,
        )
        return True

    for line in _tapology_reader_lines(markdown):
        for text, url, _title in _tapology_reader_links(line):
            if "/fightcenter/fighters/" not in url:
                continue
            full_url = urljoin(TAPOLOGY_BASE_URL, url)
            _score_site_search_result(scored_urls, query, full_url, text)
    return True


def _tapology_markdown_to_plain(markdown: str) -> str:
    text = str(markdown or "")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r" \1 ", text)
    text = text.replace("\\_", "_")
    text = text.replace("**", " ")
    text = text.replace("####", " ")
    text = text.replace("#", " ")
    text = re.sub(r"\s*\|\s*", " | ", text)
    return _clean_text(text)


def _tapology_markdown_field(plain_text: str, label: str, stop_labels: tuple[str, ...]) -> str:
    label_pattern = re.escape(label)
    stop_pattern = "|".join(re.escape(stop_label) for stop_label in stop_labels)
    pattern = rf"{label_pattern}\s*:\s*(.*?)(?=\s*(?:{stop_pattern})\s*:|\s*\|\s*|$)"
    match = re.search(pattern, plain_text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = _clean_text(match.group(1))
    return "" if value.upper() in {"N/A", "NA", "??", "--"} else value


def _parse_tapology_reader_title_name(markdown: str) -> str:
    match = re.search(
        r"Title:\s*(.*?)\s*(?:\(\".*?\"\))?\s*\|\s*MMA Fighter Page \|\s*Tapology",
        str(markdown or ""),
        flags=re.IGNORECASE,
    )
    return _clean_text(match.group(1)) if match else ""


def _parse_tapology_reader_profile(fighter_url: str, markdown: str) -> dict:
    plain_text = _tapology_markdown_to_plain(markdown)
    title_name = _parse_tapology_reader_title_name(markdown)
    name = _tapology_markdown_field(
        plain_text,
        "Name",
        (
            "Nickname",
            "Pro MMA Record",
            "Amateur MMA Record",
            "Current MMA Streak",
            "Age & Date of Birth",
            "Height",
        ),
    ) or title_name

    record = ""
    record_match = re.search(r"Pro MMA Record\s*:?\s*([0-9]+-[0-9]+-[0-9]+)", plain_text)
    if record_match:
        record = record_match.group(1)

    height_raw = _tapology_markdown_field(
        plain_text,
        "Height",
        ("Reach", "Weight Class", "Last Weigh-In", "Affiliation"),
    )
    reach_raw = _tapology_markdown_field(
        plain_text,
        "Reach",
        ("Weight Class", "Last Weigh-In", "Affiliation"),
    )
    dob_field = _tapology_markdown_field(
        plain_text,
        "Age & Date of Birth",
        ("Height", "Reach", "Weight Class", "Last Weigh-In"),
    )
    dob = ""
    for pattern in (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}\b",
    ):
        dob_match = re.search(pattern, dob_field)
        if dob_match:
            dob = dob_match.group(0)
            break

    weight_raw = _tapology_markdown_field(
        plain_text,
        "Last Weigh-In",
        ("Affiliation", "Last Fight", "Career Disclosed Earnings"),
    )
    if not weight_raw:
        top_weight_match = re.search(
            r"\bWeight\s+([0-9]+(?:\.[0-9]+)?)\s+(?:[A-Z][a-z]+weight|lbs?)\b",
            plain_text,
        )
        if top_weight_match:
            weight_raw = f"{top_weight_match.group(1)} lbs"
    elif "lbs" not in weight_raw.lower():
        weight_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", weight_raw)
        weight_raw = f"{weight_match.group(1)} lbs" if weight_match else ""

    wins = losses = draws = 0
    if record:
        try:
            wins, losses, draws = [int(part) for part in record.split("-")[:3]]
        except ValueError:
            wins = losses = draws = 0

    age_raw = dob_field
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)
    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": "",
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
        "summary_text": plain_text[:1500],
        **_empty_profile_stats(),
    }


def _response_text_is_empty(resp: object) -> bool:
    if not hasattr(resp, "text"):
        return False
    return not str(getattr(resp, "text", "") or "").strip()


def _profile_text_present(value: object) -> bool:
    text = _clean_text(value)
    return bool(text) and text.casefold() not in {"--", "n/a", "na", "??", "nan", "none"}


def _profile_numeric_present(value: object) -> bool:
    try:
        return not np.isnan(_safe_float(value))
    except Exception:
        return False


def _profile_group_has_value(profile: dict, group: str) -> bool:
    if group == "record":
        return _profile_text_present(profile.get("record"))
    if group in {"height", "reach", "weight"}:
        return _profile_numeric_present(profile.get(group)) or _profile_text_present(
            profile.get(f"{group}_raw")
        )
    if group == "stance":
        return _clean_text(profile.get("stance")).casefold() in _CANONICAL_STANCE_LABELS
    if group == "dob":
        return _profile_text_present(profile.get("dob")) or _profile_numeric_present(profile.get("age"))
    return any(_profile_text_present(profile.get(field)) for field in _PROFILE_MERGE_GROUPS.get(group, ()))


def _merge_missing_profile_fields(target: dict, source: dict) -> list[str]:
    """Fill missing profile field groups without overwriting observed values."""
    filled_groups: list[str] = []
    if not _profile_text_present(target.get("name")) and _profile_text_present(source.get("name")):
        target["name"] = source.get("name")
    if not _profile_text_present(target.get("fighter_url")) and _profile_text_present(source.get("fighter_url")):
        target["fighter_url"] = source.get("fighter_url")

    for group, fields in _PROFILE_MERGE_GROUPS.items():
        if _profile_group_has_value(target, group) or not _profile_group_has_value(source, group):
            continue
        for field in fields:
            target[field] = source.get(field)
        filled_groups.append(group)
    return filled_groups


def _profile_has_core_static_fields(profile: dict) -> bool:
    return all(
        _profile_group_has_value(profile, group)
        for group in ("record", "height", "reach", "weight", "stance", "dob")
    )


def _source_name_from_url(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower()
    if "tapology.com" in host:
        return "Tapology"
    if "sherdog.com" in host:
        return "Sherdog"
    if "fightdx.com" in host:
        return "FightDX"
    if "martialbot.com" in host:
        return "MartialBot"
    if "brave.com" in host:
        return "Brave site search"
    return host or "external source"


def _log_external_source_error_once(
    source: str,
    issue: str,
    detail: object = "",
    *,
    level: int = logging.ERROR,
) -> None:
    """Emit one alert per source/issue so external outages are visible."""
    source_label = _clean_text(source) or "external source"
    issue_label = _clean_text(issue) or "unavailable"
    key = (source_label, issue_label)
    if key in _external_source_alert_keys:
        return
    _external_source_alert_keys.add(key)

    detail_text = _clean_text(detail)
    if detail_text:
        logger.log(
            level,
            "External data source unavailable: %s - %s: %s",
            source_label,
            issue_label,
            detail_text,
        )
    else:
        logger.log(
            level,
            "External data source unavailable: %s - %s",
            source_label,
            issue_label,
        )


class TapologyRequestError(RuntimeError):
    def __init__(
        self,
        url: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.detail = detail
        message = f"Tapology request failed for {url}"
        if status_code is not None:
            message += f" (status {status_code})"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _tapology_cloudflare_issue(url: str) -> str:
    path = urlparse(str(url or "")).path
    if "/fightcenter/fighters/" in path:
        return "profile pages blocked by Cloudflare"
    if path == "/search":
        return "native search blocked by Cloudflare"
    return "blocked by Cloudflare"


def _mark_tapology_cloudflare_blocked(url: str) -> None:
    """Cache that this runtime cannot access Tapology without a proxy."""
    global _tapology_blocked
    _tapology_blocked = True
    if _tapology_browser_cloudflare_blocked:
        detail = (
            f"{url}; browser fallback is also Cloudflare-blocked. "
            "Set TAPOLOGY_PROXY_URL to an egress accepted by Tapology if this "
            "runtime needs Tapology profile access."
        )
        level = logging.WARNING
    else:
        detail = (
            f"{url}; enable TAPOLOGY_BROWSER_FALLBACK_ENABLED with Chromium/Xvfb "
            "or set TAPOLOGY_PROXY_URL if this runtime needs Tapology profile access"
        )
        level = logging.ERROR
    _log_external_source_error_once(
        "Tapology",
        _tapology_cloudflare_issue(url),
        detail,
        level=level,
    )


def _tapology_browser_error_is_cloudflare_block(exc: TapologyRequestError) -> bool:
    detail = str(getattr(exc, "detail", "") or "")
    detail_lower = detail.lower()
    return (
        exc.status_code == 403
        and "browser fallback" in detail_lower
        and "cloudflare" in detail_lower
    )


def _mark_tapology_browser_cloudflare_blocked(url: str) -> None:
    """Cache that the hosted browser path is also Cloudflare-blocked."""
    global _tapology_browser_cloudflare_blocked
    _tapology_browser_cloudflare_blocked = True
    _log_external_source_error_once(
        "Tapology",
        "browser fallback blocked by Cloudflare",
        (
            f"{url}; Tapology browser recovery will be skipped for this process "
            "unless TAPOLOGY_PROXY_URL uses an egress accepted by Tapology"
        ),
        level=logging.WARNING,
    )


def _get_soup(url: str, *, max_retries: int = 2) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup with retry on timeout."""
    source = _source_name_from_url(url)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            if _response_text_is_empty(resp):
                _log_external_source_error_once(source, "empty response body", url)
                raise RuntimeError(f"{source} returned an empty response for {url}")
            time.sleep(REQUEST_DELAY)
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < max_retries:
                backoff = REQUEST_DELAY * attempt
                logger.debug("Request to %s timed out (attempt %d/%d); retrying in %.1fs", url, attempt, max_retries, backoff)
                time.sleep(backoff)
        except requests.exceptions.RequestException as exc:
            _log_external_source_error_once(source, "request failed", f"{url}: {exc}")
            raise
    _log_external_source_error_once(source, "request timed out", f"{url}: {last_exc}")
    raise last_exc  # type: ignore[misc]


def _tapology_browser_profiles() -> list[dict[str, object]]:
    import platform as _plat

    primary_platform = "linux" if _plat.system().lower() == "linux" else "windows"
    platforms = [primary_platform]
    platforms.extend(platform for platform in ("linux", "windows") if platform != primary_platform)
    return [
        {"browser": "chrome", "platform": platform, "mobile": False}
        for platform in platforms
    ]


def _tapology_user_agent(browser_profile: dict[str, object]) -> str:
    if browser_profile.get("platform") == "linux":
        return (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )


def _build_tapology_scraper():
    if cloudscraper is None:
        raise RuntimeError("Tapology scraping requires the optional 'cloudscraper' dependency")

    profiles = _tapology_browser_profiles()
    browser_profile = profiles[min(_tapology_scraper_profile_index, len(profiles) - 1)]
    scraper = cloudscraper.create_scraper(
        browser=dict(browser_profile)
    )
    scraper.headers.update(
        {
            "User-Agent": _tapology_user_agent(browser_profile),
            "Accept": HEADERS["Accept"],
            "Accept-Language": HEADERS["Accept-Language"],
        }
    )
    proxies = _tapology_proxies()
    if proxies:
        scraper.proxies.update(proxies)
        logger.info("Tapology proxy enabled: %s", _proxy_target(TAPOLOGY_PROXY_URL))
    return scraper


def _switch_tapology_scraper_profile() -> bool:
    """Move to the next browser profile after a Cloudflare challenge."""
    global _tapology_scraper, _tapology_scraper_profile_index
    profiles = _tapology_browser_profiles()
    if _tapology_scraper_profile_index + 1 >= len(profiles):
        return False
    _tapology_scraper_profile_index += 1
    _tapology_scraper = None
    profile = profiles[_tapology_scraper_profile_index]
    logger.info(
        "Retrying Tapology with alternate browser profile: %s/%s",
        profile.get("browser"),
        profile.get("platform"),
    )
    return True


def _check_tapology_blocked() -> bool:
    """Return cached block state without making a broad Tapology probe.

    Tapology currently allows the search endpoint while Cloudflare-challenging
    fighter profile pages from some egress IPs. Probing /fightcenter up front
    incorrectly disables reachable search and site-candidate recovery.
    """
    return _tapology_blocked is True


def _get_tapology_soup(
    url: str,
    *,
    params: dict | None = None,
    max_retries: int | None = None,
    retry_statuses: set[int] | None = None,
) -> BeautifulSoup:
    """Fetch a Tapology page with challenge-aware retries."""
    global _tapology_scraper, _last_tapology_request_at, _tapology_blocked

    def _try_browser_fallback(reason: str) -> BeautifulSoup | None:
        if not _tapology_browser_fallback_available():
            return None
        try:
            logger.info(
                "Tapology %s for %s; retrying through hosted browser fallback",
                reason,
                _tapology_fetch_url(url, params),
            )
            return _get_tapology_soup_with_browser(url, params)
        except TapologyRequestError as browser_exc:
            browser_cloudflare_blocked = _tapology_browser_error_is_cloudflare_block(browser_exc)
            alert_level = logging.WARNING if browser_cloudflare_blocked else logging.ERROR
            if browser_cloudflare_blocked:
                _mark_tapology_browser_cloudflare_blocked(_tapology_fetch_url(url, params))
            _log_external_source_error_once(
                "Tapology",
                "browser fallback failed",
                f"{_tapology_fetch_url(url, params)}: {browser_exc}",
                level=alert_level,
            )
            return None

    if _tapology_blocked is True:
        browser_soup = _try_browser_fallback("requests path is blocked")
        if browser_soup is not None:
            return browser_soup
        raise TapologyRequestError(
            url,
            status_code=403,
            detail="Tapology blocked from this environment",
        )

    max_attempts = max(1, int(max_retries or TAPOLOGY_MAX_RETRIES))
    retry_statuses = set(retry_statuses or {403, 429, 503})
    last_error: Exception | None = None
    last_status: int | None = None
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        if _tapology_scraper is None:
            try:
                _tapology_scraper = _build_tapology_scraper()
            except Exception as exc:
                browser_soup = _try_browser_fallback(f"requests session unavailable ({exc})")
                if browser_soup is not None:
                    return browser_soup
                raise

        sleep_for = TAPOLOGY_REQUEST_DELAY - (time.monotonic() - _last_tapology_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            resp = _tapology_scraper.get(
                url,
                params=params,
                timeout=TAPOLOGY_TIMEOUT_SECONDS,
                proxies=_tapology_proxies(),
            )
            _last_tapology_request_at = time.monotonic()
        except Exception as exc:  # pragma: no cover - network-only branch
            last_error = exc
            if attempt >= max_attempts:
                break
            backoff = TAPOLOGY_REQUEST_DELAY * attempt
            logger.warning(
                "Tapology request to %s failed (%s); retrying in %.1fs",
                url,
                exc,
                backoff,
            )
            _tapology_scraper = _build_tapology_scraper() if cloudscraper is not None else None
            time.sleep(backoff)
            continue

        if resp.status_code == 200 and resp.text:
            if _is_cloudflare_challenge(resp):
                detail = _tapology_error_detail(resp)
                last_status = int(resp.status_code)
                last_error = TapologyRequestError(
                    url,
                    status_code=resp.status_code,
                    detail=detail,
                )
                if detail == "Cloudflare challenge":
                    if _switch_tapology_scraper_profile():
                        max_attempts += 1
                        logger.info(
                            "Tapology request to %s hit a Cloudflare challenge; retrying with alternate browser profile",
                            url,
                        )
                        continue
                    browser_soup = _try_browser_fallback("profile/search page hit Cloudflare")
                    if browser_soup is not None:
                        return browser_soup
                    _mark_tapology_cloudflare_blocked(url)
                    raise last_error
                if attempt >= max_attempts:
                    break
                backoff = TAPOLOGY_REQUEST_DELAY * (2 ** attempt)
                logger.info(
                    "Tapology request to %s returned a Cloudflare challenge with status %s "
                    "(attempt %d/%d); rebuilding session and retrying in %.1fs",
                    url,
                    resp.status_code,
                    attempt,
                    max_attempts,
                    backoff,
                )
                _tapology_scraper = _build_tapology_scraper()
                time.sleep(backoff)
                continue
            return BeautifulSoup(resp.text, "lxml")

        last_status = int(resp.status_code)
        if resp.status_code in retry_statuses:
            detail = _tapology_error_detail(resp)
            last_error = TapologyRequestError(
                url,
                status_code=resp.status_code,
                detail=detail,
            )
            if detail == "Cloudflare challenge":
                if _switch_tapology_scraper_profile():
                    max_attempts += 1
                    logger.info(
                        "Tapology request to %s hit a Cloudflare challenge; retrying with alternate browser profile",
                        url,
                    )
                    continue
                browser_soup = _try_browser_fallback("request returned Cloudflare")
                if browser_soup is not None:
                    return browser_soup
                _mark_tapology_cloudflare_blocked(url)
                raise last_error
            if attempt >= max_attempts:
                break
            backoff = TAPOLOGY_REQUEST_DELAY * (2 ** attempt)  # exponential: 6, 12, 24, 48s
            logger.log(
                logging.INFO if detail == "Cloudflare challenge" else logging.WARNING,
                "Tapology request to %s returned %s%s (attempt %d/%d); "
                "rebuilding session and retrying in %.1fs",
                url,
                resp.status_code,
                f" ({detail})" if detail else "",
                attempt,
                max_attempts,
                backoff,
            )
            _tapology_scraper = _build_tapology_scraper()
            time.sleep(backoff)
            continue

        try:
            resp.raise_for_status()
        except Exception as exc:
            detail = _tapology_error_detail(resp)
            if detail == "Cloudflare challenge":
                if _switch_tapology_scraper_profile():
                    max_attempts += 1
                    logger.info(
                        "Tapology request to %s hit a Cloudflare challenge; retrying with alternate browser profile",
                        url,
                    )
                    continue
                browser_soup = _try_browser_fallback("request raised Cloudflare")
                if browser_soup is not None:
                    return browser_soup
                _mark_tapology_cloudflare_blocked(url)
                raise TapologyRequestError(
                    url,
                    status_code=resp.status_code,
                    detail=detail,
                ) from exc
            issue = f"request returned status {resp.status_code}"
            if detail:
                issue = f"{issue} ({detail})"
            _log_external_source_error_once("Tapology", issue, f"{url}: {exc}")
            raise TapologyRequestError(
                url,
                status_code=resp.status_code,
                detail=detail,
            ) from exc

        _log_external_source_error_once("Tapology", "empty response body", url)
        raise TapologyRequestError(url, status_code=resp.status_code, detail="empty response body")

    detail = last_error.detail if isinstance(last_error, TapologyRequestError) else ""
    if detail == "Cloudflare challenge":
        browser_soup = _try_browser_fallback("requests retries exhausted on Cloudflare")
        if browser_soup is not None:
            return browser_soup
        _mark_tapology_cloudflare_blocked(url)
        raise TapologyRequestError(url, status_code=last_status, detail=detail) from last_error

    issue = f"request returned status {last_status}" if last_status is not None else "request failed"
    if detail:
        issue = f"{issue} ({detail})"
    _log_external_source_error_once("Tapology", issue, f"{url}: {last_error}")
    raise TapologyRequestError(url, status_code=last_status, detail=detail) from last_error


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _titleize_slug(value: str) -> str:
    text = re.sub(r"^\d+-", "", str(value or "").strip().lower())
    text = text.replace("-", " ").strip()
    tokens = text.split()
    if len(tokens) > 2 and tokens[-1].isalpha() and len(tokens[-1]) <= 4:
        tokens = tokens[:-1]
    text = " ".join(tokens)
    text = re.sub(r"\bm 1\b", "m-1", text)
    replacements = {
        "lfa": "LFA",
        "ufc": "UFC",
        "kotc": "KOTC",
        "cffc": "CFFC",
        "cwfc": "CWFC",
        "ec": "EC",
    }
    tokens = []
    for token in text.split():
        tokens.append(replacements.get(token, token.capitalize()))
    return " ".join(tokens)


def _slugify_person_name(name: str) -> str:
    return normalize_person_name(name).replace(" ", "-")


def _split_camel_token(token: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(token or ""))


def _name_query_variants(fighter_name: str) -> list[str]:
    tokens = str(fighter_name or "").strip().split()
    if not tokens:
        return []

    variants = [fighter_name]
    variants.extend(_MANUAL_SEARCH_ALIASES.get(normalize_person_name(fighter_name), []))

    first_token = tokens[0]
    if len(first_token) == 2 and first_token.isalpha():
        dotted_initials = ".".join(first_token.upper()) + "."
        variants.append(f"{dotted_initials} {' '.join(tokens[1:])}".strip())

    spaced_name = " ".join(_split_camel_token(token) for token in tokens).strip()
    if spaced_name and spaced_name != fighter_name:
        variants.append(spaced_name)

    saint_forms = list(dict.fromkeys(variants))
    for variant in saint_forms:
        if re.search(r"\bsaint[\s-]", variant, flags=re.IGNORECASE):
            variants.append(re.sub(r"\bsaint[\s-]", "St ", variant, flags=re.IGNORECASE))
            variants.append(re.sub(r"\bsaint[\s-]", "St. ", variant, flags=re.IGNORECASE))
            variants.append(re.sub(r"\bsaint[\s-]", "St-", variant, flags=re.IGNORECASE))
        if re.search(r"\bst\.?[\s-]", variant, flags=re.IGNORECASE):
            variants.append(re.sub(r"\bst\.?[\s-]", "Saint ", variant, flags=re.IGNORECASE))

    if re.search(r"\bjunior\b", fighter_name, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bjunior\b", "Jr.", fighter_name, flags=re.IGNORECASE))
    if re.search(r"\bjr\.?\b", fighter_name, flags=re.IGNORECASE):
        variants.append(re.sub(r"\bjr\.?\b", "Junior", fighter_name, flags=re.IGNORECASE))

    return list(dict.fromkeys(variant.strip() for variant in variants if str(variant).strip()))


def _strip_name_nicknames(value: str) -> str:
    text = str(value or "")
    text = re.sub(r'"[^"]+"', " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return _clean_text(text)


def _name_variants(value: str, href: str = "") -> set[str]:
    variants: set[str] = set()
    text = _clean_text(str(value or ""))
    if text:
        variants.add(text)
        stripped = _strip_name_nicknames(text)
        if stripped:
            variants.add(stripped)

    slug = str(href or "").rstrip("/").split("/")[-1]
    slug = re.sub(r"^\d+-", "", slug)
    slug_text = _clean_text(slug.replace("-", " "))
    if slug_text:
        variants.add(slug_text)
        stripped_slug = _strip_name_nicknames(slug_text)
        if stripped_slug:
            variants.add(stripped_slug)

    return {variant for variant in variants if variant}


def _sleep_after_request(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _extract_site_search_result_url(href: str) -> str:
    text = str(href or "").strip()
    if not text:
        return ""

    if text.startswith("//"):
        text = f"https:{text}"

    parsed = urlparse(text)
    if parsed.netloc.endswith("google.com") and parsed.path == "/url":
        target = parse_qs(parsed.query).get("q", [""])[0]
        return unquote(target).strip()
    if (not parsed.netloc or parsed.netloc.endswith("duckduckgo.com")) and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target).strip()

    return text


def _score_site_search_result(
    scored_urls: dict[str, int],
    fighter_name: str,
    candidate_url: str,
    candidate_text: str = "",
) -> None:
    actual_url = _extract_site_search_result_url(candidate_url)
    if not actual_url:
        return

    candidate_name = _clean_text(candidate_text)
    score = _best_name_score(fighter_name, candidate_name, actual_url)
    if score <= 0:
        return
    scored_urls[actual_url] = max(scored_urls.get(actual_url, 0), score)


def _search_site_candidates_with_brave_api(
    fighter_name: str,
    query: str,
    *,
    site_query: str,
    required_path_fragment: str,
    scored_urls: dict[str, int],
) -> bool:
    """Use Brave's official API when configured; return True when attempted."""
    global _site_search_disabled
    if not BRAVE_SEARCH_API_KEY:
        return False

    resp = requests.get(
        BRAVE_SEARCH_API_URL,
        params={
            "q": f'site:{site_query} "{query}"',
            "count": 10,
            "search_lang": "en",
        },
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
        },
        timeout=BRAVE_SEARCH_TIMEOUT_SECONDS,
    )
    if resp.status_code in {401, 403, 422, 429}:
        _log_external_source_error_once(
            "Brave site search",
            "API unavailable",
            f"{site_query} for {fighter_name} returned status {resp.status_code}",
        )
        _site_search_disabled = True
        return True
    resp.raise_for_status()

    payload = resp.json()
    results = ((payload or {}).get("web") or {}).get("results") or []
    if not isinstance(results, list):
        _log_external_source_error_once(
            "Brave site search",
            "API returned unexpected response",
            f"{site_query} for {query}",
        )
        return True

    for result in results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        if required_path_fragment not in url:
            continue
        text = " ".join(
            part
            for part in (
                str(result.get("title") or ""),
                str(result.get("description") or ""),
            )
            if part
        )
        _score_site_search_result(scored_urls, fighter_name, url, text)
    return True


def _search_site_candidates_with_duckduckgo_html(
    fighter_name: str,
    query: str,
    *,
    site_query: str,
    required_path_fragment: str,
    scored_urls: dict[str, int],
) -> bool:
    """Use DuckDuckGo's HTML results as a no-key site-search fallback."""
    global _duckduckgo_site_search_disabled
    if not DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED or _duckduckgo_site_search_disabled:
        return False

    response = requests.get(
        DUCKDUCKGO_SEARCH_HTML_URL,
        params={"q": f"{query} {site_query}"},
        headers=HEADERS,
        timeout=BRAVE_SEARCH_TIMEOUT_SECONDS,
    )
    if response.status_code in {403, 429}:
        _log_external_source_error_once(
            "DuckDuckGo site search",
            "HTML search blocked or rate limited",
            f"{site_query} for {fighter_name} returned status {response.status_code}",
            level=logging.WARNING,
        )
        _duckduckgo_site_search_disabled = True
        return True
    if response.status_code == 202 or _duckduckgo_response_is_challenge(response.text):
        _log_external_source_error_once(
            "DuckDuckGo site search",
            "HTML search challenge",
            f"{site_query} for {fighter_name} returned an anti-bot challenge",
            level=logging.WARNING,
        )
        _duckduckgo_site_search_disabled = True
        return True
    response.raise_for_status()
    if _response_text_is_empty(response):
        _log_external_source_error_once(
            "DuckDuckGo site search",
            "empty response body",
            f"{site_query} for {query}",
        )
        return True

    soup = BeautifulSoup(response.text, "lxml")
    for link in soup.find_all("a", href=True):
        actual_url = _extract_site_search_result_url(link.get("href", ""))
        if not actual_url or required_path_fragment not in actual_url:
            continue
        _score_site_search_result(
            scored_urls,
            fighter_name,
            actual_url,
            link.get_text(" ", strip=True),
        )
    return True


def _duckduckgo_response_is_challenge(text: str) -> bool:
    haystack = str(text or "").lower()
    return (
        "duckduckgo.com/anomaly.js" in haystack
        or "unfortunately, bots use duckduckgo too" in haystack
        or "challenge-form" in haystack
    )


def _search_site_candidates(
    fighter_name: str,
    *,
    site_query: str,
    required_path_fragment: str,
) -> list[tuple[str, int]]:
    global _site_search_disabled
    if _site_search_disabled:
        return []

    scored_urls: dict[str, int] = {}
    query_variants = _name_query_variants(fighter_name)[:2]
    if not query_variants:
        return []

    for query in query_variants:
        try:
            attempted_api = _search_site_candidates_with_brave_api(
                fighter_name,
                query,
                site_query=site_query,
                required_path_fragment=required_path_fragment,
                scored_urls=scored_urls,
            )
            if attempted_api:
                _sleep_after_request(REQUEST_DELAY)
                if _site_search_disabled:
                    break
                continue
            if not BRAVE_SEARCH_HTML_FALLBACK_ENABLED:
                attempted_duckduckgo = _search_site_candidates_with_duckduckgo_html(
                    fighter_name,
                    query,
                    site_query=site_query,
                    required_path_fragment=required_path_fragment,
                    scored_urls=scored_urls,
                )
                if attempted_duckduckgo:
                    _sleep_after_request(REQUEST_DELAY)
                    continue
                logger.debug(
                    "Skipping site-search fallback for '%s' on %s because no search fallback is configured",
                    fighter_name,
                    site_query,
                )
                _site_search_disabled = True
                break
            resp = requests.get(
                BRAVE_SEARCH_HTML_URL,
                params={"q": f'site:{site_query} "{query}"'},
                headers=HEADERS,
                timeout=BRAVE_SEARCH_TIMEOUT_SECONDS,
            )
            if resp.status_code in {403, 429}:
                _log_external_source_error_once(
                    "Brave site search",
                    "HTML search blocked or rate limited",
                    f"{site_query} for {fighter_name} returned status {resp.status_code}",
                    level=logging.WARNING,
                )
                logger.info(
                    "Brave HTML site search unavailable for '%s' on %s (status %s); disabling site-search fallback for this session",
                    fighter_name,
                    site_query,
                    resp.status_code,
                )
                _site_search_disabled = True
                break
            resp.raise_for_status()
            if _response_text_is_empty(resp):
                _log_external_source_error_once(
                    "Brave site search",
                    "empty response body",
                    f"{site_query} for {query}",
                )
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            _sleep_after_request(REQUEST_DELAY)
        except Exception as exc:
            _log_external_source_error_once(
                "Brave site search",
                "request failed",
                f"{site_query} for {query}: {exc}",
            )
            logger.warning(
                "Brave site search failed for '%s' on %s: %s",
                query,
                site_query,
                exc,
            )
            continue

        for link in soup.find_all("a", href=True):
            actual_url = _extract_site_search_result_url(link.get("href", ""))
            if not actual_url or required_path_fragment not in actual_url:
                continue
            _score_site_search_result(
                scored_urls,
                fighter_name,
                actual_url,
                link.get_text(" ", strip=True),
            )
        if not scored_urls:
            attempted_duckduckgo = _search_site_candidates_with_duckduckgo_html(
                fighter_name,
                query,
                site_query=site_query,
                required_path_fragment=required_path_fragment,
                scored_urls=scored_urls,
            )
            if attempted_duckduckgo:
                _sleep_after_request(REQUEST_DELAY)

    return sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)


def _safe_float(value, default=np.nan) -> float:
    if value is None or value == "" or value == "--" or value == "N/A":
        return default
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _inches_to_cm(value: float) -> float:
    if value is None or np.isnan(value):
        return np.nan
    return round(float(value) * 2.54, 2)


def _parse_height_cm(raw: str) -> float:
    """Parse height from any common format to centimeters.

    Handles: "193 cm", "175cm", "5'10\"", "5'10\" (177.8 cm)", "5' 10", etc.
    An explicit cm value is checked first so strings like "193 cm" are not
    misread as 19'3" by the feet'inches heuristic.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    # Try direct cm value first (avoids feet'inches misparsing of "193 cm")
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", raw)
    if cm_match:
        return round(float(cm_match.group(1)), 2)
    # Try feet'inches
    match = re.search(r"(\d+)'?\s*(\d+)", raw)
    if match:
        inches = int(match.group(1)) * 12 + int(match.group(2))
        return _inches_to_cm(inches)
    return np.nan


def _parse_reach_cm(raw: str) -> float:
    """Parse reach from any common format to centimeters.

    Handles: "74\"", "74", "74 (in)", "188 cm", "1.85m", etc.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    # Try cm first (longer number likely cm)
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", raw)
    if cm_match:
        return round(float(cm_match.group(1)), 2)
    # Try meters (e.g. "1.85m")
    m_match = re.search(r"(\d+\.\d+)\s*m\b", raw)
    if m_match:
        return round(float(m_match.group(1)) * 100, 2)
    # Try inches with quote mark (e.g. 75.0")
    inch_match = re.search(r'(\d+(?:\.\d+)?)\s*["\u2033]', raw)
    if inch_match:
        return _inches_to_cm(float(inch_match.group(1)))
    # Fallback: bare number
    match = re.search(r"(\d+)", raw)
    if match:
        val = float(match.group(1))
        # Sanity: reach in inches is typically 60-85; in cm it's 150-220
        if val > 120:
            return round(val, 2)  # Already cm
        return _inches_to_cm(val)
    return np.nan


def _parse_weight_lbs(raw: str) -> float:
    """Parse weight to lbs from any common format.

    Handles: "185 lbs", "170 lbs / 77.11 kg", "185", etc.
    """
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    match = re.search(r"(\d+)", raw)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_age_from_raw(raw: str) -> float:
    """Parse a direct age value like '25' or '25 years'."""
    if not raw or raw in ("--", "N/A", "??"):
        return np.nan
    match = re.search(r"(\d+)", raw)
    if match:
        val = float(match.group(1))
        if 15 <= val <= 65:  # Sanity check for fighter age range
            return val
    return np.nan


def _parse_dob_to_age(dob_str: str) -> float:
    """Parse DOB string to age in years.

    Handles: "Sep 22, 1989", "September 22, 1989", "1989-09-22", etc.
    """
    if not dob_str or dob_str in ("--", "N/A", "-", "??"):
        return np.nan
    from datetime import datetime
    now = datetime.now()
    for fmt in ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try:
            dob = datetime.strptime(dob_str.strip(), fmt)
            age = (now - dob).days / 365.25
            return round(age, 1)
        except ValueError:
            continue
    return np.nan


def _empty_fight_dict() -> dict:
    """Return a fight dict with all per-fight stats set to NaN."""
    return {
        "kd": np.nan,
        "sig_str_landed": np.nan,
        "sig_str_attempted": np.nan,
        "td_landed": np.nan,
        "td_attempted": np.nan,
        "sub_att": np.nan,
        "rev": np.nan,
        "ctrl_seconds": np.nan,
        "opp_kd": np.nan,
        "opp_sig_str_landed": np.nan,
        "opp_sig_str_attempted": np.nan,
        "opp_td_landed": np.nan,
        "opp_td_attempted": np.nan,
        "opp_sub_att": np.nan,
        "opp_rev": np.nan,
        "opp_ctrl_seconds": np.nan,
        "detail_url": "",
    }


def _empty_profile_stats() -> dict:
    """Return career rate stats as NaN (not available on fallback sources)."""
    return {
        "slpm": np.nan,
        "str_acc": np.nan,
        "sapm": np.nan,
        "str_def": np.nan,
        "td_avg": np.nan,
        "td_acc": np.nan,
        "td_def": np.nan,
        "sub_avg": np.nan,
    }


# ---------------------------------------------------------------------------
# Sherdog scraper
# ---------------------------------------------------------------------------

def search_sherdog(fighter_name: str) -> Optional[str]:
    """
    Search Sherdog for a fighter by name. Returns their full profile URL.

    Uses the fightfinder search endpoint and fuzzy-matches the results.
    """
    if fighter_name in _sherdog_url_cache:
        return _sherdog_url_cache[fighter_name]

    if not normalize_person_name(fighter_name):
        return None

    best_url = None
    best_score = 0

    for query in _name_query_variants(fighter_name):
        try:
            search_url = f"{SHERDOG_SEARCH_URL}?SearchTxt={requests.utils.quote(query)}"
            soup = _get_soup(search_url)
        except Exception as e:
            logger.warning(f"Sherdog search failed for '{query}': {e}")
            continue

        table = soup.find("table", class_="fightfinder_result")
        candidate_links = []
        if table:
            for row in table.find_all("tr"):
                link = row.find("a", href=lambda h: h and "/fighter/" in h)
                if link:
                    candidate_links.append(link)
        else:
            candidate_links = soup.find_all("a", href=lambda h: h and "/fighter/" in h)

        for link in candidate_links:
            found_name = _clean_text(link.text)
            href = link.get("href", "")
            full_url = f"{SHERDOG_BASE_URL}{href}" if href.startswith("/") else href
            score = _best_name_score(fighter_name, found_name, href)
            if score >= 100:
                _sherdog_url_cache[fighter_name] = full_url
                return full_url

            if score > best_score:
                best_score = score
                best_url = full_url

    if best_url and best_score >= 10:
        _sherdog_url_cache[fighter_name] = best_url
        return best_url

    return None


def _parse_sherdog_profile(soup: BeautifulSoup, fighter_url: str) -> dict:
    """Parse Sherdog profile attributes from an already-fetched page."""
    name_el = soup.find("h1")
    name = _clean_text(name_el.text) if name_el else ""

    wins, losses, draws = 0, 0, 0
    for div in soup.find_all("div", class_="winloses"):
        spans = div.find_all("span")
        if len(spans) >= 2:
            label = spans[0].text.strip().lower()
            count = _safe_float(spans[1].text.strip(), default=0)
            if label == "wins":
                wins = int(count)
            elif label == "losses":
                losses = int(count)
            elif label == "draws":
                draws = int(count)

    height = np.nan
    weight = np.nan
    age = np.nan
    height_raw = ""
    weight_raw = ""
    age_raw = ""
    dob = ""

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 2:
            continue

        label = tds[0].text.strip().upper()
        value = tds[1].text.strip()

        if label == "HEIGHT":
            height_raw = value
            match = re.search(r"(\d+)'(\d+)", value)
            if match:
                height = _inches_to_cm(int(match.group(1)) * 12 + int(match.group(2)))
        elif label == "WEIGHT":
            weight_raw = value
            match = re.search(r"(\d+)\s*lbs", value)
            if match:
                weight = float(match.group(1))
        elif label == "AGE":
            age_raw = value
            match = re.search(r"(\d+)\s*/", value)
            if match:
                age = float(match.group(1))
            dob_match = re.search(r"/\s*([A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})", value)
            if dob_match:
                dob = dob_match.group(1).strip()

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": f"{wins}-{losses}-{draws}",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": height,
        "reach": np.nan,
        "weight_raw": weight_raw,
        "weight": weight,
        "stance": "",
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
        **_empty_profile_stats(),
    }


def _parse_sherdog_fight_table(table: Tag, fighter_name: str) -> list[dict]:
    fights: list[dict] = []
    rows = table.find_all("tr")

    for row in rows[1:]:
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        try:
            result_text = tds[0].text.strip().lower()
            if result_text == "win":
                won = 1
            elif result_text == "draw":
                won = 0
            else:
                won = 0

            opp_link = tds[1].find("a")
            opponent = _clean_text(opp_link.text) if opp_link else _clean_text(tds[1].text)

            event_text = tds[2].get_text(" ", strip=True)
            date_match = re.search(
                r"([A-Z][a-z]{2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", event_text
            )
            event_date = None
            if date_match:
                try:
                    event_date = datetime.strptime(
                        f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
                        "%b %d %Y",
                    )
                except ValueError:
                    pass

            method_cell = tds[3]
            method_b = method_cell.find("b")
            method = _clean_text(method_b.text) if method_b else _clean_text(method_cell.get_text())

            round_text = tds[4].text.strip()
            round_finished = int(round_text) if round_text.isdigit() else None

            event_link = tds[2].find("a")
            event_name = _clean_text(event_link.text) if event_link else _clean_text(event_text)
            event_name_lower = event_name.lower()
            is_title = "title" in event_name_lower or "championship" in event_name_lower

            fights.append(
                {
                    "result": result_text,
                    "event_date": event_date,
                    "event_name": event_name,
                    "opponent": opponent,
                    "won": won,
                    "method": method,
                    "round_finished": round_finished,
                    "is_title_bout": is_title,
                    **_empty_fight_dict(),
                }
            )
        except Exception as e:
            logger.debug(f"Sherdog: failed to parse fight row for {fighter_name}: {e}")
            continue

    fights.reverse()
    return fights


def _nearest_sherdog_section_text(table: Tag) -> str:
    """Return the closest preceding non-empty section text for a Sherdog fight table."""
    current: Tag | None = table
    while current is not None:
        for sibling in current.previous_siblings:
            if isinstance(sibling, Tag) and sibling.name == "table" and "fighter" in (sibling.get("class") or []):
                break
            if isinstance(sibling, Tag):
                text = _clean_text(sibling.get_text(" ", strip=True))
            else:
                text = _clean_text(str(sibling))
            if text:
                return text.lower()
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return ""


def _find_sherdog_amateur_table(fight_tables: list[Tag]) -> Tag | None:
    if len(fight_tables) < 2:
        return None
    amateur_table = fight_tables[1]
    return amateur_table if "amateur" in _nearest_sherdog_section_text(amateur_table) else None


def scrape_sherdog_page(fighter_url: str, fighter_name: str) -> tuple[dict, list[dict]]:
    """
    Scrape a Sherdog fighter profile page in a single request.

    Returns (profile_dict, fights_list). Profile matches UFCStats format;
    career rate stats are NaN. Fights are in chronological order (oldest first).
    """
    soup = _get_soup(fighter_url)
    profile = _parse_sherdog_profile(soup, fighter_url)

    fight_tables = soup.find_all("table", class_="fighter")
    if not fight_tables:
        return profile, []

    return profile, _parse_sherdog_fight_table(fight_tables[0], fighter_name)


def scrape_sherdog_amateur_fights(fighter_url: str, fighter_name: str) -> tuple[dict, list[dict]]:
    """Scrape the amateur Sherdog fight table when the page exposes one."""
    soup = _get_soup(fighter_url)
    profile = _parse_sherdog_profile(soup, fighter_url)
    fight_tables = soup.find_all("table", class_="fighter")
    amateur_table = _find_sherdog_amateur_table(fight_tables)
    if amateur_table is None:
        return profile, []
    return profile, _parse_sherdog_fight_table(amateur_table, fighter_name)


# ---------------------------------------------------------------------------
# Tapology scraper
# ---------------------------------------------------------------------------

def _name_score(query_key: str, candidate_key: str) -> int:
    if not query_key or not candidate_key:
        return 0
    if query_key == candidate_key:
        return 100

    query_compact = query_key.replace(" ", "")
    candidate_compact = candidate_key.replace(" ", "")
    if query_compact and query_compact == candidate_compact:
        return 95

    query_tokens = query_key.split()
    candidate_tokens = candidate_key.split()
    score = 0
    if query_tokens and candidate_tokens and query_tokens[-1] == candidate_tokens[-1]:
        score += 6
    if query_tokens and candidate_tokens and query_tokens[0] == candidate_tokens[0]:
        score += 6
    elif query_tokens and candidate_tokens and query_tokens[0][0] == candidate_tokens[0][0]:
        score += 2

    score += 2 * len(set(query_tokens) & set(candidate_tokens))

    if query_key in candidate_key or candidate_key in query_key:
        score += 8
    if query_compact and candidate_compact and (query_compact in candidate_compact or candidate_compact in query_compact):
        score += 8

    if query_tokens and len(query_tokens) <= len(candidate_tokens):
        for idx in range(len(candidate_tokens) - len(query_tokens) + 1):
            if candidate_tokens[idx:idx + len(query_tokens)] == query_tokens:
                score += 10
                break

    if len(query_tokens) == len(candidate_tokens):
        score += 3 * sum(
            1
            for query_token, candidate_token in zip(query_tokens, candidate_tokens)
            if SequenceMatcher(None, query_token, candidate_token).ratio() >= 0.8
        )

    ratio = SequenceMatcher(None, query_compact, candidate_compact).ratio()
    if ratio >= 0.97:
        score += 20
    elif ratio >= 0.92:
        score += 12
    elif ratio >= 0.85:
        score += 8
    elif ratio >= 0.75:
        score += 4

    return score


def _best_name_score(query: str, candidate_name: str, href: str = "") -> int:
    candidate_variants = _name_variants(candidate_name, href)
    for variant in candidate_variants:
        if same_person_name(query, variant):
            return 100

    query_keys = {
        normalize_person_name(query),
        normalize_cross_source_name(query),
    }
    candidate_keys = {
        key
        for variant in candidate_variants
        for key in (normalize_person_name(variant), normalize_cross_source_name(variant))
        if key
    }

    best_score = 0
    for query_key in query_keys:
        for candidate_key in candidate_keys:
            best_score = max(best_score, _name_score(query_key, candidate_key))
    return best_score


def _name_token_sets(value: str) -> list[tuple[str, ...]]:
    token_sets: list[tuple[str, ...]] = []
    for key in (normalize_person_name(value), normalize_cross_source_name(value)):
        tokens = tuple(token for token in key.split() if token)
        if tokens and tokens not in token_sets:
            token_sets.append(tokens)
    return token_sets


def _candidate_has_required_name_tokens(
    fighter_name: str,
    candidate_name: str = "",
    href: str = "",
) -> bool:
    """Require enough identity evidence before fetching/accepting candidates."""
    query_token_sets = _name_token_sets(fighter_name)
    if not query_token_sets:
        return False

    candidate_token_sets: list[tuple[str, ...]] = []
    for variant in _name_variants(candidate_name, href):
        for tokens in _name_token_sets(variant):
            if tokens not in candidate_token_sets:
                candidate_token_sets.append(tokens)
    if not candidate_token_sets:
        return False

    for query_tokens in query_token_sets:
        for candidate_tokens in candidate_token_sets:
            if query_tokens == candidate_tokens or query_tokens == tuple(reversed(candidate_tokens)):
                return True
            if len(query_tokens) == 1 and query_tokens[0] in candidate_tokens:
                return True
            if len(query_tokens) >= 2 and query_tokens[0] in candidate_tokens and query_tokens[-1] in candidate_tokens:
                return True
    return False


def _fightdx_person_url_from_href(href: str) -> str:
    url = urljoin(f"{FIGHTDX_SITE_BASE_URL.rstrip('/')}/", str(href or "").strip())
    parsed = urlparse(url)
    expected_host = urlparse(FIGHTDX_SITE_BASE_URL).netloc
    if parsed.netloc != expected_host or not parsed.path.startswith("/person/"):
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"



def _tapology_stat_card_values(soup: BeautifulSoup, label: str) -> list[str]:
    label_el = soup.find(string=lambda s: isinstance(s, str) and s.strip() == label)
    if not label_el or not getattr(label_el, "parent", None) or not getattr(label_el.parent, "parent", None):
        return []
    card = label_el.parent.parent
    return [_clean_text(text) for text in card.stripped_strings if _clean_text(text)]


def _parse_tapology_title_name(soup: BeautifulSoup) -> str:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.match(r'^(.*?)\s*(?:\(".*"\))?\s*\|\s*MMA Fighter Page \|\s*Tapology$', title)
    return match.group(1).strip() if match else ""


def search_tapology_candidates(fighter_name: str, limit: int = 5) -> list[str]:
    global _tapology_blocked, _tapology_search_blocked
    # Honor cached block state, but avoid a fresh reachability probe here so
    # unit tests and mocked search paths do not depend on live network state.
    # When the requests path is blocked but hosted browser recovery is available,
    # _get_tapology_soup() can still recover Tapology search/profile pages. When
    # both are blocked, skip Tapology-origin search but still allow search-index
    # URL discovery for the reader fallback.
    tapology_origin_blocked = _tapology_blocked is True and not _tapology_browser_fallback_available()
    scored_urls: dict[str, int] = {}
    if not tapology_origin_blocked and not _tapology_search_blocked:
        for query in _name_query_variants(fighter_name):
            try:
                soup = _get_tapology_soup(
                    TAPOLOGY_SEARCH_URL,
                    params={"term": query},
                    max_retries=1,
                    retry_statuses={429, 503},
                )
            except TapologyRequestError as exc:
                if exc.status_code == 403:
                    if (
                        exc.detail == "Tapology blocked from this environment"
                        or exc.detail == "Cloudflare challenge"
                    ):
                        _tapology_blocked = True
                        if exc.detail == "Cloudflare challenge":
                            _mark_tapology_cloudflare_blocked(TAPOLOGY_SEARCH_URL)
                        else:
                            _log_external_source_error_once(
                                "Tapology",
                                "blocked from this environment",
                                "Tapology search candidates skipped",
                            )
                        break
                    _tapology_search_blocked = True
                    _log_external_source_error_once(
                        "Tapology",
                        "native search returned 403",
                        "disabling native search for this runtime",
                    )
                    break
                _log_external_source_error_once("Tapology", "search failed", exc)
                logger.warning("Tapology search failed for '%s': %s", query, exc)
                continue
            except Exception as exc:
                _log_external_source_error_once("Tapology", "search failed", exc)
                logger.warning("Tapology search failed for '%s': %s", query, exc)
                continue

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if "/fightcenter/fighters/" not in href:
                    continue
                candidate_name = _clean_text(link.get_text(" ", strip=True).replace('"', " "))
                score = _best_name_score(fighter_name, candidate_name, href)
                if score <= 0:
                    continue
                full_url = f"{TAPOLOGY_BASE_URL}{href}" if href.startswith("/") else href
                previous = scored_urls.get(full_url, 0)
                if score > previous:
                    scored_urls[full_url] = score

    should_try_reader_search = (
        _tapology_reader_available()
        and (_tapology_blocked is True or _tapology_search_blocked or not scored_urls)
    )
    if should_try_reader_search:
        for query in _name_query_variants(fighter_name)[:2]:
            attempted_reader = _search_tapology_candidates_with_reader(
                fighter_name,
                query,
                scored_urls=scored_urls,
            )
            if attempted_reader:
                _sleep_after_request(REQUEST_DELAY)
            if scored_urls:
                break

    tapology_origin_blocked = _tapology_blocked is True and not _tapology_browser_fallback_available()
    should_try_site_search = _tapology_search_blocked or not scored_urls
    if tapology_origin_blocked:
        should_try_site_search = should_try_site_search and _tapology_reader_available()
    if should_try_site_search:
        for full_url, score in _search_site_candidates(
            fighter_name,
            site_query="tapology.com/fightcenter/fighters",
            required_path_fragment="/fightcenter/fighters/",
        ):
            previous = scored_urls.get(full_url, 0)
            if score > previous:
                scored_urls[full_url] = score

    ranked_urls = sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)
    return [url for url, score in ranked_urls if score >= 8][:limit]


def search_tapology(fighter_name: str) -> Optional[str]:
    """Search Tapology for a fighter by name and return their full profile URL."""
    if fighter_name in _tapology_url_cache:
        return _tapology_url_cache[fighter_name]

    candidates = search_tapology_candidates(fighter_name, limit=1)
    if candidates:
        best_url = candidates[0]
        _tapology_url_cache[fighter_name] = best_url
        return best_url
    return None


def _parse_tapology_profile_soup(fighter_url: str, soup: BeautifulSoup) -> dict:
    age_card = _tapology_stat_card_values(soup, "Age")
    height_card = _tapology_stat_card_values(soup, "Height")
    reach_card = _tapology_stat_card_values(soup, "Reach")
    weight_card = _tapology_stat_card_values(soup, "Weight")

    dob = age_card[1].strip() if len(age_card) >= 2 and re.match(r"^\d{4}-\d{2}-\d{2}$", age_card[1].strip()) else ""
    height_raw = ""
    if len(height_card) >= 3:
        height_raw = f"{height_card[1]} ({height_card[2]})"
    elif len(height_card) >= 2:
        height_raw = height_card[1]

    reach_raw = ""
    if len(reach_card) >= 2 and reach_card[1] not in {"N/A", "??"}:
        reach_raw = reach_card[1]
        if len(reach_card) >= 3 and reach_card[2] not in {"N/A", "??"}:
            reach_raw = f"{reach_raw} ({reach_card[2]})"

    weight_raw = ""
    if len(weight_card) >= 2 and weight_card[1] not in {"N/A", "??"}:
        weight_raw = f"{weight_card[1]} lbs"

    record = ""
    summary_text = ""
    for div in soup.find_all("div", class_=True):
        text = " ".join(div.stripped_strings)
        if "Name:" in text and "Pro MMA Record:" in text and len(text) < 1500:
            summary_text = text
            record_match = re.search(r"Pro MMA Record:\s*([0-9\-]+)", text)
            if record_match:
                record = record_match.group(1).strip()
            break

    wins = losses = draws = 0
    if record:
        parts = record.split("-")
        if len(parts) >= 3:
            try:
                wins = int(parts[0])
                losses = int(parts[1])
                draws = int(parts[2])
            except ValueError:
                wins = losses = draws = 0

    age_raw = age_card[1].strip() if len(age_card) >= 2 else ""
    # Compute age: prefer DOB (more precise), fall back to raw age string
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)

    return {
        "name": _parse_tapology_title_name(soup),
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": "",
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
        "summary_text": summary_text,
        **_empty_profile_stats(),
    }


def scrape_tapology_profile(fighter_url: str) -> dict:
    """Scrape a Tapology fighter page for static profile attributes."""
    try:
        soup = _get_tapology_soup(fighter_url)
        return _parse_tapology_profile_soup(fighter_url, soup)
    except TapologyRequestError as exc:
        if not _tapology_reader_available():
            raise
        try:
            markdown = _get_tapology_markdown_with_reader(fighter_url)
            return _parse_tapology_reader_profile(fighter_url, markdown)
        except TapologyRequestError as reader_exc:
            raise reader_exc from exc


def search_martialbot(fighter_name: str) -> Optional[str]:
    """Search MartialBot for a fighter by name and return their full profile URL."""
    if fighter_name in _martialbot_url_cache:
        return _martialbot_url_cache[fighter_name]

    best_url = None
    best_score = 0
    for query in _name_query_variants(fighter_name):
        try:
            response = requests.get(
                MARTIALBOT_SEARCH_URL,
                params={"name": query, "sport": "mma"},
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            if _response_text_is_empty(response):
                _log_external_source_error_once(
                    "MartialBot",
                    "search returned empty response",
                    query,
                )
                continue
            payload = response.json()
            _sleep_after_request(MARTIALBOT_REQUEST_DELAY)
        except Exception as exc:
            _log_external_source_error_once(
                "MartialBot",
                "search failed",
                f"{query}: {exc}",
            )
            logger.warning("MartialBot search failed for '%s': %s", query, exc)
            continue

        for result in payload.get("results", []):
            candidate_name = _clean_text(
                str(result.get("display_name") or result.get("name") or "")
            )
            candidate_id = str(result.get("id") or "").strip()
            candidate_href = f"/mma/fighters/{candidate_id}"
            if not _candidate_has_required_name_tokens(
                query,
                candidate_name,
                candidate_href,
            ):
                continue
            score = _best_name_score(
                query,
                candidate_name,
                candidate_href,
            )
            if score > best_score and candidate_id:
                best_score = score
                best_url = f"{MARTIALBOT_BASE_URL}/mma/fighters/{candidate_id}"

    if best_url and best_score >= _FALLBACK_PROFILE_MATCH_MIN_SCORE:
        _martialbot_url_cache[fighter_name] = best_url
        return best_url
    return None


def _decode_turbo_stream(rows: object) -> object:
    """Decode a React Router single-fetch (turbo-stream) payload.

    MartialBot is a client-rendered app: the structured fighter bio is exposed
    via the ".data" route, encoded as a flat array of nodes where objects and
    arrays reference their children by array index. Object keys are themselves
    index references prefixed with "_". Negative references are sentinels
    (undefined/null/NaN/Infinity); only concrete bio values are needed, so they
    collapse to None.
    """
    if not isinstance(rows, list) or not rows:
        return None

    def resolve(ref: object, stack: frozenset) -> object:
        if isinstance(ref, bool):
            return ref
        if not isinstance(ref, int) or ref < 0 or ref >= len(rows) or ref in stack:
            return None
        node = rows[ref]
        if isinstance(node, dict):
            next_stack = stack | {ref}
            resolved: dict = {}
            for key, child_ref in node.items():
                if isinstance(key, str) and key.startswith("_"):
                    try:
                        name = rows[int(key[1:])]
                    except (ValueError, IndexError):
                        continue
                else:
                    name = key
                resolved[name] = resolve(child_ref, next_stack)
            return resolved
        if isinstance(node, list):
            next_stack = stack | {ref}
            return [resolve(child_ref, next_stack) for child_ref in node]
        return node

    return resolve(0, frozenset())


def _find_martialbot_fighter_node(node: object) -> Optional[dict]:
    """Locate the fighter bio dict within a decoded MartialBot profile payload.

    The bio carries a ``name`` plus physical attributes (``height_cm`` etc.) and
    profile metadata (``record``/``weightClasses``); this distinguishes it from
    the per-fight participant blocks, which use bare ``height``/``reach`` keys.
    """
    if isinstance(node, dict):
        has_identity = "name" in node and any(
            field in node for field in ("height_cm", "reach_cm", "birthdate")
        )
        has_profile = any(
            field in node for field in ("record", "weightClasses", "latest_weight_class")
        )
        if has_identity and has_profile:
            return node
        for value in node.values():
            found = _find_martialbot_fighter_node(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_martialbot_fighter_node(value)
            if found is not None:
                return found
    return None


def _fetch_martialbot_fighter(fighter_url: str) -> dict:
    """Fetch and decode the structured fighter bio for a MartialBot profile URL."""
    data_url = f"{fighter_url.rstrip('/')}.data"
    try:
        response = requests.get(data_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        if _response_text_is_empty(response):
            _log_external_source_error_once(
                "MartialBot",
                "profile data returned empty response",
                data_url,
            )
            return {}
        payload = _decode_turbo_stream(response.json())
    except Exception as exc:
        _log_external_source_error_once(
            "MartialBot",
            "profile data fetch failed",
            f"{data_url}: {exc}",
        )
        raise
    _sleep_after_request(MARTIALBOT_REQUEST_DELAY)
    fighter = _find_martialbot_fighter_node(payload) or {}
    if not fighter:
        _log_external_source_error_once(
            "MartialBot",
            "profile data missing fighter node",
            data_url,
        )
    return fighter


def _parse_record_triplet(record: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)-(\d+)(?:-(\d+))?", str(record or "").strip())
    if not match:
        return 0, 0, 0
    wins = int(match.group(1))
    losses = int(match.group(2))
    draws = int(match.group(3) or 0)
    return wins, losses, draws


def _parse_fightdx_heading_name(soup: BeautifulSoup) -> str:
    heading = soup.find("h1")
    if heading:
        return _clean_text(heading.get_text(" ", strip=True))
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if "|" in title:
        return _clean_text(title.split("|", 1)[0])
    return _clean_text(title)


def _is_fightdx_transient_failure(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    )


def _fightdx_cooldown_remaining_seconds() -> float:
    return max(0.0, _fightdx_unavailable_until - time.monotonic())


def _fightdx_temporarily_unavailable() -> bool:
    return _fightdx_cooldown_remaining_seconds() > 0


def _mark_fightdx_transient_failure(exc: BaseException) -> None:
    global _fightdx_unavailable_until
    if not _is_fightdx_transient_failure(exc) or FIGHTDX_FAILURE_COOLDOWN_SECONDS <= 0:
        return
    _fightdx_unavailable_until = max(
        _fightdx_unavailable_until,
        time.monotonic() + FIGHTDX_FAILURE_COOLDOWN_SECONDS,
    )


def _get_fightdx_soup(url: str) -> BeautifulSoup:
    if _fightdx_temporarily_unavailable():
        raise RuntimeError(
            f"FightDX temporarily unavailable for "
            f"{_fightdx_cooldown_remaining_seconds():.0f}s after a transient request failure"
        )
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        if _response_text_is_empty(response):
            _log_external_source_error_once(
                "FightDX",
                "profile returned empty response",
                url,
            )
            raise RuntimeError(f"FightDX returned an empty response for {url}")
        _sleep_after_request(REQUEST_DELAY)
        return BeautifulSoup(response.text, "lxml")
    except Exception as exc:
        _log_external_source_error_once(
            "FightDX",
            "profile scrape failed",
            f"{url}: {exc}",
            level=logging.WARNING if _is_fightdx_transient_failure(exc) else logging.ERROR,
        )
        _mark_fightdx_transient_failure(exc)
        raise


def _load_fightdx_person_urls() -> list[str]:
    global _fightdx_person_urls_cache

    if _fightdx_temporarily_unavailable():
        return []
    if _fightdx_person_urls_cache is not None:
        return _fightdx_person_urls_cache

    person_sitemap_urls: list[str] = []
    try:
        response = requests.get(
            FIGHTDX_SITEMAP_INDEX_URL,
            headers=HEADERS,
            timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        person_sitemap_urls = [
            loc.get_text(strip=True)
            for loc in soup.find_all("loc")
            if "_people.xml" in loc.get_text(strip=True)
        ]
        if not person_sitemap_urls:
            _log_external_source_error_once(
                "FightDX",
                "sitemap index returned no person sitemaps",
                FIGHTDX_SITEMAP_INDEX_URL,
            )
    except Exception as exc:
        _log_external_source_error_once(
            "FightDX",
            "sitemap index lookup failed",
            exc,
            level=logging.WARNING,
        )
        logger.warning("FightDX sitemap index lookup failed: %s", exc)
        _mark_fightdx_transient_failure(exc)
        _fightdx_person_urls_cache = []
        return _fightdx_person_urls_cache

    person_urls: list[str] = []
    seen_urls: set[str] = set()
    for sitemap_url in person_sitemap_urls:
        try:
            response = requests.get(
                sitemap_url,
                headers=HEADERS,
                timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            if _response_text_is_empty(response):
                _log_external_source_error_once(
                    "FightDX",
                    "sitemap page returned empty response",
                    sitemap_url,
                )
                continue
            for loc in soup.find_all("loc"):
                candidate_url = loc.get_text(strip=True)
                if "/person/" not in candidate_url or candidate_url in seen_urls:
                    continue
                seen_urls.add(candidate_url)
                person_urls.append(candidate_url)
            _sleep_after_request(FIGHTDX_SITEMAP_REQUEST_DELAY)
        except Exception as exc:
            _log_external_source_error_once(
                "FightDX",
                "sitemap page lookup failed",
                f"{sitemap_url}: {exc}",
                level=logging.WARNING,
            )
            logger.warning("FightDX sitemap page lookup failed for '%s': %s", sitemap_url, exc)
            _mark_fightdx_transient_failure(exc)
            continue

    _fightdx_person_urls_cache = person_urls
    return _fightdx_person_urls_cache


def _search_fightdx_sitemap_candidates(fighter_name: str, limit: int = 5) -> list[str]:
    if _fightdx_temporarily_unavailable():
        return []
    scored_urls: dict[str, int] = {}
    for candidate_url in _load_fightdx_person_urls():
        if not _candidate_has_required_name_tokens(fighter_name, href=candidate_url):
            continue
        score = _best_name_score(fighter_name, "", candidate_url)
        if score <= 0:
            continue
        previous = scored_urls.get(candidate_url, 0)
        if score > previous:
            scored_urls[candidate_url] = score

    ranked_urls = sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)
    return [url for url, score in ranked_urls if score >= 8][:limit]


def _search_fightdx_site_candidates(fighter_name: str, limit: int = 5) -> tuple[bool, list[str]]:
    if _fightdx_temporarily_unavailable():
        return True, []
    scored_urls: dict[str, int] = {}
    try:
        response = requests.get(
            FIGHTDX_SEARCH_URL,
            params={"query": fighter_name},
            headers=HEADERS,
            timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            _log_external_source_error_once(
                "FightDX",
                f"search returned status {response.status_code}",
                fighter_name,
                level=logging.WARNING,
            )
            return False, []
        if _response_text_is_empty(response):
            _log_external_source_error_once(
                "FightDX",
                "search returned empty response",
                fighter_name,
                level=logging.WARNING,
            )
            return False, []

        soup = BeautifulSoup(response.text, "lxml")
        for link in soup.find_all("a", href=True):
            candidate_url = _fightdx_person_url_from_href(link["href"])
            if not candidate_url:
                continue
            candidate_text = _clean_text(link.get_text(" ", strip=True))
            if not _candidate_has_required_name_tokens(
                fighter_name,
                candidate_text,
                candidate_url,
            ):
                continue
            score = _best_name_score(fighter_name, candidate_text, candidate_url)
            if score <= 0:
                continue
            previous = scored_urls.get(candidate_url, 0)
            if score > previous:
                scored_urls[candidate_url] = score
    except Exception as exc:
        _log_external_source_error_once(
            "FightDX",
            "search failed",
            f"{fighter_name}: {exc}",
            level=logging.WARNING,
        )
        logger.warning("FightDX search failed for '%s': %s", fighter_name, exc)
        _mark_fightdx_transient_failure(exc)
        return _is_fightdx_transient_failure(exc), []

    ranked_urls = sorted(scored_urls.items(), key=lambda item: item[1], reverse=True)
    return True, [url for url, score in ranked_urls if score >= 8][:limit]


def _resolve_fightdx_candidate_url(
    fighter_name: str,
    candidate_url: str,
    source_label: str,
) -> str | None:
    if _fightdx_temporarily_unavailable():
        return None
    try:
        response = requests.get(
            candidate_url,
            headers=HEADERS,
            timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            if response.status_code != 404:
                _log_external_source_error_once(
                    "FightDX",
                    f"{source_label} candidate returned status {response.status_code}",
                    candidate_url,
                )
            return None
        if _response_text_is_empty(response):
            _log_external_source_error_once(
                "FightDX",
                f"{source_label} candidate returned empty response",
                candidate_url,
            )
            return None
        soup = BeautifulSoup(response.text, "lxml")
        candidate_name = _parse_fightdx_heading_name(soup)
        if not _candidate_has_required_name_tokens(fighter_name, candidate_name):
            return None
        verified_score = _best_name_score(fighter_name, candidate_name)
        if verified_score < 8:
            return None
        _fightdx_url_cache[fighter_name] = candidate_url
        _sleep_after_request(REQUEST_DELAY)
        return candidate_url
    except Exception as exc:
        _log_external_source_error_once(
            "FightDX",
            f"{source_label} lookup failed",
            f"{fighter_name}: {exc}",
            level=logging.WARNING,
        )
        logger.warning("FightDX %s lookup failed for '%s': %s", source_label, fighter_name, exc)
        _mark_fightdx_transient_failure(exc)
        return None


def search_fightdx(fighter_name: str) -> Optional[str]:
    """Resolve a FightDX profile URL from the fighter's normalized slug."""
    if _fightdx_temporarily_unavailable():
        return None
    if fighter_name in _fightdx_url_cache:
        return _fightdx_url_cache[fighter_name]

    slug = _slugify_person_name(fighter_name)
    if not slug:
        return None

    url = f"{FIGHTDX_BASE_URL}/{slug}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=FIGHTDX_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            if _response_text_is_empty(response):
                _log_external_source_error_once(
                    "FightDX",
                    "profile returned empty response",
                    url,
                )
                return None
            soup = BeautifulSoup(response.text, "lxml")
            candidate_name = _parse_fightdx_heading_name(soup)
            if _candidate_has_required_name_tokens(fighter_name, candidate_name):
                score = _best_name_score(fighter_name, candidate_name)
                if score >= _FALLBACK_PROFILE_MATCH_MIN_SCORE:
                    _fightdx_url_cache[fighter_name] = url
                    _sleep_after_request(REQUEST_DELAY)
                    return url
        elif response.status_code not in {404}:
            _log_external_source_error_once(
                "FightDX",
                f"profile lookup returned status {response.status_code}",
                url,
            )
    except Exception as exc:
        _log_external_source_error_once(
            "FightDX",
            "profile lookup failed",
            f"{fighter_name}: {exc}",
            level=logging.WARNING,
        )
        logger.warning("FightDX lookup failed for '%s': %s", fighter_name, exc)
        _mark_fightdx_transient_failure(exc)
        if _is_fightdx_transient_failure(exc):
            _fightdx_url_cache[fighter_name] = ""
            return None

    search_attempted, search_candidates = _search_fightdx_site_candidates(fighter_name)
    for candidate_url in search_candidates:
        resolved_url = _resolve_fightdx_candidate_url(fighter_name, candidate_url, "search")
        if resolved_url:
            return resolved_url
    if search_attempted:
        _fightdx_url_cache[fighter_name] = ""
        return None

    for candidate_url in _search_fightdx_sitemap_candidates(fighter_name):
        resolved_url = _resolve_fightdx_candidate_url(fighter_name, candidate_url, "sitemap")
        if resolved_url:
            return candidate_url
    _fightdx_url_cache[fighter_name] = ""
    return None


def scrape_fightdx_profile(fighter_url: str) -> dict:
    """Scrape a FightDX fighter page for static profile attributes."""
    soup = _get_fightdx_soup(fighter_url)
    name = _parse_fightdx_heading_name(soup)

    details: dict[str, str] = {}
    for label in soup.select("span.info-stat-label"):
        value = label.find_next_sibling("span", class_="info-stat-value")
        if not value:
            continue
        label_text = _clean_text("".join(str(node) for node in label.contents))
        value_text = _clean_text("".join(str(node) for node in value.contents))
        if not label_text:
            continue
        details[label_text] = value_text

    height_raw = details.get("Height", "")
    reach_raw = details.get("Reach", "")
    weight_raw = details.get("Weight", "")
    age_raw = details.get("Age", "")
    dob = details.get("Date of Birth", "")
    dob = "" if dob in {"", "-"} else dob
    record = details.get("Record", "")
    wins, losses, draws = _parse_record_triplet(record)

    # Compute age: prefer DOB, fall back to raw age string
    age = _parse_dob_to_age(dob) if dob else _parse_age_from_raw(age_raw)

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": details.get("Style", ""),
        "age_raw": age_raw,
        "age": age,
        "dob": dob,
        **_empty_profile_stats(),
    }


def _extract_espn_fighter_url(item: dict) -> str:
    for link in item.get("links", []) or []:
        href = _clean_text(link.get("href", ""))
        if "/mma/fighter/_/id/" in href:
            return href
    return ""


def search_espn(fighter_name: str) -> Optional[str]:
    """Search ESPN's public player search API for an MMA fighter profile URL."""
    if fighter_name in _espn_url_cache:
        return _espn_url_cache[fighter_name]

    if not normalize_person_name(fighter_name):
        return None

    best_url = None
    best_score = 0
    for query in _name_query_variants(fighter_name):
        try:
            response = requests.get(
                ESPN_SEARCH_URL,
                params={"query": query, "type": "player"},
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            _sleep_after_request(REQUEST_DELAY)
        except Exception as exc:
            _log_external_source_error_once(
                "ESPN",
                "player search failed",
                f"{query}: {exc}",
            )
            logger.warning("ESPN player search failed for '%s': %s", query, exc)
            continue

        for item in payload.get("items", []) or []:
            if _clean_text(item.get("sport", "")).casefold() != "mma":
                continue
            fighter_url = _extract_espn_fighter_url(item)
            if not fighter_url:
                continue
            candidate_name = _clean_text(item.get("displayName") or item.get("shortName") or "")
            score = _best_name_score(fighter_name, candidate_name, fighter_url)
            if score >= 100:
                _espn_url_cache[fighter_name] = fighter_url
                return fighter_url
            if score > best_score:
                best_score = score
                best_url = fighter_url

    if best_url and best_score >= _FALLBACK_PROFILE_MATCH_MIN_SCORE:
        _espn_url_cache[fighter_name] = best_url
        return best_url
    return None


def _espn_athlete_api_url(fighter_url: str) -> str:
    match = re.search(r"/id/(\d+)", str(fighter_url or ""))
    if not match:
        raise ValueError(f"Could not parse ESPN athlete id from URL: {fighter_url}")
    return ESPN_CORE_ATHLETE_API_URL.format(athlete_id=match.group(1))


def _espn_height_raw(display_value: object, numeric_value: object) -> str:
    text = _clean_text(display_value)
    if text and text not in {"--", "0", "0.0"}:
        return text

    value = _safe_float(numeric_value)
    if np.isnan(value) or value <= 0:
        return ""
    return f"{_inches_to_cm(value):g} cm"


def _espn_reach_raw(display_value: object, numeric_value: object) -> str:
    text = _clean_text(display_value)
    if text and text not in {"--", "0", "0.0"}:
        return text

    value = _safe_float(numeric_value)
    if np.isnan(value) or value <= 0:
        return ""
    return f'{value:g}"'


def _espn_weight_raw(display_value: object, numeric_value: object) -> str:
    text = _clean_text(display_value)
    if text and text not in {"--", "0", "0.0"}:
        return text

    value = _safe_float(numeric_value)
    if np.isnan(value) or value <= 0:
        return ""
    return f"{value:g} lbs"


def scrape_espn_profile(fighter_url: str) -> dict:
    """Fetch structured ESPN MMA athlete profile data."""
    try:
        response = requests.get(
            _espn_athlete_api_url(fighter_url),
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        if _response_text_is_empty(response):
            _log_external_source_error_once("ESPN", "profile returned empty response", fighter_url)
            raise RuntimeError(f"ESPN returned an empty response for {fighter_url}")
        payload = response.json()
    except Exception as exc:
        if "empty response" not in str(exc):
            _log_external_source_error_once(
                "ESPN",
                "profile lookup failed",
                f"{fighter_url}: {exc}",
            )
        raise
    _sleep_after_request(REQUEST_DELAY)

    name = _clean_text(payload.get("displayName") or payload.get("fullName") or "")
    height_raw = _espn_height_raw(payload.get("displayHeight"), payload.get("height"))
    reach_raw = _espn_reach_raw(payload.get("displayReach"), payload.get("reach"))
    weight_raw = _espn_weight_raw(payload.get("displayWeight"), payload.get("weight"))

    stance_text = _clean_text((payload.get("stance") or {}).get("text"))
    if stance_text in {"--", "0"}:
        stance_text = ""

    dob_raw = _clean_text(payload.get("dateOfBirth"))
    dob = dob_raw.split("T", 1)[0] if "T" in dob_raw else dob_raw
    age = _parse_dob_to_age(dob) if dob else np.nan

    return {
        "name": name,
        "fighter_url": fighter_url,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": stance_text,
        "dob": dob,
        "age": age,
        **_empty_profile_stats(),
    }


def scrape_martialbot_profile(fighter_url: str) -> dict:
    """Scrape a MartialBot fighter page for static profile attributes."""
    fighter = _fetch_martialbot_fighter(fighter_url)

    name = _clean_text(fighter.get("name", ""))
    record_data = fighter.get("record") if isinstance(fighter.get("record"), dict) else {}
    wins = int(_safe_float(record_data.get("wins"), 0.0))
    losses = int(_safe_float(record_data.get("losses"), 0.0))
    draws = int(_safe_float(record_data.get("draws"), 0.0))
    record = f"{wins}-{losses}-{draws}" if name else ""

    # MartialBot reports height/reach as already-converted cm strings (e.g.
    # "193 cm"), stance lowercased ("orthodox"), and an exact ISO date of birth.
    # It exposes weight class rather than a pound/kg figure, so weight is left
    # blank (the profile-supplement builder does not recover weight from here).
    height_raw = _clean_text(fighter.get("height_cm", ""))
    reach_raw = _clean_text(fighter.get("reach_cm", ""))
    stance = _clean_text(fighter.get("stance", ""))
    stance = _MARTIALBOT_STANCE_ALIASES.get(stance.casefold(), stance)
    dob = _clean_text(fighter.get("birthdate", ""))
    weight_raw = ""

    age = _parse_dob_to_age(dob) if dob else np.nan

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_raw": height_raw,
        "height": _parse_height_cm(height_raw),
        "reach_raw": reach_raw,
        "reach": _parse_reach_cm(reach_raw),
        "weight_raw": weight_raw,
        "weight": _parse_weight_lbs(weight_raw),
        "stance": stance,
        "age_raw": "",
        "age": age,
        "dob": dob,
        **_empty_profile_stats(),
    }


_TAPOLOGY_READER_LINK_RE = re.compile(
    r"(!?)\[([^\]]+)\]\((\S+?)(?:\s+\"([^\"]*)\")?\)"
)
_TAPOLOGY_EVENT_DATE_RE = re.compile(r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}$")
_TAPOLOGY_RESULT_CODE_MAP = {
    "W": ("win", 1),
    "L": ("loss", 0),
    "D": ("draw", 0),
    "DRAW": ("draw", 0),
    "NC": ("nc", 0),
    "N/C": ("nc", 0),
}
_TAPOLOGY_METHOD_CODES = {"DEC", "SUB", "TKO", "KO", "DQ", "NC", "DRAW"}


def _tapology_method_label(
    method_code: str,
    method_detail: str = "",
    secondary_detail: str = "",
) -> str:
    method_code = _clean_text(method_code).upper()
    method_detail = _clean_text(method_detail)
    secondary_detail = _clean_text(secondary_detail)
    if method_code == "DEC":
        decision_detail = f"{secondary_detail} {method_detail}".lower()
        if "split" in decision_detail:
            return "Decision (Split)"
        if "majority" in decision_detail:
            return "Decision (Majority)"
        return "Decision (Unanimous)"
    if method_code == "SUB":
        finish = secondary_detail or _clean_text(method_detail.split("·", 1)[0])
        return f"Submission ({finish})" if finish else "Submission"
    if method_code in {"TKO", "KO"}:
        finish = secondary_detail or _clean_text(method_detail.split("·", 1)[0])
        return f"{method_code} ({finish})" if finish else method_code
    if method_code == "DQ":
        return "Disqualification"
    if method_code == "NC":
        return "No Contest"
    if method_code == "DRAW":
        return "Draw"
    return method_detail or method_code


def _tapology_finish_round(*texts: str) -> float | int:
    for text in texts:
        match = re.search(r"\bR(?:ound)?\.?\s*(\d+)\b", str(text or ""), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return np.nan


def _tapology_reader_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.replace("\\_", "_").replace("**", "")
        line = _clean_text(line)
        if line:
            lines.append(line)
    return lines


def _tapology_reader_links(line: str) -> list[tuple[str, str, str]]:
    links: list[tuple[str, str, str]] = []
    for match in _TAPOLOGY_READER_LINK_RE.finditer(str(line or "")):
        if match.group(1):
            continue
        links.append(
            (
                _clean_text(match.group(2)),
                _clean_text(match.group(3)),
                _clean_text(match.group(4)),
            )
        )
    return links


def _find_tapology_reader_link(
    lines: list[str],
    start_idx: int,
    *,
    url_fragment: str,
    title_fragment: str = "",
    date_text: bool = False,
    max_scan: int = 10,
) -> tuple[int, str] | None:
    title_fragment = title_fragment.casefold()
    stop_idx = min(len(lines), start_idx + max_scan)
    for idx in range(start_idx, stop_idx):
        for text, url, title in _tapology_reader_links(lines[idx]):
            if url_fragment not in url:
                continue
            if title_fragment and title and title_fragment not in title.casefold():
                continue
            if date_text != bool(_TAPOLOGY_EVENT_DATE_RE.match(text)):
                continue
            return idx, text
    return None


def _tapology_reader_title_bout(lines: list[str], result_idx: int) -> int:
    context = " ".join(lines[max(0, result_idx - 8):result_idx]).casefold()
    return 1 if ("title fight" in context or "title bout" in context) else 0


def _parse_tapology_reader_fights(markdown: str, *, division: str = "pro") -> list[dict]:
    division_key = _clean_text(division).casefold() or "pro"
    if division_key not in {"pro", "professional"}:
        return []
    if "pro mma record" not in str(markdown or "").lower():
        return []

    lines = _tapology_reader_lines(markdown)
    fights: list[dict] = []
    for idx, line in enumerate(lines):
        result_code = _clean_text(line).upper()
        if result_code not in _TAPOLOGY_RESULT_CODE_MAP:
            continue
        if idx + 1 >= len(lines):
            continue

        method_code = _clean_text(lines[idx + 1]).upper()
        if method_code not in _TAPOLOGY_METHOD_CODES:
            continue

        opponent_link = _find_tapology_reader_link(
            lines,
            idx + 2,
            url_fragment="/fightcenter/fighters/",
            title_fragment="Fighter Page",
            max_scan=8,
        )
        if opponent_link is None:
            continue
        opponent_idx, opponent = opponent_link

        method_link = _find_tapology_reader_link(
            lines,
            opponent_idx + 1,
            url_fragment="/fightcenter/bouts/",
            title_fragment="Bout Page",
            max_scan=10,
        )
        if method_link is None:
            continue
        method_idx, method_detail = method_link

        event_link = _find_tapology_reader_link(
            lines,
            method_idx + 1,
            url_fragment="/fightcenter/events/",
            title_fragment="Event Page",
            date_text=False,
            max_scan=8,
        )
        if event_link is None:
            continue
        event_idx, event_name = event_link

        secondary_detail = ""
        secondary_link = _find_tapology_reader_link(
            lines,
            event_idx + 1,
            url_fragment="/fightcenter/bouts/",
            title_fragment="Bout Page",
            max_scan=4,
        )
        if secondary_link is not None:
            secondary_detail = secondary_link[1]

        date_link = _find_tapology_reader_link(
            lines,
            event_idx + 1,
            url_fragment="/fightcenter/events/",
            date_text=True,
            max_scan=10,
        )
        if date_link is None:
            continue
        date_idx, date_text = date_link

        try:
            event_date = datetime.strptime(date_text, "%Y %b %d")
        except ValueError:
            continue

        status, won = _TAPOLOGY_RESULT_CODE_MAP[result_code]
        detail_window = lines[method_idx:date_idx + 1]
        fights.append(
            {
                "event_date": event_date,
                "event_name": event_name,
                "organization": event_name,
                "opponent": opponent,
                "result": status,
                "won": won,
                "method": _tapology_method_label(method_code, method_detail, secondary_detail),
                "round_finished": _tapology_finish_round(*detail_window),
                "is_title_bout": _tapology_reader_title_bout(lines, idx),
                **_empty_fight_dict(),
            }
        )

    fights.reverse()
    return fights


def _parse_tapology_fights_soup(
    fighter_url: str,
    fighter_name: str,
    soup: BeautifulSoup,
    division: str = "pro",
) -> list[dict]:
    fights: list[dict] = []
    division_aliases = {
        "pro": {"pro", "professional"},
        "am": {"am", "amateur"},
    }
    allowed_divisions = division_aliases.get(division, {division})

    for block in soup.select("[data-bout-id]"):
        if block.get("data-sport") != "mma":
            continue
        if str(block.get("data-division") or "").strip().lower() not in allowed_divisions:
            continue

        status = str(block.get("data-status") or "").strip().lower()
        if status in {"cancelled", "booking", "scheduled", "upcoming"}:
            continue

        block_texts = [
            _clean_text(text)
            for text in block.stripped_strings
            if _clean_text(text)
        ]

        result_row = block.find("div", class_="result")
        result_children = result_row.find_all("div", recursive=False) if result_row else []
        method_code = ""
        if len(result_children) >= 2:
            method_code = _clean_text(result_children[1].get_text(" ", strip=True)).upper()

        fighter_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/fighters/" in h,
            title=lambda t: t and "Fighter Page" in t,
        )
        if not fighter_links:
            continue
        opponent = _clean_text(fighter_links[0].get_text(" ", strip=True))
        if not opponent:
            continue

        bout_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/bouts/" in h,
            title=lambda t: t and "Bout Page" in t,
        )
        bout_texts = [
            _clean_text(link.get_text(" ", strip=True))
            for link in bout_links
            if _clean_text(link.get_text(" ", strip=True))
        ]
        method_detail = bout_texts[0] if bout_texts else ""
        secondary_detail = bout_texts[1] if len(bout_texts) > 1 else ""

        event_links = block.find_all(
            "a",
            href=lambda h: h and "/fightcenter/events/" in h,
        )
        event_name = ""
        date_text = ""
        for link in event_links:
            text = _clean_text(link.get_text(" ", strip=True))
            if not text:
                continue
            if re.match(r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}$", text):
                date_text = text
            elif len(text) > len(event_name):
                event_name = text

        if not date_text:
            for link in block.find_all("a", href=True):
                text = _clean_text(link.get_text(" ", strip=True))
                if re.match(r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}$", text):
                    date_text = text
                    break

        event_date = None
        if date_text:
            try:
                event_date = datetime.strptime(date_text, "%Y %b %d")
            except ValueError:
                event_date = None

        promotion_name = ""
        promo_link = block.find("a", href=lambda h: h and "/fightcenter/promotions/" in h)
        if promo_link:
            slug = str(promo_link.get("href") or "").rstrip("/").split("/")[-1]
            promotion_name = _titleize_slug(slug)
        if not promotion_name:
            for idx, text in enumerate(block_texts[:-1]):
                if text in {"League:", "Promotion:"}:
                    promotion_name = block_texts[idx + 1]
                    break
        if not promotion_name:
            promotion_name = event_name

        title_bout = 1 if block.find(class_="fighterBeltIcon") else 0

        fights.append(
            {
                "event_date": event_date,
                "event_name": event_name,
                "organization": promotion_name,
                "opponent": opponent,
                "result": status,
                "won": 1 if status == "win" else 0,
                "method": _tapology_method_label(method_code, method_detail, secondary_detail),
                "round_finished": _tapology_finish_round(method_detail, secondary_detail),
                "is_title_bout": title_bout,
                **_empty_fight_dict(),
            }
        )

    fights.reverse()
    return fights


def scrape_tapology_fights(
    fighter_url: str,
    fighter_name: str,
    division: str = "pro",
) -> list[dict]:
    """Scrape Tapology fight history blocks for a fighter page."""
    try:
        soup = _get_tapology_soup(fighter_url)
        return _parse_tapology_fights_soup(fighter_url, fighter_name, soup, division)
    except TapologyRequestError as exc:
        if not _tapology_reader_available():
            raise
        try:
            markdown = _get_tapology_markdown_with_reader(fighter_url)
            return _parse_tapology_reader_fights(markdown, division=division)
        except TapologyRequestError as reader_exc:
            raise reader_exc from exc


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _merge_static_fallback_profile_sources(
    fighter_name: str,
    merged_profile: dict | None,
) -> dict | None:
    # ESPN is generally stronger for physical fields/stance; MartialBot carries
    # records for fighters ESPN does not expose fully.
    for source_name, search_fn, scrape_fn in (
        ("ESPN", search_espn, scrape_espn_profile),
        ("MartialBot", search_martialbot, scrape_martialbot_profile),
        ("FightDX", search_fightdx, scrape_fightdx_profile),
    ):
        try:
            source_url = search_fn(fighter_name)
            if not source_url:
                continue
            logger.info(f"  Found {fighter_name} on {source_name}: {source_url}")
            source_profile = scrape_fn(source_url)
            if not source_profile or not source_profile.get("name"):
                continue
            if merged_profile is None:
                merged_profile = source_profile
            else:
                filled_groups = _merge_missing_profile_fields(merged_profile, source_profile)
                if filled_groups:
                    logger.info(
                        "  Filled %s fallback profile fields from %s: %s",
                        fighter_name,
                        source_name,
                        ", ".join(filled_groups),
                    )
            if _profile_has_core_static_fields(merged_profile):
                break
        except Exception as e:
            logger.warning(f"{source_name} fallback failed for {fighter_name}: {e}")
    return merged_profile


def fallback_lookup(fighter_name: str) -> Optional[tuple[dict, list[dict]]]:
    """
    Try fallback sources (Sherdog → Tapology) for a fighter's data.

    Returns (profile_dict, fights_list) or None if all sources fail.
    Both dicts match the UFCStats format used by _compute_rolling_for_fighter.
    """
    merged_profile: dict | None = None
    merged_fights: list[dict] = []

    # Try Sherdog first
    try:
        sherdog_url = search_sherdog(fighter_name)
        if sherdog_url:
            logger.info(f"  Found {fighter_name} on Sherdog: {sherdog_url}")
            profile, fights = scrape_sherdog_page(sherdog_url, fighter_name)
            if profile and profile.get("name"):
                merged_profile = profile
                merged_fights = fights
    except Exception as e:
        logger.warning(f"Sherdog fallback failed for {fighter_name}: {e}")

    if merged_profile is not None and merged_fights:
        merged_profile = _merge_static_fallback_profile_sources(fighter_name, merged_profile)
        if _profile_has_core_static_fields(merged_profile):
            return merged_profile, merged_fights

    # Try Tapology for static profile recovery when Sherdog fails or lacks fields
    try:
        tapology_url = search_tapology(fighter_name)
        if tapology_url:
            logger.info(f"  Found {fighter_name} on Tapology: {tapology_url}")
            profile = scrape_tapology_profile(tapology_url)
            if profile and profile.get("name"):
                try:
                    fights = scrape_tapology_fights(tapology_url, fighter_name)
                except Exception as fight_exc:
                    logger.warning(
                        "Tapology fight history failed for %s after profile recovery: %s",
                        fighter_name,
                        fight_exc,
                    )
                    fights = []
                if merged_profile is None:
                    merged_profile = profile
                else:
                    filled_groups = _merge_missing_profile_fields(merged_profile, profile)
                    if filled_groups:
                        logger.info(
                            "  Filled %s fallback profile fields from Tapology: %s",
                            fighter_name,
                            ", ".join(filled_groups),
                        )
                if len(fights) > len(merged_fights):
                    merged_fights = fights
    except Exception as e:
        logger.warning(f"Tapology fallback failed for {fighter_name}: {e}")

    merged_profile = _merge_static_fallback_profile_sources(fighter_name, merged_profile)

    if merged_profile and merged_profile.get("name"):
        return merged_profile, merged_fights

    return None


def clear_fallback_cache(*, preserve_environment_blocks: bool = False):
    """Clear all fallback scraper caches.

    ``preserve_environment_blocks`` keeps process-level outage state that is not
    a data cache, such as Tapology Cloudflare blocks for the current runtime.
    """
    _sherdog_url_cache.clear()
    _tapology_url_cache.clear()
    _tapology_browser_html_cache.clear()
    _tapology_reader_markdown_cache.clear()
    _martialbot_url_cache.clear()
    _fightdx_url_cache.clear()
    _espn_url_cache.clear()
    if not preserve_environment_blocks:
        _external_source_alert_keys.clear()
    global _fightdx_person_urls_cache, _fightdx_unavailable_until
    global _tapology_scraper, _tapology_scraper_profile_index
    global _last_tapology_request_at, _last_tapology_browser_request_at
    global _tapology_blocked, _tapology_search_blocked, _tapology_browser_unavailable
    global _tapology_browser_cloudflare_blocked, _site_search_disabled
    global _tapology_reader_unavailable, _duckduckgo_site_search_disabled
    global _tapology_browser_profile_dir
    _fightdx_person_urls_cache = None
    _fightdx_unavailable_until = 0.0
    _tapology_scraper = None
    _tapology_scraper_profile_index = 0
    _last_tapology_request_at = 0.0
    _last_tapology_browser_request_at = 0.0
    if not preserve_environment_blocks:
        _tapology_blocked = None
        _tapology_search_blocked = False
        _tapology_browser_unavailable = False
        _tapology_browser_cloudflare_blocked = False
        _tapology_reader_unavailable = False
        _duckduckgo_site_search_disabled = False
    if _tapology_browser_profile_dir:
        shutil.rmtree(_tapology_browser_profile_dir, ignore_errors=True)
    _tapology_browser_profile_dir = None
    if not preserve_environment_blocks:
        _site_search_disabled = False
