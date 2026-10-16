"""Public roster.watch website acquisition for the Roster Watch membership feed.

The owner designated the roster.watch website as the recurring authoritative
membership source (the September 1, 2026 CSV was a one-time kickstart baseline;
an owner-supplied CSV is only an optional override).  This module scrapes the
public current-roster page, preserves every raw page immutably with retrieval
time, size and SHA-256, parses the roster table into the existing 37-column
Roster Watch membership table, and proves completeness before any membership
reconciliation runs.

Site facts established by the sealed u31r1 discovery captures (2026-09-03):
  * ``https://www.roster.watch/`` is the canonical host (the apex 301s to it);
  * the whole current roster is one static HTML page; ``#roster-tbl.filter-current``
    holds one ``<tr>`` per member with ``data-*`` attributes for the membership
    fields, including ``data-fighter`` and ``data-dob``; pagination is purely
    client-side JavaScript, so one complete page is the complete roster;
  * a member may lack a fighter page slug (``/fighters/<slug>.html``), so the
    cross-snapshot continuity key stays exact source name plus date of birth;
  * Cloudflare serves a managed challenge to library-signature user agents and
    ``robots.txt`` allows ``User-agent: *`` on ``/``; the plain identifying user
    agent below passed.

No login, no Patreon, no cookies, no redirect following, no retries.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.data.name_utils import normalize_person_name

ROSTERWATCH_SITE_HOST = "www.roster.watch"
ROSTERWATCH_SITE_ROSTER_URL = "https://www.roster.watch/"
ROSTERWATCH_SITE_USER_AGENT = (
    "UFCBettingBot-rosterwatch-sync/1.0 (owner-operated research bot)"
)
ROSTERWATCH_SITE_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
ROSTERWATCH_SITE_REQUEST_TIMEOUT = (10.0, 60.0)
ROSTERWATCH_SITE_REQUEST_CEILING = 1
ROSTERWATCH_SITE_TABLE_SELECTOR = "table#roster-tbl"
ROSTERWATCH_SITE_CURRENT_TABLE_CLASS = "filter-current"
ROSTERWATCH_SITE_MINIMUM_ROWS = 100

ROSTERWATCH_SITE_COLUMNS = (
    "fighter",
    "date_of_birth",
    "tuf",
    "r2ufc",
    "dwcs",
    "pride",
    "sf",
    "wec",
    "bmf",
    "hof",
    "country",
    "gender",
    "weightclass",
    "current_rank",
    "rank_flag",
    "current_p4p",
    "p4p_flag",
    "peak_rank",
    "peak_p4p",
    "bouts",
    "wins",
    "win_types",
    "losses",
    "loss_types",
    "streak",
    "best_streak",
    "bonus",
    "slot",
    "elo",
    "peak_elo",
    "opp_elo",
    "debut",
    "debut_result",
    "prev_bout",
    "prev_result",
    "next_date",
    "next_opp",
)

_BOOLEAN_BADGES = ("tuf", "r2ufc", "dwcs", "pride", "sf", "wec", "bmf")
_NA = "NA"


class RosterWatchSiteError(ValueError):
    """Fail-closed roster.watch site acquisition or parse error.

    ``page`` carries the rejected response (when one was received) so the
    caller can preserve it as rejected evidence without treating it as a roster.
    """

    def __init__(self, message: str, *, page: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.page = dict(page) if page is not None else None


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _valid_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def looks_like_bot_challenge(
    status_code: int, headers: Mapping[str, str], content: bytes
) -> bool:
    """Return true for a Cloudflare managed challenge or similar interstitial."""
    lowered = {str(k).casefold(): str(v) for k, v in headers.items()}
    if lowered.get("cf-mitigated", "").casefold() == "challenge":
        return True
    head = content[:4000].decode("utf-8", errors="replace").casefold()
    if "<title>just a moment" in head and "challenges.cloudflare.com" in head:
        return True
    return status_code in {403, 503} and "cf_chl" in head


def fetch_rosterwatch_site_page(
    url: str = ROSTERWATCH_SITE_ROSTER_URL,
    *,
    session: requests.Session | None = None,
    user_agent: str = ROSTERWATCH_SITE_USER_AGENT,
    timeout: tuple[float, float] = ROSTERWATCH_SITE_REQUEST_TIMEOUT,
) -> dict[str, object]:
    """Fetch exactly one public roster.watch page with no redirects and no retries."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() != ROSTERWATCH_SITE_HOST:
        raise RosterWatchSiteError(f"roster.watch page URL must be https on {ROSTERWATCH_SITE_HOST}: {url}")
    client = session or requests.Session()
    client.trust_env = False
    requested_at = datetime.now(timezone.utc).isoformat()
    response = client.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": ROSTERWATCH_SITE_ACCEPT,
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
        timeout=timeout,
        allow_redirects=False,
    )
    retrieved_at = datetime.now(timezone.utc).isoformat()
    status = int(response.status_code)
    headers = {str(k): str(v) for k, v in response.headers.items()}
    content = bytes(response.content)
    page = {
        "url": url,
        "requested_at_utc": requested_at,
        "retrieved_at_utc": retrieved_at,
        "status_code": status,
        "headers": headers,
        "byte_count": len(content),
        "sha256": _sha256(content),
        "content": content,
        "user_agent": user_agent,
        "redirects_followed": 0,
        "retries": 0,
    }
    if 300 <= status < 400:
        raise RosterWatchSiteError(
            f"roster.watch page redirected (HTTP {status} to {headers.get('Location', '')!r}); redirects are never followed",
            page=page,
        )
    if looks_like_bot_challenge(status, headers, content):
        raise RosterWatchSiteError(f"roster.watch served a bot challenge (HTTP {status}); no challenge is solved", page=page)
    if status != 200:
        raise RosterWatchSiteError(f"roster.watch page returned HTTP {status}", page=page)
    advertised = headers.get("Content-Length")
    if advertised is not None and advertised.strip().isdigit() and int(advertised) != len(content):
        raise RosterWatchSiteError("roster.watch page body length does not match Content-Length", page=page)
    media_type = headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
    if media_type != "text/html":
        raise RosterWatchSiteError(f"roster.watch page media type is not text/html: {media_type!r}", page=page)
    if not content:
        raise RosterWatchSiteError("roster.watch page body is empty", page=page)
    return page


def _text_or_na(value: object) -> str:
    text = _clean(value)
    return text if text else _NA


def _badge(value: object) -> str:
    return "TRUE" if _clean(value).casefold() == "true" else "FALSE"


def _event_cell(td) -> tuple[str, str]:
    if td is None:
        return _NA, _NA
    link = td.find("a")
    sub = td.select_one(".td-subtext")
    date_text = _clean(link.get_text()) if link else ""
    sub_text = _clean(sub.get_text()) if sub else ""
    return (date_text or _NA), (sub_text or _NA)


def parse_rosterwatch_site_row(row) -> dict[str, str]:
    """Map one ``<tr>`` of the current-roster table onto the 37 Roster Watch columns."""
    attrs = row.attrs
    fighter = _clean(attrs.get("data-fighter"))
    dob = _clean(attrs.get("data-dob"))
    if not fighter:
        raise RosterWatchSiteError("roster.watch row lacks data-fighter")
    if not _valid_iso_date(dob):
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} lacks a valid data-dob: {dob!r}")
    country_td = row.select_one("td.td-country")
    countries = [
        _clean(div.get("title"))
        for div in (country_td.select("div.emoji") if country_td else [])
        if _clean(div.get("title"))
    ]
    try:
        weightclasses = json.loads(attrs.get("data-weightclass") or "[]")
    except json.JSONDecodeError as exc:
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} has invalid data-weightclass") from exc
    if not isinstance(weightclasses, list) or not all(isinstance(item, str) for item in weightclasses):
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} has a non-list data-weightclass")
    rank_tds = row.select("td.td-rank")
    rank_flag = _clean(rank_tds[0].select_one(".rank-flag").get_text()) if len(rank_tds) > 0 and rank_tds[0].select_one(".rank-flag") else ""
    p4p_flag = _clean(rank_tds[1].select_one(".rank-flag").get_text()) if len(rank_tds) > 1 and rank_tds[1].select_one(".rank-flag") else ""
    record_tds = row.select("td.td-record")
    win_types = _clean(record_tds[0].select_one(".td-subtext").get_text()) if len(record_tds) > 0 and record_tds[0].select_one(".td-subtext") else ""
    loss_types = _clean(record_tds[1].select_one(".td-subtext").get_text()) if len(record_tds) > 1 and record_tds[1].select_one(".td-subtext") else ""
    event_tds = row.select("td.td-event")
    debut, debut_result = _event_cell(event_tds[0] if len(event_tds) > 0 else None)
    prev_bout, prev_result = _event_cell(event_tds[1] if len(event_tds) > 1 else None)
    next_date, next_opp = _event_cell(event_tds[2] if len(event_tds) > 2 else None)
    # data-* attributes are the authoritative machine values; cell text only supplies
    # the result strings that the site renders without a data attribute.
    debut_attr = _clean(attrs.get("data-debut"))
    prev_attr = _clean(attrs.get("data-prev"))
    next_attr = _clean(attrs.get("data-next"))
    if debut_attr and debut != _NA and debut_attr != debut:
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} has inconsistent debut date")
    if prev_attr and prev_bout != _NA and prev_attr != prev_bout:
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} has inconsistent previous-bout date")
    if next_attr and next_date != _NA and next_attr != next_date:
        raise RosterWatchSiteError(f"roster.watch row for {fighter!r} has inconsistent next-bout date")
    link = row.select_one("a[href^='/fighters/']")
    record = {
        "fighter": fighter,
        "date_of_birth": dob,
        "tuf": _badge(attrs.get("data-tuf")),
        "r2ufc": _badge(attrs.get("data-r2ufc")),
        "dwcs": _badge(attrs.get("data-dwcs")),
        "pride": _badge(attrs.get("data-pride")),
        "sf": _badge(attrs.get("data-sf")),
        "wec": _badge(attrs.get("data-wec")),
        "bmf": _badge(attrs.get("data-bmf")),
        "hof": "TRUE" if _clean(attrs.get("data-hof")).casefold() == "true" else _NA,
        "country": ", ".join(countries),
        "gender": _text_or_na(attrs.get("data-gender")),
        "weightclass": ", ".join(_clean(item) for item in weightclasses),
        "current_rank": _text_or_na(attrs.get("data-rank")),
        "rank_flag": rank_flag or _NA,
        "current_p4p": _text_or_na(attrs.get("data-p4p")),
        "p4p_flag": p4p_flag or _NA,
        "peak_rank": _text_or_na(attrs.get("data-peak")),
        "peak_p4p": _text_or_na(attrs.get("data-peak_p4p")),
        "bouts": _text_or_na(attrs.get("data-bouts")),
        "wins": _text_or_na(attrs.get("data-wins")),
        "win_types": win_types or _NA,
        "losses": _text_or_na(attrs.get("data-losses")),
        "loss_types": loss_types or _NA,
        "streak": _text_or_na(attrs.get("data-streak")),
        "best_streak": _text_or_na(attrs.get("data-best_streak")),
        "bonus": _text_or_na(attrs.get("data-bonus")),
        "slot": _text_or_na(attrs.get("data-slot")),
        "elo": _text_or_na(attrs.get("data-elo")),
        "peak_elo": _text_or_na(attrs.get("data-peak_elo")),
        "opp_elo": _text_or_na(attrs.get("data-opp_elo")),
        "debut": debut_attr or debut,
        "debut_result": debut_result,
        "prev_bout": prev_attr or prev_bout,
        "prev_result": prev_result,
        "next_date": next_attr or next_date,
        "next_opp": next_opp,
    }
    record["_page_slug"] = _clean(link.get("href")) if link else ""
    return record


def parse_rosterwatch_site_roster(content: bytes) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Parse the complete current-roster page; every table row must map or the parse fails."""
    if not isinstance(content, bytes) or not content:
        raise RosterWatchSiteError("roster.watch page payload is empty")
    soup = BeautifulSoup(content, "lxml")
    tables = soup.select(ROSTERWATCH_SITE_TABLE_SELECTOR)
    if len(tables) != 1:
        raise RosterWatchSiteError(f"expected exactly one roster table, found {len(tables)}")
    table = tables[0]
    if ROSTERWATCH_SITE_CURRENT_TABLE_CLASS not in (table.get("class") or []):
        raise RosterWatchSiteError("roster table is not the current-roster table")
    tbody = table.find("tbody")
    if tbody is None:
        raise RosterWatchSiteError("roster table has no tbody")
    body_rows = tbody.find_all("tr", recursive=False)
    if len(body_rows) < ROSTERWATCH_SITE_MINIMUM_ROWS:
        raise RosterWatchSiteError(f"roster table is implausibly small ({len(body_rows)} rows)")
    rows: list[dict[str, str]] = []
    exact_names: set[str] = set()
    normalized_names: set[str] = set()
    slugs: set[str] = set()
    rows_without_slug: list[str] = []
    for row in body_rows:
        record = parse_rosterwatch_site_row(row)
        fighter = record["fighter"]
        normalized = normalize_person_name(fighter)
        if not normalized:
            raise RosterWatchSiteError(f"roster.watch row {fighter!r} has an empty normalized key")
        if fighter in exact_names or normalized in normalized_names:
            raise RosterWatchSiteError(f"roster.watch page contains duplicate fighter identity {fighter!r}")
        exact_names.add(fighter)
        normalized_names.add(normalized)
        slug = record.pop("_page_slug")
        if slug:
            if slug in slugs:
                raise RosterWatchSiteError(f"roster.watch page contains duplicate fighter page slug {slug!r}")
            slugs.add(slug)
        else:
            rows_without_slug.append(fighter)
        rows.append(record)
    metadata = {
        "schema_version": 1,
        "source": "rosterwatch_site",
        "page_sha256": _sha256(content),
        "page_bytes": len(content),
        "table_selector": ROSTERWATCH_SITE_TABLE_SELECTOR,
        "table_class": ROSTERWATCH_SITE_CURRENT_TABLE_CLASS,
        "pages_fetched": 1,
        "pages_required": 1,
        "pagination": "client_side_javascript_only_single_html_page",
        "tbody_row_count": len(body_rows),
        "parsed_row_count": len(rows),
        "displayed_total_basis": "tbody_row_count; the visible page-info total is rendered client-side from the same rows",
        "displayed_total": len(body_rows),
        "rows_with_page_slug": len(slugs),
        "rows_without_page_slug": rows_without_slug,
        "duplicate_names": 0,
        "duplicate_slugs": 0,
        "complete": len(rows) == len(body_rows),
        "status": "passed",
    }
    if not metadata["complete"]:
        raise RosterWatchSiteError("roster.watch parse did not cover every table row")
    return rows, metadata


def rosterwatch_csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    """Serialize parsed rows as the exact 37-column Roster Watch CSV (UTF-8, LF, no BOM)."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(ROSTERWATCH_SITE_COLUMNS), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in ROSTERWATCH_SITE_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    except FileExistsError:
        if path.is_file() and path.read_bytes() == content:
            return
        raise RosterWatchSiteError(f"immutable capture collision at {path}")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    try:
        path.chmod(0o444)
    except OSError:
        pass


def preserve_rosterwatch_site_page(page: Mapping[str, object], capture_root: Path | str) -> dict[str, object]:
    """Write the raw page and its retrieval metadata immutably; return the capture record."""
    root = Path(capture_root)
    content = page["content"]
    assert isinstance(content, bytes)
    retrieved = str(page["retrieved_at_utc"])
    stamp = retrieved.replace(":", "").replace("+00:00", "Z")
    digest = str(page["sha256"])
    page_path = root / f"{stamp}--{digest}.html"
    meta_path = root / f"{stamp}--{digest}.meta.json"
    metadata = {
        "schema_version": 1,
        "source": "rosterwatch_site",
        "url": page["url"],
        "requested_at_utc": page["requested_at_utc"],
        "retrieved_at_utc": retrieved,
        "status_code": page["status_code"],
        "byte_count": page["byte_count"],
        "sha256": digest,
        "headers": page["headers"],
        "user_agent": page["user_agent"],
        "redirects_followed": page["redirects_followed"],
        "retries": page["retries"],
        "page_path": page_path.name,
    }
    _write_immutable(page_path, content)
    _write_immutable(meta_path, (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return {**metadata, "page_path": str(page_path), "metadata_path": str(meta_path)}


def acquire_rosterwatch_site_roster(
    *,
    capture_root: Path | str,
    session: requests.Session | None = None,
    url: str = ROSTERWATCH_SITE_ROSTER_URL,
    user_agent: str = ROSTERWATCH_SITE_USER_AGENT,
) -> dict[str, object]:
    """Fetch, preserve, parse and prove one complete current-roster page.

    Returns the derived 37-column CSV bytes with the immutable page capture
    record and the parse/completeness metadata.  Any failure raises
    :class:`RosterWatchSiteError`; the caller retains its last accepted roster.
    A rejected response (redirect, challenge, wrong status, truncated body,
    unparseable roster) is preserved immutably under ``<capture_root>-rejected``
    as evidence only; it never becomes an accepted snapshot.
    """
    capture_root = Path(capture_root)
    rejected_root = capture_root.with_name(capture_root.name + "-rejected")
    try:
        page = fetch_rosterwatch_site_page(url, session=session, user_agent=user_agent)
    except RosterWatchSiteError as exc:
        if exc.page is not None and exc.page.get("content"):
            preserve_rosterwatch_site_page(exc.page, rejected_root)
        raise
    try:
        rows, parse_metadata = parse_rosterwatch_site_roster(page["content"])
    except RosterWatchSiteError as exc:
        preserve_rosterwatch_site_page(page, rejected_root)
        raise RosterWatchSiteError(str(exc), page=page) from exc
    capture = preserve_rosterwatch_site_page(page, capture_root)
    csv_bytes = rosterwatch_csv_bytes(rows)
    return {
        "csv_bytes": csv_bytes,
        "csv_sha256": _sha256(csv_bytes),
        "row_count": len(rows),
        "request_count": 1,
        "capture": capture,
        "parse": parse_metadata,
        "retrieved_at_utc": page["retrieved_at_utc"],
        "retrieved_date_utc": str(page["retrieved_at_utc"])[:10],
    }
