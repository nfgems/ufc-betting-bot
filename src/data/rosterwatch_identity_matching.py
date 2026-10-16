"""Deterministic Roster Watch identity decisions from immutable captured evidence.

This module has no HTTP client. Discovery returns URLs for a separately reviewed
transport plan; parsing and decisions can always be replayed without a network.
Persisted decisions carry the exact input bodies, so reload repeats the evidence
checks instead of trusting a saved ``resolved`` flag or a caller's assertions.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup


SCHEMA_VERSION = 1
NAME_BASIS = "rosterwatch_phase2_name_ladder_v1"
BOOKING_BASIS = "rosterwatch_phase2_official_booking_v1"
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_EMPTY = {"", "na", "nan", "none", "null", "--"}
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[a-f0-9]{16}\Z")
# Existing approvals are scoped to complete names and the permanent page. Their
# documentary provenance is frozen; none grants a nickname rule for a newcomer.
REVIEWED_GIVEN_NAME_VARIANTS = (
    {"names": ["Nick Galanti", "Nicholas Galanti"],
     "page_url": "http://ufcstats.com/fighter-details/8988705468d04ef7",
     "source": "owner_decision_2_2026-09-03",
     "evidence_document_sha256": "ac10f8ab18a76d560895987468d16ca350be9ce026950c1801e22f06815fbcdf"},
    {"names": ["Nikolija Milosevic", "Nina Milosevic"],
     "page_url": "http://ufcstats.com/fighter-details/eabece0edc2d62ca",
     "source": "reviewed_historical_alias_source_specific_names_2026-09-03",
     "evidence_document_sha256": "ac10f8ab18a76d560895987468d16ca350be9ce026950c1801e22f06815fbcdf"},
)
_CHALLENGE = re.compile(
    r"cf-chl-|challenge-platform|checking your browser|just a moment|/__c/|"
    r"enable javascript and cookies to continue|access denied|captcha", re.I
)


def _text(value: object) -> str:
    value = "" if value is None else str(value).strip()
    return "" if value.casefold() in _EMPTY else value


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def normalize_identity_name(name: object) -> str:
    """The charter's first rung, with suffixes removed as whole tokens."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", _text(name).casefold())
                    if not unicodedata.combining(c))
    plain = "".join(" " if unicodedata.category(c).startswith("P") else c for c in plain)
    return " ".join(token for token in plain.split() if token not in _SUFFIXES)


def name_ladder_rung(
    source_name: object, page_name: object, *, variants: Sequence[Mapping] = (),
    page_url: str = "", captures: Sequence[Mapping] = (),
) -> int | None:
    """Compare names only. A returned rung is never an identity decision.

    Rung 3 tables are page-scoped evidence, not a universal nickname dictionary.
    The variant's captured Person must explicitly connect both complete names.
    """
    left, right = normalize_identity_name(source_name), normalize_identity_name(page_name)
    if not left or not right:
        return None
    if left == right or left.replace(" ", "") == right.replace(" ", ""):
        return 1
    a, b = Counter(left.split()), Counter(right.split())
    if a <= b or b <= a:
        return 2
    if left.split()[-1] != right.split()[-1]:
        return None
    for variant in REVIEWED_GIVEN_NAME_VARIANTS:
        if variant["page_url"] == page_url and {left, right} == {normalize_identity_name(n) for n in variant["names"]}:
            return 3
    for variant in variants:
        if not _same_identity_page(_text(variant.get("page_url")), page_url):
            continue
        names = {normalize_identity_name(n) for n in variant.get("names", [])}
        if {left, right} != names:
            continue
        for capture in captures:
            if capture.get("decoded_sha256") != variant.get("capture_sha256"):
                continue
            for person in _people(capture):
                aliases = {normalize_identity_name(n) for n in person["names"]}
                if {left, right} <= aliases and any(_same_identity_page(page_url, url) for url in person["identity_urls"]):
                    return 3
    return None


def _iso_date(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass
    text = text.replace("Sept.", "Sep").replace(".", "")
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def _url(value: object, kind: str = "") -> str:
    text = _text(value)
    try:
        parsed = urlsplit(text)
        if parsed.username or parsed.password or parsed.port or parsed.fragment:
            return ""
    except ValueError:
        return ""
    host, path = (parsed.hostname or "").casefold(), parsed.path
    if host == "ufcstats.com" and parsed.scheme == "http":
        if re.fullmatch(r"/(fighter|event|fight)-details/[a-f0-9]{16}", path) and not parsed.query:
            return text if not kind or path.startswith("/" + kind + "-details/") else ""
        if path == "/statistics/fighters" and (not kind or kind == "listing"):
            query = parse_qs(parsed.query)
            if set(query) == {"char", "page"} and query["page"] == ["all"] and re.fullmatch(r"[a-z]", query["char"][0]):
                return text
    if host in {"www.ufc.com", "ufc.com"} and parsed.scheme == "https" and not parsed.query:
        if re.fullmatch(r"/(athlete|event)/[^/?#]+", path):
            if not kind or path.startswith("/" + kind + "/"):
                return text
    return ""


def surname_listing_url(name: object) -> str:
    tokens = normalize_identity_name(name).split()
    initial = tokens[-1][0] if tokens else ""
    return f"http://ufcstats.com/statistics/fighters?char={initial}&page=all" if re.fullmatch("[a-z]", initial) else ""


def _ufc_carrier_value(url: str) -> str:
    """Existing registry's logical key; never used to propose a request URL."""
    return "https://www.ufc.com" + urlsplit(url).path.casefold() if _url(url, "athlete") else ""


def _identity_page_key(url: str) -> str:
    return _ufc_carrier_value(url) or url


def _same_identity_page(left: str, right: str) -> bool:
    return bool(left and right) and _identity_page_key(left) == _identity_page_key(right)


def _supported_ufc_carrier(url: str) -> bool:
    return bool(re.fullmatch(r"https://www\.ufc\.com/athlete/[a-z0-9-]+", _ufc_carrier_value(url)))


def _logical_carrier(carrier: str) -> str:
    if carrier.startswith("ufc:"):
        return "ufc:" + (_ufc_carrier_value(carrier[4:]) or carrier[4:])
    return carrier


def _occupied_conflict(carrier: str, owner: str, occupied: Mapping) -> bool:
    key = _logical_carrier(carrier)
    return any(_logical_carrier(_text(candidate)) == key and claimant != owner
               for candidate, claimant in occupied.items())


def _normalize_capture(record: Mapping) -> dict:
    capture = dict(record.get("capture") or record)
    url = _text(capture.get("url") or record.get("url"))
    if not _url(url):
        # Other permitted captured pages can supply links, but not page identity.
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc not in {"www.ufc.com", "ufc.com", "www.roster.watch", "roster.watch"}:
            raise ValueError("Identity capture URL is outside the permitted source hosts")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Invalid captured source URL")
    if "body_text" in capture:
        body = capture["body_text"].encode("utf-8")
    elif "body_base64" in capture:
        body = base64.b64decode(capture["body_base64"], validate=True)
    elif "body" in capture:
        body = capture["body"]
        body = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    elif capture.get("decoded_path"):
        body = Path(capture["decoded_path"]).read_bytes()
    else:
        body = b""
    expected = _text(capture.get("decoded_sha256") or capture.get("sha256"))
    if not _HASH.fullmatch(expected) or hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Identity capture hash mismatch")
    if capture.get("decoded_size", len(body)) != len(body):
        raise ValueError("Identity capture size mismatch")
    captured_at = _text(capture.get("captured_at") or capture.get("completed_at_utc") or record.get("completed_at_utc"))
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("Identity capture requires a zoned acquisition timestamp") from exc
    error = _text(record.get("error") or capture.get("error") or capture.get("decode_error")
                  or capture.get("read_error") or capture.get("source_unavailable_reason"))
    try:
        body_text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        body_text, error = "", "capture_not_utf8"
    eligible = (capture.get("semantic_evidence_eligible") is True
                and capture.get("capture_complete") is True
                and capture.get("decode_complete") is True
                and capture.get("status_code", record.get("status_code")) == 200
                and capture.get("ufcstats_challenge_detected") is not True
                and bool(body) and not error and not _CHALLENGE.search(body_text))
    # Persist bytes in base64: it preserves all captured bytes even for a rejected
    # non-UTF8 response and is independent of an external path on later reload.
    return {"url": url, "body_base64": base64.b64encode(body).decode("ascii"),
            "decoded_sha256": expected, "decoded_size": len(body), "captured_at": captured_at,
            "status_code": capture.get("status_code", record.get("status_code")),
            "capture_complete": capture.get("capture_complete") is True,
            "decode_complete": capture.get("decode_complete") is True,
            "semantic_evidence_eligible": bool(eligible),
            "error": error or ("source_unavailable_or_incomplete" if not eligible else "")}


def _soup(capture: Mapping) -> BeautifulSoup | None:
    if not capture.get("semantic_evidence_eligible"):
        return None
    return BeautifulSoup(base64.b64decode(capture["body_base64"]).decode("utf-8"), "html.parser")


def _walk_json(value: object):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


_UNPARSED = object()


def _people(capture: Mapping, *, soup=_UNPARSED) -> list[dict]:
    soup = _soup(capture) if soup is _UNPARSED else soup
    if soup is None or not _url(capture["url"], "athlete"):
        return []
    people = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        for item in _walk_json(payload):
            types = item.get("@type", [])
            if "Person" not in ([types] if isinstance(types, str) else types):
                continue
            person_url = _url(item.get("url") or item.get("@id"), "athlete")
            captured_url = _url(capture["url"], "athlete")
            if not _same_identity_page(person_url, captured_url) or not _text(item.get("name")):
                continue
            alternate = item.get("alternateName", [])
            alternate = [alternate] if isinstance(alternate, str) else alternate
            same_as = item.get("sameAs", [])
            same_as = [same_as] if isinstance(same_as, str) else same_as
            people.append({"name": _text(item["name"]), "names": [_text(item["name"])] + [_text(n) for n in alternate],
                           "date_of_birth": _iso_date(item.get("birthDate")),
                           "published_dob": _text(item.get("birthDate")),
                           "identity_urls": list(dict.fromkeys([captured_url, person_url] + [u for v in same_as if (u := _url(v))])),
                           "url": captured_url, "capture_sha256": capture["decoded_sha256"]})
    return people


def _links(capture: Mapping, *, soup=_UNPARSED) -> set[str]:
    tree = _soup(capture) if soup is _UNPARSED else soup
    if tree is None:
        return set()
    links = {_url(urljoin(capture["url"], a.get("href", ""))) for a in tree.select("a[href]")}
    for person in _people(capture, soup=soup):
        links.update(person["identity_urls"])
    return links - {""}


def _named_athlete_links(capture: Mapping, *, soup=_UNPARSED) -> dict[str, set[str]]:
    soup = _soup(capture) if soup is _UNPARSED else soup
    result = {}
    if soup is not None:
        for link in soup.select("a[href]"):
            url = _url(urljoin(capture["url"], link["href"]), "athlete")
            if url:
                result.setdefault(url, set()).add(link.get_text(" ", strip=True))
    return result


def parse_ufcstats_listing(capture: Mapping) -> list[dict]:
    """Parse a hash-checked surname-initial listing, retaining page ownership."""
    capture = _normalize_capture(capture)
    return _listing_rows(capture)


def _listing_rows(capture: Mapping, *, soup=_UNPARSED) -> list[dict]:
    if not _url(capture["url"], "listing"):
        return []
    soup = _soup(capture) if soup is _UNPARSED else soup
    if soup is None:
        return []
    rows = []
    for tr in soup.select("table.b-statistics__table tbody tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        first = cells[0].find("a", href=True)
        last = cells[1].find("a", href=True)
        if first is None or last is None:
            continue
        page = _url(urljoin(capture["url"], first["href"]), "fighter")
        if not page or page != _url(urljoin(capture["url"], last["href"]), "fighter"):
            continue
        rows.append({"page": page, "name": " ".join([first.get_text(" ", strip=True), last.get_text(" ", strip=True)]),
                     "surname": last.get_text(" ", strip=True), "listing_url": capture["url"],
                     "capture_sha256": capture["decoded_sha256"]})
    return rows


def _stats_bouts(soup: BeautifulSoup, page: str, default_date: str = "") -> list[dict]:
    bouts = []
    for tr in soup.select("table.b-fight-details__table tbody tr"):
        fighters = []
        for link in tr.select("a[href]"):
            url = _url(urljoin(page, link["href"]), "fighter")
            if url and url not in [f["url"] for f in fighters]:
                fighters.append({"url": url, "name": link.get_text(" ", strip=True)})
        if len(fighters) != 2:
            continue
        event_link = next((a for a in tr.select("a[href]") if _url(urljoin(page, a["href"]), "event")), None)
        event_url = _url(urljoin(page, event_link["href"]), "event") if event_link else _url(page, "event")
        bout_date = default_date
        if event_link:
            container = event_link.find_parent("td")
            for p in container.find_all("p") if container else []:
                bout_date = _iso_date(p.get_text(" ", strip=True)) or bout_date
        if event_url and bout_date:
            bouts.append({"date": bout_date, "event_url": event_url, "fighters": fighters})
    return bouts


def _stats_profile(capture: Mapping, *, soup=_UNPARSED) -> dict | None:
    if not _url(capture["url"], "fighter"):
        return None
    soup = _soup(capture) if soup is _UNPARSED else soup
    if soup is None:
        return None
    names = soup.select(".b-content__title-highlight")
    if len(names) != 1 or not names[0].get_text(" ", strip=True):
        return None
    dobs = []
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if re.match(r"^DOB\s*:", text, re.I):
            dobs.append(re.sub(r"^DOB\s*:\s*", "", text, flags=re.I))
    if len(dobs) != 1:
        return None
    record = soup.select_one(".b-content__title-record")
    return {"site": "ufcstats", "page": capture["url"], "name": names[0].get_text(" ", strip=True),
            "date_of_birth": _iso_date(dobs[0]), "published_dob": _text(dobs[0]),
            "record": record.get_text(" ", strip=True) if record else "",
            "bouts": _stats_bouts(soup, capture["url"]), "capture_sha256": capture["decoded_sha256"]}


def _events(capture: Mapping, *, soup=_UNPARSED) -> list[dict]:
    page = capture["url"]
    soup = _soup(capture) if soup is _UNPARSED else soup
    if soup is None:
        return []
    if _url(page, "event") and urlsplit(page).hostname == "ufcstats.com":
        dates = []
        for li in soup.select("li.b-list__box-list-item"):
            text = li.get_text(" ", strip=True)
            if re.match(r"^Date\s*:", text, re.I):
                dates.append(_iso_date(re.sub(r"^Date\s*:\s*", "", text, flags=re.I)))
        if len(set(dates)) != 1 or not dates[0]:
            return []
        return [{**b, "site": "ufcstats", "capture_sha256": capture["decoded_sha256"]}
                for b in _stats_bouts(soup, page, dates[0])]
    if not _url(page, "event"):
        return []
    times = soup.select('.field--name-fight-card-time-main time[datetime]')
    dates = {_iso_date(t["datetime"]) for t in times} - {""}
    if len(dates) != 1:
        return []
    bouts = []
    for row in soup.select(".c-listing-fight__names-row"):
        fighters = []
        for corner in row.select(".c-listing-fight__corner-name"):
            link = corner.find("a", href=True)
            url = _url(urljoin(page, link["href"]), "athlete") if link else ""
            if url:
                fighters.append({"url": url, "name": link.get_text(" ", strip=True)})
        if len(fighters) == 2 and not _same_identity_page(fighters[0]["url"], fighters[1]["url"]):
            bouts.append({"date": next(iter(dates)), "event_url": page, "fighters": fighters,
                          "site": "ufc", "capture_sha256": capture["decoded_sha256"]})
    return bouts


def _source_context(record: Mapping, supplied: Mapping | None) -> dict:
    supplied = dict(supplied or {})
    context = dict(supplied.get("row") or supplied)
    context["fighter"] = _text(context.get("fighter") or record.get("current_source_name") or record.get("rw_fighter_raw") or record.get("rw_fighter") or record.get("fighter"))
    context["date_of_birth"] = _text(context.get("date_of_birth") or record.get("current_source_dob") or record.get("rw_dob_raw") or record.get("rw_dob") or record.get("date_of_birth"))
    context["snapshot_sha256"] = _text(supplied.get("snapshot_sha256") or supplied.get("source_sha256") or context.get("snapshot_sha256"))
    for key in ("next_date", "next_opp", "prev_bout", "prev_result", "debut", "debut_result", "wins", "losses", "bouts"):
        context[key] = _text(context.get(key))
    return context


def _confirmed_opponent(name: str, opponents: Sequence[Mapping]) -> Mapping | None:
    matches = []
    for opponent in opponents:
        if opponent.get("external_carrier_status") != "resolved":
            continue
        names = [opponent.get(key) for key in ("current_source_name", "rw_fighter", "rw_fighter_raw", "fighter", "official_name", "current_name")]
        if any(normalize_identity_name(n) == normalize_identity_name(name) for n in names if _text(n)):
            matches.append(opponent)
    owners = {_text(m.get("internal_owner_id")) for m in matches}
    return matches[0] if len(owners) == 1 and len(matches) == 1 and "" not in owners else None


def _opponent_urls(opponent: Mapping) -> tuple[str, str]:
    stats = _text(opponent.get("ufcstats_url"))
    value = _text(opponent.get("preferred_carrier_value"))
    if opponent.get("preferred_carrier_type") == "ufcstats" and _ID.fullmatch(value):
        stats = "http://ufcstats.com/fighter-details/" + value
    official = _text(opponent.get("official_athlete_url") or opponent.get("canonical_athlete_url"))
    if opponent.get("preferred_carrier_type") == "ufc":
        official = value
    return _url(stats, "fighter"), _url(official, "athlete")


_PARSE_CACHE_MAX_BYTES = 32 * 1024 * 1024
_PARSE_CACHE_MAX_ENTRIES = 128
_PARSE_CACHE_EPOCH = "rosterwatch_capture_projection_v1"


def _capture_projection(capture: Mapping) -> dict:
    """Extract plain document facts; no owner, freshness or decision is cached."""
    soup = _soup(capture)
    profile = _stats_profile(capture, soup=soup)
    people = _people(capture, soup=soup)
    events = _events(capture, soup=soup)
    links = _links(capture, soup=soup)
    listings = None
    if _url(capture["url"], "listing"):
        # Preserve the public listing parser's additional byte checks.
        listings = _listing_rows(_normalize_capture(capture), soup=soup)
    named = _named_athlete_links(capture, soup=soup)
    return {"profile": profile, "people": people, "events": events,
            "links": sorted(links), "listings": listings,
            "named_athlete_links": {url: sorted(names) for url, names in named.items()}}


class _CaptureParseCache:
    """Private, bounded memo for one synchronous complete binding-validation pass.

    The constructor still normalizes every input before this cache is consulted.
    Retained values are immutable serialized extraction bytes, never trees,
    providers, normalized captures or decisions. Every hit returns fresh nested
    values. The caller clears this cache on return/error; no process-wide cache
    or persisted reuse exists. Byte accounting bounds encoded retained values;
    transient parser/provider objects remain subject to the enclosing limits.
    """
    def __init__(self):
        self.entries = OrderedDict()
        self.retained_bytes = 0

    def clear(self):
        self.entries.clear()
        self.retained_bytes = 0

    def extract(self, capture: Mapping) -> dict:
        # Full normalized metadata matters: a body hash alone loses URL,
        # eligibility and acquisition time (including default event dates).
        # Function identities also keep test/parser replacements out of a hit.
        key = (_PARSE_CACHE_EPOCH, _digest(capture), _capture_projection,
               _soup, _stats_profile, _stats_bouts, _people, _walk_json, _events,
               _links, _listing_rows, _named_athlete_links, _url, _text,
               _iso_date, _identity_page_key, _same_identity_page,
               _ufc_carrier_value, BeautifulSoup)
        encoded = self.entries.get(key)
        if encoded is not None:
            if _url(capture["url"], "listing"):
                _normalize_capture(capture)
            self.entries.move_to_end(key)
            return json.loads(encoded)
        projection = _capture_projection(capture)
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
        if len(encoded) > _PARSE_CACHE_MAX_BYTES or _PARSE_CACHE_MAX_ENTRIES <= 0:
            return projection
        while self.entries and (len(self.entries) >= _PARSE_CACHE_MAX_ENTRIES
                                or self.retained_bytes + len(encoded) > _PARSE_CACHE_MAX_BYTES):
            _, previous = self.entries.popitem(last=False)
            self.retained_bytes -= len(previous)
        self.entries[key] = encoded
        self.retained_bytes += len(encoded)
        return json.loads(encoded)


class CapturedIdentityProvider:
    """Search and decide using reviewed captures; never sends a request.

    ``requested_urls`` are unmet evidence needs, not executed requests. A caller
    may place them in a new independently reviewed transport plan. A manifest can
    use transport response records or normalized embedded captures.
    """
    def __init__(self, captures: Sequence[Mapping], *, variants: Sequence[Mapping] = (),
                 _parsed_capture_cache=None):
        if _parsed_capture_cache is not None and type(_parsed_capture_cache) is not _CaptureParseCache:
            raise TypeError("Internal capture parsing cache required")
        self.captures = [_normalize_capture(capture) for capture in captures]
        # Identical repeated input does not introduce ambiguity or alias growth.
        self.captures = list({_digest(c): c for c in self.captures}.values())
        self.variants = [dict(v) for v in variants]
        self.profiles = []
        self.people = []
        self.events = []
        self.links = set()
        self.listings = {}
        self.named_athlete_links = {}
        for capture in self.captures:
            facts = (_capture_projection(capture) if _parsed_capture_cache is None
                     else _parsed_capture_cache.extract(capture))
            if profile := facts["profile"]:
                self.profiles.append(profile)
            self.people.extend(facts["people"])
            self.events.extend(facts["events"])
            self.links.update(facts["links"])
            if facts["listings"] is not None:
                self.listings[capture["decoded_sha256"]] = facts["listings"]
            for url, names in facts["named_athlete_links"].items():
                self.named_athlete_links.setdefault(url, set()).update(names)

    def _rung(self, source: str, page: str, url: str) -> int | None:
        return name_ladder_rung(source, page, variants=self.variants, page_url=url, captures=self.captures)

    def _alias_connection(self, name: str, profile: Mapping) -> list[dict]:
        result = []
        for person in self.people:
            if profile["page"] not in person["identity_urls"]:
                continue
            if any(self._rung(name, n, person["url"]) in (1, 2, 3) for n in person["names"]):
                result.append({"kind": "published_identity_relationship", "url": person["url"],
                               "capture_sha256": person["capture_sha256"], "published_names": person["names"]})
        return result

    def _bout_support(self, profile: Mapping, context: Mapping, opponents: Sequence[Mapping], *, booking: bool,
                      occupied: Mapping | None = None) -> list[dict]:
        if booking:
            source_date, opponent_name = _iso_date(context.get("next_date")), _text(context.get("next_opp"))
        else:
            source_date = _iso_date(context.get("prev_bout"))
            result = re.fullmatch(r"(?:W|L|D|NC)\s*-\s*(.+)", _text(context.get("prev_result")), re.I)
            opponent_name = result.group(1) if result else ""
        opponent = _confirmed_opponent(opponent_name, opponents) if opponent_name and source_date else None
        if opponent is None:
            return []
        stats_url, official_url = _opponent_urls(opponent)
        if not stats_url or not official_url or stats_url == profile["page"]:
            return []
        if occupied is not None and occupied.get("ufcstats:" + stats_url.rsplit("/", 1)[1]) != opponent["internal_owner_id"]:
            return []
        stats_bouts = [e for e in self.events if e["site"] == "ufcstats"] if booking else profile.get("bouts", [])
        supporting = []
        for bout in stats_bouts:
            if bout["date"] != source_date or {f["url"] for f in bout["fighters"]} != {profile["page"], stats_url}:
                continue
            for official in self.events:
                if official["site"] != "ufc" or official["date"] != source_date:
                    continue
                other = [f for f in official["fighters"] if _ufc_carrier_value(f["url"]) != _ufc_carrier_value(official_url)]
                if len(other) != 1 or _ufc_carrier_value(official_url) not in {_ufc_carrier_value(f["url"]) for f in official["fighters"]}:
                    continue
                target = other[0]
                # The candidate and corroborating official card must agree by a
                # normal name rung or an explicit same-person link, never DOB alone.
                connected = any(profile["page"] in p["identity_urls"] and _ufc_carrier_value(p["url"]) == _ufc_carrier_value(target["url"]) for p in self.people)
                if not connected and self._rung(profile["name"], target["name"], target["url"]) not in (1, 2, 3):
                    continue
                contradicting = [p for p in self.people if _ufc_carrier_value(p["url"]) == _ufc_carrier_value(target["url"]) and p["published_dob"]
                                 and p["date_of_birth"] != context["date_of_birth"]]
                supporting.append({"kind": "official_booking" if booking else "independent_bout",
                                   "date": source_date, "opponent_owner": opponent["internal_owner_id"],
                                   "opponent_ufcstats_url": stats_url, "opponent_ufc_url": official_url,
                                   "ufcstats_event_url": bout["event_url"],
                                   "ufcstats_capture_sha256": bout.get("capture_sha256", profile["capture_sha256"]),
                                   "ufc_event_url": official["event_url"],
                                   "ufc_capture_sha256": official["capture_sha256"],
                                   "published_dob_conflict": bool(contradicting),
                                   "official_athlete_url": target["url"], "official_name": target["name"]})
        return supporting

    def _candidates(self, context: Mapping, opponents: Sequence[Mapping],
                    occupied: Mapping | None = None) -> tuple[list[dict], set[str], list[str]]:
        name = context["fighter"]
        listing = surname_listing_url(name)
        requested, reasons, selected = set(), [], {}
        matching_captures = [c for c in self.captures if c["url"] == listing]
        if not listing:
            reasons.append("surname_initial_not_supported")
        elif not matching_captures:
            requested.add(listing)
            reasons.append("ufcstats_listing_not_captured")
        elif not all(c["semantic_evidence_eligible"] for c in matching_captures):
            reasons.append("ufcstats_listing_unavailable")
        surname = normalize_identity_name(name).split()[-1] if normalize_identity_name(name) else ""
        for capture in matching_captures:
            for row in self.listings.get(capture["decoded_sha256"], []):
                candidate_surname = normalize_identity_name(row["surname"])
                # Prefixes widen discovery only; materially different names
                # still need the independent rung-4 evidence below.
                surname_hit = surname and (surname in normalize_identity_name(row["name"]).split()
                                           or (len(surname) >= 4 and candidate_surname.startswith(surname)))
                if surname_hit:
                    selected[row["page"]] = row
                    if not any(p["page"] == row["page"] for p in self.profiles):
                        requested.add(row["page"])
        # Bookings can discover an entirely differently named fighter through
        # the confirmed opponent's already evidenced page and its bout links.
        opponent = _confirmed_opponent(_text(context.get("next_opp")), opponents)
        booking_date = _iso_date(context.get("next_date"))
        if opponent is not None and booking_date:
            stats_url, _ = _opponent_urls(opponent)
            if stats_url and not any(p["page"] == stats_url for p in self.profiles):
                requested.add(stats_url)
            for profile in self.profiles:
                if profile["page"] != stats_url:
                    continue
                for bout in profile.get("bouts", []):
                    if bout["date"] != booking_date:
                        continue
                    if bout["event_url"] not in {c["url"] for c in self.captures}:
                        requested.add(bout["event_url"])
                    for fighter in bout["fighters"]:
                        if fighter["url"] != stats_url:
                            selected.setdefault(fighter["url"], {"page": fighter["url"], "name": fighter["name"],
                                "opponent_profile": stats_url, "capture_sha256": profile["capture_sha256"]})
                            if not any(p["page"] == fighter["url"] for p in self.profiles):
                                requested.add(fighter["url"])
        candidates = []
        for profile in self.profiles:
            alias = self._alias_connection(name, profile)
            booking = self._bout_support(profile, context, opponents, booking=True, occupied=occupied)
            history = self._bout_support(profile, context, opponents, booking=False, occupied=occupied)
            if profile["page"] in selected or alias or booking or history:
                candidates.append({**profile, "discovery": selected.get(profile["page"]),
                                   "alias_support": alias, "booking_support": booking, "history_support": history})
                for bout in profile.get("bouts", []):
                    if bout["date"] == _iso_date(context.get("next_date")) and bout["event_url"] not in {c["url"] for c in self.captures}:
                        requested.add(bout["event_url"])
        for url, names in self.named_athlete_links.items():
            if _url(url, "athlete") and not any(_same_identity_page(p["url"], url) for p in self.people):
                # Only request athlete links whose anchor text matches this
                # member, or links from an already evidenced relevant bout.
                if any(self._rung(name, anchor, url) in (1, 2, 3) for anchor in names):
                    requested.add(url)
        for page, row in selected.items():
            if not any(p["page"] == page for p in candidates):
                matching = [c for c in self.captures if c["url"] == page]
                reasons.append("candidate_page_unavailable" if matching else "candidate_page_not_captured")
                candidates.append({"site": "ufcstats", "page": page, "name": row["name"], "date_of_birth": "",
                                   "capture_sha256": "", "discovery": row, "unavailable": True})
        return candidates, requested, sorted(set(reasons))

    def lookup(self, record: Mapping, site: str, *, source_context: Mapping | None = None,
               confirmed_opponents: Sequence[Mapping] | None = None) -> dict:
        context = _source_context(record, source_context)
        candidates, requested, reasons = self._candidates(context, confirmed_opponents or [])
        pages = [{"site": p["site"], "page": p["page"], "name": p["name"],
                  "date_of_birth": p["date_of_birth"], "evidence": {"capture_sha256": p["capture_sha256"], "discovery": p.get("discovery")},
                  "reason_not_bound": "candidate_page_unavailable" if p.get("unavailable") else "identity_decision_required"}
                 for p in candidates if site == "ufcstats"]
        if site == "ufc":
            pages = [{"site": "ufc", "page": p["url"], "name": p["name"], "date_of_birth": p["date_of_birth"],
                      "evidence": {"capture_sha256": p["capture_sha256"]}, "reason_not_bound": "identity_decision_required"}
                     for p in self.people if self._rung(context["fighter"], p["name"], p["url"]) or
                     any(self._rung(context["fighter"], n, p["url"]) for n in p["names"])]
            reasons = self._ufc_route_reasons(context, candidates)
        return {"candidate_pages": pages, "request_count": 0, "requested_urls": sorted(requested), "reason_codes": reasons}

    def _ufc_route_reasons(self, context: Mapping, candidates: Sequence[Mapping]) -> list[str]:
        """Distinguish absent route evidence from a captured-but-failed page."""
        name = context["fighter"]
        routes = {url for url, names in self.named_athlete_links.items()
                  if any(self._rung(name, anchor, url) in (1, 2, 3) for anchor in names)}
        routes.update(p["url"] for p in self.people
                      if any(self._rung(name, alias, p["url"]) in (1, 2, 3) for alias in p["names"]))
        for candidate in candidates:
            routes.update(s["official_athlete_url"] for key in ("booking_support", "history_support")
                          for s in candidate.get(key, []) if s.get("official_athlete_url"))
            routes.update(p["url"] for p in self.people if candidate["page"] in p["identity_urls"])
        if not routes:
            return ["ufc_athlete_route_not_evidenced"]
        if any(_same_identity_page(p["url"], route) for p in self.people for route in routes):
            return []
        if any(_same_identity_page(c["url"], route) for c in self.captures for route in routes):
            return ["ufc_athlete_page_unavailable"]
        return ["ufc_athlete_page_not_captured"]

    def decide(self, record: Mapping, observations: object = None, *, source_context: Mapping | None = None,
               confirmed_opponents: Sequence[Mapping] | None = None, occupied_carriers: Mapping | None = None,
               checked_at: str | None = None) -> dict:
        del observations  # Caller-written candidates/booleans cannot prove identity.
        context = _source_context(record, source_context)
        opponents = list(confirmed_opponents or [])
        occupied = dict(occupied_carriers or {})
        timestamp = checked_at or datetime.now(timezone.utc).isoformat()
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed_time.tzinfo is None:
            raise ValueError("Identity decision date requires timezone")
        owner = _text(record.get("internal_owner_id"))
        if not owner or not context["fighter"] or _iso_date(context["date_of_birth"]) != context["date_of_birth"] or not _HASH.fullmatch(context["snapshot_sha256"]):
            raise ValueError("Identity decision requires owner, source name, ISO DOB and snapshot hash")
        candidates, requested, reasons = self._candidates(context, opponents, occupied)
        eligible, pages = [], []
        for profile in candidates:
            reason = "candidate_page_unavailable" if profile.get("unavailable") else ""
            rung = self._rung(context["fighter"], profile["name"], profile["page"])
            supporting = profile.get("alias_support", []) + profile.get("history_support", [])
            identity_support = bool(supporting)
            booking = profile.get("booking_support", [])
            bout_support = profile.get("history_support", []) + booking
            dob = profile["date_of_birth"]
            published = profile.get("published_dob", "")
            # Every published nonempty DOB is binding evidence: malformed or
            # contradictory values cannot be ignored as if absent.
            if not reason and published and dob != context["date_of_birth"]:
                reason = "published_dob_conflict" if dob else "published_dob_invalid"
            linked_pages = {_identity_page_key(p["url"]) for p in self.people if profile["page"] in p["identity_urls"]}
            linked_people = [p for p in self.people if _identity_page_key(p["url"]) in linked_pages]
            if not reason and any(p["published_dob"] and p["date_of_birth"] != context["date_of_birth"] for p in linked_people):
                reason = "corroborating_published_dob_conflict"
            if not reason and any(item.get("published_dob_conflict") for item in bout_support):
                reason = "corroborating_published_dob_conflict"
            if not reason and len({_ufc_carrier_value(item["official_athlete_url"]) for item in bout_support}) > 1:
                reason = "ambiguous_official_booking_targets"
            if not reason and any(not _supported_ufc_carrier(item["official_athlete_url"]) for item in bout_support):
                reason = "unsupported_ufc_carrier_shape"
            if not reason and any(_occupied_conflict("ufc:" + item["official_athlete_url"], owner, occupied) for item in bout_support):
                reason = "corroborating_page_owned_by_another_fighter"
            if not dob and not published and any(p["date_of_birth"] == context["date_of_birth"] for p in linked_people):
                dob = context["date_of_birth"]
                supporting += [{"kind": "linked_source_dob", "url": p["url"], "capture_sha256": p["capture_sha256"]}
                               for p in linked_people if p["date_of_birth"] == dob]
            basis = NAME_BASIS
            if not reason and booking:
                basis, rung, supporting = BOOKING_BASIS, None, booking
            elif not reason and not dob:
                reason = "published_dob_missing_without_same_identity_source"
            elif not reason and rung is None and identity_support:
                rung = 4
            elif not reason and rung is None:
                reason = "material_name_difference_needs_independent_evidence"
            carrier = "ufcstats:" + profile["page"].rsplit("/", 1)[1]
            if not reason:
                reason = _carrier_conflict(record, carrier, occupied)
            page = {"site": "ufcstats", "page": profile["page"], "name": profile["name"],
                    "date_of_birth": profile["date_of_birth"], "ladder_rung": rung,
                    "reason_not_bound": reason, "evidence": {"capture_sha256": profile["capture_sha256"],
                    "discovery": profile.get("discovery"), "supporting": supporting}}
            pages.append(page)
            if not reason:
                official = next((p for p in linked_people if p["date_of_birth"] == context["date_of_birth"]), None)
                official_url = official["url"] if official else (booking[0]["official_athlete_url"] if booking else "")
                official_name = official["name"] if official else (booking[0]["official_name"] if booking else "")
                if official_url and not _supported_ufc_carrier(official_url):
                    page["reason_not_bound"] = "unsupported_ufc_carrier_shape"
                    continue
                if official_url and _occupied_conflict("ufc:" + official_url, owner, occupied):
                    page["reason_not_bound"] = "corroborating_page_owned_by_another_fighter"
                    continue
                eligible.append({"carrier_type": "ufcstats", "carrier_value": carrier.split(":", 1)[1],
                                 "source_names": {"rosterwatch": context["fighter"], "ufcstats": profile["name"], "ufc": official_name},
                                 "matching_basis": basis, "ladder_rung": rung, "page": profile["page"],
                                 "official_athlete_url": _ufc_carrier_value(official_url), "record": profile.get("record", "")})
        # UFC-only identities use a positively identified athlete Person, with
        # the same DOB/name requirements. Multiple pages remain ambiguous.
        official_candidates = []
        for person in self.people:
            rung = self._rung(context["fighter"], person["name"], person["url"])
            alias = any(self._rung(context["fighter"], n, person["url"]) in (1, 2, 3) for n in person["names"][1:])
            if rung is None and not alias:
                continue
            reason = "" if person["date_of_birth"] == context["date_of_birth"] else (
                "published_dob_conflict" if person["published_dob"] else "published_dob_missing_without_same_identity_source")
            peers = [other for other in self.people if _same_identity_page(other["url"], person["url"])]
            # Every observation of this logical athlete contributes its links,
            # including a peer with a different published name or source host.
            identity_urls = {url for peer in peers for url in peer["identity_urls"]}
            if not reason and any(other["published_dob"] and other["date_of_birth"] != context["date_of_birth"] for other in peers):
                reason = "corroborating_published_dob_conflict"
            if not reason and any(profile["page"] in identity_urls and profile.get("published_dob")
                                  and profile["date_of_birth"] != context["date_of_birth"] for profile in self.profiles):
                reason = "corroborating_published_dob_conflict"
            if not reason:
                reason = _carrier_conflict(record, "ufc:" + person["url"], occupied)
            pages.append({"site": "ufc", "page": person["url"], "name": person["name"],
                          "date_of_birth": person["date_of_birth"], "ladder_rung": rung or 4,
                          "reason_not_bound": reason, "evidence": {"capture_sha256": person["capture_sha256"]}})
            if not reason:
                official_candidates.append({"carrier_type": "ufc", "carrier_value": _ufc_carrier_value(person["url"]),
                    "source_names": {"rosterwatch": context["fighter"], "ufcstats": "", "ufc": person["name"]},
                    "matching_basis": NAME_BASIS, "ladder_rung": rung or 4, "page": person["url"],
                    "official_athlete_url": _ufc_carrier_value(person["url"]), "record": ""})
        if not eligible:
            eligible = official_candidates
        elif len({p["carrier_value"] for p in official_candidates}) > 1:
            reasons.append("ambiguous_official_pages")
            eligible = []
        eligible = list({_digest({**e, "page": _identity_page_key(e["page"])}): e for e in eligible}.values())
        # Multiple captures of one page that disagree are conflicts too.
        conflicts = [p for p in pages if "dob_conflict" in p["reason_not_bound"] or p["reason_not_bound"] == "published_dob_invalid"]
        if eligible and any(_same_identity_page(p["page"], e["page"])
                            or _same_identity_page(p["page"], e.get("official_athlete_url", ""))
                            for p in conflicts for e in eligible):
            reasons.append("conflicting_captures_for_candidate")
            eligible = []
        if len(eligible) > 1:
            reasons.append("ambiguous_candidates")
            eligible = []
        selected = eligible[0] if eligible else None
        reasons = [] if selected else sorted(set(reasons + self._ufc_route_reasons(context, candidates)
                                                 + [p["reason_not_bound"] for p in pages if p["reason_not_bound"]]))
        if not selected and not reasons:
            reasons = ["no_supported_candidate"]
        flags = []
        if selected and selected["ladder_rung"] in (3, 4):
            flags.append("name_variant")
        if selected:
            record_match = re.search(r"(?:Record:\s*)?(\d+)-(\d+)-(\d+)", selected.pop("record"))
            if record_match and context["wins"].isdigit() and context["losses"].isdigit():
                if (int(context["wins"]), int(context["losses"])) != (int(record_match[1]), int(record_match[2])):
                    flags.append("record_mismatch")
        evidence = {"captures": self.captures, "variants": self.variants, "source_context": context,
                    "decision_record": {k: _text(record.get(k)) for k in ("internal_owner_id", "preferred_carrier_type", "preferred_carrier_value")},
                    "confirmed_opponents": opponents, "checked_at": timestamp}
        decision = {"schema_version": SCHEMA_VERSION, "status": "resolved" if selected else "pending",
                    "internal_owner_id": owner, "carrier_type": "", "carrier_value": "", "source_names": {},
                    "matching_basis": "", "ladder_rung": None, "reason_codes": reasons, "soft_flags": flags,
                    "candidate_pages": pages, "requested_urls": sorted(requested), "request_count": 0,
                    "evidence": evidence, "learned_aliases": []}
        if selected:
            decision.update(selected)
            for page in decision["candidate_pages"]:
                if not _same_identity_page(page["page"], selected["page"]) and not page["reason_not_bound"]:
                    page["reason_not_bound"] = "corroborating_official_identity"
            if selected["ladder_rung"] in (2, 3, 4) or selected["matching_basis"] == BOOKING_BASIS:
                alias = {"schema_version": 1, "internal_owner_id": owner, "source": "rosterwatch",
                         "source_name": context["fighter"], "page_name": selected["source_names"].get(selected["carrier_type"], ""),
                         "page": selected["page"], "matching_basis": selected["matching_basis"],
                         "ladder_rung": selected["ladder_rung"], "date": timestamp[:10],
                         "capture_sha256": next(p["evidence"]["capture_sha256"] for p in pages if p["page"] == selected["page"])}
                alias["alias_id"] = _digest({k: v for k, v in alias.items() if k != "date"})
                decision["learned_aliases"] = [alias]
        decision["decision_sha256"] = _digest(decision)
        return decision


def _carrier_conflict(record: Mapping, carrier: str, occupied: Mapping) -> str:
    owner = _text(record.get("internal_owner_id"))
    if carrier.startswith("ufc:") and not _supported_ufc_carrier(carrier[4:]):
        return "unsupported_ufc_carrier_shape"
    existing_type = _text(record.get("preferred_carrier_type"))
    existing_value = _text(record.get("preferred_carrier_value"))
    if existing_type in {"ufcstats", "ufc"} and existing_value and _logical_carrier(existing_type + ":" + existing_value) != _logical_carrier(carrier):
        return "locked_carrier_reassignment_forbidden"
    if _occupied_conflict(carrier, owner, occupied):
        return "candidate_page_owned_by_another_fighter"
    return ""


def validate_identity_decision(decision: Mapping, record: Mapping, *, occupied_carriers: Mapping | None = None,
                               _parsed_capture_cache=None) -> dict:
    """Reproduce a persisted decision from its exact bodies; raise on any drift."""
    if decision.get("schema_version") != SCHEMA_VERSION or decision.get("status") != "resolved":
        raise ValueError("Only confirmed versioned decisions may supply persisted carriers")
    unsigned = {k: v for k, v in decision.items() if k != "decision_sha256"}
    if decision.get("decision_sha256") != _digest(unsigned):
        raise ValueError("Persisted identity decision digest mismatch")
    if decision.get("internal_owner_id") != record.get("internal_owner_id"):
        raise ValueError("Persisted identity owner mismatch")
    carrier = _text(decision.get("carrier_type")) + ":" + _text(decision.get("carrier_value"))
    conflict = _carrier_conflict(record, carrier, occupied_carriers or {})
    if conflict:
        raise ValueError(conflict)
    secondary = _text(decision.get("official_athlete_url"))
    if secondary and _occupied_conflict("ufc:" + secondary, _text(record.get("internal_owner_id")), occupied_carriers or {}):
        raise ValueError("corroborating_page_owned_by_another_fighter")
    evidence = decision.get("evidence") or {}
    context = evidence.get("source_context") or {}
    current = _source_context(record, None)
    if current["fighter"] and normalize_identity_name(current["fighter"]) != normalize_identity_name(context.get("fighter")):
        raise ValueError("Persisted source-name mismatch")
    if current["date_of_birth"] and current["date_of_birth"] != context.get("date_of_birth"):
        raise ValueError("Persisted source-DOB mismatch")
    provider = CapturedIdentityProvider(evidence.get("captures", []), variants=evidence.get("variants", []),
                                        _parsed_capture_cache=_parsed_capture_cache)
    original_record = evidence.get("decision_record") or {}
    if original_record.get("internal_owner_id") != record.get("internal_owner_id"):
        raise ValueError("Persisted decision input owner mismatch")
    replay = provider.decide(original_record, source_context=context, confirmed_opponents=evidence.get("confirmed_opponents", []),
                             occupied_carriers=occupied_carriers, checked_at=evidence.get("checked_at"))
    if replay != dict(decision):
        raise ValueError("Persisted identity evidence does not reproduce the decision")
    return replay
