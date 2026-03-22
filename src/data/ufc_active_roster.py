"""Official UFC active-roster scraping helpers."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically
from src.data.name_utils import normalize_cross_source_name, same_person_name

logger = logging.getLogger(__name__)

OFFICIAL_ACTIVE_ROSTER_URL = "https://www.ufc.com/athletes/all?filters%5B0%5D=status%3A23"
OFFICIAL_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_REQUEST_DELAY_SECONDS = 0.35


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _get_soup(url: str, *, session: requests.Session | None = None) -> BeautifulSoup:
    client = session or requests.Session()
    response = client.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(_REQUEST_DELAY_SECONDS)
    return BeautifulSoup(response.text, "lxml")


def _absolute_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return urljoin("https://www.ufc.com", text)


def _slug_to_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = urlparse(text).path if "://" in text else text
    slug = str(path).rstrip("/").rsplit("/", 1)[-1]
    return _clean_text(slug.replace("-", " "))


def _parse_roster_card(card) -> dict[str, object] | None:
    link = card.find("a", href=lambda href: href and "/athlete/" in href)
    name_el = card.select_one("span.c-listing-athlete__name")
    if link is None or name_el is None:
        return None

    division_el = card.select_one("span.c-listing-athlete__title .field__item")
    record_el = card.select_one("span.c-listing-athlete__record")
    nickname_el = card.select_one("span.c-listing-athlete__nickname")
    athlete_url = _absolute_url(link.get("href"))
    slug_name = _slug_to_name(athlete_url)

    return {
        "official_name": _clean_text(name_el.get_text(" ", strip=True)),
        "official_athlete_url": athlete_url,
        "nickname": _clean_text(nickname_el.get_text(" ", strip=True)) if nickname_el else "",
        "division": _clean_text(division_el.get_text(" ", strip=True)) if division_el else "",
        "record": _clean_text(record_el.get_text(" ", strip=True)) if record_el else "",
        "slug_name": slug_name,
    }


def _parse_alternate_slug_names(soup: BeautifulSoup) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for link in soup.select("link[rel='alternate'][href]"):
        href = str(link.get("href") or "").strip()
        if "/athlete/" not in href:
            continue
        name = _slug_to_name(href)
        key = normalize_cross_source_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _filter_alternate_slug_names(names: list[str], *, reference_name: str) -> list[str]:
    reference_tokens = [token for token in normalize_cross_source_name(reference_name).split() if token]
    if not reference_tokens:
        return names

    filtered: list[str] = []
    seen: set[str] = set()
    reference_first = reference_tokens[0]
    reference_last = reference_tokens[-1]
    for name in names:
        tokens = [token for token in normalize_cross_source_name(name).split() if token]
        if not tokens:
            continue
        if tokens[0] != reference_first and tokens[-1] != reference_last:
            continue
        key = normalize_cross_source_name(name)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(name)
    return filtered


def _parse_bio_fields(soup: BeautifulSoup) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in soup.select("div.c-bio__field"):
        label_el = field.select_one("div.c-bio__label")
        text_el = field.select_one("div.c-bio__text")
        if label_el is None or text_el is None:
            continue
        label = _clean_text(label_el.get_text(" ", strip=True))
        text = _clean_text(text_el.get_text(" ", strip=True))
        if label and text:
            fields[label] = text
    return fields


def _normalize_official_inches_measurement(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return f"{text} in"
    return text


def scrape_official_athlete_profile(
    athlete_url: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    soup = _get_soup(athlete_url, session=session)
    hero_tags = [_clean_text(tag.get_text(" ", strip=True)) for tag in soup.select("p.hero-profile__tag")]
    hero_tags = [tag for tag in hero_tags if tag]
    hero_name_el = soup.select_one("h1.hero-profile__name")
    division_el = soup.select_one("p.hero-profile__division-title")
    record_el = soup.select_one("p.hero-profile__division-body")
    hero_name = _clean_text(hero_name_el.get_text(" ", strip=True)) if hero_name_el else ""
    division = _clean_text(division_el.get_text(" ", strip=True)) if division_el else ""
    record = _clean_text(record_el.get_text(" ", strip=True)) if record_el else ""
    bio_fields = _parse_bio_fields(soup)

    status = ""
    source_tag = ""
    division_key = normalize_cross_source_name(division)
    for tag in hero_tags:
        lower = tag.casefold()
        tag_key = normalize_cross_source_name(tag)
        if "division" in lower and not division:
            division = tag
            division_key = normalize_cross_source_name(tag)
        elif lower == "active" and not status:
            status = tag
        elif (
            tag_key
            and division_key
            and (tag_key == division_key or tag_key in division_key or division_key in tag_key)
        ):
            continue
        elif not source_tag:
            source_tag = tag

    alternate_slug_names = _filter_alternate_slug_names(
        _parse_alternate_slug_names(soup),
        reference_name=hero_name or _slug_to_name(athlete_url),
    )

    return {
        "profile_name": hero_name,
        "profile_division": division,
        "profile_record": record,
        "profile_status": status,
        "profile_source_tag": source_tag,
        "birthplace": bio_fields.get("Place of Birth", ""),
        "age": bio_fields.get("Age", ""),
        "height": _normalize_official_inches_measurement(bio_fields.get("Height", "")),
        "reach": _normalize_official_inches_measurement(bio_fields.get("Reach", "")),
        "weight": bio_fields.get("Weight", ""),
        "octagon_debut": bio_fields.get("Octagon Debut", ""),
        "alternate_slug_names": "|".join(alternate_slug_names),
    }


def _load_local_ufcstats_candidates() -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = {}

    def _append_candidate(name: object, url: object, source: str) -> None:
        name_text = _clean_text(name)
        url_text = str(url or "").strip()
        key = normalize_cross_source_name(name_text)
        if not key or not url_text:
            return
        candidates.setdefault(key, []).append(
            {"name": name_text, "fighter_url": url_text, "source": source}
        )

    scraped_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    if scraped_path.exists():
        scraped_df = pd.read_csv(scraped_path, usecols=["name", "fighter_url"])
        for _, row in scraped_df.dropna(subset=["fighter_url"]).iterrows():
            _append_candidate(row.get("name"), row.get("fighter_url"), "ufc_fighters_scraped")

    details_path = RAW_DATA_DIR / "ufc-fighter-details.csv"
    if details_path.exists():
        details_df = pd.read_csv(details_path, usecols=["FIRST", "LAST", "URL"])
        name_series = (details_df["FIRST"].fillna("").astype(str) + " " + details_df["LAST"].fillna("").astype(str)).str.strip()
        for index, url in details_df["URL"].items():
            _append_candidate(name_series.iloc[index], url, "ufc_fighter_details")

    return candidates


def _resolve_local_ufcstats_profile(
    row: dict[str, object],
    *,
    candidates: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    alias_names = [
        row.get("official_name"),
        row.get("slug_name"),
        *str(row.get("alternate_slug_names") or "").split("|"),
    ]
    match_pool: dict[str, dict[str, str]] = {}
    for alias in alias_names:
        key = normalize_cross_source_name(alias)
        if not key:
            continue
        for candidate in candidates.get(key, []):
            match_pool[candidate["fighter_url"]] = candidate

    if len(match_pool) == 1:
        candidate = next(iter(match_pool.values()))
        return {
            "ufcstats_name": candidate["name"],
            "ufcstats_url": candidate["fighter_url"],
            "ufcstats_resolution": candidate["source"],
        }

    preferred: list[dict[str, str]] = []
    for candidate in match_pool.values():
        if any(same_person_name(alias, candidate["name"]) for alias in alias_names if alias):
            preferred.append(candidate)
    if len(preferred) == 1:
        candidate = preferred[0]
        return {
            "ufcstats_name": candidate["name"],
            "ufcstats_url": candidate["fighter_url"],
            "ufcstats_resolution": candidate["source"],
        }

    return {
        "ufcstats_name": "",
        "ufcstats_url": "",
        "ufcstats_resolution": "",
    }


def _resolve_via_live_search(
    row: dict[str, object],
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    from src.data.fighter_lookup import search_fighter_url
    from src.data.scraper import scrape_fighter

    queries = [
        row.get("official_name"),
        row.get("slug_name"),
        *str(row.get("alternate_slug_names") or "").split("|"),
    ]
    seen_queries: set[str] = set()
    for query in queries:
        query_text = _clean_text(query)
        key = normalize_cross_source_name(query_text)
        if not key or key in seen_queries:
            continue
        seen_queries.add(key)
        fighter_url = search_fighter_url(query_text)
        if not fighter_url:
            continue
        try:
            profile = scrape_fighter(fighter_url)
        except Exception as exc:
            logger.debug("Failed to scrape UFCStats profile %s for %s: %s", fighter_url, query_text, exc)
            profile = None
        if not profile:
            return {
                "ufcstats_name": "",
                "ufcstats_url": fighter_url,
                "ufcstats_resolution": "live_search_url_only",
            }
        return {
            "ufcstats_name": _clean_text(profile.get("name")),
            "ufcstats_url": fighter_url,
            "ufcstats_resolution": "live_search",
        }
    return {
        "ufcstats_name": "",
        "ufcstats_url": "",
        "ufcstats_resolution": "",
    }


def scrape_official_active_roster(
    *,
    fetch_profile_details: bool = True,
    max_pages: int | None = None,
    resolve_ufcstats: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    rows: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    page = 0
    candidates = _load_local_ufcstats_candidates() if resolve_ufcstats else {}

    while True:
        if max_pages is not None and page >= max_pages:
            break
        page_url = OFFICIAL_ACTIVE_ROSTER_URL if page == 0 else f"{OFFICIAL_ACTIVE_ROSTER_URL}&page={page}"
        soup = _get_soup(page_url, session=client)
        cards = soup.select("div.c-listing-athlete-flipcard")
        if not cards:
            break

        page_added = 0
        for card in cards:
            parsed = _parse_roster_card(card)
            if not parsed:
                continue
            athlete_url = str(parsed.get("official_athlete_url") or "").strip()
            if not athlete_url or athlete_url in seen_urls:
                continue
            seen_urls.add(athlete_url)
            if fetch_profile_details:
                try:
                    parsed.update(scrape_official_athlete_profile(athlete_url, session=client))
                except Exception as exc:
                    logger.warning("Failed to scrape official athlete profile %s: %s", athlete_url, exc)
            else:
                parsed.setdefault("profile_name", "")
                parsed.setdefault("profile_division", "")
                parsed.setdefault("profile_record", "")
                parsed.setdefault("profile_status", "")
                parsed.setdefault("profile_source_tag", "")
                parsed.setdefault("birthplace", "")
                parsed.setdefault("age", "")
                parsed.setdefault("height", "")
                parsed.setdefault("reach", "")
                parsed.setdefault("weight", "")
                parsed.setdefault("octagon_debut", "")
                parsed.setdefault("alternate_slug_names", "")

            if resolve_ufcstats:
                resolution = _resolve_local_ufcstats_profile(parsed, candidates=candidates)
                if not resolution.get("ufcstats_url"):
                    resolution = _resolve_via_live_search(parsed, session=client)
                parsed.update(resolution)
            rows.append(parsed)
            page_added += 1

        logger.info("Scraped official UFC active roster page %d with %d athlete cards", page, page_added)
        page += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["official_name", "official_athlete_url"]).reset_index(drop=True)
    return df


def sync_official_active_roster(
    *,
    output_path: Path | None = None,
    fetch_profile_details: bool = True,
    max_pages: int | None = None,
    resolve_ufcstats: bool = True,
) -> pd.DataFrame:
    output_path = Path(output_path) if output_path is not None else OFFICIAL_ACTIVE_ROSTER_PATH
    df = scrape_official_active_roster(
        fetch_profile_details=fetch_profile_details,
        max_pages=max_pages,
        resolve_ufcstats=resolve_ufcstats,
    )
    write_csv_atomically(df, output_path, refuse_empty=True)
    logger.info("Saved official UFC active roster with %d rows to %s", len(df), output_path)
    return df
