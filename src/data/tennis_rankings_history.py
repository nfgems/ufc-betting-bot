"""Official ATP/WTA rankings-history collection and safe match enrichment."""

from __future__ import annotations

import json
import logging
import re
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.tennis_player_profiles import canonical_player_id

logger = logging.getLogger(__name__)

TENNIS_RAW_DIR = RAW_DATA_DIR / "tennis"
TENNIS_RANKINGS_SUMMARY_PATH = PROCESSED_DATA_DIR / "tennis" / "rankings_history_enrichment_summary.json"

ATP_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles"
WTA_RANKINGS_API_URL = "https://api.wtatennis.com/tennis/players/ranked"

REQUEST_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
RANKINGS_REQUEST_PAUSE_SECONDS = 0.15
WTA_RANKINGS_PAGE_SIZE = 100
ATP_FALLBACK_RANK_RANGES = (
    "1-500",
    "501-1000",
    "1001-1500",
    "1501-2000",
    "2001-3000",
    "3001-5000",
)


def _rankings_dir_for_tour(tour: str) -> Path:
    return TENNIS_RAW_DIR / str(tour or "").lower() / "rankings"


def _rankings_snapshot_path(tour: str, snapshot_date: str) -> Path:
    return _rankings_dir_for_tour(tour) / f"{str(tour).lower()}_singles_rankings_{snapshot_date}.csv"


def _wta_snapshot_resolution_path() -> Path:
    return TENNIS_RAW_DIR / "wta" / "wta_rankings_snapshot_resolution.json"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_BROWSER_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


def _request_session(session: Optional[requests.Session] = None, *, tour: Optional[str] = None) -> requests.Session:
    session = session or requests.Session()
    if str(session.headers.get("User-Agent") or "").startswith("python-requests/"):
        session.headers["User-Agent"] = _BROWSER_USER_AGENT
    else:
        session.headers.setdefault("User-Agent", _BROWSER_USER_AGENT)
    for key, value in _BROWSER_HEADERS.items():
        session.headers.setdefault(key, value)
    if str(tour or "").lower() == "wta":
        session.headers.setdefault("account", "wta")
    return session


def _get_with_retries(session: requests.Session, url: str, timeout: int = 60, max_attempts: int = 5, **kwargs) -> requests.Response:
    response: Optional[requests.Response] = None
    for attempt in range(max_attempts):
        response = session.get(url, timeout=timeout, **kwargs)
        if response.status_code not in REQUEST_RETRY_STATUS_CODES:
            return response
        if attempt >= max_attempts - 1:
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = max(float(retry_after), 1.0)
        elif response.status_code == 429:
            wait_seconds = float(10 * (attempt + 1))
        else:
            wait_seconds = float(min(2**attempt, 8))
        logger.warning(
            "Retrying tennis rankings request %s after HTTP %s in %.1f seconds (attempt %s/%s)",
            url,
            response.status_code,
            wait_seconds,
            attempt + 2,
            max_attempts,
        )
        time.sleep(wait_seconds)

    if response is None:
        raise requests.RequestException(f"Failed to fetch {url}")
    return response


def _safe_int(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d[\d,]*", text)
    if match is None:
        return None
    text = match.group(0).replace(",", "")
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _normalized_name_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _initial_surname_key(value: object) -> str:
    tokens = _normalized_name_key(value).split()
    if len(tokens) < 2 or not tokens[0]:
        return ""
    surname = " ".join(tokens[1:]).strip()
    if not surname:
        return ""
    return f"{tokens[0][0]} {surname}".strip()


def _slug_to_full_name(slug: object) -> str:
    text = str(slug or "").strip().replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return " ".join(token.capitalize() for token in text.split())


def _unique_non_missing_value(series: pd.Series, *, numeric: bool = False):
    if numeric:
        values = pd.to_numeric(series, errors="coerce").dropna()
        unique_values = sorted(set(values.round(6).tolist()))
        if len(unique_values) == 1:
            value = unique_values[0]
            return int(value) if float(value).is_integer() else float(value)
        return None

    values = series.fillna("").astype(str).str.strip()
    values = values[values != ""]
    unique_values = sorted(set(values.tolist()))
    if len(unique_values) == 1:
        return unique_values[0]
    return None


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.astype("string").str.strip().str.lower()
        return cleaned.isna() | cleaned.eq("") | cleaned.eq("nan") | cleaned.eq("none")
    return series.isna()


def _fill_missing_column(frame: pd.DataFrame, destination: str, values: pd.Series) -> None:
    if destination not in frame.columns:
        frame[destination] = pd.NA
    mask = _missing_mask(frame[destination]) & values.notna()
    if not mask.any():
        return
    if pd.api.types.is_numeric_dtype(frame[destination]) or pd.api.types.is_numeric_dtype(values):
        frame.loc[mask, destination] = pd.to_numeric(values.loc[mask], errors="coerce")
    else:
        frame.loc[mask, destination] = values.loc[mask].astype(object)


def _ensure_rankings_dirs() -> None:
    _rankings_dir_for_tour("atp").mkdir(parents=True, exist_ok=True)
    _rankings_dir_for_tour("wta").mkdir(parents=True, exist_ok=True)
    TENNIS_RANKINGS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _atp_rankings_available_dates(session: Optional[requests.Session] = None) -> list[str]:
    session = _request_session(session, tour="atp")
    response = _get_with_retries(session, ATP_RANKINGS_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    select = soup.select_one("select#dateWeek-filter")
    if select is None:
        raise ValueError("ATP rankings page did not expose a dateWeek filter.")

    dates: list[str] = []
    for option in select.select("option"):
        value = str(option.get("value") or "").strip()
        label = " ".join(option.get_text(" ", strip=True).split())
        candidate = value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else ""
        if not candidate and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", label):
            candidate = label.replace(".", "-")
        if candidate:
            dates.append(candidate)
    return sorted(set(dates))


def _resolve_atp_snapshot_dates(request_dates: list[str], available_dates: list[str]) -> dict[str, str]:
    return _resolve_snapshot_dates_from_available(request_dates, available_dates)


def _resolve_snapshot_dates_from_available(request_dates: list[str], available_dates: list[str]) -> dict[str, str]:
    available_index = [pd.Timestamp(value).date() for value in available_dates]
    resolved: dict[str, str] = {}
    for request_date in request_dates:
        request_day = pd.Timestamp(request_date).date()
        index = bisect_right(available_index, request_day) - 1
        if index >= 0:
            resolved[request_date] = available_dates[index]
    return resolved


def _cached_rankings_snapshot_dates(tour: str) -> list[str]:
    tour_norm = str(tour or "").strip().lower()
    directory = _rankings_dir_for_tour(tour_norm)
    if not directory.exists():
        return []

    pattern = re.compile(rf"^{re.escape(tour_norm)}_singles_rankings_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    snapshot_dates: list[str] = []
    for path in directory.glob(f"{tour_norm}_singles_rankings_*.csv"):
        match = pattern.match(path.name)
        if match is not None:
            snapshot_dates.append(match.group(1))
    return sorted(set(snapshot_dates))


def _empty_rankings_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["tour", "snapshot_date", "player_id", "display_name", "full_name", "rank", "rank_points"]
    )


def _parse_atp_rankings_snapshot_html(html: str, *, snapshot_date: str, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    rows: list[dict[str, object]] = []
    for row in soup.select("table.mega-table tbody tr.lower-row"):
        link = row.select_one("td.player a[href*='/en/players/']")
        if link is None:
            continue

        href = str(link.get("href") or "").strip()
        href_parts = [part for part in href.split("/") if part]
        if len(href_parts) < 4:
            continue

        player_id = canonical_player_id("atp", href_parts[-2])
        full_name = _slug_to_full_name(href_parts[-3])
        display_name = " ".join(link.get_text(" ", strip=True).split())
        rank = _safe_int(row.select_one("td.rank").get_text(" ", strip=True) if row.select_one("td.rank") else None)
        points_node = row.select_one("td.points")
        rank_points = _safe_int(points_node.get_text(" ", strip=True) if points_node else None)
        if not player_id or rank is None:
            continue

        rows.append(
            {
                "tour": "atp",
                "snapshot_date": snapshot_date,
                "player_id": player_id,
                "display_name": display_name,
                "full_name": full_name,
                "rank": rank,
                "rank_points": rank_points,
                "source_url": source_url,
                "fetched_at_utc": _now_utc_iso(),
            }
        )

    if not rows:
        return _empty_rankings_snapshot_frame()

    frame = pd.DataFrame(rows)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["rank_points"] = pd.to_numeric(frame["rank_points"], errors="coerce")
    frame = frame.sort_values(["rank", "player_id"]).drop_duplicates(subset=["player_id"], keep="first")
    return frame.reset_index(drop=True)


def fetch_atp_rankings_snapshot(snapshot_date: str, session: Optional[requests.Session] = None) -> pd.DataFrame:
    session = _request_session(session, tour="atp")
    response = _get_with_retries(
        session,
        ATP_RANKINGS_URL,
        timeout=60,
        params={"dateWeek": snapshot_date, "rankRange": "0-5000"},
    )
    response.raise_for_status()

    frame = _parse_atp_rankings_snapshot_html(response.text, snapshot_date=snapshot_date, source_url=response.url)
    if not frame.empty:
        return frame

    if "Error while rendering the view [Singles Rankings]" not in response.text:
        return frame

    logger.warning(
        "ATP rankings archive failed for %s with rankRange=0-5000; retrying in rank-range chunks.",
        snapshot_date,
    )
    frames: list[pd.DataFrame] = []
    for index, rank_range in enumerate(ATP_FALLBACK_RANK_RANGES):
        range_response = _get_with_retries(
            session,
            ATP_RANKINGS_URL,
            timeout=60,
            params={"dateWeek": snapshot_date, "rankRange": rank_range},
        )
        range_response.raise_for_status()
        range_frame = _parse_atp_rankings_snapshot_html(
            range_response.text,
            snapshot_date=snapshot_date,
            source_url=range_response.url,
        )
        if not range_frame.empty:
            frames.append(range_frame)
        if index < len(ATP_FALLBACK_RANK_RANGES) - 1:
            time.sleep(RANKINGS_REQUEST_PAUSE_SECONDS)

    if not frames:
        return _empty_rankings_snapshot_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["rank", "player_id"]).drop_duplicates(subset=["player_id"], keep="first")
    return combined.reset_index(drop=True)


def _load_wta_snapshot_resolution() -> dict[str, str]:
    path = _wta_snapshot_resolution_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if key and value}


def _write_wta_snapshot_resolution(mapping: dict[str, str]) -> None:
    _ensure_rankings_dirs()
    path = _wta_snapshot_resolution_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)


def resolve_wta_snapshot_date(
    request_date: str,
    *,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    mapping = _load_wta_snapshot_resolution()
    if not force and request_date in mapping:
        return mapping[request_date]

    session = _request_session(session, tour="wta")
    response = _get_with_retries(
        session,
        WTA_RANKINGS_API_URL,
        timeout=60,
        params={
            "page": 0,
            "pageSize": 1,
            "type": "rankSingles",
            "sort": "asc",
            "metric": "SINGLES",
            "at": request_date,
            "name": "",
            "nationality": "",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        return None
    ranked_at = str((payload[0] or {}).get("rankedAt") or "").strip()
    if not ranked_at:
        return None
    snapshot_date = str(pd.Timestamp(ranked_at).date())
    mapping[request_date] = snapshot_date
    _write_wta_snapshot_resolution(mapping)
    return snapshot_date


def fetch_wta_rankings_snapshot(snapshot_date: str, session: Optional[requests.Session] = None) -> pd.DataFrame:
    session = _request_session(session, tour="wta")
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for page in range(100):
        response = _get_with_retries(
            session,
            WTA_RANKINGS_API_URL,
            timeout=60,
            params={
                "page": page,
                "pageSize": WTA_RANKINGS_PAGE_SIZE,
                "type": "rankSingles",
                "sort": "asc",
                "metric": "SINGLES",
                "at": snapshot_date,
                "name": "",
                "nationality": "",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            break

        for item in payload:
            player = (item or {}).get("player") or {}
            player_id = canonical_player_id("wta", player.get("id"))
            if not player_id or player_id in seen_ids:
                continue
            seen_ids.add(player_id)
            rows.append(
                {
                    "tour": "wta",
                    "snapshot_date": str(pd.Timestamp((item or {}).get("rankedAt") or snapshot_date).date()),
                    "player_id": player_id,
                    "display_name": str(player.get("fullName") or "").strip(),
                    "full_name": str(player.get("fullName") or "").strip(),
                    "rank": _safe_int(item.get("ranking")),
                    "rank_points": _safe_int(item.get("points")),
                    "source_url": response.url,
                    "fetched_at_utc": _now_utc_iso(),
                }
            )

        if len(payload) < WTA_RANKINGS_PAGE_SIZE:
            break
        time.sleep(RANKINGS_REQUEST_PAUSE_SECONDS)

    return pd.DataFrame(rows)


def load_rankings_snapshot(
    tour: str,
    snapshot_date: str,
    *,
    fetch_missing: bool = False,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    tour_norm = str(tour or "").lower()
    path = _rankings_snapshot_path(tour_norm, snapshot_date)
    if path.exists() and not force:
        return pd.read_csv(path, low_memory=False)

    if not fetch_missing:
        return pd.DataFrame(columns=["tour", "snapshot_date", "player_id", "display_name", "full_name", "rank", "rank_points"])

    _ensure_rankings_dirs()
    fetch_fn = fetch_atp_rankings_snapshot if tour_norm == "atp" else fetch_wta_rankings_snapshot
    frame = fetch_fn(snapshot_date, session=session)
    if not frame.empty:
        frame.to_csv(path, index=False)
    elif force and path.exists():
        path.unlink()
    return frame


def load_rankings_snapshots(
    tour: str,
    snapshot_dates: list[str],
    *,
    fetch_missing: bool = False,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for snapshot_date in sorted(set(snapshot_dates)):
        frame = load_rankings_snapshot(
            tour,
            snapshot_date,
            fetch_missing=fetch_missing,
            force=force,
            session=session,
        )
        if not frame.empty:
            frames.append(frame)
            time.sleep(RANKINGS_REQUEST_PAUSE_SECONDS)
    if not frames:
        return _empty_rankings_snapshot_frame()
    combined = pd.concat(frames, ignore_index=True)
    combined["tour"] = combined["tour"].astype(str).str.lower()
    combined["player_id"] = combined.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["rank"] = pd.to_numeric(combined["rank"], errors="coerce")
    combined["rank_points"] = pd.to_numeric(combined["rank_points"], errors="coerce")
    return combined.reset_index(drop=True)


def _load_rankings_snapshots_best_effort(
    tour: str,
    snapshot_dates: list[str],
    *,
    fetch_missing: bool = False,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    unique_snapshot_dates = sorted(set(snapshot_dates))
    if not unique_snapshot_dates:
        return _empty_rankings_snapshot_frame()

    try:
        return load_rankings_snapshots(
            tour,
            unique_snapshot_dates,
            fetch_missing=fetch_missing,
            force=force,
            session=session,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load %s rankings snapshots for %s; retrying with cached files only: %s",
            str(tour or "").upper(),
            ", ".join(unique_snapshot_dates),
            exc,
        )

    try:
        return load_rankings_snapshots(
            tour,
            unique_snapshot_dates,
            fetch_missing=False,
            force=False,
            session=session,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load cached %s rankings snapshots for %s: %s",
            str(tour or "").upper(),
            ", ".join(unique_snapshot_dates),
            exc,
        )
        return _empty_rankings_snapshot_frame()


def _build_rankings_id_lookup(rankings_df: pd.DataFrame) -> pd.DataFrame:
    if rankings_df.empty:
        return pd.DataFrame(columns=["tour", "snapshot_date", "player_id_key", "rank", "rank_points"])
    working = rankings_df.copy()
    working["tour"] = working["tour"].astype(str).str.lower()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["player_id_key"] = working.apply(lambda row: canonical_player_id(row.get("tour"), row.get("player_id")), axis=1)
    rows: list[dict[str, object]] = []
    for (tour, snapshot_date, player_id_key), group in working.groupby(["tour", "snapshot_date", "player_id_key"], dropna=False):
        rows.append(
            {
                "tour": tour,
                "snapshot_date": snapshot_date,
                "player_id_key": player_id_key,
                "rank": _unique_non_missing_value(group["rank"], numeric=True),
                "rank_points": _unique_non_missing_value(group["rank_points"], numeric=True),
            }
        )
    return pd.DataFrame(rows)


def _build_rankings_name_lookup(rankings_df: pd.DataFrame) -> pd.DataFrame:
    if rankings_df.empty:
        return pd.DataFrame(columns=["tour", "snapshot_date", "player_name_key", "rank", "rank_points"])
    working = rankings_df.copy()
    working["tour"] = working["tour"].astype(str).str.lower()
    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    name_rows: list[dict[str, object]] = []
    for name_column in ["display_name", "full_name"]:
        if name_column not in working.columns:
            continue
        subset = working[["tour", "snapshot_date", name_column, "rank", "rank_points"]].copy()
        subset = subset.rename(columns={name_column: "player_name"})
        subset["player_name_key"] = subset["player_name"].map(_normalized_name_key)
        subset = subset[subset["player_name_key"].astype(str).str.len() > 0]
        if not subset.empty:
            name_rows.extend(subset.to_dict("records"))

        abbreviated = working[["tour", "snapshot_date", name_column, "rank", "rank_points"]].copy()
        abbreviated = abbreviated.rename(columns={name_column: "player_name"})
        abbreviated["player_name_key"] = abbreviated["player_name"].map(_initial_surname_key)
        abbreviated = abbreviated[abbreviated["player_name_key"].astype(str).str.len() > 0]
        if not abbreviated.empty:
            name_rows.extend(abbreviated.to_dict("records"))

    if not name_rows:
        return pd.DataFrame(columns=["tour", "snapshot_date", "player_name_key", "rank", "rank_points"])

    combined = pd.DataFrame(name_rows)
    rows: list[dict[str, object]] = []
    for (tour, snapshot_date, player_name_key), group in combined.groupby(["tour", "snapshot_date", "player_name_key"], dropna=False):
        rank = _unique_non_missing_value(group["rank"], numeric=True)
        rank_points = _unique_non_missing_value(group["rank_points"], numeric=True)
        if rank is None and rank_points is None:
            continue
        rows.append(
            {
                "tour": tour,
                "snapshot_date": snapshot_date,
                "player_name_key": player_name_key,
                "rank": rank,
                "rank_points": rank_points,
            }
        )
    return pd.DataFrame(rows)


def enrich_tennis_matches_with_rankings_history(
    matches_df: pd.DataFrame,
    *,
    fetch_missing: bool = False,
    force_download: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if matches_df.empty:
        return matches_df.copy()

    working = matches_df.copy()
    working["tour"] = working["tour"].astype(str).str.lower()
    working["event_date"] = pd.to_datetime(working["event_date"], errors="coerce")

    required_columns = {
        "tour",
        "event_date",
        "player_a_id",
        "player_b_id",
        "player_a",
        "player_b",
        "player_a_rank",
        "player_b_rank",
        "player_a_rank_points",
        "player_b_rank_points",
    }
    missing_columns = [column for column in required_columns if column not in working.columns]
    if missing_columns:
        raise ValueError(
            "Tennis match frame is missing required columns for rankings-history enrichment: "
            + ", ".join(missing_columns)
        )

    needs_rankings = (
        _missing_mask(working["player_a_rank"])
        | _missing_mask(working["player_b_rank"])
        | _missing_mask(working["player_a_rank_points"])
        | _missing_mask(working["player_b_rank_points"])
    )
    if not needs_rankings.any():
        return working

    request_dates_by_tour: dict[str, list[str]] = {}
    request_dates = working.loc[needs_rankings, ["tour", "event_date"]].dropna().copy()
    request_dates["request_date"] = request_dates["event_date"].dt.strftime("%Y-%m-%d")
    for tour, group in request_dates.groupby("tour"):
        request_dates_by_tour[tour] = sorted(set(group["request_date"].tolist()))

    session = session or requests.Session()
    snapshot_mapping_rows: list[dict[str, str]] = []
    rankings_frames: list[pd.DataFrame] = []

    atp_request_dates = request_dates_by_tour.get("atp", [])
    if atp_request_dates:
        atp_mapping: dict[str, str] = {}
        try:
            atp_available_dates = _atp_rankings_available_dates(session=_request_session(session, tour="atp"))
            atp_mapping.update(_resolve_atp_snapshot_dates(atp_request_dates, atp_available_dates))
        except Exception as exc:
            logger.warning("Failed to resolve ATP rankings snapshot dates from official source: %s", exc)

        unresolved_atp_dates = sorted(set(atp_request_dates) - set(atp_mapping))
        if unresolved_atp_dates:
            cached_atp_dates = _cached_rankings_snapshot_dates("atp")
            cached_atp_mapping = _resolve_snapshot_dates_from_available(unresolved_atp_dates, cached_atp_dates)
            if cached_atp_mapping:
                logger.info(
                    "Resolved %s ATP rankings snapshot dates from cached files",
                    len(cached_atp_mapping),
                )
                atp_mapping.update(cached_atp_mapping)

        snapshot_mapping_rows.extend(
            {"tour": "atp", "request_date": request_date, "snapshot_date": snapshot_date}
            for request_date, snapshot_date in atp_mapping.items()
        )
        atp_snapshots = _load_rankings_snapshots_best_effort(
            "atp",
            list(atp_mapping.values()),
            fetch_missing=fetch_missing,
            force=force_download,
            session=_request_session(session, tour="atp"),
        )
        if not atp_snapshots.empty:
            rankings_frames.append(atp_snapshots)

    wta_request_dates = request_dates_by_tour.get("wta", [])
    if wta_request_dates:
        wta_mapping: dict[str, str] = {}
        for request_date in wta_request_dates:
            snapshot_date: Optional[str] = None
            try:
                snapshot_date = resolve_wta_snapshot_date(
                    request_date,
                    force=force_download,
                    session=_request_session(session, tour="wta"),
                )
            except Exception as exc:
                logger.warning("Failed to resolve WTA rankings snapshot date for %s: %s", request_date, exc)
            if snapshot_date:
                wta_mapping[request_date] = snapshot_date

        unresolved_wta_dates = sorted(set(wta_request_dates) - set(wta_mapping))
        if unresolved_wta_dates:
            cached_wta_dates = _cached_rankings_snapshot_dates("wta")
            cached_wta_mapping = _resolve_snapshot_dates_from_available(unresolved_wta_dates, cached_wta_dates)
            if cached_wta_mapping:
                logger.info(
                    "Resolved %s WTA rankings snapshot dates from cached files",
                    len(cached_wta_mapping),
                )
                wta_mapping.update(cached_wta_mapping)

        snapshot_mapping_rows.extend(
            {"tour": "wta", "request_date": request_date, "snapshot_date": snapshot_date}
            for request_date, snapshot_date in wta_mapping.items()
        )
        wta_snapshots = _load_rankings_snapshots_best_effort(
            "wta",
            list(wta_mapping.values()),
            fetch_missing=fetch_missing,
            force=force_download,
            session=_request_session(session, tour="wta"),
        )
        if not wta_snapshots.empty:
            rankings_frames.append(wta_snapshots)

    if not rankings_frames or not snapshot_mapping_rows:
        return working

    rankings_df = pd.concat(rankings_frames, ignore_index=True).drop_duplicates(
        subset=["tour", "snapshot_date", "player_id", "display_name", "full_name"],
        keep="last",
    )
    snapshot_map = pd.DataFrame(snapshot_mapping_rows).drop_duplicates(subset=["tour", "request_date"], keep="last")

    working["request_date"] = working["event_date"].dt.strftime("%Y-%m-%d")
    working = working.merge(snapshot_map, on=["tour", "request_date"], how="left")

    id_lookup = _build_rankings_id_lookup(rankings_df)
    name_lookup = _build_rankings_name_lookup(rankings_df)

    for prefix in ["player_a", "player_b"]:
        id_key_column = f"{prefix}_id_key"
        working[id_key_column] = working.apply(lambda row: canonical_player_id(row.get("tour"), row.get(f"{prefix}_id")), axis=1)
        renamed_id_lookup = id_lookup.rename(
            columns={
                "player_id_key": id_key_column,
                "rank": f"{prefix}_history_rank",
                "rank_points": f"{prefix}_history_rank_points",
            }
        )
        working = working.merge(renamed_id_lookup, on=["tour", "snapshot_date", id_key_column], how="left")
        _fill_missing_column(working, f"{prefix}_rank", working[f"{prefix}_history_rank"])
        _fill_missing_column(working, f"{prefix}_rank_points", working[f"{prefix}_history_rank_points"])
        working = working.drop(columns=[id_key_column, f"{prefix}_history_rank", f"{prefix}_history_rank_points"])

        for key_builder, suffix in [(_normalized_name_key, "name"), (_initial_surname_key, "abbr")]:
            key_column = f"{prefix}_{suffix}_ranking_key"
            working[key_column] = working[prefix].map(key_builder)
            renamed_name_lookup = name_lookup.rename(
                columns={
                    "player_name_key": key_column,
                    "rank": f"{prefix}_{suffix}_history_rank",
                    "rank_points": f"{prefix}_{suffix}_history_rank_points",
                }
            )
            working = working.merge(renamed_name_lookup, on=["tour", "snapshot_date", key_column], how="left")
            _fill_missing_column(working, f"{prefix}_rank", working[f"{prefix}_{suffix}_history_rank"])
            _fill_missing_column(working, f"{prefix}_rank_points", working[f"{prefix}_{suffix}_history_rank_points"])
            working = working.drop(
                columns=[
                    key_column,
                    f"{prefix}_{suffix}_history_rank",
                    f"{prefix}_{suffix}_history_rank_points",
                ]
            )

    working = working.drop(columns=["request_date", "snapshot_date"])
    return working


def enrich_live_tennis_matchups_with_current_rankings(
    matchups_df: pd.DataFrame,
    *,
    fetch_missing: bool = True,
    force_download: bool = False,
    session: Optional[requests.Session] = None,
    overwrite_existing: bool = True,
) -> pd.DataFrame:
    """Best-effort live enrichment for current ATP/WTA ranks and rank points."""
    if matchups_df.empty:
        return matchups_df.copy()

    working = matchups_df.copy()
    required_columns = {"tour", "fighter_a", "fighter_b"}
    missing_columns = [column for column in required_columns if column not in working.columns]
    if missing_columns:
        raise ValueError(
            "Tennis live matchup frame is missing required columns for rankings enrichment: "
            + ", ".join(missing_columns)
        )

    target_columns = [
        "player_a_rank",
        "player_b_rank",
        "player_a_rank_points",
        "player_b_rank_points",
    ]
    original_values: dict[str, pd.Series] = {}
    for column in target_columns:
        if column not in working.columns:
            working[column] = pd.NA
        original_values[column] = working[column].copy()

    seed = working.copy()
    if "commence_time" in seed.columns:
        seed["event_date"] = pd.to_datetime(seed["commence_time"], errors="coerce")
        if "event_date" in working.columns:
            fallback_event_dates = pd.to_datetime(working["event_date"], errors="coerce")
            missing_event_dates = seed["event_date"].isna() & fallback_event_dates.notna()
            seed.loc[missing_event_dates, "event_date"] = fallback_event_dates.loc[missing_event_dates]
    else:
        seed["event_date"] = pd.to_datetime(seed.get("event_date"), errors="coerce")

    for source_column, fallback_column in [("player_a", "fighter_a"), ("player_b", "fighter_b")]:
        if source_column not in seed.columns:
            seed[source_column] = seed[fallback_column]
        else:
            source_missing = _missing_mask(seed[source_column])
            seed.loc[source_missing, source_column] = seed.loc[source_missing, fallback_column]

    if "player_a_id" not in seed.columns:
        seed["player_a_id"] = seed.get("fighter_a_id", "")
    if "player_b_id" not in seed.columns:
        seed["player_b_id"] = seed.get("fighter_b_id", "")

    if overwrite_existing:
        for column in target_columns:
            seed[column] = pd.NA

    try:
        enriched = enrich_tennis_matches_with_rankings_history(
            seed,
            fetch_missing=fetch_missing,
            force_download=force_download,
            session=session,
        )
    except Exception as exc:
        logger.warning("Failed to enrich live tennis matchups with current rankings: %s", exc)
        return working

    for column in target_columns:
        if column not in enriched.columns:
            continue
        working[column] = enriched[column]
        _fill_missing_column(working, column, original_values[column])

    covered_rows = int(
        (
            ~_missing_mask(working["player_a_rank"])
            & ~_missing_mask(working["player_b_rank"])
        ).sum()
    )
    logger.info(
        "Live tennis rankings enrichment covered %s/%s matchup rows",
        covered_rows,
        len(working),
    )
    return working


def summarize_tennis_rankings_enrichment(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict[str, object]:
    tracked_columns = [
        "player_a_rank",
        "player_b_rank",
        "player_a_rank_points",
        "player_b_rank_points",
    ]

    before = before_df.copy()
    after = after_df.copy()
    before["event_date"] = pd.to_datetime(before["event_date"], errors="coerce")
    after["event_date"] = pd.to_datetime(after["event_date"], errors="coerce")

    def coverage_payload(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        payload: dict[str, dict[str, int]] = {}
        for column in tracked_columns:
            payload[column] = {
                "non_null_rows": int((~_missing_mask(frame[column])).sum()) if column in frame.columns else 0,
                "total_rows": int(len(frame)),
            }
        return payload

    def coverage_by_year(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
        output: dict[str, dict[str, dict[str, int]]] = {}
        years = frame["event_date"].dt.year.dropna().astype(int).sort_values().unique()
        for year in years:
            subset = frame[frame["event_date"].dt.year == int(year)].copy()
            output[str(year)] = coverage_payload(subset)
        return output

    filled_counts: dict[str, int] = {}
    for column in tracked_columns:
        before_non_null = int((~_missing_mask(before[column])).sum()) if column in before.columns else 0
        after_non_null = int((~_missing_mask(after[column])).sum()) if column in after.columns else 0
        filled_counts[column] = max(after_non_null - before_non_null, 0)

    return {
        "summary_generated_at_utc": _now_utc_iso(),
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "filled_counts": {str(key): int(value) for key, value in filled_counts.items()},
        "coverage_before": coverage_payload(before),
        "coverage_after": coverage_payload(after),
        "coverage_after_by_year": coverage_by_year(after),
    }


def write_tennis_rankings_enrichment_summary(summary: dict[str, object], path: Optional[Path] = None) -> Path:
    _ensure_rankings_dirs()
    output_path = path or TENNIS_RANKINGS_SUMMARY_PATH
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return output_path
