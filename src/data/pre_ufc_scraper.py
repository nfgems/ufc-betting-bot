"""
On-demand pre-UFC career scraper.

Scrapes a fighter's pre-UFC fight history from Sherdog AND Tapology,
keeping whichever source returns the most complete data.  Used both by
the batch script (scripts/build_pre_ufc_career_supplement.py) and at
live-prediction time inside fighter_lookup.build_fight_features().
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fallback_scrapers import (
    scrape_sherdog_page,
    scrape_tapology_profile,
    scrape_tapology_fights,
    search_sherdog,
    search_tapology,
    search_tapology_candidates,
)
from src.data.io_utils import write_csv_atomically
from src.data.name_utils import (
    normalize_cross_source_name,
    normalize_person_name,
    same_person_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAT_COLUMNS = [
    "kd", "sig_str_landed", "sig_str_attempted",
    "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds",
]

BASE_COLUMNS = [
    "event_date", "fighter_a", "fighter_b", "winner", "method",
    "finish_round", "num_rounds", "weight_class", "title_bout",
    "source", "organization",
]

OUTPUT_COLUMNS = BASE_COLUMNS + [
    f"{prefix}{stat}" for prefix in ("a_", "b_") for stat in STAT_COLUMNS
]

PROMOTION_CANONICAL_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbellator(?: mma)?\b", re.IGNORECASE), "Bellator"),
    (re.compile(r"\bone(?: championship| fc)?\b", re.IGNORECASE), "ONE Championship"),
    (re.compile(r"\bworld series of fighting\b|\bwsof\b", re.IGNORECASE), "WSOF"),
    (re.compile(r"\blegacy fighting alliance\b|\blfa\b", re.IGNORECASE), "LFA"),
    (re.compile(r"\bcage warriors\b", re.IGNORECASE), "Cage Warriors"),
    (re.compile(r"\bdana white'?s contender series\b|\bcontender series\b", re.IGNORECASE),
     "Dana White's Contender Series"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _same_fighter_identity(query_name: str, resolved_name: str) -> bool:
    if same_person_name(query_name, resolved_name):
        return True
    query_key = normalize_person_name(query_name).replace(" ", "")
    resolved_key = normalize_person_name(resolved_name).replace(" ", "")
    if query_key and query_key == resolved_key:
        return True
    query_cross = normalize_cross_source_name(query_name).replace(" ", "")
    resolved_cross = normalize_cross_source_name(resolved_name).replace(" ", "")
    return bool(query_cross and query_cross == resolved_cross)


def sherdog_method_to_ufc_method(method: str) -> str:
    """Normalize Sherdog method strings to match UFCStats format."""
    m = method.strip().lower()
    if "tko" in m or "knockout" in m or re.search(r"\bko\b", m):
        return "KO/TKO"
    if "technical submission" in m or "submission" in m or re.search(r"\bsub\b", m):
        return "SUB"
    if "decision" in m or re.search(r"\bdec\b", m):
        if "unanimous" in m:
            return "U-DEC"
        if "split" in m:
            return "S-DEC"
        if "majority" in m:
            return "M-DEC"
        return "U-DEC"
    if "nc" in m or "no contest" in m:
        return "Overturned"
    if "draw" in m:
        return "Draw"
    if "dq" in m or "disqualification" in m:
        return "DQ"
    return method


def _infer_num_rounds(method: str, round_finished) -> float:
    if round_finished in (None, "", 0) or pd.isna(round_finished):
        return np.nan
    if method in {"U-DEC", "S-DEC", "M-DEC", "Draw"}:
        return float(round_finished)
    return np.nan


def _infer_organization(event_name: str) -> str:
    """Extract a promotion label from the Sherdog/Tapology event name."""
    text = str(event_name or "").strip()
    if not text:
        return ""

    prefix = text
    for separator in (" - ", ": "):
        if separator in prefix:
            prefix = prefix.split(separator, 1)[0].strip()
            break

    prefix = re.sub(r"\s+\d+[A-Za-z]*$", "", prefix).strip()
    prefix = re.sub(r"\s{2,}", " ", prefix)

    for pattern, replacement in PROMOTION_CANONICAL_MAP:
        if pattern.search(prefix):
            return replacement

    if prefix:
        return prefix

    match = re.match(r"([A-Za-z0-9'&./ -]+?)(?:\s+\d+\b|$)", text)
    return match.group(1).strip() if match else text


# ---------------------------------------------------------------------------
# Core: scrape pre-UFC fights from a single source
# ---------------------------------------------------------------------------

def _rows_from_fights(
    fights: list[dict],
    fighter_name: str,
    first_ufc_date: str | None,
    source: str,
) -> list[dict]:
    """Convert raw scraped fights to supplement-schema rows, filtering to pre-UFC only."""
    if not fights:
        return []

    first_ufc = pd.to_datetime(first_ufc_date, errors="coerce") if first_ufc_date else pd.NaT
    rows = []

    for fight in fights:
        fight_date = fight.get("event_date")
        if fight_date is None:
            continue

        fight_date = pd.to_datetime(fight_date)
        if pd.notna(first_ufc) and fight_date >= first_ufc:
            continue

        opponent = fight.get("opponent", "Unknown")
        result = str(fight.get("result", "")).strip().lower()
        method_raw = fight.get("method", "")
        method = sherdog_method_to_ufc_method(method_raw)
        round_finished = fight.get("round_finished")
        organization = (
            str(fight.get("organization") or "").strip()
            or _infer_organization(fight.get("event_name", ""))
        )

        if result == "win":
            winner = fighter_name
        elif result in {"draw", "nc"} or "draw" in method_raw.lower() or "no contest" in method_raw.lower():
            winner = np.nan
        else:
            winner = opponent

        row = {
            "event_date": fight_date.strftime("%Y-%m-%d"),
            "fighter_a": fighter_name,
            "fighter_b": opponent,
            "winner": winner,
            "method": method,
            "finish_round": round_finished,
            "num_rounds": _infer_num_rounds(method, round_finished),
            "weight_class": "",
            "title_bout": 1 if fight.get("is_title_bout") else 0,
            "source": source,
            "organization": organization,
        }

        for prefix in ["a_", "b_"]:
            for stat in STAT_COLUMNS:
                row[f"{prefix}{stat}"] = np.nan

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main entry point: scrape one fighter from all sources, return the best
# ---------------------------------------------------------------------------

def scrape_fighter_pre_ufc_fights(
    fighter_name: str,
    first_ufc_date: str | None,
) -> list[dict]:
    """
    Scrape a fighter's pre-UFC fight history from Sherdog AND Tapology.

    Always tries both sources and returns whichever found more pre-UFC
    fights.  Never discards real data from one source based on an
    arbitrary threshold.

    Returns fight rows in supplement CSV schema (fights_cleaned-compatible),
    with per-fight stats set to NaN (not available from regional sources).
    """
    sherdog_rows: list[dict] = []
    tapology_rows: list[dict] = []

    # --- Source 1: Sherdog ---
    url = search_sherdog(fighter_name)
    if url:
        try:
            profile, fights = scrape_sherdog_page(url, fighter_name)
            if profile and profile.get("name") and not _same_fighter_identity(profile.get("name"), fighter_name):
                logger.warning(
                    "  Sherdog match mismatch for '%s': resolved to '%s' (%s)",
                    fighter_name, profile.get("name"), url,
                )
            else:
                sherdog_rows = _rows_from_fights(fights, fighter_name, first_ufc_date, "sherdog")
        except Exception as e:
            logger.warning(f"  Sherdog scrape failed for '{fighter_name}': {e}")

    # --- Source 2: Tapology ---
    tapology_urls = []
    primary_tapology_url = search_tapology(fighter_name)
    if primary_tapology_url:
        tapology_urls.append(primary_tapology_url)
    for candidate_url in search_tapology_candidates(fighter_name, limit=5):
        if candidate_url not in tapology_urls:
            tapology_urls.append(candidate_url)

    for tapology_url in tapology_urls:
        try:
            profile = scrape_tapology_profile(tapology_url)
            resolved_name = str(profile.get("name") or "").strip()
            if not resolved_name or not _same_fighter_identity(fighter_name, resolved_name):
                logger.warning(
                    "  Tapology match mismatch for '%s': resolved to '%s' (%s)",
                    fighter_name,
                    resolved_name or "?",
                    tapology_url,
                )
                continue
            candidate_rows = _rows_from_fights(
                scrape_tapology_fights(tapology_url, fighter_name),
                fighter_name, first_ufc_date, "tapology",
            )
        except Exception as e:
            logger.warning(f"  Tapology scrape failed for '{fighter_name}': {e}")
            continue
        if len(candidate_rows) > len(tapology_rows):
            tapology_rows = candidate_rows

    # --- Pick whichever source returned more fights ---
    if len(tapology_rows) > len(sherdog_rows):
        best_rows = tapology_rows
    else:
        best_rows = sherdog_rows  # Sherdog wins ties (tends to be more accurate)

    if not best_rows:
        logger.debug(f"  No pre-UFC fight history found for '{fighter_name}'")
        return []

    logger.info(
        f"  {fighter_name}: {len(best_rows)} pre-UFC fights "
        f"(sherdog={len(sherdog_rows)}, tapology={len(tapology_rows)})"
    )
    return best_rows


# ---------------------------------------------------------------------------
# Summary computation for a single fighter (used by on-demand lookup)
# ---------------------------------------------------------------------------

def compute_single_fighter_summary(rows: list[dict]) -> dict:
    """Compute pre-UFC career summary features from raw fight rows.

    Returns a dict with keys: pre_ufc_total_fights, pre_ufc_wins,
    pre_ufc_losses, pre_ufc_win_pct, pre_ufc_ko_rate, pre_ufc_sub_rate,
    pre_ufc_dec_rate, pre_ufc_org_tier_best.
    """
    if not rows:
        return {}

    total = len(rows)
    wins = sum(1 for r in rows if r.get("winner") == r.get("fighter_a"))
    losses = sum(
        1 for r in rows
        if r.get("winner") and r.get("winner") != r.get("fighter_a")
        and not (isinstance(r.get("winner"), float) and np.isnan(r["winner"]))
    )

    if wins > 0:
        methods = [str(r.get("method", "")).lower() for r in rows if r.get("winner") == r.get("fighter_a")]
        ko_wins = sum(1 for m in methods if re.search(r"ko|tko|punch|kick|knee|elbow|slam|stomp", m))
        sub_wins = sum(1 for m in methods if re.search(
            r"sub|submission|choke|armbar|triangle|guillotine|rear.naked|kimura|"
            r"americana|heel.hook|ankle.lock|arm.triangle|darce|anaconda|twister|"
            r"calf.slicer|neck.crank", m))
        dec_wins = sum(1 for m in methods if re.search(r"dec|decision|unanimous|split|majority", m))
        ko_rate = float(ko_wins) / wins
        sub_rate = float(sub_wins) / wins
        dec_rate = float(dec_wins) / wins
    else:
        ko_rate = np.nan
        sub_rate = np.nan
        dec_rate = np.nan

    win_pct = wins / total if total > 0 else np.nan

    # Org tier — import lazily to avoid circular deps
    from src.features.build_features import _encode_org_tier

    org_tiers = []
    for r in rows:
        org = r.get("organization", "")
        tier = _encode_org_tier(org)
        if not np.isnan(tier):
            org_tiers.append(tier)
    best_tier = min(org_tiers) if org_tiers else np.nan

    return {
        "pre_ufc_total_fights": total,
        "pre_ufc_wins": wins,
        "pre_ufc_losses": losses,
        "pre_ufc_win_pct": win_pct,
        "pre_ufc_ko_rate": ko_rate,
        "pre_ufc_sub_rate": sub_rate,
        "pre_ufc_dec_rate": dec_rate,
        "pre_ufc_org_tier_best": best_tier,
    }


# ---------------------------------------------------------------------------
# Persistence: append new fighter rows to the supplement CSV
# ---------------------------------------------------------------------------

def append_to_supplement(rows: list[dict], output_path: Path) -> None:
    """Append newly scraped pre-UFC fight rows to the supplement CSV.

    Creates the file if it doesn't exist.  Deduplicates against existing
    rows by (event_date, fighter_a_key, fighter_b_key).
    """
    new_df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df.reindex(columns=OUTPUT_COLUMNS)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Dedupe
    combined["_fa_key"] = combined["fighter_a"].map(normalize_person_name)
    combined["_fb_key"] = combined["fighter_b"].map(normalize_person_name)
    combined["_org_len"] = combined["organization"].fillna("").astype(str).str.len()
    combined = combined.sort_values("_org_len", ascending=False)
    combined = combined.drop_duplicates(
        subset=["event_date", "_fa_key", "_fb_key"], keep="first"
    )
    combined = combined.drop(columns=["_fa_key", "_fb_key", "_org_len"], errors="ignore")
    combined = combined.reindex(columns=OUTPUT_COLUMNS)
    write_csv_atomically(combined, output_path)
