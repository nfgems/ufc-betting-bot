"""
On-demand pre-UFC career scraper.

Reads a fighter's pre-UFC history from identity-validated retained responses.
The collection transport is deliberately separate from this processor. Used by
the batch script (scripts/build_pre_ufc_career_supplement.py) and at
live-prediction time inside fighter_lookup.build_fight_features().
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fallback_scrapers import (
    scrape_sherdog_amateur_fights,
    scrape_sherdog_page,
    scrape_tapology_profile,
    scrape_tapology_fights,
    search_sherdog,
    search_tapology,
    search_tapology_candidates,
)
from src.data.io_utils import write_csv_atomically
from src.data.provider_observation_history import (
    ProviderObservationError,
    finalize_retained_history_rows,
    load_retained_provider_observation,
)
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

PROVENANCE_COLUMNS = [
    "fighter_a_id", "fighter_b_id", "history_row_id", "history_division",
    "observation_id", "captured_at_utc", "provider_id", "parser_revision",
    "eligibility_policy", "evidence_path", "evidence_sha256", "first_ufc_date",
    "available_before",
    "source_observation_id", "source_response_sha256", "source_envelope_sha256",
    "source_parser_revision", "source_eligibility_policy", "source_captured_at_utc",
    "source_evidence_path", "source_evidence_sha256", "source_provider_id",
    "source_fighter_id",
    "source_division",
    "source_attachment_path", "source_attachment_sha256",
]

OUTPUT_COLUMNS = BASE_COLUMNS + PROVENANCE_COLUMNS + [
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
    if query_cross and query_cross == resolved_cross:
        return True

    query_tokens = normalize_person_name(query_name).split()
    resolved_tokens = normalize_person_name(resolved_name).split()
    pairs = [
        (query_tokens, resolved_tokens),
        (resolved_tokens, query_tokens),
    ]
    for shorter, longer in pairs:
        if len(shorter) < 2 or len(longer) <= len(shorter):
            continue
        if shorter[0] == longer[0] and shorter[-1] == longer[-1]:
            return True
        if shorter[0] == longer[-1] and shorter[-1] == longer[0]:
            return True
    return False


_SUBMISSION_HOLD_TERMS = (
    "choke", "armbar", "arm bar", "kimura", "guillotine", "triangle", "americana",
    "heel hook", "kneebar", "knee bar", "toe hold", "toehold", "crank", "twister",
    "omoplata", "gogoplata", "ezekiel", "darce", "d'arce", "anaconda", "arm-triangle",
    "arm triangle", "leg lock", "leglock", "ankle lock", "wrist lock", "north-south",
    "peruvian necktie", "japanese necktie", "von flue", "smother", "keylock", "key lock",
    "calf slicer", "bicep slicer", "tap", "verbal submission",
)


def sherdog_method_to_ufc_method(method: str) -> str:
    """Normalize Sherdog method strings to match UFCStats format.

    Order matters: knockouts (including the dotted ``K.O`` spelling) and
    submissions (including bare hold names such as ``Triangle Choke``) are
    recognised before the no-contest test, which is a whole-word test so that
    ``(Punches)`` is never read as ``nc``.
    """
    m = method.strip().lower()
    if "tko" in m or "knockout" in m or re.search(r"\bk\.?o\b", m):
        return "KO/TKO"
    if (
        "technical submission" in m
        or "submission" in m
        or re.search(r"\bsub\b", m)
        or any(term in m for term in _SUBMISSION_HOLD_TERMS)
    ):
        return "SUB"
    if "decision" in m or re.search(r"\bdec\b", m):
        if "unanimous" in m:
            return "U-DEC"
        if "split" in m:
            return "S-DEC"
        if "majority" in m:
            return "M-DEC"
        return "U-DEC"
    if "no contest" in m or re.search(r"\bnc\b", m):
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


def _has_observed_win_method(value: object) -> bool:
    """A missing/unsupported method cannot establish a zero method-specific win rate."""
    if value is None or pd.isna(value):
        return False
    return sherdog_method_to_ufc_method(str(value)) in {"KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC", "DQ"}


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
    *,
    fighter_id: str = "",
    observation: Mapping[str, object] | None = None,
    available_before: object = None,
    division: str = "pro",
) -> list[dict]:
    """Convert raw scraped fights to supplement-schema rows, filtering to pre-UFC only."""
    if not fights:
        return []

    first_ufc = pd.to_datetime(first_ufc_date, errors="coerce", utc=True) if first_ufc_date else pd.NaT
    cutoff = pd.to_datetime(available_before, errors="coerce", utc=True)
    capture = pd.to_datetime((observation or {}).get("captured_at_utc"), errors="coerce", utc=True)
    if observation is not None and (pd.isna(first_ufc) or pd.isna(cutoff) or pd.isna(capture)):
        raise ProviderObservationError("pre_ufc_history_timing_missing")
    rows = []

    for fight in fights:
        fight_date = fight.get("event_date")
        if fight_date is None:
            continue

        fight_date = pd.to_datetime(fight_date, errors="coerce", utc=True)
        if pd.isna(fight_date):
            continue
        if pd.notna(first_ufc) and fight_date >= first_ufc:
            continue
        if observation is not None and (fight_date >= cutoff or fight_date >= capture):
            continue

        opponent = str(fight.get("opponent") or "").strip()
        if not opponent:
            continue
        result = str(fight.get("result", "")).strip().lower()
        method_raw = str(fight.get("method") or "")
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
        elif result == "loss":
            winner = opponent
        else:
            # A scheduled/unknown result cannot be converted to a loss.
            continue

        title = fight.get("is_title_bout")
        title_value = np.nan if title is None or pd.isna(title) else 1 if title is True or title == 1 else 0 if title is False or title == 0 else np.nan
        row = {
            "event_date": fight_date.strftime("%Y-%m-%d"),
            "fighter_a": fighter_name,
            "fighter_b": opponent,
            "winner": winner,
            "method": method,
            "finish_round": round_finished,
            "num_rounds": _infer_num_rounds(method, round_finished),
            "weight_class": "",
            # An unobserved title flag (None) is written as NaN for every row,
            # legacy or provider-bound; only a real 0/1 observation is kept.
            "title_bout": title_value,
            "source": source,
            "organization": organization,
        }

        for prefix in ["a_", "b_"]:
            for stat in STAT_COLUMNS:
                row[f"{prefix}{stat}"] = np.nan

        if observation is not None:
            metadata = dict(observation.get("row_metadata") or {})
            for key in PROVENANCE_COLUMNS:
                if key in observation:
                    metadata[key] = observation[key]
            row.update({key: metadata[key] for key in PROVENANCE_COLUMNS if key in metadata})
            row.update({
                "fighter_a_id": fighter_id,
                "fighter_b_id": np.nan,
                "history_division": division,
                "first_ufc_date": first_ufc.isoformat(),
                "available_before": cutoff.isoformat(),
            })
            identity = [fighter_id, division, row["event_date"], normalize_person_name(opponent),
                        organization.casefold()]
            row["history_row_id"] = hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        rows.append(row)

    return rows


def _scrape_fighter_history(
    fighter_name: str,
    first_ufc_date: str | None,
    *,
    sherdog_scraper: Callable[[str, str], tuple[dict, list[dict]]] | None = None,
    tapology_division: str,
    label: str,
    fighter_id: str | None = None,
    retained_observations: Sequence[Mapping[str, object] | Path] | None = None,
    available_before: object = None,
    verified_bindings: set[tuple[str, str, str]] | None = None,
    verified_binding_metadata: Mapping | None = None,
    verified_binding_name_keys: Mapping | None = None,
    history_root: Path | None = None,
    ordinary_provider_sources: Sequence[str] | None = None,
    accepted_membership: Mapping[str, object] | None = None,
    decision_paths: Sequence[Path] = (),
    provider_transport=None,
    provider_response_format: str = "html",
    provider_capture_cache: dict | None = None,
    provider_capture_log: dict | None = None,
    expected_decision_pins: Sequence[Mapping[str, object]] = (),
) -> list[dict]:
    """Validate every retained source before selecting one complete history.

    The historical scraper argument is kept for call compatibility, but is never
    invoked: missing custody/identity inputs are processing refusals, not searches.
    """
    if not fighter_id or not re.fullmatch(r"[0-9a-f]{16}", fighter_id):
        raise ProviderObservationError("pre_ufc_stable_identity_missing")
    ordinary_captures = []
    if ordinary_provider_sources is not None:
        from src.data.provider_membership import load_accepted_rosterwatch_membership
        from src.data.provider_observation_history import _time, _provider_number
        from src.data.provider_response_capture import acquire_provider_response
        if retained_observations is not None:
            raise ProviderObservationError("pre_ufc_ordinary_and_retained_inputs_conflict")
        if not accepted_membership or not decision_paths or history_root is None or available_before is None or not verified_bindings:
            raise ProviderObservationError("pre_ufc_ordinary_membership_decisions_history_and_cutoff_required")
        _time(available_before)
        if pd.isna(pd.to_datetime(first_ufc_date, errors="coerce", utc=True)):
            raise ProviderObservationError("pre_ufc_history_timing_missing")
        if any(not Path(path).is_file() for path in decision_paths):
            raise ProviderObservationError("pre_ufc_independent_decision_input_missing")
        from src.data.provider_consumer_references import _pin
        current_pins = [_pin(path) for path in decision_paths]
        if expected_decision_pins and list(expected_decision_pins) != current_pins:
            raise ProviderObservationError("pre_ufc_independent_decision_inputs_changed")
        expected_decision_pins = current_pins
        members = load_accepted_rosterwatch_membership(
            Path(accepted_membership["path"]), expected_sha256=accepted_membership["sha256"],
            expected_generation=accepted_membership["generation"],
        )
        if fighter_id not in set(members["stable_ufcstats_id"]):
            raise ProviderObservationError("pre_ufc_fighter_not_in_accepted_membership")
        sources = tuple(dict.fromkeys(ordinary_provider_sources))
        if not sources or any(source not in {"sherdog", "tapology"} for source in sources):
            raise ProviderObservationError("pre_ufc_ordinary_provider_source_invalid")
        routes = []
        for source in sources:
            bindings = {binding for binding in verified_bindings if binding[:2] == (fighter_id, source)}
            if len(bindings) != 1:
                raise ProviderObservationError("pre_ufc_unambiguous_reviewed_binding_required_before_fetch")
            binding = next(iter(bindings))
            url = str((verified_binding_metadata or {}).get(binding, {}).get("requested_url") or binding[2])
            if provider_response_format == "tapology_reader" and url.startswith("https://r.jina.ai/"):
                url = url.removeprefix("https://r.jina.ai/")
            if not _provider_number(source, url) or _provider_number(source, url) != _provider_number(source, binding[2]):
                raise ProviderObservationError("pre_ufc_reviewed_binding_route_unsupported")
            routes.append((source, url))
        descriptors = []
        cache = provider_capture_cache if provider_capture_cache is not None else {}
        if provider_capture_cache is not None and provider_capture_log is None:
            raise ProviderObservationError("pre_ufc_capture_cache_requires_original_attempt_log")
        capture_logs = provider_capture_log if provider_capture_log is not None else {}
        for source, url in routes:
            key = (fighter_id, source, url, provider_response_format)
            attempt_log = capture_logs.setdefault(key, [])
            if key not in cache:
                try:
                    cache[key] = acquire_provider_response(source, url, history_root=history_root,
                        transport=provider_transport, response_format=provider_response_format, capture_log=attempt_log)
                except Exception as exc:
                    cache[key] = exc
                    raise
            if isinstance(cache[key], Exception):
                raise cache[key]
            descriptors.append(cache[key])
            ordinary_captures.extend(attempt_log)
        retained_observations = descriptors
    if not retained_observations or not verified_bindings or available_before is None or history_root is None:
        raise ProviderObservationError("pre_ufc_retained_inputs_required")
    choices: list[tuple[str, str, list[dict]]] = []
    seen: set[str] = set()
    bout_claims: dict[tuple[str, str], dict] = {}
    for descriptor in retained_observations:
        observation = load_retained_provider_observation(
            descriptor, fighter_id=fighter_id, fighter_name=fighter_name,
            available_before=available_before, verified_bindings=verified_bindings,
            verified_binding_metadata=verified_binding_metadata,
            verified_binding_name_keys=verified_binding_name_keys,
            division=tapology_division, history_root=history_root,
            consumer={"kind": "pre_ufc", "id": f"{fighter_id}:{tapology_division}"},
        )
        observation_id = str(observation["observation_id"])
        if observation_id in seen:
            continue
        seen.add(observation_id)
        rows = _rows_from_fights(
            observation["fights"], fighter_name, first_ufc_date, observation["source"],
            fighter_id=fighter_id, observation=observation,
            available_before=available_before, division=tapology_division,
        )
        rows = finalize_retained_history_rows(rows, history_root=history_root)
        if ordinary_provider_sources is not None:
            from src.data.provider_consumer_references import record_provider_consumer_artifact
            for row in rows:
                record_provider_consumer_artifact(
                    Path(row["source_attachment_path"]), evidence_paths=[Path(row["source_evidence_path"])],
                    accepted_membership=accepted_membership, decision_paths=decision_paths,
                    expected_decision_pins=expected_decision_pins,
                    capture_descriptors=ordinary_captures,
                    history_root=history_root, kind="career_attachment",
                )
                row["provider_reference_schema"] = 1
        for row in rows:
            bout = (row["event_date"], normalize_person_name(row["fighter_b"]))
            prior = bout_claims.get(bout)
            if prior is not None:
                for field in ("winner", "method", "finish_round"):
                    left, right = prior.get(field), row.get(field)
                    left_missing = left is None or pd.isna(left) or left == ""
                    right_missing = right is None or pd.isna(right) or right == ""
                    if field == "winner" and left_missing != right_missing:
                        raise ProviderObservationError("pre_ufc_conflicting_bout_result")
                    if not left_missing and not right_missing and left != right:
                        raise ProviderObservationError(f"pre_ufc_conflicting_bout_{field}")
            bout_claims[bout] = row
        rows = list({row["history_row_id"]: row for row in rows}.values())
        choices.append((str(observation["source"]), observation_id, rows))
    choices.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
    best_rows = choices[0][2] if choices else []
    logger.info("%s: %d independently bound %s fights", fighter_name, len(best_rows), label)
    return best_rows


# ---------------------------------------------------------------------------
# Main entry point: scrape one fighter from all sources, return the best
# ---------------------------------------------------------------------------

def scrape_fighter_pre_ufc_fights(
    fighter_name: str,
    first_ufc_date: str | None,
    **retained_inputs,
) -> list[dict]:
    """
    Read validated retained professional history without invoking transport.

    Returns fight rows in supplement CSV schema (fights_cleaned-compatible),
    with per-fight stats set to NaN (not available from regional sources).
    """
    return _scrape_fighter_history(
        fighter_name,
        first_ufc_date,
        sherdog_scraper=scrape_sherdog_page,
        tapology_division="pro",
        label="pre-UFC",
        **retained_inputs,
    )


def scrape_fighter_amateur_fights(
    fighter_name: str,
    first_ufc_date: str | None,
    **retained_inputs,
) -> list[dict]:
    """Scrape a fighter's amateur MMA history from Sherdog and Tapology."""
    return _scrape_fighter_history(
        fighter_name,
        first_ufc_date,
        sherdog_scraper=scrape_sherdog_amateur_fights,
        tapology_division="am",
        label="amateur pre-UFC",
        **retained_inputs,
    )


# ---------------------------------------------------------------------------
# Summary computation for a single fighter (no src/ caller today: live and
# training both use build_features._compute_pre_ufc_summary)
# ---------------------------------------------------------------------------

def _compute_single_fighter_summary(
    rows: list[dict],
    *,
    prefix: str,
    include_org_tier: bool,
) -> dict:
    """Compute prefixed summary features from scraped supplement-style rows."""
    if not rows:
        return {}

    total = len(rows)
    wins = sum(1 for r in rows if r.get("winner") == r.get("fighter_a"))
    losses = sum(
        1 for r in rows
        if r.get("winner") and r.get("winner") != r.get("fighter_a")
        and not (isinstance(r.get("winner"), float) and np.isnan(r["winner"]))
    )

    retained = _bound_history_mask(pd.DataFrame(rows)).any()
    complete_win_methods = all(_has_observed_win_method(row.get("method")) for row in rows
                               if row.get("winner") == row.get("fighter_a"))
    if wins > 0 and (not retained or complete_win_methods):
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

    summary = {
        f"{prefix}total_fights": total,
        f"{prefix}wins": wins,
        f"{prefix}losses": losses,
        f"{prefix}win_pct": win_pct,
        f"{prefix}ko_rate": ko_rate,
        f"{prefix}sub_rate": sub_rate,
        f"{prefix}dec_rate": dec_rate,
    }

    if include_org_tier:
        from src.features.build_features import _encode_org_tier

        org_tiers = []
        for r in rows:
            org = r.get("organization", "")
            tier = _encode_org_tier(org, r.get("event_date"))
            if not np.isnan(tier):
                org_tiers.append(tier)
        summary[f"{prefix}org_tier_best"] = min(org_tiers) if org_tiers else np.nan

    return summary


def compute_single_fighter_summary(rows: list[dict]) -> dict:
    """Compute pre-UFC career summary features from raw fight rows."""
    return _compute_single_fighter_summary(
        rows,
        prefix="pre_ufc_",
        include_org_tier=True,
    )


def compute_single_fighter_amateur_summary(rows: list[dict]) -> dict:
    """Compute amateur career summary features from raw fight rows."""
    return _compute_single_fighter_summary(
        rows,
        prefix="amateur_",
        include_org_tier=False,
    )


# Spellings under which one person appears in the supplement because the two
# sources (or the two fighters' pages) name them differently. Used ONLY to fold
# the duplicate-check key, never to rewrite a stored name. Each pair is either
# a ring name (Patricio "Pitbull" Freire) or a Sherdog spelling that was tied
# to the UFCStats spelling by the same dated bout against the same opponent
# on 2026-09-16 (acquisition/sherdog01/anchor_gate_results.jsonl). Pure
# token-order and spacing variants (Meng Ding / Ding Meng, Won Bin Ki /
# WonBin Ki) need no entry: the key is order- and space-insensitive.
_SUPPLEMENT_REVIEWED_SPELLINGS = {
    "Patricio Pitbull": "Patricio Freire",
    "Ahmad Sohail Hassanzada": "Ahmad Hassanzada",
    "Gary Balletto Jr.": "Gary Balletto",
    "Lucas Triverio Camacho": "Lucas Camacho",
    "Matthew Adams": "Matt Adams",
    "Matthieu Letho Duclos": "Matthieu Duclos",
    "Nina Nikolija Milosevic": "Nina Milosevic",
    "Cameron Nelson": "Cam Nelson",
    "Gabriel Lourenco": "Gabriel Lorenco",
    "Joseph Kropschot": "Joe Kropschot",
    "Jonathan Kunneman": "Jon Kunneman",
    "Adrian Luna Martinetti": "Juan Adrian Martinetti",
    "Kyung Pyo Kim": "KyeungPyo Kim",
    "Nick Galanti": "Nicholas Galanti",
    "Oliver Schmid": "Ollie Schmid",
    "Jilunzhou Qiu": "Qiu Lun",
    "Reginaldo Geraldo Jr.": "Reginaldo Junior",
    "Roman Puga": "Roman Gabriel Puga",
    "Roque Conceicao Moreira Junior": "Roque Conceicao",
    "Sean King": "Sean King III",
    "Stanley Dorsainvil": "Stan Dorsainvil",
    "Tommy Cuozzi": "Tommy Cuozzi Jr.",
    "Vanilton Antunes": "Vanilto Antunes",
    "Vlastislav Cepo": "Vlasto Cepo",
    # Added 2026-09-17 (owner decision: residual same-day duplicate pairs). Each
    # pair is tied by one fighter's own record listing the same bout on the same
    # day under both opponent spellings with the same result (evidence in
    # acquisition/sherdog01/reviewed_spellings_2026-09-17.json). Both Cerqueira
    # spellings are tracked under one UFCStats id (09a4447655fd4b7b).
    "Rafael Cerquiera": "Rafael Cerqueira",
    "Freddy Emiliano Linares": "Emiliano Linares",
    "Antonio Rogerio Nogueira": "Rogerio Nogueira",
    "Anistavio Medeiros de Figueiredo": "Anistavio Medeiros",
    "Leomana Martinez": "Mana Martinez",
    "Leonardo Mafra Teixeira": "Leonardo Mafra",
    "Wendell de Oliveira Marques": "Wendell Oliveira Marques",
    "Tsuyoshi Kosaka": "Tsuyoshi Kohsaka",
    "Darya Zheleznyakova": "Daria Zhelezniakova",
    "Tahir Abdullaev": "Tahir Abdullayev",
}
_SUPPLEMENT_REVIEWED_SPELLING_KEYS = {
    normalize_person_name(source): normalize_person_name(target)
    for source, target in _SUPPLEMENT_REVIEWED_SPELLINGS.items()
}


def _supplement_pair_name_keys(value: object) -> tuple[str, str]:
    """(token-order-insensitive key, space-free key) for one name in the duplicate check."""
    key = normalize_person_name(str(value or ""))
    key = _SUPPLEMENT_REVIEWED_SPELLING_KEYS.get(key, key)
    tokens = key.split()
    return "".join(sorted(tokens)), "".join(tokens)


def _dedupe_supplement_rows(rows: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Dedupe mirrored supplement rows by event date, unordered pair, and organization.

    This row-level rule is deliberately exact on spellings: one person can be
    tracked under two UFCStats spellings (Patricio Freire / Patricio Pitbull,
    same stable ID) with a complete own-page scrape under each, and the
    per-fighter collapse and summary key on the stateful identity
    (``fighter_key``), which never folds two reviewed ring-name spellings, so
    folding spellings here would move bouts from one tracked spelling to the
    other. Label-variant and alias duplicates of the same bout are collapsed
    later, per fighter identity, in
    build_features._drop_duplicate_supplement_bouts.
    """
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return df
    bound_mask = _bound_history_mask(df)
    if bound_mask.any():
        bound = _dedupe_bound_history(df.loc[bound_mask])
        legacy = _dedupe_supplement_rows(df.loc[~bound_mask])
        combined = pd.concat([legacy, bound], ignore_index=True)
        combined.attrs = dict(df.attrs)
        return combined
    required = {"event_date", "fighter_a", "fighter_b", "organization"}
    if not required.issubset(df.columns):
        return df

    deduped = df.copy()
    event_dates = pd.to_datetime(deduped["event_date"], errors="coerce")
    deduped["_event_date_key"] = event_dates.dt.strftime("%Y-%m-%d")
    missing_dates = deduped["_event_date_key"].isna()
    deduped.loc[missing_dates, "_event_date_key"] = (
        deduped.loc[missing_dates, "event_date"].fillna("").astype(str).str.strip()
    )

    fighter_a_keys = deduped["fighter_a"].fillna("").astype(str).map(normalize_person_name)
    fighter_b_keys = deduped["fighter_b"].fillna("").astype(str).map(normalize_person_name)
    pair_keys = [
        tuple(sorted((fighter_a_key, fighter_b_key)))
        for fighter_a_key, fighter_b_key in zip(fighter_a_keys, fighter_b_keys)
    ]
    deduped["_pair_lo"] = [pair[0] for pair in pair_keys]
    deduped["_pair_hi"] = [pair[1] for pair in pair_keys]
    deduped["_org_key"] = deduped["organization"].fillna("").astype(str).str.strip().str.lower()
    deduped["_org_len"] = deduped["organization"].fillna("").astype(str).str.len()

    deduped = deduped.sort_values("_org_len", ascending=False, kind="stable")
    deduped = deduped.drop_duplicates(
        subset=["_event_date_key", "_pair_lo", "_pair_hi", "_org_key"],
        keep="first",
    )
    return deduped.drop(
        columns=["_event_date_key", "_pair_lo", "_pair_hi", "_org_key", "_org_len"],
        errors="ignore",
    )


def _bound_history_mask(frame: pd.DataFrame) -> pd.Series:
    marked = pd.Series(False, index=frame.index)
    for column in PROVENANCE_COLUMNS:
        if column not in {"fighter_a_id", "fighter_b_id", "first_ufc_date", "available_before"} and column in frame:
            marked |= frame[column].fillna("").astype(str).str.strip().ne("")
    return marked


def _dedupe_bound_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep each stable owner's bout once with its earliest original evidence."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    from src.data.provider_observation_history import verify_retained_history_row

    for row in work.to_dict("records"):
        row_reference = frame.attrs.get("provider_row_references", {}).get(row.get("source_attachment_path"))
        verify_retained_history_row(row, references=row_reference or frame.attrs.get("provider_references"))
    work["__opponent"] = work["fighter_b"].map(normalize_person_name)
    work["__date"] = pd.to_datetime(work["event_date"], errors="coerce", utc=True)
    keys = ["fighter_a_id", "history_division", "__date", "__opponent"]
    for _identity, group in work.groupby(keys, dropna=False):
        outcomes = set()
        for row in group.to_dict("records"):
            winner = row.get("winner")
            outcomes.add("draw" if pd.isna(winner) else "win" if winner == row["fighter_a"] else "loss")
        if len(outcomes) > 1:
            raise ProviderObservationError("pre_ufc_conflicting_bout_result")
        for column in ("method", "finish_round"):
            observed = group[column].dropna()
            observed = observed.loc[observed.astype(str).str.strip().ne("")]
            if observed.nunique() > 1:
                raise ProviderObservationError(f"pre_ufc_conflicting_bout_{column}")
    work["__capture"] = pd.to_datetime(work["source_captured_at_utc"], errors="coerce", utc=True)
    work = work.sort_values("__capture", kind="stable", na_position="last")
    return work.drop_duplicates(keys, keep="first").drop(columns=["__capture", "__opponent", "__date"])


# ---------------------------------------------------------------------------
# Persistence: append new fighter rows to the supplement CSV
# ---------------------------------------------------------------------------

def append_to_supplement(
    rows: list[dict],
    output_path: Path,
    *,
    dedupe_mirrors: bool = False,
) -> None:
    """Append newly scraped pre-UFC fight rows to the supplement CSV.

    Creates the file if it doesn't exist.  Deduplicates against existing
    rows by (event_date, fighter_a_key, fighter_b_key).
    """
    from src.data.provider_consumer_references import snapshot_provider_references
    previous = snapshot_provider_references(output_path) if output_path.exists() else None
    new_df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df.reindex(columns=[*OUTPUT_COLUMNS, *(["provider_reference_schema"] if "provider_reference_schema" in new_df else [])])

    if output_path.exists():
        existing = pd.read_csv(output_path, dtype={column: object for column in PROVENANCE_COLUMNS})
        combined = pd.concat([existing, new_df], ignore_index=True)
        from src.data.provider_consumer_references import verify_provider_consumer_artifact
        prior_document = verify_provider_consumer_artifact(output_path)
        if prior_document is not None:
            # Newly captured rows still name local originals; merging their
            # references is required before an unavailable prior carrier can
            # be verified. Existing-only verification uses the saved manifest.
            existing.attrs["provider_references"] = prior_document
            _dedupe_bound_history(existing.loc[_bound_history_mask(existing)])
            combined.attrs["provider_row_references"] = {value: prior_document
                for value in existing.get("source_attachment_path", []) if isinstance(value, str) and value}
    else:
        combined = new_df

    if dedupe_mirrors:
        combined = _dedupe_supplement_rows(combined)

    retained_attrs = dict(combined.attrs)

    bound_mask = _bound_history_mask(combined)
    bound = _dedupe_bound_history(combined.loc[bound_mask])
    combined = combined.loc[~bound_mask].copy()

    # Dedupe
    combined["_fa_key"] = combined["fighter_a"].map(normalize_person_name)
    combined["_fb_key"] = combined["fighter_b"].map(normalize_person_name)
    combined["_org_len"] = combined["organization"].fillna("").astype(str).str.len()
    combined = combined.sort_values("_org_len", ascending=False)
    combined = combined.drop_duplicates(
        subset=["event_date", "_fa_key", "_fb_key"], keep="first"
    )
    combined = combined.drop(columns=["_fa_key", "_fb_key", "_org_len"], errors="ignore")
    combined = pd.concat([combined, bound], ignore_index=True)
    combined.attrs = retained_attrs
    combined = combined.reindex(columns=[*OUTPUT_COLUMNS, *(["provider_reference_schema"] if "provider_reference_schema" in combined else [])])
    write_csv_atomically(combined, output_path)
    parents = [Path(value) for value in combined.get("source_attachment_path", [])
               if isinstance(value, str) and value and Path(value + ".provider-references.json").exists()]
    if parents or previous is not None:
        from src.data.provider_consumer_references import record_provider_consumer_artifact
        record_provider_consumer_artifact(output_path, evidence_paths=[Path(value) for value in
            combined["source_attachment_path"] if isinstance(value, str) and value],
            accepted_membership=None, decision_paths=(), history_root=output_path.parent,
            kind="career_csv", parent_artifacts=parents,
            parent_references=[previous] if previous is not None else ())
