"""
LLM Operator — the decision-making brain that receives model outputs,
conducts its own research, and makes final bet/no-bet decisions.

The XGBoost model produces a probability. The operator treats that as ONE
input among many. It runs its own research pipeline, synthesizes everything,
and makes the final call.

Pipeline:
    model.predict() → value.detect_value() → operator.evaluate() → execute_bet()
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd

from src.config import DATA_DIR, LOGS_DIR
from src.data.name_utils import same_person_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPERATOR_ENABLED = os.getenv("LLM_OPERATOR_ENABLED", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
OPERATOR_MODE: Literal["gate", "advisory"] = (
    "gate" if os.getenv("LLM_OPERATOR_MODE", "gate").strip().lower() == "gate"
    else "advisory"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_OPERATOR_MODEL", "gemini-2.5-pro")

# Paths
OPERATOR_DIR = DATA_DIR / "operator"
OPERATOR_DIR.mkdir(parents=True, exist_ok=True)
BLIND_SPOTS_PATH = OPERATOR_DIR / "blind_spots.json"
DECISION_LOG_PATH = OPERATOR_DIR / "decision_log.jsonl"  # append-only, one JSON object per line

# Exposure limits
MAX_BETS_PER_EVENT = 3  # Flag concentration risk above this

# Session-level decision cache: fight_key → (OperatorDecision, epoch)
# Prevents re-evaluating the same fight across loop cycles and across
# value/conviction trader passes within a single cycle.
_decision_cache: dict[str, tuple["OperatorDecision", float]] = {}
_decision_cache_lock = threading.Lock()
# Per-key locks: prevents two threads from evaluating the same fight
# concurrently (they'd both miss the cache and double-call the LLM).
_decision_inflight: dict[str, threading.Lock] = {}

# Re-evaluate a fight after this many seconds (4 hours) so that new
# information (injuries, weigh-in results, etc.) can be incorporated.
CACHE_TTL_SECONDS = float(os.getenv("LLM_OPERATOR_CACHE_TTL", str(4 * 3600)))

# Disk-backed cache file — survives process restarts.
_DECISION_CACHE_FILE = OPERATOR_DIR / "decision_cache.json"


def _fight_cache_key(fighter_a: str, fighter_b: str) -> str:
    """Canonical cache key for a fight (order-independent)."""
    pair = sorted([fighter_a.strip().lower(), fighter_b.strip().lower()])
    return f"{pair[0]}|{pair[1]}"


def _normalize_event_date(value: object) -> str:
    """Normalize event identifiers down to a durable YYYY-MM-DD key when possible."""
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return text.casefold()
    return parsed.strftime("%Y-%m-%d")


def _existing_bet_matches_fight(
    existing_bet: dict,
    *,
    fighter_a: str,
    fighter_b: str,
    event_date: str = "",
) -> bool:
    """Check whether an existing ledger row refers to this same fight."""
    existing_a = str(existing_bet.get("fighter") or existing_bet.get("fighter_a") or "").strip()
    existing_b = str(existing_bet.get("opponent") or existing_bet.get("fighter_b") or "").strip()
    if not existing_a or not existing_b:
        return False

    names_match = (
        same_person_name(fighter_a, existing_a) and same_person_name(fighter_b, existing_b)
    ) or (
        same_person_name(fighter_a, existing_b) and same_person_name(fighter_b, existing_a)
    )
    if not names_match:
        return False

    candidate_event_date = _normalize_event_date(event_date)
    existing_event_date = _normalize_event_date(
        existing_bet.get("event_date") or existing_bet.get("market_event_date")
    )
    if candidate_event_date and existing_event_date and candidate_event_date != existing_event_date:
        return False

    return True


def _has_existing_bet_for_fight(
    *,
    fighter_a: str,
    fighter_b: str,
    existing_bets: list[dict] | None,
    event_date: str = "",
) -> bool:
    """Return True when the fight is already present in the current ledgers."""
    return any(
        _existing_bet_matches_fight(
            existing_bet,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            event_date=event_date,
        )
        for existing_bet in (existing_bets or [])
    )


def _save_decision_cache_to_disk() -> None:
    """Persist the in-memory decision cache to disk so it survives restarts."""
    try:
        serializable: dict[str, dict] = {}
        for key, (decision, cached_at) in _decision_cache.items():
            serializable[key] = {
                "decision": asdict(decision),
                "cached_at": cached_at,
            }
        _DECISION_CACHE_FILE.write_text(json.dumps(serializable, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to persist operator decision cache to disk: %s", exc)


def _load_decision_cache_from_disk() -> None:
    """Load persisted decision cache from disk into memory (called once at import)."""
    if not _DECISION_CACHE_FILE.exists():
        return
    try:
        data = json.loads(_DECISION_CACHE_FILE.read_text(encoding="utf-8"))
        now = time.time()
        restored = 0
        for key, entry in data.items():
            cached_at = float(entry.get("cached_at", 0))
            if now - cached_at >= CACHE_TTL_SECONDS:
                continue
            d = entry.get("decision", {})
            decision = OperatorDecision(
                verdict=d.get("verdict", "PASS"),
                confidence=float(d.get("confidence", 0.0)),
                model_prob=float(d.get("model_prob", 0.5)),
                operator_prob=float(d.get("operator_prob", 0.5)),
                rationale=d.get("rationale", ""),
                research_summary=dict(d.get("research_summary") or {}),
                risk_flags=list(d.get("risk_flags") or []),
                timestamp=d.get("timestamp", ""),
                fighter_a=d.get("fighter_a", ""),
                fighter_b=d.get("fighter_b", ""),
                bet_on=d.get("bet_on", ""),
                bet_side=d.get("bet_side", ""),
                edge=float(d.get("edge", 0.0)),
                market_prob=float(d.get("market_prob", 0.0)),
                provenance=dict(d.get("provenance") or {}),
            )
            _decision_cache[key] = (decision, cached_at)
            restored += 1
        if restored:
            logger.info("Restored %d operator decision cache entries from disk", restored)
    except Exception as exc:
        logger.debug("Failed to load operator decision cache from disk: %s", exc)


def clear_decision_cache() -> None:
    """Clear the session decision cache (e.g. when a new event starts)."""
    with _decision_cache_lock:
        _decision_cache.clear()
    try:
        _DECISION_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    logger.info("Operator decision cache cleared")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResearchFindings:
    """Structured output from the research pipeline."""

    recency_flags: list[str] = field(default_factory=list)
    matchup_analysis: str = ""
    motivation_flags: list[str] = field(default_factory=list)
    social_signals: dict = field(default_factory=dict)
    blind_spot_matches: list[str] = field(default_factory=list)
    exposure_warning: str = ""


@dataclass
class OperatorDecision:
    """Final decision from the LLM Operator for a single bet."""

    verdict: Literal["PASS", "BLOCK"]
    confidence: float  # 0.0–1.0, operator's own confidence
    model_prob: float  # what the model said
    operator_prob: float  # operator's adjusted probability
    rationale: str  # written explanation (ALWAYS logged)
    research_summary: dict  # structured research findings
    risk_flags: list[str]  # any flags raised during research
    timestamp: str  # ISO timestamp
    fighter_a: str = ""
    fighter_b: str = ""
    bet_on: str = ""
    bet_side: str = ""
    edge: float = 0.0
    market_prob: float = 0.0
    provenance: dict = field(default_factory=dict)


# Load persisted cache from disk at import time so process restarts don't
# lose cached decisions (the most common cause of duplicate API calls).
_load_decision_cache_from_disk()


# ---------------------------------------------------------------------------
# 1. Recency Context
# ---------------------------------------------------------------------------

def _check_recency_context(
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> list[str]:
    """Flag regime changes that rolling averages can't capture."""
    flags = []

    # Long layoffs (2+ years)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        days_key = f"{side}_days_since_last_fight"
        days = features.get(days_key)
        if days is not None:
            try:
                days = float(days)
            except (TypeError, ValueError):
                continue
            if days > 730:
                flags.append(
                    f"{name} has not fought in {int(days)} days "
                    f"({days / 365:.1f} years) — long layoff risk"
                )
            elif days > 365:
                flags.append(
                    f"{name} returning after {int(days)} day layoff "
                    f"({days / 365:.1f} years)"
                )

    # Short-notice replacements
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        notice_key = f"{side}_is_short_notice"
        if features.get(notice_key):
            flags.append(f"{name} is a short-notice replacement")

    # Weight class changes (check if fighting at unusual weight)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        wc_change_key = f"{side}_weight_class_change"
        if features.get(wc_change_key):
            flags.append(f"{name} is fighting at a new weight class")

    # Debut fighters
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        debut_key = f"{side}_is_debut"
        num_fights_key = f"{side}_num_fights"
        is_debut = features.get(debut_key, False)
        num_fights = features.get(num_fights_key, 0)
        try:
            num_fights = int(num_fights) if num_fights is not None else 0
        except (TypeError, ValueError):
            num_fights = 0
        if is_debut or num_fights == 0:
            flags.append(f"{name} is making their UFC debut — limited data")

    return flags


# ---------------------------------------------------------------------------
# 2. Style Matchup Reasoning (extracted from feature vector)
# ---------------------------------------------------------------------------

def _analyze_matchup_from_features(
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> str:
    """Build a matchup narrative from the feature vector for LLM synthesis."""
    lines = []

    def _get(key, default=None):
        val = features.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _fmt_pct(val) -> str:
        if val is None:
            return "unknown"
        try:
            val = float(val)
        except (TypeError, ValueError):
            return "unknown"
        if 0.0 <= val <= 1.0:
            val *= 100.0
        return f"{val:.0f}%"

    # Striking differential
    a_slpm = _get("a_roll_slpm")
    b_slpm = _get("b_roll_slpm")
    a_str_acc = _get("a_roll_str_acc")
    b_str_acc = _get("b_roll_str_acc")
    if a_slpm and b_slpm:
        acc_a = _fmt_pct(a_str_acc)
        acc_b = _fmt_pct(b_str_acc)
        lines.append(
            f"Striking output: {fighter_a} {a_slpm:.1f} SLpM "
            f"({acc_a} acc) vs {fighter_b} {b_slpm:.1f} SLpM "
            f"({acc_b} acc)"
        )

    # Grappling differential
    a_td_avg = _get("a_roll_td_avg")
    b_td_avg = _get("b_roll_td_avg")
    a_td_acc = _get("a_roll_td_acc")
    b_td_acc = _get("b_roll_td_acc")
    a_td_def = _get("a_roll_td_def")
    b_td_def = _get("b_roll_td_def")
    if a_td_avg and b_td_avg:
        td_acc_a = _fmt_pct(a_td_acc)
        td_acc_b = _fmt_pct(b_td_acc)
        td_def_a = _fmt_pct(a_td_def)
        td_def_b = _fmt_pct(b_td_def)
        lines.append(
            f"Takedowns: {fighter_a} {a_td_avg:.1f}/fight "
            f"({td_acc_a} acc, {td_def_a} def) vs "
            f"{fighter_b} {b_td_avg:.1f}/fight "
            f"({td_acc_b} acc, {td_def_b} def)"
        )

    # Wrestler vs striker mismatch (td_def is in 0-100 range)
    if a_td_avg and b_td_avg:
        if a_td_avg > 3.0 and b_td_def is not None and b_td_def < 55:
            lines.append(
                f"MISMATCH: {fighter_a} is an active wrestler vs "
                f"{fighter_b}'s weak TDD ({b_td_def:.0f}%)"
            )
        if b_td_avg > 3.0 and a_td_def is not None and a_td_def < 55:
            lines.append(
                f"MISMATCH: {fighter_b} is an active wrestler vs "
                f"{fighter_a}'s weak TDD ({a_td_def:.0f}%)"
            )

    # Stance matchup
    a_stance = features.get("a_stance", "")
    b_stance = features.get("b_stance", "")
    if a_stance and b_stance and a_stance != b_stance:
        lines.append(f"Stance matchup: {fighter_a} ({a_stance}) vs {fighter_b} ({b_stance})")

    # Win method profiles (rates are in 0-100 range)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        ko_rate = _get(f"{side}_ko_rate")
        sub_rate = _get(f"{side}_sub_rate")
        dec_rate = _get(f"{side}_dec_rate")
        if ko_rate is not None and sub_rate is not None:
            lines.append(
                f"{name} finishes: KO {_fmt_pct(ko_rate)}, Sub {_fmt_pct(sub_rate)}, "
                f"Dec {_fmt_pct(dec_rate)}" if dec_rate is not None
                else f"{name} finishes: KO {_fmt_pct(ko_rate)}, Sub {_fmt_pct(sub_rate)}"
            )

    # Pre-UFC career context
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        org_tier = features.get(f"{side}_pre_ufc_org_tier")
        pre_record = features.get(f"{side}_pre_ufc_record_depth")
        if org_tier is not None:
            lines.append(f"{name} pre-UFC org tier: {org_tier}")
        if pre_record is not None:
            lines.append(f"{name} pre-UFC record depth: {pre_record}")

    return "\n".join(lines) if lines else "Insufficient feature data for matchup analysis."


# ---------------------------------------------------------------------------
# 3. Motivation / Stakes Signals
# ---------------------------------------------------------------------------

def _check_motivation_signals(
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> list[str]:
    """Flag motivation-related signals from available features."""
    flags = []

    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        # Losing streak — potential desperation or contract fight
        streak = features.get(f"{side}_lose_streak", 0)
        try:
            streak = int(streak) if streak is not None else 0
        except (TypeError, ValueError):
            streak = 0
        if streak >= 3:
            flags.append(
                f"{name} is on a {streak}-fight losing streak — "
                "potential contract fight / must-win"
            )
        elif streak == 2:
            flags.append(
                f"{name} has lost 2 straight — possible urgency"
            )

        # Win streak — riding momentum
        w_streak = features.get(f"{side}_current_win_streak", 0)
        try:
            w_streak = int(w_streak) if w_streak is not None else 0
        except (TypeError, ValueError):
            w_streak = 0
        if w_streak >= 5:
            flags.append(
                f"{name} is on a {w_streak}-fight win streak — "
                "high momentum, likely motivated"
            )

        # Age concerns
        age = features.get(f"{side}_age")
        if age is not None:
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = None
            if age is not None and age >= 38:
                flags.append(
                    f"{name} is {age:.0f} years old — "
                    "potential age/retirement factor"
                )

    return flags


# ---------------------------------------------------------------------------
# 4. Historical Model Blind Spots
# ---------------------------------------------------------------------------

def load_blind_spots() -> list[dict]:
    """Load known model failure patterns from disk."""
    if not BLIND_SPOTS_PATH.exists():
        return []
    try:
        with open(BLIND_SPOTS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load blind spots: %s", exc)
        return []


def save_blind_spots(blind_spots: list[dict]) -> None:
    """Persist blind spot patterns to disk."""
    with open(BLIND_SPOTS_PATH, "w") as f:
        json.dump(blind_spots, f, indent=2)


def _check_blind_spots(
    features: dict,
    fighter_a: str,
    fighter_b: str,
    model_prob_a: float,
    market_prob_a: float,
) -> list[str]:
    """Check if the current fight matches any known model blind spots."""
    blind_spots = load_blind_spots()
    if not blind_spots:
        return []

    matches = []
    for spot in blind_spots:
        pattern = spot.get("pattern", {})
        matched = True

        for key, condition in pattern.items():
            feature_val = features.get(key)
            if feature_val is None:
                matched = False
                break

            if isinstance(condition, dict):
                # Threshold conditions: {"op": "gt", "value": 3.0}
                op = condition.get("op", "eq")
                threshold = condition.get("value")
                try:
                    feature_val = float(feature_val)
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    matched = False
                    break

                if op == "gt" and not (feature_val > threshold):
                    matched = False
                elif op == "lt" and not (feature_val < threshold):
                    matched = False
                elif op == "gte" and not (feature_val >= threshold):
                    matched = False
                elif op == "lte" and not (feature_val <= threshold):
                    matched = False
                elif op == "eq" and feature_val != threshold:
                    matched = False
            else:
                # Plain value — equality check (supports strings and numbers)
                try:
                    if float(feature_val) != float(condition):
                        matched = False
                except (TypeError, ValueError):
                    if str(feature_val) != str(condition):
                        matched = False

        if matched:
            matches.append(
                f"Blind spot match: {spot.get('description', 'unknown pattern')} "
                f"(historical accuracy: {spot.get('accuracy', 'N/A')})"
            )

    return matches


# ---------------------------------------------------------------------------
# 6. Correlated Exposure Check
# ---------------------------------------------------------------------------

def _check_correlated_exposure(
    event_title: str,
    existing_bets: list[dict],
    event_date: str = "",
) -> str:
    """Flag concentration risk when multiple bets are on the same event.

    Matches on event_title OR event_date — each compared against its own
    field on the existing bet, not cross-compared.
    """
    if not existing_bets:
        return ""

    # Normalise the candidate event date to YYYY-MM-DD for comparison.
    candidate_date = _normalize_event_date(event_date) or _normalize_event_date(event_title)
    candidate_title = (event_title or "").strip()

    same_event_bets = []
    for b in existing_bets:
        bet_title = (b.get("event_title") or "").strip()
        bet_date = _normalize_event_date(b.get("event_date") or b.get("market_event_date"))
        if candidate_title and bet_title and candidate_title == bet_title:
            same_event_bets.append(b)
        elif candidate_date and bet_date and candidate_date == bet_date:
            same_event_bets.append(b)
        # Don't double-count if both matched
    # Deduplicate (a bet could match on both title and date)
    same_event_bets = list({id(b): b for b in same_event_bets}.values())

    count = len(same_event_bets)
    if count >= MAX_BETS_PER_EVENT:
        return (
            f"CONCENTRATION RISK: Already {count} bets on this event. "
            f"One bad judging night or doctor stoppage affects all. "
            f"Consider reducing position size."
        )
    elif count >= 2:
        return (
            f"Moderate exposure: {count} existing bets on this event. "
            f"Adding another increases correlated risk."
        )
    return ""


# ---------------------------------------------------------------------------
# Research aggregator
# ---------------------------------------------------------------------------

def run_research_pipeline(
    *,
    features: dict,
    fighter_a: str,
    fighter_b: str,
    model_prob_a: float,
    market_prob_a: float,
    event_title: str = "",
    event_date: str = "",
    existing_bets: list[dict] | None = None,
) -> ResearchFindings:
    """Run all research layers and aggregate findings."""
    findings = ResearchFindings()

    # 1. Recency context
    findings.recency_flags = _check_recency_context(features, fighter_a, fighter_b)

    # 2. Matchup analysis
    findings.matchup_analysis = _analyze_matchup_from_features(
        features, fighter_a, fighter_b
    )

    # 3. Motivation signals
    findings.motivation_flags = _check_motivation_signals(
        features, fighter_a, fighter_b
    )

    # 4. Blind spot matching
    findings.blind_spot_matches = _check_blind_spots(
        features, fighter_a, fighter_b, model_prob_a, market_prob_a
    )

    # 6. Correlated exposure
    findings.exposure_warning = _check_correlated_exposure(
        event_title, existing_bets or [], event_date=event_date,
    )

    return findings


# ---------------------------------------------------------------------------
# LLM Synthesis — the "brain"
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""\
You are an expert MMA fight analyst acting as a sanity check for a data-driven \
UFC betting model. Your job is to PASS or BLOCK bets — nothing else. You do not \
control sizing.

TODAY'S DATE: {today}. You are evaluating bets on UPCOMING fights that have not \
happened yet. If your web search shows a fight has already occurred, BLOCK it — \
the data is stale. Only evaluate fights that are scheduled in the future.

You have access to WEB SEARCH. USE IT. For every fight, you MUST search for:
1. Both fighters' names to understand who they are, their records, and recent form
2. The specific matchup to find any news, injury reports, or expert analysis
3. Any recent developments (camp changes, weight class moves, personal issues)

Do not rely solely on your training data — it may be outdated. Search the web to \
get current information about both fighters before making your decision.

ABOUT THE MODEL: This is a backtested, calibrated XGBoost model that has shown \
a profitable edge over historical UFC data. It blends model probabilities with \
market odds and applies multiple statistical filters before generating a bet. \
It is good at what it does and you should TRUST it on close fights between \
evenly matched fighters. That's where its statistical edge works best.

However, the model has blind spots:
- It only sees rolling statistical averages. It doesn't know WHO fighters are, \
their reputation, their championship pedigree, or their career trajectory.
- It can't assess competition quality. A 10-0 record in regional shows looks the \
same as 10-0 in the UFC to the model.
- It doesn't understand stylistic context beyond raw numbers. A wrestler with 4.0 \
TD/fight who has never faced an elite anti-wrestler looks the same as one who has.
- It can miss regime changes: new gyms, new weight classes, long layoffs, aging \
fighters who have visibly declined but whose rolling stats haven't caught up yet.

YOUR ROLE: You receive the model's bet recommendation along with both fighters' \
names, statistical profiles, and weight class. Search the web and use your \
knowledge of MMA to answer:

1. WHO ARE THESE FIGHTERS? Search for both fighters. Find their REAL current \
record (W-L-D), their UFC ranking (if any), and their career trajectory. Is one \
fighter clearly a level above the other in ways stats don't capture (e.g., former \
champion, ranked contender vs unranked, elite prospect, known journeyman)?

2. VERIFY THE RECORDS. Search for each fighter's actual record on Sherdog, \
Tapology, or UFC.com. Report the verified record you found in your response. \
If the records suggest a massive experience or quality gap the model can't see, \
that's a reason to BLOCK.

3. DOES THIS BET MAKE SENSE? Would a knowledgeable MMA fan look at this bet \
and think it's reasonable, or would they immediately see something the model missed?

4. ARE THERE RED FLAGS the model can't see? Search for recent news — injuries, \
cancellations, weight miss history, visible decline, terrible stylistic matchup \
that stats understate, fighter known to quit when adversity hits, etc.

HANDLING UNKNOWN FIGHTERS: If you search for a fighter and find very little \
information (no Wikipedia page, no Sherdog profile, no notable results), that \
fighter is likely regional-level. If the model is betting on an unknown fighter \
against a well-known UFC veteran or ranked opponent, BLOCK. If both fighters are \
relatively unknown, PASS — the model's stats are the best information available.

WEIGHT CLASS CONTEXT: Interpret stats differently by weight class. Heavyweights \
have lower output and more KO finishes — 2.5 SLpM is normal. Lightweights and \
below have higher volume — 4+ SLpM is common. Don't flag low output in heavy \
divisions or high output in lighter divisions as unusual.

ABOUT THE STATS YOU RECEIVE: The statistical profiles below are VERIFIED rolling \
averages computed from UFCStats.com scraped data. They are correct. Do NOT claim \
the data is "corrupted", "placeholder", "impossible", or "fabricated" — it is \
real, computed data. If a stat looks unusual, consider that it may reflect the \
fighter's actual career performance rather than a data error. You are not a data \
quality auditor — focus on fighter assessment and bet evaluation.

DECISION RULES:
- Default to PASS. The model is backtested and profitable. Only block bets \
where you see something the model clearly cannot.
- BLOCK when: the model is betting on a clearly inferior fighter against a \
significantly better one, or when there's a major factor the stats can't capture \
(e.g., betting on a regional-level fighter against a former world champion).
- BLOCK when: your web research reveals something specific that makes this bet \
obviously wrong — not vague concerns, but concrete findings.
- DO NOT block bets just because the edge is small or because you're uncertain. \
Uncertainty is the model's job to price. Your job is catching obvious misreads.
- DO NOT override the model on close fights between evenly matched fighters. \
That's exactly where the model's statistical edge works best.
- DO NOT cite data quality issues as a reason to BLOCK. The stats are verified.

BEFORE DECIDING, you must read the statistical profiles provided and confirm \
the key numbers. Copy the exact stats from the profiles into your response — do \
NOT guess, round differently, or substitute numbers from web sources. The \
profiles below are what the MODEL uses. Your job is to evaluate the BET, not \
the data.

After completing your research, respond with ONLY a JSON object (no markdown fencing):
{{
    "stats_confirmed": {{
        "fighter_a_str_acc": <copy from profile, e.g. 43>,
        "fighter_a_td_acc": <copy from profile>,
        "fighter_a_td_def": <copy from profile>,
        "fighter_b_str_acc": <copy from profile>,
        "fighter_b_td_acc": <copy from profile>,
        "fighter_b_td_def": <copy from profile>
    }},
    "verified_records": {{
        "fighter_a": "<W-L-D record from web search, e.g. 12-1-0>",
        "fighter_b": "<W-L-D record from web search, e.g. 22-3-0>",
        "fighter_a_ranking": "<UFC ranking or 'unranked'>",
        "fighter_b_ranking": "<UFC ranking or 'unranked'>",
        "source": "<where you found this, e.g. Sherdog, UFC.com>"
    }},
    "verdict": "PASS" | "BLOCK",
    "rationale": "2-3 sentences explaining your reasoning, citing what you found",
    "fighter_assessment": "Brief assessment of both fighters based on your research",
    "risk_flags": ["flag1", "flag2"]
}}
"""


def _fval(features: dict, key: str, fmt: str = ".1f") -> str:
    """Format a feature value, returning 'N/A' for missing data."""
    val = features.get(key)
    if val is None or str(val) in ("", "nan", "None"):
        return "N/A"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return "N/A"


def _fpct(features: dict, key: str) -> str:
    """Format a feature value as a percentage string."""
    val = features.get(key)
    if val is None or str(val) in ("", "nan", "None"):
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if 0.0 <= val <= 1.0:
        val *= 100.0
    return f"{val:.0f}"


def _build_fighter_narrative(features: dict, prefix: str, name: str) -> str:
    """Build a plain-English summary of what the model sees for one fighter."""
    parts = []

    # Experience
    num_fights = _fval(features, f"{prefix}_num_fights", ".0f")
    if num_fights != "N/A":
        parts.append(f"{num_fights} UFC fights on record")

    # Streaks — only mention if notable
    w_streak = features.get(f"{prefix}_current_win_streak")
    l_streak = features.get(f"{prefix}_lose_streak")
    try:
        w = int(float(w_streak)) if w_streak is not None else 0
    except (TypeError, ValueError):
        w = 0
    try:
        l = int(float(l_streak)) if l_streak is not None else 0
    except (TypeError, ValueError):
        l = 0
    if w >= 3:
        parts.append(f"on a {w}-fight win streak")
    elif l >= 2:
        parts.append(f"on a {l}-fight losing streak")

    # Striking — characterise the style, not the numbers
    slpm = _fval(features, f"{prefix}_roll_slpm")
    str_acc = _fpct(features, f"{prefix}_roll_str_acc")
    if slpm != "N/A":
        try:
            slpm_f = float(slpm)
            vol = "high-volume" if slpm_f >= 5.0 else "moderate-volume" if slpm_f >= 3.0 else "low-volume"
            parts.append(f"{vol} striker ({slpm} SLpM, {str_acc}% acc)")
        except (TypeError, ValueError):
            pass

    # Grappling — characterise the threat level
    td_avg = _fval(features, f"{prefix}_roll_td_avg")
    td_acc = _fpct(features, f"{prefix}_roll_td_acc")
    td_def = _fpct(features, f"{prefix}_roll_td_def")
    if td_avg != "N/A":
        try:
            td_f = float(td_avg)
            if td_f >= 3.0:
                parts.append(f"active wrestler ({td_avg} TD/fight, {td_acc}% acc)")
            elif td_f >= 1.0:
                parts.append(f"moderate grappling threat ({td_avg} TD/fight)")
            else:
                parts.append(f"primarily a striker on the feet ({td_avg} TD/fight)")
        except (TypeError, ValueError):
            pass
    if td_def != "N/A":
        try:
            td_def_f = float(td_def)
            if td_def_f < 55:
                parts.append(f"vulnerable to takedowns ({td_def}% TDD)")
            elif td_def_f >= 85:
                parts.append(f"strong takedown defense ({td_def}% TDD)")
        except (TypeError, ValueError):
            pass

    # Age — only if notable
    age = _fval(features, f"{prefix}_age", ".0f")
    if age != "N/A":
        try:
            age_f = float(age)
            if age_f >= 37:
                parts.append(f"age {age} (potential decline factor)")
            elif age_f <= 24:
                parts.append(f"age {age} (young, still developing)")
        except (TypeError, ValueError):
            pass

    # Layoff
    days = features.get(f"{prefix}_days_since_last_fight")
    if days is not None:
        try:
            d = int(float(days))
            if d > 365:
                parts.append(f"hasn't fought in {d} days ({d / 365:.1f} years)")
        except (TypeError, ValueError):
            pass

    if not parts:
        return f"**{name}:** Limited data available."
    return f"**{name}:** " + ". ".join(p.capitalize() if i == 0 else p for i, p in enumerate(parts)) + "."


def _build_model_narrative(features: dict, fighter_a: str, fighter_b: str) -> str:
    """Build a combined narrative of both fighters from the model's perspective."""
    lines = [
        _build_fighter_narrative(features, "a", fighter_a),
        _build_fighter_narrative(features, "b", fighter_b),
    ]

    # Key matchup note — only flag clear mismatches
    a_td_avg = features.get("a_roll_td_avg")
    b_td_def = features.get("b_roll_td_def")
    b_td_avg = features.get("b_roll_td_avg")
    a_td_def = features.get("a_roll_td_def")
    try:
        if (a_td_avg and b_td_def and
                float(a_td_avg) > 3.0 and float(b_td_def) < 55):
            lines.append(
                f"**Key mismatch:** The model sees {fighter_a} as an active "
                f"wrestler against {fighter_b}'s weak takedown defense."
            )
        elif (b_td_avg and a_td_def and
                float(b_td_avg) > 3.0 and float(a_td_def) < 55):
            lines.append(
                f"**Key mismatch:** The model sees {fighter_b} as an active "
                f"wrestler against {fighter_a}'s weak takedown defense."
            )
    except (TypeError, ValueError):
        pass

    return "\n\n".join(lines)


def _build_stat_reference(features: dict, fighter_a: str, fighter_b: str) -> str:
    """Compact stat block for the stats_confirmed echo check."""
    lines = []
    for prefix, name in [("a", fighter_a), ("b", fighter_b)]:
        str_acc = _fpct(features, f"{prefix}_roll_str_acc")
        td_acc = _fpct(features, f"{prefix}_roll_td_acc")
        td_def = _fpct(features, f"{prefix}_roll_td_def")
        lines.append(f"{name}: str_acc={str_acc}%, td_acc={td_acc}%, td_def={td_def}%")
    return "\n".join(lines)


def _build_synthesis_prompt(
    *,
    fighter_a: str,
    fighter_b: str,
    bet_on: str,
    bet_side: str,
    model_prob: float,
    market_prob: float,
    blended_prob: float,
    edge: float,
    features: dict,
    findings: ResearchFindings,
    weight_class: str = "",
) -> str:
    """Build the user prompt for Gemini synthesis."""
    sections = []

    wc_label = f" ({weight_class})" if weight_class else ""
    sections.append(f"## Fight: {fighter_a} vs {fighter_b}{wc_label}")
    sections.append(
        f"The model wants to bet on **{bet_on}**.\n"
        f"- Model probability: {model_prob:.1%}\n"
        f"- Market probability: {market_prob:.1%}\n"
        f"- Blended probability: {blended_prob:.1%}\n"
        f"- Edge: {edge:.1%}"
    )

    # -- Model's view: narrative summary of what the model "sees" --
    sections.append("## What the Model Sees")
    sections.append(
        "The stats below are the rolling averages the model used to make its "
        "prediction. Your job is NOT to re-analyze these numbers — the model "
        "already did that. Your job is to check whether the model's picture of "
        "these fighters matches REALITY based on your web research."
    )
    sections.append(_build_model_narrative(features, fighter_a, fighter_b))

    # Compact stat reference (for stats_confirmed echo — do not remove)
    sections.append("## Stat Reference (for confirmation)")
    sections.append(_build_stat_reference(features, fighter_a, fighter_b))

    # Recency flags
    if findings.recency_flags:
        sections.append("## Context Flags")
        for flag in findings.recency_flags:
            sections.append(f"- {flag}")

    # Motivation signals
    if findings.motivation_flags:
        sections.append("## Motivation Signals")
        for flag in findings.motivation_flags:
            sections.append(f"- {flag}")

    # Blind spots
    if findings.blind_spot_matches:
        sections.append("## Known Model Blind Spots Matched")
        for match in findings.blind_spot_matches:
            sections.append(f"- {match}")

    # Exposure
    if findings.exposure_warning:
        sections.append("## Exposure Warning")
        sections.append(findings.exposure_warning)

    sections.append(
        "\n## Your Task\n"
        f"Using your knowledge of MMA: who are {fighter_a} and {fighter_b}? "
        f"Does betting on {bet_on} make sense, or is the model missing something obvious? "
        f"PASS or BLOCK."
    )

    return "\n\n".join(sections)


def _call_llm_synthesis(prompt: str) -> dict:
    """Dispatch fight research to Gemini (with Google Search grounding).

    Returns passthrough PASS if Gemini is unavailable or not configured.
    """
    if GEMINI_API_KEY:
        result = _call_gemini_synthesis(prompt)
        if result is not None:
            return result
        logger.warning("Gemini call failed after retries — passthrough PASS")

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured — operator passthrough")

    return {
        "verdict": "PASS",
        "rationale": "Operator passthrough: Gemini unavailable",
        "fighter_assessment": "",
        "risk_flags": ["llm_unavailable"],
    }


def _call_gemini_synthesis(prompt: str, *, _max_retries: int = 4) -> dict | None:
    """Call Gemini with Google Search grounding. Returns None on failure."""
    text = ""
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        last_exc = None
        for attempt in range(_max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "system_instruction": _build_system_prompt(),
                        "tools": [{"google_search": {}}],
                        "temperature": 0.3,
                    },
                )
                break  # success
            except Exception as exc:
                last_exc = exc
                # Retry on 503 / overload; bail on anything else
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    wait = [1, 2, 4, 10][attempt]  # 1s, 2s, 4s, 10s
                    logger.warning(
                        "Gemini 503 (attempt %d/%d) — retrying in %ds",
                        attempt + 1, _max_retries, wait,
                    )
                    import time
                    time.sleep(wait)
                else:
                    raise
        else:
            # All retries exhausted
            raise last_exc  # type: ignore[misc]

        text = response.text.strip()

        # Log grounding sources for audit
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            gm = getattr(candidate, "grounding_metadata", None)
            if gm and hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                sources = []
                for chunk in gm.grounding_chunks[:10]:
                    if hasattr(chunk, "web") and chunk.web:
                        sources.append(f"{chunk.web.title}: {chunk.web.uri}")
                if sources:
                    logger.info(
                        "Gemini used %d web sources: %s",
                        len(sources),
                        "; ".join(sources[:3]) + ("..." if len(sources) > 3 else ""),
                    )

        # Parse JSON from response
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Gemini with google_search grounding sometimes returns narrative
        # text with JSON embedded. Try to extract it.
        match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise json.JSONDecodeError("No JSON object found in response", text, 0)

    except ImportError:
        logger.warning("google-genai package not installed")
        return None
    except json.JSONDecodeError as exc:
        raw_preview = text[:300] if text else "(empty)"
        logger.warning("Failed to parse Gemini response as JSON: %s — raw: %s", exc, raw_preview)
        return None  # fall back to passthrough PASS
    except Exception as exc:
        logger.warning("Gemini API error: %s", exc)
        return None


_STATS_CONFIRMED_MAP = {
    "fighter_a_str_acc": "a_roll_str_acc",
    "fighter_a_td_acc": "a_roll_td_acc",
    "fighter_a_td_def": "a_roll_td_def",
    "fighter_b_str_acc": "b_roll_str_acc",
    "fighter_b_td_acc": "b_roll_td_acc",
    "fighter_b_td_def": "b_roll_td_def",
}

# Maximum allowed absolute difference between the LLM's echoed stat and the
# real feature value.  Accounts for rounding (profile shows "43%" for 43.15).
_STATS_TOLERANCE = 3.0


def _check_stats_confirmed(synthesis: dict, features: dict) -> list[str]:
    """Compare LLM-echoed stats against real features.

    Returns a list of mismatch descriptions.  Empty list means the LLM read
    the data correctly (or didn't return the field at all).
    """
    confirmed = synthesis.get("stats_confirmed")
    if not isinstance(confirmed, dict):
        return []
    mismatches = []
    for llm_key, feature_key in _STATS_CONFIRMED_MAP.items():
        llm_val = confirmed.get(llm_key)
        real_val = features.get(feature_key)
        if llm_val is None or real_val is None:
            continue
        try:
            llm_val = float(llm_val)
            real_val = float(real_val)
        except (TypeError, ValueError):
            continue
        if 0.0 <= llm_val <= 1.0 and real_val > 1.0:
            llm_val *= 100.0
        if abs(llm_val - real_val) > _STATS_TOLERANCE:
            mismatches.append(
                f"{llm_key}: LLM said {llm_val:.1f}, actual {real_val:.1f}"
            )
    return mismatches


def _build_correction_prompt(
    original_prompt: str,
    original_rationale: str,
    mismatches: list[str],
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> str:
    """Build a follow-up prompt that corrects the LLM with the real stats."""
    # Build a verified stats block from the real features.
    def _fmt(key: str) -> str:
        val = features.get(key)
        if val is None or str(val) in ("", "nan", "None"):
            return "N/A"
        try:
            v = float(val)
            return f"{v:.1f}%"
        except (TypeError, ValueError):
            return "N/A"

    stats_block = (
        f"## CORRECTION — Your previous response used WRONG stats\n\n"
        f"Your previous answer contained these errors:\n"
        + "\n".join(f"- {m}" for m in mismatches)
        + f"\n\nHere are the VERIFIED stats from the model's feature pipeline. "
        f"These are the real numbers. Use ONLY these:\n\n"
        f"**{fighter_a}:**\n"
        f"- Striking accuracy: {_fmt('a_roll_str_acc')}\n"
        f"- Takedown accuracy: {_fmt('a_roll_td_acc')}\n"
        f"- Takedown defense: {_fmt('a_roll_td_def')}\n"
        f"- SLpM: {_fmt('a_roll_slpm')}\n"
        f"- TD avg/fight: {_fmt('a_roll_td_avg')}\n\n"
        f"**{fighter_b}:**\n"
        f"- Striking accuracy: {_fmt('b_roll_str_acc')}\n"
        f"- Takedown accuracy: {_fmt('b_roll_td_acc')}\n"
        f"- Takedown defense: {_fmt('b_roll_td_def')}\n"
        f"- SLpM: {_fmt('b_roll_slpm')}\n"
        f"- TD avg/fight: {_fmt('b_roll_td_avg')}\n\n"
        f"Now re-evaluate this bet using the CORRECT stats above combined with "
        f"your web research. Your previous verdict was based on wrong data so "
        f"start fresh. Respond with the same JSON format."
    )

    return f"{original_prompt}\n\n{stats_block}"


def _guard_data_hallucination(
    synthesis: dict,
    features: dict,
    fighter_a: str,
    fighter_b: str,
    *,
    original_prompt: str = "",
    _retry: bool = False,
) -> dict:
    """Validate the LLM read real stats.  If it didn't, retry with corrections.

    If the LLM's echoed stats don't match reality, we re-call the LLM with the
    correct stats explicitly injected so it can make a properly informed
    decision.  We never auto-pass — the fight may genuinely deserve a BLOCK.
    """
    mismatches = _check_stats_confirmed(synthesis, features)

    if not mismatches:
        # Stats match (or weren't returned) — accept the verdict as-is.
        return synthesis

    if _retry:
        # Already retried once — accept whatever the retry returned but annotate.
        logger.warning(
            "Operator for %s vs %s still misread stats on retry — accepting "
            "verdict with annotation: %s",
            fighter_a,
            fighter_b,
            "; ".join(mismatches),
        )
        synthesis = dict(synthesis)
        synthesis.setdefault("risk_flags", []).append("stats_mismatch_after_retry")
        return synthesis

    # First attempt had wrong stats — retry with explicit corrections.
    detail_str = "; ".join(mismatches)
    logger.warning(
        "Operator for %s vs %s echoed wrong stats (%s) — retrying with "
        "corrected data",
        fighter_a,
        fighter_b,
        detail_str,
    )

    correction_prompt = _build_correction_prompt(
        original_prompt=original_prompt,
        original_rationale=synthesis.get("rationale", ""),
        mismatches=mismatches,
        features=features,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
    )
    retry_synthesis = _call_llm_synthesis(correction_prompt)

    # Validate the retry too (but don't recurse further).
    retry_synthesis = _guard_data_hallucination(
        retry_synthesis,
        features,
        fighter_a,
        fighter_b,
        original_prompt=original_prompt,
        _retry=True,
    )
    retry_synthesis = dict(retry_synthesis)
    retry_synthesis.setdefault("risk_flags", []).append("stats_corrected_retry")
    return retry_synthesis




# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------

def _log_decision(decision: OperatorDecision) -> None:
    """Append decision to the persistent audit log (JSONL — one record per line)."""
    try:
        with open(DECISION_LOG_PATH, "a") as f:
            f.write(json.dumps(asdict(decision), default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to log operator decision: %s", exc)


def load_decision_log() -> list[dict]:
    """Read all operator decisions from the JSONL audit log."""
    if not DECISION_LOG_PATH.exists():
        return []
    decisions = []
    with open(DECISION_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return decisions


# ---------------------------------------------------------------------------
# Public API — evaluate a single bet
# ---------------------------------------------------------------------------

def evaluate_bet(
    *,
    fighter_a: str,
    fighter_b: str,
    bet_on: str,
    bet_side: str,
    model_prob: float,
    blended_prob: float,
    market_prob: float,
    edge: float,
    features: dict,
    provenance: dict | None = None,
    weight_class: str = "",
    event_title: str = "",
    event_date: str = "",
    existing_bets: list[dict] | None = None,
) -> OperatorDecision:
    """
    Run the full operator pipeline for a single bet candidate.

    Returns an OperatorDecision with the verdict and rationale.
    The operator NEVER crashes the trading loop — any unhandled error
    results in a PASS (let the model's bet through).
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if _has_existing_bet_for_fight(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        existing_bets=existing_bets,
        event_date=event_date,
    ):
        logger.info(
            "Operator skip for %s vs %s — fight already has a recorded bet/order",
            fighter_a,
            fighter_b,
        )
        return OperatorDecision(
            verdict="PASS",
            confidence=1.0,
            model_prob=model_prob,
            operator_prob=model_prob,
            rationale="Operator skipped: fight already has a recorded bet/order",
            research_summary={},
            risk_flags=["existing_bet"],
            timestamp=timestamp,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            bet_on=bet_on,
            bet_side=bet_side,
            edge=edge,
            market_prob=market_prob,
            provenance=dict(provenance or {}),
        )

    # Check session cache — avoid re-evaluating the same fight.
    # Per-key lock prevents concurrent threads from both missing the
    # cache for the same fight and making duplicate LLM calls.
    cache_key = _fight_cache_key(fighter_a, fighter_b)

    # Acquire a per-key lock so only one thread evaluates a given fight.
    with _decision_cache_lock:
        if cache_key not in _decision_inflight:
            _decision_inflight[cache_key] = threading.Lock()
        key_lock = _decision_inflight[cache_key]

    with key_lock:
        # Re-check cache under the per-key lock (another thread may have
        # populated it while we were waiting).
        with _decision_cache_lock:
            if cache_key in _decision_cache:
                cached, cached_at = _decision_cache[cache_key]
                age = time.time() - cached_at
                if age < CACHE_TTL_SECONDS:
                    logger.info(
                        "Operator cache hit for %s vs %s — reusing %s verdict "
                        "(age %.0fm, saved an API call)",
                        fighter_a, fighter_b, cached.verdict, age / 60,
                    )
                    return cached
                else:
                    logger.info(
                        "Operator cache expired for %s vs %s (age %.1fh) — re-evaluating",
                        fighter_a, fighter_b, age / 3600,
                    )
                    del _decision_cache[cache_key]

        try:
            # Run research
            findings = run_research_pipeline(
                features=features,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                model_prob_a=model_prob if bet_side == "a" else 1 - model_prob,
                market_prob_a=market_prob if bet_side == "a" else 1 - market_prob,
                event_title=event_title,
                event_date=event_date,
                existing_bets=existing_bets,
            )

            # Build synthesis prompt and call LLM
            prompt = _build_synthesis_prompt(
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                bet_on=bet_on,
                bet_side=bet_side,
                model_prob=model_prob,
                market_prob=market_prob,
                blended_prob=blended_prob,
                edge=edge,
                features=features,
                findings=findings,
                weight_class=weight_class,
            )

            synthesis = _call_llm_synthesis(prompt)

            # Guard: if the LLM misread the stats, retry with corrections
            # so it can make a properly informed decision.
            synthesis = _guard_data_hallucination(
                synthesis, features, fighter_a, fighter_b,
                original_prompt=prompt,
            )

            # Build decision — PASS/BLOCK only
            verdict = synthesis.get("verdict", "PASS").upper()
            if verdict not in ("PASS", "BLOCK"):
                logger.warning("Invalid verdict %r from operator — defaulting to PASS", verdict)
                verdict = "PASS"

            research_summary = asdict(findings) if findings else {}
            if synthesis.get("verified_records"):
                research_summary["verified_records"] = synthesis["verified_records"]
            if synthesis.get("fighter_assessment"):
                research_summary["fighter_assessment"] = synthesis["fighter_assessment"]

            decision = OperatorDecision(
                verdict=verdict,
                confidence=1.0,
                model_prob=model_prob,
                operator_prob=model_prob,
                rationale=synthesis.get("rationale", "No rationale provided"),
                research_summary=research_summary,
                risk_flags=synthesis.get("risk_flags", []),
                timestamp=timestamp,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                bet_on=bet_on,
                bet_side=bet_side,
                edge=edge,
                market_prob=market_prob,
                provenance=dict(provenance or {}),
            )

        except Exception as exc:
            # Operator must NEVER crash the trading loop
            logger.error(
                "Operator pipeline error for %s vs %s (defaulting to PASS): %s",
                fighter_a, fighter_b, exc,
            )
            decision = OperatorDecision(
                verdict="PASS",
                confidence=1.0,
                model_prob=model_prob,
                operator_prob=model_prob,
                rationale=f"Operator error (defaulting to PASS): {exc}",
                research_summary={},
                risk_flags=["operator_error"],
                timestamp=timestamp,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                bet_on=bet_on,
                bet_side=bet_side,
                edge=edge,
                market_prob=market_prob,
                provenance=dict(provenance or {}),
            )

        # Always log
        _log_decision(decision)

        # Cache the decision for this session
        with _decision_cache_lock:
            _decision_cache[cache_key] = (decision, time.time())
        _save_decision_cache_to_disk()

    logger.info(
        "Operator verdict for %s: %s (flags: %s, bundle=%s, model_spec=%s, processed=%s, sources=%s/%s)",
        bet_on,
        decision.verdict,
        ", ".join(decision.risk_flags) if decision.risk_flags else "none",
        decision.provenance.get("bundle_id", "n/a"),
        decision.provenance.get("model_spec_name", "n/a"),
        decision.provenance.get("processed_snapshot_max_event_date", "n/a"),
        decision.provenance.get("fighter_a_source", "n/a"),
        decision.provenance.get("fighter_b_source", "n/a"),
    )

    return decision


# ---------------------------------------------------------------------------
# Batch evaluation — process a DataFrame of bet candidates
# ---------------------------------------------------------------------------

def evaluate_bets(
    bets: pd.DataFrame,
    *,
    features_by_fight: dict[str, dict] | None = None,
    provenance_by_fight: dict[str, dict] | None = None,
    event_title: str = "",
    existing_bets: list[dict] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str = "bets",
) -> pd.DataFrame:
    """
    Evaluate a DataFrame of bet candidates through the operator.

    Args:
        bets: DataFrame from find_value_bets or find_conviction_bets
        features_by_fight: mapping of "fighterA|fighterB" → feature dict
        provenance_by_fight: mapping of "fighterA|fighterB" → runtime/source metadata
        event_title: current event name for exposure checks
        existing_bets: list of already-placed bets for exposure check

    Returns:
        Filtered DataFrame with only PASS bets, plus operator columns added.
        Sizing is unchanged — the operator only gates, never adjusts size.
    """
    if bets.empty:
        return bets

    if not OPERATOR_ENABLED:
        logger.debug("LLM Operator is disabled — passing all bets through")
        bets = bets.copy()
        bets["operator_verdict"] = "PASS"
        bets["operator_rationale"] = "Operator disabled"
        return bets

    features_by_fight = features_by_fight or {}
    provenance_by_fight = provenance_by_fight or {}
    decisions = []
    approved_rows = []

    def _report_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(message)
        except Exception as exc:
            logger.debug("Operator progress callback failed: %s", exc)

    total_bets = len(bets)
    for position, (_, bet) in enumerate(bets.iterrows(), start=1):
        fighter_a = bet.get("fighter_a", "")
        fighter_b = bet.get("fighter_b", "")
        fight_key = f"{fighter_a}|{fighter_b}"
        _report_progress(
            f"Cycle active: operator evaluating {progress_label} {position}/{total_bets}: {fighter_a} vs {fighter_b}"
        )

        features = features_by_fight.get(fight_key, {})
        provenance = provenance_by_fight.get(fight_key, {})

        decision = evaluate_bet(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            bet_on=bet.get("bet_on", ""),
            bet_side=bet.get("bet_side", ""),
            model_prob=float(bet.get("model_prob", 0.5)),
            blended_prob=float(bet.get("blended_prob", 0.5)),
            market_prob=float(bet.get("market_prob", 0.5)),
            edge=float(bet.get("edge", 0.0)),
            features=features,
            provenance=provenance,
            weight_class=str(bet.get("weight_class", "")),
            event_title=event_title,
            event_date=str(bet.get("event_date", "")),
            existing_bets=existing_bets,
        )

        decisions.append(decision)

        if decision.verdict == "BLOCK":
            logger.info(
                "Operator BLOCKED bet on %s: %s",
                bet.get("bet_on", "?"),
                decision.rationale[:100],
            )
            if OPERATOR_MODE == "gate":
                continue

        row = bet.copy()
        row["operator_verdict"] = decision.verdict
        row["operator_rationale"] = decision.rationale
        row["operator_risk_flags"] = ", ".join(decision.risk_flags)

        approved_rows.append(row)

    if not approved_rows:
        cols = list(bets.columns) + ["operator_verdict", "operator_rationale", "operator_risk_flags"]
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(approved_rows)
    blocked = sum(1 for d in decisions if d.verdict == "BLOCK")
    logger.info(
        "Operator: %d/%d bets passed, %d blocked",
        len(approved_rows),
        len(bets),
        blocked,
    )
    return result
