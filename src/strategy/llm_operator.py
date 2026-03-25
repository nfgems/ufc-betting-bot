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
from typing import Literal, Optional

import pandas as pd

from src.config import DATA_DIR, LOGS_DIR

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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
) -> str:
    """Flag concentration risk when multiple bets are on the same event.

    Matches on event_title, event_date, or event_id — whichever is available.
    """
    if not existing_bets:
        return ""

    same_event_bets = [
        b for b in existing_bets
        if (event_title and (
            b.get("event_title", "") == event_title
            or b.get("event_date", "") == event_title
            or b.get("event_id", "") == event_title
        ))
    ]

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
        event_title, existing_bets or []
    )

    return findings


# ---------------------------------------------------------------------------
# Claude API Synthesis — the "brain"
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

1. WHO ARE THESE FIGHTERS? Search for both fighters. What is their actual \
reputation, skill level, and career trajectory? Is one fighter clearly a level \
above the other in ways stats don't capture (e.g., former champion from another \
org, elite prospect, known journeyman)?

2. DOES THIS BET MAKE SENSE? Would a knowledgeable MMA fan look at this bet \
and think it's reasonable, or would they immediately see something the model missed?

3. ARE THERE RED FLAGS the model can't see? Search for recent news — injuries, \
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

After completing your research, respond with ONLY a JSON object (no markdown fencing):
{{
    "verdict": "PASS" | "BLOCK",
    "rationale": "2-3 sentences explaining your reasoning, citing what you found",
    "fighter_assessment": "Brief assessment of both fighters based on your research",
    "risk_flags": ["flag1", "flag2"]
}}
"""


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
    """Build the user prompt for Claude/Gemini synthesis."""
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

    # Fighter stat profiles for Claude to reason about
    sections.append("## Fighter Statistical Profiles")

    def _fighter_profile(prefix, name):
        lines = [f"**{name}:**"]
        def _g(key, fmt=".1f"):
            val = features.get(key)
            if val is None or str(val) in ("", "nan", "None"):
                return "N/A"
            try:
                return f"{float(val):{fmt}}"
            except (TypeError, ValueError):
                return "N/A"
        def _g_pct(key):
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
        num_fights = _g(f"{prefix}_num_fights", ".0f")
        lines.append(f"- UFC fights: {num_fights}")

        win_streak = _g(f"{prefix}_current_win_streak", ".0f")
        lose_streak = _g(f"{prefix}_lose_streak", ".0f")
        lines.append(f"- Win streak: {win_streak} | Lose streak: {lose_streak}")

        slpm = _g(f"{prefix}_roll_slpm")
        str_acc = _g_pct(f"{prefix}_roll_str_acc")
        lines.append(f"- Striking: {slpm} SLpM, {str_acc}% accuracy")

        td_avg = _g(f"{prefix}_roll_td_avg")
        td_acc = _g_pct(f"{prefix}_roll_td_acc")
        td_def = _g_pct(f"{prefix}_roll_td_def")
        lines.append(f"- Takedowns: {td_avg}/fight, {td_acc}% acc, {td_def}% def")

        age = _g(f"{prefix}_age", ".0f")
        if age != "N/A":
            lines.append(f"- Age: {age}")

        days = features.get(f"{prefix}_days_since_last_fight")
        if days is not None:
            try:
                d = int(float(days))
                lines.append(f"- Days since last fight: {d}")
            except (TypeError, ValueError):
                pass

        # Pre-UFC context
        org_tier = features.get(f"{prefix}_pre_ufc_org_tier")
        if org_tier is not None:
            lines.append(f"- Pre-UFC org tier: {org_tier}")

        ko_rate = features.get(f"{prefix}_ko_rate")
        sub_rate = features.get(f"{prefix}_sub_rate")
        if ko_rate is not None and sub_rate is not None:
            try:
                ko_pct = float(ko_rate) * 100.0 if 0.0 <= float(ko_rate) <= 1.0 else float(ko_rate)
                sub_pct = float(sub_rate) * 100.0 if 0.0 <= float(sub_rate) <= 1.0 else float(sub_rate)
                lines.append(
                    f"- Finish rates: KO {ko_pct:.0f}%, Sub {sub_pct:.0f}%"
                )
            except (TypeError, ValueError):
                pass

        return "\n".join(lines)

    sections.append(_fighter_profile("a", fighter_a))
    sections.append(_fighter_profile("b", fighter_b))

    # Recency flags
    if findings.recency_flags:
        sections.append("## Context Flags")
        for flag in findings.recency_flags:
            sections.append(f"- {flag}")

    # Matchup analysis from features
    if findings.matchup_analysis and "Insufficient" not in findings.matchup_analysis:
        sections.append("## Statistical Matchup Notes")
        sections.append(findings.matchup_analysis)

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
    """Dispatch fight research to an LLM for PASS/BLOCK decision.

    Tries Gemini first (free, with Google Search grounding), falls back to
    Claude Sonnet if Gemini fails, and passthrough if neither is configured.
    """
    # Try Gemini first (free tier with Google Search grounding)
    if GEMINI_API_KEY:
        result = _call_gemini_synthesis(prompt)
        if result is not None:
            return result
        logger.warning("Gemini call failed — falling back to Claude")

    # Fall back to Claude if Gemini fails or isn't configured
    if ANTHROPIC_API_KEY:
        return _call_anthropic_synthesis(prompt)

    logger.warning("No LLM API key configured — operator falling back to passthrough")
    return {
        "verdict": "PASS",
        "rationale": "Operator passthrough: no API keys configured (GEMINI_API_KEY or ANTHROPIC_API_KEY)",
        "fighter_assessment": "",
        "risk_flags": ["no_api_key"],
    }


def _call_gemini_synthesis(prompt: str, *, _max_retries: int = 3) -> dict | None:
    """Call Gemini with Google Search grounding. Returns None on failure."""
    text = ""
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        last_exc = None
        for attempt in range(_max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
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
                    wait = 2 ** attempt  # 1s, 2s, 4s
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
        return None  # fall back to Claude
    except Exception as exc:
        logger.warning("Gemini API error: %s", exc)
        return None


def _call_anthropic_synthesis(prompt: str) -> dict:
    """Fallback: Call Claude API for synthesis (no web search without extra cost)."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Claude fallback has no web search tools — prepend an override so
        # the model doesn't emit <search> tags instead of returning JSON.
        no_search_note = (
            "IMPORTANT: You do NOT have web search tools in this mode. "
            "Ignore any instructions about searching the web. Use only your "
            "training knowledge to assess both fighters. If you lack information "
            "on a fighter, note that in your assessment and lean toward PASS. "
            "You MUST respond with ONLY a valid JSON object.\n\n"
        )
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2048,
            system=no_search_note + _build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()

        return json.loads(text)

    except ImportError:
        logger.warning("anthropic package not installed — operator passthrough")
        return {
            "verdict": "PASS",
            "rationale": "Operator passthrough: anthropic package not installed",
            "fighter_assessment": "",
            "risk_flags": ["no_sdk"],
        }
    except json.JSONDecodeError as exc:
        raw_preview = text[:500] if text else "(empty)"
        logger.warning("Failed to parse Claude response as JSON: %s — raw: %s", exc, raw_preview)
        return {
            "verdict": "PASS",
            "rationale": f"Claude parse error (defaulting to PASS): {exc}",
            "fighter_assessment": "",
            "risk_flags": ["parse_error"],
        }
    except Exception as exc:
        logger.warning("Claude API error: %s", exc)
        return {
            "verdict": "PASS",
            "rationale": f"Claude API error (defaulting to PASS): {exc}",
            "fighter_assessment": "",
            "risk_flags": ["api_error"],
        }


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
    existing_bets: list[dict] | None = None,
) -> OperatorDecision:
    """
    Run the full operator pipeline for a single bet candidate.

    Returns an OperatorDecision with the verdict and rationale.
    The operator NEVER crashes the trading loop — any unhandled error
    results in a PASS (let the model's bet through).
    """
    timestamp = datetime.now(timezone.utc).isoformat()

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

            # Build decision — PASS/BLOCK only
            verdict = synthesis.get("verdict", "PASS").upper()
            if verdict not in ("PASS", "BLOCK"):
                logger.warning("Invalid verdict %r from operator — defaulting to PASS", verdict)
                verdict = "PASS"

            decision = OperatorDecision(
                verdict=verdict,
                confidence=1.0,
                model_prob=model_prob,
                operator_prob=model_prob,
                rationale=synthesis.get("rationale", "No rationale provided"),
                research_summary=asdict(findings) if findings else {},
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

    for idx, bet in bets.iterrows():
        fighter_a = bet.get("fighter_a", "")
        fighter_b = bet.get("fighter_b", "")
        fight_key = f"{fighter_a}|{fighter_b}"

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
