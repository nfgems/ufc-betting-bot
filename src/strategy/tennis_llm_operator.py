"""Gemini-based sanity-check operator for tennis live trade candidates."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

TENNIS_LLM_VETO_ENABLED = os.getenv("TENNIS_LLM_VETO_ENABLED", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
TENNIS_LLM_VETO_FAIL_CLOSED = os.getenv("TENNIS_LLM_VETO_FAIL_CLOSED", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
TENNIS_LLM_VETO_MODEL = os.getenv("TENNIS_LLM_VETO_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TENNIS_LLM_OPERATOR_DIR = DATA_DIR / "operator"
TENNIS_LLM_OPERATOR_DIR.mkdir(parents=True, exist_ok=True)
TENNIS_LLM_VETO_LOG_PATH = TENNIS_LLM_OPERATOR_DIR / "tennis_veto_log.jsonl"

# Session-level veto cache: match_key → (TennisVetoDecision, epoch)
# Prevents re-evaluating the same match across loop cycles within a session.
_veto_cache: dict[str, tuple["TennisVetoDecision", float]] = {}
VETO_CACHE_TTL_SECONDS = float(os.getenv("TENNIS_LLM_VETO_CACHE_TTL", str(4 * 3600)))


def _match_cache_key(fighter_a: str, fighter_b: str) -> str:
    """Canonical cache key for a match (order-independent)."""
    pair = sorted([fighter_a.strip().lower(), fighter_b.strip().lower()])
    return f"{pair[0]}|{pair[1]}"


def clear_veto_cache() -> None:
    """Clear the session veto cache."""
    _veto_cache.clear()
    logger.info("Tennis veto decision cache cleared")


TENNIS_VETO_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["AUTO_BLOCK", "AUTO_SKIP", "NO_VETO"],
        },
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "source_summary": {"type": "array", "items": {"type": "string"}},
        "coverage_quality": {
            "type": "string",
            "enum": ["sparse", "moderate", "rich"],
        },
        "contradiction_strength": {
            "type": "string",
            "enum": ["none", "weak", "moderate", "strong"],
        },
    },
    "required": [
        "verdict",
        "confidence",
        "rationale",
        "risk_flags",
        "source_summary",
        "coverage_quality",
        "contradiction_strength",
    ],
}


@dataclass
class TennisVetoDecision:
    """Final Gemini sanity-check decision for a single tennis trade candidate."""

    verdict: Literal["AUTO_BLOCK", "AUTO_SKIP", "NO_VETO"]
    confidence: float
    rationale: str
    risk_flags: list[str]
    timestamp: str
    coverage_quality: str = "unknown"
    contradiction_strength: str = "unknown"
    policy_adjustment: str = ""
    fighter_a: str = ""
    fighter_b: str = ""
    decision_fighter: str = ""
    tournament_name: str = ""
    tour: str = ""
    source_summary: list[str] = field(default_factory=list)
    policy_exempt: bool = False


def _split_reason_string(value: object) -> list[str]:
    if value is None:
        return []
    reasons: list[str] = []
    for token in str(value).split(","):
        normalized = token.strip()
        if normalized:
            reasons.append(normalized)
    return reasons


def _reason_string(values: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return ",".join(deduped)


def _format_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return f"{numeric:.4f}"


def _format_optional_probability(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric or not (0.0 < numeric < 1.0):
        return "n/a"
    return f"{numeric:.4f}"


def _normalize_coverage_quality(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"sparse", "moderate", "rich"}:
        return normalized
    if normalized in {"low", "limited", "minimal", "light", "thin"}:
        return "sparse"
    if normalized in {"medium", "medium-high", "medium high", "average"}:
        return "moderate"
    if normalized in {"high", "strong", "extensive", "full"}:
        return "rich"
    return "unknown"


def _normalize_contradiction_strength(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "weak", "moderate", "strong"}:
        return normalized
    if normalized in {"no", "neutral", "low", "minor", "small", "slight"}:
        return "weak"
    if normalized in {"medium", "medium-high", "medium high", "mixed", "material", "meaningful", "noticeable"}:
        return "moderate"
    if normalized in {"high", "severe", "substantial", "significant", "considerable", "clear"}:
        return "strong"
    return "unknown"


def _normalize_confidence(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapped = {
            "very low": 0.15,
            "low": 0.30,
            "medium": 0.50,
            "medium-low": 0.40,
            "medium low": 0.40,
            "medium-high": 0.70,
            "medium high": 0.70,
            "high": 0.85,
            "very high": 0.95,
        }.get(normalized)
        if mapped is not None:
            return mapped
    try:
        confidence = float(value or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return min(max(confidence, 0.0), 1.0)


def _build_system_prompt() -> str:
    return (
        "You are a conservative but tennis-aware live-trading sanity check. "
        "Do not search for or reason about withdrawals, injuries, illnesses, retirements, or walkover risk. Ignore those topics entirely. "
        "Do not override the structured rules or invent certainty. "
        "Your job is to sanity-check whether an automated pre-match bet still makes sense in the real tennis world. "
        "Focus on things like breakout trajectory, phenom reputation, recent visible level, surface fit, age/experience context, "
        "elite pedigree, whether the market/model framing seems sensible, wrong opponent or wrong tournament context, "
        "match already started/completed, or clearly stale/inconsistent market context. "
        "Web search is available and should be used. "
        "Lack of public information is neutral, not negative. "
        "Do not penalize a player for being obscure or lightly covered. "
        "The numeric rank, rank-points, and recent-form values in the input packet are model-side snapshots and may lag public live updates. "
        "Do not issue AUTO_BLOCK or AUTO_SKIP solely because public websites show fresher rankings, rank points, or recent records than the packet. "
        "If that is your only concern, return NO_VETO and include the risk flag packet_snapshot_lag_only. "
        "Positive breakout or phenom context can support NO_VETO, but never replace the model or structured rules. "
        "Any AUTO_BLOCK or AUTO_SKIP verdict must include at least one concise risk flag describing the concrete contradiction. "
        "Return JSON only with keys: verdict, confidence, rationale, risk_flags, source_summary, coverage_quality, contradiction_strength. "
        "Allowed verdicts: AUTO_BLOCK, AUTO_SKIP, NO_VETO. "
        "Use AUTO_BLOCK only for strong real-world reasons the bet should not stand. "
        "Use AUTO_SKIP when the sanity check is materially unresolved, mixed, or contextually shaky. "
        "Use NO_VETO when the bet passes a reasonable real-world tennis sanity check, including cases where a young or lightly-sampled player still looks legitimately strong."
    )


def _build_player_context(row: dict[str, object], side: str) -> str:
    label = "A" if side == "a" else "B"
    name = str(row.get(f"fighter_{side}") or "").strip()
    age = row.get(f"{side}_age", "")
    rank = row.get(f"{side}_rank", "")
    rank_points = row.get(f"{side}_rank_points", "")
    num_matches = row.get(f"{side}_num_matches", "")
    recent_form = _format_float(row.get(f"{side}_recent_form", 0.0))
    surface_elo = _format_float(row.get(f"{side}_surface_elo", 0.0))
    elo = _format_float(row.get(f"{side}_elo", 0.0))
    return (
        f"Player {label}: {name} | age={age} | rank={rank} | rank_points={rank_points} | "
        f"prior_matches={num_matches} | recent_form={recent_form} | elo={elo} | surface_elo={surface_elo}"
    )


def _build_user_prompt(row: dict[str, object]) -> str:
    lines = [
        "Evaluate whether this automated tennis bet passes a real-world sanity check.",
        f"Player A: {row.get('fighter_a', '')}",
        f"Player B: {row.get('fighter_b', '')}",
        f"Proposed side: {row.get('decision_fighter', '')}",
        f"Tour: {row.get('tour', '')}",
        f"Tournament: {row.get('tournament_name', '')}",
        f"Commence time: {row.get('commence_time', '')}",
        f"Surface: {row.get('surface', '')}",
        _build_player_context(row, "a"),
        _build_player_context(row, "b"),
        f"Model probability on proposed side: {_format_float(row.get('decision_model_prob', 0.0))}",
        f"Reference market probability on proposed side: {_format_float(row.get('decision_market_prob', 0.0))}",
        f"Execution market price on proposed side: {_format_optional_probability(row.get('execution_price'))}",
        f"Reference edge: {_format_float(row.get('decision_edge', 0.0))}",
        f"Execution edge: {_format_float(row.get('execution_edge', 0.0))}",
        f"Bookmaker count: {int(float(row.get('num_bookmakers', 0) or 0))}",
        f"Minimum prior matches in matchup: {row.get('min_player_matches', '')}",
        f"Structured automation status: {row.get('automation_status', '')}",
        f"Structured reasons: {row.get('automation_reasons', '') or 'none'}",
        f"Second-source status: {row.get('second_source_status', '')}",
        f"Second-source detail: {row.get('second_source_detail', '') or 'none'}",
    ]
    lines.append(
        "Do a grounded real-world tennis sanity check. "
        "Web search is on, but do not search for withdrawals or injuries. "
        "Treat sparse public information as neutral, not negative. "
        "Do not downgrade a player just because coverage is light. "
        "Treat the packet's rank, rank-points, and recent-form values as model snapshots that may lag public live updates; that alone is not a veto reason. "
        "If your only concern is fresher public stats than the packet, return NO_VETO and include risk flag packet_snapshot_lag_only. "
        "If you return AUTO_BLOCK or AUTO_SKIP, include at least one concise risk flag describing the contradiction. "
        "If the bet still looks sensible in context, return NO_VETO. "
        "Only return AUTO_BLOCK or AUTO_SKIP if the broader tennis context makes the bet look clearly wrong or too shaky."
    )
    return "\n".join(lines)


def _log_tennis_veto_decision(decision: TennisVetoDecision) -> None:
    try:
        with open(TENNIS_LLM_VETO_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to log tennis veto decision: %s", exc)


def _fallback_decision(
    *,
    verdict: Literal["AUTO_BLOCK", "AUTO_SKIP", "NO_VETO"],
    rationale: str,
    risk_flags: list[str] | None = None,
    row: dict[str, object] | None = None,
) -> TennisVetoDecision:
    row = row or {}
    contradiction_strength = "unknown"
    if verdict == "AUTO_BLOCK":
        contradiction_strength = "strong"
    elif verdict == "AUTO_SKIP":
        contradiction_strength = "moderate"
    return TennisVetoDecision(
        verdict=verdict,
        confidence=0.0,
        rationale=rationale,
        risk_flags=list(risk_flags or []),
        timestamp=datetime.now(timezone.utc).isoformat(),
        coverage_quality="unknown",
        contradiction_strength=contradiction_strength,
        policy_adjustment="system_fail_closed" if verdict in {"AUTO_BLOCK", "AUTO_SKIP"} else "",
        fighter_a=str(row.get("fighter_a") or ""),
        fighter_b=str(row.get("fighter_b") or ""),
        decision_fighter=str(row.get("decision_fighter") or ""),
        tournament_name=str(row.get("tournament_name") or ""),
        tour=str(row.get("tour") or ""),
        source_summary=[],
        policy_exempt=True,
    )


def _default_contradiction_strength_for_verdict(verdict: str) -> str:
    if verdict == "AUTO_BLOCK":
        return "strong"
    if verdict == "AUTO_SKIP":
        return "moderate"
    if verdict == "NO_VETO":
        return "none"
    return "unknown"


def _is_snapshot_lag_flag(flag: str) -> bool:
    normalized = str(flag or "").strip().lower()
    if not normalized:
        return False
    snapshot_terms = (
        "packet",
        "snapshot",
        "stale",
        "outdated",
        "rank",
        "ranking",
        "recent_form",
        "recent form",
        "form",
        "underestimating",
        "overstated",
        "understated",
        "public",
        "model_",
    )
    return any(term in normalized for term in snapshot_terms)


def _is_hard_real_world_flag(flag: str) -> bool:
    normalized = str(flag or "").strip().lower()
    if not normalized:
        return False
    hard_terms = (
        "defending",
        "champion",
        "head_to_head",
        "head-to-head",
        "wrong_opponent",
        "wrong_tournament",
        "already_started",
        "already_completed",
        "identity",
        "misframed",
        "market_context",
    )
    return any(term in normalized for term in hard_terms)


def _normalize_veto_payload(
    payload: dict[str, object],
    *,
    row: dict[str, object],
) -> TennisVetoDecision:
    verdict = str(payload.get("verdict") or "AUTO_SKIP").strip().upper()
    if verdict not in {"AUTO_BLOCK", "AUTO_SKIP", "NO_VETO"}:
        return _fallback_decision(
            verdict="AUTO_SKIP",
            rationale="Gemini veto returned an invalid verdict",
            risk_flags=["invalid_veto_verdict"],
            row=row,
        )

    confidence = _normalize_confidence(payload.get("confidence", 0.0))

    rationale = str(payload.get("rationale") or "").strip() or "No rationale supplied by tennis veto operator"
    source_summary_raw = payload.get("source_summary") or []
    if isinstance(source_summary_raw, list):
        source_summary = [str(item).strip() for item in source_summary_raw if str(item).strip()]
    else:
        source_text = str(source_summary_raw or "").strip()
        source_summary = [source_text] if source_text else []

    risk_flags_raw = payload.get("risk_flags") or []
    if isinstance(risk_flags_raw, list):
        risk_flags = [str(item).strip() for item in risk_flags_raw if str(item).strip()]
    else:
        risk_flags = _split_reason_string(risk_flags_raw)

    coverage_quality = _normalize_coverage_quality(payload.get("coverage_quality"))
    contradiction_strength = _normalize_contradiction_strength(payload.get("contradiction_strength"))

    if verdict in {"AUTO_BLOCK", "AUTO_SKIP"} and contradiction_strength == "unknown":
        contradiction_strength = _default_contradiction_strength_for_verdict(verdict)
        if "invalid_contradiction_strength" not in risk_flags:
            risk_flags.append("invalid_contradiction_strength")
        if "inferred_contradiction_strength_from_verdict" not in risk_flags:
            risk_flags.append("inferred_contradiction_strength_from_verdict")

    return TennisVetoDecision(
        verdict=verdict,
        confidence=confidence,
        rationale=rationale,
        risk_flags=risk_flags,
        timestamp=datetime.now(timezone.utc).isoformat(),
        coverage_quality=coverage_quality,
        contradiction_strength=contradiction_strength,
        policy_adjustment="",
        fighter_a=str(row.get("fighter_a") or ""),
        fighter_b=str(row.get("fighter_b") or ""),
        decision_fighter=str(row.get("decision_fighter") or ""),
        tournament_name=str(row.get("tournament_name") or ""),
        tour=str(row.get("tour") or ""),
        source_summary=source_summary,
        policy_exempt=False,
    )


def _apply_asymmetric_policy(decision: TennisVetoDecision) -> TennisVetoDecision:
    """Constrain Gemini so sparse coverage and weak contradiction stay neutral."""
    if decision.policy_exempt:
        return decision

    contradiction_strength = decision.contradiction_strength
    coverage_quality = decision.coverage_quality
    original_verdict = decision.verdict
    policy_adjustment = ""
    risk_flags = {flag.strip() for flag in decision.risk_flags if str(flag).strip()}

    if "packet_snapshot_lag_only" in risk_flags:
        decision.verdict = "NO_VETO"
        policy_adjustment = "packet_snapshot_lag_only_forced_no_veto"

    if not policy_adjustment and original_verdict in {"AUTO_BLOCK", "AUTO_SKIP"} and risk_flags:
        has_snapshot_lag_flags = any(_is_snapshot_lag_flag(flag) for flag in risk_flags)
        has_hard_real_world_flags = any(_is_hard_real_world_flag(flag) for flag in risk_flags)
        if has_snapshot_lag_flags and not has_hard_real_world_flags:
            decision.verdict = "NO_VETO"
            policy_adjustment = "snapshot_lag_reasoning_forced_no_veto"

    if not policy_adjustment and original_verdict in {"AUTO_BLOCK", "AUTO_SKIP"} and not risk_flags:
        decision.verdict = "NO_VETO"
        policy_adjustment = "negative_verdict_without_risk_flags_forced_no_veto"

    if not policy_adjustment and original_verdict == "AUTO_BLOCK":
        if contradiction_strength == "moderate":
            decision.verdict = "AUTO_SKIP"
            policy_adjustment = "downgraded_block_to_skip"
        elif contradiction_strength in {"none", "weak", "unknown"}:
            decision.verdict = "NO_VETO"
            policy_adjustment = "downgraded_block_to_no_veto"
    elif not policy_adjustment and original_verdict == "AUTO_SKIP":
        if contradiction_strength in {"none", "weak", "unknown"}:
            decision.verdict = "NO_VETO"
            policy_adjustment = "downgraded_skip_to_no_veto"
    elif not policy_adjustment and original_verdict == "NO_VETO":
        if contradiction_strength in {"moderate", "strong"}:
            policy_adjustment = "inconsistent_no_veto_metadata"

    if (
        coverage_quality == "sparse"
        and contradiction_strength in {"none", "weak", "unknown"}
        and policy_adjustment != "packet_snapshot_lag_only_forced_no_veto"
    ):
        decision.verdict = "NO_VETO"
        policy_adjustment = "sparse_coverage_forced_no_veto"

    if policy_adjustment:
        decision.policy_adjustment = policy_adjustment
        if policy_adjustment not in decision.risk_flags:
            decision.risk_flags.append(policy_adjustment)
        decision.rationale = f"{decision.rationale} [Policy adjustment: {policy_adjustment}]".strip()

    return decision


def _call_gemini_veto(prompt: str) -> tuple[dict[str, object] | None, list[str]]:
    text = ""
    grounding_sources: list[str] = []
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=TENNIS_LLM_VETO_MODEL,
            contents=f"{_build_system_prompt()}\n\n---\n\n{prompt}",
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0.1,
            },
        )

        text = response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            if grounding_metadata and getattr(grounding_metadata, "grounding_chunks", None):
                for chunk in grounding_metadata.grounding_chunks[:10]:
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        title = getattr(web, "title", "") or getattr(web, "uri", "")
                        grounding_sources.append(f"{title}: {web.uri}")

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0].strip()

        try:
            return json.loads(text), grounding_sources
        except json.JSONDecodeError:
            formatter_prompt = (
                "Convert the grounded tennis sanity-check analysis below into strict JSON only. "
                "Do not add facts that are not already supported by the grounded analysis. "
                "Use this schema exactly: "
                '{"verdict":"AUTO_BLOCK|AUTO_SKIP|NO_VETO","confidence":0.0,'
                '"rationale":"...",'
                '"risk_flags":["..."],'
                '"source_summary":["..."],'
                '"coverage_quality":"sparse|moderate|rich",'
                '"contradiction_strength":"none|weak|moderate|strong"}. '
                "If the only issue is that public rankings, rank points, or recent form are fresher than the packet, use NO_VETO and include risk flag packet_snapshot_lag_only. "
                "Any AUTO_BLOCK or AUTO_SKIP verdict must include at least one concise risk flag. "
                "If the analysis says the bet basically makes sense, use NO_VETO. "
                "If the analysis says the case is materially shaky or unresolved, use AUTO_SKIP. "
                "If the analysis says the bet is clearly wrong, use AUTO_BLOCK.\n\n"
                f"Grounded analysis:\n{text}\n\n"
                "Grounding sources:\n"
                + "\n".join(grounding_sources[:10])
            )
            formatter_response = client.models.generate_content(
                model=TENNIS_LLM_VETO_MODEL,
                contents=formatter_prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_json_schema": TENNIS_VETO_RESPONSE_JSON_SCHEMA,
                },
            )
            formatter_text = formatter_response.text.strip()
            if formatter_text.startswith("```"):
                formatter_text = formatter_text.split("\n", 1)[1] if "\n" in formatter_text else formatter_text
                formatter_text = formatter_text.rsplit("```", 1)[0].strip()
            payload = json.loads(formatter_text)
            return payload, grounding_sources
    except ImportError:
        logger.warning("google-genai package not installed for tennis veto operator")
        return None, grounding_sources
    except json.JSONDecodeError as exc:
        raw_preview = text[:300] if text else "(empty)"
        logger.warning("Failed to parse tennis Gemini response as JSON: %s — raw: %s", exc, raw_preview)
        return None, grounding_sources
    except Exception as exc:
        logger.warning("Tennis Gemini veto API error: %s", exc)
        return None, grounding_sources


def evaluate_tennis_veto_candidate(row: dict[str, object]) -> TennisVetoDecision:
    """Evaluate one structured tennis trade candidate through the Gemini veto layer."""
    if not TENNIS_LLM_VETO_ENABLED:
        return _fallback_decision(
            verdict="NO_VETO",
            rationale="Tennis LLM veto disabled",
            row=row,
        )

    if not GEMINI_API_KEY:
        decision = _fallback_decision(
            verdict="AUTO_SKIP" if TENNIS_LLM_VETO_FAIL_CLOSED else "NO_VETO",
            rationale="Gemini veto enabled but GEMINI_API_KEY is missing",
            risk_flags=["missing_gemini_api_key"],
            row=row,
        )
        _log_tennis_veto_decision(decision)
        return decision

    payload, grounding_sources = _call_gemini_veto(_build_user_prompt(row))
    if payload is None:
        decision = _fallback_decision(
            verdict="AUTO_SKIP" if TENNIS_LLM_VETO_FAIL_CLOSED else "NO_VETO",
            rationale="Gemini veto call failed",
            risk_flags=["gemini_veto_error"],
            row=row,
        )
        _log_tennis_veto_decision(decision)
        return decision

    decision = _apply_asymmetric_policy(_normalize_veto_payload(payload, row=row))
    if grounding_sources and not decision.source_summary:
        decision.source_summary = grounding_sources[:5]
    _log_tennis_veto_decision(decision)
    return decision


def _coerce_tennis_veto_decision(
    result: TennisVetoDecision | dict[str, object],
    *,
    row: dict[str, object],
) -> TennisVetoDecision:
    if isinstance(result, TennisVetoDecision):
        return _apply_asymmetric_policy(result)
    if isinstance(result, dict):
        return _apply_asymmetric_policy(_normalize_veto_payload(result, row=row))
    return _apply_asymmetric_policy(_fallback_decision(
        verdict="AUTO_SKIP",
        rationale="Unsupported tennis veto result type",
        risk_flags=["invalid_veto_result"],
        row=row,
    ))


def apply_tennis_llm_veto(
    decisions: pd.DataFrame,
    *,
    evaluator=None,
) -> pd.DataFrame:
    """Apply the Gemini veto layer to structured tennis trade candidates."""
    if decisions.empty:
        return decisions.copy()

    evaluator = evaluator or evaluate_tennis_veto_candidate
    rows: list[dict[str, object]] = []

    for row in decisions.to_dict("records"):
        updated = dict(row)
        structured_status = str(updated.get("automation_status") or "")
        execution_status = str(updated.get("execution_status") or "")

        if structured_status != "auto_eligible":
            updated["llm_veto_status"] = "NO_VETO"
            updated["llm_veto_reason"] = f"structured_status_{structured_status or 'unknown'}"
            updated["llm_veto_flags"] = ""
            updated["llm_veto_confidence"] = 0.0
            updated["llm_veto_sources"] = ""
            updated["llm_coverage_quality"] = "unknown"
            updated["llm_contradiction_strength"] = "unknown"
            updated["llm_policy_adjustment"] = ""
            rows.append(updated)
            continue

        if execution_status != "bettable_now":
            updated["llm_veto_status"] = "NO_VETO"
            updated["llm_veto_reason"] = "not_live_trade_candidate"
            updated["llm_veto_flags"] = ""
            updated["llm_veto_confidence"] = 0.0
            updated["llm_veto_sources"] = ""
            updated["llm_coverage_quality"] = "unknown"
            updated["llm_contradiction_strength"] = "unknown"
            updated["llm_policy_adjustment"] = ""
            rows.append(updated)
            continue

        # Check session cache — avoid re-evaluating the same match
        fighter_a = str(updated.get("fighter_a") or "")
        fighter_b = str(updated.get("fighter_b") or "")
        cache_key = _match_cache_key(fighter_a, fighter_b) if fighter_a and fighter_b else ""

        decision = None
        if cache_key and cache_key in _veto_cache:
            cached, cached_at = _veto_cache[cache_key]
            age = time.time() - cached_at
            if age < VETO_CACHE_TTL_SECONDS:
                logger.info(
                    "Tennis veto cache hit for %s vs %s — reusing %s verdict "
                    "(age %.0fm, saved a Gemini call)",
                    fighter_a, fighter_b, cached.verdict, age / 60,
                )
                decision = cached
            else:
                logger.info(
                    "Tennis veto cache expired for %s vs %s (age %.1fh) — re-evaluating",
                    fighter_a, fighter_b, age / 3600,
                )
                del _veto_cache[cache_key]

        if decision is None:
            decision = _coerce_tennis_veto_decision(evaluator(updated), row=updated)
            # Don't cache transient error decisions — retry next loop cycle
            error_flags = {"gemini_veto_error", "missing_gemini_api_key"}
            if cache_key and not (error_flags & set(decision.risk_flags)):
                _veto_cache[cache_key] = (decision, time.time())

        updated["llm_veto_status"] = decision.verdict
        updated["llm_veto_reason"] = decision.rationale
        updated["llm_veto_flags"] = _reason_string(decision.risk_flags)
        updated["llm_veto_confidence"] = decision.confidence
        updated["llm_veto_sources"] = _reason_string(decision.source_summary)
        updated["llm_coverage_quality"] = decision.coverage_quality
        updated["llm_contradiction_strength"] = decision.contradiction_strength
        updated["llm_policy_adjustment"] = decision.policy_adjustment

        if decision.verdict in {"AUTO_BLOCK", "AUTO_SKIP"}:
            automation_status = "auto_block" if decision.verdict == "AUTO_BLOCK" else "auto_skip"
            updated["structured_status"] = automation_status
            updated["automation_status"] = automation_status
            updated["execution_allowed"] = False
            updated["execution_status"] = "llm_auto_block" if decision.verdict == "AUTO_BLOCK" else "llm_auto_skip"
            updated["trade_ready"] = False
            updated["trade_status"] = "blocked" if decision.verdict == "AUTO_BLOCK" else "auto_skipped"

            automation_reasons = _split_reason_string(updated.get("automation_reasons"))
            trade_reasons = _split_reason_string(updated.get("trade_reasons"))
            automation_reasons.append(decision.verdict.lower())
            trade_reasons.append(decision.verdict.lower())
            automation_reasons.extend(decision.risk_flags)
            trade_reasons.extend(decision.risk_flags)
            updated["automation_reasons"] = _reason_string(automation_reasons)
            updated["trade_reasons"] = _reason_string(trade_reasons)

        rows.append(updated)

    return pd.DataFrame(rows)
