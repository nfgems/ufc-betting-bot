import pandas as pd

from src.strategy import tennis_llm_operator
from src.strategy.tennis_llm_operator import (
    TennisVetoDecision,
    _build_system_prompt,
    _build_user_prompt,
    apply_tennis_llm_veto,
    evaluate_tennis_veto_candidate,
)


def _base_trade_candidate() -> dict:
    return {
        "fighter_a": "Player A",
        "fighter_b": "Player B",
        "decision_fighter": "Player A",
        "tournament_name": "Miami Open",
        "tour": "atp",
        "surface": "Hard",
        "a_age": 18.2,
        "b_age": 27.4,
        "a_rank": 54,
        "b_rank": 21,
        "a_rank_points": 1120,
        "b_rank_points": 2145,
        "a_num_matches": 7,
        "b_num_matches": 74,
        "a_recent_form": 0.8,
        "b_recent_form": 0.5,
        "a_elo": 1680,
        "b_elo": 1715,
        "a_surface_elo": 1710,
        "b_surface_elo": 1688,
        "automation_status": "auto_eligible",
        "execution_status": "bettable_now",
        "execution_allowed": True,
        "trade_ready": True,
        "trade_status": "tradeable_now",
        "trade_reasons": "",
        "automation_reasons": "",
    }


def test_apply_tennis_llm_veto_skips_non_trade_candidate_without_calling_evaluator():
    decisions = pd.DataFrame(
        [
            {
                **_base_trade_candidate(),
                "execution_status": "missing_execution_price",
                "trade_ready": False,
                "trade_status": "reference_only_eligible",
            }
        ]
    )

    def _unexpected_call(_row):
        raise AssertionError("LLM evaluator should not be called for non-trade candidates")

    reviewed = apply_tennis_llm_veto(decisions, evaluator=_unexpected_call)
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_veto_reason"] == "not_live_trade_candidate"
    assert row["trade_status"] == "reference_only_eligible"


def test_apply_tennis_llm_veto_resets_stale_fields_when_row_is_no_longer_tradeable():
    decisions = pd.DataFrame(
        [
            {
                **_base_trade_candidate(),
                "execution_status": "missing_execution_price",
                "trade_ready": False,
                "trade_status": "reference_only_eligible",
                "llm_veto_status": "AUTO_SKIP",
                "llm_veto_reason": "stale_reason",
                "llm_veto_flags": "stale_flag",
                "llm_veto_confidence": 0.91,
                "llm_veto_sources": "stale_source",
                "llm_coverage_quality": "rich",
                "llm_contradiction_strength": "moderate",
                "llm_policy_adjustment": "stale_adjustment",
            }
        ]
    )

    reviewed = apply_tennis_llm_veto(decisions, evaluator=lambda _row: {"verdict": "AUTO_BLOCK"})
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_veto_reason"] == "not_live_trade_candidate"
    assert row["llm_veto_flags"] == ""
    assert row["llm_veto_confidence"] == 0.0
    assert row["llm_veto_sources"] == ""
    assert row["llm_coverage_quality"] == "unknown"
    assert row["llm_contradiction_strength"] == "unknown"
    assert row["llm_policy_adjustment"] == ""


def test_apply_tennis_llm_veto_auto_blocks_trade_candidate():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_BLOCK",
            "confidence": 0.92,
            "rationale": "Market context looks misframed for this matchup",
            "risk_flags": ["market_context_misframed"],
            "coverage_quality": "rich",
            "contradiction_strength": "strong",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_BLOCK"
    assert row["structured_status"] == "auto_block"
    assert row["automation_status"] == "auto_block"
    assert row["execution_status"] == "llm_auto_block"
    assert bool(row["trade_ready"]) is False
    assert row["trade_status"] == "blocked"
    assert row["llm_coverage_quality"] == "rich"
    assert row["llm_contradiction_strength"] == "strong"
    assert "auto_block" in row["automation_reasons"]
    assert "market_context_misframed" in row["trade_reasons"]


def test_apply_tennis_llm_veto_auto_skips_trade_candidate():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.61,
            "rationale": "Too much uncertainty around how real the current level is",
            "risk_flags": ["context_uncertain"],
            "coverage_quality": "moderate",
            "contradiction_strength": "moderate",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_SKIP"
    assert row["structured_status"] == "auto_skip"
    assert row["automation_status"] == "auto_skip"
    assert row["execution_status"] == "llm_auto_skip"
    assert bool(row["trade_ready"]) is False
    assert row["trade_status"] == "auto_skipped"
    assert row["llm_coverage_quality"] == "moderate"
    assert row["llm_contradiction_strength"] == "moderate"
    assert "auto_skip" in row["automation_reasons"]
    assert "context_uncertain" in row["trade_reasons"]


def test_apply_tennis_llm_veto_no_veto_leaves_trade_ready():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda row: TennisVetoDecision(
            verdict="NO_VETO",
            confidence=0.25,
            rationale="No concrete red flags found",
            risk_flags=[],
            timestamp="2026-03-22T12:00:00+00:00",
            coverage_quality="sparse",
            contradiction_strength="none",
            fighter_a=row["fighter_a"],
            fighter_b=row["fighter_b"],
            decision_fighter=row["decision_fighter"],
            tournament_name=row["tournament_name"],
            tour=row["tour"],
        ),
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_coverage_quality"] == "sparse"
    assert row["llm_contradiction_strength"] == "none"
    assert row["automation_status"] == "auto_eligible"
    assert bool(row["trade_ready"]) is True
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_sparse_weak_contradiction_forces_no_veto():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.72,
            "rationale": "Very little coverage and only weak concern",
            "risk_flags": ["weak_concern"],
            "coverage_quality": "sparse",
            "contradiction_strength": "weak",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "sparse_coverage_forced_no_veto"
    assert row["automation_status"] == "auto_eligible"
    assert bool(row["trade_ready"]) is True
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_downgrades_weak_skip_to_no_veto():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.64,
            "rationale": "Weak contradiction only",
            "risk_flags": ["weak_contradiction"],
            "coverage_quality": "rich",
            "contradiction_strength": "weak",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "downgraded_skip_to_no_veto"
    assert row["automation_status"] == "auto_eligible"
    assert bool(row["trade_ready"]) is True


def test_apply_tennis_llm_veto_downgrades_moderate_block_to_skip():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_BLOCK",
            "confidence": 0.81,
            "rationale": "Moderate contradiction only",
            "risk_flags": ["moderate_contradiction"],
            "coverage_quality": "rich",
            "contradiction_strength": "moderate",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_SKIP"
    assert row["llm_policy_adjustment"] == "downgraded_block_to_skip"
    assert row["automation_status"] == "auto_skip"
    assert row["trade_status"] == "auto_skipped"


def test_apply_tennis_llm_veto_normalizes_real_gemini_label_variants():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "NO_VETO",
            "confidence": "HIGH",
            "rationale": "Grounded tennis context supports staying out of the way",
            "risk_flags": [],
            "source_summary": "Single grounded summary string",
            "coverage_quality": "HIGH",
            "contradiction_strength": "LOW",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_veto_confidence"] == 0.85
    assert row["llm_coverage_quality"] == "rich"
    assert row["llm_contradiction_strength"] == "weak"
    assert row["llm_veto_sources"] == "Single grounded summary string"
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_keeps_supportive_no_veto_despite_inconsistent_moderate_contradiction():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "NO_VETO",
            "confidence": 0.66,
            "rationale": "Some moderate contradiction exists",
            "risk_flags": ["moderate_contradiction_present"],
            "coverage_quality": "rich",
            "contradiction_strength": "moderate",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "inconsistent_no_veto_metadata"
    assert row["automation_status"] == "auto_eligible"
    assert row["execution_status"] == "bettable_now"
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_fails_closed_on_invalid_verdict_payload():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "PASS",
            "confidence": 0.73,
            "rationale": "Old-schema output",
            "risk_flags": ["old_schema_payload"],
            "coverage_quality": "moderate",
            "contradiction_strength": "none",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_SKIP"
    assert row["automation_status"] == "auto_skip"
    assert row["execution_status"] == "llm_auto_skip"
    assert "invalid_veto_verdict" in row["llm_veto_flags"]
    assert row["trade_status"] == "auto_skipped"


def test_apply_tennis_llm_veto_infers_invalid_negative_contradiction_strength_from_verdict():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.58,
            "rationale": "The broader context looks shaky",
            "risk_flags": ["context_uncertain"],
            "coverage_quality": "moderate",
            "contradiction_strength": "MEDIUM-SEVERE",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_SKIP"
    assert row["llm_contradiction_strength"] == "moderate"
    assert "invalid_contradiction_strength" in row["llm_veto_flags"]
    assert "inferred_contradiction_strength_from_verdict" in row["llm_veto_flags"]
    assert row["trade_status"] == "auto_skipped"


def test_apply_tennis_llm_veto_forces_no_veto_on_packet_snapshot_lag_only_flag():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_BLOCK",
            "confidence": 0.81,
            "rationale": "Public rankings are fresher than the packet",
            "risk_flags": ["packet_snapshot_lag_only"],
            "coverage_quality": "rich",
            "contradiction_strength": "strong",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "packet_snapshot_lag_only_forced_no_veto"
    assert row["automation_status"] == "auto_eligible"
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_forces_no_veto_on_negative_verdict_without_risk_flags():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.67,
            "rationale": "The context feels shaky",
            "risk_flags": [],
            "coverage_quality": "rich",
            "contradiction_strength": "moderate",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "negative_verdict_without_risk_flags_forced_no_veto"
    assert row["automation_status"] == "auto_eligible"
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_forces_no_veto_on_snapshot_lag_only_reasoning_flags():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_SKIP",
            "confidence": 0.78,
            "rationale": "The packet looks stale for the opponent",
            "risk_flags": ["outdated_opponent_ranking_and_form", "model_underestimating_opponent"],
            "coverage_quality": "rich",
            "contradiction_strength": "strong",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "NO_VETO"
    assert row["llm_policy_adjustment"] == "snapshot_lag_reasoning_forced_no_veto"
    assert row["automation_status"] == "auto_eligible"
    assert row["trade_status"] == "tradeable_now"


def test_apply_tennis_llm_veto_keeps_negative_verdict_when_hard_real_world_flag_exists():
    decisions = pd.DataFrame([_base_trade_candidate()])

    reviewed = apply_tennis_llm_veto(
        decisions,
        evaluator=lambda _row: {
            "verdict": "AUTO_BLOCK",
            "confidence": 0.93,
            "rationale": "Opponent is defending champion and packet also understates them",
            "risk_flags": ["opponent_rank_stale", "opponent_is_defending_tournament_champion"],
            "coverage_quality": "rich",
            "contradiction_strength": "strong",
        },
    )
    row = reviewed.iloc[0]

    assert row["llm_veto_status"] == "AUTO_BLOCK"
    assert row["automation_status"] == "auto_block"
    assert row["trade_status"] == "blocked"


def test_evaluate_tennis_veto_candidate_fails_closed_without_api_key(monkeypatch):
    monkeypatch.setattr(tennis_llm_operator, "TENNIS_LLM_VETO_ENABLED", True)
    monkeypatch.setattr(tennis_llm_operator, "TENNIS_LLM_VETO_FAIL_CLOSED", True)
    monkeypatch.setattr(tennis_llm_operator, "GEMINI_API_KEY", "")
    monkeypatch.setattr(tennis_llm_operator, "_log_tennis_veto_decision", lambda _decision: None)

    decision = evaluate_tennis_veto_candidate(_base_trade_candidate())

    assert decision.verdict == "AUTO_SKIP"
    assert "missing" in decision.rationale.lower()
    assert decision.policy_exempt is True


def test_tennis_system_prompt_explicitly_avoids_withdrawal_search():
    prompt = _build_system_prompt().lower()

    assert "do not search for or reason about withdrawals" in prompt
    assert "lack of public information is neutral" in prompt
    assert "positive breakout or phenom context can support no_veto" in prompt
    assert "packet_snapshot_lag_only" in prompt
    assert "must include at least one concise risk flag" in prompt


def test_tennis_user_prompt_explains_packet_snapshot_lag_is_not_a_veto_reason():
    prompt = _build_user_prompt(_base_trade_candidate()).lower()

    assert "may lag public live updates" in prompt
    assert "packet_snapshot_lag_only" in prompt


def test_evaluate_tennis_veto_candidate_uses_mocked_gemini_payload(monkeypatch):
    monkeypatch.setattr(tennis_llm_operator, "TENNIS_LLM_VETO_ENABLED", True)
    monkeypatch.setattr(tennis_llm_operator, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(tennis_llm_operator, "_log_tennis_veto_decision", lambda _decision: None)
    monkeypatch.setattr(
        tennis_llm_operator,
        "_call_gemini_veto",
        lambda _prompt: (
            {
                "verdict": "NO_VETO",
                "confidence": 0.73,
                "rationale": "No concrete issue found",
                "risk_flags": [],
                "source_summary": ["ATP Tour: https://www.atptour.com"],
                "coverage_quality": "moderate",
                "contradiction_strength": "none",
            },
            [],
        ),
    )

    decision = evaluate_tennis_veto_candidate(_base_trade_candidate())

    assert decision.verdict == "NO_VETO"
    assert decision.confidence == 0.73
    assert decision.coverage_quality == "moderate"
    assert decision.contradiction_strength == "none"
    assert decision.source_summary == ["ATP Tour: https://www.atptour.com"]
