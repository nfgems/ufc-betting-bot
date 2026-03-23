"""Tests for the LLM Operator decision-making pipeline."""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.strategy.llm_operator import (
    OPERATOR_DIR,
    GrokIntel,
    OperatorDecision,
    ResearchFindings,
    _analyze_matchup_from_features,
    _check_blind_spots,
    _check_correlated_exposure,
    _check_motivation_signals,
    _check_recency_context,
    _parse_grok_response,
    evaluate_bet,
    evaluate_bets,
    load_blind_spots,
    run_research_pipeline,
    save_blind_spots,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_features():
    return {
        "a_days_since_last_fight": 800,
        "b_days_since_last_fight": 90,
        "a_is_short_notice": False,
        "b_is_short_notice": True,
        "a_is_debut": False,
        "b_is_debut": False,
        "a_num_fights": 12,
        "b_num_fights": 5,
        "a_roll_slpm": 4.5,
        "b_roll_slpm": 3.2,
        "a_roll_str_acc": 0.48,
        "b_roll_str_acc": 0.42,
        "a_roll_td_avg": 3.5,
        "b_roll_td_avg": 0.5,
        "a_roll_td_acc": 0.45,
        "b_roll_td_acc": 0.30,
        "a_roll_td_def": 0.70,
        "b_roll_td_def": 0.45,
        "a_stance": "Orthodox",
        "b_stance": "Southpaw",
        "a_lose_streak": 0,
        "b_lose_streak": 3,
        "a_current_win_streak": 5,
        "b_current_win_streak": 0,
        "a_age": 30,
        "b_age": 39,
        "a_ko_rate": 0.4,
        "a_sub_rate": 0.2,
        "a_dec_rate": 0.4,
        "b_ko_rate": 0.6,
        "b_sub_rate": 0.1,
        "b_dec_rate": 0.3,
    }


@pytest.fixture
def sample_bets():
    return pd.DataFrame([
        {
            "fighter_a": "Fighter Alpha",
            "fighter_b": "Fighter Beta",
            "bet_on": "Fighter Alpha",
            "bet_side": "a",
            "model_prob": 0.65,
            "blended_prob": 0.58,
            "market_prob": 0.50,
            "edge": 0.08,
            "decimal_odds": 2.0,
            "event_date": "2026-04-01",
            "weight_class": "Welterweight",
        },
        {
            "fighter_a": "Fighter Gamma",
            "fighter_b": "Fighter Delta",
            "bet_on": "Fighter Delta",
            "bet_side": "b",
            "model_prob": 0.70,
            "blended_prob": 0.62,
            "market_prob": 0.55,
            "edge": 0.07,
            "decimal_odds": 1.82,
            "event_date": "2026-04-01",
            "weight_class": "Lightweight",
        },
    ])


# ---------------------------------------------------------------------------
# Recency context
# ---------------------------------------------------------------------------

class TestRecencyContext:
    def test_long_layoff_detected(self, sample_features):
        flags = _check_recency_context(sample_features, "Alpha", "Beta")
        assert any("800 days" in f for f in flags)
        assert any("long layoff" in f.lower() for f in flags)

    def test_short_notice_detected(self, sample_features):
        flags = _check_recency_context(sample_features, "Alpha", "Beta")
        assert any("short-notice" in f.lower() for f in flags)

    def test_debut_detected(self):
        features = {"a_is_debut": True, "b_num_fights": 5}
        flags = _check_recency_context(features, "Alpha", "Beta")
        assert any("debut" in f.lower() for f in flags)

    def test_no_flags_normal_fight(self):
        features = {
            "a_days_since_last_fight": 90,
            "b_days_since_last_fight": 120,
            "a_is_short_notice": False,
            "b_is_short_notice": False,
            "a_num_fights": 10,
            "b_num_fights": 8,
        }
        flags = _check_recency_context(features, "Alpha", "Beta")
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# Matchup analysis
# ---------------------------------------------------------------------------

class TestMatchupAnalysis:
    def test_mismatch_detected(self, sample_features):
        analysis = _analyze_matchup_from_features(
            sample_features, "Alpha", "Beta"
        )
        assert "MISMATCH" in analysis
        assert "Alpha" in analysis

    def test_stance_matchup_noted(self, sample_features):
        analysis = _analyze_matchup_from_features(
            sample_features, "Alpha", "Beta"
        )
        assert "Orthodox" in analysis
        assert "Southpaw" in analysis

    def test_empty_features(self):
        analysis = _analyze_matchup_from_features({}, "Alpha", "Beta")
        assert "Insufficient" in analysis


# ---------------------------------------------------------------------------
# Motivation signals
# ---------------------------------------------------------------------------

class TestMotivationSignals:
    def test_losing_streak_flagged(self, sample_features):
        flags = _check_motivation_signals(sample_features, "Alpha", "Beta")
        assert any("3-fight losing streak" in f for f in flags)

    def test_win_streak_flagged(self, sample_features):
        flags = _check_motivation_signals(sample_features, "Alpha", "Beta")
        assert any("5-fight win streak" in f for f in flags)

    def test_age_flagged(self, sample_features):
        flags = _check_motivation_signals(sample_features, "Alpha", "Beta")
        assert any("39 years old" in f for f in flags)

    def test_no_flags_normal(self):
        features = {
            "a_current_lose_streak": 0,
            "b_current_lose_streak": 1,
            "a_current_win_streak": 2,
            "b_current_win_streak": 1,
            "a_age": 28,
            "b_age": 30,
        }
        flags = _check_motivation_signals(features, "Alpha", "Beta")
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# Blind spots
# ---------------------------------------------------------------------------

class TestBlindSpots:
    def test_load_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )
        assert load_blind_spots() == []

    def test_save_and_load(self, tmp_path, monkeypatch):
        path = tmp_path / "blind_spots.json"
        monkeypatch.setattr("src.strategy.llm_operator.BLIND_SPOTS_PATH", path)
        spots = [
            {
                "description": "Model overvalues grapplers vs strikers with bad TDD",
                "pattern": {"a_roll_td_avg": {"op": "gt", "value": 3.0}},
                "accuracy": "62%",
            }
        ]
        save_blind_spots(spots)
        loaded = load_blind_spots()
        assert len(loaded) == 1
        assert loaded[0]["description"] == spots[0]["description"]

    def test_blind_spot_match(self, tmp_path, monkeypatch, sample_features):
        path = tmp_path / "blind_spots.json"
        monkeypatch.setattr("src.strategy.llm_operator.BLIND_SPOTS_PATH", path)
        spots = [
            {
                "description": "Overvalues grapplers",
                "pattern": {"a_roll_td_avg": {"op": "gt", "value": 3.0}},
                "accuracy": "60%",
            }
        ]
        save_blind_spots(spots)
        matches = _check_blind_spots(sample_features, "Alpha", "Beta", 0.65, 0.50)
        assert len(matches) == 1
        assert "Overvalues grapplers" in matches[0]

    def test_blind_spot_no_match(self, tmp_path, monkeypatch, sample_features):
        path = tmp_path / "blind_spots.json"
        monkeypatch.setattr("src.strategy.llm_operator.BLIND_SPOTS_PATH", path)
        spots = [
            {
                "description": "Something else",
                "pattern": {"a_roll_td_avg": {"op": "gt", "value": 10.0}},
            }
        ]
        save_blind_spots(spots)
        matches = _check_blind_spots(sample_features, "Alpha", "Beta", 0.65, 0.50)
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Correlated exposure
# ---------------------------------------------------------------------------

class TestCorrelatedExposure:
    def test_no_existing_bets(self):
        assert _check_correlated_exposure("UFC 350", []) == ""

    def test_moderate_exposure(self):
        bets = [
            {"event_title": "UFC 350"},
            {"event_title": "UFC 350"},
        ]
        warning = _check_correlated_exposure("UFC 350", bets)
        assert "Moderate exposure" in warning

    def test_concentration_risk(self):
        bets = [
            {"event_title": "UFC 350"},
            {"event_title": "UFC 350"},
            {"event_title": "UFC 350"},
        ]
        warning = _check_correlated_exposure("UFC 350", bets)
        assert "CONCENTRATION RISK" in warning

    def test_different_event(self):
        bets = [
            {"event_title": "UFC 349"},
            {"event_title": "UFC 349"},
            {"event_title": "UFC 349"},
        ]
        warning = _check_correlated_exposure("UFC 350", bets)
        assert warning == ""


# ---------------------------------------------------------------------------
# Research pipeline
# ---------------------------------------------------------------------------

class TestResearchPipeline:
    def test_runs_all_layers(self, sample_features, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "")
        findings = run_research_pipeline(
            features=sample_features,
            fighter_a="Alpha",
            fighter_b="Beta",
            model_prob_a=0.65,
            market_prob_a=0.50,
        )
        assert isinstance(findings, ResearchFindings)
        assert len(findings.recency_flags) > 0
        assert findings.matchup_analysis != ""
        assert len(findings.motivation_flags) > 0
        assert findings.grok_intel.skipped is True


# ---------------------------------------------------------------------------
# evaluate_bet — with mocked Claude API
# ---------------------------------------------------------------------------

class TestEvaluateBet:
    def test_evaluate_with_mocked_claude(self, sample_features, tmp_path, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "")
        monkeypatch.setattr("src.strategy.llm_operator.ANTHROPIC_API_KEY", "fake-key")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )

        mock_synthesis_result = {
            "verdict": "BLOCK",
            "rationale": "Fighter Alpha has a long layoff and is facing a short-notice opponent with unknown preparation",
            "fighter_assessment": "Alpha is returning after 800+ days, Beta is a short-notice step-in",
            "risk_flags": ["long_layoff", "short_notice_opponent"],
        }

        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda prompt: mock_synthesis_result,
        )

        decision = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            features=sample_features,
        )

        assert isinstance(decision, OperatorDecision)
        assert decision.verdict == "BLOCK"
        assert "long_layoff" in decision.risk_flags
        assert decision.rationale != ""

        # Check decision was logged (JSONL format — one JSON object per line)
        log_path = tmp_path / "decision_log.jsonl"
        assert log_path.exists()
        lines = [l for l in log_path.read_text().strip().splitlines() if l.strip()]
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["verdict"] == "BLOCK"

    def test_evaluate_passthrough_no_api_key(self, sample_features, tmp_path, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "")
        monkeypatch.setattr("src.strategy.llm_operator.ANTHROPIC_API_KEY", "")
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )

        decision = evaluate_bet(
            fighter_a="Alpha",
            fighter_b="Beta",
            bet_on="Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            features=sample_features,
        )

        # Without API key, should passthrough as PASS
        assert decision.verdict == "PASS"
        assert "passthrough" in decision.rationale.lower()


# ---------------------------------------------------------------------------
# evaluate_bets (batch) — operator disabled
# ---------------------------------------------------------------------------

class TestEvaluateBetsBatch:
    def test_disabled_operator_passes_all(self, sample_bets, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", False)
        result = evaluate_bets(sample_bets)
        assert len(result) == len(sample_bets)
        assert all(result["operator_verdict"] == "PASS")

    def test_enabled_operator_filters(self, sample_bets, tmp_path, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "gate")
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "")
        monkeypatch.setattr("src.strategy.llm_operator.ANTHROPIC_API_KEY", "fake")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )

        # First bet: PASS, second bet: BLOCK
        call_count = [0]

        def mock_call_claude(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "verdict": "PASS",
                    "rationale": "Both fighters are evenly matched, model edge is reasonable",
                    "fighter_assessment": "Competitive matchup",
                    "risk_flags": [],
                }
            return {
                "verdict": "BLOCK",
                "rationale": "Model is picking an inferior fighter against a clearly better opponent",
                "fighter_assessment": "Massive talent gap the model can't see",
                "risk_flags": ["talent_mismatch"],
            }

        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            mock_call_claude,
        )

        result = evaluate_bets(sample_bets)
        assert len(result) == 1
        assert result.iloc[0]["bet_on"] == "Fighter Alpha"

    def test_empty_bets(self, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        result = evaluate_bets(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Grok social layer
# ---------------------------------------------------------------------------

class TestGrokSocial:
    def test_skipped_without_key(self, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "")
        from src.strategy.llm_operator import _query_grok_social

        result = _query_grok_social("Alpha", "Beta")
        assert isinstance(result, GrokIntel)
        assert result.skipped is True

    def test_caching(self, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GROK_API_KEY", "fake-key")
        from src.strategy.llm_operator import _GROK_CACHE, _query_grok_social
        import time

        # Pre-populate cache with a GrokIntel object
        cached_intel = GrokIntel(
            overall_signal="cached data",
            data_quality="rich",
        )
        _GROK_CACHE["Alpha|Beta"] = (time.time(), cached_intel)

        result = _query_grok_social("Alpha", "Beta")
        assert isinstance(result, GrokIntel)
        assert result.overall_signal == "cached data"

        # Clean up
        _GROK_CACHE.pop("Alpha|Beta", None)

    def test_parse_structured_response(self):
        raw = json.dumps({
            "injury_reports": [
                {"fighter": "Alpha", "detail": "Knee injury reported", "source_type": "journalist", "confidence": "rumored"}
            ],
            "camp_intel": [],
            "betting_sentiment": {"direction": "fighter_b", "summary": "Sharp money on Beta"},
            "insider_reports": [
                {"detail": "Alpha changed gyms 3 weeks ago", "source_type": "journalist", "credibility": "high"}
            ],
            "weight_cut_flags": [],
            "overall_signal": "Alpha has a rumored knee injury and recently changed gyms",
            "data_quality": "rich",
        })
        intel = _parse_grok_response(raw, "Alpha", "Beta")
        assert intel.data_quality == "rich"
        assert len(intel.injury_reports) == 1
        assert intel.injury_reports[0]["confidence"] == "rumored"
        assert intel.betting_sentiment["direction"] == "fighter_b"
        assert len(intel.insider_reports) == 1
        assert intel.overall_signal != ""

    def test_parse_invalid_json_falls_back(self):
        raw = "No relevant tweets found for this matchup."
        intel = _parse_grok_response(raw, "Alpha", "Beta")
        assert intel.data_quality == "sparse"
        assert intel.raw_response == raw

    def test_parse_empty_response(self):
        intel = _parse_grok_response("", "Alpha", "Beta")
        assert intel.data_quality == "none"
