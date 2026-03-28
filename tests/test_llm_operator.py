"""Tests for the LLM Operator decision-making pipeline."""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.strategy.llm_operator import (
    OPERATOR_DIR,
    OperatorDecision,
    ResearchFindings,
    _analyze_matchup_from_features,
    _build_synthesis_prompt,
    _check_blind_spots,
    _check_correlated_exposure,
    _check_motivation_signals,
    _check_recency_context,
    clear_decision_cache,
    evaluate_bet,
    evaluate_bets,
    load_blind_spots,
    run_research_pipeline,
    save_blind_spots,
)


@pytest.fixture(autouse=True)
def _clear_operator_cache():
    """Clear the operator decision cache before each test."""
    clear_decision_cache()


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

    def test_fractional_rates_render_as_percentages(self, sample_features):
        analysis = _analyze_matchup_from_features(sample_features, "Alpha", "Beta")
        assert "4.5 SLpM (48% acc)" in analysis
        assert "3.5/fight (45% acc, 70% def)" in analysis
        assert "Alpha finishes: KO 40%, Sub 20%, Dec 40%" in analysis
        assert "Beta finishes: KO 60%, Sub 10%, Dec 30%" in analysis


class TestPromptFormatting:
    def test_fighter_profile_narrative_format(self, sample_features):
        prompt = _build_synthesis_prompt(
            fighter_a="Alpha",
            fighter_b="Beta",
            bet_on="Alpha",
            bet_side="a",
            model_prob=0.65,
            market_prob=0.50,
            blended_prob=0.58,
            edge=0.08,
            features=sample_features,
            findings=ResearchFindings(),
            weight_class="Welterweight",
        )
        # Narrative should describe fighters, not dump a stat sheet
        assert "What the Model Sees" in prompt
        assert "**Alpha:**" in prompt
        assert "**Beta:**" in prompt
        # Compact stat reference block should still have numbers for echo check
        assert "Stat Reference" in prompt
        assert "str_acc=" in prompt
        assert "td_acc=" in prompt
        assert "td_def=" in prompt


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


# ---------------------------------------------------------------------------
# evaluate_bet — with mocked LLM synthesis
# ---------------------------------------------------------------------------

class TestEvaluateBet:
    def test_evaluate_with_mocked_llm_synthesis(self, sample_features, tmp_path, monkeypatch):

        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
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

    def test_skips_llm_when_fight_already_has_recorded_bet(
        self,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        log_path = tmp_path / "decision_log.jsonl"
        monkeypatch.setattr("src.strategy.llm_operator.DECISION_LOG_PATH", log_path)
        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt: pytest.fail("LLM should not run for an already-bet fight"),
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
            event_date="2026-04-01",
            existing_bets=[
                {
                    "fighter": "Fighter Alpha",
                    "opponent": "Fighter Beta",
                    "event_date": "2026-04-01",
                }
            ],
        )

        assert decision.verdict == "PASS"
        assert "already has a recorded bet/order" in decision.rationale.lower()
        assert not log_path.exists()

    def test_existing_bet_skip_does_not_block_different_event_date(
        self,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )

        call_count = [0]

        def _mock_call(_prompt):
            call_count[0] += 1
            return {
                "verdict": "BLOCK",
                "rationale": "Different event date, so this should be evaluated normally",
                "fighter_assessment": "Rematch or separate booking",
                "risk_flags": ["evaluated_normally"],
            }

        monkeypatch.setattr("src.strategy.llm_operator._call_llm_synthesis", _mock_call)

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
            event_date="2026-06-01",
            existing_bets=[
                {
                    "fighter": "Fighter Alpha",
                    "opponent": "Fighter Beta",
                    "event_date": "2026-04-01",
                }
            ],
        )

        assert call_count[0] == 1
        assert decision.verdict == "BLOCK"


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

        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake")
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

        def mock_call_llm(prompt):
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
            mock_call_llm,
        )

        result = evaluate_bets(sample_bets)
        assert len(result) == 1
        assert result.iloc[0]["bet_on"] == "Fighter Alpha"

    def test_empty_bets(self, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        result = evaluate_bets(pd.DataFrame())
        assert result.empty

    def test_existing_bet_short_circuits_llm_but_keeps_row(
        self,
        sample_bets,
        monkeypatch,
    ):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "gate")
        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt: pytest.fail("LLM should not run for an already-bet fight"),
        )

        result = evaluate_bets(
            sample_bets.iloc[[0]],
            existing_bets=[
                {
                    "fighter": "Fighter Alpha",
                    "opponent": "Fighter Beta",
                    "event_date": "2026-04-01",
                }
            ],
        )

        assert len(result) == 1
        assert result.iloc[0]["operator_verdict"] == "PASS"
        assert "recorded bet/order" in result.iloc[0]["operator_rationale"].lower()
        assert "existing_bet" in result.iloc[0]["operator_risk_flags"]

    def test_evaluate_bets_reports_progress(self, sample_bets, monkeypatch):
        messages = []

        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "gate")
        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt: {
                "verdict": "PASS",
                "rationale": "Looks fine",
                "fighter_assessment": "No veto",
                "risk_flags": [],
            },
        )

        result = evaluate_bets(
            sample_bets.iloc[[0]],
            progress_callback=messages.append,
            progress_label="value bets",
        )

        assert len(result) == 1
        assert messages == [
            "Cycle active: operator evaluating value bets 1/1: Fighter Alpha vs Fighter Beta"
        ]

    def test_enabled_operator_logs_runtime_provenance(
        self,
        sample_bets,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "gate")
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt: {
                "verdict": "PASS",
                "rationale": "Runtime provenance looks sane",
                "fighter_assessment": "No veto flags",
                "risk_flags": [],
            },
        )

        evaluate_bets(
            sample_bets.iloc[[0]],
            features_by_fight={"Fighter Alpha|Fighter Beta": sample_features},
            provenance_by_fight={
                "Fighter Alpha|Fighter Beta": {
                    "bundle_id": "bundle-1",
                    "model_spec_name": "prod_spec",
                    "processed_snapshot_max_event_date": "2026-03-21",
                    "fighter_a_source": "processed",
                    "fighter_b_source": "ufcstats",
                }
            },
        )

        log_path = tmp_path / "decision_log.jsonl"
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["provenance"]["bundle_id"] == "bundle-1"

    def test_advisory_mode_preserves_blocked_rows(self, sample_bets, tmp_path, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "advisory")
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake")
        monkeypatch.setattr(
            "src.strategy.llm_operator.DECISION_LOG_PATH",
            tmp_path / "decision_log.jsonl",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt: {
                "verdict": "BLOCK",
                "rationale": "Keep the row, but annotate it in advisory mode",
                "fighter_assessment": "Flagged but retained",
                "risk_flags": ["advisory_flag"],
            },
        )

        result = evaluate_bets(sample_bets.iloc[[0]])

        assert len(result) == 1
        assert result.iloc[0]["operator_verdict"] == "BLOCK"
        assert "advisory" in result.iloc[0]["operator_rationale"].lower()
