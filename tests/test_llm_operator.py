"""Tests for the LLM Operator decision-making pipeline."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.strategy import llm_operator
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

    def test_layoff_uses_exact_event_date_when_available(self):
        features = {"a_days_since_last_fight": 686}
        flags = _check_recency_context(
            features,
            "Alpha",
            "Beta",
            event_date="2026-04-04",
        )
        expected_last_fight = (
            pd.Timestamp("2026-04-04") - pd.Timedelta(days=686)
        ).strftime("%B %d, %Y")
        assert any(expected_last_fight in f for f in flags)
        assert any("returns on April 04, 2026" in f for f in flags)

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
            event_date="2026-04-04",
        )
        # Narrative should describe fighters, not dump a stat sheet
        assert "What the Model Sees" in prompt
        assert "**Alpha:**" in prompt
        assert "**Beta:**" in prompt
        assert "Date Anchor" in prompt
        assert "scheduled for April 04, 2026" in prompt
        assert "Treat layoff math as anchored to the scheduled fight date" in prompt
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

    def test_operator_prompt_keeps_block_threshold_high(self):
        system_prompt = llm_operator._build_operator_synthesis_system_prompt()
        task_prompt = llm_operator._build_operator_synthesis_prompt_from_research(
            "base prompt",
            {
                "fight_status": "upcoming",
                "memo_text": "Research memo.",
                "model_pick_concerns": "Opponent has one viable path.",
            },
        )

        assert "Default to PASS" in system_prompt
        assert "Do not block just because the opponent has a viable path to victory" in system_prompt
        assert "Richer research should improve explanation quality, not lower the BLOCK threshold" in system_prompt
        assert "model_read_concerns but should still PASS" in task_prompt

    def test_operator_research_cache_key_is_model_pick_scoped(self):
        base_key = "2026-05-16|alpha|beta"

        assert (
            llm_operator._operator_research_cache_key(base_key, "Fighter Alpha")
            == "2026-05-16|alpha|beta|model_pick:fighter alpha"
        )
        assert llm_operator._operator_research_cache_key(base_key, "") == base_key

    def test_synthesis_prompt_includes_local_matchup_and_model_signals(self, sample_features):
        features = dict(sample_features)
        features.update(
            {
                "a_opp_strength": 0.71,
                "b_opp_strength": 0.54,
                "a_striker_edge": 0.18,
                "b_striker_edge": 0.09,
                "a_grappler_edge": 0.12,
                "b_grappler_edge": 0.21,
                "a_wc_rank_feat": 8,
                "b_wc_rank_feat": 16,
                "a_pre_ufc_org_tier_best": 2,
                "b_pre_ufc_org_tier_best": 4,
                "diff_opp_strength": 0.17,
                "diff_striker_edge": 0.09,
                "diff_grappler_edge": -0.09,
            }
        )
        findings = run_research_pipeline(
            features=features,
            fighter_a="Alpha",
            fighter_b="Beta",
            model_prob_a=0.65,
            market_prob_a=0.50,
        )

        prompt = _build_synthesis_prompt(
            fighter_a="Alpha",
            fighter_b="Beta",
            bet_on="Alpha",
            bet_side="a",
            model_prob=0.65,
            market_prob=0.50,
            blended_prob=0.58,
            edge=0.08,
            features=features,
            findings=findings,
        )

        assert "## Local Matchup Analysis" in prompt
        assert findings.matchup_analysis in prompt
        assert "## Model Matchup Signals" in prompt
        assert "Recent opponent strength: Alpha 0.71 vs Beta 0.54" in prompt
        assert "Striker-edge interaction: Alpha 0.18 vs Beta 0.09" in prompt
        assert "Grappler-edge differential: -0.09" in prompt


class TestGeminiJsonParsing:
    def test_configured_gemini_models_skips_retired_models(self, monkeypatch):
        monkeypatch.setattr(llm_operator, "GEMINI_MODEL", "gemini-3.1-pro-preview")
        monkeypatch.setattr(
            llm_operator,
            "GEMINI_FALLBACK_MODELS",
            (
                "gemini-3-pro-preview",
                "gemini-3.5-flash",
                "gemini-3.1-pro-preview",
                "gemini-2.5-pro",
            ),
        )

        assert llm_operator._configured_gemini_models() == [
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "gemini-2.5-pro",
        ]

    def test_gemini_cancelled_errors_are_treated_as_transient(self):
        assert llm_operator._is_gemini_transient_error(RuntimeError("499 CANCELLED"))

    def test_extract_gemini_grounding_sources_handles_dict_and_camel_case_payloads(self):
        response = {
            "candidates": [
                {
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/fight"}},
                            {"web": {"uri": "https://example.com/fight"}},
                            {
                                "retrievedContext": {
                                    "uri": "https://example.com/context"
                                }
                            },
                        ]
                    }
                },
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(uri="https://example.com/second"),
                            )
                        ]
                    )
                ),
            ]
        }

        assert llm_operator._extract_gemini_grounding_sources(response) == [
            "https://example.com/fight",
            "https://example.com/context",
            "https://example.com/second",
        ]

    def test_parse_grounded_research_response_extracts_matchup_sections(self):
        raw = (
            "FIGHT STATUS:\n"
            "upcoming\n"
            "RESEARCH MEMO:\n"
            "The matchup is live and both fighters are booked.\n"
            "RECENT FORM:\n"
            "Alpha beat two UFC veterans and lost to a ranked contender.\n"
            "LEVEL OF COMPETITION:\n"
            "Alpha has faced stronger recent UFC opposition.\n"
            "STYLE MATCHUP:\n"
            "Alpha is the cleaner striker; Beta has the wrestling threat.\n"
            "PATHS TO VICTORY:\n"
            "Alpha wins at range. Beta wins with control time.\n"
            "CONCERNS FOR MODEL PICK:\n"
            "Beta's wrestling could stress Alpha's takedown defense.\n"
            "VERIFIED RECORDS:\n"
            "fighter_a: 10-2-0\n"
            "fighter_b: 9-3-0\n"
            "fighter_a_ranking: unranked\n"
            "fighter_b_ranking: unranked\n"
            "source: Sherdog\n"
            "KEY FLAGS:\n"
            "- wrestling swing factor\n"
        )

        parsed = llm_operator._parse_grounded_research_response(raw)

        assert parsed["fight_status"] == "upcoming"
        assert parsed["recent_form"].startswith("Alpha beat")
        assert parsed["level_of_competition"].startswith("Alpha has faced")
        assert parsed["style_matchup"].startswith("Alpha is the cleaner striker")
        assert parsed["paths_to_victory"].startswith("Alpha wins")
        assert parsed["model_pick_concerns"].startswith("Beta's wrestling")
        assert parsed["verified_records"]["fighter_a"] == "10-2-0"
        assert parsed["key_flags"] == ["wrestling swing factor"]

    def test_parse_gemini_json_response_extracts_nested_payload_from_wrapper_text(self):
        payload = {
            "stats_confirmed": {
                "fighter_a_str_acc": 51,
                "fighter_a_td_acc": 20,
                "fighter_a_td_def": 43,
                "fighter_b_str_acc": 38,
                "fighter_b_td_acc": 47,
                "fighter_b_td_def": 59,
            },
            "verified_records": {
                "fighter_a": "22-6-0",
                "fighter_b": "13-3-0",
                "fighter_a_ranking": "unranked",
                "fighter_b_ranking": "unranked",
                "source": "Sherdog",
            },
            "verdict": "PASS",
            "rationale": "The matchup looks correctly framed.",
            "fighter_assessment": "No obvious veto concerns.",
            "risk_flags": [],
        }
        raw = f"Research complete.\n```json\n{json.dumps(payload)}\n```\nUse the JSON above."

        parsed = llm_operator._parse_gemini_json_response(
            raw,
            fallback_json_key="verdict",
        )

        assert parsed["verdict"] == "PASS"
        assert parsed["verified_records"]["fighter_a"] == "22-6-0"
        assert parsed["stats_confirmed"]["fighter_b_td_def"] == 59

    def test_call_gemini_synthesis_from_research_falls_back_after_malformed_primary_response(self, monkeypatch):
        primary_response = MagicMock()
        primary_response.text = '{"verdict":"PASS","risk_flags":'
        primary_response.candidates = []

        fallback_payload = {
            "verdict": "PASS",
            "rationale": "Fallback model returned valid JSON.",
            "fighter_assessment": "",
            "risk_flags": [],
        }
        fallback_response = MagicMock()
        fallback_response.text = json.dumps(fallback_payload)
        fallback_response.candidates = []

        client = MagicMock()
        client.models.generate_content.side_effect = [
            primary_response,
            fallback_response,
        ]

        monkeypatch.setattr(llm_operator, "_get_gemini_client", lambda *args, **kwargs: client)
        monkeypatch.setattr(
            llm_operator,
            "_configured_gemini_models",
            lambda: ["gemini-primary", "gemini-fallback"],
        )
        monkeypatch.setattr(llm_operator, "GEMINI_MODEL", "gemini-primary")

        parsed, telemetry = llm_operator._call_gemini_synthesis_from_research(
            "test prompt",
            system_instruction="system",
            response_json_schema={"type": "object"},
            fallback_json_key="verdict",
            success_log_label="Gemini operator synthesis",
            _max_retries=1,
        )

        assert parsed == fallback_payload
        assert telemetry["model_used"] == "gemini-fallback"
        assert telemetry["fallback_reached"] is True
        assert telemetry["schema_parse_success"] is True
        assert "tools" not in client.models.generate_content.call_args.kwargs["config"]
        assert client.models.generate_content.call_count == 2

    def test_call_gemini_synthesis_does_not_lower_thinking_for_gemini_3(self, monkeypatch):
        payload = {
            "verdict": "PASS",
            "rationale": "Schema synthesis completed.",
            "fighter_assessment": "",
            "risk_flags": [],
        }
        response = MagicMock()
        response.text = json.dumps(payload)
        response.candidates = []

        client = MagicMock()
        client.models.generate_content.return_value = response

        monkeypatch.setattr(llm_operator, "_get_gemini_client", lambda *args, **kwargs: client)
        monkeypatch.setattr(
            llm_operator,
            "_configured_gemini_models",
            lambda: ["gemini-3.1-pro-preview"],
        )
        monkeypatch.setattr(llm_operator, "GEMINI_MODEL", "gemini-3.1-pro-preview")

        parsed, telemetry = llm_operator._call_gemini_synthesis_from_research(
            "test prompt",
            system_instruction="system",
            response_json_schema={"type": "object"},
            fallback_json_key="verdict",
            success_log_label="Gemini operator synthesis",
            _max_retries=1,
        )

        assert parsed == payload
        assert telemetry["model_used"] == "gemini-3.1-pro-preview"
        config = client.models.generate_content.call_args.kwargs["config"]
        assert "thinking_config" not in config
        assert "temperature" not in config
        assert "tools" not in config

    def test_call_gemini_synthesis_fails_over_after_configured_transient_primary_retries(
        self,
        monkeypatch,
    ):
        fallback_payload = {
            "verdict": "PASS",
            "rationale": "Fallback model returned valid JSON.",
            "fighter_assessment": "",
            "risk_flags": [],
        }
        fallback_response = MagicMock()
        fallback_response.text = json.dumps(fallback_payload)
        fallback_response.candidates = []

        client = MagicMock()
        client.models.generate_content.side_effect = [
            RuntimeError("503 UNAVAILABLE"),
            RuntimeError("504 DEADLINE_EXCEEDED"),
            fallback_response,
        ]

        monkeypatch.setattr(llm_operator, "_get_gemini_client", lambda *args, **kwargs: client)
        monkeypatch.setattr(
            llm_operator,
            "_configured_gemini_models",
            lambda: ["gemini-primary", "gemini-fallback"],
        )
        monkeypatch.setattr(llm_operator, "GEMINI_MODEL", "gemini-primary")
        monkeypatch.setattr(llm_operator, "GEMINI_PRIMARY_MODEL_RETRIES", 2)
        monkeypatch.setattr(llm_operator, "GEMINI_FALLBACK_RETRIES_PER_MODEL", 1)
        monkeypatch.setattr(llm_operator.time, "sleep", lambda _seconds: None)

        parsed, telemetry = llm_operator._call_gemini_synthesis_from_research(
            "test prompt",
            system_instruction="system",
            response_json_schema={"type": "object"},
            fallback_json_key="verdict",
            success_log_label="Gemini operator synthesis",
        )

        assert parsed == fallback_payload
        assert telemetry["models_attempted"] == ["gemini-primary", "gemini-fallback"]
        assert telemetry["model_used"] == "gemini-fallback"
        assert client.models.generate_content.call_count == 3
        assert [
            call.kwargs["model"]
            for call in client.models.generate_content.call_args_list
        ] == ["gemini-primary", "gemini-primary", "gemini-fallback"]

    def test_call_gemini_research_does_not_lower_thinking_for_gemini_3(self, monkeypatch, tmp_path):
        response = MagicMock()
        response.text = (
            "FIGHT STATUS:\n"
            "upcoming\n"
            "RESEARCH MEMO:\n"
            "Compact grounded memo.\n"
            "VERIFIED RECORDS:\n"
            "fighter_a: 10-1-0\n"
            "fighter_b: 9-2-0\n"
            "fighter_a_ranking: unranked\n"
            "fighter_b_ranking: unranked\n"
            "source: Sherdog\n"
            "KEY FLAGS:\n"
            "- none"
        )
        response.candidates = [
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(uri="https://example.com/fight"),
                        )
                    ]
                )
            )
        ]

        client = MagicMock()
        client.models.generate_content.return_value = response
        captured = {}

        def _mock_get_client(timeout_ms=None):
            captured["timeout_ms"] = timeout_ms
            return client

        monkeypatch.setattr(llm_operator, "_get_gemini_client", _mock_get_client)
        monkeypatch.setattr(
            llm_operator,
            "_GEMINI_RESEARCH_CACHE_FILE",
            tmp_path / "gemini_research_cache.json",
        )
        monkeypatch.setattr(llm_operator, "GEMINI_RESEARCH_TIMEOUT_MS", 60000)
        monkeypatch.setattr(
            llm_operator,
            "_configured_gemini_models",
            lambda: ["gemini-3.1-pro-preview"],
        )
        monkeypatch.setattr(llm_operator, "GEMINI_MODEL", "gemini-3.1-pro-preview")

        result, telemetry = llm_operator._call_gemini_research(
            "prompt",
            cache_key="2026-04-19|alpha|beta|thinking",
            success_log_label="Gemini operator research",
            _max_retries=1,
        )

        assert result is not None
        assert telemetry["model_used"] == "gemini-3.1-pro-preview"
        assert captured["timeout_ms"] == 60000
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config["tools"] == [{"google_search": {}}]
        assert "thinking_config" not in config
        assert "temperature" not in config

    def test_call_gemini_research_uses_short_ttl_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            llm_operator,
            "_GEMINI_RESEARCH_CACHE_FILE",
            tmp_path / "gemini_research_cache.json",
        )
        monkeypatch.setattr(llm_operator, "GEMINI_RESEARCH_CACHE_TTL_SECONDS", 900.0)

        call_count = [0]

        def _mock_stage(*args, **kwargs):
            call_count[0] += 1
            return (
                {
                    "fight_status": "upcoming",
                    "memo_text": "Compact grounded memo.",
                    "verified_records": {
                        "fighter_a": "10-1-0",
                        "fighter_b": "9-2-0",
                        "fighter_a_ranking": "unranked",
                        "fighter_b_ranking": "unranked",
                        "source": "Sherdog",
                    },
                    "key_flags": ["none"],
                    "raw_text": "FIGHT STATUS:\nupcoming",
                },
                ["https://example.com/fight"],
                {
                    "models_attempted": ["gemini-primary"],
                    "model_used": "gemini-primary",
                    "fallback_reached": False,
                    "search_enabled": True,
                    "search_success": True,
                    "schema_mode": False,
                    "schema_parse_success": None,
                    "failure_class": "",
                },
            )

        monkeypatch.setattr(llm_operator, "_call_gemini_stage", _mock_stage)

        first, first_telemetry = llm_operator._call_gemini_research(
            "prompt",
            cache_key="2026-04-19|alpha|beta",
            success_log_label="Gemini operator research",
        )
        second, second_telemetry = llm_operator._call_gemini_research(
            "prompt",
            cache_key="2026-04-19|alpha|beta",
            success_log_label="Gemini operator research",
        )

        assert call_count[0] == 1
        assert first["memo_text"] == second["memo_text"]
        assert second["cached"] is True
        assert first_telemetry["cached"] is False
        assert second_telemetry["cached"] is True


class TestOperatorFailureCaching:
    def test_llm_unavailable_cache_entries_expire_on_short_ttl(self, monkeypatch):
        monkeypatch.setattr(
            llm_operator,
            "LLM_OPERATOR_FAILURE_CACHE_TTL_SECONDS",
            60.0,
        )
        decision = OperatorDecision(
            verdict="PASS",
            confidence=1.0,
            model_prob=0.6,
            operator_prob=0.6,
            rationale="Operator passthrough: Gemini failed after retries",
            research_summary={},
            risk_flags=["llm_unavailable", "llm_failed_after_retries"],
            timestamp="2026-04-07T18:00:00+00:00",
            fighter_a="Alpha",
            fighter_b="Beta",
            bet_on="Alpha",
            bet_side="a",
            edge=0.05,
            market_prob=0.55,
            event_date="2026-04-19T00:00:00+00:00",
            event_title="UFC Test",
            decision_key="2026-04-19|alpha|beta",
            provenance={},
        )

        assert llm_operator._decision_cache_is_fresh(decision, cached_at=150.0, now=200.0)
        assert not llm_operator._decision_cache_is_fresh(decision, cached_at=100.0, now=200.0)


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
            lambda prompt, **_: mock_synthesis_result,
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

    def test_stats_correction_retry_reuses_grounded_research(
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
        monkeypatch.setattr(
            "src.strategy.llm_operator.BLIND_SPOTS_PATH",
            tmp_path / "blind_spots.json",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator._DECISION_CACHE_FILE",
            tmp_path / "decision_cache.json",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator._GEMINI_RESEARCH_CACHE_FILE",
            tmp_path / "gemini_research_cache.json",
        )

        research_calls = [0]
        synthesis_prompts = []

        def _mock_research(prompt, *, cache_key, success_log_label, _max_retries=None):
            research_calls[0] += 1
            return (
                {
                    "fight_status": "upcoming",
                    "memo_text": "Grounded memo says Alpha is returning from a long layoff.",
                    "verified_records": {
                        "fighter_a": "12-1-0",
                        "fighter_b": "10-3-0",
                        "fighter_a_ranking": "unranked",
                        "fighter_b_ranking": "unranked",
                        "source": "Sherdog",
                    },
                    "key_flags": ["long layoff"],
                    "sources": ["https://example.com/fight"],
                    "model_used": "gemini-3.1-pro-preview",
                    "cached": False,
                    "failure_class": "",
                },
                {
                    "models_attempted": ["gemini-3.1-pro-preview"],
                    "model_used": "gemini-3.1-pro-preview",
                    "fallback_reached": False,
                    "search_enabled": True,
                    "search_success": True,
                    "schema_mode": False,
                    "schema_parse_success": None,
                    "failure_class": "",
                    "cached": False,
                },
            )

        def _mock_synthesis(
            prompt,
            *,
            system_instruction,
            response_json_schema,
            fallback_json_key,
            success_log_label,
            _max_retries=None,
        ):
            synthesis_prompts.append(prompt)
            if len(synthesis_prompts) == 1:
                return (
                    {
                        "stats_confirmed": {
                            "fighter_a_str_acc": 99,
                            "fighter_a_td_acc": 99,
                            "fighter_a_td_def": 99,
                            "fighter_b_str_acc": 99,
                            "fighter_b_td_acc": 99,
                            "fighter_b_td_def": 99,
                        },
                        "verified_records": {
                            "fighter_a": "12-1-0",
                            "fighter_b": "10-3-0",
                            "fighter_a_ranking": "unranked",
                            "fighter_b_ranking": "unranked",
                            "source": "Sherdog",
                        },
                        "verdict": "BLOCK",
                        "rationale": "Initial synthesis misread the stats.",
                        "fighter_assessment": "Needs retry.",
                        "risk_flags": ["initial_read"],
                    },
                    {
                        "models_attempted": ["gemini-3.1-pro-preview"],
                        "model_used": "gemini-3.1-pro-preview",
                        "fallback_reached": False,
                        "search_enabled": False,
                        "search_success": None,
                        "schema_mode": True,
                        "schema_parse_success": True,
                        "failure_class": "",
                    },
                )
            return (
                {
                    "stats_confirmed": {
                        "fighter_a_str_acc": 0.48,
                        "fighter_a_td_acc": 0.45,
                        "fighter_a_td_def": 0.70,
                        "fighter_b_str_acc": 0.42,
                        "fighter_b_td_acc": 0.30,
                        "fighter_b_td_def": 0.45,
                    },
                    "verified_records": {
                        "fighter_a": "12-1-0",
                        "fighter_b": "10-3-0",
                        "fighter_a_ranking": "unranked",
                        "fighter_b_ranking": "unranked",
                        "source": "Sherdog",
                    },
                    "verdict": "BLOCK",
                    "rationale": "Corrected synthesis reused the existing research.",
                    "fighter_assessment": "Alpha still carries the bigger contextual risk.",
                    "risk_flags": ["long_layoff"],
                },
                {
                    "models_attempted": ["gemini-3-pro-preview"],
                    "model_used": "gemini-3-pro-preview",
                    "fallback_reached": True,
                    "search_enabled": False,
                    "search_success": None,
                    "schema_mode": True,
                    "schema_parse_success": True,
                    "failure_class": "",
                },
            )

        monkeypatch.setattr(llm_operator, "_call_gemini_research", _mock_research)
        monkeypatch.setattr(llm_operator, "_call_gemini_synthesis_from_research", _mock_synthesis)

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
            event_date="2026-04-19",
        )

        assert research_calls[0] == 1
        assert len(synthesis_prompts) == 2
        assert "Grounded memo says Alpha is returning from a long layoff." in synthesis_prompts[0]
        assert "CORRECTION" in synthesis_prompts[1]
        assert decision.verdict == "BLOCK"
        assert "stats_corrected_retry" in decision.risk_flags
        assert decision.research_summary["grounded_research"]["memo_text"].startswith("Grounded memo")
        assert (
            decision.provenance["llm_stage_telemetry"]["research"]["model_used"]
            == "gemini-3.1-pro-preview"
        )
        assert (
            decision.provenance["llm_stage_telemetry"]["synthesis"]["model_used"]
            == "gemini-3-pro-preview"
        )

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

    def test_standalone_pick_uses_staged_research_and_synthesis(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(
            "src.strategy.llm_operator._GEMINI_PICK_CACHE_FILE",
            tmp_path / "gemini_pick_cache.json",
        )
        monkeypatch.setattr(
            "src.strategy.llm_operator._GEMINI_RESEARCH_CACHE_FILE",
            tmp_path / "gemini_research_cache.json",
        )

        monkeypatch.setattr(
            llm_operator,
            "_call_gemini_research",
            lambda prompt, *, cache_key, success_log_label, _max_retries=None: (
                {
                    "fight_status": "upcoming",
                    "memo_text": "Grounded memo favors Alpha's pace and recent activity.",
                    "verified_records": {
                        "fighter_a": "12-1-0",
                        "fighter_b": "10-3-0",
                        "source": "Sherdog",
                    },
                    "key_flags": ["recent form edge"],
                    "sources": ["https://example.com/fight"],
                    "model_used": "gemini-3.1-pro-preview",
                    "cached": False,
                    "failure_class": "",
                },
                {
                    "models_attempted": ["gemini-3.1-pro-preview"],
                    "model_used": "gemini-3.1-pro-preview",
                    "fallback_reached": False,
                    "search_enabled": True,
                    "search_success": True,
                    "schema_mode": False,
                    "schema_parse_success": None,
                    "failure_class": "",
                    "cached": False,
                },
            ),
        )
        monkeypatch.setattr(
            llm_operator,
            "_call_gemini_synthesis_from_research",
            lambda prompt, *, system_instruction, response_json_schema, fallback_json_key, success_log_label, _max_retries=None: (
                {
                    "pick": "Fighter Alpha",
                    "confidence": 0.64,
                    "rationale": "Alpha's recent activity and pressure look more dependable.",
                    "fighter_assessment": "Alpha has the cleaner recent form.",
                    "risk_flags": ["recent_form_edge"],
                    "verified_records": {
                        "fighter_a": "12-1-0",
                        "fighter_b": "10-3-0",
                        "source": "Sherdog",
                    },
                },
                {
                    "models_attempted": ["gemini-3-flash-preview"],
                    "model_used": "gemini-3-flash-preview",
                    "fallback_reached": True,
                    "search_enabled": False,
                    "search_success": None,
                    "schema_mode": True,
                    "schema_parse_success": True,
                    "failure_class": "",
                },
            ),
        )

        pick = llm_operator.gemini_standalone_pick(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            weight_class="Welterweight",
            event_date="2026-04-19",
            event_title="UFC Test",
        )

        assert pick["pick"] == "Fighter Alpha"
        assert pick["sources"] == ["https://example.com/fight"]
        assert pick["verified_records"]["source"] == "Sherdog"
        assert pick["stage_telemetry"]["research"]["model_used"] == "gemini-3.1-pro-preview"
        assert pick["stage_telemetry"]["synthesis"]["model_used"] == "gemini-3-flash-preview"

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
            lambda _prompt, **_: pytest.fail("LLM should not run for an already-bet fight"),
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

        def _mock_call(_prompt, **_):
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

    def test_reuses_persisted_cache_when_memory_cache_is_empty(
        self,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        cache_path = tmp_path / "decision_cache.json"
        log_path = tmp_path / "decision_log.jsonl"
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr("src.strategy.llm_operator.DECISION_LOG_PATH", log_path)
        monkeypatch.setattr("src.strategy.llm_operator._DECISION_CACHE_FILE", cache_path)
        monkeypatch.setattr("src.strategy.llm_operator.CACHE_TTL_SECONDS", 0.0)

        call_count = [0]

        def _mock_call(_prompt, **_):
            call_count[0] += 1
            return {
                "verdict": "PASS",
                "rationale": "One stable decision per booked fight",
                "fighter_assessment": "Cached decision",
                "risk_flags": ["stable_cache"],
            }

        monkeypatch.setattr("src.strategy.llm_operator._call_llm_synthesis", _mock_call)

        first = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            features=sample_features,
            event_date="2027-04-01T23:00:00Z",
        )

        assert call_count[0] == 1
        assert cache_path.exists()

        llm_operator._decision_cache.clear()

        monkeypatch.setattr(
            "src.strategy.llm_operator._call_llm_synthesis",
            lambda _prompt, **_: pytest.fail("Persisted cache should satisfy the second lookup"),
        )

        second = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.43,
            edge=0.15,
            features=sample_features,
            event_date="2027-04-01T23:00:00Z",
        )

        assert second.verdict == first.verdict
        assert second.rationale == first.rationale
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1

    def test_reuses_fight_cache_across_trader_contexts(
        self,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        cache_path = tmp_path / "decision_cache.json"
        log_path = tmp_path / "decision_log.jsonl"
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr("src.strategy.llm_operator.DECISION_LOG_PATH", log_path)
        monkeypatch.setattr("src.strategy.llm_operator._DECISION_CACHE_FILE", cache_path)

        call_count = [0]

        def _mock_call(_prompt, **_):
            call_count[0] += 1
            return {
                "verdict": "PASS",
                "rationale": "One operator read per fight, shared by S and C",
                "fighter_assessment": "Shared fight-level decision",
                "risk_flags": [],
            }

        monkeypatch.setattr("src.strategy.llm_operator._call_llm_synthesis", _mock_call)

        first = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.03,
            features=sample_features,
            event_date="2027-04-01",
            decision_context="S",
        )
        second = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.15,
            features=sample_features,
            event_date="2027-04-01",
            decision_context="C",
        )

        assert call_count[0] == 1
        assert second.rationale == first.rationale
        assert second.decision_key == "2027-04-01|fighter alpha|fighter beta"
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1

    def test_cache_is_scoped_by_event_date(
        self,
        sample_features,
        tmp_path,
        monkeypatch,
    ):
        cache_path = tmp_path / "decision_cache.json"
        monkeypatch.setattr("src.strategy.llm_operator.GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr("src.strategy.llm_operator._DECISION_CACHE_FILE", cache_path)

        call_count = [0]

        def _mock_call(_prompt, **_):
            call_count[0] += 1
            return {
                "verdict": "PASS" if call_count[0] == 1 else "BLOCK",
                "rationale": f"decision-{call_count[0]}",
                "fighter_assessment": "Event-scoped cache",
                "risk_flags": [],
            }

        monkeypatch.setattr("src.strategy.llm_operator._call_llm_synthesis", _mock_call)

        first = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            features=sample_features,
            event_date="2027-04-01",
        )
        cached = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.44,
            edge=0.14,
            features=sample_features,
            event_date="2027-04-01",
        )
        rematch = evaluate_bet(
            fighter_a="Fighter Alpha",
            fighter_b="Fighter Beta",
            bet_on="Fighter Alpha",
            bet_side="a",
            model_prob=0.65,
            blended_prob=0.58,
            market_prob=0.50,
            edge=0.08,
            features=sample_features,
            event_date="2027-06-01",
        )

        assert call_count[0] == 2
        assert cached.rationale == first.rationale
        assert rematch.rationale != first.rationale


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

        def mock_call_llm(prompt, **_):
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
            lambda _prompt, **_: pytest.fail("LLM should not run for an already-bet fight"),
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
            lambda _prompt, **_: {
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

    def test_evaluate_bets_dedupes_same_fight_within_batch(
        self,
        sample_bets,
        monkeypatch,
    ):
        duplicated = pd.concat([sample_bets.iloc[[0]], sample_bets.iloc[[0]]], ignore_index=True)
        messages = []
        call_count = [0]

        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_ENABLED", True)
        monkeypatch.setattr("src.strategy.llm_operator.OPERATOR_MODE", "gate")

        def _mock_call(_prompt, **_):
            call_count[0] += 1
            return {
                "verdict": "PASS",
                "rationale": "Reuse the same fight decision inside one batch",
                "fighter_assessment": "Deduped",
                "risk_flags": [],
            }

        monkeypatch.setattr("src.strategy.llm_operator._call_llm_synthesis", _mock_call)

        result = evaluate_bets(
            duplicated,
            progress_callback=messages.append,
            progress_label="value bets",
        )

        assert len(result) == 2
        assert call_count[0] == 1
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
            lambda _prompt, **_: {
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
            lambda _prompt, **_: {
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
