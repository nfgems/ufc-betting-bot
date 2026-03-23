import pandas as pd
import pytest

from src.strategy.tennis_decision import (
    annotate_tennis_reference_edges,
    apply_tennis_automation_controls,
    build_tennis_execution_decisions,
)


def test_annotate_tennis_reference_edges_marks_qualified_reference_edge():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.62,
                "prob_b": 0.38,
                "a_fair_prob_avg": 0.54,
                "b_fair_prob_avg": 0.46,
                "num_bookmakers": 4,
                "a_num_matches": 14,
                "b_num_matches": 18,
            }
        ]
    )

    decisions = annotate_tennis_reference_edges(predictions, min_edge=0.02)
    row = decisions.iloc[0]

    assert row["decision_side"] == "a"
    assert row["decision_fighter"] == "Player A"
    assert row["decision_status"] == "reference_qualified"
    assert row["decision_edge"] == pytest.approx(0.08)
    assert row["required_edge"] == pytest.approx(0.02)
    assert row["uncertainty_penalty"] == pytest.approx(0.0)


def test_annotate_tennis_reference_edges_applies_low_history_and_confidence_penalties():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.57,
                "prob_b": 0.43,
                "a_fair_prob_avg": 0.54,
                "b_fair_prob_avg": 0.46,
                "num_bookmakers": 5,
                "a_num_matches": 4,
                "b_num_matches": 16,
            }
        ]
    )

    decisions = annotate_tennis_reference_edges(predictions, min_edge=0.02)
    row = decisions.iloc[0]

    assert row["decision_status"] == "insufficient_reference_edge"
    assert row["min_player_matches"] == pytest.approx(4.0)
    assert row["uncertainty_penalty"] == pytest.approx(0.03)
    assert row["required_edge"] == pytest.approx(0.05)
    assert "low_history_3_to_5" in row["decision_reasons"]
    assert "low_confidence" in row["decision_reasons"]
    assert "insufficient_reference_edge" in row["decision_reasons"]


def test_annotate_tennis_reference_edges_requires_minimum_bookmakers():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.64,
                "prob_b": 0.36,
                "a_fair_prob_avg": 0.52,
                "b_fair_prob_avg": 0.48,
                "num_bookmakers": 2,
                "a_num_matches": 20,
                "b_num_matches": 24,
            }
        ]
    )

    decisions = annotate_tennis_reference_edges(predictions, min_edge=0.02)
    row = decisions.iloc[0]

    assert row["decision_status"] == "insufficient_bookmakers"
    assert "insufficient_bookmakers" in row["decision_reasons"]


def test_build_tennis_execution_decisions_marks_bettable_now_for_side_a():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.62,
                "prob_b": 0.38,
                "a_fair_prob_avg": 0.54,
                "b_fair_prob_avg": 0.46,
                "num_bookmakers": 4,
                "a_num_matches": 14,
                "b_num_matches": 18,
                "price_yes": 0.50,
                "price_no": 0.50,
            }
        ]
    )

    decisions = build_tennis_execution_decisions(predictions, min_edge=0.02, bankroll=500.0)
    row = decisions.iloc[0]

    assert row["decision_side"] == "a"
    assert row["execution_status"] == "bettable_now"
    assert row["execution_price"] == pytest.approx(0.50)
    assert row["execution_edge"] == pytest.approx(0.12)
    assert row["execution_decimal_odds"] == pytest.approx(2.0)
    assert row["stake_usd"] > 0.0


def test_build_tennis_execution_decisions_uses_price_no_for_side_b():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.35,
                "prob_b": 0.65,
                "a_fair_prob_avg": 0.44,
                "b_fair_prob_avg": 0.56,
                "num_bookmakers": 4,
                "a_num_matches": 14,
                "b_num_matches": 18,
                "price_yes": 0.52,
                "price_no": 0.48,
            }
        ]
    )

    decisions = build_tennis_execution_decisions(predictions, min_edge=0.02, bankroll=500.0)
    row = decisions.iloc[0]

    assert row["decision_side"] == "b"
    assert row["decision_fighter"] == "Player B"
    assert row["execution_price"] == pytest.approx(0.48)
    assert row["execution_edge"] == pytest.approx(0.17)
    assert row["execution_status"] == "bettable_now"
    assert row["stake_usd"] > 0.0


def test_build_tennis_execution_decisions_blocks_small_execution_edge():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.62,
                "prob_b": 0.38,
                "a_fair_prob_avg": 0.54,
                "b_fair_prob_avg": 0.46,
                "num_bookmakers": 4,
                "a_num_matches": 14,
                "b_num_matches": 18,
                "price_yes": 0.61,
                "price_no": 0.39,
            }
        ]
    )

    decisions = build_tennis_execution_decisions(predictions, min_edge=0.02, bankroll=500.0)
    row = decisions.iloc[0]

    assert row["decision_status"] == "reference_qualified"
    assert row["execution_status"] == "insufficient_execution_edge"
    assert row["execution_edge"] == pytest.approx(0.01)
    assert row["stake_usd"] == pytest.approx(0.0)


def test_annotate_tennis_reference_edges_flags_suspicious_market_mismatch():
    predictions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "prob_a": 0.78,
                "prob_b": 0.22,
                "a_fair_prob_avg": 0.18,
                "b_fair_prob_avg": 0.82,
                "num_bookmakers": 6,
                "a_num_matches": 30,
                "b_num_matches": 35,
            }
        ]
    )

    decisions = annotate_tennis_reference_edges(predictions, min_edge=0.02, suspicious_edge_threshold=0.35)
    row = decisions.iloc[0]

    assert row["decision_side"] == "a"
    assert row["decision_status"] == "suspicious_market_mismatch"
    assert row["decision_edge"] == pytest.approx(0.60)
    assert "suspicious_market_mismatch" in row["decision_reasons"]


def test_apply_tennis_automation_controls_auto_skips_low_history_reference():
    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "reference_qualified",
                "execution_status": "bettable_now",
                "min_player_matches": 4,
                "num_bookmakers": 5,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(decisions, legacy_blocklist_path=None)
    row = reviewed.iloc[0]

    assert row["automation_status"] == "auto_skip"
    assert bool(row["execution_allowed"]) is False
    assert row["execution_status"] == "auto_skip"
    assert row["trade_status"] == "auto_skipped"
    assert "low_history_auto_skip" in row["automation_reasons"]


def test_apply_tennis_automation_controls_respects_auto_blocklist_and_overrides_execution_status(tmp_path):
    blocklist_path = tmp_path / "auto_blocklist.csv"
    pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "reason": "hard block",
                "active": 1,
            }
        ]
    ).to_csv(blocklist_path, index=False)

    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "reference_qualified",
                "execution_status": "bettable_now",
                "decision_market_prob": 0.55,
                "min_player_matches": 24,
                "num_bookmakers": 7,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(
        decisions,
        blocklist_path=blocklist_path,
        legacy_blocklist_path=None,
    )
    row = reviewed.iloc[0]

    assert row["automation_status"] == "auto_block"
    assert bool(row["execution_allowed"]) is False
    assert row["execution_status"] == "auto_block"
    assert row["trade_status"] == "blocked"
    assert "auto_blocklist" in row["automation_reasons"]


def test_apply_tennis_automation_controls_auto_skips_thin_market_without_second_source():
    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "reference_qualified",
                "decision_market_prob": 0.56,
                "min_player_matches": 18,
                "num_bookmakers": 3,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(decisions, legacy_blocklist_path=None)
    row = reviewed.iloc[0]

    assert bool(row["second_source_required"]) is True
    assert row["second_source_status"] == "unavailable"
    assert row["automation_status"] == "auto_skip"
    assert row["trade_status"] == "auto_skipped"
    assert "thin_market_auto_skip" in row["automation_reasons"]
    assert "second_source_unavailable" in row["automation_reasons"]


def test_apply_tennis_automation_controls_confirms_thin_market_with_execution_market_second_source():
    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "reference_qualified",
                "decision_side": "a",
                "decision_market_prob": 0.54,
                "execution_status": "bettable_now",
                "execution_price": 0.58,
                "min_player_matches": 18,
                "num_bookmakers": 3,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(decisions, legacy_blocklist_path=None)
    row = reviewed.iloc[0]

    assert bool(row["second_source_required"]) is True
    assert row["second_source_status"] == "confirmed"
    assert row["automation_status"] == "auto_eligible"
    assert bool(row["trade_ready"]) is True
    assert row["trade_status"] == "tradeable_now"
    assert "thin_market_second_source_confirmed" in row["automation_reasons"]


def test_apply_tennis_automation_controls_auto_skips_suspicious_market_mismatch():
    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "suspicious_market_mismatch",
                "min_player_matches": 20,
                "num_bookmakers": 6,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(decisions, legacy_blocklist_path=None)
    row = reviewed.iloc[0]

    assert row["automation_status"] == "auto_skip"
    assert row["trade_status"] == "auto_skipped"
    assert "suspicious_market_mismatch" in row["automation_reasons"]


def test_apply_tennis_automation_controls_auto_skips_large_second_source_disagreement():
    decisions = pd.DataFrame(
        [
            {
                "fighter_a": "Player A",
                "fighter_b": "Player B",
                "decision_status": "reference_qualified",
                "decision_side": "a",
                "decision_market_prob": 0.55,
                "execution_status": "bettable_now",
                "execution_price": 0.80,
                "min_player_matches": 18,
                "num_bookmakers": 6,
            }
        ]
    )

    reviewed = apply_tennis_automation_controls(decisions, legacy_blocklist_path=None)
    row = reviewed.iloc[0]

    assert row["second_source_status"] == "contradicted"
    assert row["automation_status"] == "auto_skip"
    assert row["trade_status"] == "auto_skipped"
    assert "second_source_market_disagreement" in row["automation_reasons"]
