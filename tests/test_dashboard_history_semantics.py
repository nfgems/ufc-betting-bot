import json
from pathlib import Path

import pytest

from src.web.refresh_dashboard import _private_legacy_module


@pytest.fixture(scope="module")
def legacy():
    return _private_legacy_module()


@pytest.fixture
def synthetic_positions():
    return json.loads(Path(__file__).with_name("fixtures_dashboard_history_audit.json").read_text(encoding="utf-8-sig"))["positions"]


def closed(legacy, position, **kwargs):
    return legacy._profile_closed_row(position, matched_bets=kwargs.get("matched_bets", []),
                                     trades=kwargs.get("trades", []), redeem_trades=kwargs.get("redeem_trades", []))


def test_closed_positions_do_not_invent_settlement_or_resolution(legacy, synthetic_positions):
    before = json.dumps(synthetic_positions, sort_keys=True)
    for position in synthetic_positions:
        row = closed(legacy, position)
        assert row["status"] == ("closed_profit" if position["realizedPnl"] > 0 else "closed_loss")
        assert row["result_pnl"] == position["realizedPnl"]
        assert row["settled_at"] is None and row["placed_at"] is None
        assert row["market_end_date"] == position["endDate"]
        assert row["resolution_confirmed"] is False
        assert row["resolution_source"] is None
        assert not row["traders"] and row["trader_label"] is None
    assert json.dumps(synthetic_positions, sort_keys=True) == before


def test_template_fixture_rows_match_the_closed_position_adapter(legacy, synthetic_positions):
    fixture = json.loads(Path(__file__).with_name("fixtures_dashboard_history_audit.json").read_text(encoding="utf-8-sig"))
    assert len(fixture["render_rows"]) == len(synthetic_positions) == 3
    for position, expected in zip(synthetic_positions, fixture["render_rows"]):
        actual = closed(legacy, position)
        assert {key: actual[key] for key in expected} == expected


def test_actual_exit_time_is_retained_but_market_expiry_is_separate(legacy, synthetic_positions):
    position = synthetic_positions[0]
    row = closed(legacy, position, trades=[{"side": "SELL", "timestamp": 1781279100}])
    assert row["settled_at"] == "2026-06-12T15:45:00+00:00"
    assert row["market_end_date"] == position["endDate"]
    assert row["status"] == "closed_profit"
    explicit = closed(legacy, {**position, "closedAt": "2026-06-01T17:00:00Z"})
    assert explicit["settled_at"] == "2026-06-01T17:00:00Z"


@pytest.mark.parametrize("price,pnl,status", [(1, -3, "won"), (0, 3, "lost"), (.5, 3, "closed_profit")])
def test_resolution_requires_explicit_evidence_and_uses_outcome_not_return(legacy, synthetic_positions, price, pnl, status):
    position = {**synthetic_positions[0], "curPrice": price, "realizedPnl": pnl}
    assert closed(legacy, position)["status"] in {"closed_profit", "closed_loss"}
    result = closed(legacy, {**position, "resolved": True})
    assert result["status"] == status
    assert result["resolution_confirmed"] is (status in {"won", "lost"})


def test_conflicting_original_results_do_not_claim_resolution(legacy, synthetic_positions):
    bets = [{"status": "won", "settled_at": "2026-06-12T15:45:00Z"},
            {"status": "lost", "settled_at": "2026-06-12T15:45:00Z"}]
    assert closed(legacy, synthetic_positions[0], matched_bets=bets)["resolution_confirmed"] is False


def test_passed_original_settlement_evidence_retains_actual_time(legacy, synthetic_positions):
    result = closed(legacy, synthetic_positions[0], matched_bets=[
        {"status": "won", "settled_at": "2026-06-12T15:45:00Z", "trader": "S"}])
    assert result["status"] == "won" and result["resolution_confirmed"] is True
    assert result["settled_at"] == "2026-06-12T15:45:00Z"
    assert result["resolution_source"] == "matched_original_settlement"


def test_unavailable_closed_return_does_not_become_flat_zero(legacy, synthetic_positions):
    position = {key: value for key, value in synthetic_positions[0].items() if key != "realizedPnl"}
    result = closed(legacy, position)
    assert result["status"] == "closed" and result["result_pnl"] is None
    assert result["resolution_confirmed"] is False


def test_counts_and_headlines_keep_closed_and_resolved_sources_distinct(legacy):
    rows = [{"status": state, "amount": 10, "sport": "ufc", "result_pnl": pnl}
            for state, pnl in [("closed_profit", 5), ("closed_loss", -3), ("closed_flat", 0),
                               ("won", -1), ("lost", 2), ("sold", 1)]]
    rows.append({"status": "open", "sport": "ufc", "amount": 4, "unrealized_pnl": 1, "value": 5})
    raw = {"total_pnl": -321.5, "realized_pnl": -300.25, "unrealized_pnl": -21.25, "current_value": 8.5}
    profile = {"total_pnl": 123.75, "positions_value": 8.5, "predictions": 37}
    result = legacy._build_profile_summary(rows=rows, raw_live_pnl=raw, profile_snapshot=profile,
                                          live_source="live", degraded=False)
    assert result["total_pnl"] == profile["total_pnl"]
    assert result["realized_pnl"] == raw["realized_pnl"]
    assert result["unrealized_pnl"] == raw["unrealized_pnl"]
    assert result["open_bets"] == 1 and result["closed_bets"] == 6
    assert result["settled_bets"] == 2 and result["wins"] == 1 and result["losses"] == 1
    assert result["closed_profit"] == 1 and result["closed_loss"] == 1 and result["closed_flat"] == 1
    scoped = legacy._scope_profile_bets_payload({"summary": result, "bets": rows}, "ufc")["summary"]
    assert scoped["profile_total_pnl"] == profile["total_pnl"]
    assert scoped["closed_bets"] == 6 and scoped["settled_bets"] == 2
    assert scoped["realized_pnl"] == 4  # includes all closed returns, not only resolved results


def test_win_rate_is_unavailable_without_confirmed_outcomes(legacy):
    rows = [{"status": "closed_profit", "sport": "ufc", "result_pnl": 5}]
    result = legacy._build_profile_summary(rows=rows, raw_live_pnl={"total_pnl": 5}, profile_snapshot={},
                                          live_source="live", degraded=False)
    assert result["wins"] == 0 and result["losses"] == 0 and result["settled_bets"] == 0
    assert result["win_rate"] is None
    assert legacy._scope_profile_bets_payload({"summary": result, "bets": rows}, "ufc")["summary"]["win_rate"] is None
