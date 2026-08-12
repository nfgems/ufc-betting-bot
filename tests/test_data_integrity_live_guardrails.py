import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.data import fallback_scrapers, method_odds
from src.polymarket.markets import parse_fight_market
from src.strategy.value import (
    _coerce_probability,
    _filter_rejection_reason,
    _passes_quality_filters,
    find_conviction_bets,
    find_value_bets,
)


def _prediction_row(**overrides):
    row = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "prob_a": 0.80,
        "prob_b": 0.20,
        "a_market_prob": 0.55,
        "b_market_prob": 0.45,
        "no_odds_prob_a": 0.70,
        "no_odds_prob_b": 0.30,
        "a_num_fights": 8,
        "b_num_fights": 8,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_probability_coercion_rejects_nonfinite_and_out_of_range_values(value):
    assert _coerce_probability(value) is None


def test_mixed_card_nan_no_odds_probability_cannot_create_a_value_bet():
    predictions = pd.DataFrame(
        [
            _prediction_row(
                fighter_a="Invalid Alpha",
                fighter_b="Invalid Beta",
                no_odds_prob_a=None,
            ),
            _prediction_row(
                fighter_a="Valid Alpha",
                fighter_b="Valid Beta",
            ),
        ]
    )

    result = find_value_bets(predictions, min_edge=0.01)

    assert result["bet_on"].tolist() == ["Valid Alpha"]


@pytest.mark.parametrize("no_odds_prob", [None, np.nan])
def test_defensive_value_gate_rejects_invalid_no_odds_probability_when_required(
    no_odds_prob,
):
    assert not _passes_quality_filters(
        blended_prob=0.70,
        market_prob=0.55,
        edge=0.15,
        fighter_name="Alpha Fighter",
        no_odds_prob=no_odds_prob,
        a_num_fights=8,
        b_num_fights=8,
        require_model_agreement=True,
    )


def test_signal_only_value_gates_allow_missing_no_odds_when_agreement_is_disabled():
    kwargs = {
        "blended_prob": 0.70,
        "market_prob": 0.55,
        "edge": 0.15,
        "fighter_name": "Alpha Fighter",
        "no_odds_prob": None,
        "a_num_fights": 8,
        "b_num_fights": 8,
        "require_model_agreement": False,
    }

    assert _passes_quality_filters(**kwargs)
    assert _filter_rejection_reason(**kwargs) is None


def test_conviction_requires_valid_no_odds_probability_and_known_experience():
    valid = find_conviction_bets(pd.DataFrame([_prediction_row()]))
    unknown_experience = find_conviction_bets(
        pd.DataFrame([_prediction_row(a_num_fights=None)])
    )
    invalid_no_odds = find_conviction_bets(
        pd.DataFrame([_prediction_row(no_odds_prob_a=np.nan)])
    )

    assert valid["bet_on"].tolist() == ["Alpha Fighter"]
    assert unknown_experience.empty
    assert invalid_no_odds.empty


def _fight_market(**overrides):
    market = {
        "id": "market-1",
        "conditionId": "condition-1",
        "question": "Alpha Fighter vs Beta Fighter",
        "outcomes": json.dumps(["Alpha Fighter", "Beta Fighter"]),
        "clobTokenIds": json.dumps(["token-a", "token-b"]),
        "outcomePrices": json.dumps(["0.60", "0.40"]),
        "active": True,
        "closed": False,
        "gameStartTime": "2026-08-20T20:00:00Z",
    }
    market.update(overrides)
    return market


@pytest.mark.parametrize(
    "prices",
    [[], ["0.60"], [None, "0.40"], ["0.60", None]],
)
def test_polymarket_market_requires_both_published_outcome_prices(prices):
    parsed = parse_fight_market(
        _fight_market(
            outcomePrices=json.dumps(prices),
            bestBid="0.58",
            bestAsk="0.62",
        )
    )

    assert parsed is None


def test_markerless_bfo_page_is_never_used_for_method_odds():
    soup = BeautifulSoup(
        """
        <html><body><h1>Bruno Silva vs Brad Tavares</h1><table>
          <tr><th>Bruno Silva wins by TKO/KO</th><td>+200</td></tr>
          <tr><th>Anderson Silva wins by TKO/KO</th><td>+900</td></tr>
        </table></body></html>
        """,
        "lxml",
    )

    assert method_odds._parse_bfo_method_odds(
        soup, "Bruno Silva", "Brad Tavares"
    ) is None


def _scoped_bfo_soup():
    return BeautifulSoup(
        """
        <html><body><table class="odds-table">
          <tr id="mu-100"><td class="t-b-fcc">Bruno Silva</td></tr>
          <tr><td class="t-b-fcc">Brad Tavares</td></tr>
          <tr><th>Bruno Silva wins by TKO/KO</th><td>+200</td></tr>
          <tr><th>Brad Tavares wins by decision</th><td>+150</td></tr>
        </table></body></html>
        """,
        "lxml",
    )


def test_bfo_snapshot_record_carries_v2_page_provenance():
    source_url = "https://www.bestfightodds.com/events/ufc-test"

    class Resolver:
        def resolve_event_page(self, _fights):
            return source_url, _scoped_bfo_soup()

    records = method_odds._collect_bfo_records_for_event_group(
        [
            {
                "fighter_a": "Bruno Silva",
                "fighter_b": "Brad Tavares",
                "event_id": "evt-1",
                "commence_time": "2026-08-20T20:00:00Z",
            }
        ],
        Resolver(),
    )

    assert len(records) == 1
    record = records[0]
    assert record["source_url"] == source_url
    assert len(record["page_sha256"]) == 64
    assert record["parser_contract"] == method_odds.BFO_PARSER_CONTRACT
    assert record["fighter_a_norm"] == "bruno silva"
    assert record["commence_time"] == "2026-08-20T20:00:00Z"


def test_method_odds_inference_ignores_schema_v1_without_migrating_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    snapshot_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(
        microsecond=0
    ).isoformat()
    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.30,
        },
        source="bestfightodds",
        source_url="https://www.bestfightodds.com/events/ufc-test",
        page_sha256="a" * 64,
        parser_contract=method_odds.BFO_PARSER_CONTRACT,
        captured_at=snapshot_time,
        event_id="evt-1",
    )
    path = tmp_path / "method_odds_20260820_200000.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_time": snapshot_time,
                "status": "success",
                "record_count": 1,
                "records": [record],
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )

    assert all(np.isnan(value) for value in result.values())
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_method_odds_inference_rejects_corrupt_v2_record_provenance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    snapshot_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.30,
        },
        source="bestfightodds",
        source_url="https://www.bestfightodds.com/events/ufc-test",
        page_sha256="not-a-sha256",
        parser_contract=method_odds.BFO_PARSER_CONTRACT,
        captured_at=snapshot_time,
        commence_time="2026-08-20T20:00:00Z",
        event_id="evt-1",
    )
    (tmp_path / "method_odds_20260820_200000.json").write_text(
        json.dumps(
            {
                "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
                "snapshot_time": snapshot_time,
                "status": "success",
                "record_count": 1,
                "records": [record],
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )

    assert all(np.isnan(value) for value in result.values())


def test_sherdog_missing_record_is_unknown_but_literal_zero_is_observed():
    missing = fallback_scrapers._parse_sherdog_profile(
        BeautifulSoup("<html><body><h1>Unknown Fighter</h1></body></html>", "lxml"),
        "https://www.sherdog.com/fighter/unknown",
    )
    observed = fallback_scrapers._parse_sherdog_profile(
        BeautifulSoup(
            """
            <html><body><h1>Known Fighter</h1>
              <div class="winloses"><span>Wins</span><span>0</span></div>
              <div class="winloses"><span>Losses</span><span>2</span></div>
              <div class="winloses"><span>Draws</span><span>0</span></div>
            </body></html>
            """,
            "lxml",
        ),
        "https://www.sherdog.com/fighter/known",
    )

    assert missing["record"] == ""
    assert all(np.isnan(missing[field]) for field in ("wins", "losses", "draws"))
    assert observed["record"] == "0-2-0"
    assert observed["wins"] == 0
    assert observed["draws"] == 0
