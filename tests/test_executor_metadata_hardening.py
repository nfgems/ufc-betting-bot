from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.polymarket.executor import OrderExecutor
from src.polymarket.tracker import BetLedger
from src.strategy.bankroll import BankrollManager


class _ProductionCapableClob:
    """Small CLOB double that exposes the production market-info capability."""

    def __init__(self, *, open_orders=None):
        self.metadata_calls = 0
        self.open_order_calls = 0
        self.limit_calls = 0
        self.open_orders = list(open_orders or [])

    def get_clob_market_info(self, _condition_id):
        self.metadata_calls += 1
        raise TimeoutError("market metadata timed out")

    def get_open_orders(self):
        self.open_order_calls += 1
        return list(self.open_orders)

    def create_limit_order(self, **_kwargs):
        self.limit_calls += 1
        return {"orderID": f"limit-{self.limit_calls}", "status": "live"}


class _CanonicalZeroFeeClob(_ProductionCapableClob):
    def get_clob_market_info(self, _condition_id):
        self.metadata_calls += 1
        return {
            "mts": "0.01",
            "nr": False,
            "t": [
                {"t": "token-yes"},
                {"t": "token-no"},
            ],
        }


class _MalformedFeeClob(_CanonicalZeroFeeClob):
    def get_clob_market_info(self, _condition_id):
        payload = super().get_clob_market_info(_condition_id)
        payload["fd"] = {"r": "not-a-number", "e": "0"}
        return payload


def _bet(**overrides) -> pd.Series:
    values = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "bet_on": "Alpha",
        "model_prob": 0.75,
        "blended_prob": 0.75,
        "market_prob": 0.60,
        "edge": 0.15,
        "decimal_odds": 1.6667,
        "bet_side": "a",
        "token_id_yes": "token-yes",
        "token_id_no": "token-no",
        "market_id": "market-1",
        "condition_id": "condition-1",
        "tick_size": "0.01",
        "neg_risk": False,
        "override_bet_size": 25.0,
        "event_date": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    }
    values.update(overrides)
    return pd.Series(values)


def _executor(tmp_path, clob, **kwargs) -> OrderExecutor:
    executor = OrderExecutor(
        bankroll=BankrollManager(
            initial_bankroll=100.0,
            max_bet_fraction=1.0,
            auto_detect_balance=False,
        ),
        clob_client=clob,
        dry_run=False,
        **kwargs,
    )
    executor.ledger = BetLedger(path=tmp_path / "ledger.json")
    executor._authoritative_wallet_conflict = lambda **_kwargs: (False, "")
    executor._check_liquidity = lambda *_args, **_kwargs: {
        "ok": True,
        "adjusted_size": 25.0,
        "available_liquidity": 100.0,
        "best_ask_liquidity": 100.0,
        "slippage": 0.0,
        "best_ask": 0.60,
        "reason": "",
    }
    return executor


@pytest.mark.parametrize("placement", ["primary", "near_miss"])
def test_duplicate_gate_reconciles_and_skips_metadata_hydration(tmp_path, placement):
    clob = _ProductionCapableClob(
        open_orders=[
            {
                "id": "recovered-order",
                "asset_id": "token-yes",
                "price": "0.60",
                "original_size": "10",
                "status": "live",
            }
        ]
    )
    executor = _executor(tmp_path, clob)
    executor.ledger.add_bet(
        fighter="Alpha",
        opponent="Beta",
        side="a",
        amount=6.0,
        price=0.60,
        shares=10.0,
        token_id="token-yes",
        market_id="market-1",
        condition_id="condition-1",
        model_prob=0.70,
        market_prob=0.60,
        edge=0.10,
        decimal_odds=1.6667,
        dry_run=False,
        order_type="marketable_limit",
        placement_state="pending_submit",
    )
    audit = []
    executor.decision_audit_callback = lambda _bet, decision: audit.append(decision)

    bet = _bet(edge=0.02 if placement == "near_miss" else 0.15)
    if placement == "near_miss":
        result = executor._place_near_miss_limit(bet, pd.DataFrame())
    else:
        result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.open_order_calls == 1
    assert clob.metadata_calls == 0
    assert audit[-1]["status"] == "already_bet"
    assert audit[-1]["gate"] == "duplicate_open_position"
    recovered = BetLedger(path=executor.ledger.path).bets[0]
    assert recovered["order_id"] == "recovered-order"
    assert recovered["placement_state"] == "submitted"


@pytest.mark.parametrize(
    ("placement", "expected_gate"),
    [("primary", "bet_window"), ("near_miss", "limit_bid_window")],
)
def test_closed_window_gate_skips_metadata_hydration(
    tmp_path,
    placement,
    expected_gate,
):
    clob = _ProductionCapableClob()
    executor = _executor(tmp_path, clob)
    audit = []
    executor.decision_audit_callback = lambda _bet, decision: audit.append(decision)
    bet = _bet(
        edge=0.02 if placement == "near_miss" else 0.15,
        event_date=(datetime.now(timezone.utc) + timedelta(days=4)).isoformat(),
    )

    if placement == "near_miss":
        result = executor._place_near_miss_limit(bet, pd.DataFrame())
    else:
        result = executor._place_bet(bet, pd.DataFrame())

    assert result is None
    assert clob.metadata_calls == 0
    assert audit[-1]["gate"] == expected_gate


def test_marketable_candidate_fails_closed_without_canonical_fee_metadata(tmp_path):
    clob = _ProductionCapableClob()
    executor = _executor(tmp_path, clob, force_market_order=True)
    audit = []
    executor.decision_audit_callback = lambda _bet, decision: audit.append(decision)

    result = executor._place_bet(_bet(), pd.DataFrame())

    assert result is None
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 0
    assert executor.ledger.open_bets == []
    assert executor.bankroll.bankroll == pytest.approx(100.0)
    assert audit[-1]["gate"] == "taker_fee_metadata"


def test_absent_fee_details_is_canonical_zero_fee_for_marketable_candidate(tmp_path):
    clob = _CanonicalZeroFeeClob()
    executor = _executor(tmp_path, clob, force_market_order=True)

    result = executor._place_bet(_bet(), pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert result["order_type"] == "marketable_limit"
    assert result["fee_rate"] == pytest.approx(0.0)
    assert result["fee_exponent"] == pytest.approx(0.0)
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 1


def test_malformed_explicit_fee_details_do_not_coerce_to_zero(tmp_path):
    clob = _MalformedFeeClob()
    executor = _executor(tmp_path, clob, force_market_order=True)

    result = executor._place_bet(_bet(), pd.DataFrame())

    assert result is None
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 0
    assert executor.ledger.open_bets == []


@pytest.mark.parametrize(
    ("fee_details", "expected_rate", "expected_exponent"),
    [
        ({"r": "0.1"}, 0.1, 0.0),
        ({"e": "2"}, 0.0, 2.0),
    ],
)
def test_nonempty_canonical_fee_details_default_missing_fields_to_zero(
    fee_details,
    expected_rate,
    expected_exponent,
):
    metadata, _token_ids = OrderExecutor._metadata_from_clob_market_info(
        {
            "mts": "0.01",
            "nr": False,
            "t": [{"t": "token-yes"}],
            "fd": fee_details,
        }
    )

    assert metadata["fee_source"] == "clob"
    assert metadata["fee_rate"] == pytest.approx(expected_rate)
    assert metadata["fee_exponent"] == pytest.approx(expected_exponent)


def test_tick_and_neg_risk_without_tokens_do_not_claim_canonical_zero_fee():
    metadata, token_ids = OrderExecutor._metadata_from_clob_market_info(
        {"mts": "0.01", "nr": False}
    )

    assert token_ids == []
    assert "fee_source" not in metadata
    assert "fee_rate" not in metadata


def test_forced_resting_maker_uses_noncanonical_fee_fallback(tmp_path):
    clob = _ProductionCapableClob()
    executor = _executor(tmp_path, clob, force_limit_order=True)

    result = executor._place_bet(_bet(), pd.DataFrame())

    assert result is not None
    assert result["status"] == "placed"
    assert result["order_type"] == "limit_bid"
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 1


def test_low_gross_edge_uses_resting_maker_without_canonical_fee_metadata(tmp_path):
    clob = _ProductionCapableClob()
    executor = _executor(tmp_path, clob)

    result = executor._place_bet(
        _bet(model_prob=0.61, blended_prob=0.61, edge=0.01),
        pd.DataFrame(),
    )

    assert result is not None
    assert result["status"] == "placed"
    assert result["order_type"] == "limit_bid"
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 1


def test_near_miss_maker_uses_noncanonical_fee_fallback(tmp_path):
    clob = _ProductionCapableClob()
    executor = _executor(tmp_path, clob)
    executor.bankroll.kelly_bet_size = lambda *_args, **_kwargs: 5.0

    result = executor._place_near_miss_limit(
        _bet(
            model_prob=0.61,
            blended_prob=0.61,
            edge=0.01,
            override_bet_size=None,
        ),
        pd.DataFrame(),
    )

    assert result is not None
    assert result["status"] == "placed"
    assert result["order_type"] == "near_miss_limit"
    assert clob.metadata_calls == 1
    assert clob.limit_calls == 1
