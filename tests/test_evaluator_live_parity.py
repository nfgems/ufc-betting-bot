"""Cross-system evaluator-vs-live S/C semantic parity regression.

DIR-AUD-P2-021: feed identical predictions/markets to the sweep evaluator
(`_evaluate_config`, legacy pricing) and the live runtime (`run_duo_traders`,
dry run, real executor sizing) under the same locked strategy config, and
require identical bet sets, sides, and sizes within execution-model
tolerances. This exercises selection, kelly and conviction sizing (including
the override cap), the S-then-C allocation handoff, and same-fight exclusion
with no selector, trader, or executor monkeypatched away.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategy import duo_trader
from src.strategy import execution_audit
from src.strategy.duo_trader_sweep import SweepConfig, _evaluate_config
from src.strategy.runtime_strategy import shared_strategy_constants_snapshot

_EVENT_DATE = "2026-12-05"
_INITIAL_BANKROLL = 1000.0

# (fighter_a, fighter_b, prob_a, no_odds_prob_a, market_prob_a, target)
_FIGHTS = [
    # Clear S bet on side a: blend edge 0.035 >= 0.03, no-odds agreement holds.
    ("Alpha One", "Bravo One", 0.70, 0.68, 0.60, 1),
    # C-only bet on side a: blend edge 0.007 < 0.03, conviction gates pass.
    ("Alpha Two", "Bravo Two", 0.80, 0.62, 0.78, 1),
    # No bet anywhere.
    ("Alpha Three", "Bravo Three", 0.50, 0.50, 0.50, 0),
    # S bet on side b (orientation coverage).
    ("Alpha Four", "Bravo Four", 0.28, 0.30, 0.38, 0),
    # S bet on side a AND a conviction candidate: C must skip the S fight.
    ("Alpha Five", "Bravo Five", 0.78, 0.75, 0.68, 1),
    # C-only bet sized into the c_max_bet_fraction cap (0.04 + 5*0.01 > 0.07).
    ("Alpha Six", "Bravo Six", 0.95, 0.60, 0.93, 1),
]

_EXPECTED_S_BETS = {"Alpha One", "Bravo Four", "Alpha Five"}
_EXPECTED_C_BETS = {"Alpha Two", "Alpha Six"}


def _locked_strategy() -> dict:
    return {
        "name": "parity_locked",
        "s_min_edge": 0.03,
        "s_blend_weight": 0.35,
        "s_model_agreement_min_edge": 0.02,
        "c_min_model_prob": 0.70,
        "c_min_no_odds_prob": 0.55,
        "c_share": 0.40,
        "c_bet_fraction": 0.04,
        "c_max_bet_fraction": 0.07,
        "c_min_edge": 0.01,
        "c_max_decimal_odds": 2.5,
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.30,
    }


def _sweep_config(strategy: dict) -> SweepConfig:
    return SweepConfig(
        name=strategy["name"],
        s_min_edge=strategy["s_min_edge"],
        s_blend_weight=strategy["s_blend_weight"],
        s_model_agreement_min_edge=strategy["s_model_agreement_min_edge"],
        c_min_model_prob=strategy["c_min_model_prob"],
        c_min_no_odds_prob=strategy["c_min_no_odds_prob"],
        c_share=strategy["c_share"],
        c_bet_fraction=strategy["c_bet_fraction"],
        c_max_bet_fraction=strategy["c_max_bet_fraction"],
        c_min_edge=strategy["c_min_edge"],
        c_max_decimal_odds=strategy["c_max_decimal_odds"],
        m_enabled=False,
    )


def _prediction_rows() -> list[dict]:
    rows = []
    for fighter_a, fighter_b, prob_a, no_odds_a, market_a, target in _FIGHTS:
        rows.append(
            {
                "event_date": _EVENT_DATE,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "target": target,
                "prob_a": prob_a,
                "prob_b": 1.0 - prob_a,
                "no_odds_prob_a": no_odds_a,
                "no_odds_prob_b": 1.0 - no_odds_a,
                "market_prob_a": market_a,
                "a_num_fights": 6,
                "b_num_fights": 6,
            }
        )
    return rows


def _evaluator_frame() -> pd.DataFrame:
    frame = pd.DataFrame(_prediction_rows())
    frame["a_market_prob"] = frame.pop("market_prob_a")
    frame["b_market_prob"] = 1.0 - frame["a_market_prob"]
    return frame


def _live_predictions() -> pd.DataFrame:
    frame = pd.DataFrame(_prediction_rows())
    return frame.drop(columns=["market_prob_a", "target"])


def _live_markets() -> pd.DataFrame:
    rows = []
    for index, (fighter_a, fighter_b, _prob_a, _no_odds_a, market_a, _target) in (
        enumerate(_FIGHTS)
    ):
        rows.append(
            {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "event_date": _EVENT_DATE,
                "event_title": "Parity Card",
                "market_id": f"mkt-{index}",
                "condition_id": f"cond-{index}",
                "token_id_yes": f"tok-{index}-yes",
                "token_id_no": f"tok-{index}-no",
                "price_yes": market_a,
                "price_no": round(1.0 - market_a, 6),
                "tick_size": 0.01,
                "neg_risk": False,
                "fee_rate": 0.0,
                "fee_exponent": 1,
                "fee_source": "clob",
                "volume": 10_000.0,
                "liquidity": 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_evaluator_and_live_runtime_agree_on_s_c_selection_and_sizing(
    tmp_path,
    monkeypatch,
):
    strategy = _locked_strategy()

    evaluator_result = _evaluate_config(
        [(1, _evaluator_frame())],
        _sweep_config(strategy),
        initial_bankroll=_INITIAL_BANKROLL,
        bet_start_date="2022-01-01",
        execution_mode="legacy",
    )
    evaluator_log = evaluator_result["bet_log"]
    assert not evaluator_log.empty
    evaluator_s = {
        row["bet_on"]: row
        for _, row in evaluator_log[evaluator_log["trader"] == "S"].iterrows()
    }
    evaluator_c = {
        row["bet_on"]: row
        for _, row in evaluator_log[evaluator_log["trader"] == "C"].iterrows()
    }
    assert set(evaluator_s) == _EXPECTED_S_BETS
    assert set(evaluator_c) == _EXPECTED_C_BETS

    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", tmp_path / "single_ledger.json")
    monkeypatch.setattr(
        duo_trader, "CONVICTION_LEDGER", tmp_path / "conviction_ledger.json"
    )
    monkeypatch.setattr(
        execution_audit, "persist_cycle_payload", lambda *_args, **_kwargs: None
    )

    live_result = duo_trader.run_duo_traders(
        predictions=_live_predictions(),
        markets=_live_markets(),
        clob=None,
        dry_run=True,
        min_edge=strategy["s_min_edge"],
        bankroll_basis=duo_trader.WalletBankrollBasis(
            _INITIAL_BANKROLL, _INITIAL_BANKROLL, "test"
        ),
        strategy_config=strategy,
        confirmed_shared_constants=shared_strategy_constants_snapshot(),
    )

    live_s = {order["fighter"]: order for order in live_result["trader_s"]["orders"]}
    live_c = {order["fighter"]: order for order in live_result["trader_c"]["orders"]}

    # Identical selection: same fights, same sides (fighter identity encodes
    # the side), on both S and C, including the same-fight C exclusion.
    assert set(live_s) == set(evaluator_s) == _EXPECTED_S_BETS
    assert set(live_c) == set(evaluator_c) == _EXPECTED_C_BETS
    assert live_result["trader_s"]["near_miss_orders"] == []
    assert live_result["trader_m"]["deferred"] is True
    assert live_result["trader_m"]["orders"] == []

    # Identical pricing basis: legacy evaluator fills at the snapshot market
    # probability, which is exactly the dry-run marketable price.
    for fighter, order in {**live_s, **live_c}.items():
        evaluator_row = {**evaluator_s, **evaluator_c}[fighter]
        assert order["price"] == pytest.approx(
            float(evaluator_row["market_prob"]), abs=1e-6
        )

    # Identical sizing: S kelly*multiplier and C conviction sizing with the
    # override cap. Tolerance covers only share/min-notional cent rounding.
    for fighter in _EXPECTED_S_BETS:
        assert live_s[fighter]["bet_size_usd"] == pytest.approx(
            float(evaluator_s[fighter]["bet_size"]), abs=0.05
        )
    for fighter in _EXPECTED_C_BETS:
        assert live_c[fighter]["bet_size_usd"] == pytest.approx(
            float(evaluator_c[fighter]["bet_size"]), abs=0.05
        )

    # The conviction cap fight must actually exercise the cap on both systems.
    conviction_allocation = live_result["trader_c"]["allocation"]
    assert live_c["Alpha Six"]["bet_size_usd"] == pytest.approx(
        strategy["c_max_bet_fraction"] * conviction_allocation, abs=0.05
    )

    # Identical S->C bankroll handoff: C allocation is the configured share of
    # the cash remaining after S reserves its stakes, on both systems.
    evaluator_s_wagered = float(
        evaluator_log.loc[evaluator_log["trader"] == "S", "bet_size"].sum()
    )
    expected_allocation = (_INITIAL_BANKROLL - evaluator_s_wagered) * strategy[
        "c_share"
    ]
    assert conviction_allocation == pytest.approx(expected_allocation, abs=0.1)
