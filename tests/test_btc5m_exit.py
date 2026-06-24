from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.polymarket.btc5m_exit import (
    Btc5mExitContext,
    Btc5mExitShadowState,
    evaluate_exit_policy,
    fair_value_from_context,
    net_bid_price,
)


NOW = datetime(2026, 6, 20, 20, 13, tzinfo=timezone.utc)


def _state_with_vol() -> Btc5mExitShadowState:
    return Btc5mExitShadowState(
        marks=[
            (NOW - timedelta(seconds=30), 100_000.0),
            (NOW - timedelta(seconds=20), 100_020.0),
            (NOW - timedelta(seconds=10), 99_990.0),
        ]
    )


def _context(
    *,
    profile_name: str = "balanced",
    strategy_style: str = "momentum",
    side: str = "up",
    entry_price: float = 0.50,
    best_bid: float = 0.50,
    midpoint: float | None = None,
    btc_current_price: float = 100_090.0,
    btc_start_price: float = 100_000.0,
    seconds_left: float = 90.0,
    entry_btc_move_usd: float | None = 90.0,
    state: Btc5mExitShadowState | None = None,
    profile_attrs: dict | None = None,
) -> Btc5mExitContext:
    shares = 10.0
    return Btc5mExitContext(
        bet={"btc_move_usd": entry_btc_move_usd, "status": "open"},
        profile=SimpleNamespace(name=profile_name, **(profile_attrs or {})),
        profile_name=profile_name,
        strategy_style=strategy_style,
        side=side,
        entry_price=entry_price,
        shares=shares,
        amount=shares * entry_price,
        btc_start_price=btc_start_price,
        btc_current_price=btc_current_price,
        seconds_left=seconds_left,
        held_book=SimpleNamespace(best_bid=best_bid, midpoint=midpoint if midpoint is not None else best_bid),
        fee_schedule={"rate": 0.07},
        recorded_at=NOW,
        shadow_state=state or Btc5mExitShadowState(),
    )


def test_net_bid_subtracts_crypto_taker_fee():
    assert net_bid_price(0.50, {"rate": 0.07}) == 0.4825
    assert round(net_bid_price(0.90, {"rate": 0.07}), 6) == 0.8937


def test_fair_value_is_monotonic_for_up_and_down():
    state = _state_with_vol()
    up_high = fair_value_from_context(
        _context(side="up", btc_current_price=100_050.0, state=state)
    )
    up_low = fair_value_from_context(
        _context(side="up", btc_current_price=99_950.0, state=state)
    )
    down_high = fair_value_from_context(
        _context(side="down", btc_current_price=99_950.0, state=state)
    )
    down_low = fair_value_from_context(
        _context(side="down", btc_current_price=100_050.0, state=state)
    )

    assert up_high is not None and up_low is not None
    assert down_high is not None and down_low is not None
    assert up_high > up_low
    assert down_high > down_low


def test_fair_value_unavailable_holds_instead_of_exiting():
    decision = evaluate_exit_policy(
        _context(
            profile_name="balanced",
            side="up",
            entry_btc_move_usd=100.0,
            btc_current_price=100_090.0,
            best_bid=0.51,
        )
    )

    assert decision.action == "hold"
    assert decision.fair_value is None
    assert decision.trigger == "momentum_hold"


def test_cheap_side_uses_fair_value_take_profit_without_fixed_stop():
    decision = evaluate_exit_policy(
        _context(
            profile_name="cheap_side",
            strategy_style="cheap_side",
            side="down",
            entry_price=0.20,
            best_bid=0.36,
            btc_current_price=100_050.0,
            seconds_left=30.0,
            state=_state_with_vol(),
        )
    )

    assert decision.action == "shadow_exit"
    assert decision.model == "cheap_side_reversion"
    assert decision.trigger == "fair_value_take_profit"


def test_cheap_side_take_profit_target_is_profile_tunable():
    # Identical book/price situation: the default +0.08 target locks the win, but a
    # profile that demands a far larger +0.30 profit before taking it keeps holding.
    common = dict(
        profile_name="cheap_side",
        strategy_style="cheap_side",
        side="down",
        entry_price=0.20,
        best_bid=0.36,
        btc_current_price=100_050.0,
        seconds_left=30.0,
    )
    default_decision = evaluate_exit_policy(_context(state=_state_with_vol(), **common))
    assert default_decision.trigger == "fair_value_take_profit"
    assert default_decision.diagnostics["tp_profit_target"] == 0.08

    tuned_decision = evaluate_exit_policy(
        _context(
            state=_state_with_vol(),
            profile_attrs={"tp_profit_target": 0.30, "tp_trail_arm": 0.40},
            **common,
        )
    )
    assert tuned_decision.action == "hold"
    assert tuned_decision.diagnostics["tp_profit_target"] == 0.30


def test_momentum_profile_exits_on_btc_reversal():
    decision = evaluate_exit_policy(
        _context(
            profile_name="balanced",
            side="up",
            entry_btc_move_usd=100.0,
            btc_current_price=100_030.0,
            best_bid=0.42,
        )
    )

    assert decision.action == "shadow_exit"
    assert decision.trigger == "btc_reversal_stop"


def test_confidence_profile_exits_on_favorite_flip():
    decision = evaluate_exit_policy(
        _context(
            profile_name="confidence",
            side="up",
            best_bid=0.56,
            midpoint=0.55,
            btc_current_price=100_020.0,
        )
    )

    assert decision.action == "shadow_exit"
    assert decision.trigger == "favorite_flip"


def test_late_profile_only_exits_on_emergency_flip():
    decision = evaluate_exit_policy(
        _context(
            profile_name="late",
            side="up",
            best_bid=0.12,
            btc_current_price=99_990.0,
            seconds_left=12.0,
        )
    )

    assert decision.action == "shadow_exit"
    assert decision.trigger == "start_price_flip"


def test_late_capture_uses_catastrophe_trigger_only():
    decision = evaluate_exit_policy(
        _context(
            profile_name="late_capture",
            strategy_style="probability_capture",
            side="up",
            entry_price=0.90,
            best_bid=0.20,
            btc_current_price=99_990.0,
            seconds_left=20.0,
        )
    )

    assert decision.action == "shadow_exit"
    assert decision.trigger == "start_price_flip"


def test_trailing_state_updates_without_mutating_bet():
    state = Btc5mExitShadowState()
    bet = {"btc_move_usd": 100.0, "status": "open"}
    first = _context(
        profile_name="balanced",
        side="up",
        best_bid=0.60,
        btc_current_price=100_100.0,
        entry_btc_move_usd=100.0,
        state=state,
    )
    first = Btc5mExitContext(**{**first.__dict__, "bet": bet})
    first_decision = evaluate_exit_policy(first)

    second = _context(
        profile_name="balanced",
        side="up",
        best_bid=0.54,
        btc_current_price=100_100.0,
        entry_btc_move_usd=100.0,
        state=state,
    )
    second = Btc5mExitContext(**{**second.__dict__, "bet": bet})
    second_decision = evaluate_exit_policy(second)

    assert first_decision.action == "hold"
    assert second_decision.action == "shadow_exit"
    assert second_decision.trigger == "trailing_stop"
    assert bet == {"btc_move_usd": 100.0, "status": "open"}
