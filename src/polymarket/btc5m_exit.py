"""Shadow exit models for BTC 5-minute Polymarket positions.

The functions in this module are intentionally side-effect light: they update
only in-memory shadow state and return hypothetical decisions. They do not
place orders or mutate ledgers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


BTC5M_EXIT_SHADOW_SCHEMA_VERSION = 1


@dataclass
class Btc5mExitShadowState:
    marks: list[tuple[datetime, float]] = field(default_factory=list)
    high_bid: float | None = None
    trailing_armed: bool = False

    def record_mark(
        self,
        *,
        recorded_at: datetime,
        btc_price: float,
        best_bid: float | None,
        max_marks: int = 60,
    ) -> None:
        self.marks.append((recorded_at, float(btc_price)))
        if len(self.marks) > max_marks:
            self.marks = self.marks[-max_marks:]
        if best_bid is not None:
            bid = float(best_bid)
            self.high_bid = bid if self.high_bid is None else max(self.high_bid, bid)


@dataclass(frozen=True)
class Btc5mExitContext:
    bet: dict
    profile: Any
    profile_name: str
    strategy_style: str
    side: str
    entry_price: float
    shares: float
    amount: float
    btc_start_price: float
    btc_current_price: float
    seconds_left: float
    held_book: Any
    fee_schedule: dict
    recorded_at: datetime
    shadow_state: Btc5mExitShadowState

    @property
    def btc_move_usd(self) -> float:
        return self.btc_current_price - self.btc_start_price

    @property
    def entry_btc_move_usd(self) -> float | None:
        return _optional_float(self.bet.get("btc_move_usd"))


@dataclass(frozen=True)
class Btc5mExitDecision:
    action: str
    model: str
    trigger: str
    reason: str
    net_bid: float | None
    fair_value: float | None
    hypothetical_pnl: float | None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": BTC5M_EXIT_SHADOW_SCHEMA_VERSION,
            "action": self.action,
            "model": self.model,
            "trigger": self.trigger,
            "reason": self.reason,
            "net_bid": _round_optional(self.net_bid),
            "fair_value": _round_optional(self.fair_value),
            "hypothetical_pnl": _round_optional(self.hypothetical_pnl, 4),
            "diagnostics": self.diagnostics,
        }


_SHADOW_STATES: dict[str, Btc5mExitShadowState] = {}


def get_exit_shadow_state(key: str) -> Btc5mExitShadowState:
    if key not in _SHADOW_STATES:
        _SHADOW_STATES[key] = Btc5mExitShadowState()
    return _SHADOW_STATES[key]


def fee_rate_from_schedule(fee_schedule: dict | None) -> float:
    if not isinstance(fee_schedule, dict):
        return 0.0
    return max(_safe_float(fee_schedule.get("rate"), 0.0), 0.0)


def fee_per_share(price: float | None, fee_schedule: dict | None) -> float | None:
    if price is None:
        return None
    price = float(price)
    if price <= 0:
        return 0.0
    return fee_rate_from_schedule(fee_schedule) * price * (1.0 - price)


def net_bid_price(best_bid: float | None, fee_schedule: dict | None) -> float | None:
    if best_bid is None:
        return None
    bid = max(float(best_bid), 0.0)
    fee = fee_per_share(bid, fee_schedule) or 0.0
    return max(bid - fee, 0.0)


def observed_sigma_per_sqrt_second(state: Btc5mExitShadowState) -> float | None:
    if len(state.marks) < 3:
        return None
    values: list[float] = []
    for (left_time, left_price), (right_time, right_price) in zip(state.marks, state.marks[1:]):
        dt = max((right_time - left_time).total_seconds(), 0.0)
        if dt <= 0:
            continue
        values.append(((right_price - left_price) ** 2) / dt)
    if not values:
        return None
    sigma = math.sqrt(sum(values) / len(values))
    return sigma if sigma > 0 else None


def fair_value_from_context(context: Btc5mExitContext) -> float | None:
    seconds_left = max(float(context.seconds_left), 0.0)
    distance = float(context.btc_current_price) - float(context.btc_start_price)
    side = context.side.lower()
    if seconds_left <= 0:
        if side == "up":
            return 1.0 if distance >= 0 else 0.0
        return 1.0 if distance < 0 else 0.0

    sigma = observed_sigma_per_sqrt_second(context.shadow_state)
    if sigma is None:
        return None
    remaining_sd = sigma * math.sqrt(seconds_left)
    if remaining_sd <= 0:
        return None

    p_up = _normal_cdf(distance / remaining_sd)
    return p_up if side == "up" else 1.0 - p_up


def evaluate_exit_policy(context: Btc5mExitContext) -> Btc5mExitDecision:
    best_bid = _book_value(context.held_book, "best_bid")
    context.shadow_state.record_mark(
        recorded_at=context.recorded_at,
        btc_price=context.btc_current_price,
        best_bid=best_bid,
    )
    fair_value = fair_value_from_context(context)
    net_bid = net_bid_price(best_bid, context.fee_schedule)
    pnl = _hypothetical_pnl(context, net_bid)
    diagnostics = _base_diagnostics(context, best_bid=best_bid, fair_value=fair_value)
    if net_bid is None:
        return _hold(context, "no_bid", "No bid available for a shadow exit.", net_bid, fair_value, pnl, diagnostics)

    profile_name = context.profile_name.lower()
    style = context.strategy_style.lower()
    if style == "cheap_side" or profile_name == "cheap_side":
        return _evaluate_cheap_side(context, net_bid, fair_value, pnl, diagnostics)
    if profile_name == "late_capture" or style == "probability_capture":
        return _evaluate_late_capture(context, net_bid, fair_value, pnl, diagnostics)
    if profile_name == "late":
        return _evaluate_late(context, net_bid, fair_value, pnl, diagnostics)
    if profile_name == "confidence":
        return _evaluate_confidence(context, net_bid, fair_value, pnl, diagnostics)
    return _evaluate_momentum(context, net_bid, fair_value, pnl, diagnostics)


def _evaluate_cheap_side(
    context: Btc5mExitContext,
    net_bid: float,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    model = "cheap_side_reversion"
    fv_premium = _profile_float(context.profile, "tp_fair_value_premium", 0.03)
    profit_target = _profile_float(context.profile, "tp_profit_target", 0.08)
    trail_arm = _profile_float(context.profile, "tp_trail_arm", 0.10)
    trail_giveback = _profile_float(context.profile, "tp_trail_giveback", 0.06)
    diagnostics.update(
        {
            "tp_fair_value_premium": fv_premium,
            "tp_profit_target": profit_target,
            "tp_trail_arm": trail_arm,
            "tp_trail_giveback": trail_giveback,
        }
    )
    if fair_value is not None and net_bid >= fair_value + fv_premium and net_bid >= context.entry_price + profit_target:
        return _exit(context, model, "fair_value_take_profit", "Net bid is above fair value and entry profit target.", net_bid, fair_value, pnl, diagnostics)

    high_bid = context.shadow_state.high_bid
    if high_bid is not None and high_bid >= context.entry_price + trail_arm:
        context.shadow_state.trailing_armed = True
    if context.shadow_state.trailing_armed and high_bid is not None and _book_value(context.held_book, "best_bid") <= high_bid - trail_giveback:
        return _exit(context, model, "trailing_take_profit", "Cheap-side position gave back enough post-entry profit.", net_bid, fair_value, pnl, diagnostics)

    adverse_sigma = _adverse_remaining_sigma(context)
    diagnostics["adverse_remaining_sigma"] = _round_optional(adverse_sigma)
    if context.seconds_left <= 45 and adverse_sigma is not None and adverse_sigma >= 2.0:
        return _exit(context, model, "thesis_dead", "Held side is far out of the money with little time left.", net_bid, fair_value, pnl, diagnostics)

    return _hold(context, "cheap_side_hold", "No cheap-side shadow exit trigger fired.", net_bid, fair_value, pnl, diagnostics)


def _evaluate_momentum(
    context: Btc5mExitContext,
    net_bid: float,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    profile_name = context.profile_name.lower()
    reversal_ratio = {"conservative": 0.45, "balanced": 0.35, "aggressive": 0.25}.get(profile_name, 0.35)
    trail_arm = {"conservative": 0.06, "balanced": 0.08, "aggressive": 0.10}.get(profile_name, 0.08)
    trail_giveback = {"conservative": 0.04, "balanced": 0.05, "aggressive": 0.07}.get(profile_name, 0.05)
    diagnostics.update(
        {
            "reversal_ratio": reversal_ratio,
            "trail_arm": trail_arm,
            "trail_giveback": trail_giveback,
        }
    )
    if _momentum_reversed(context, reversal_ratio):
        return _exit(context, "momentum", "btc_reversal_stop", "BTC gave back enough of the entry impulse.", net_bid, fair_value, pnl, diagnostics)

    high_bid = context.shadow_state.high_bid
    if high_bid is not None and high_bid >= context.entry_price + trail_arm:
        context.shadow_state.trailing_armed = True
    if context.shadow_state.trailing_armed and high_bid is not None and _book_value(context.held_book, "best_bid") <= high_bid - trail_giveback:
        return _exit(context, "momentum", "trailing_stop", "Momentum position gave back enough post-entry profit.", net_bid, fair_value, pnl, diagnostics)

    if fair_value is not None and net_bid >= fair_value + 0.03:
        return _exit(context, "momentum", "fair_value_overbid", "Net bid is materially above fair value.", net_bid, fair_value, pnl, diagnostics)
    return _hold(context, "momentum_hold", "No momentum shadow exit trigger fired.", net_bid, fair_value, pnl, diagnostics)


def _evaluate_confidence(
    context: Btc5mExitContext,
    net_bid: float,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    midpoint = _book_value(context.held_book, "midpoint")
    diagnostics["held_midpoint"] = _round_optional(midpoint)
    if _start_crossed_against_side(context) and net_bid >= 0.10:
        return _exit(context, "confidence", "btc_thesis_break", "BTC crossed the start price against the held side.", net_bid, fair_value, pnl, diagnostics)
    if (midpoint is not None and midpoint <= 0.58) or (fair_value is not None and fair_value <= 0.60):
        return _exit(context, "confidence", "favorite_flip", "Held side is no longer clearly favored.", net_bid, fair_value, pnl, diagnostics)
    if fair_value is not None and net_bid >= fair_value + 0.03:
        return _exit(context, "confidence", "fair_value_overbid", "Net bid is materially above fair value.", net_bid, fair_value, pnl, diagnostics)
    return _hold(context, "confidence_hold", "No confidence shadow exit trigger fired.", net_bid, fair_value, pnl, diagnostics)


def _evaluate_late(
    context: Btc5mExitContext,
    net_bid: float,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    if _start_crossed_against_side(context) and net_bid >= 0.05:
        return _exit(context, "late_emergency", "start_price_flip", "BTC crossed against the held side late in the window.", net_bid, fair_value, pnl, diagnostics)
    if fair_value is not None and net_bid >= fair_value + 0.07:
        return _exit(context, "late_emergency", "stale_book_gift", "Late net bid is far above fair value.", net_bid, fair_value, pnl, diagnostics)
    return _hold(context, "late_hold", "No late emergency shadow exit trigger fired.", net_bid, fair_value, pnl, diagnostics)


def _evaluate_late_capture(
    context: Btc5mExitContext,
    net_bid: float,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    threshold = max(0.05, (fair_value + 0.03) if fair_value is not None else 0.05)
    diagnostics["late_capture_exit_threshold"] = round(threshold, 6)
    if _start_crossed_against_side(context) and net_bid >= threshold:
        return _exit(context, "late_capture", "start_price_flip", "Late-capture side flipped against the position.", net_bid, fair_value, pnl, diagnostics)
    if fair_value is not None and fair_value <= 0.75 and net_bid >= threshold:
        return _exit(context, "late_capture", "fair_value_collapse", "Late-capture fair value collapsed enough to shadow exit.", net_bid, fair_value, pnl, diagnostics)
    return _hold(context, "late_capture_hold", "No late-capture catastrophe trigger fired.", net_bid, fair_value, pnl, diagnostics)


def _momentum_reversed(context: Btc5mExitContext, ratio: float) -> bool:
    entry_move = context.entry_btc_move_usd
    if entry_move is None or abs(entry_move) <= 0:
        return False
    if context.side == "up" and entry_move > 0:
        return context.btc_move_usd <= entry_move * ratio
    if context.side == "down" and entry_move < 0:
        return -context.btc_move_usd <= (-entry_move) * ratio
    return False


def _adverse_remaining_sigma(context: Btc5mExitContext) -> float | None:
    sigma = observed_sigma_per_sqrt_second(context.shadow_state)
    if sigma is None:
        return None
    remaining_sd = sigma * math.sqrt(max(float(context.seconds_left), 0.0))
    if remaining_sd <= 0:
        return None
    if context.side == "up":
        adverse_distance = max(context.btc_start_price - context.btc_current_price, 0.0)
    else:
        adverse_distance = max(context.btc_current_price - context.btc_start_price, 0.0)
    return adverse_distance / remaining_sd


def _start_crossed_against_side(context: Btc5mExitContext) -> bool:
    if context.side == "up":
        return context.btc_current_price < context.btc_start_price
    return context.btc_current_price > context.btc_start_price


def _exit(
    context: Btc5mExitContext,
    model: str,
    trigger: str,
    reason: str,
    net_bid: float | None,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    return Btc5mExitDecision(
        action="shadow_exit",
        model=model,
        trigger=trigger,
        reason=reason,
        net_bid=net_bid,
        fair_value=fair_value,
        hypothetical_pnl=pnl,
        diagnostics=diagnostics,
    )


def _hold(
    context: Btc5mExitContext,
    trigger: str,
    reason: str,
    net_bid: float | None,
    fair_value: float | None,
    pnl: float | None,
    diagnostics: dict,
) -> Btc5mExitDecision:
    return Btc5mExitDecision(
        action="hold",
        model=f"{context.profile_name.lower()}_exit",
        trigger=trigger,
        reason=reason,
        net_bid=net_bid,
        fair_value=fair_value,
        hypothetical_pnl=pnl,
        diagnostics=diagnostics,
    )


def _base_diagnostics(
    context: Btc5mExitContext,
    *,
    best_bid: float | None,
    fair_value: float | None,
) -> dict:
    sigma = observed_sigma_per_sqrt_second(context.shadow_state)
    return {
        "best_bid": _round_optional(best_bid),
        "entry_price": round(context.entry_price, 6),
        "btc_move_usd": round(context.btc_move_usd, 4),
        "entry_btc_move_usd": _round_optional(context.entry_btc_move_usd, 4),
        "seconds_left": round(float(context.seconds_left), 3),
        "fair_value_available": fair_value is not None,
        "observed_sigma_per_sqrt_second": _round_optional(sigma, 8),
        "mark_count": len(context.shadow_state.marks),
        "trailing_armed": context.shadow_state.trailing_armed,
        "high_bid": _round_optional(context.shadow_state.high_bid),
    }


def _hypothetical_pnl(context: Btc5mExitContext, net_bid: float | None) -> float | None:
    if net_bid is None:
        return None
    return (float(context.shares) * float(net_bid)) - float(context.amount)


def _book_value(book: Any, name: str) -> float | None:
    value = getattr(book, name, None)
    if callable(value):
        value = value()
    return _optional_float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _profile_float(profile: Any, name: str, default: float) -> float:
    """Read a numeric profile attribute, falling back to the historical default.

    Works for both Btc5mProfile dataclasses and lightweight stand-ins (e.g. test
    SimpleNamespaces) that omit the take-profit tuning attributes entirely.
    """
    parsed = _optional_float(getattr(profile, name, None))
    return default if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _round_optional(value: float | None, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
