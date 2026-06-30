"""Forward BTC 5-minute profile opportunity harness.

The harness polls the live BTC 5m market once per cycle, then evaluates every
selected profile against that same market, BTC price, and UP/DOWN book snapshot.
It writes only dry-run ledger rows under its own run directory.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.config import (
    BTC5M_OPPORTUNITY_TARGET_MARKETS,
    BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
    BTC5M_PAPER_SETTLEMENT_SOURCE,
    BTC5M_POLL_SECONDS,
    LOGS_DIR,
)
from src.polymarket.btc_5m import (
    BTC5M_ORDER_TYPE,
    BTC5M_WINDOW_SECONDS,
    Btc5mMarket,
    Btc5mMarketClient,
    Btc5mProfile,
    BtcPriceSnapshot,
    OrderBookSummary,
    PublicClobOrderBookClient,
    _available_cash,
    _book_for_direction,
    _coerce_utc,
    _entry_price_for_book,
    _entry_window_skip,
    _parse_profile_names,
    _place_order,
    _risk_check,
    _safe_float,
    _sized_notional,
    _strategy_style,
    _trading_hours_skip,
    build_btc5m_settlement_client,
    build_btc_price_client_for_profile,
    evaluate_signal,
    resolve_btc5m_profile,
    settle_btc5m_paper_ledger,
    utcnow,
)
from src.polymarket.btc5m_exit import (
    Btc5mExitContext,
    Btc5mExitShadowState,
    evaluate_exit_policy,
)
from src.polymarket.tracker import BetLedger

logger = logging.getLogger(__name__)

LIVE_LIKE_MODE = "live_like"
ALL_SNAPSHOTS_MODE = "all_snapshots"
BTC5M_OPPORTUNITY_MODES = (LIVE_LIKE_MODE, ALL_SNAPSHOTS_MODE)
# Used only when the comparison harness runs with risk limits disabled: exposure
# is widened to this (not 0, which would zero position sizing and block trades).
UNBOUNDED_EXPOSURE_USD = 1_000_000_000.0
HOLD_TO_SETTLEMENT_POLICY = "hold_to_settlement"
TAKE_PROFIT_EXIT_POLICY = "take_profit_exit"
BTC5M_OPPORTUNITY_EXIT_POLICIES = (HOLD_TO_SETTLEMENT_POLICY, TAKE_PROFIT_EXIT_POLICY)
BTC5M_OPPORTUNITY_PROFILE_GROUP_DEFS = {
    "late_capture": {
        "label": "Track A - late capture",
        "primary_mode": LIVE_LIKE_MODE,
        "primary_exit_policy": HOLD_TO_SETTLEMENT_POLICY,
        "description": "Probability-capture profiles; judge deployment candidates on live-like hold-to-settlement.",
    },
    "cheap_side": {
        "label": "Track B - cheap-side reversion",
        "primary_mode": LIVE_LIKE_MODE,
        "primary_exit_policy": TAKE_PROFIT_EXIT_POLICY,
        "description": "Cheap-side mean-reversion profiles; judge only on live-like take-profit exits.",
    },
    "other": {
        "label": "Other profiles",
        "primary_mode": LIVE_LIKE_MODE,
        "primary_exit_policy": HOLD_TO_SETTLEMENT_POLICY,
        "description": "Profiles outside the late-capture and cheap-side experiment tracks.",
    },
}


@dataclass(frozen=True)
class Btc5mOpportunitySnapshot:
    market: Btc5mMarket
    price_snapshot: BtcPriceSnapshot
    up_book: OrderBookSummary
    down_book: OrderBookSummary
    now: datetime

    @property
    def market_key(self) -> str:
        return _market_key(self.market)


@dataclass
class OpportunityProfileStats:
    mode: str
    profile: str
    ledger_path: Path
    markets_seen: set[str] = field(default_factory=set)
    snapshots_seen: int = 0
    entry_window_hit_markets: set[str] = field(default_factory=set)
    outside_entry_window: int = 0
    outside_trading_hours: int = 0
    signal_skips_by_reason: Counter[str] = field(default_factory=Counter)
    risk_skips_by_reason: Counter[str] = field(default_factory=Counter)
    signal_trade_markets: set[str] = field(default_factory=set)
    actual_trade_markets: set[str] = field(default_factory=set)
    trade_outside_entry_window: int = 0
    settlement_errors: list[str] = field(default_factory=list)

    def to_report(self, ledger: BetLedger) -> dict:
        summary = ledger.get_summary(fresh=True)
        return {
            "mode": self.mode,
            "profile": self.profile,
            "ledger_path": str(self.ledger_path),
            "markets_seen": len(self.markets_seen),
            "snapshots_seen": self.snapshots_seen,
            "entry_window_hit_markets": len(self.entry_window_hit_markets),
            "outside_entry_window": self.outside_entry_window,
            "outside_trading_hours": self.outside_trading_hours,
            "signal_skips_by_reason": dict(sorted(self.signal_skips_by_reason.items())),
            "risk_skips_by_reason": dict(sorted(self.risk_skips_by_reason.items())),
            "signal_trade_markets": len(self.signal_trade_markets),
            "actual_trade_markets": len(self.actual_trade_markets),
            "trade_outside_entry_window": self.trade_outside_entry_window,
            "settled_trades": summary["settled_bets"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "realized_pnl": round(summary["realized_pnl"], 2),
            "roi": round(summary["roi"], 4),
            "win_rate": round(summary["win_rate"], 4),
            "open_bets": summary["open_bets"],
            "total_bets": summary["total_bets"],
            "settlement_errors": list(self.settlement_errors[-20:]),
        }


@dataclass
class VirtualTakeProfitPosition:
    mode: str
    profile: str
    ledger_bet_id: int
    market_key: str
    market_slug: str
    market_id: str
    condition_id: str
    side: str
    entry_price: float
    shares: float
    amount: float
    entry_time: datetime
    window_start: datetime
    window_end: datetime
    btc_start_price: float
    btc_entry_price: float
    entry_btc_move_usd: float
    strategy_style: str
    fee_schedule: dict
    bet: dict
    shadow_state: Btc5mExitShadowState = field(default_factory=Btc5mExitShadowState)
    status: str = "open"
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_trigger: str | None = None
    exit_decision: dict | None = None
    realized_pnl: float | None = None
    hold_status: str = "open"
    hold_pnl: float | None = None
    delta_pnl_vs_hold: float | None = None

    def update_hold_result(self, bet: dict | None) -> None:
        if not bet:
            return
        status = str(bet.get("status") or "")
        self.hold_status = status
        result_pnl = bet.get("result_pnl")
        if result_pnl is not None:
            self.hold_pnl = round(float(result_pnl), 2)
        if status in {"won", "lost"} and self.status == "open":
            self.status = "settled"
            self.realized_pnl = self.hold_pnl
        if self.realized_pnl is not None and self.hold_pnl is not None:
            self.delta_pnl_vs_hold = round(float(self.realized_pnl) - float(self.hold_pnl), 2)

    def close_early(self, *, decision: dict, recorded_at: datetime) -> None:
        self.status = "early_exited"
        self.exit_time = recorded_at
        self.exit_price = decision.get("net_bid")
        self.exit_reason = str(decision.get("reason") or "")
        self.exit_trigger = str(decision.get("trigger") or "shadow_exit")
        self.exit_decision = dict(decision)
        pnl = decision.get("hypothetical_pnl")
        self.realized_pnl = round(float(pnl), 2) if pnl is not None else None
        if self.realized_pnl is not None and self.hold_pnl is not None:
            self.delta_pnl_vs_hold = round(float(self.realized_pnl) - float(self.hold_pnl), 2)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "exit_policy": TAKE_PROFIT_EXIT_POLICY,
            "ledger_bet_id": self.ledger_bet_id,
            "market_key": self.market_key,
            "market_slug": self.market_slug,
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "side": self.side,
            "entry_price": round(self.entry_price, 6),
            "shares": round(self.shares, 4),
            "amount": round(self.amount, 2),
            "entry_time": self.entry_time.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "btc_start_price": round(self.btc_start_price, 4),
            "btc_entry_price": round(self.btc_entry_price, 4),
            "entry_btc_move_usd": round(self.entry_btc_move_usd, 4),
            "status": self.status,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_trigger": self.exit_trigger,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
            "hold_status": self.hold_status,
            "hold_pnl": self.hold_pnl,
            "delta_pnl_vs_hold": self.delta_pnl_vs_hold,
        }


def build_btc5m_opportunity_paths(run_id: str | None = None) -> tuple[str, Path, Path, Path]:
    """Return default run_id, ledger_dir, report_path, and markdown_path."""
    resolved_run_id = str(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = LOGS_DIR / "btc5m_opportunity" / resolved_run_id
    report_path = run_dir / "report.json"
    return resolved_run_id, run_dir / "ledgers", report_path, report_path.with_suffix(".md")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _mode_profile_ledger_path(ledger_dir: Path, mode: str, profile_name: str) -> Path:
    return Path(ledger_dir) / _safe_name(mode) / f"{_safe_name(profile_name)}.json"


def _market_key(market: Btc5mMarket) -> str:
    return str(market.condition_id or market.slug or market.market_id).strip()


def _market_record(market: Btc5mMarket) -> dict:
    return {
        "key": _market_key(market),
        "slug": market.slug,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "window_start": market.window_start.isoformat(),
        "window_end": market.window_end.isoformat(),
    }


def _ledger_has_market_trade(ledger: BetLedger, market: Btc5mMarket) -> bool:
    market_key = _market_key(market)
    for bet in ledger.get_bets(fresh=True):
        if str(bet.get("order_type", "") or "") != BTC5M_ORDER_TYPE:
            continue
        bet_condition = str(bet.get("condition_id", "") or "").strip()
        bet_slug = str(bet.get("market_slug", "") or "").strip()
        bet_market_id = str(bet.get("market_id", "") or "").strip()
        if bet_condition and bet_condition == market.condition_id:
            return True
        if bet_slug and bet_slug == market.slug:
            return True
        if bet_market_id and bet_market_id == market.market_id:
            return True
        if market_key and market_key in {bet_condition, bet_slug, bet_market_id}:
            return True
    return False


def _ledger_bet_by_id(ledger: BetLedger, bet_id: int) -> dict | None:
    for bet in ledger.get_bets(fresh=True):
        if int(bet.get("id", -1)) == int(bet_id):
            return dict(bet)
    return None


def _ledger_market_key(bet: dict) -> str:
    for field_name in ("condition_id", "market_slug", "market_id"):
        value = str(bet.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _report_datetime(value, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        return _coerce_utc(value)
    text = str(value).strip()
    if not text:
        return default
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _coerce_utc(datetime.fromisoformat(text))
    except ValueError:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_counter(value) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, count in value.items():
        counter[str(key)] = _safe_int(count)
    return counter


def _counted_key_set(prefix: str, count: int, keys: list[str] | tuple[str, ...] = ()) -> set[str]:
    target = max(_safe_int(count), 0)
    values: list[str] = []
    seen: set[str] = set()
    for key in keys:
        text = str(key or "").strip()
        if not text or text in seen:
            continue
        values.append(text)
        seen.add(text)
        if len(values) >= target:
            return set(values)
    while len(values) < target:
        values.append(f"__resume_{prefix}_{len(values)}")
    return set(values)


def _has_open_btc5m_bets(ledgers: dict[tuple[str, str], BetLedger]) -> bool:
    for ledger in ledgers.values():
        for bet in ledger.get_open_bets(fresh=True):
            if str(bet.get("order_type", "") or "") == BTC5M_ORDER_TYPE:
                return True
    return False


def _closed_positions(positions: list[VirtualTakeProfitPosition]) -> list[VirtualTakeProfitPosition]:
    return [position for position in positions if position.realized_pnl is not None]


def _position_pnl_summary(positions: list[VirtualTakeProfitPosition]) -> dict:
    closed = _closed_positions(positions)
    early_exits = [position for position in positions if position.status == "early_exited"]
    fallback_settled = [position for position in positions if position.status == "settled"]
    hold_settled = [position for position in positions if position.hold_pnl is not None]
    deltas = [position.delta_pnl_vs_hold for position in positions if position.delta_pnl_vs_hold is not None]
    pnl = sum(float(position.realized_pnl or 0.0) for position in closed)
    wagered = sum(float(position.amount or 0.0) for position in closed)
    wins = sum(1 for position in closed if float(position.realized_pnl or 0.0) > 0)
    losses = sum(1 for position in closed if float(position.realized_pnl or 0.0) <= 0)
    exit_reasons = Counter(position.exit_trigger or "early_exit" for position in early_exits)
    return {
        "exit_policy": TAKE_PROFIT_EXIT_POLICY,
        "total_trades": len(positions),
        "open_trades": len([position for position in positions if position.status == "open"]),
        "closed_trades": len(closed),
        "settled_trades": len(closed),
        "early_exits": len(early_exits),
        "settlement_fallbacks": len(fallback_settled),
        "paired_hold_settled": len(hold_settled),
        "wins": wins,
        "losses": losses,
        "realized_pnl": round(pnl, 2),
        "roi": round((pnl / wagered) if wagered else 0.0, 4),
        "win_rate": round((wins / len(closed)) if closed else 0.0, 4),
        "avg_exit_pnl": round(
            sum(float(position.realized_pnl or 0.0) for position in early_exits) / len(early_exits),
            4,
        )
        if early_exits
        else None,
        "avg_hold_to_settlement_pnl": round(
            sum(float(position.hold_pnl or 0.0) for position in hold_settled) / len(hold_settled),
            4,
        )
        if hold_settled
        else None,
        "delta_pnl_vs_hold": round(sum(float(delta) for delta in deltas), 2),
        "avg_delta_pnl_vs_hold": round(sum(float(delta) for delta in deltas) / len(deltas), 4) if deltas else None,
        "exit_reasons_by_reason": dict(sorted(exit_reasons.items())),
    }


def _hold_policy_report(payload: dict) -> dict:
    settled = int(payload.get("settled_trades", 0) or 0)
    realized_pnl = float(payload.get("realized_pnl", 0.0) or 0.0)
    return {
        "exit_policy": HOLD_TO_SETTLEMENT_POLICY,
        "total_trades": int(payload.get("total_bets", 0) or 0),
        "open_trades": int(payload.get("open_bets", 0) or 0),
        "closed_trades": settled,
        "settled_trades": settled,
        "early_exits": 0,
        "settlement_fallbacks": settled,
        "wins": int(payload.get("wins", 0) or 0),
        "losses": int(payload.get("losses", 0) or 0),
        "realized_pnl": round(realized_pnl, 2),
        "roi": float(payload.get("roi", 0.0) or 0.0),
        "win_rate": float(payload.get("win_rate", 0.0) or 0.0),
        "avg_exit_pnl": None,
        "avg_hold_to_settlement_pnl": round((realized_pnl / settled) if settled else 0.0, 4) if settled else None,
        "delta_pnl_vs_hold": 0.0,
        "avg_delta_pnl_vs_hold": 0.0 if settled else None,
        "exit_reasons_by_reason": {},
    }


def _rank_mode_profiles(mode_profiles: dict[str, dict]) -> list[dict]:
    ranked = []
    for profile_name, payload in mode_profiles.items():
        ranked.append(
            {
                "profile": profile_name,
                "settled_trades": int(payload.get("settled_trades", 0) or 0),
                "actual_trade_markets": int(payload.get("actual_trade_markets", 0) or 0),
                "realized_pnl": float(payload.get("realized_pnl", 0.0) or 0.0),
                "roi": float(payload.get("roi", 0.0) or 0.0),
                "win_rate": float(payload.get("win_rate", 0.0) or 0.0),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["settled_trades"] > 0,
            item["realized_pnl"],
            item["roi"],
            item["win_rate"],
            item["actual_trade_markets"],
        ),
        reverse=True,
    )


def _rank_exit_policy_profiles(mode_profiles: dict[str, dict], exit_policy: str) -> list[dict]:
    ranked = []
    for profile_name, payload in mode_profiles.items():
        policy = (payload.get("exit_policies") or {}).get(exit_policy, {})
        ranked.append(
            {
                "profile": profile_name,
                "exit_policy": exit_policy,
                "settled_trades": int(policy.get("settled_trades", 0) or 0),
                "early_exits": int(policy.get("early_exits", 0) or 0),
                "actual_trade_markets": int(payload.get("actual_trade_markets", 0) or 0),
                "realized_pnl": float(policy.get("realized_pnl", 0.0) or 0.0),
                "roi": float(policy.get("roi", 0.0) or 0.0),
                "win_rate": float(policy.get("win_rate", 0.0) or 0.0),
                "delta_pnl_vs_hold": float(policy.get("delta_pnl_vs_hold", 0.0) or 0.0),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["settled_trades"] > 0,
            item["realized_pnl"],
            item["roi"],
            item["win_rate"],
            item["actual_trade_markets"],
        ),
        reverse=True,
    )


def _profile_group_key(profile: Btc5mProfile) -> str:
    style = _strategy_style(profile)
    name = profile.name.strip().lower()
    if style == "cheap_side" or name.startswith("cheap_"):
        return "cheap_side"
    if style == "probability_capture" or name.startswith("late_capture"):
        return "late_capture"
    return "other"


def _build_profile_groups(profiles: dict[str, Btc5mProfile]) -> dict[str, dict]:
    grouped_names: dict[str, list[str]] = {}
    for profile_name, profile in profiles.items():
        grouped_names.setdefault(_profile_group_key(profile), []).append(profile_name)

    groups: dict[str, dict] = {}
    for group_key, profile_names in grouped_names.items():
        definition = BTC5M_OPPORTUNITY_PROFILE_GROUP_DEFS[group_key]
        groups[group_key] = {
            **definition,
            "profiles": profile_names,
        }
    return groups


def _filter_mode_profiles(mode_profiles: dict[str, dict], profile_names: list[str]) -> dict[str, dict]:
    wanted = set(profile_names)
    return {
        profile_name: payload
        for profile_name, payload in mode_profiles.items()
        if profile_name in wanted
    }


def _build_group_rankings(profiles: dict[str, dict[str, dict]], groups: dict[str, dict]) -> dict[str, dict]:
    group_rankings: dict[str, dict] = {}
    for mode in BTC5M_OPPORTUNITY_MODES:
        group_rankings[mode] = {}
        mode_profiles = profiles.get(mode) or {}
        for group_key, group in groups.items():
            group_mode_profiles = _filter_mode_profiles(mode_profiles, group.get("profiles") or [])
            group_rankings[mode][group_key] = {
                "primary_exit_policy": group["primary_exit_policy"],
                "ranking": _rank_exit_policy_profiles(group_mode_profiles, group["primary_exit_policy"]),
                "exit_policy_ranking": {
                    policy: _rank_exit_policy_profiles(group_mode_profiles, policy)
                    for policy in BTC5M_OPPORTUNITY_EXIT_POLICIES
                },
            }
    return group_rankings


class Btc5mOpportunityHarness:
    def __init__(
        self,
        *,
        profile_names: str | list[str] | tuple[str, ...] = "all",
        ledger_dir: Path,
        settlement_source: str | None = BTC5M_PAPER_SETTLEMENT_SOURCE,
        settlement_delay_seconds: float = BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
        market_client: Btc5mMarketClient | None = None,
        price_client=None,
        book_client: PublicClobOrderBookClient | None = None,
        settlement_client=None,
        ignore_risk_limits: bool = False,
    ):
        self.profile_names = _parse_profile_names(profile_names)
        self.ignore_risk_limits = bool(ignore_risk_limits)
        self.profiles: dict[str, Btc5mProfile] = {
            name: self._prepare_profile(name) for name in self.profile_names
        }
        self.ledger_dir = Path(ledger_dir)
        self.settlement_source = str(settlement_source or BTC5M_PAPER_SETTLEMENT_SOURCE or "polymarket")
        self.settlement_delay_seconds = float(settlement_delay_seconds)
        self.market_client = market_client or Btc5mMarketClient()
        self.snapshot_profile = self.profiles[self.profile_names[0]]
        self.price_client = price_client or build_btc_price_client_for_profile(self.snapshot_profile)
        self.book_client = book_client or PublicClobOrderBookClient()
        self.settlement_client = settlement_client or build_btc5m_settlement_client(
            settlement_source=self.settlement_source,
            price_client=self.price_client,
            market_client=self.market_client,
        )
        self.ledgers: dict[tuple[str, str], BetLedger] = {}
        self.stats: dict[tuple[str, str], OpportunityProfileStats] = {}
        self.take_profit_positions: dict[tuple[str, str], list[VirtualTakeProfitPosition]] = {}
        self.market_records: dict[str, dict] = {}
        self.cycles = 0

        for mode in BTC5M_OPPORTUNITY_MODES:
            for profile_name in self.profile_names:
                ledger_path = _mode_profile_ledger_path(self.ledger_dir, mode, profile_name)
                self.ledgers[(mode, profile_name)] = BetLedger(path=ledger_path)
                self.stats[(mode, profile_name)] = OpportunityProfileStats(
                    mode=mode,
                    profile=profile_name,
                    ledger_path=ledger_path,
                )
                self.take_profit_positions[(mode, profile_name)] = []

    def resume_from_report(self, report: dict) -> None:
        """Restore aggregate comparison state from a prior report."""
        if not isinstance(report, dict):
            return
        self.cycles = max(self.cycles, _safe_int(report.get("cycles")))
        self._resume_market_records(report)
        market_keys = list(self.market_records.keys())
        profiles = report.get("profiles") or {}
        for mode in BTC5M_OPPORTUNITY_MODES:
            mode_profiles = profiles.get(mode) or {}
            for profile_name in self.profile_names:
                key = (mode, profile_name)
                payload = mode_profiles.get(profile_name) or {}
                if not isinstance(payload, dict):
                    continue
                ledger = self.ledgers[key]
                ledger_market_keys = [
                    market_key
                    for bet in ledger.get_bets(fresh=True)
                    if str(bet.get("order_type", "") or "") == BTC5M_ORDER_TYPE
                    for market_key in [_ledger_market_key(bet)]
                    if market_key
                ]
                self._resume_stats(
                    self.stats[key],
                    payload,
                    market_keys=market_keys,
                    ledger_market_keys=ledger_market_keys,
                )
        self._resume_take_profit_positions(report)
        self._sync_take_profit_hold_results()

    def _resume_market_records(self, report: dict) -> None:
        records = report.get("markets") or []
        if isinstance(records, list):
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                key = str(
                    record.get("key")
                    or record.get("condition_id")
                    or record.get("slug")
                    or record.get("market_id")
                    or f"__resume_market_{index}"
                ).strip()
                if not key:
                    continue
                restored = dict(record)
                restored["key"] = key
                self.market_records[key] = restored
        target_count = max(_safe_int(report.get("distinct_markets_seen")), 0)
        while len(self.market_records) < target_count:
            key = f"__resume_market_{len(self.market_records)}"
            self.market_records[key] = {"key": key, "resume_placeholder": True}

    def _resume_stats(
        self,
        stats: OpportunityProfileStats,
        payload: dict,
        *,
        market_keys: list[str],
        ledger_market_keys: list[str],
    ) -> None:
        prefix = f"{stats.mode}_{stats.profile}"
        stats.markets_seen = _counted_key_set(
            f"{prefix}_markets_seen",
            _safe_int(payload.get("markets_seen")),
            market_keys,
        )
        stats.snapshots_seen = _safe_int(payload.get("snapshots_seen"))
        stats.entry_window_hit_markets = _counted_key_set(
            f"{prefix}_entry_window_hit_markets",
            _safe_int(payload.get("entry_window_hit_markets")),
            market_keys,
        )
        stats.outside_entry_window = _safe_int(payload.get("outside_entry_window"))
        stats.outside_trading_hours = _safe_int(payload.get("outside_trading_hours"))
        stats.signal_skips_by_reason = _safe_counter(payload.get("signal_skips_by_reason"))
        stats.risk_skips_by_reason = _safe_counter(payload.get("risk_skips_by_reason"))
        stats.signal_trade_markets = _counted_key_set(
            f"{prefix}_signal_trade_markets",
            _safe_int(payload.get("signal_trade_markets")),
            ledger_market_keys,
        )
        stats.actual_trade_markets = _counted_key_set(
            f"{prefix}_actual_trade_markets",
            _safe_int(payload.get("actual_trade_markets")),
            ledger_market_keys,
        )
        stats.trade_outside_entry_window = _safe_int(payload.get("trade_outside_entry_window"))
        errors = payload.get("settlement_errors") or []
        if isinstance(errors, list):
            stats.settlement_errors = [str(error) for error in errors]

    def _resume_take_profit_positions(self, report: dict) -> None:
        payload = report.get("take_profit_positions") or {}
        if not isinstance(payload, dict):
            return
        for mode in BTC5M_OPPORTUNITY_MODES:
            mode_positions = payload.get(mode) or {}
            if not isinstance(mode_positions, dict):
                continue
            for profile_name in self.profile_names:
                rows = mode_positions.get(profile_name) or []
                if not isinstance(rows, list):
                    continue
                key = (mode, profile_name)
                ledger = self.ledgers[key]
                positions: list[VirtualTakeProfitPosition] = []
                for row in rows:
                    position = self._resume_take_profit_position(
                        mode=mode,
                        profile_name=profile_name,
                        row=row,
                        ledger=ledger,
                    )
                    if position is not None:
                        positions.append(position)
                self.take_profit_positions[key] = positions

    def _resume_take_profit_position(
        self,
        *,
        mode: str,
        profile_name: str,
        row: dict,
        ledger: BetLedger,
    ) -> VirtualTakeProfitPosition | None:
        if not isinstance(row, dict):
            return None
        ledger_bet_id = _safe_int(row.get("ledger_bet_id"), -1)
        if ledger_bet_id < 0:
            return None
        bet = _ledger_bet_by_id(ledger, ledger_bet_id) or {}
        window_start = _report_datetime(
            row.get("window_start") or bet.get("window_start") or bet.get("market_event_date"),
            datetime.now(timezone.utc),
        )
        window_end = _report_datetime(
            row.get("window_end") or bet.get("window_end") or bet.get("event_date"),
            window_start,
        )
        entry_time = _report_datetime(row.get("entry_time") or bet.get("placed_at"), window_start)
        exit_time = _report_datetime(row.get("exit_time"))
        fee_schedule = bet.get("fee_schedule") if isinstance(bet.get("fee_schedule"), dict) else {}
        return VirtualTakeProfitPosition(
            mode=mode,
            profile=profile_name,
            ledger_bet_id=ledger_bet_id,
            market_key=str(row.get("market_key") or _ledger_market_key(bet) or ""),
            market_slug=str(row.get("market_slug") or bet.get("market_slug") or ""),
            market_id=str(row.get("market_id") or bet.get("market_id") or ""),
            condition_id=str(row.get("condition_id") or bet.get("condition_id") or ""),
            side=str(row.get("side") or bet.get("side") or ""),
            entry_price=float(row.get("entry_price") or bet.get("price") or 0.0),
            shares=float(row.get("shares") or bet.get("shares") or 0.0),
            amount=float(row.get("amount") or bet.get("amount") or 0.0),
            entry_time=entry_time or datetime.now(timezone.utc),
            window_start=window_start or datetime.now(timezone.utc),
            window_end=window_end or window_start or datetime.now(timezone.utc),
            btc_start_price=float(row.get("btc_start_price") or bet.get("btc_window_start_price") or 0.0),
            btc_entry_price=float(row.get("btc_entry_price") or bet.get("btc_current_price") or 0.0),
            entry_btc_move_usd=float(row.get("entry_btc_move_usd") or bet.get("btc_move_usd") or 0.0),
            strategy_style=str(bet.get("strategy_style") or _strategy_style(self.profiles[profile_name])),
            fee_schedule=fee_schedule,
            bet=dict(bet),
            status=str(row.get("status") or "open"),
            exit_time=exit_time,
            exit_price=row.get("exit_price"),
            exit_reason=row.get("exit_reason"),
            exit_trigger=row.get("exit_trigger"),
            realized_pnl=row.get("realized_pnl"),
            hold_status=str(row.get("hold_status") or bet.get("status") or "open"),
            hold_pnl=row.get("hold_pnl"),
            delta_pnl_vs_hold=row.get("delta_pnl_vs_hold"),
        )

    def _prepare_profile(self, name: str) -> Btc5mProfile:
        profile = resolve_btc5m_profile(name)
        if self.ignore_risk_limits:
            # Comparison-only: drop every portfolio risk throttle so each profile
            # competes purely on its signal. The ONLY remaining gate is one order
            # per profile per market (duplicate_market dedup, enforced by the
            # harness). Exposure is widened (not zeroed) because a 0 cap would
            # zero the sized notional and block all trades.
            profile = replace(
                profile,
                max_trades_per_day=0,
                cooldown_windows_after_trade=0,
                daily_loss_limit_usd=0.0,
                max_open_exposure_usd=UNBOUNDED_EXPOSURE_USD,
            )
        return profile

    @property
    def distinct_markets_seen(self) -> int:
        return len(self.market_records)

    def has_open_take_profit_positions(self) -> bool:
        return any(
            position.status == "open"
            for positions in self.take_profit_positions.values()
            for position in positions
        )

    def settle_ledgers(self, *, now: datetime | None = None) -> dict:
        current = _coerce_utc(now)
        total = {"settled": 0, "skipped": 0, "errors": []}
        for key, ledger in self.ledgers.items():
            settlement = settle_btc5m_paper_ledger(
                ledger=ledger,
                settlement_client=self.settlement_client,
                now=current,
                settlement_delay_seconds=self.settlement_delay_seconds,
            )
            total["settled"] += settlement.get("settled", 0)
            total["skipped"] += settlement.get("skipped", 0)
            errors = list(settlement.get("errors", []) or [])
            if errors:
                self.stats[key].settlement_errors.extend(errors)
                total["errors"].extend(errors)
        self._sync_take_profit_hold_results()
        return total

    def _sync_take_profit_hold_results(self) -> None:
        for key, positions in self.take_profit_positions.items():
            ledger = self.ledgers[key]
            for position in positions:
                position.update_hold_result(_ledger_bet_by_id(ledger, position.ledger_bet_id))

    def _create_take_profit_counterpart(
        self,
        *,
        mode: str,
        profile: Btc5mProfile,
        ledger: BetLedger,
        order: dict,
        signal: dict,
        snapshot: Btc5mOpportunitySnapshot,
    ) -> None:
        ledger_bet_id = order.get("ledger_bet_id")
        if ledger_bet_id is None:
            return
        key = (mode, profile.name)
        if any(position.ledger_bet_id == int(ledger_bet_id) for position in self.take_profit_positions[key]):
            return
        bet = _ledger_bet_by_id(ledger, int(ledger_bet_id))
        if bet is None:
            return
        position = VirtualTakeProfitPosition(
            mode=mode,
            profile=profile.name,
            ledger_bet_id=int(ledger_bet_id),
            market_key=snapshot.market_key,
            market_slug=snapshot.market.slug,
            market_id=snapshot.market.market_id,
            condition_id=snapshot.market.condition_id,
            side=str(signal.get("direction") or bet.get("side") or ""),
            entry_price=float(order.get("price") or bet.get("price") or 0.0),
            shares=float(order.get("shares") or bet.get("shares") or 0.0),
            amount=float(order.get("amount") or bet.get("amount") or 0.0),
            entry_time=snapshot.now,
            window_start=snapshot.market.window_start,
            window_end=snapshot.market.window_end,
            btc_start_price=float(snapshot.price_snapshot.window_start_price),
            btc_entry_price=float(snapshot.price_snapshot.current_price),
            entry_btc_move_usd=float(snapshot.price_snapshot.move_usd),
            strategy_style=_strategy_style(profile),
            fee_schedule=snapshot.market.fee_schedule,
            bet=dict(bet),
        )
        self.take_profit_positions[key].append(position)

    def _process_take_profit_exits(self, snapshot: Btc5mOpportunitySnapshot) -> int:
        closed = 0
        for key, positions in self.take_profit_positions.items():
            mode, profile_name = key
            profile = self.profiles[profile_name]
            ledger = self.ledgers[key]
            for position in positions:
                position.update_hold_result(_ledger_bet_by_id(ledger, position.ledger_bet_id))
                if position.status != "open" or position.market_key != snapshot.market_key:
                    continue
                held_book = _book_for_direction(position.side, up_book=snapshot.up_book, down_book=snapshot.down_book)
                context = Btc5mExitContext(
                    bet=position.bet,
                    profile=profile,
                    profile_name=profile.name,
                    strategy_style=position.strategy_style,
                    side=position.side,
                    entry_price=position.entry_price,
                    shares=position.shares,
                    amount=position.amount,
                    btc_start_price=position.btc_start_price,
                    btc_current_price=snapshot.price_snapshot.current_price,
                    seconds_left=(position.window_end - snapshot.now).total_seconds(),
                    held_book=held_book,
                    fee_schedule=position.fee_schedule,
                    recorded_at=snapshot.now,
                    shadow_state=position.shadow_state,
                )
                decision = evaluate_exit_policy(context).to_dict()
                if decision.get("action") != "shadow_exit" or decision.get("net_bid") is None:
                    continue
                position.close_early(decision=decision, recorded_at=snapshot.now)
                closed += 1
                logger.info(
                    "BTC 5m opportunity take-profit exit: mode=%s profile=%s market=%s bet=%s trigger=%s pnl=%s",
                    mode,
                    profile.name,
                    position.market_slug,
                    position.ledger_bet_id,
                    position.exit_trigger,
                    position.realized_pnl,
                )
        return closed

    def run_exit_tracking_once(
        self,
        *,
        now: datetime | None = None,
        market_slug: str | None = None,
    ) -> dict:
        current = _coerce_utc(now)
        settlement = self.settle_ledgers(now=current)
        try:
            snapshot, idle = self.fetch_snapshot(now=current, market_slug=market_slug)
        except Exception as exc:
            logger.warning("BTC 5m opportunity exit snapshot fetch failed: %s", exc)
            return {"status": "error", "reason": f"exit_snapshot_fetch_failed: {exc}", "settlement": settlement}
        if snapshot is None:
            return {**(idle or {"status": "idle", "reason": "no_snapshot"}), "settlement": settlement}
        if snapshot.market_key not in self.market_records:
            return {
                "status": "idle",
                "reason": "current_market_not_in_sample",
                "market_slug": snapshot.market.slug,
                "settlement": settlement,
            }
        closed = self._process_take_profit_exits(snapshot)
        return {
            "status": "ok",
            "market_key": snapshot.market_key,
            "market_slug": snapshot.market.slug,
            "take_profit_exits": closed,
            "settlement": settlement,
        }

    def fetch_snapshot(
        self,
        *,
        now: datetime | None = None,
        market_slug: str | None = None,
    ) -> tuple[Btc5mOpportunitySnapshot | None, dict | None]:
        current = _coerce_utc(now)
        market = self.market_client.get_market(
            now=current,
            market_slug=market_slug,
            market_slug_prefix=self.snapshot_profile.market_slug_prefix,
        )
        if market is None:
            return None, {"status": "idle", "reason": "no_btc5m_market_found"}
        if current < market.window_start:
            return None, {
                "status": "idle",
                "reason": "BTC 5m market window has not started yet.",
                "reason_code": "market_not_started",
                "market_slug": market.slug,
            }
        if current >= market.window_end:
            return None, {
                "status": "idle",
                "reason": "BTC 5m market window is already closed.",
                "reason_code": "market_closed",
                "market_slug": market.slug,
            }
        if not market.active or market.closed or not market.accepting_orders:
            return None, {
                "status": "idle",
                "reason": "BTC 5m market is not accepting orders.",
                "reason_code": "market_not_tradeable",
                "market_slug": market.slug,
            }

        price_snapshot = self.price_client.get_snapshot(
            window_start=market.window_start,
            now=current,
        )
        up_book = self.book_client.summarize(market.token_up)
        down_book = self.book_client.summarize(market.token_down)
        return (
            Btc5mOpportunitySnapshot(
                market=market,
                price_snapshot=price_snapshot,
                up_book=up_book,
                down_book=down_book,
                now=current,
            ),
            None,
        )

    def run_once(
        self,
        *,
        now: datetime | None = None,
        market_slug: str | None = None,
    ) -> dict:
        current = _coerce_utc(now)
        self.cycles += 1
        settlement = self.settle_ledgers(now=current)
        try:
            snapshot, idle = self.fetch_snapshot(now=current, market_slug=market_slug)
        except Exception as exc:
            logger.warning("BTC 5m opportunity snapshot fetch failed: %s", exc)
            return {
                "status": "error",
                "reason": f"snapshot_fetch_failed: {exc}",
                "cycle": self.cycles,
                "settlement": settlement,
            }
        if snapshot is None:
            return {
                **(idle or {"status": "idle", "reason": "no_snapshot"}),
                "cycle": self.cycles,
                "settlement": settlement,
            }

        self.market_records[snapshot.market_key] = _market_record(snapshot.market)
        take_profit_exits = self._process_take_profit_exits(snapshot)
        evaluations: dict[str, dict[str, dict]] = {mode: {} for mode in BTC5M_OPPORTUNITY_MODES}
        for mode in BTC5M_OPPORTUNITY_MODES:
            for profile_name, profile in self.profiles.items():
                evaluations[mode][profile_name] = self._evaluate_profile_snapshot(
                    mode=mode,
                    profile=profile,
                    snapshot=snapshot,
                )

        return {
            "status": "ok",
            "cycle": self.cycles,
            "market_key": snapshot.market_key,
            "market_slug": snapshot.market.slug,
            "settlement": settlement,
            "take_profit_exits": take_profit_exits,
            "evaluations": evaluations,
        }

    def _evaluate_profile_snapshot(
        self,
        *,
        mode: str,
        profile: Btc5mProfile,
        snapshot: Btc5mOpportunitySnapshot,
    ) -> dict:
        market = snapshot.market
        current = snapshot.now
        market_key = snapshot.market_key
        stats = self.stats[(mode, profile.name)]
        ledger = self.ledgers[(mode, profile.name)]

        if profile.market_slug_prefix and not market.slug.startswith(f"{profile.market_slug_prefix}-"):
            stats.signal_skips_by_reason["market_prefix_mismatch"] += 1
            return {
                "status": "skipped",
                "phase": "signal",
                "reason_code": "market_prefix_mismatch",
                "reason": (
                    f"Profile {profile.name} expects {profile.market_slug_prefix} markets, "
                    f"but snapshot market is {market.slug}."
                ),
            }

        stats.markets_seen.add(market_key)
        stats.snapshots_seen += 1

        seconds_left = (market.window_end - current).total_seconds()
        entry_skip = _entry_window_skip(profile, seconds_left)
        outside_entry_window = entry_skip is not None
        if outside_entry_window:
            stats.outside_entry_window += 1
        else:
            stats.entry_window_hit_markets.add(market_key)

        hours_skip = _trading_hours_skip(profile, current)
        if hours_skip is not None:
            stats.outside_trading_hours += 1

        if mode == LIVE_LIKE_MODE and entry_skip is not None:
            reason_code = str(entry_skip.get("reason_code") or "outside_entry_window")
            stats.signal_skips_by_reason[reason_code] += 1
            return {
                "status": "skipped",
                "phase": "signal",
                "reason_code": reason_code,
                "reason": entry_skip.get("reason"),
            }
        if hours_skip is not None:
            reason_code = str(hours_skip.get("reason_code") or "outside_trading_hours")
            stats.signal_skips_by_reason[reason_code] += 1
            return {
                "status": "skipped",
                "phase": "signal",
                "reason_code": reason_code,
                "reason": hours_skip.get("reason"),
            }

        signal = evaluate_signal(
            market=market,
            price_snapshot=snapshot.price_snapshot,
            up_book=snapshot.up_book,
            down_book=snapshot.down_book,
            profile=profile,
            now=current,
            ignore_entry_window=(mode == ALL_SNAPSHOTS_MODE),
        )
        if signal.get("action") != "trade":
            reason_code = str(signal.get("reason_code") or "signal_skip")
            stats.signal_skips_by_reason[reason_code] += 1
            return {
                "status": "skipped",
                "phase": "signal",
                "reason_code": reason_code,
                "reason": signal.get("reason"),
                "signal": signal,
            }

        stats.signal_trade_markets.add(market_key)

        if _ledger_has_market_trade(ledger, market):
            stats.risk_skips_by_reason["duplicate_market"] += 1
            return {
                "status": "skipped",
                "phase": "risk",
                "reason_code": "duplicate_market",
                "reason": f"Already simulated a BTC 5m trade for market {market.slug}.",
                "signal": signal,
            }

        risk = _risk_check(
            ledger=ledger,
            market=market,
            profile=profile,
            dry_run=True,
            now=current,
        )
        if risk.get("action") != "ok":
            reason_code = str(risk.get("reason_code") or "risk_skip")
            stats.risk_skips_by_reason[reason_code] += 1
            return {
                "status": "skipped",
                "phase": "risk",
                "reason_code": reason_code,
                "reason": risk.get("reason"),
                "signal": signal,
                "risk": risk.get("risk"),
            }

        direction = str(signal["direction"])
        chosen_book = _book_for_direction(direction, up_book=snapshot.up_book, down_book=snapshot.down_book)
        entry_price, entry_price_source = _entry_price_for_book(profile, chosen_book)
        if entry_price is None:
            stats.risk_skips_by_reason["sizing_entry_price"] += 1
            return {
                "status": "skipped",
                "phase": "risk",
                "reason_code": "sizing_entry_price",
                "reason": f"No {entry_price_source.replace('_', ' ')} available for sizing.",
                "signal": signal,
                "risk": risk["risk"],
            }

        amount, size_reason = _sized_notional(
            profile=profile,
            available_cash=_available_cash(dry_run=True, clob_client=None),
            open_exposure=_safe_float(risk["risk"].get("open_exposure")),
            price=entry_price,
            min_order_size=max(market.order_min_size, chosen_book.min_order_size),
        )
        if amount <= 0:
            stats.risk_skips_by_reason["sizing"] += 1
            return {
                "status": "skipped",
                "phase": "risk",
                "reason_code": "sizing",
                "reason": size_reason,
                "signal": signal,
                "risk": risk["risk"],
            }

        order = _place_order(
            direction=direction,
            amount=amount,
            market=market,
            book=chosen_book,
            ledger=ledger,
            dry_run=True,
            clob_client=None,
            price_snapshot=snapshot.price_snapshot,
            signal=signal,
            profile=profile,
            reason=signal["reason"],
            current_time=current,
            realtime_submit_deadline=False,
        )
        if order.get("status") != "dry_run":
            reason_code = str(order.get("reason_code") or order.get("status") or "order_not_placed")
            stats.risk_skips_by_reason[reason_code] += 1
            return {
                "status": "skipped",
                "phase": "risk",
                "reason_code": reason_code,
                "reason": order.get("reason") or order.get("error") or "dry-run order was not recorded",
                "signal": signal,
                "risk": risk["risk"],
                "order": order,
            }

        stats.actual_trade_markets.add(market_key)
        if mode == ALL_SNAPSHOTS_MODE and outside_entry_window:
            stats.trade_outside_entry_window += 1
        self._create_take_profit_counterpart(
            mode=mode,
            profile=profile,
            ledger=ledger,
            order=order,
            signal=signal,
            snapshot=snapshot,
        )
        return {
            "status": "traded",
            "reason_code": "trade",
            "outside_entry_window": outside_entry_window,
            "signal": signal,
            "risk": risk["risk"],
            "order": order,
        }

    def build_report(
        self,
        *,
        run_id: str,
        target_markets: int,
        status: str,
        started_at: datetime,
        completed_at: datetime | None = None,
    ) -> dict:
        profiles: dict[str, dict[str, dict]] = {mode: {} for mode in BTC5M_OPPORTUNITY_MODES}
        for mode in BTC5M_OPPORTUNITY_MODES:
            for profile_name in self.profile_names:
                key = (mode, profile_name)
                payload = self.stats[key].to_report(self.ledgers[key])
                payload["exit_policies"] = {
                    HOLD_TO_SETTLEMENT_POLICY: _hold_policy_report(payload),
                    TAKE_PROFIT_EXIT_POLICY: _position_pnl_summary(self.take_profit_positions[key]),
                }
                profiles[mode][profile_name] = payload
        profile_groups = _build_profile_groups(self.profiles)
        return {
            "run_id": run_id,
            "status": status,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat() if completed_at else None,
            "target_markets": target_markets,
            "distinct_markets_seen": self.distinct_markets_seen,
            "cycles": self.cycles,
            "settlement_source": self.settlement_source,
            "settlement_delay_seconds": self.settlement_delay_seconds,
            "modes": list(BTC5M_OPPORTUNITY_MODES),
            "profiles": profiles,
            "ranking": {
                mode: _rank_mode_profiles(profiles[mode])
                for mode in BTC5M_OPPORTUNITY_MODES
            },
            "exit_policy_ranking": {
                mode: {
                    policy: _rank_exit_policy_profiles(profiles[mode], policy)
                    for policy in BTC5M_OPPORTUNITY_EXIT_POLICIES
                }
                for mode in BTC5M_OPPORTUNITY_MODES
            },
            "profile_groups": profile_groups,
            "group_ranking": _build_group_rankings(profiles, profile_groups),
            "take_profit_positions": {
                mode: {
                    profile_name: [
                        position.to_dict()
                        for position in self.take_profit_positions[(mode, profile_name)]
                    ]
                    for profile_name in self.profile_names
                }
                for mode in BTC5M_OPPORTUNITY_MODES
            },
            "markets": list(self.market_records.values()),
        }

    @staticmethod
    def write_reports(report: dict, *, report_path: Path, markdown_path: Path | None = None) -> None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if markdown_path is not None:
            markdown_path = Path(markdown_path)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict) -> str:
    lines = [
        f"# BTC 5m Opportunity Report - {report.get('run_id')}",
        "",
        f"- Status: {report.get('status')}",
        f"- Markets: {report.get('distinct_markets_seen')} / {report.get('target_markets')}",
        f"- Cycles: {report.get('cycles')}",
        f"- Settlement source: {report.get('settlement_source')}",
        "",
    ]
    rankings = report.get("ranking", {}) or {}
    policy_rankings = report.get("exit_policy_ranking", {}) or {}
    profiles = report.get("profiles", {}) or {}
    profile_groups = report.get("profile_groups", {}) or {}
    group_rankings = report.get("group_ranking", {}) or {}
    if profile_groups:
        lines.extend(
            [
                "## Strategy Tracks",
                "",
                "| Track | Primary Read | Profiles | Notes |",
                "| --- | --- | --- | --- |",
            ]
        )
        for group in profile_groups.values():
            lines.append(
                "| {label} | {mode} / {policy} | {profiles} | {description} |".format(
                    label=group.get("label"),
                    mode=group.get("primary_mode"),
                    policy=group.get("primary_exit_policy"),
                    profiles=", ".join(group.get("profiles") or []),
                    description=group.get("description"),
                )
            )
        lines.append("")
    for mode in BTC5M_OPPORTUNITY_MODES:
        lines.extend([f"## {mode}", ""])
        if mode == ALL_SNAPSHOTS_MODE:
            lines.extend(
                [
                    "_Reference only: all_snapshots ignores entry timing and is not directly deployable._",
                    "",
                ]
            )
        lines.append("### Overall hold-to-settlement ranking")
        lines.extend(_ranking_table(rankings.get(mode, []) or []))
        lines.append("")
        lines.append("### Overall take-profit ranking")
        lines.extend(_ranking_table((policy_rankings.get(mode, {}) or {}).get(TAKE_PROFIT_EXIT_POLICY, []) or []))
        for group_key, group in profile_groups.items():
            group_payload = ((group_rankings.get(mode) or {}).get(group_key) or {})
            primary_policy = str(group_payload.get("primary_exit_policy") or group.get("primary_exit_policy"))
            primary_label = "hold-to-settlement" if primary_policy == HOLD_TO_SETTLEMENT_POLICY else "take-profit"
            if mode == group.get("primary_mode"):
                heading = f"### {group.get('label')} primary {primary_label} ranking"
            else:
                heading = f"### {group.get('label')} {primary_label} reference ranking"
            lines.extend(["", heading])
            lines.extend(_ranking_table(group_payload.get("ranking") or []))
        lines.extend(["", "| Profile | Seen | Entry Hits | Outside Entry | Signal Trades | Actual Trades | Top Signal Skips | Top Risk Skips |"])
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for profile_name, payload in (profiles.get(mode, {}) or {}).items():
            lines.append(
                "| {profile} | {seen} | {hits} | {outside} | {signals} | {actual} | {signal_skips} | {risk_skips} |".format(
                    profile=profile_name,
                    seen=int(payload.get("markets_seen", 0) or 0),
                    hits=int(payload.get("entry_window_hit_markets", 0) or 0),
                    outside=int(payload.get("outside_entry_window", 0) or 0),
                    signals=int(payload.get("signal_trade_markets", 0) or 0),
                    actual=int(payload.get("actual_trade_markets", 0) or 0),
                    signal_skips=_format_counter(payload.get("signal_skips_by_reason", {})),
                    risk_skips=_format_counter(payload.get("risk_skips_by_reason", {})),
                )
            )
        lines.extend(["", "| Profile | Exit Policy | Closed | Early Exits | P&L | ROI | Win Rate | Delta vs Hold | Top Exit Reasons |"])
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for profile_name, payload in (profiles.get(mode, {}) or {}).items():
            for policy_name in BTC5M_OPPORTUNITY_EXIT_POLICIES:
                policy = (payload.get("exit_policies") or {}).get(policy_name, {})
                lines.append(
                    "| {profile} | {policy} | {closed} | {early} | {pnl:+.2f} | {roi:.2%} | {win:.1%} | {delta:+.2f} | {reasons} |".format(
                        profile=profile_name,
                        policy=policy_name,
                        closed=int(policy.get("closed_trades", 0) or 0),
                        early=int(policy.get("early_exits", 0) or 0),
                        pnl=float(policy.get("realized_pnl", 0.0) or 0.0),
                        roi=float(policy.get("roi", 0.0) or 0.0),
                        win=float(policy.get("win_rate", 0.0) or 0.0),
                        delta=float(policy.get("delta_pnl_vs_hold", 0.0) or 0.0),
                        reasons=_format_counter(policy.get("exit_reasons_by_reason", {})),
                    )
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _ranking_table(rows: list[dict]) -> list[str]:
    lines = ["| Rank | Profile | Trades | Settled | P&L | ROI | Win Rate | Delta vs Hold |"]
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {profile} | {trades} | {settled} | {pnl:+.2f} | {roi:.2%} | {win:.1%} | {delta:+.2f} |".format(
                rank=idx,
                profile=row.get("profile"),
                trades=int(row.get("actual_trade_markets", 0) or 0),
                settled=int(row.get("settled_trades", 0) or 0),
                pnl=float(row.get("realized_pnl", 0.0) or 0.0),
                roi=float(row.get("roi", 0.0) or 0.0),
                win=float(row.get("win_rate", 0.0) or 0.0),
                delta=float(row.get("delta_pnl_vs_hold", 0.0) or 0.0),
            )
        )
    return lines


def _format_counter(values: dict) -> str:
    if not values:
        return "-"
    items = sorted(values.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{key}: {value}" for key, value in items[:3])


def _load_resume_report(report_path: Path, *, run_id: str, target_markets: int) -> dict | None:
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("BTC 5m opportunity resume report load failed: %s", exc)
        return None
    if not isinstance(report, dict):
        return None
    prior_run_id = str(report.get("run_id") or "")
    if prior_run_id and prior_run_id != run_id:
        return None
    prior_seen = _safe_int(report.get("distinct_markets_seen"))
    if prior_seen <= 0 or prior_seen >= target_markets:
        return None
    return report


def run_btc5m_opportunity(
    *,
    profile_names: str | list[str] | tuple[str, ...] = "all",
    target_markets: int = BTC5M_OPPORTUNITY_TARGET_MARKETS,
    poll_seconds: float = BTC5M_POLL_SECONDS,
    settlement_source: str | None = BTC5M_PAPER_SETTLEMENT_SOURCE,
    ledger_dir: Path | None = None,
    report_path: Path | None = None,
    markdown_path: Path | None = None,
    market_slug: str | None = None,
    max_cycles: int | None = None,
    market_client: Btc5mMarketClient | None = None,
    price_client=None,
    book_client: PublicClobOrderBookClient | None = None,
    settlement_client=None,
    settlement_delay_seconds: float = BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS,
    run_id: str | None = None,
    final_settlement_timeout_seconds: float | None = None,
    ignore_risk_limits: bool = False,
    now_fn: Callable[[], datetime] = utcnow,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    run_id, default_ledger_dir, default_report_path, default_markdown_path = build_btc5m_opportunity_paths(run_id)
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else default_ledger_dir
    report_path = Path(report_path) if report_path is not None else default_report_path
    markdown_path = (
        Path(markdown_path)
        if markdown_path is not None
        else (report_path.with_suffix(".md") if report_path is not None else default_markdown_path)
    )
    target_markets = max(int(target_markets), 1)
    started_at = _coerce_utc(now_fn())
    harness = Btc5mOpportunityHarness(
        profile_names=profile_names,
        ledger_dir=ledger_dir,
        settlement_source=settlement_source,
        settlement_delay_seconds=settlement_delay_seconds,
        market_client=market_client,
        price_client=price_client,
        book_client=book_client,
        settlement_client=settlement_client,
        ignore_risk_limits=ignore_risk_limits,
    )
    resume_report = _load_resume_report(report_path, run_id=run_id, target_markets=target_markets)
    if resume_report is not None:
        harness.resume_from_report(resume_report)
        started_at = _report_datetime(resume_report.get("started_at"), started_at) or started_at
        logger.info(
            "BTC 5m opportunity resumed from %s: seen=%s/%s cycles=%s",
            report_path,
            harness.distinct_markets_seen,
            target_markets,
            harness.cycles,
        )

    status = "running"
    while harness.distinct_markets_seen < target_markets:
        if max_cycles is not None and harness.cycles >= max_cycles:
            status = "stopped"
            break
        current = _coerce_utc(now_fn())
        result = harness.run_once(now=current, market_slug=market_slug)
        logger.info(
            "BTC 5m opportunity cycle %s: status=%s market=%s seen=%s/%s",
            harness.cycles,
            result.get("status"),
            result.get("market_slug"),
            harness.distinct_markets_seen,
            target_markets,
        )
        if harness.distinct_markets_seen >= target_markets:
            status = "complete"
            break
        report = harness.build_report(
            run_id=run_id,
            target_markets=target_markets,
            status=status,
            started_at=started_at,
        )
        harness.write_reports(report, report_path=report_path, markdown_path=markdown_path)
        sleep_seconds = max(float(poll_seconds), 0.0)
        if sleep_seconds:
            sleep_fn(sleep_seconds)

    if status == "complete" and (
        _has_open_btc5m_bets(harness.ledgers) or harness.has_open_take_profit_positions()
    ):
        timeout_seconds = (
            float(final_settlement_timeout_seconds)
            if final_settlement_timeout_seconds is not None
            else BTC5M_WINDOW_SECONDS + max(float(settlement_delay_seconds), 0.0) + max(float(poll_seconds), 0.0)
        )
        deadline = time.monotonic() + max(timeout_seconds, 0.0)
        while _has_open_btc5m_bets(harness.ledgers) or harness.has_open_take_profit_positions():
            current = _coerce_utc(now_fn())
            harness.run_exit_tracking_once(now=current, market_slug=market_slug)
            if (
                not _has_open_btc5m_bets(harness.ledgers)
                and not harness.has_open_take_profit_positions()
            ) or time.monotonic() >= deadline:
                break
            sleep_seconds = min(max(float(poll_seconds), 1.0), max(deadline - time.monotonic(), 0.0))
            if sleep_seconds <= 0:
                break
            sleep_fn(sleep_seconds)

    completed_at = _coerce_utc(now_fn())
    report = harness.build_report(
        run_id=run_id,
        target_markets=target_markets,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
    )
    harness.write_reports(report, report_path=report_path, markdown_path=markdown_path)
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report["ledger_dir"] = str(ledger_dir)
    return report
