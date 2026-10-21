"""Callable final-publication consumer for the existing UFC portfolio.

The refresh producer remains the sole publication owner. This finite reader
uses the accepted models and original UFC accounting only after the real-money
startup controls pass; off/dry-run evaluation remains isolated by its caller.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading

from src.config import (INJURY_BLOCK_BETS, LOGS_DIR, MIN_EDGE_THRESHOLD,
                        REFRESH_TRADING_OBSERVATION_MAX_AGE_SECONDS)

logger = logging.getLogger(__name__)
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


class TradingConsumerBusy(RuntimeError):
    pass


class OriginalAccountingUnresolved(RuntimeError):
    """Original order evidence blocks execution, not current observations."""

    def __init__(self, count):
        self.count = count
        super().__init__("Original real submission ownership/fill remains unresolved; new orders blocked")


@contextmanager
def trading_consumer_lock(path):
    """One finite market/prediction/execution reader per retained state root."""
    path = Path(path).resolve()
    with _locks_guard:
        process_lock = _locks.setdefault(str(path), threading.Lock())
    if not process_lock.acquire(blocking=False):
        raise TradingConsumerBusy("Trading consumer already owns this state root")
    stream = None
    acquired = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TradingConsumerBusy("Trading consumer already owns this state root") from exc
        acquired = True
        yield
    finally:
        if stream is not None:
            if acquired:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        process_lock.release()


def _pair(row):
    from src.data.name_utils import normalize_person_name
    if any(not isinstance(row.get(key), str) or not row[key].strip()
           for key in ("fighter_a", "fighter_b")):
        raise ValueError("Current matchup requires two observed fighter names")
    return tuple(sorted(normalize_person_name(str(row.get(key, "")))
                        for key in ("fighter_a", "fighter_b")))


def _event_date(value):
    from zoneinfo import ZoneInfo
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("A current bout-specific aware commence time is required")
    return stamp.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _bound_markets(markets, predictions):
    """Remove fuzzy, ambiguous, off-card or malformed outcome/token mappings."""
    import pandas as pd
    allowed = {_pair(row): row["official_event_date"] for row in predictions}
    retained, seen_pairs, seen_tokens = [], set(), set()
    for row in markets.to_dict("records"):
        pair = _pair(row)
        if pair not in allowed or _event_date(row.get("event_date")) != allowed[pair]:
            continue
        tokens = [row.get("token_id_yes"), row.get("token_id_no")]
        if (any(not isinstance(token, str) or not token.strip() or token != token.strip() for token in tokens)
                or tokens[0] == tokens[1] or any(token in seen_tokens for token in tokens)
                or pair in seen_pairs):
            raise ValueError("Ambiguous or missing market outcome/token mapping")
        if row.get("closed") is True or row.get("active") is False or row.get("accepting_orders") is False:
            continue
        seen_pairs.add(pair)
        seen_tokens.update(tokens)
        retained.append(row)
    return pd.DataFrame(retained)


def run_refresh_live_cycle(*, root, baseline, dry_run=True,
                           min_edge=MIN_EDGE_THRESHOLD, progress_callback=None,
                           readonly_clob=None, bankroll_basis=None, observation_dir=None,
                           model_name=None, clob_client=None, host="0.0.0.0",
                           startup_source="refresh-consumer"):
    """Evaluate a final published pair through the existing S/C/M executor."""
    from src.live_control import assert_real_trading_allowed, resolve_live_mode_from_env
    # Check before imports which create clients/state or access external APIs.
    mode = resolve_live_mode_from_env()
    if dry_run is not True:
        if dry_run is not False or mode != "real":
            raise RuntimeError("Real refresh execution requires LIVE_TRADING_MODE=real")
        assert_real_trading_allowed(model_name=model_name, host=host, startup_source=startup_source)
        from src.trading_controls import require_unpaused
        require_unpaused()
    elif mode not in {"off", "dry-run"}:
        raise RuntimeError("Orders-disabled refresh evaluation requires off or dry-run mode")
    try:
        with trading_consumer_lock(Path(LOGS_DIR) / "refresh-trading.lock"):
            return _run_cycle(root=root, baseline=baseline, min_edge=min_edge,
                              progress_callback=progress_callback,
                              readonly_clob=readonly_clob, bankroll_basis=bankroll_basis,
                              observation_dir=observation_dir, dry_run=dry_run,
                              model_name=model_name, clob_client=clob_client)
    except TradingConsumerBusy as exc:
        return {"status": "busy", "reason": str(exc), "trading_enabled": False,
                "total_orders": 0}


def _run_cycle(*, root, baseline, min_edge, progress_callback,
               readonly_clob=None, bankroll_basis=None, observation_dir=None,
               dry_run=True, model_name=None, clob_client=None):
    import pandas as pd
    from src.data.refresh_service import RefreshService
    from src.data.refresh_trading import PublishedTradingSnapshot
    from src.data.odds_client import OddsClient
    from src.polymarket.markets import get_ufc_fight_markets
    from src.strategy.duo_trader import run_duo_traders
    from src.bot import _live_prediction_quality_assessment, _withhold_unconfirmed_identity
    from src.data.line_tracker import detect_injury_or_cancellation
    from src.betting_window import bet_window_status

    consumer = PublishedTradingSnapshot(RefreshService(root=root, baseline=baseline)).read()
    original_accounting = None
    blocking_conditions = []
    clob = readonly_clob
    if not dry_run:
        _assert_selected_live_model(consumer, model_name)
        from src.polymarket.client import ClobClientWrapper
        clob = clob_client if clob_client is not None else ClobClientWrapper()
        try:
            original_accounting = _maintain_original_accounting(clob)
        except OriginalAccountingUnresolved as exc:
            # Missing exchange evidence never cancels an intent or permits a
            # replacement. Still collect this cycle's real market observations.
            blocking_conditions.append("original_order_unresolved")
            original_accounting = {"status": "blocked",
                                   "unresolved_original_submissions": exc.count}
    freshness = None
    if observation_dir is not None:
        from src.data.refresh_live_observations import observe_publication
        freshness = observe_publication(consumer, observation_dir)
    if callable(progress_callback):
        progress_callback("Cycle active: reading final refresh publication and current markets")
    odds = OddsClient()
    consensus = odds.get_consensus_odds(odds.odds_to_dataframe(odds.get_live_odds()))
    observed_at = datetime.now(timezone.utc)
    official = {}
    for row in consumer.rows:
        if _pair(row) in official:
            raise ValueError("Published card has an ambiguous fighter matchup")
        official[_pair(row)] = row
    # Market membership comes from the entire accepted card, independently of
    # bookmaker coverage. Saved rows supply a date, never a current bout clock.
    card_contexts = [{"fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"],
                      "event_date": row["event_date"], "official_event_date": row["event_date"]}
                     for row in consumer.rows]
    current, withheld, current_pairs = [], [], set()
    for row in consensus.to_dict("records"):
        card = official.get(_pair(row))
        if card is None or _event_date(row.get("commence_time")) != card["event_date"]:
            withheld.append({"fighter_a": row.get("fighter_a"), "fighter_b": row.get("fighter_b"),
                             "reason": "not_bound_to_published_card"})
            continue
        if _pair(row) in current_pairs:
            raise ValueError("Current bookmaker feed has an ambiguous duplicate matchup")
        current_pairs.add(_pair(row))
        current.append(row)
    # Empty/out-of-card feeds cannot be turned into invented bookmaker inputs.
    markets = get_ufc_fight_markets(bout_contexts=card_contexts + current, require_fresh=True)
    markets_observed_at = datetime.now(timezone.utc).isoformat()
    predictions = consumer.score(current, observed_at=observed_at,
                                 decision_at=datetime.now(timezone.utc),
                                 max_observation_age_seconds=REFRESH_TRADING_OBSERVATION_MAX_AGE_SECONDS)
    markets = _bound_markets(markets, card_contexts)
    # The legacy executor also has a fuzzy matcher. Keep its input restricted
    # to exact scored pairs; full-card display coverage must not widen trading.
    execution_markets = _bound_markets(markets, predictions)
    for row in predictions:
        quality = _live_prediction_quality_assessment(
            row["model_features"], row["feature_provenance"],
            fighter_a=row["fighter_a"], fighter_b=row["fighter_b"],
            a_fights=row["a_num_fights"], b_fights=row["b_num_fights"])
        reasons = list(quality["reasons"]) + blocking_conditions
        if freshness:
            reasons.extend(freshness["trade_block_reasons"])
        if row.get("market_inputs_available") is not True:
            reasons.append("missing_current_bookmaker_inputs")
        if _withhold_unconfirmed_identity(row):
            reasons.append("fighter_identity_unconfirmed")
        if INJURY_BLOCK_BETS:
            injury = detect_injury_or_cancellation(
                row["fighter_a"], row["fighter_b"],
                current_odds={"a_prob": row.get("a_fair_prob_avg"),
                              "b_prob": row.get("b_fair_prob_avg")},
                event_id=row.get("event_id"), commence_time=row.get("commence_time"))
            if injury.get("severity") == "block":
                reasons.append(str(injury.get("reason") or "injury_or_cancellation"))
        row.update(data_quality=quality, trade_blocked=bool(reasons),
                   trade_block_reason="; ".join(reasons),
                   market_observed_at=markets_observed_at,
                   bet_window=bet_window_status(row.get("commence_time"), fail_closed=True))
    executable = pd.DataFrame([row for row in predictions if not row["trade_blocked"]])
    result = {"status": "idle", "reason": "no_executable_opportunities", "total_orders": 0}
    if not blocking_conditions and not executable.empty and not execution_markets.empty:
        # The original S -> C -> M runner retains its existing timing, stake,
        # risk, ledger, no-odds and exposure rules.
        result = run_duo_traders(executable, execution_markets, clob=clob, dry_run=dry_run,
                                min_edge=min_edge, event_title=consumer.report.get("event_date", ""),
                                progress_callback=progress_callback, bankroll_basis=bankroll_basis)
    result.setdefault("status", "evaluated")
    if blocking_conditions:
        result.update(status="blocked", reason="original_order_unresolved")
    real_orders = [order for name in ("trader_s", "trader_c", "trader_m")
                   for group in ("orders", "near_miss_orders")
                   for order in result.get(name, {}).get(group, [])
                   if order.get("dry_run") is False]
    result.update(trading_enabled=not dry_run and not blocking_conditions, orders_disabled=dry_run,
                  blocking_conditions=blocking_conditions,
                  # A failed/unknown result is not proof that POST never reached
                  # the exchange. Keep that distinction in the saved cycle.
                  real_submissions=0 if not real_orders else None,
                  confirmed_placed_orders=sum(order.get("status") == "placed" for order in real_orders),
                  original_accounting_maintenance=original_accounting,
                  accepted_generation=consumer.publication["generation_id"],
                  publication_job_id=consumer.publication["job_id"],
                  bookmaker_observed_at=observed_at.isoformat(),
                  market_observed_at=markets_observed_at,
                  prediction_count=len(predictions), executable_prediction_count=len(executable),
                  matched_market_count=len(markets), withheld=withheld,
                  current_card={"published_bout_count": len(consumer.rows),
                      "bookmaker_bout_count": len(current),
                      "missing_bookmaker_bout_count": len(official) - len(current),
                      "bet_window_open_count": sum(row["bet_window"].get("open") is True for row in predictions)},
                  predictions=predictions, freshness=freshness,
                  snapshot_binding=consumer.binding,
                  mapped_markets=[{key: row.get(key) for key in
                      ("fighter_a", "fighter_b", "event_date", "market_id",
                       "token_id_yes", "token_id_no", "price_yes", "price_no")}
                      for row in markets.to_dict("records")])
    logger.info("Final-publication portfolio evaluation: generation=%s predictions=%s markets=%s mode=%s",
                result["accepted_generation"], len(predictions), len(markets),
                "dry-run" if dry_run else "real")
    return result


def _assert_selected_live_model(consumer, model_name):
    """The LIVE_MODEL control must select the final publication's primary."""
    from src.live_control import resolve_live_model_name, _resolve_model_artifact_path
    path = _resolve_model_artifact_path(resolve_live_model_name(model_name)).resolve()
    expected = consumer.snapshot["models"]["primary"]
    if (not path.is_file() or path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
        raise RuntimeError("LIVE_MODEL does not match the final accepted primary model")


def original_accounting_ledgers():
    """Keep every original ledger, including retired/shared-owner history."""
    from src.polymarket.tracker import BetLedger
    from src.strategy.duo_trader import get_all_trader_ledgers
    paths = {Path(path).resolve() for _, path in get_all_trader_ledgers() if path is not None}
    paths.update(path.resolve() for path in Path(LOGS_DIR).glob("bet_ledger*.json"))
    ledgers = []
    for path in sorted(paths, key=str):
        if not path.exists():
            continue
        document = json.loads(path.read_bytes())
        rows = document.get("bets")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError("Original accounting ledger has an invalid shape")
        ids = [row.get("id") for row in rows]
        if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
            raise RuntimeError("Original accounting ledger has ambiguous original IDs")
        if any(row.get("status") == "open" and type(row.get("dry_run")) is not bool for row in rows):
            raise RuntimeError("Original active accounting row has an unclassified real/paper mode")
        ledgers.append(BetLedger(path))
    return ledgers


def _maintain_original_accounting(clob):
    """Recover saved order identities and settle confirmed original positions.

    No token/economic match assigns a manual holding to a trader. Uncertainty
    in any original real submission prevents new orders until exact exchange
    evidence resolves it; there is no synthetic cash refund during recovery.
    """
    from src.polymarket.executor import OrderExecutor
    from src.polymarket.tracker import auto_settle_from_polymarket, _has_actual_fill
    from src.strategy.bankroll import BankrollManager
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER, MODEL_TRACKER_LEDGER
    from src.trading_controls import require_unpaused
    require_unpaused()
    ledgers = original_accounting_ledgers()
    enabled = {Path(path).resolve() for path in (SINGLE_LEDGER, CONVICTION_LEDGER, MODEL_TRACKER_LEDGER)}
    executor = OrderExecutor(BankrollManager(initial_bankroll=0, auto_detect_balance=False),
                             clob_client=clob, dry_run=False)
    open_orders = None
    unresolved, reconciled, settled = [], 0, 0
    for ledger in ledgers:
        executor.ledger = ledger
        for row in ledger.get_open_bets(fresh=True):
            if row.get("dry_run") is not False:
                continue
            if row.get("placement_state") not in {"pending_submit", "unknown", "submitted", "resting", "pending"}:
                continue
            if ledger.path.resolve() not in enabled:
                raise RuntimeError("Dormant original owner has an uncertain intent; separate evidence required")
            if open_orders is None:
                open_orders = clob.get_open_orders()
                if not isinstance(open_orders, list):
                    raise RuntimeError("Complete original-order recovery read is unavailable")
            original = {**row, "_ledger_path": str(ledger.path)}
            if row.get("placement_state") in {"pending_submit", "unknown"}:
                recovered = executor._reconcile_unresolved_submission(original, open_orders=open_orders)
            else:
                if not row.get("order_id"):
                    raise RuntimeError("Original resting order has no durable exchange ID; new orders blocked")
                state = executor._inspect_limit_order_state(original, open_orders)
                if state["state"] == "closed":
                    outcome = executor._reconcile_closed_limit_order(original, reason=state["reason"],
                        order_data=state["order"], ledger=ledger, release_reserved_cash=False)
                    if outcome == "unchanged":
                        unresolved.append({"ledger": ledger.path.name, "id": row["id"],
                                           "order_id": row.get("order_id")})
                        continue
                    recovered = executor._reload_ledger_entry(original)
                elif state["state"] == "resting":
                    recovered = original
                else:
                    unresolved.append({"ledger": ledger.path.name, "id": row["id"],
                                       "order_id": row.get("order_id")})
                    continue
            if recovered.get("status") == "open" and recovered.get("placement_state") in {"pending_submit", "unknown"}:
                unresolved.append({"ledger": ledger.path.name, "id": row["id"],
                                   "order_id": row.get("order_id")})
            else:
                reconciled += 1
    if unresolved:
        raise OriginalAccountingUnresolved(len(unresolved))
    # Original gross evidence remains unchanged. Cash cost is established only
    # by exact exchange-trade and activity linkage before financial finalization.
    from src.polymarket.fill_cashflow import (
        CashDebitUnavailable, fill_binding, observed_cash_debit, verified_cash_debit,
    )
    cash_confirmed = cash_unknown = 0
    for ledger in ledgers:
        if ledger.path.resolve() not in enabled:
            continue
        executor.ledger = ledger
        for row in ledger.get_open_bets(fresh=True):
            if row.get("dry_run") is not False or not _has_actual_fill(row):
                continue
            try:
                verified_cash_debit(row)
                cash_confirmed += 1
                continue
            except CashDebitUnavailable:
                pass
            binding = fill_binding(row)
            require_unpaused()
            marked = ledger.record_cash_debit(row["id"], expected_binding=binding,
                fields={"cash_debit_status": "unknown",
                        "cash_debit_unknown_reason": "observed_activity_not_yet_confirmed"})
            if not marked.ok:
                cash_unknown += 1
                continue
            try:
                placed = datetime.fromisoformat(str(row.get("placed_at") or "").replace("Z", "+00:00"))
                if placed.tzinfo is None:
                    raise CashDebitUnavailable("original_placement_time_unavailable")
                found, order = executor._lookup_order_from_trade_history(row, str(row.get("order_id") or ""))
                if not found or not order:
                    raise CashDebitUnavailable("original_trade_set_unavailable")
                activity = clob.get_trade_activity(start=max(0, int(placed.timestamp()) - 300),
                                                   end=int(datetime.now(timezone.utc).timestamp()))
                fields = observed_cash_debit(row, order, activity)
            except Exception as exc:
                fields = {"cash_debit_status": "unknown", "cash_debit_unknown_reason":
                          str(exc) if isinstance(exc, CashDebitUnavailable)
                          else "cash_debit_read_unavailable"}
            require_unpaused()
            saved = ledger.record_cash_debit(row["id"], expected_binding=binding, fields=fields)
            try:
                if not saved.ok:
                    raise CashDebitUnavailable("cash_debit_enrichment_not_saved")
                verified_cash_debit(saved.bet)
                cash_confirmed += 1
            except CashDebitUnavailable:
                cash_unknown += 1
    class RealPositions:
        def __init__(self, ledger): self.ledger = ledger
        def get_open_bets(self, *, fresh=False):
            return [row for row in self.ledger.get_open_bets(fresh=fresh)
                    if row.get("dry_run") is False and _has_actual_fill(row)]
        def settle_bet(self, *args, **kwargs):
            require_unpaused()
            return self.ledger.settle_bet(*args, **kwargs)
    for ledger in ledgers:
        if ledger.path.resolve() in enabled:
            settled += auto_settle_from_polymarket(RealPositions(ledger), clob_client=clob)
    return {"original_ledger_count": len(ledgers), "recovered_original_submissions": reconciled,
            "settled_original_real_positions": settled, "unresolved_original_submissions": 0,
            "manual_positions_adopted": 0,
            "original_open_cash_debits_confirmed": cash_confirmed,
            "original_open_cash_debits_unknown": cash_unknown}
