"""
Order executor — places and manages bets on Polymarket based on model signals.
"""

import logging
import math
import time
from typing import Optional

import pandas as pd

from src.polymarket.client import ClobClientWrapper
from src.polymarket.markets import get_ufc_fight_markets
from src.strategy.value import find_value_bets, implied_prob_to_decimal_odds
from src.strategy.bankroll import BankrollManager
from src.config import (
    MIN_EDGE_THRESHOLD,
    MIN_BOOK_LIQUIDITY,
    MAX_SLIPPAGE,
    MAX_BET_VS_BOOK_RATIO,
    LIMIT_BID_TTL_HOURS,
    LIMIT_BID_PRE_EVENT_HOURS,
)
from src.polymarket.tracker import BetLedger

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Executes orders on Polymarket based on model predictions."""

    def __init__(
        self,
        bankroll: BankrollManager,
        clob_client: Optional[ClobClientWrapper] = None,
        dry_run: bool = True,
    ):
        """
        Args:
            bankroll: BankrollManager instance
            clob_client: Authenticated CLOB client (None = dry run)
            dry_run: If True, log orders but don't actually place them
        """
        self.bankroll = bankroll
        self.clob = clob_client or ClobClientWrapper()
        self.dry_run = dry_run
        self.order_log: list[dict] = []
        self.ledger = BetLedger()

    def execute_value_bets(
        self,
        predictions: pd.DataFrame,
        markets: pd.DataFrame,
        min_edge: float = MIN_EDGE_THRESHOLD,
    ) -> list[dict]:
        """
        Match model predictions to Polymarket markets and place orders.

        Args:
            predictions: DataFrame with prob_a, prob_b for each fight
            markets: DataFrame from get_ufc_fight_markets()
            min_edge: minimum edge to place a bet

        Returns list of order results.
        """
        if markets.empty:
            logger.warning("No markets available")
            return []

        # Match predictions to markets by fighter names
        matched = self._match_predictions_to_markets(predictions, markets)
        if matched.empty:
            logger.warning("No predictions matched to active markets")
            return []

        # Find value bets
        value_bets = find_value_bets(matched, min_edge=min_edge)
        if value_bets.empty:
            logger.info("No value bets identified")
            return []

        orders = []
        for _, bet in value_bets.iterrows():
            order = self._place_bet(bet, markets)
            if order:
                orders.append(order)
            time.sleep(1)  # Rate limiting

        return orders

    def _match_predictions_to_markets(
        self,
        predictions: pd.DataFrame,
        markets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match model predictions to Polymarket markets by fuzzy fighter name matching."""
        matched_rows = []

        for _, pred in predictions.iterrows():
            pred_a = str(pred.get("fighter_a", "")).lower().strip()
            pred_b = str(pred.get("fighter_b", "")).lower().strip()

            for _, market in markets.iterrows():
                mkt_a = str(market.get("fighter_a", "")).lower().strip()
                mkt_b = str(market.get("fighter_b", "")).lower().strip()

                # Check if fighters match (in either order)
                match_direct = (
                    _name_match(pred_a, mkt_a) and _name_match(pred_b, mkt_b)
                )
                match_reverse = (
                    _name_match(pred_a, mkt_b) and _name_match(pred_b, mkt_a)
                )

                if match_direct:
                    row = pred.to_dict()
                    # Market YES token = fighter_a wins
                    row["a_market_prob"] = market.get("price_yes") or 0.5
                    row["b_market_prob"] = market.get("price_no") or 0.5
                    row["token_id_yes"] = market.get("token_id_yes", "")
                    row["token_id_no"] = market.get("token_id_no", "")
                    row["market_id"] = market.get("market_id", "")
                    row["tick_size"] = market.get("tick_size", "0.01")
                    row["neg_risk"] = market.get("neg_risk", False)
                    row["volume"] = market.get("volume", 0)
                    matched_rows.append(row)
                    break

                elif match_reverse:
                    row = pred.to_dict()
                    # Swap: market YES = pred fighter_b
                    row["a_market_prob"] = market.get("price_no") or 0.5
                    row["b_market_prob"] = market.get("price_yes") or 0.5
                    row["token_id_yes"] = market.get("token_id_no", "")
                    row["token_id_no"] = market.get("token_id_yes", "")
                    row["market_id"] = market.get("market_id", "")
                    row["tick_size"] = market.get("tick_size", "0.01")
                    row["neg_risk"] = market.get("neg_risk", False)
                    row["volume"] = market.get("volume", 0)
                    matched_rows.append(row)
                    break

        result = pd.DataFrame(matched_rows)
        logger.info(f"Matched {len(result)} predictions to markets")
        return result

    def _check_liquidity(
        self,
        token_id: str,
        price: float,
        desired_size_usd: float,
        fighter: str,
    ) -> dict:
        """
        Check orderbook liquidity before placing an order.

        Returns dict with:
            - ok: whether the order should proceed
            - adjusted_size: recommended bet size (may be reduced)
            - available_liquidity: total USD available at or near price
            - slippage: estimated price impact
            - reason: why the order was blocked (if ok=False)
        """
        result = {
            "ok": True,
            "adjusted_size": desired_size_usd,
            "available_liquidity": 0.0,
            "slippage": 0.0,
            "best_ask": None,
            "reason": "",
        }

        try:
            book = self.clob.get_orderbook(token_id)
        except Exception as e:
            logger.warning(f"Could not fetch orderbook for {fighter}: {e}")
            result["ok"] = False
            result["reason"] = f"orderbook fetch failed: {e}"
            return result

        # We're buying, so we look at ask side (sellers)
        asks = book.get("asks", [])
        if not asks:
            result["ok"] = False
            result["reason"] = "no asks in orderbook"
            return result

        # Walk the ask side to calculate available liquidity and slippage
        total_shares = 0.0
        total_cost = 0.0
        best_ask = float(asks[0]["price"])
        result["best_ask"] = best_ask

        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            level_cost = level_price * level_size

            total_shares += level_size
            total_cost += level_cost

            # Stop if we've found enough to fill our order
            if total_cost >= desired_size_usd * 1.5:
                break

        result["available_liquidity"] = total_cost

        # Check 1: Minimum liquidity
        if total_cost < MIN_BOOK_LIQUIDITY:
            result["ok"] = False
            result["reason"] = f"insufficient liquidity (${total_cost:.0f} < ${MIN_BOOK_LIQUIDITY:.0f} min)"
            return result

        # Check 2: Don't take too much of the book
        max_size_from_book = total_cost * MAX_BET_VS_BOOK_RATIO
        if desired_size_usd > max_size_from_book:
            result["adjusted_size"] = max_size_from_book
            logger.info(
                f"  Reducing bet on {fighter}: ${desired_size_usd:.2f} -> "
                f"${max_size_from_book:.2f} (25% of ${total_cost:.0f} book)"
            )

        # Check 3: Estimate slippage (walk the book for our order size)
        filled_cost = 0.0
        filled_shares = 0.0
        worst_price = best_ask
        order_size = result["adjusted_size"]

        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            remaining = order_size - filled_cost

            if remaining <= 0:
                break

            take_cost = min(level_price * level_size, remaining)
            take_shares = take_cost / level_price
            filled_cost += take_cost
            filled_shares += take_shares
            worst_price = level_price

        if filled_shares > 0:
            avg_fill_price = filled_cost / filled_shares
            slippage = (avg_fill_price - best_ask) / best_ask if best_ask > 0 else 0
            result["slippage"] = slippage

            if slippage > MAX_SLIPPAGE:
                result["ok"] = False
                result["reason"] = (
                    f"slippage too high ({slippage:.1%} > {MAX_SLIPPAGE:.0%}) "
                    f"for ${order_size:.2f} order"
                )
                return result

        return result

    def _place_bet(self, bet: pd.Series, markets: pd.DataFrame) -> Optional[dict]:
        """Place a single bet on Polymarket."""
        fighter = bet["bet_on"]
        model_prob = bet["model_prob"]
        blended_prob = bet.get("blended_prob", model_prob)
        market_prob = bet["market_prob"]
        edge = bet["edge"]
        odds = bet["decimal_odds"]

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
        else:
            token_id = bet.get("token_id_no", "")

        if not token_id:
            logger.warning(f"No token ID for {fighter}")
            return None

        # Calculate preliminary bet size (using snapshot odds — may be recalculated below)
        override = bet.get("override_bet_size")
        if override is not None and override > 0:
            bet_size = override
        else:
            bet_size = self.bankroll.kelly_bet_size(blended_prob, odds)
        if bet_size <= 0:
            return None

        # Check orderbook liquidity before placing
        use_limit_bid = False
        if not self.dry_run:
            liq = self._check_liquidity(token_id, market_prob, bet_size, fighter)
            if not liq["ok"]:
                logger.warning(f"Skipping {fighter}: {liq['reason']}")
                return None

            # Re-verify edge against the LIVE Polymarket ask price.
            # The edge was originally calculated against a snapshot price
            # that may be stale. The actual execution price is what matters.
            live_ask = liq.get("best_ask")
            if live_ask is None or live_ask <= 0:
                logger.warning(
                    f"Skipping {fighter}: could not get live ask price from orderbook"
                )
                return None

            live_edge = blended_prob - live_ask
            use_limit_bid = live_edge < MIN_EDGE_THRESHOLD

            if use_limit_bid:
                # Don't place duplicate limit bids for the same fighter
                existing = [
                    b for b in self.ledger.open_bets
                    if b.get("fighter") == fighter
                    and b.get("order_type") in ("limit_bid", "limit")
                    and not b.get("dry_run")
                ]
                if existing:
                    logger.info(
                        f"  Skipping {fighter}: already have open limit bid "
                        f"(#{existing[0]['id']} @ ${existing[0]['price']:.4f})"
                    )
                    return None

                # Ask is too expensive for a market buy — place a resting
                # limit bid at a price that guarantees our minimum edge.
                tick = float(bet.get("tick_size", "0.01"))
                bid_price = math.floor((blended_prob - MIN_EDGE_THRESHOLD) / tick) * tick
                bid_price = round(bid_price, 4)

                if bid_price <= 0 or bid_price >= live_ask:
                    logger.info(
                        f"  Skipping {fighter}: no viable bid price "
                        f"(blended {blended_prob:.1%}, ask ${live_ask:.4f})"
                    )
                    return None

                price = bid_price
                edge = blended_prob - bid_price
                odds = implied_prob_to_decimal_odds(bid_price)
                logger.info(
                    f"  {fighter}: ask ${live_ask:.4f} too expensive "
                    f"(edge {live_edge:+.1%}), placing limit bid @ ${bid_price:.4f} "
                    f"(edge if filled: {edge:+.1%})"
                )
            else:
                # Ask price has edge — proceed with market buy
                price = live_ask
                edge = live_edge
                odds = implied_prob_to_decimal_odds(live_ask)
                logger.info(
                    f"  {fighter}: live ask ${live_ask:.4f} "
                    f"(snapshot was ${market_prob:.4f}), "
                    f"edge {live_edge:+.1%}"
                )

            # Recalculate bet size with live odds (skip for override/conviction bets)
            if override is None or override <= 0:
                bet_size = self.bankroll.kelly_bet_size(blended_prob, odds)
                if bet_size <= 0:
                    return None

            # Apply liquidity adjustments from the orderbook check
            if not use_limit_bid:
                bet_size = min(bet_size, liq["adjusted_size"])
            if liq["slippage"] > 0 and not use_limit_bid:
                logger.info(
                    f"  {fighter}: ${liq['available_liquidity']:.0f} book liquidity, "
                    f"{liq['slippage']:.1%} est. slippage"
                )
        else:
            price = market_prob
            # In dry run, still log what we'd check
            logger.info(
                f"  [DRY RUN] Would check orderbook for {fighter} "
                f"(token: {token_id[:16]}...)"
            )

        # Calculate shares: bet_size / price
        shares = bet_size / price if price > 0 else 0

        order_info = {
            "fighter": fighter,
            "side": "BUY",
            "token_id": token_id,
            "price": round(price, 4),
            "shares": round(shares, 2),
            "bet_size_usd": bet_size,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": edge,
            "dry_run": self.dry_run,
        }

        if self.dry_run:
            order_type = "limit_bid" if use_limit_bid else "market"
            logger.info(
                f"[DRY RUN] Would place: {order_type.upper()} BUY {shares:.1f} shares "
                f"of {fighter} @ ${price:.4f} (${bet_size:.2f} total) | "
                f"Edge: {edge:.1%}"
            )
            order_info["status"] = "dry_run"
            order_info["order_type"] = order_type
        elif use_limit_bid:
            # Place a resting limit bid — gets filled if price drops to our level
            try:
                tick_size = str(bet.get("tick_size", "0.01"))
                response = self.clob.create_limit_order(
                    token_id=token_id,
                    side="BUY",
                    price=price,
                    size=shares,
                    tick_size=tick_size,
                    neg_risk=bet.get("neg_risk", False),
                )
                order_info["response"] = response
                order_info["status"] = "placed"
                order_info["order_type"] = "limit_bid"
                logger.info(
                    f"Limit bid placed for {fighter}: "
                    f"BUY {shares:.1f} @ ${price:.4f} (${bet_size:.2f}) | "
                    f"Edge if filled: {edge:.1%} | {response}"
                )
            except Exception as e:
                order_info["status"] = "failed"
                order_info["error"] = str(e)
                logger.error(f"Failed to place limit bid for {fighter}: {e}")
        else:
            # Market buy — ask price has edge
            try:
                tick_size = str(bet.get("tick_size", "0.01"))
                response = self.clob.create_market_order(
                    token_id=token_id,
                    side="BUY",
                    amount=bet_size,
                    tick_size=tick_size,
                    neg_risk=bet.get("neg_risk", False),
                )
                order_info["response"] = response
                order_info["status"] = "placed"
                order_info["order_type"] = "market"
                logger.info(
                    f"Market order filled for {fighter}: "
                    f"${bet_size:.2f} | Edge: {edge:.1%} | {response}"
                )
            except Exception as e:
                # Fall back to limit order at live ask if market order fails
                logger.warning(
                    f"Market order failed for {fighter}: {e} — "
                    f"falling back to limit order"
                )
                try:
                    tick_size = str(bet.get("tick_size", "0.01"))
                    response = self.clob.create_limit_order(
                        token_id=token_id,
                        side="BUY",
                        price=price,
                        size=shares,
                        tick_size=tick_size,
                        neg_risk=bet.get("neg_risk", False),
                    )
                    order_info["response"] = response
                    order_info["status"] = "placed"
                    order_info["order_type"] = "limit"
                    logger.info(f"Limit order placed for {fighter}: {response}")
                except Exception as e2:
                    order_info["status"] = "failed"
                    order_info["error"] = str(e2)
                    logger.error(f"Failed to place order for {fighter}: {e2}")

        # Record bet in bankroll manager and persistent ledger
        if order_info["status"] in ("placed", "dry_run"):
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=odds,
                model_prob=model_prob,
                market_prob=market_prob,
            )

            # Determine opponent name
            opponent = ""
            if bet["bet_side"] == "a":
                opponent = str(bet.get("fighter_b", ""))
            else:
                opponent = str(bet.get("fighter_a", ""))

            # Extract order ID from CLOB response (if available)
            resp = order_info.get("response", {})
            clob_order_id = None
            if isinstance(resp, dict):
                clob_order_id = resp.get("orderID") or resp.get("id")

            self.ledger.add_bet(
                fighter=fighter,
                opponent=opponent,
                side=bet["bet_side"],
                amount=bet_size,
                price=price,
                shares=shares,
                token_id=token_id,
                market_id=str(bet.get("market_id", "")),
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                decimal_odds=odds,
                dry_run=self.dry_run,
                event_date=str(bet.get("event_date", "")),
                order_type=order_info.get("order_type"),
                order_id=clob_order_id,
            )

        self.order_log.append(order_info)
        return order_info

    def cancel_stale_limit_bids(self, ledger: Optional[BetLedger] = None) -> int:
        """
        Cancel open limit bids that are stale or approaching event time.

        A limit bid is cancelled if:
        - The event is within LIMIT_BID_PRE_EVENT_HOURS of starting, OR
        - The fight has already started (event_date <= now), OR
        - The bid has been resting longer than LIMIT_BID_TTL_HOURS

        Returns the number of orders cancelled.
        """
        from datetime import datetime, timezone, timedelta

        target_ledger = ledger or self.ledger
        now = datetime.now(timezone.utc)
        ttl = timedelta(hours=LIMIT_BID_TTL_HOURS)
        pre_event_buffer = timedelta(hours=LIMIT_BID_PRE_EVENT_HOURS)
        cancelled = 0

        for bet in list(target_ledger.bets):
            if bet.get("status") != "open":
                continue
            if bet.get("order_type") not in ("limit_bid", "limit"):
                continue
            if bet.get("dry_run"):
                continue

            fighter = bet.get("fighter", "?")
            order_id = bet.get("order_id")
            cancel_reason = None

            # Check 1: fight is about to start (cancel before event begins)
            event_date = bet.get("event_date")
            if event_date:
                try:
                    if "T" in str(event_date):
                        fight_time = datetime.fromisoformat(
                            str(event_date).replace("Z", "+00:00")
                        )
                    else:
                        fight_time = datetime.fromisoformat(str(event_date)).replace(
                            tzinfo=timezone.utc
                        )
                    if fight_time.tzinfo is None:
                        fight_time = fight_time.replace(tzinfo=timezone.utc)
                    cancel_deadline = fight_time - pre_event_buffer
                    if now >= cancel_deadline:
                        if now >= fight_time:
                            cancel_reason = "fight started"
                        else:
                            mins_left = int((fight_time - now).total_seconds() / 60)
                            cancel_reason = f"pre-event pull ({mins_left}min to event)"
                except (ValueError, TypeError):
                    pass

            # Check 2: bid has exceeded TTL
            if not cancel_reason:
                placed_at = bet.get("placed_at")
                if placed_at:
                    try:
                        placed_time = datetime.fromisoformat(str(placed_at))
                        if placed_time.tzinfo is None:
                            placed_time = placed_time.replace(tzinfo=timezone.utc)
                        if now - placed_time >= ttl:
                            cancel_reason = f"exceeded {LIMIT_BID_TTL_HOURS}h TTL"
                    except (ValueError, TypeError):
                        pass

            if not cancel_reason:
                continue

            if not order_id:
                logger.warning(
                    f"Cannot cancel limit bid for {fighter}: no order ID stored "
                    f"(bet #{bet['id']})"
                )
                continue

            try:
                self.clob.cancel_order(order_id)
                logger.info(
                    f"Cancelled limit bid for {fighter}: "
                    f"order {order_id} ({cancel_reason})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to cancel order {order_id} for {fighter}: {e}"
                )
                continue

            # Mark as cancelled in the ledger only after successful exchange cancel
            target_ledger.cancel_bet(bet["id"])
            cancelled += 1

        if cancelled:
            logger.info(f"Cancelled {cancelled} stale limit bid(s)")
        return cancelled

    def get_order_log(self) -> pd.DataFrame:
        """Get log of all orders placed."""
        return pd.DataFrame(self.order_log)


def cancel_all_stale_limit_bids(clob_client: Optional[ClobClientWrapper] = None) -> int:
    """
    Cancel stale limit bids across all trader ledgers.

    Called from the live betting loop before placing new bets.
    """
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER
    from src.strategy.bankroll import BankrollManager

    client = clob_client or ClobClientWrapper()
    total = 0

    for label, path in [("S", SINGLE_LEDGER), ("C", CONVICTION_LEDGER)]:
        ledger = BetLedger(path=path)
        executor = OrderExecutor(
            bankroll=BankrollManager(initial_bankroll=0, auto_detect_balance=False),
            clob_client=client,
            dry_run=False,
        )
        n = executor.cancel_stale_limit_bids(ledger=ledger)
        if n:
            logger.info(f"Trader {label}: cancelled {n} stale limit bid(s)")
        total += n

    return total


def _name_match(name1: str, name2: str) -> bool:
    """
    Fuzzy match two fighter names.
    Handles variations like "Jon Jones" vs "Jonathan Jones" or "Jon 'Bones' Jones".
    """
    if not name1 or not name2:
        return False

    # Exact match
    if name1 == name2:
        return True

    # Remove nicknames in quotes
    import re
    clean1 = re.sub(r"['\"].*?['\"]", "", name1).strip()
    clean2 = re.sub(r"['\"].*?['\"]", "", name2).strip()
    if clean1 == clean2:
        return True

    # Last name match (for cases like "Jon Jones" vs "Jonathan Jones")
    parts1 = clean1.split()
    parts2 = clean2.split()
    if parts1 and parts2 and parts1[-1] == parts2[-1]:
        # Same last name — require first name containment (not just same initial)
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] in parts2[0] or parts2[0] in parts1[0]:
                return True

    return False
