"""
Order executor — places and manages bets on Polymarket based on model signals.
"""

import logging
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
            "reason": "",
        }

        try:
            book = self.clob.get_orderbook(token_id)
        except Exception as e:
            logger.warning(f"Could not fetch orderbook for {fighter}: {e}")
            # If we can't check, proceed with caution (small size)
            result["adjusted_size"] = min(desired_size_usd, MIN_BOOK_LIQUIDITY * 0.5)
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
                f"  Reducing bet on {fighter}: ${desired_size_usd:.2f} → "
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
        market_prob = bet["market_prob"]
        edge = bet["edge"]
        odds = bet["decimal_odds"]

        # Calculate bet size using Kelly
        bet_size = self.bankroll.kelly_bet_size(model_prob, odds)
        if bet_size <= 0:
            return None

        # Determine which token to buy
        if bet["bet_side"] == "a":
            token_id = bet.get("token_id_yes", "")
            price = market_prob  # Buy YES at market probability
        else:
            token_id = bet.get("token_id_no", "")
            price = market_prob

        if not token_id:
            logger.warning(f"No token ID for {fighter}")
            return None

        # Check orderbook liquidity before placing
        if not self.dry_run:
            liq = self._check_liquidity(token_id, price, bet_size, fighter)
            if not liq["ok"]:
                logger.warning(f"Skipping {fighter}: {liq['reason']}")
                return None
            bet_size = liq["adjusted_size"]
            if liq["slippage"] > 0:
                logger.info(
                    f"  {fighter}: ${liq['available_liquidity']:.0f} book liquidity, "
                    f"{liq['slippage']:.1%} est. slippage"
                )
        else:
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
            logger.info(
                f"[DRY RUN] Would place: BUY {shares:.1f} shares of {fighter} "
                f"@ ${price:.4f} (${bet_size:.2f} total) | "
                f"Edge: {edge:.1%}"
            )
            order_info["status"] = "dry_run"
        else:
            try:
                # Use market order (FOK) for immediate execution
                response = self.clob.create_market_order(
                    token_id=token_id,
                    side="BUY",
                    amount=bet_size,
                )
                order_info["response"] = response
                order_info["status"] = "placed"
                order_info["order_type"] = "market"
                logger.info(
                    f"Market order filled for {fighter}: "
                    f"${bet_size:.2f} | Edge: {edge:.1%} | {response}"
                )
            except Exception as e:
                # Fall back to limit order if market order fails
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
            )

        self.order_log.append(order_info)
        return order_info

    def get_order_log(self) -> pd.DataFrame:
        """Get log of all orders placed."""
        return pd.DataFrame(self.order_log)


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
