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
from src.config import MIN_EDGE_THRESHOLD

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
                logger.info(f"Order placed for {fighter}: {response}")
            except Exception as e:
                order_info["status"] = "failed"
                order_info["error"] = str(e)
                logger.error(f"Failed to place order for {fighter}: {e}")

        # Record bet in bankroll manager
        if order_info["status"] in ("placed", "dry_run"):
            self.bankroll.place_bet(
                amount=bet_size,
                fighter=fighter,
                decimal_odds=odds,
                model_prob=model_prob,
                market_prob=market_prob,
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

    # Last name match (for cases like "J. Jones" vs "Jon Jones")
    parts1 = clean1.split()
    parts2 = clean2.split()
    if parts1 and parts2 and parts1[-1] == parts2[-1]:
        # Same last name — check if first names are compatible
        if len(parts1) >= 2 and len(parts2) >= 2:
            # First name starts with same letter or one contains the other
            if (parts1[0][0] == parts2[0][0] or
                parts1[0] in parts2[0] or parts2[0] in parts1[0]):
                return True

    # Substring match (one name contained in the other)
    if len(name1) > 5 and len(name2) > 5:
        if name1 in name2 or name2 in name1:
            return True

    return False
