"""
Manual live smoke test: place a $1 real-money order on the first available UFC market.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_bet")

import pandas as pd
from src.polymarket.client import ClobClientWrapper
from src.polymarket.markets import get_ufc_fight_markets


def main():
    confirmation = input(
        "This script will place a real-money order on Polymarket. "
        "Type YES to continue: "
    ).strip()
    if confirmation != "YES":
        logger.info("Aborted manual live bet smoke test.")
        return

    # 1. Initialize client and check balance
    logger.info("=== STEP 1: Connect and check balance ===")
    clob = ClobClientWrapper()
    balance_before = clob.get_cash_balance()
    logger.info(f"Balance BEFORE: ${balance_before:.2f}")

    # 2. Find an available UFC market
    logger.info("\n=== STEP 2: Find a UFC market ===")
    markets = get_ufc_fight_markets()
    if markets.empty:
        logger.error("No active UFC markets found on Polymarket. Cannot test.")
        return

    # Pick the market with the most volume (most liquid)
    markets["volume"] = pd.to_numeric(markets["volume"], errors="coerce").fillna(0)
    market = markets.sort_values("volume", ascending=False).iloc[0]
    logger.info(
        f"Selected: {market['fighter_a']} vs {market['fighter_b']}\n"
        f"  Price YES ({market['fighter_a']}): {market['price_yes']}\n"
        f"  Price NO  ({market['fighter_b']}): {market['price_no']}\n"
        f"  Volume: ${market['volume']}\n"
        f"  Token YES: {market['token_id_yes'][:20]}..."
    )

    # 3. Place a $1 limit order on the YES side (fighter_a)
    logger.info("\n=== STEP 3: Place $1 test bet ===")
    token_id = market["token_id_yes"]
    price = round(float(market["price_yes"]), 2)

    # Buy at the current price — $1 worth of shares
    shares = round(1.0 / price, 2) if price > 0 else 0
    logger.info(
        f"Placing: BUY {shares} shares of {market['fighter_a']} "
        f"@ ${price:.2f} (=${price * shares:.2f} total)"
    )

    try:
        tick_size = str(market.get("tick_size", "0.01"))
        response = clob.create_limit_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=shares,
            tick_size=tick_size,
            neg_risk=market.get("neg_risk", False),
        )
        logger.info(f"Order response: {response}")
    except Exception as e:
        logger.error(f"Order FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check balance again
    logger.info("\n=== STEP 4: Check balance after bet ===")
    balance_after = clob.get_cash_balance()
    logger.info(f"Balance AFTER:  ${balance_after:.2f}")
    logger.info(f"Difference:     ${balance_after - balance_before:.2f}")

    if abs(balance_after - balance_before) > 0.001:
        logger.info("SUCCESS — Balance updated after bet!")
    else:
        logger.info(
            "Balance unchanged — order may be pending (limit order not yet filled). "
            "Check open orders or try a market order."
        )

    # 5. Show open orders
    logger.info("\n=== STEP 5: Check open orders ===")
    try:
        orders = clob.get_open_orders()
        logger.info(f"Open orders: {len(orders) if orders else 0}")
        if orders:
            for o in orders[:5]:
                logger.info(f"  {o}")
    except Exception as e:
        logger.warning(f"Could not fetch open orders: {e}")


if __name__ == "__main__":
    main()
