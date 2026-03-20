"""Import untracked Polymarket positions into the bot ledger.

Runs at startup on Railway to prevent wallet/ledger mismatch errors.
Only imports positions that are not already tracked in any ledger.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from src.config import LOGS_DIR, POLYMARKET_DATA_API_URL, POLYMARKET_FUNDER_ADDRESS
from src.polymarket.tracker import BetLedger

SINGLE_LEDGER = LOGS_DIR / "bet_ledger_single.json"


def main():
    wallet = POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        print("[reconcile] No POLYMARKET_FUNDER_ADDRESS set, skipping")
        return

    print(f"[reconcile] Fetching positions for {wallet[:10]}...")
    resp = requests.get(
        f"{POLYMARKET_DATA_API_URL}/positions",
        params={"user": wallet},
        timeout=30,
    )
    resp.raise_for_status()
    positions = [p for p in resp.json() if float(p.get("size", 0)) > 0]
    print(f"[reconcile] Found {len(positions)} live positions on Polymarket")

    if not positions:
        return

    # Load all existing ledger entries to find tracked token IDs
    tracked_tokens = set()
    for ledger_name in ("bet_ledger_single.json", "bet_ledger_conviction.json", "bet_ledger.json"):
        ledger_path = LOGS_DIR / ledger_name
        if ledger_path.exists():
            ledger = BetLedger(path=ledger_path)
            for bet in ledger.get_open_bets():
                token = str(bet.get("token_id", "") or "").strip()
                if token:
                    tracked_tokens.add(token)

    # Import untracked positions
    target_ledger = BetLedger(path=SINGLE_LEDGER)
    imported = 0

    for pos in positions:
        asset_id = str(pos.get("asset", pos.get("token_id", "")))
        if asset_id in tracked_tokens:
            continue

        size = float(pos.get("size", 0))
        title = pos.get("title", pos.get("question", "unknown"))
        avg_price = float(pos.get("avgPrice", pos.get("avg_price", 0.5)))
        market_id = str(pos.get("market", pos.get("condition_id", "")))
        outcome = pos.get("outcome", "Yes")

        # Parse fighter names from title
        parts = title.split(":")[-1].strip() if ":" in title else title
        vs_parts = parts.split(" vs. ") if " vs. " in parts else parts.split(" vs ")
        fighter = vs_parts[0].strip() if vs_parts else "Unknown"
        opponent_raw = vs_parts[1].strip() if len(vs_parts) > 1 else "Unknown"
        if "(" in opponent_raw:
            opponent_raw = opponent_raw[:opponent_raw.index("(")].strip()

        amount = round(size * avg_price, 2)

        bet = target_ledger.add_bet(
            fighter=fighter,
            opponent=opponent_raw,
            side=outcome,
            amount=amount,
            price=avg_price,
            shares=round(size, 2),
            token_id=asset_id,
            market_id=market_id,
            model_prob=0.0,
            market_prob=avg_price,
            edge=0.0,
            decimal_odds=round(1.0 / avg_price, 4) if avg_price > 0 else 0,
            dry_run=False,
            order_type="imported",
            status="open",
            placement_state="filled",
        )
        print(f"[reconcile] Imported: {title} (size={size:.2f}, price={avg_price:.4f}) -> bet #{bet['id']}")
        imported += 1

    if imported:
        print(f"[reconcile] Imported {imported} untracked positions into {SINGLE_LEDGER}")
    else:
        print("[reconcile] All positions already tracked, nothing to import")


if __name__ == "__main__":
    main()
