"""Import untracked Polymarket positions into the bot ledger.

Runs at startup on Railway to prevent wallet/ledger mismatch errors.
Only imports positions that are not already tracked in any ledger.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from src.config import LOGS_DIR, POLYMARKET_DATA_API_URL, POLYMARKET_FUNDER_ADDRESS
from src.polymarket.executor import _reconcile_import_positions
from src.polymarket.market_lookup import load_supported_market_token_lookup
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

    token_lookup = load_supported_market_token_lookup()
    if not token_lookup:
        print("[reconcile] No supported market token metadata available, skipping import")
        return

    imported = _reconcile_import_positions(
        positions,
        token_lookup,
        tracked_tokens,
        import_ledger_path=SINGLE_LEDGER,
    )

    if imported:
        print(f"[reconcile] Imported {imported} untracked positions into {SINGLE_LEDGER}")
    else:
        print("[reconcile] All positions already tracked, nothing to import")


if __name__ == "__main__":
    main()
