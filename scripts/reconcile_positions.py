"""Import untracked Polymarket positions into the bot ledger.

Runs at startup on Railway to prevent wallet/ledger mismatch errors.
Only imports positions that are not already tracked in any ledger.
"""

import sys
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from src.config import LOGS_DIR, POLYMARKET_DATA_API_URL, POLYMARKET_FUNDER_ADDRESS
from src.polymarket.executor import _reconcile_import_positions
from src.polymarket.market_lookup import load_supported_market_token_lookup
from src.polymarket.tracker import BetLedger
from src.strategy.duo_trader import SINGLE_LEDGER, get_all_trader_ledgers

LEGACY_LEDGER = LOGS_DIR / "bet_ledger.json"
TRACKER_TRADERS = {"M", "G"}
TRACKER_IMPORT_REPAIR_REASON = "duplicate_tracker_reconciliation_import"
SHARE_COVERAGE_TOLERANCE = 0.02


def _ledger_registry() -> list[tuple[str, Path]]:
    """Return every active trader ledger plus the pre-duo legacy ledger."""
    registry = [
        (str(label or "").strip().upper(), Path(path))
        for label, path in get_all_trader_ledgers()
        if path is not None
    ]
    registry.append(("LEGACY", Path(LEGACY_LEDGER)))

    deduped: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in registry:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((label, path))
    return deduped


def _open_ledger_rows() -> list[dict]:
    rows: list[dict] = []
    for trader, ledger_path in _ledger_registry():
        if not ledger_path.exists():
            continue
        ledger = BetLedger(path=ledger_path)
        for bet in ledger.get_open_bets():
            rows.append(
                {
                    **bet,
                    "_trader": trader,
                    "_ledger_path": str(ledger_path),
                }
            )
    return rows


def _parse_placed_at(raw) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _positive_float(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) and value > 0 else 0.0


def _positive_int(raw) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        if not isfinite(raw) or not raw.is_integer():
            return None
        value = int(raw)
        return value if value > 0 else None
    value_text = str(raw or "").strip()
    if not value_text.isdigit():
        return None
    value = int(value_text)
    return value if value > 0 else None


def _live_position_sizes(live_positions: list[dict]) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for position in live_positions:
        token_id = str(
            position.get("asset", position.get("token_id", "")) or ""
        ).strip()
        size = _positive_float(position.get("size"))
        if token_id and size > 0:
            sizes[token_id] = max(sizes.get(token_id, 0.0), size)
    return sizes


def _confirmed_tracker_shares(row: dict) -> float:
    order_type = str(row.get("order_type", "") or "").strip().lower()
    if order_type == "imported":
        return 0.0

    actual_shares = (
        _positive_float(row.get("actual_filled_shares"))
        or _positive_float(row.get("filled_shares"))
    )
    if actual_shares > 0:
        return actual_shares
    if order_type == "filled_limit":
        return _positive_float(row.get("shares"))
    if (
        str(row.get("placement_state", "") or "").strip().lower()
        in {"filled", "matched"}
    ):
        return _positive_float(row.get("shares"))
    return 0.0


def _repair_tracker_shadow_imports(
    open_rows: list[dict],
    live_positions: list[dict],
) -> int:
    """Cancel false S imports that merely mirror earlier M/G wallet exposure.

    Polymarket reports one aggregate wallet position per token. Older startup
    reconciliation omitted M/G ledgers, so their combined shares were imported
    later as a synthetic S row. Repair only exact, still-live matches and fail
    closed for ambiguous or malformed ledger history.
    """
    single_path = Path(SINGLE_LEDGER).resolve()
    live_sizes = _live_position_sizes(live_positions)
    tracker_rows_by_token: dict[str, list[dict]] = {}
    imported_single_rows_by_token: dict[str, list[dict]] = {}

    for row in open_rows:
        token_id = str(row.get("token_id", "") or "").strip()
        raw_path = str(row.get("_ledger_path", "") or "").strip()
        if (
            not token_id
            or not raw_path
            or row.get("status") != "open"
            or row.get("dry_run") is not False
        ):
            continue
        row_path = Path(raw_path).resolve()
        trader = str(row.get("_trader", "") or "").strip().upper()
        if (
            row_path == single_path
            and str(row.get("order_type", "") or "").strip().lower() == "imported"
            and str(row.get("placement_state", "") or "").strip().lower()
            == "filled"
            and not str(row.get("order_id", "") or "").strip()
        ):
            imported_single_rows_by_token.setdefault(token_id, []).append(row)
        elif trader in TRACKER_TRADERS:
            tracker_shares = _confirmed_tracker_shares(row)
            if tracker_shares > 0:
                tracker_rows_by_token.setdefault(token_id, []).append(
                    {**row, "_confirmed_shares": tracker_shares}
                )

    if not imported_single_rows_by_token:
        return 0

    single_ledger = BetLedger(path=SINGLE_LEDGER)
    repaired = 0
    for token_id, imported_rows in imported_single_rows_by_token.items():
        if len(imported_rows) != 1:
            continue
        imported = imported_rows[0]
        imported_at = _parse_placed_at(imported.get("placed_at"))
        imported_shares = _positive_float(imported.get("shares"))
        imported_id = _positive_int(imported.get("id"))
        live_size = live_sizes.get(token_id, 0.0)
        if (
            imported_at is None
            or imported_shares <= 0
            or imported_id is None
            or (
                abs(live_size - imported_shares)
                > SHARE_COVERAGE_TOLERANCE + 1e-9
            )
        ):
            continue

        qualifying_by_trader: dict[str, list[dict]] = {}
        for tracker in tracker_rows_by_token.get(token_id, []):
            tracker_at = _parse_placed_at(tracker.get("placed_at"))
            tracker_shares = _positive_float(tracker.get("_confirmed_shares"))
            if (
                tracker_at is None
                or tracker_at >= imported_at
                or tracker_shares <= 0
            ):
                continue
            trader = str(tracker.get("_trader", "") or "").strip().upper()
            qualifying_by_trader.setdefault(trader, []).append(tracker)

        if not qualifying_by_trader or any(
            len(rows) != 1 for rows in qualifying_by_trader.values()
        ):
            continue

        earlier_tracker_shares = sum(
            _positive_float(rows[0].get("_confirmed_shares"))
            for rows in qualifying_by_trader.values()
        )
        if (
            abs(earlier_tracker_shares - imported_shares)
            > SHARE_COVERAGE_TOLERANCE + 1e-9
        ):
            continue

        result = single_ledger.cancel_bet(
            imported_id,
            reason=TRACKER_IMPORT_REPAIR_REASON,
            expected_order_types={"imported"},
        )
        if result.ok:
            print(
                "[reconcile] Cancelled false S import "
                f"#{imported['id']} for token {token_id[:16]}...; "
                f"{earlier_tracker_shares:.2f} earlier M/G shares cover "
                f"{imported_shares:.2f} imported shares"
            )
            repaired += 1
    return repaired


def main():
    wallet = POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        raise RuntimeError("POLYMARKET_FUNDER_ADDRESS is required for live reconciliation")

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

    # Repair old startup imports before determining which wallet positions are
    # already owned by S/C/M, the retired G ledger, or the pre-duo legacy ledger.
    open_rows = _open_ledger_rows()
    repaired = _repair_tracker_shadow_imports(open_rows, positions)
    if repaired:
        open_rows = _open_ledger_rows()

    tracked_tokens = {
        str(bet.get("token_id", "") or "").strip()
        for bet in open_rows
        if (
            bet.get("dry_run") is False
            and str(bet.get("token_id", "") or "").strip()
        )
    }

    token_lookup = load_supported_market_token_lookup()
    if not token_lookup:
        raise RuntimeError("Supported market token metadata is required to reconcile live positions")

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
