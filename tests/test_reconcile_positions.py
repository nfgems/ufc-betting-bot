from pathlib import Path

import pytest

from scripts import reconcile_positions
from src.polymarket.tracker import BetLedger


TOKEN = "walker-token"
EARLY_M = "2026-07-23T05:13:21.591794-04:00"
EARLY_G = "2026-07-23T05:25:32.389101-04:00"
IMPORTED_AT = "2026-07-23T22:29:48.066880-04:00"
LATER = "2026-07-24T03:00:00+00:00"


def _configure_ledgers(monkeypatch, tmp_path):
    paths = {
        "S": tmp_path / "single.json",
        "C": tmp_path / "conviction.json",
        "M": tmp_path / "model_tracker.json",
        "G": tmp_path / "gemini_tracker.json",
        "LEGACY": tmp_path / "legacy.json",
    }
    monkeypatch.setattr(reconcile_positions, "SINGLE_LEDGER", paths["S"])
    monkeypatch.setattr(reconcile_positions, "LEGACY_LEDGER", paths["LEGACY"])
    monkeypatch.setattr(
        reconcile_positions,
        "get_all_trader_ledgers",
        lambda: [(label, paths[label]) for label in ("S", "C", "M", "G")],
    )
    return paths


def _add_bet(
    path: Path,
    *,
    token_id: str = TOKEN,
    shares: float,
    placed_at: str,
    order_type: str,
    placement_state: str,
    dry_run: bool = False,
    order_id: str | None = None,
    actual_filled_shares: float | None = None,
):
    ledger = BetLedger(path=path)
    metadata = {}
    if actual_filled_shares is not None:
        metadata["actual_filled_shares"] = actual_filled_shares
    bet = ledger.add_bet(
        fighter="Valter Walker",
        opponent="Thomas Petersen",
        side="b",
        amount=round(shares * 0.63, 2),
        price=0.63,
        shares=shares,
        token_id=token_id,
        market_id="2885013",
        model_prob=0.76,
        market_prob=0.63,
        edge=0.13,
        decimal_odds=1.5873,
        dry_run=dry_run,
        order_type=order_type,
        order_id=order_id,
        placement_state=placement_state,
        metadata=metadata,
    )
    result = ledger.update_bet_fields(bet["id"], placed_at=placed_at)
    assert result.ok
    return bet["id"]


def _walker_live_position(size=6.36):
    return [{"asset": TOKEN, "size": size}]


def test_repairs_exact_walker_tracker_shadow_import_idempotently(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=3.18,
        placed_at=EARLY_M,
        order_type="filled_limit",
        placement_state="submitted",
    )
    _add_bet(
        paths["G"],
        shares=3.18,
        placed_at=EARLY_G,
        order_type="filled_limit",
        placement_state="submitted",
    )
    imported_id = _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(),
    )

    assert repaired == 1
    imported = BetLedger(path=paths["S"]).get_bets()[0]
    assert imported["id"] == imported_id
    assert imported["status"] == "cancelled"
    assert (
        imported["cancel_reason"]
        == reconcile_positions.TRACKER_IMPORT_REPAIR_REASON
    )
    assert len(BetLedger(path=paths["M"]).get_open_bets()) == 1
    assert len(BetLedger(path=paths["G"]).get_open_bets()) == 1
    assert (
        reconcile_positions._repair_tracker_shadow_imports(
            reconcile_positions._open_ledger_rows(),
            _walker_live_position(),
        )
        == 0
    )


@pytest.mark.parametrize(
    ("imported_shares", "expected_repaired"),
    [
        (6.38, 1),
        (6.39, 0),
        (6.33, 0),
    ],
)
def test_tracker_shadow_repair_requires_two_cent_exact_coverage(
    monkeypatch,
    tmp_path,
    imported_shares,
    expected_repaired,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=6.36,
        placed_at=EARLY_M,
        order_type="filled_limit",
        placement_state="submitted",
    )
    _add_bet(
        paths["S"],
        shares=imported_shares,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(size=max(imported_shares, 6.36)),
    )

    assert repaired == expected_repaired


@pytest.mark.parametrize(
    (
        "order_type",
        "placement_state",
        "order_id",
        "placed_at",
        "dry_run",
    ),
    [
        ("marketable_limit", "submitted", "order-1", EARLY_M, False),
        ("marketable_limit", "unknown", None, EARLY_M, False),
        ("filled_limit", "submitted", None, LATER, False),
        ("filled_limit", "submitted", None, EARLY_M, True),
        ("imported", "filled", None, EARLY_M, False),
    ],
)
def test_tracker_shadow_repair_ignores_unconfirmed_or_ineligible_tracker_rows(
    monkeypatch,
    tmp_path,
    order_type,
    placement_state,
    order_id,
    placed_at,
    dry_run,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=6.36,
        placed_at=placed_at,
        order_type=order_type,
        placement_state=placement_state,
        order_id=order_id,
        dry_run=dry_run,
    )
    _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(),
    )

    assert repaired == 0
    assert BetLedger(path=paths["S"]).get_bets()[0]["status"] == "open"


def test_tracker_shadow_repair_accepts_explicit_actual_fill_evidence(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=10.0,
        actual_filled_shares=6.36,
        placed_at=EARLY_M,
        order_type="marketable_limit",
        placement_state="submitted",
        order_id="order-1",
    )
    _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(),
    )

    assert repaired == 1


def test_tracker_shadow_repair_rejects_ambiguous_same_trader_rows(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    for placed_at in (EARLY_M, EARLY_G):
        _add_bet(
            paths["M"],
            shares=3.18,
            placed_at=placed_at,
            order_type="filled_limit",
            placement_state="submitted",
        )
    _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(),
    )

    assert repaired == 0


@pytest.mark.parametrize(
    "live_positions",
    [
        [],
        _walker_live_position(6.33),
        _walker_live_position(12.72),
    ],
)
def test_tracker_shadow_repair_requires_matching_live_wallet_position(
    monkeypatch,
    tmp_path,
    live_positions,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=6.36,
        placed_at=EARLY_M,
        order_type="filled_limit",
        placement_state="submitted",
    )
    _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        live_positions,
    )

    assert repaired == 0


def test_tracker_shadow_repair_rejects_multiple_imports_for_same_token(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["M"],
        shares=6.36,
        placed_at=EARLY_M,
        order_type="filled_limit",
        placement_state="submitted",
    )
    for placed_at in (IMPORTED_AT, "2026-07-23T23:00:00-04:00"):
        _add_bet(
            paths["S"],
            shares=6.36,
            placed_at=placed_at,
            order_type="imported",
            placement_state="filled",
        )

    repaired = reconcile_positions._repair_tracker_shadow_imports(
        reconcile_positions._open_ledger_rows(),
        _walker_live_position(),
    )

    assert repaired == 0
    assert len(BetLedger(path=paths["S"]).get_open_bets()) == 2


def test_main_repairs_shadow_without_reimporting_tracker_position(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    for label, placed_at in (("M", EARLY_M), ("G", EARLY_G)):
        _add_bet(
            paths[label],
            shares=3.18,
            placed_at=placed_at,
            order_type="filled_limit",
            placement_state="submitted",
        )
    _add_bet(
        paths["S"],
        shares=6.36,
        placed_at=IMPORTED_AT,
        order_type="imported",
        placement_state="filled",
    )
    live_positions = [
        {"asset": TOKEN, "size": 6.36},
        {"asset": "genuinely-untracked", "size": 2.0},
    ]

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return live_positions

    captured = {}

    def _capture_import(positions, token_lookup, tracked_tokens, **kwargs):
        captured["positions"] = positions
        captured["tracked_tokens"] = set(tracked_tokens)
        captured["import_ledger_path"] = kwargs["import_ledger_path"]
        return 1

    monkeypatch.setattr(
        reconcile_positions,
        "POLYMARKET_FUNDER_ADDRESS",
        "0x1234567890",
    )
    monkeypatch.setattr(reconcile_positions.requests, "get", lambda *a, **k: _Response())
    monkeypatch.setattr(
        reconcile_positions,
        "load_supported_market_token_lookup",
        lambda: {TOKEN: {}, "genuinely-untracked": {}},
    )
    monkeypatch.setattr(
        reconcile_positions,
        "_reconcile_import_positions",
        _capture_import,
    )

    reconcile_positions.main()

    assert TOKEN in captured["tracked_tokens"]
    assert "genuinely-untracked" not in captured["tracked_tokens"]
    assert captured["import_ledger_path"] == paths["S"]
    assert BetLedger(path=paths["S"]).get_bets()[0]["status"] == "cancelled"


def test_main_does_not_treat_dry_run_ledger_row_as_live_tracking(
    monkeypatch,
    tmp_path,
):
    paths = _configure_ledgers(monkeypatch, tmp_path)
    _add_bet(
        paths["S"],
        shares=2.0,
        placed_at=EARLY_M,
        order_type="marketable_limit",
        placement_state="dry_run",
        dry_run=True,
    )

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return [{"asset": TOKEN, "size": 2.0}]

    captured = {}

    def _capture_import(positions, token_lookup, tracked_tokens, **kwargs):
        captured["tracked_tokens"] = set(tracked_tokens)
        return 1

    monkeypatch.setattr(
        reconcile_positions,
        "POLYMARKET_FUNDER_ADDRESS",
        "0x1234567890",
    )
    monkeypatch.setattr(reconcile_positions.requests, "get", lambda *a, **k: _Response())
    monkeypatch.setattr(
        reconcile_positions,
        "load_supported_market_token_lookup",
        lambda: {TOKEN: {}},
    )
    monkeypatch.setattr(
        reconcile_positions,
        "_reconcile_import_positions",
        _capture_import,
    )

    reconcile_positions.main()

    assert TOKEN not in captured["tracked_tokens"]


def test_entrypoint_migrates_both_tracker_ledgers():
    entrypoint = (
        Path(__file__).resolve().parents[1] / "entrypoint.sh"
    ).read_text(encoding="utf-8")

    assert "copy_log_file bet_ledger_model_tracker.json" in entrypoint
    assert "copy_log_file bet_ledger_gemini_tracker.json" in entrypoint
