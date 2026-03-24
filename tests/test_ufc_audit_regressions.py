import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from bs4 import BeautifulSoup

import src.bot as bot_module
import src.strategy.duo_trader as duo_trader
from src.data import fighter_lookup, historical_backfill, ufc_refresh
from src.data.kaggle_loader import load_kaggle_dataset
from src.features import build_features as build_features_module
from src.model import training_spec
from src.model.train import prepare_train_test
from src.polymarket import tracker as tracker_module
from src.polymarket.tracker import BetLedger, load_all_trader_ledgers
from src.strategy import model_variants
from src.web import app as web_app

_AUDIT_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_model_feature_nulls.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("audit_model_feature_nulls", _AUDIT_SCRIPT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
audit_model_feature_nulls = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules.setdefault(_AUDIT_SPEC.name, audit_model_feature_nulls)
_AUDIT_SPEC.loader.exec_module(audit_model_feature_nulls)


@pytest.fixture(autouse=True)
def _reset_dashboard_access(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")
    monkeypatch.delenv("WEB_DASHBOARD_TOKEN", raising=False)


@lru_cache(maxsize=1)
def _load_full_live_contract_v4_train_split() -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parent.parent
    legacy_path = repo_root / "data" / "raw" / "ufc-master.csv"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Missing required dataset artifact: {legacy_path}")

    legacy_df = load_kaggle_dataset(legacy_path)
    variants = ufc_refresh.build_training_dataset_variants(legacy_df=legacy_df)
    spec = training_spec.full_live_contract_v4_spec()
    raw_features = build_features_module.build_features(variants[spec.dataset_variant])
    materialized = training_spec.materialize_spec_transforms(raw_features, spec)
    train_split, _, _ = prepare_train_test(
        materialized.copy(),
        cutoff_date=spec.train_cutoff_date,
        min_fights=2,
        feature_cols=list(spec.feature_cols),
    )
    return train_split


def _require_full_live_contract_v4_train_split() -> pd.DataFrame:
    try:
        return _load_full_live_contract_v4_train_split()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _add_bet(ledger: BetLedger, fighter: str, token_id: str, market_id: str) -> None:
    ledger.add_bet(
        fighter=fighter,
        opponent="Opponent",
        side="a",
        amount=10.0,
        price=0.5,
        shares=20.0,
        token_id=token_id,
        market_id=market_id,
        model_prob=0.6,
        market_prob=0.5,
        edge=0.1,
        decimal_odds=2.0,
        dry_run=True,
    )


def _seed_overlapping_trader_ledgers(tmp_path, monkeypatch) -> int:
    single = tmp_path / "single.json"
    conviction = tmp_path / "conviction.json"
    monkeypatch.setattr(duo_trader, "SINGLE_LEDGER", single)
    monkeypatch.setattr(duo_trader, "CONVICTION_LEDGER", conviction)

    _add_bet(BetLedger(path=single), "A", "token-a", "market-a")
    _add_bet(BetLedger(path=conviction), "C", "token-c", "market-c")
    _add_bet(BetLedger(path=single), "E", "token-e", "market-e")

    merged = load_all_trader_ledgers()
    return next(bet["id"] for bet in merged.bets if bet["fighter"] == "C")


def test_bet_ledger_reloads_latest_state_before_each_write(tmp_path):
    path = tmp_path / "ledger.json"
    first = BetLedger(path=path)
    second = BetLedger(path=path)

    _add_bet(first, "Fighter A", "token-1", "market-1")
    _add_bet(second, "Fighter B", "token-2", "market-2")

    final = BetLedger(path=path)
    assert [bet["fighter"] for bet in final.bets] == ["Fighter A", "Fighter B"]
    assert [bet["id"] for bet in final.bets] == [1, 2]


def test_fresh_bet_snapshots_are_detached_from_live_ledger_state(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BetLedger(path=path)
    _add_bet(ledger, "Fighter A", "token-1", "market-1")

    fresh_bets = ledger.get_bets(fresh=True)
    fresh_open_bets = ledger.get_open_bets(fresh=True)

    fresh_bets[0]["fighter"] = "Mutated Fighter"
    fresh_open_bets[0]["status"] = "cancelled"

    assert ledger.bets[0]["fighter"] == "Fighter A"
    assert ledger.bets[0]["status"] == "open"
    assert ledger.get_bets(fresh=True)[0]["fighter"] == "Fighter A"
    assert ledger.get_open_bets(fresh=True)[0]["status"] == "open"


def test_cli_settle_uses_displayed_merged_ids(tmp_path, monkeypatch):
    display_id = _seed_overlapping_trader_ledgers(tmp_path, monkeypatch)

    bot_module.cmd_settle(SimpleNamespace(auto=False, bet_id=display_id, result="win"))

    single = BetLedger(path=duo_trader.SINGLE_LEDGER)
    conviction = BetLedger(path=duo_trader.CONVICTION_LEDGER)
    assert single.bets[1]["status"] == "open"
    assert conviction.bets[0]["status"] == "won"


def test_web_manual_settle_uses_displayed_merged_ids(tmp_path, monkeypatch):
    display_id = _seed_overlapping_trader_ledgers(tmp_path, monkeypatch)
    client = web_app.app.test_client()

    response = client.post(f"/api/settle/{display_id}/loss")

    assert response.status_code == 200
    single = BetLedger(path=duo_trader.SINGLE_LEDGER)
    conviction = BetLedger(path=duo_trader.CONVICTION_LEDGER)
    assert single.bets[1]["status"] == "open"
    assert conviction.bets[0]["status"] == "lost"


def test_settle_bet_rejects_stale_resolution_after_external_cancel(tmp_path, monkeypatch):
    display_id = _seed_overlapping_trader_ledgers(tmp_path, monkeypatch)
    target = tracker_module.resolve_merged_bet_reference(display_id, require_open=True)

    cancelled = BetLedger(path=target["ledger_path"]).cancel_bet(
        target["original_id"],
        reason="external_cancel",
    )
    result = BetLedger(path=target["ledger_path"]).settle_bet(target["original_id"], True)

    assert cancelled.ok is True
    assert result.status == "not_open"
    conviction = BetLedger(path=duo_trader.CONVICTION_LEDGER)
    assert conviction.bets[0]["status"] == "cancelled"


def test_ledger_lock_file_does_not_grow_across_multiple_mutations(tmp_path):
    path = tmp_path / "ledger.json"
    lock_path = tracker_module._lock_path_for_ledger(path)
    lock_path.write_bytes(b"000000")

    ledger = BetLedger(path=path)
    _add_bet(ledger, "Alpha", "token-a", "market-a")
    ledger.cancel_bet(1, reason="test_cancel")
    _add_bet(ledger, "Beta", "token-b", "market-b")

    assert lock_path.stat().st_size == 1


def test_scrape_fighter_profile_converts_height_and_reach_to_cm(monkeypatch):
    html = """
    <html><body>
      <h2 class="b-content__title"><span>Brandon Vera</span></h2>
      <span class="b-content__title-record">Record: 17-9-0</span>
      <li class="b-list__box-list-item">Height: 6' 3"</li>
      <li class="b-list__box-list-item">Weight: 205 lbs.</li>
      <li class="b-list__box-list-item">Reach: 76"</li>
      <li class="b-list__box-list-item">STANCE: Orthodox</li>
      <li class="b-list__box-list-item">DOB: Oct 15, 1977</li>
      <li class="b-list__box-list-item_type_block">SLpM: 3.2</li>
      <li class="b-list__box-list-item_type_block">Str. Acc.: 50%</li>
      <li class="b-list__box-list-item_type_block">SApM: 2.1</li>
      <li class="b-list__box-list-item_type_block">Str. Def: 55%</li>
      <li class="b-list__box-list-item_type_block">TD Avg.: 1.0</li>
      <li class="b-list__box-list-item_type_block">TD Acc.: 40%</li>
      <li class="b-list__box-list-item_type_block">TD Def.: 60%</li>
      <li class="b-list__box-list-item_type_block">Sub. Avg.: 0.3</li>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    monkeypatch.setattr(fighter_lookup, "_get_soup", lambda _: soup)

    profile = fighter_lookup.scrape_fighter_profile("http://example.test/fighter")

    assert profile["height"] == pytest.approx(190.5)
    assert profile["reach"] == pytest.approx(193.04)


def test_kaggle_loader_converts_string_height_and_reach_inputs_to_cm(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "r_fighter": "Alpha",
                "b_fighter": "Beta",
                "winner": "Alpha",
                "r_height": "5'10\"",
                "b_height": "72",
                "r_reach": "72\"",
                "b_reach": "74.5",
            }
        ]
    )
    path = tmp_path / "kaggle.csv"
    raw.to_csv(path, index=False)

    loaded = load_kaggle_dataset(path)
    row = loaded.iloc[0]

    assert row["a_height"] == pytest.approx(177.8)
    assert row["b_height"] == pytest.approx(182.88)
    assert row["a_reach"] == pytest.approx(182.88)
    assert row["b_reach"] == pytest.approx(189.23)


def test_kaggle_loader_parses_metric_and_feet_reach_formats(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "r_fighter": "Alpha",
                "b_fighter": "Beta",
                "winner": "Alpha",
                "r_reach": "6'1\" (1.85m)",
                "b_reach": "1.91m",
            }
        ]
    )
    path = tmp_path / "kaggle_reach_formats.csv"
    raw.to_csv(path, index=False)

    loaded = load_kaggle_dataset(path)
    row = loaded.iloc[0]

    assert row["a_reach"] == pytest.approx(185.0)
    assert row["b_reach"] == pytest.approx(191.0)


def test_kaggle_loader_preserves_boolean_title_bout_values(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "r_fighter": "Alpha",
                "b_fighter": "Beta",
                "winner": "Alpha",
                "TitleBout": True,
                "EmptyArena": False,
            }
        ]
    )
    path = tmp_path / "kaggle.csv"
    raw.to_csv(path, index=False)

    loaded = load_kaggle_dataset(path)
    row = loaded.iloc[0]

    assert row["title_bout"] == pytest.approx(1.0)
    assert row["empty_arena"] == pytest.approx(0.0)


def test_build_features_uses_prior_fight_counts_and_live_lookup_returns_completed_fights(tmp_path, monkeypatch):
    # Mock away pre-UFC supplement to avoid collision with real fighter named "Alpha"
    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: Path("/nonexistent/supplement.csv"),
    )
    fights = pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": pd.Timestamp("2020-01-01"),
                "winner": "Alpha",
                "weight_class": "Lightweight",
            },
            {
                "fighter_a": "Gamma",
                "fighter_b": "Alpha",
                "event_date": pd.Timestamp("2020-02-01"),
                "winner": "Alpha",
                "weight_class": "Lightweight",
            },
            {
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "event_date": pd.Timestamp("2020-03-01"),
                "winner": "Delta",
                "weight_class": "Lightweight",
            },
        ]
    )

    features = build_features_module.build_features(fights)
    alpha_rows = features.loc[
        (features["fighter_a"] == "Alpha") | (features["fighter_b"] == "Alpha")
    ].sort_values("event_date")

    alpha_prior = alpha_rows.apply(
        lambda row: row["a_num_fights"] if row["fighter_a"] == "Alpha" else row["b_num_fights"],
        axis=1,
    )
    assert list(alpha_prior) == [0, 1, 2]

    monkeypatch.setattr(build_features_module, "PROCESSED_DATA_DIR", tmp_path)
    build_features_module.save_features(features)

    assert build_features_module.get_fighter_ufc_fight_count("Alpha") == 3


def test_backfill_resume_retries_only_missing_fights(tmp_path, monkeypatch):
    output_dir = tmp_path / "historical"
    output_dir.mkdir()
    monkeypatch.setattr(historical_backfill, "BACKFILL_DIR", output_dir)
    monkeypatch.setattr(historical_backfill, "REQUEST_DELAY", 0)

    fights = pd.DataFrame(
        [
            {"fighter_a": "Alpha", "fighter_b": "Beta", "event_date": pd.Timestamp("2024-01-01")},
            {"fighter_a": "Gamma", "fighter_b": "Delta", "event_date": pd.Timestamp("2024-01-01")},
        ]
    )

    existing = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "query_date": "2023-12-25",
                "offset_days": 7,
                "a_fair_prob": 0.6,
                "b_fair_prob": 0.4,
                "a_decimal_odds": 1.67,
                "b_decimal_odds": 2.5,
                "num_bookmakers": 1,
            }
        ]
    )
    existing.to_csv(output_dir / "historical_odds.csv", index=False)

    class _FakeOddsClient:
        def __init__(self):
            self.calls = []

        def get_historical_odds(self, query_iso):
            self.calls.append(query_iso)
            return [
                {
                    "home_team": "Alpha",
                    "away_team": "Beta",
                    "bookmakers": [
                        {
                            "title": "Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Alpha", "price": 1.67},
                                        {"name": "Beta", "price": 2.5},
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "home_team": "Gamma",
                    "away_team": "Delta",
                    "bookmakers": [
                        {
                            "title": "Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Gamma", "price": 1.8},
                                        {"name": "Delta", "price": 2.2},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ]

    fake_client = _FakeOddsClient()
    monkeypatch.setattr(historical_backfill, "OddsClient", lambda: fake_client)

    result = historical_backfill.run_backfill(fights_df=fights, offsets=[7], resume=True)

    assert fake_client.calls == ["2023-12-25T12:00:00Z"]
    assert len(result) == 2
    keys = {
        (row.event_date, row.fighter_a, row.fighter_b, int(row.offset_days))
        for row in result.itertuples(index=False)
    }
    assert keys == {
        ("2024-01-01", "Alpha", "Beta", 7),
        ("2024-01-01", "Gamma", "Delta", 7),
    }


def test_backfill_resume_key_normalization_handles_case_spacing_diacritics(tmp_path, monkeypatch):
    output_dir = tmp_path / "historical"
    output_dir.mkdir()
    monkeypatch.setattr(historical_backfill, "BACKFILL_DIR", output_dir)
    monkeypatch.setattr(historical_backfill, "REQUEST_DELAY", 0)

    fights = pd.DataFrame(
        [
            {"fighter_a": "Jose Aldo", "fighter_b": "Renato Moicano", "event_date": pd.Timestamp("2024-01-01")},
            {"fighter_a": "Charles Johnson", "fighter_b": "Bruno Silva", "event_date": pd.Timestamp("2024-01-01")},
        ]
    )

    existing = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "  José   Áldo ",
                "fighter_b": "Renato   Moicano",
                "query_date": "2023-12-25",
                "offset_days": 7,
                "a_fair_prob": 0.6,
                "b_fair_prob": 0.4,
                "a_decimal_odds": 1.67,
                "b_decimal_odds": 2.5,
                "num_bookmakers": 1,
            },
            {
                "event_date": "2024-01-01",
                "fighter_a": "bruno silva",
                "fighter_b": "charles johnson",
                "query_date": "2023-12-25",
                "offset_days": 7,
                "a_fair_prob": 0.55,
                "b_fair_prob": 0.45,
                "a_decimal_odds": 1.82,
                "b_decimal_odds": 2.22,
                "num_bookmakers": 1,
            },
        ]
    )
    existing.to_csv(output_dir / "historical_odds.csv", index=False)

    class _FakeOddsClient:
        def __init__(self):
            self.calls = []

        def get_historical_odds(self, query_iso):
            self.calls.append(query_iso)
            return [
                {
                    "home_team": "Charles Johnson",
                    "away_team": "Bruno Silva",
                    "bookmakers": [
                        {
                            "title": "Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Charles Johnson", "price": 1.8},
                                        {"name": "Bruno Silva", "price": 2.2},
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "home_team": "José Aldo",
                    "away_team": "Renato Moicano",
                    "bookmakers": [
                        {
                            "title": "Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "José Aldo", "price": 1.67},
                                        {"name": "Renato Moicano", "price": 2.5},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ]

    fake_client = _FakeOddsClient()
    monkeypatch.setattr(historical_backfill, "OddsClient", lambda: fake_client)

    result = historical_backfill.run_backfill(fights_df=fights, offsets=[7], resume=True)

    assert fake_client.calls == ["2023-12-25T12:00:00Z"]
    assert len(result) == 3
    normalized_keys = {
        (
            historical_backfill._normalize_backfill_name(row.fighter_a),
            historical_backfill._normalize_backfill_name(row.fighter_b),
            int(row.offset_days),
        )
        for row in result.itertuples(index=False)
    }
    assert ("jose aldo", "renato moicano", 7) in normalized_keys
    assert ("charles johnson", "bruno silva", 7) in normalized_keys
    assert ("bruno silva", "charles johnson", 7) in normalized_keys


def test_audit_reports_split_level_dead_columns_even_when_trainable_pool_is_not_dead(monkeypatch):
    features = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2021-12-01"),
                "a_num_fights": 3,
                "b_num_fights": 3,
                "target": 1,
                "always_on": 1.0,
                "cutoff_dead": pd.NA,
            },
            {
                "event_date": pd.Timestamp("2022-02-01"),
                "a_num_fights": 3,
                "b_num_fights": 3,
                "target": 0,
                "always_on": 1.0,
                "cutoff_dead": 5.0,
            },
        ]
    )
    spec = training_spec.NamedModelTrainingSpec(
        name="split_dead_probe",
        feature_cols=["always_on", "cutoff_dead"],
        train_cutoff_date="2022-01-01",
    )

    monkeypatch.setattr(audit_model_feature_nulls, "_resolve_spec", lambda _name: spec)
    monkeypatch.setattr(audit_model_feature_nulls, "_load_feature_frame", lambda _variant: features.copy())
    monkeypatch.setattr(
        training_spec,
        "materialize_spec_transforms",
        lambda frame, _spec: frame.copy(),
    )

    result = audit_model_feature_nulls.run_audit(
        spec_name=spec.name,
        dataset_variant=None,
        min_fights=2,
        recent_rows=10,
        null_threshold_pct=20.0,
    )

    assert result.trainable_rows == 2
    assert result.train_split_rows == 1
    assert result.test_split_rows == 1
    assert result.dead_contract_columns_trainable == []
    assert result.dead_contract_columns_train_split == ["cutoff_dead"]
    assert result.high_null_columns_train_split[0]["column"] == "cutoff_dead"


def test_audit_defaults_to_spec_dataset_variant_when_no_override_is_passed(monkeypatch):
    features = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2021-12-01"),
                "a_num_fights": 3,
                "b_num_fights": 3,
                "target": 1,
                "always_on": 1.0,
            }
        ]
    )
    spec = training_spec.NamedModelTrainingSpec(
        name="variant_default_probe",
        feature_cols=["always_on"],
        dataset_variant="best_of_both_full_history",
        train_cutoff_date="2022-01-01",
    )
    captured = {}

    def fake_load_feature_frame(variant):
        captured["dataset_variant"] = variant
        return features.copy()

    monkeypatch.setattr(audit_model_feature_nulls, "_resolve_spec", lambda _name: spec)
    monkeypatch.setattr(
        audit_model_feature_nulls,
        "_load_feature_frame",
        fake_load_feature_frame,
    )
    monkeypatch.setattr(
        training_spec,
        "materialize_spec_transforms",
        lambda frame, _spec: frame.copy(),
    )

    result = audit_model_feature_nulls.run_audit(
        spec_name=spec.name,
        dataset_variant=None,
        min_fights=2,
        recent_rows=10,
        null_threshold_pct=20.0,
    )

    assert captured["dataset_variant"] == "best_of_both_full_history"
    assert result.dataset_variant == "best_of_both_full_history"


def test_backfill_name_normalization_handles_true_unicode_diacritics():
    assert historical_backfill._normalize_backfill_name("Jos\u00e9 \u00c1ldo") == "jose aldo"


def test_build_features_ewm_uses_prior_fight_counts(monkeypatch):
    def fake_build_features(_fights_df):
        fighter_fights = pd.DataFrame(
            {
                "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
                "won": [1, 0, 1],
                **{stat: [0.0, 0.0, 0.0] for stat in model_variants.STAT_COLUMNS},
            }
        )
        return build_features_module._compute_rolling_stats(fighter_fights)

    monkeypatch.setattr(build_features_module, "build_features", fake_build_features)

    result = model_variants.build_features_ewm(pd.DataFrame())

    assert result["num_fights"].tolist() == [0, 1, 2]


def test_full_live_contract_v4_unexpected_train_split_null_columns_do_not_expand():
    train_split = _require_full_live_contract_v4_train_split()
    spec = training_spec.full_live_contract_v4_spec()

    null_columns = {
        column
        for column in spec.feature_cols
        if column in train_split.columns and train_split[column].isna().any()
    }
    expected_honest_null_columns = {
        # Profile gaps (early UFC fighters with missing bios)
        "a_age", "b_age", "diff_age",
        "a_height", "b_height", "diff_height",
        "a_reach", "b_reach", "diff_reach",
        "a_weight", "b_weight", "diff_weight",
        # Debut fighters: all rolling stats are NaN (no fabricated defaults)
        "a_roll_slpm", "b_roll_slpm", "diff_roll_slpm",
        "a_roll_sapm", "b_roll_sapm", "diff_roll_sapm",
        "a_roll_str_acc", "b_roll_str_acc", "diff_roll_str_acc",
        "a_roll_str_def", "b_roll_str_def", "diff_roll_str_def",
        "a_roll_sub_avg", "b_roll_sub_avg", "diff_roll_sub_avg",
        "a_roll_td_acc", "b_roll_td_acc", "diff_roll_td_acc",
        "a_roll_td_avg", "b_roll_td_avg", "diff_roll_td_avg",
        "a_roll_td_def", "b_roll_td_def", "diff_roll_td_def",
        "a_roll_sig_str_landed", "b_roll_sig_str_landed", "diff_roll_sig_str_landed",
        "a_roll_td_landed", "b_roll_td_landed", "diff_roll_td_landed",
        "a_roll_kd", "b_roll_kd", "diff_roll_kd",
        "a_strike_diff", "b_strike_diff", "diff_strike_diff",
        # Debut fighters: experimental features NaN
        "a_fight_pace", "b_fight_pace", "diff_fight_pace",
        "a_ctrl_efficiency", "b_ctrl_efficiency", "diff_ctrl_efficiency",
        # Finish rates: NaN for fighters with 0 wins (no fabricated defaults)
        "a_ko_rate", "b_ko_rate", "diff_ko_rate",
        "a_sub_rate", "b_sub_rate", "diff_sub_rate",
        "a_dec_rate", "b_dec_rate", "diff_dec_rate",
        "a_win_pct", "b_win_pct", "diff_win_pct",
        # Stance: unknown stance is NaN
        "a_stance_enc", "b_stance_enc", "same_stance",
        # Context features
        "num_rounds_feat", "weight_class_enc",
        # Style interaction edges propagate NaN from missing ko/sub/defense stats
        "a_striker_edge", "b_striker_edge", "diff_striker_edge",
        "a_grappler_edge", "b_grappler_edge", "diff_grappler_edge",
    }

    assert null_columns <= expected_honest_null_columns


def test_full_live_contract_v4_remaining_roll_accuracy_and_defense_nulls_are_denominator_driven():
    train_split = _require_full_live_contract_v4_train_split()

    checks = [
        ("a_roll_td_acc", "a_roll_td_attempted"),
        ("b_roll_td_acc", "b_roll_td_attempted"),
        ("a_roll_td_def", "a_roll_opp_td_attempted"),
        ("b_roll_td_def", "b_roll_opp_td_attempted"),
        ("a_roll_str_def", "a_roll_opp_sig_str_attempted"),
        ("b_roll_str_def", "b_roll_opp_sig_str_attempted"),
    ]

    for feature_col, denominator_col in checks:
        null_rows = train_split.loc[train_split[feature_col].isna()]
        denominators = null_rows[denominator_col]
        assert (denominators.fillna(0) == 0).all(), (
            f"{feature_col} has nulls without a zero-or-missing denominator in {denominator_col}"
        )

    diff_checks = [
        ("diff_roll_td_acc", "a_roll_td_acc", "b_roll_td_acc"),
        ("diff_roll_td_def", "a_roll_td_def", "b_roll_td_def"),
        ("diff_roll_str_def", "a_roll_str_def", "b_roll_str_def"),
    ]
    for diff_col, a_col, b_col in diff_checks:
        null_rows = train_split.loc[train_split[diff_col].isna()]
        assert not (null_rows[a_col].notna() & null_rows[b_col].notna()).any(), (
            f"{diff_col} is null even though both source features are populated"
        )


def test_full_live_contract_v4_remaining_profile_and_context_nulls_match_known_honest_exceptions():
    train_split = _require_full_live_contract_v4_train_split()

    num_rounds_rows = train_split.loc[
        train_split["num_rounds_feat"].isna(),
        ["event_name", "fighter_a", "fighter_b"],
    ]
    expected_num_rounds_rows = {
        ("UFC 2: No Way Out", "Patrick Smith", "Johnny Rhodes"),
        ("UFC 2: No Way Out", "Royce Gracie", "Remco Pardoel"),
        ("UFC 2: No Way Out", "Royce Gracie", "Jason DeLucia"),
        ("UFC 4: Revenge of the Warriors", "Royce Gracie", "Keith Hackney"),
    }
    assert set(map(tuple, num_rounds_rows.itertuples(index=False, name=None))) == expected_num_rounds_rows

    # Profile gaps are expected for very early UFC fighters and some missing bios.
    # With honest NaN semantics (no fabricated defaults), more fighters may show gaps.
    # Assert that the gap rate stays below a reasonable threshold rather than
    # hardcoding an exact fighter set which is brittle to data pipeline changes.
    age_null_pct = (train_split["a_age"].isna().sum() + train_split["b_age"].isna().sum()) / (2 * len(train_split)) * 100
    reach_null_pct = (train_split["a_reach"].isna().sum() + train_split["b_reach"].isna().sum()) / (2 * len(train_split)) * 100
    assert age_null_pct < 2.0, f"Age null rate too high: {age_null_pct:.1f}%"
    assert reach_null_pct < 3.5, f"Reach null rate too high: {reach_null_pct:.1f}%"
    # Height and weight should be near-complete (allowing small gaps for early UFC)
    height_null_pct = (train_split["a_height"].isna().sum() + train_split["b_height"].isna().sum()) / (2 * len(train_split)) * 100
    weight_null_pct = (train_split["a_weight"].isna().sum() + train_split["b_weight"].isna().sum()) / (2 * len(train_split)) * 100
    assert height_null_pct < 1.0, f"Height null rate too high: {height_null_pct:.1f}%"
    assert weight_null_pct < 1.0, f"Weight null rate too high: {weight_null_pct:.1f}%"
