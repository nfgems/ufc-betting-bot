import numpy as np
import pandas as pd

import src.strategy.backtest as backtest
import src.strategy.lab_stats as lab_stats
import src.strategy.model_lab as model_lab
import src.strategy.model_variants as model_variants
import src.strategy.run_evaluation as run_evaluation


def _make_model_lab_features_frame() -> pd.DataFrame:
    training_dates = pd.date_range("2018-01-01", "2023-12-20", periods=100)
    rows = []
    for idx, event_date in enumerate(training_dates):
        rows.append({
            "event_date": event_date,
            "fighter_a": f"Train A {idx}",
            "fighter_b": f"Train B {idx}",
            "target": idx % 2,
            "dummy_feature": float(idx),
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "seed_prob_a": 0.55,
            "seed_prob_b": 0.45,
        })

    rows.extend([
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Skip A",
            "fighter_b": "Skip B",
            "target": 1,
            "dummy_feature": 100.0,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "seed_prob_a": 0.70,
            "seed_prob_b": 0.30,
        },
        {
            "event_date": pd.Timestamp("2024-01-02"),
            "fighter_a": "Bet A",
            "fighter_b": "Bet B",
            "target": 0,
            "dummy_feature": 101.0,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "seed_prob_a": 0.70,
            "seed_prob_b": 0.30,
        },
        {
            "event_date": pd.Timestamp("2024-01-03"),
            "fighter_a": "No Odds A",
            "fighter_b": "No Odds B",
            "target": 1,
            "dummy_feature": 102.0,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": np.nan,
            "b_market_prob": np.nan,
            "seed_prob_a": 0.70,
            "seed_prob_b": 0.30,
        },
        {
            "event_date": pd.Timestamp("2024-01-04"),
            "fighter_a": "No Bet A",
            "fighter_b": "No Bet B",
            "target": 1,
            "dummy_feature": 103.0,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "seed_prob_a": 0.50,
            "seed_prob_b": 0.50,
        },
        {
            "event_date": pd.Timestamp("2024-01-05"),
            "fighter_a": "No Bet 2 A",
            "fighter_b": "No Bet 2 B",
            "target": 0,
            "dummy_feature": 104.0,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "seed_prob_a": 0.50,
            "seed_prob_b": 0.50,
        },
    ])
    for window_start in ("2024-07-01", "2025-01-01"):
        for offset in range(5):
            event_date = pd.Timestamp(window_start) + pd.Timedelta(days=offset)
            rows.append({
                "event_date": event_date,
                "fighter_a": f"Reserved A {event_date.date()}",
                "fighter_b": f"Reserved B {event_date.date()}",
                "target": offset % 2,
                "dummy_feature": 200.0 + len(rows),
                "a_num_fights": 5,
                "b_num_fights": 5,
                "a_market_prob": 0.50,
                "b_market_prob": 0.50,
                "seed_prob_a": 0.50,
                "seed_prob_b": 0.50,
            })
    return pd.DataFrame(rows)


def test_backtest_bankroll_history_counts_only_settled_bets(monkeypatch):
    monkeypatch.setattr(backtest, "_passes_filters", lambda *args, **kwargs: True)

    predictions = pd.DataFrame([
        {
            "event_date": pd.Timestamp("2023-12-20"),
            "fighter_a": "Skip A",
            "fighter_b": "Skip B",
            "target": 1,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_no_odds_prob_a": 0.70,
            "xgboost_no_odds_prob_b": 0.30,
        },
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Bet A",
            "fighter_b": "Bet B",
            "target": 0,
            "a_market_prob": 0.50,
            "b_market_prob": 0.50,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_no_odds_prob_a": 0.70,
            "xgboost_no_odds_prob_b": 0.30,
        },
        {
            "event_date": pd.Timestamp("2024-01-02"),
            "fighter_a": "No Odds A",
            "fighter_b": "No Odds B",
            "target": 1,
            "a_market_prob": np.nan,
            "b_market_prob": np.nan,
            "a_num_fights": 5,
            "b_num_fights": 5,
            "odds_source": "test",
            "xgboost_no_odds_prob_a": 0.70,
            "xgboost_no_odds_prob_b": 0.30,
        },
    ])

    result = backtest._simulate_backtest_predictions(
        predictions,
        backtest.NO_ODDS_SIGNAL_ONLY_STRATEGY,
        initial_bankroll=100.0,
        min_edge=0.01,
        bet_start_date="2024-01-01",
    )

    assert result["stats"]["total_bets"] == 1
    assert len(result["bankroll_history"]) == 2
    assert result["stats"]["max_drawdown_duration"] == 1


def test_model_lab_drawdown_duration_counts_only_settled_bets(monkeypatch):
    monkeypatch.setattr(
        run_evaluation,
        "_preflight_selection_fold_manifest",
        lambda *_args, **_kwargs: "0" * 64,
    )
    monkeypatch.setattr(model_lab, "_passes_filters", lambda *args, **kwargs: True)
    bankroll_manager_cls = model_lab.BankrollManager
    monkeypatch.setattr(
        model_lab,
        "BankrollManager",
        lambda initial_bankroll, kelly_fraction, max_bet_fraction: bankroll_manager_cls(
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            auto_detect_balance=False,
        ),
    )
    monkeypatch.setattr(model_lab, "_variant_feature_columns", lambda *_args, **_kwargs: ["dummy_feature"])
    monkeypatch.setattr(model_lab, "_merge_historical_odds", lambda predictions: predictions)
    monkeypatch.setattr(
        model_lab,
        "_resolve_market_odds",
        lambda predictions, *_args, **_kwargs: (predictions, "test"),
    )
    monkeypatch.setattr(
        model_lab,
        "train_variant_model",
        lambda *_args, **_kwargs: {"model": object(), "feature_cols": ["dummy_feature"], "col_medians": np.array([0.0])},
    )
    monkeypatch.setattr(
        model_lab,
        "_predict_batch_with_model",
        lambda features_df, _model_result: features_df.assign(
            prob_a=features_df["seed_prob_a"],
            prob_b=features_df["seed_prob_b"],
        ),
    )

    variant = model_variants.ALL_VARIANTS["baseline"]()
    features_df = _make_model_lab_features_frame()
    result = model_lab.run_variant_walkforward(
        features_df,
        variant,
        initial_bankroll=100.0,
        bet_start_date="2024-01-02",
    )

    dd = lab_stats.compute_max_drawdown(result["bankroll_history"])

    assert result["stats"]["total_bets"] == 1
    assert len(result["bankroll_history"]) == 2
    assert dd["max_drawdown_duration"] == 1
