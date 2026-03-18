"""Tests for the finalist sweep framework."""

import numpy as np
import pandas as pd
import pytest

import src.strategy.finalist_sweep as finalist_sweep_mod
from src.strategy.finalist_sweep import (
    compare_finalist_sweeps,
    narrow_sweep_around,
    run_finalist_sweep,
    _narrow_grid_around,
    _row_to_sweep_config,
)
from src.strategy.duo_trader_sweep import SweepConfig
from src.strategy.selection_gate import SweepTargetSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PREDICTIONS = pd.DataFrame(
    [
        {
            "event_date": pd.Timestamp("2024-01-01"),
            "fighter_a": "Fighter A",
            "fighter_b": "Fighter B",
            "target": 1,
            "prob_a": 0.76,
            "prob_b": 0.24,
            "a_market_prob": 0.60,
            "b_market_prob": 0.40,
            "no_odds_prob_a": 0.71,
            "no_odds_prob_b": 0.29,
            "a_num_fights": 6,
            "b_num_fights": 6,
            "closing_prob_a": 0.64,
            "closing_prob_b": 0.36,
        }
    ]
)


def _mock_predictions(**kwargs):
    """Mock walk-forward predictions returning a single fold."""
    return [(1, _PREDICTIONS.copy())]


def _tiny_sweep_configs(**kwargs):
    """Return a minimal sweep grid for fast tests."""
    variant_name = kwargs.get("variant_name", "baseline")
    return [
        SweepConfig(
            name=f"{variant_name}_test_config",
            variant_name=variant_name,
            s_min_edge=0.02,
            s_blend_weight=0.30,
            c_min_model_prob=0.65,
            c_min_no_odds_prob=0.50,
            c_share=1.0,
        )
    ]


# ---------------------------------------------------------------------------
# _narrow_grid_around
# ---------------------------------------------------------------------------


def test_narrow_grid_around_symmetric():
    grid = _narrow_grid_around(0.025, 0.003, n_points=5)
    assert len(grid) == 5
    assert 0.025 in grid
    # Should be symmetric around centre
    assert grid[0] < grid[-1]


def test_narrow_grid_around_clips_negative():
    grid = _narrow_grid_around(0.003, 0.005, n_points=5)
    # Should not contain values <= 0
    assert all(v > 0 for v in grid)


# ---------------------------------------------------------------------------
# narrow_sweep_around
# ---------------------------------------------------------------------------


def test_narrow_sweep_around_produces_configs():
    top_configs = [
        SweepConfig(
            name="top1",
            variant_name="baseline",
            s_min_edge=0.025,
            s_blend_weight=0.30,
            s_model_agreement_min_edge=0.01,
            c_min_model_prob=0.70,
            c_min_no_odds_prob=0.55,
            c_share=0.50,
            c_min_edge=0.0,
            c_max_decimal_odds=2.5,
        )
    ]
    narrow = narrow_sweep_around(top_configs, variant_name="baseline")
    assert len(narrow) > 0
    assert all(c.name.startswith("narrow_") for c in narrow)
    assert all(c.variant_name == "baseline" for c in narrow)


def test_narrow_sweep_around_empty_input():
    assert narrow_sweep_around([], variant_name="baseline") == []


def test_narrow_sweep_around_respects_max_configs():
    top_configs = [
        SweepConfig(name="a", variant_name="baseline", s_min_edge=0.02),
        SweepConfig(name="b", variant_name="baseline", s_min_edge=0.03),
    ]
    narrow = narrow_sweep_around(top_configs, variant_name="baseline", max_configs=5)
    assert len(narrow) <= 5


# ---------------------------------------------------------------------------
# _row_to_sweep_config
# ---------------------------------------------------------------------------


def test_row_to_sweep_config_roundtrip():
    row = {
        "config": "test_cfg",
        "s_min_edge": 0.025,
        "s_blend_weight": 0.35,
        "s_model_agreement_min_edge": 0.02,
        "c_min_model_prob": 0.70,
        "c_min_no_odds_prob": 0.55,
        "c_share": 0.50,
        "c_min_edge": 0.01,
        "c_max_decimal_odds": 2.5,
    }
    config = _row_to_sweep_config(row, "baseline")
    assert config.name == "test_cfg"
    assert config.s_min_edge == 0.025
    assert config.c_share == 0.50
    assert config.variant_name == "baseline"


# ---------------------------------------------------------------------------
# run_finalist_sweep (mocked)
# ---------------------------------------------------------------------------


def test_run_finalist_sweep_single_variant(monkeypatch):
    # Patch on the finalist_sweep module (where the names are bound)
    monkeypatch.setattr(
        finalist_sweep_mod,
        "_generate_walk_forward_predictions",
        _mock_predictions,
    )
    monkeypatch.setattr(
        finalist_sweep_mod,
        "build_sweep_configs",
        _tiny_sweep_configs,
    )

    results = run_finalist_sweep(
        ["baseline"],
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
        run_narrow=False,
    )

    # Key is name_dataset_feature
    result_key = "baseline_append_only_2026_production"
    assert result_key in results
    data = results[result_key]
    assert "production_row" in data
    assert "broad_summary" in data
    assert "best_config" in data
    assert data["narrow_summary"] is None
    assert data["production_row"]["is_baseline"] is True


def test_run_finalist_sweep_with_narrow(monkeypatch):
    monkeypatch.setattr(
        finalist_sweep_mod,
        "_generate_walk_forward_predictions",
        _mock_predictions,
    )
    monkeypatch.setattr(
        finalist_sweep_mod,
        "build_sweep_configs",
        _tiny_sweep_configs,
    )

    results = run_finalist_sweep(
        ["baseline"],
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
        run_narrow=True,
    )

    result_key = "baseline_append_only_2026_production"
    data = results[result_key]
    # Narrow sweep should have run (may or may not produce configs depending on grid)
    assert "narrow_summary" in data
    if data["narrow_summary"]:
        assert data["primary_summary"] == data["narrow_summary"]


def test_run_finalist_sweep_passes_spec_retrain_months(monkeypatch):
    captured: dict[str, object] = {}

    def fake_predictions(**kwargs):
        captured.update(kwargs)
        return _mock_predictions()

    monkeypatch.setattr(
        finalist_sweep_mod,
        "_generate_walk_forward_predictions",
        fake_predictions,
    )
    monkeypatch.setattr(
        finalist_sweep_mod,
        "build_sweep_configs",
        _tiny_sweep_configs,
    )

    spec = SweepTargetSpec(
        candidate_id="combined_best_append_only_2026_production",
        model_variant="combined_best",
        dataset_variant="append_only_2026",
        feature_family="production",
        calibration_method="sigmoid",
        retrain_months=8,
    )

    run_finalist_sweep(
        [spec],
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
        run_narrow=False,
    )

    assert captured["calibration_method"] == "sigmoid"
    assert captured["retrain_months"] == 8


# ---------------------------------------------------------------------------
# compare_finalist_sweeps
# ---------------------------------------------------------------------------


def test_compare_finalist_sweeps_produces_dataframe(monkeypatch):
    monkeypatch.setattr(
        finalist_sweep_mod,
        "_generate_walk_forward_predictions",
        _mock_predictions,
    )
    monkeypatch.setattr(
        finalist_sweep_mod,
        "build_sweep_configs",
        _tiny_sweep_configs,
    )

    sweep_results = run_finalist_sweep(
        ["baseline"],
        initial_bankroll=100.0,
        bet_start_date="2022-01-01",
        run_narrow=False,
    )

    df = compare_finalist_sweeps(sweep_results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "variant" in df.columns
    assert "roi" in df.columns
    assert "s_edge_profit_gap" in df.columns
    assert "s_edge_profit_flag" in df.columns
    assert "volume_adjusted_flag" in df.columns
    assert "profit_per_bet" in df.columns


def test_compare_finalist_sweeps_empty_preserves_schema():
    df = compare_finalist_sweeps({})

    assert df.empty
    assert "variant" in df.columns
    assert "best_config" in df.columns
    assert "s_edge_profit_gap" in df.columns
    assert "s_edge_profit_flag" in df.columns
    assert "volume_adjusted_flag" in df.columns


def test_compare_flags_volume_winner():
    """Test that a candidate winning only by volume is flagged."""
    sweep_results = {
        "high_volume": {
            "best_config": {
                "config": "hv_cfg",
                "profit": 20.0,
                "roi": 0.05,
                "avg_clv": 0.01,
                "max_dd": 0.10,
                "total_bets": 200,
                "s_bets": 150,
                "s_roi": 0.04,
                "s_avg_clv": 0.01,
                "c_bets": 50,
                "c_roi": 0.06,
                "c_avg_clv": 0.02,
            },
            "production_row": {
                "profit": 15.0,
                "roi": 0.08,
                "total_bets": 100,
            },
        },
    }

    df = compare_finalist_sweeps(sweep_results)
    # profit_per_bet: 20/200=0.10 vs prod 15/100=0.15 -- worse per-bet, 2x more bets
    assert bool(df.iloc[0]["volume_adjusted_flag"]) is True


def test_narrow_only_primary_summary():
    """Verify that compare_finalist_sweeps trusts explicit best_config."""
    sweep_results = {
        "candidate": {
            "primary_summary": [
                {
                    "config": "narrow_winner",
                    "profit": 100.0,
                    "roi": 0.10,
                    "total_bets": 50,
                }
            ],
            "best_config": {
                "config": "broad_winner",
                "profit": 90.0,
                "roi": 0.09,
                "total_bets": 45,
            },
            "production_row": {
                "profit": 10.0,
                "roi": 0.01,
                "total_bets": 10,
            }
        }
    }
    
    df = compare_finalist_sweeps(sweep_results)
    assert df.iloc[0]["best_config"] == "broad_winner"
    assert df.iloc[0]["roi"] == 0.09
