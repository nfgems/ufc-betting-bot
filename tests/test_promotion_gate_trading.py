"""Tests for the trading bar checks in the promotion gate."""

import pandas as pd
import pytest
from src.strategy.promotion_gate import PromotionGate

@pytest.fixture
def gate():
    return PromotionGate()

def test_volume_sanity_fails_trading_bar(gate):
    """Verify that excessive bet volume (>150% of baseline) fails the trading bar."""
    baseline_log = pd.DataFrame([{"bet_size": 10.0}] * 100) # 100 bets
    candidate_log = pd.DataFrame([{"bet_size": 10.0, "trader": "S"}] * 151) # 151 bets (>150%)
    
    baseline_sweep = {
        "roi": 0.05,
        "total_profit": 50.0,
        "max_drawdown": 0.10,
        "bet_log": baseline_log
    }
    candidate_sweep = {
        "roi": 0.06,
        "total_profit": 60.0,
        "max_drawdown": 0.08,
        "bet_log": candidate_log
    }
    
    result = gate.check_trading_bar(candidate_sweep, baseline_sweep)
    assert result["passes"] is False
    assert any("EXCESS_BETS" in r for r in result["reasons"])

def test_trader_c_audit_fails_trading_bar(gate):
    """Verify that over-betting marginal underdogs in Trader C fails the trading bar."""
    # S-bets are clean
    s_bets = [{"trader": "S", "bet_size": 10.0, "won": 1, "market_prob": 0.5, "model_prob": 0.6, "decimal_odds": 2.0}] * 50
    # C-bets have marginal underdogs (odds > 3.0, model_prob < 0.70)
    c_bets = [{"trader": "C", "bet_size": 10.0, "won": 0, "market_prob": 0.2, "model_prob": 0.35, "decimal_odds": 4.5}] * 10
    
    candidate_log = pd.DataFrame(s_bets + c_bets)
    baseline_log = pd.DataFrame(s_bets) # Baseline is just S-bets
    
    baseline_sweep = {
        "roi": 0.05,
        "total_profit": 50.0,
        "max_drawdown": 0.10,
        "bet_log": baseline_log
    }
    candidate_sweep = {
        "roi": 0.06,
        "total_profit": 60.0,
        "max_drawdown": 0.08,
        "bet_log": candidate_log,
        "avg_clv": 0.02 # CLV must be present if baseline has it, or just to avoid issues
    }
    
    result = gate.check_trading_bar(candidate_sweep, baseline_sweep)
    assert result["passes"] is False
    assert any("MARGINAL_UNDERDOGS" in r for r in result["reasons"])

def test_trading_bar_passes_clean_sweep(gate):
    """Verify that a clean sweep with better ROI and acceptable drawdown passes."""
    s_bets = [{"trader": "S", "bet_size": 10.0, "won": 1, "market_prob": 0.5, "model_prob": 0.6, "decimal_odds": 2.0}] * 50
    
    log = pd.DataFrame(s_bets)
    
    baseline_sweep = {
        "roi": 0.05,
        "total_profit": 50.0,
        "max_drawdown": 0.10,
        "bet_log": log,
        "avg_clv": 0.01
    }
    candidate_sweep = {
        "roi": 0.07, # Better ROI
        "total_profit": 70.0, # Better profit
        "max_drawdown": 0.09, # Acceptable drawdown
        "bet_log": log,
        "avg_clv": 0.02 # Better CLV
    }
    
    result = gate.check_trading_bar(candidate_sweep, baseline_sweep)
    assert result["passes"] is True
    assert not any("FAIL" in r for r in result["reasons"])


def test_trading_bar_uses_single_trader_blend_edge_when_available(gate):
    candidate_log = pd.DataFrame(
        [
            {
                "trader": "S",
                "bet_size": 10.0,
                "profit": 1.0,
                "won": 1,
                "edge": 0.01,
                "blend_edge": 0.01,
                "raw_edge": 0.03,
                "market_prob": 0.50,
                "model_prob": 0.53,
                "decimal_odds": 2.0,
            }
            for _ in range(10)
        ]
        + [
            {
                "trader": "C",
                "bet_size": 10.0,
                "profit": 0.0,
                "won": 0,
                "edge": 0.50,
                "blend_edge": pd.NA,
                "raw_edge": 0.50,
                "market_prob": 0.50,
                "model_prob": 1.0,
                "decimal_odds": 2.0,
            }
        ]
    )
    baseline_log = pd.DataFrame([{"bet_size": 10.0, "profit": 0.0}] * 20)

    baseline_sweep = {
        "roi": 0.05,
        "total_profit": 50.0,
        "max_drawdown": 0.10,
        "bet_log": baseline_log,
        "avg_clv": 0.01,
    }
    candidate_sweep = {
        "roi": 0.06,
        "total_profit": 60.0,
        "max_drawdown": 0.08,
        "bet_log": candidate_log,
        "avg_clv": 0.02,
    }

    result = gate.check_trading_bar(candidate_sweep, baseline_sweep)
    assert result["passes"] is False
    assert any("possible luck or data leakage" in r for r in result["reasons"])
