"""Unit test for dual trader coordination logic."""

import pandas as pd
from src.strategy.dual_trader import _resolve_conflicts, _split_bankroll


def test_opposite_sides_both_cancelled():
    """If traders bet opposite sides of the same fight, both should be cancelled."""
    bets_a = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.05,
    }])
    bets_b = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Alex Pereira",
        "bet_side": "b",
        "edge": 0.04,
    }])

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    assert len(result_a) == 0, f"Expected 0 bets for A, got {len(result_a)}"
    assert len(result_b) == 0, f"Expected 0 bets for B, got {len(result_b)}"
    print("PASS: Opposite sides -> both cancelled")


def test_same_side_higher_edge_wins():
    """If both want the same side, higher edge trader takes it."""
    bets_a = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.07,
    }])
    bets_b = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.04,
    }])

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    assert len(result_a) == 1, f"Expected 1 bet for A (higher edge), got {len(result_a)}"
    assert len(result_b) == 0, f"Expected 0 bets for B (lower edge), got {len(result_b)}"
    print("PASS: Same side -> higher edge (A) wins")


def test_same_side_b_higher_edge():
    """If B has higher edge on same side, B takes it."""
    bets_a = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.03,
    }])
    bets_b = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.08,
    }])

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    assert len(result_a) == 0, f"Expected 0 bets for A, got {len(result_a)}"
    assert len(result_b) == 1, f"Expected 1 bet for B (higher edge), got {len(result_b)}"
    print("PASS: Same side -> higher edge (B) wins")


def test_no_conflict_different_fights():
    """Different fights should not conflict."""
    bets_a = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.05,
    }])
    bets_b = pd.DataFrame([{
        "fighter_a": "Islam Makhachev",
        "fighter_b": "Ilia Topuria",
        "bet_on": "Islam Makhachev",
        "bet_side": "a",
        "edge": 0.06,
    }])

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    assert len(result_a) == 1, f"Expected 1 bet for A, got {len(result_a)}"
    assert len(result_b) == 1, f"Expected 1 bet for B, got {len(result_b)}"
    print("PASS: Different fights -> no conflict, both proceed")


def test_empty_bets():
    """One or both empty should pass through."""
    bets_a = pd.DataFrame([{
        "fighter_a": "Jon Jones",
        "fighter_b": "Alex Pereira",
        "bet_on": "Jon Jones",
        "bet_side": "a",
        "edge": 0.05,
    }])
    bets_b = pd.DataFrame()

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    assert len(result_a) == 1, f"Expected 1 bet for A, got {len(result_a)}"
    assert len(result_b) == 0, f"Expected 0 bets for B, got {len(result_b)}"
    print("PASS: Empty B -> A passes through")


def test_multiple_fights_mixed():
    """Mix of conflicts and non-conflicts across multiple fights."""
    bets_a = pd.DataFrame([
        {
            "fighter_a": "Jon Jones", "fighter_b": "Alex Pereira",
            "bet_on": "Jon Jones", "bet_side": "a", "edge": 0.05,
        },
        {
            "fighter_a": "Islam Makhachev", "fighter_b": "Ilia Topuria",
            "bet_on": "Islam Makhachev", "bet_side": "a", "edge": 0.04,
        },
        {
            "fighter_a": "Khamzat Chimaev", "fighter_b": "Sean Strickland",
            "bet_on": "Khamzat Chimaev", "bet_side": "a", "edge": 0.06,
        },
    ])
    bets_b = pd.DataFrame([
        {
            # Same fight as A[0], OPPOSITE side -> both cancelled
            "fighter_a": "Jon Jones", "fighter_b": "Alex Pereira",
            "bet_on": "Alex Pereira", "bet_side": "b", "edge": 0.04,
        },
        {
            # Same fight as A[1], SAME side, B has higher edge -> B takes it
            "fighter_a": "Islam Makhachev", "fighter_b": "Ilia Topuria",
            "bet_on": "Islam Makhachev", "bet_side": "a", "edge": 0.07,
        },
    ])

    result_a, result_b = _resolve_conflicts(bets_a, bets_b)
    # A should keep only Chimaev fight (Jones cancelled, Islam goes to B)
    assert len(result_a) == 1, f"Expected 1 bet for A, got {len(result_a)}"
    assert result_a.iloc[0]["bet_on"] == "Khamzat Chimaev", \
        f"Expected A to keep Chimaev, got {result_a.iloc[0]['bet_on']}"
    # B should keep only Islam fight (Jones cancelled, Islam won)
    assert len(result_b) == 1, f"Expected 1 bet for B, got {len(result_b)}"
    assert result_b.iloc[0]["bet_on"] == "Islam Makhachev", \
        f"Expected B to keep Islam, got {result_b.iloc[0]['bet_on']}"
    print("PASS: Multiple fights mixed conflicts resolved correctly")


def test_bankroll_split():
    """Verify bankroll splits 50/50."""
    a, b = _split_bankroll(dry_run=True)
    assert a == b, f"Expected equal split, got A={a}, B={b}"
    assert a > 0, f"Expected positive allocation, got {a}"
    print(f"PASS: Bankroll split 50/50 -> ${a:.2f} each")


if __name__ == "__main__":
    test_opposite_sides_both_cancelled()
    test_same_side_higher_edge_wins()
    test_same_side_b_higher_edge()
    test_no_conflict_different_fights()
    test_empty_bets()
    test_multiple_fights_mixed()
    test_bankroll_split()
    print("\nAll tests passed!")
