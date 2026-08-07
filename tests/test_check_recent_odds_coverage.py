import sys

import pandas as pd

from scripts import check_recent_odds_coverage as coverage


def _write_features(path):
    pd.DataFrame(
        {
            "event_date": ["2026-08-01"],
            "target": [1.0],
            "a_num_fights": [3],
            "b_num_fights": [4],
            "a_implied_prob": [0.6],
            "b_implied_prob": [0.4],
            "diff_implied_prob": [0.2],
        }
    ).to_csv(path, index=False)


def test_fixed_complete_odds_floor_fails_closed(tmp_path, monkeypatch, capsys):
    features = tmp_path / "features.csv"
    _write_features(features)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_recent_odds_coverage.py",
            "--features-path",
            str(features),
            "--skip-non-regression",
            "--minimum-complete-odds-rows",
            "2",
        ],
    )

    assert coverage.main() == 1
    assert "1 < 2" in capsys.readouterr().err


def test_fixed_complete_odds_floor_accepts_exact_minimum(tmp_path, monkeypatch):
    features = tmp_path / "features.csv"
    _write_features(features)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_recent_odds_coverage.py",
            "--features-path",
            str(features),
            "--skip-non-regression",
            "--minimum-complete-odds-rows",
            "1",
        ],
    )

    assert coverage.main() == 0
