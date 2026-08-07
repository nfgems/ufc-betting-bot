import sys

import pandas as pd

from scripts import recover_bfo_moneyline_gaps as recover_bfo


def test_main_uses_explicit_fights_snapshot(tmp_path, monkeypatch):
    fights_path = tmp_path / "pre_recovery" / "fights_cleaned.csv"
    fights_path.parent.mkdir()
    fights_path.write_text("event_date,fighter_a,fighter_b\n", encoding="utf-8")
    recovered_path = tmp_path / "historical_odds_bfo_recovered_run.csv"
    unresolved_path = tmp_path / "historical_odds_bfo_unresolved_run.csv"
    provenance_path = tmp_path / "historical_odds_bfo_recovered_run.provenance.jsonl"
    observed = {}

    def fake_load_true_missing_queue(start_date, end_date, *, fights_path):
        observed.update(
            start_date=start_date,
            end_date=end_date,
            fights_path=fights_path,
        )
        return pd.DataFrame()

    monkeypatch.setattr(
        recover_bfo,
        "load_true_missing_queue",
        fake_load_true_missing_queue,
    )
    monkeypatch.setattr(
        recover_bfo,
        "recover_queue",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_bfo_moneyline_gaps.py",
            "--fights-path",
            str(fights_path),
            "--recovered-output",
            str(recovered_path),
            "--unresolved-output",
            str(unresolved_path),
            "--provenance-output",
            str(provenance_path),
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-08-01",
        ],
    )

    assert recover_bfo.main() == 0
    assert observed == {
        "start_date": "2026-01-01",
        "end_date": "2026-08-01",
        "fights_path": fights_path,
    }
