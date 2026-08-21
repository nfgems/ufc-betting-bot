from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_pre_ufc_career_supplement as builder
from src.data import fallback_scrapers, fighter_lookup


def _candidate(
    *,
    fighter_id: str = "1111111111111111",
    fighter_name: str = "Alpha Fighter",
    first_ufc_date: str = "2025-01-01",
    aliases: list[str] | None = None,
    source_fingerprint: str = "queue-v1",
) -> dict[str, object]:
    return builder._finalize_candidate(
        {
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "first_ufc_date": first_ufc_date,
            "aliases": aliases or [fighter_name],
            "ufc_fights": 1,
            "source_input_fingerprint": source_fingerprint,
        }
    )


def _checkpoint() -> dict[str, object]:
    return {
        "schema_version": 2,
        "processed": {},
        "failed": [],
        "identity_state": {},
    }


def _isolate_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder, "OUTPUT_PATH", tmp_path / "pre-ufc.csv")
    monkeypatch.setattr(builder, "CHECKPOINT_PATH", tmp_path / "pre-ufc-checkpoint.json")
    monkeypatch.setattr(builder, "AMATEUR_OUTPUT_PATH", tmp_path / "amateur.csv")
    monkeypatch.setattr(builder, "AMATEUR_CHECKPOINT_PATH", tmp_path / "amateur-checkpoint.json")
    monkeypatch.setattr(builder, "OFFICIAL_ROSTER_PATH", tmp_path / "missing-roster.csv")
    monkeypatch.setattr(builder.time, "sleep", lambda _seconds: None)


def test_default_queue_uses_canonical_processed_identity_chronology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chronology = tmp_path / "processed" / "fights_cleaned.csv"
    chronology.parent.mkdir()
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "fighter_a_id": "1111111111111111",
                "fighter_b_id": "2222222222222222",
                "event_date": "2025-03-08",
            },
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Gamma Fighter",
                "fighter_a_id": "1111111111111111",
                "fighter_b_id": "3333333333333333",
                "event_date": "2025-01-04",
            },
        ]
    ).to_csv(chronology, index=False)
    monkeypatch.setattr(builder, "DEFAULT_FIGHTS_PATH", chronology)

    candidates = builder.identify_ufc_fighters(builder._find_best_features_csv())

    assert candidates["ufcstats:1111111111111111"]["first_ufc_date"] == "2025-01-04"
    assert candidates["ufcstats:1111111111111111"]["fighter_name"] == "Alpha Fighter"


def test_explicit_csv_and_json_queues_require_stable_id_and_strict_boundary(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "queue.csv"
    pd.DataFrame(
        [
            {
                "fighter_id": "ABCDEF0123456789",
                "fighter_name": "Alpha Fighter",
                "first_ufc_date": "2025-03-08T20:00:00Z",
                "aliases": '["Alpha Fighter", "A. Fighter"]',
            }
        ]
    ).to_csv(csv_path, index=False)
    json_path = tmp_path / "queue.json"
    json_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "fighter_id": "2222222222222222",
                        "fighter_name": "Beta Fighter",
                        "first_ufc_date": "2025-04-12",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    csv_candidate = builder.load_candidate_file(csv_path)["ufcstats:abcdef0123456789"]
    json_candidate = builder.load_candidate_file(json_path)["ufcstats:2222222222222222"]

    assert csv_candidate["first_ufc_date"] == "2025-03-08"
    assert csv_candidate["aliases"] == ["Alpha Fighter", "A. Fighter"]
    assert json_candidate["first_ufc_date"] == "2025-04-12"

    csv_path.write_text(
        "fighter_id,fighter_name,first_ufc_date\nnot-an-id,Alpha Fighter,not-a-date\n",
        encoding="utf-8",
    )
    with pytest.raises(builder.CandidateFileError, match="16 hexadecimal"):
        builder.load_candidate_file(csv_path)


@pytest.mark.parametrize(
    ("attempt_kwargs", "expected_status"),
    [
        ({"row_count": 0}, "exhausted_zero_rows"),
        ({"row_count": None, "error": "temporary"}, "exhausted_transient_failure"),
    ],
)
def test_retry_budget_exhausts_after_three_attempts(
    attempt_kwargs: dict[str, object],
    expected_status: str,
) -> None:
    candidate = _candidate()
    key = builder._candidate_key(candidate)
    checkpoint = _checkpoint()

    attempts_and_statuses = []
    for _ in range(builder.MAX_RETRY_ATTEMPTS):
        state = builder._record_identity_attempt(checkpoint, key, candidate, **attempt_kwargs)
        attempts_and_statuses.append((state["attempts"], state["status"]))

    assert [attempts for attempts, _status in attempts_and_statuses] == [1, 2, 3]
    assert attempts_and_statuses[-1][1] == expected_status
    assert builder._state_should_scrape(state) is False


def test_alias_change_preserves_retry_budget_but_boundary_change_resets_it() -> None:
    original = _candidate()
    key = builder._candidate_key(original)
    checkpoint = _checkpoint()
    for _ in range(builder.MAX_RETRY_ATTEMPTS):
        builder._record_identity_attempt(checkpoint, key, original, row_count=0)

    alias_changed = _candidate(
        fighter_name="Alpha Alias",
        aliases=["Alpha Alias", "Alpha Fighter"],
        source_fingerprint="queue-v2",
    )
    state, reset = builder._prepare_identity_state(
        checkpoint,
        key,
        alias_changed,
        allow_legacy_migration=False,
    )
    assert reset is False
    assert (state["attempts"], state["status"]) == (3, "exhausted_zero_rows")

    boundary_changed = _candidate(first_ufc_date="2024-12-15")
    state, reset = builder._prepare_identity_state(
        checkpoint,
        key,
        boundary_changed,
        allow_legacy_migration=False,
    )
    assert reset is True
    assert (state["attempts"], state["status"]) == (0, "pending")


def test_main_persists_outputs_before_checkpoints_and_reports_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    candidate = _candidate()
    monkeypatch.setattr(
        builder,
        "identify_ufc_fighters",
        lambda _path, max_ufc_fights=None: {builder._candidate_key(candidate): candidate},
    )
    monkeypatch.setattr(builder, "scrape_fighter_pre_ufc_fights", lambda *_args: [])
    monkeypatch.setattr(builder, "scrape_fighter_amateur_fights", lambda *_args: [])
    events: list[tuple[str, str]] = []
    real_json_writer = builder.write_json_atomically

    def _record_csv(frame: pd.DataFrame, path: Path) -> Path:
        events.append(("csv", Path(path).name))
        return Path(path)

    def _record_json(payload: object, path: Path, **kwargs: object) -> Path:
        events.append(("json", Path(path).name))
        return real_json_writer(payload, path, **kwargs)

    monkeypatch.setattr(builder, "write_csv_atomically", _record_csv)
    monkeypatch.setattr(builder, "write_json_atomically", _record_json)
    summary_path = tmp_path / "summary.json"

    assert builder.main(["--summary-json", str(summary_path)]) == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["warning_codes"] == ["pre_ufc_identity_retry_incomplete"]
    # Empty observations do not rewrite supplements, but their checkpoints are
    # still atomically persisted before the final orchestration summary.
    assert events == [
        ("json", "pre-ufc-checkpoint.json"),
        ("json", "amateur-checkpoint.json"),
        ("json", "summary.json"),
    ]


def test_successful_rows_are_written_before_atomic_checkpoint_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    candidate = _candidate()
    monkeypatch.setattr(
        builder,
        "identify_ufc_fighters",
        lambda _path, max_ufc_fights=None: {builder._candidate_key(candidate): candidate},
    )
    row = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Regional Opponent",
        "event_date": "2024-01-01",
        "organization": "Regional FC",
    }
    monkeypatch.setattr(builder, "scrape_fighter_pre_ufc_fights", lambda *_args: [row])
    monkeypatch.setattr(builder, "scrape_fighter_amateur_fights", lambda *_args: [row])
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        builder,
        "write_csv_atomically",
        lambda _frame, path: events.append(("csv", Path(path).name)) or Path(path),
    )
    monkeypatch.setattr(
        builder,
        "write_json_atomically",
        lambda _payload, path, **_kwargs: events.append(("json", Path(path).name)) or Path(path),
    )

    assert builder.main([]) == 0
    assert events == [
        ("csv", "pre-ufc.csv"),
        ("json", "pre-ufc-checkpoint.json"),
        ("csv", "amateur.csv"),
        ("json", "amateur-checkpoint.json"),
    ]


def test_clear_cache_invalidates_pre_ufc_and_amateur_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fighter_lookup, "_pre_ufc_long_rows_cache", pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr(fighter_lookup, "_amateur_summary_cache", {"Alpha": {"fights": 1}})
    monkeypatch.setattr(fallback_scrapers, "clear_fallback_cache", lambda **_kwargs: None)

    fighter_lookup.clear_cache()

    assert fighter_lookup._pre_ufc_long_rows_cache is None
    assert fighter_lookup._amateur_summary_cache is None
