import json
import statistics
from pathlib import Path

import pandas as pd
import pytest

from scripts import revalidate_bfo_recovery_file as revalidate


def test_revalidate_recovery_frame_updates_confirmed_rows_and_quarantines_failures(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "event_date": "2026-08-01",
                "fighter_a": "Ludovit Klein",
                "fighter_b": "Tofiq Musayev",
                "a_fair_prob": 0.495967,
                "b_fair_prob": 0.504033,
                "a_decimal_odds": 2.864242,
                "b_decimal_odds": 2.818401,
                "num_bookmakers": 7,
                "source_url": "https://example.test/event",
            },
            {
                "event_date": "2026-08-01",
                "fighter_a": "Missing",
                "fighter_b": "Fight",
                "a_fair_prob": 0.5,
                "b_fair_prob": 0.5,
                "a_decimal_odds": 1.9,
                "b_decimal_odds": 1.9,
                "num_bookmakers": 2,
                "source_url": "https://example.test/event",
            },
        ]
    )

    def fake_find(_url, fighter_a, _fighter_b, _event_date, **_kwargs):
        if fighter_a == "Missing":
            return None
        return "A", "B", {20: -250.0, 21: -265.0, 22: -275.0}, {
            20: 200.0,
            21: 205.0,
            22: 225.0,
        }

    monkeypatch.setattr(revalidate, "find_odds_from_event", fake_find)

    corrected, quarantined = revalidate.revalidate_recovery_frame(frame)

    assert len(corrected) == 1
    assert corrected.iloc[0]["fighter_a"] == "Ludovit Klein"
    assert corrected.iloc[0]["a_fair_prob"] == 0.688898
    assert corrected.iloc[0]["num_bookmakers"] == 3
    assert len(quarantined) == 1
    assert quarantined.iloc[0]["fighter_a"] == "Missing"
    assert quarantined.iloc[0]["revalidation_error"] == (
        "insufficient_or_inconsistent_sportsbook_consensus"
    )


def test_revalidation_can_preserve_legacy_manual_and_thin_observations(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "fighter_a": "Manual",
                "fighter_b": "Source",
                "source_url": "websearch_manual",
                "num_bookmakers": 5,
            },
            {
                "fighter_a": "Thin",
                "fighter_b": "Market",
                "source_url": "https://example.test/event",
                "num_bookmakers": 2,
            },
        ]
    )
    monkeypatch.setattr(revalidate, "find_odds_from_event", lambda *_args: None)

    corrected, quarantined = revalidate.revalidate_recovery_frame(
        frame,
        preserve_legacy_observations=True,
    )

    assert list(corrected["fighter_a"]) == ["Manual", "Thin"]
    assert len(quarantined) == 2


def test_publish_writes_reconciliable_quote_provenance(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    provenance_path = tmp_path / "corrected.provenance.jsonl"
    source = _source_frame(20).assign(
        query_date=lambda frame: frame["event_date"],
        offset_days=0,
        a_fair_prob=0.5,
        b_fair_prob=0.5,
        a_decimal_odds=2.0,
        b_decimal_odds=2.0,
    )
    source.to_csv(source_path, index=False)

    def fake_find(url, fighter_a, fighter_b, event_date, *, observation):
        observation.update(
            {
                "event_page": {
                    "url": url,
                    "start_date": event_date,
                    "start_dates": [event_date],
                    "date_delta_days": 0,
                    "fetched_at_utc": "2026-08-05T12:00:00+00:00",
                    "content_sha256": "a" * 64,
                },
                "matched_bfo_rows": [
                    {
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                        "fighter_a_href": "/fighters/a-1",
                        "fighter_b_href": "/fighters/b-2",
                        "matchup_id": 123,
                        "orientation": "direct",
                        "paired_market_ids": [20, 21, 22],
                    }
                ],
            }
        )
        if fighter_a == "Fighter A19":
            observation["match_error"] = "exact_fighter_pair_not_found"
            return None
        observation["match_error"] = ""
        return fighter_a, fighter_b, {
            20: -250.0,
            21: -265.0,
            22: -275.0,
        }, {
            20: 200.0,
            21: 205.0,
            22: 225.0,
        }

    monkeypatch.setattr(revalidate, "find_odds_from_event", fake_find)

    corrected, quarantined, issues = revalidate.publish_revalidation(
        source_path,
        output_path,
        quarantine_path,
        provenance_path=provenance_path,
    )

    assert issues == []
    assert len(corrected) == 19
    assert len(quarantined) == 1
    records = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(source)
    assert [record["decision"] for record in records].count("accepted") == 19
    assert [record["decision"] for record in records].count("rejected") == 1
    assert {
        (
            record["recovery_key"]["event_date"],
            record["recovery_key"]["fighter_a"],
            record["recovery_key"]["fighter_b"],
        )
        for record in records
    } == {
        (row.event_date, row.fighter_a, row.fighter_b)
        for row in source.itertuples(index=False)
    }

    accepted = next(record for record in records if record["decision"] == "accepted")
    output = pd.read_csv(output_path, dtype={"event_date": str})
    output_row = output[
        (output["event_date"] == accepted["recovery_key"]["event_date"])
        & (output["fighter_a"] == accepted["recovery_key"]["fighter_a"])
    ].iloc[0]
    valid_quotes = [quote for quote in accepted["paired_quotes"] if quote["accepted"]]
    regenerated_fair_a = statistics.median(
        quote["a_fair_prob"] for quote in valid_quotes
    )
    regenerated_a_decimal = statistics.median(
        quote["a_decimal"] for quote in valid_quotes
    )
    assert round(regenerated_fair_a, 6) == output_row["a_fair_prob"]
    assert round(1.0 - regenerated_fair_a, 6) == output_row["b_fair_prob"]
    assert round(regenerated_a_decimal, 6) == output_row["a_decimal_odds"]
    assert accepted["csv_values"]["num_bookmakers"] == output_row["num_bookmakers"]
    assert accepted["event_page"]["content_sha256"] == "a" * 64
    assert accepted["matched_bfo_rows"][0]["matchup_id"] == 123
    assert accepted["parser"]["file_sha256"]
    assert accepted["parser"]["dirty_diff_sha256"]
    assert accepted["input_batch"] == str(source_path.resolve())
    assert accepted["output_batch"] == str(output_path.resolve())


def _source_frame(row_count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": f"2026-08-{index + 1:02d}",
                "fighter_a": f"Fighter A{index}",
                "fighter_b": f"Fighter B{index}",
                "source_url": "https://example.test/event",
                "num_bookmakers": 5,
            }
            for index in range(row_count)
        ]
    )


@pytest.mark.parametrize(
    ("output_selector", "quarantine_selector"),
    [
        ("input", "quarantine"),
        ("output", "output"),
        ("output", "input"),
    ],
)
def test_publish_requires_distinct_resolved_paths(
    tmp_path,
    output_selector,
    quarantine_selector,
):
    paths = {
        "input": tmp_path / "source.csv",
        "output": tmp_path / "corrected.csv",
        "quarantine": tmp_path / "quarantine.csv",
    }
    source = _source_frame()
    source.to_csv(paths["input"], index=False)
    original = paths["input"].read_bytes()

    with pytest.raises(revalidate.RevalidationSafetyError, match="four distinct paths"):
        revalidate.publish_revalidation(
            paths["input"],
            paths[output_selector],
            paths[quarantine_selector],
        )

    assert paths["input"].read_bytes() == original


def test_publish_rejects_provenance_path_collision(tmp_path):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _source_frame().to_csv(source_path, index=False)

    with pytest.raises(revalidate.RevalidationSafetyError, match="four distinct paths"):
        revalidate.publish_revalidation(
            source_path,
            output_path,
            quarantine_path,
            provenance_path=source_path,
        )


def test_systemic_fetch_failure_aborts_without_writing_any_artifact(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _source_frame().to_csv(source_path, index=False)
    output_path.write_text("existing output\n", encoding="utf-8")
    quarantine_path.write_text("existing quarantine\n", encoding="utf-8")
    source_before = source_path.read_bytes()
    output_before = output_path.read_bytes()
    quarantine_before = quarantine_path.read_bytes()
    monkeypatch.setattr(
        revalidate,
        "find_odds_from_event",
        lambda *_args: (_ for _ in ()).throw(OSError("BFO unavailable")),
    )

    with pytest.raises(revalidate.RevalidationSafetyError, match="operational fetch/parser"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert source_path.read_bytes() == source_before
    assert output_path.read_bytes() == output_before
    assert quarantine_path.read_bytes() == quarantine_before


def test_preserve_legacy_override_does_not_hide_all_rows_failed(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(2)
    source["source_url"] = "websearch_manual"
    source.to_csv(source_path, index=False)
    monkeypatch.setattr(revalidate, "find_odds_from_event", lambda *_args: None)

    with pytest.raises(revalidate.RevalidationSafetyError, match="every source row"):
        revalidate.publish_revalidation(
            source_path,
            output_path,
            quarantine_path,
            preserve_legacy_observations=True,
        )

    assert not output_path.exists()
    assert not quarantine_path.exists()


def test_empty_corrected_output_is_rejected_before_writes(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(2)
    source.to_csv(source_path, index=False)
    quarantined = source.assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (source.iloc[0:0].copy(), quarantined),
    )

    with pytest.raises(revalidate.RevalidationSafetyError, match="would be empty"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert not output_path.exists()
    assert not quarantine_path.exists()


def test_material_row_loss_is_rejected_before_writes(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(10)
    source.to_csv(source_path, index=False)
    corrected = source.iloc[:-1].copy()
    quarantined = source.iloc[-1:].assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )

    with pytest.raises(revalidate.RevalidationSafetyError, match="10.0%"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert not output_path.exists()
    assert not quarantine_path.exists()


def test_unsafe_override_is_explicit_and_keeps_source_unchanged(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(2)
    source.to_csv(source_path, index=False)
    source_before = source_path.read_bytes()
    corrected = source.iloc[0:0].copy()
    quarantined = source.assign(revalidation_error="OSError: BFO unavailable")
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )

    _, _, issues = revalidate.publish_revalidation(
        source_path,
        output_path,
        quarantine_path,
        allow_unsafe_output=True,
    )

    assert issues
    assert source_path.read_bytes() == source_before
    assert pd.read_csv(output_path).empty
    assert len(pd.read_csv(quarantine_path)) == 2


def test_quarantine_is_written_and_verified_before_corrected_output(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(20)
    source.to_csv(source_path, index=False)
    corrected = source.iloc[:-1].copy()
    quarantined = source.iloc[-1:].assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )
    real_writer = revalidate.write_csv_atomically
    writes: list[Path] = []

    def recording_writer(frame, path, **kwargs):
        writes.append(Path(path))
        return real_writer(frame, path, **kwargs)

    monkeypatch.setattr(revalidate, "write_csv_atomically", recording_writer)

    revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert writes == [quarantine_path.resolve(), output_path.resolve()]
    assert len(pd.read_csv(quarantine_path)) == 1
    assert len(pd.read_csv(output_path)) == 19


def test_failed_quarantine_verification_leaves_corrected_output_untouched(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(20)
    source.to_csv(source_path, index=False)
    corrected = source.iloc[:-1].copy()
    quarantined = source.iloc[-1:].assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    output_path.write_text("existing output\n", encoding="utf-8")
    output_before = output_path.read_bytes()
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )
    real_writer = revalidate.write_csv_atomically

    def corrupt_quarantine(frame, path, **kwargs):
        result = real_writer(frame, path, **kwargs)
        if Path(path) == quarantine_path.resolve():
            Path(path).write_text("corrupt\n", encoding="utf-8")
        return result

    monkeypatch.setattr(revalidate, "write_csv_atomically", corrupt_quarantine)

    with pytest.raises(revalidate.RevalidationSafetyError, match="failed to verify"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert output_path.read_bytes() == output_before


def test_failed_provenance_write_leaves_corrected_output_untouched(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(20)
    source.to_csv(source_path, index=False)
    corrected = source.iloc[:-1].copy()
    quarantined = source.iloc[-1:].assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    output_path.write_text("existing output\n", encoding="utf-8")
    output_before = output_path.read_bytes()
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )
    monkeypatch.setattr(
        revalidate,
        "write_jsonl_atomically",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(revalidate.RevalidationSafetyError, match="provenance JSONL"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert output_path.read_bytes() == output_before
    assert len(pd.read_csv(quarantine_path)) == 1


def test_corrected_write_failure_preserves_source_and_verified_quarantine(tmp_path, monkeypatch):
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "corrected.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    source = _source_frame(20)
    source.to_csv(source_path, index=False)
    source_before = source_path.read_bytes()
    corrected = source.iloc[:-1].copy()
    quarantined = source.iloc[-1:].assign(revalidation_error=revalidate.CONSENSUS_REJECTION)
    monkeypatch.setattr(
        revalidate,
        "revalidate_recovery_frame",
        lambda *_args, **_kwargs: (corrected, quarantined),
    )
    real_writer = revalidate.write_csv_atomically

    def fail_corrected_write(frame, path, **kwargs):
        if Path(path) == output_path.resolve():
            raise OSError("disk full")
        return real_writer(frame, path, **kwargs)

    monkeypatch.setattr(revalidate, "write_csv_atomically", fail_corrected_write)

    with pytest.raises(OSError, match="disk full"):
        revalidate.publish_revalidation(source_path, output_path, quarantine_path)

    assert source_path.read_bytes() == source_before
    assert not output_path.exists()
    assert len(pd.read_csv(quarantine_path)) == 1


def test_main_returns_nonzero_and_preserves_source_on_path_collision(tmp_path):
    source_path = tmp_path / "source.csv"
    quarantine_path = tmp_path / "quarantine.csv"
    _source_frame().to_csv(source_path, index=False)
    source_before = source_path.read_bytes()

    result = revalidate.main(
        [
            str(source_path),
            "--output",
            str(source_path),
            "--quarantine-output",
            str(quarantine_path),
        ]
    )

    assert result == 1
    assert source_path.read_bytes() == source_before
    assert not quarantine_path.exists()
