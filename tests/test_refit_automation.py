"""Tests for the unattended-refit plumbing: atomic-write retry, built_at
stamping, and BFO recovered-batch overwrite protection."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.recover_bfo_moneyline_gaps as recover_bfo
import scripts.reconcile_production_bundle_manifest as reconcile_script
from src.data import io_utils


def test_write_csv_atomically_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    target.write_text("old\n", encoding="utf-8")

    real_replace = io_utils.os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(5, "Access is denied", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", flaky_replace)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)

    io_utils.write_csv_atomically(pd.DataFrame({"a": [1]}), target)

    assert attempts["count"] == 3
    assert "a" in target.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_write_csv_atomically_reraises_after_exhausted_retries(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"

    def always_denied(src, dst):
        raise PermissionError(5, "Access is denied", str(dst))

    monkeypatch.setattr(io_utils.os, "replace", always_denied)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        io_utils.write_csv_atomically(pd.DataFrame({"a": [1]}), target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_stamp_manifest_built_at_now_updates_only_timestamps(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "model_spec_name": "prod_spec",
                "built_at": "2026-06-11T06:05:30Z",
                "manifest_updated_at": "2026-06-11T06:05:30Z",
                "selection_basis": "evidence",
            }
        ),
        encoding="utf-8",
    )

    stamp = reconcile_script.stamp_manifest_built_at_now(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["built_at"] == stamp
    assert payload["manifest_updated_at"] == stamp
    assert payload["built_at"] != "2026-06-11T06:05:30Z"
    assert payload["bundle_id"] == "bundle-1"
    assert payload["selection_basis"] == "evidence"
    assert not manifest_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_recover_bfo_refuses_to_overwrite_existing_batch(tmp_path, monkeypatch):
    recovered = tmp_path / "historical_odds_bfo_recovered_existing.csv"
    recovered.write_text("event_date,fighter_a,fighter_b\n2026-06-27,A,B\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_bfo_moneyline_gaps.py",
            "--recovered-output",
            str(recovered),
            "--unresolved-output",
            str(tmp_path / "unresolved.csv"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        recover_bfo.main()

    assert excinfo.value.code == 2
    assert "2026-06-27,A,B" in recovered.read_text(encoding="utf-8")


def test_recover_queue_enforces_event_date_and_emits_provenance(monkeypatch):
    queue = pd.DataFrame(
        [
            {
                "event_date": "2026-08-01",
                "event_name": "UFC Test",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
            }
        ]
    )
    monkeypatch.setattr(
        recover_bfo,
        "find_event_via_fighter",
        lambda *_args: {
            "event_url": "https://example.test/event",
            "fighter_page_url": "https://example.test/fighter-a",
            "matched_row": "Fighter B | Aug 1st 2026",
        },
    )

    def fake_find(url, fighter_a, fighter_b, event_date, *, observation):
        assert event_date == "2026-08-01"
        observation.update(
            {
                "event_page": {
                    "url": url,
                    "start_date": event_date,
                    "fetched_at_utc": "2026-08-05T12:00:00+00:00",
                    "content_sha256": "b" * 64,
                },
                "matched_bfo_rows": [
                    {
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                        "fighter_a_href": "/fighters/a-1",
                        "fighter_b_href": "/fighters/b-2",
                        "matchup_id": 456,
                        "orientation": "direct",
                        "paired_market_ids": [20, 21, 22],
                    }
                ],
            }
        )
        return fighter_a, fighter_b, {
            20: -250.0,
            21: -265.0,
            22: -275.0,
        }, {
            20: 200.0,
            21: 205.0,
            22: 225.0,
        }

    monkeypatch.setattr(recover_bfo, "find_odds_from_event", fake_find)
    records = []

    recovered, unresolved = recover_bfo.recover_queue(
        queue,
        provenance_records=records,
        output_batch="temp/recovered.csv",
    )

    assert len(recovered) == 1
    assert unresolved.empty
    assert len(records) == 1
    assert records[0]["decision"] == "accepted"
    assert records[0]["event_page"]["start_date"] == "2026-08-01"
    assert records[0]["event_page"]["content_sha256"] == "b" * 64
    assert len(records[0]["paired_quotes"]) == 3
    assert records[0]["csv_values"]["a_fair_prob"] == recovered.iloc[0]["a_fair_prob"]


def test_bfo_event_parser_uses_anchor_names_and_reviewed_sportsbook_ids(monkeypatch):
    html = """
    <html><body>
      <table class="odds-table">
        <tr><th>Fighter</th></tr>
        <tr><th>
          <a class="bfo-admin-link" href="/cnadm/matchups/44222">44222</a>
          <a href="/fighters/ludovit-klein-7610"><span class="t-b-fcc">Ludovit Klein</span></a>
        </th></tr>
        <tr><th>
          <a class="bfo-admin-link" href="/cnadm/matchups/44222">44222</a>
          <a href="/fighters/tofiq-musayev-13682"><span class="t-b-fcc">Tofiq Musayev</span></a>
        </th></tr>
        <tr><td>footer</td></tr>
      </table>
      <table class="odds-table">
        <tr>
          <th data-b="28">Polymarket</th>
          <th data-b="29"><img alt="Kalshi"></th>
          <th data-b="21">FanDuel</th>
          <th data-b="24">Caesars</th>
          <th data-b="25">BetRivers</th>
          <th data-b="22">Unreviewed Exchange</th>
          <th data-b="99">Future Exchange</th>
        </tr>
        <tr>
          <td class="but-sg" data-li="[28,1,44222]">-233</td>
          <td class="but-sg" data-li="[29,1,44222]">+1071</td>
          <td class="but-sg" data-li="[21,1,44222]">-255</td>
          <td class="but-sg" data-li="[24,1,44222]">-275</td>
          <td class="but-sg" data-li="[25,1,44222]">-265</td>
          <td class="but-sg" data-li="[22,1,44222]">+900</td>
          <td class="but-sg" data-li="[99,1,44222]">+1100</td>
        </tr>
        <tr>
          <td class="but-sg" data-li="[28,2,44222]">+223</td>
          <td class="but-sg" data-li="[29,2,44222]">-2049</td>
          <td class="but-sg" data-li="[21,2,44222]">+210</td>
          <td class="but-sg" data-li="[24,2,44222]">+225</td>
          <td class="but-sg" data-li="[25,2,44222]">+205</td>
          <td class="but-sg" data-li="[22,2,44222]">-1400</td>
          <td class="but-sg" data-li="[99,2,44222]">-1600</td>
        </tr>
        <tr><td>footer</td></tr>
      </table>
    </body></html>
    """
    soup = recover_bfo.BeautifulSoup(html, "lxml")
    monkeypatch.setattr(recover_bfo, "get_soup", lambda _url: soup)

    fights = recover_bfo.parse_event_page("https://example.test/event")

    assert fights == [
        (
            "Ludovit Klein",
            "Tofiq Musayev",
            {21: -255.0, 24: -275.0, 25: -265.0},
            {21: 210.0, 24: 225.0, 25: 205.0},
        )
    ]


def test_bfo_fighter_name_row_rejects_missing_or_ambiguous_fighter_anchors():
    missing_anchor = recover_bfo.BeautifulSoup("<tr><td>Ludovit Klein</td></tr>", "lxml").select_one("tr")
    ambiguous_anchors = recover_bfo.BeautifulSoup(
        """
        <tr><td>
          <a href="/fighters/ludovit-klein-7610">Ludovit Klein</a>
          <a href="/fighters/tofiq-musayev-13682">Tofiq Musayev</a>
        </td></tr>
        """,
        "lxml",
    ).select_one("tr")

    assert recover_bfo._fighter_identity_from_row(missing_anchor) is None
    assert recover_bfo._fighter_identity_from_row(ambiguous_anchors) is None


def test_bfo_consensus_pairs_and_devigs_each_sportsbook_before_aggregation():
    odds_a = {
        21: -255.0,
        24: -275.0,
        25: -265.0,
        20: -250.0,
        26: -265.0,
        28: 100.0,
        29: 100.0,
        99: 100.0,
    }
    odds_b = {
        21: 210.0,
        24: 225.0,
        25: 205.0,
        20: 200.0,
        26: 205.0,
        28: -100.0,
        29: -100.0,
        99: -100.0,
    }

    consensus = recover_bfo.sportsbook_consensus(odds_a, odds_b)

    assert consensus is not None
    assert consensus["num_bookmakers"] == 5
    assert consensus["a_fair_prob"] == pytest.approx(0.688898, abs=1e-6)
    assert consensus["b_fair_prob"] == pytest.approx(0.311102, abs=1e-6)
    assert consensus["a_decimal_odds"] == pytest.approx(1 + 100 / 265)
    assert consensus["b_decimal_odds"] == pytest.approx(3.05)


def test_bfo_consensus_fails_closed_on_too_few_or_widely_disagreeing_books():
    assert recover_bfo.sportsbook_consensus(
        {20: -200.0, 21: -210.0},
        {20: 170.0, 21: 175.0},
    ) is None

    assert recover_bfo.sportsbook_consensus(
        {20: -250.0, 21: -255.0, 22: -10000.0},
        {20: 205.0, 21: 210.0, 22: 5000.0},
    ) is None

    assert recover_bfo.sportsbook_consensus(
        {30: -200.0, 31: -210.0, 32: -220.0},
        {30: 170.0, 31: 175.0, 32: 180.0},
    ) is None


def test_bfo_find_odds_requires_the_complete_fighter_pair(monkeypatch):
    wrong_pair = [
        ("Jordan Smith", "Taylor Brown", {21: -110.0}, {21: -110.0}),
    ]
    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: wrong_pair)

    assert recover_bfo.find_odds_from_event(
        "https://example.test/wrong-event",
        "Chris Smith",
        "Pat Brown",
    ) is None


def test_bfo_find_odds_orients_one_exact_pair_and_rejects_duplicate_matches(monkeypatch):
    reversed_match = (
        "Tofiq Musayev",
        "Ludovit Klein",
        {21: 210.0},
        {21: -255.0},
    )
    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: [reversed_match])

    assert recover_bfo.find_odds_from_event(
        "https://example.test/event",
        "Ludovit Klein",
        "Tofiq Musayev",
    ) == ("Ludovit Klein", "Tofiq Musayev", {21: -255.0}, {21: 210.0})

    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: [reversed_match, reversed_match])
    assert recover_bfo.find_odds_from_event(
        "https://example.test/ambiguous-event",
        "Ludovit Klein",
        "Tofiq Musayev",
    ) is None


def test_bfo_reviewed_aliases_cover_live_name_drift():
    assert recover_bfo.alias_norms("Blood Diamond") >= {
        recover_bfo.norm("Mike Mathetha"),
        recover_bfo.norm("Mike Diamond"),
    }
    assert recover_bfo.norm("Rodrigo Vargas") in recover_bfo.alias_norms("Kazula Vargas")
    assert recover_bfo.norm("Sumudaerji Sumudaerji") in recover_bfo.alias_norms("Sumudaerji")


def test_bfo_alias_split_rows_merge_only_across_disjoint_paired_books(monkeypatch):
    split_rows = [
        (
            "Charlie Radtke",
            "Mike Diamond",
            {20: -300.0, 21: -320.0, 25: -335.0, 26: -335.0},
            {20: 240.0, 21: 260.0, 25: 255.0, 26: 255.0},
        ),
        (
            "Charlie Radtke",
            "Mike Mathetha",
            {23: -350.0, 24: -330.0},
            {23: 260.0, 24: 260.0},
        ),
        (
            "Blood Diamond",
            "Charlie Radtke",
            {22: 260.0},
            {22: -325.0},
        ),
    ]
    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: split_rows)

    result = recover_bfo.find_odds_from_event(
        "https://example.test/ufc-293",
        "Blood Diamond",
        "Charles Radtke",
    )

    assert result is not None
    assert set(result[2]) == {20, 21, 22, 23, 24, 25, 26}
    consensus = recover_bfo.sportsbook_consensus(result[2], result[3])
    assert consensus is not None
    assert consensus["num_bookmakers"] == 7
    assert consensus["a_fair_prob"] == pytest.approx(0.267175572519084)
    assert consensus["a_decimal_odds"] == pytest.approx(3.6)
    assert consensus["b_decimal_odds"] == pytest.approx(1.303030303030303)

    overlapping_rows = [
        split_rows[0],
        (
            "Charlie Radtke",
            "Mike Mathetha",
            {21: -330.0, 23: -350.0},
            {21: 250.0, 23: 260.0},
        ),
    ]
    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: overlapping_rows)
    assert recover_bfo.find_odds_from_event(
        "https://example.test/overlapping-alias-rows",
        "Blood Diamond",
        "Charles Radtke",
    ) is None


def test_bfo_event_jsonld_date_allows_one_day_skew_and_fails_closed(monkeypatch):
    matching_rows = [
        ("Fighter A", "Fighter B", {21: -110.0}, {21: -110.0}),
    ]
    monkeypatch.setattr(recover_bfo, "parse_event_page", lambda _url: matching_rows)

    def soup_with_dates(*dates):
        scripts = "".join(
            '<script type="application/ld+json">'
            f'{{"@type":"SportsEvent","startDate":"{value}"}}'
            "</script>"
            for value in dates
        )
        return recover_bfo.BeautifulSoup(f"<html><body>{scripts}</body></html>", "lxml")

    monkeypatch.setattr(
        recover_bfo,
        "get_soup",
        lambda _url: soup_with_dates("2026-08-02T00:00:00-04:00"),
    )
    for requested_date in ("2026-08-01", "2026-08-02", "2026-08-03"):
        assert recover_bfo.find_odds_from_event(
            "https://example.test/dated-event",
            "Fighter A",
            "Fighter B",
            requested_date,
        ) is not None
    assert recover_bfo.find_odds_from_event(
        "https://example.test/wrong-date-event",
        "Fighter A",
        "Fighter B",
        "2026-07-31",
    ) is None

    monkeypatch.setattr(recover_bfo, "get_soup", lambda _url: soup_with_dates())
    assert recover_bfo.find_odds_from_event(
        "https://example.test/missing-date-event",
        "Fighter A",
        "Fighter B",
        "2026-08-02",
    ) is None

    monkeypatch.setattr(
        recover_bfo,
        "get_soup",
        lambda _url: soup_with_dates("2026-08-02", "2026-08-09"),
    )
    assert recover_bfo.find_odds_from_event(
        "https://example.test/conflicting-date-event",
        "Fighter A",
        "Fighter B",
        "2026-08-02",
    ) is None

    monkeypatch.setattr(
        recover_bfo,
        "get_soup",
        lambda _url: soup_with_dates("2026-08-02", "not-a-date"),
    )
    assert recover_bfo.find_odds_from_event(
        "https://example.test/malformed-date-event",
        "Fighter A",
        "Fighter B",
        "2026-08-02",
    ) is None


def test_bfo_fighter_history_requires_exact_anchored_opponent_and_date(monkeypatch):
    fighter_url = "https://www.bestfightodds.com/fighters/ludovit-klein-7610"
    html = """
    <table class="team-stats-table"><tbody>
      <tr class="event-header"><td><a href="/events/wrong-surname-1">Wrong Surname</a> Aug 1st 2026</td></tr>
      <tr class="main-row">
        <th><a href="/fighters/ludovit-klein-7610">Ludovit Klein</a></th>
        <td><a href="/events/wrong-surname-1">Wrong Surname</a></td>
      </tr>
      <tr><th><a href="/fighters/rafael-musayev-9000">Rafael Musayev</a></th><td>Aug 1st 2026</td></tr>

      <tr class="event-header"><td><a href="/events/wrong-date-2">Wrong Date</a> Aug 8th 2026</td></tr>
      <tr class="main-row">
        <th><a href="/fighters/ludovit-klein-7610">Ludovit Klein</a></th>
        <td><a href="/events/wrong-date-2">Wrong Date</a></td>
      </tr>
      <tr><th><a href="/fighters/tofiq-musayev-13682">Tofiq Musayev</a></th><td>Aug 8th 2026</td></tr>

      <tr class="event-header"><td><a href="/events/correct-3">Correct Event</a> Aug 1st 2026</td></tr>
      <tr class="main-row">
        <th><a href="/fighters/ludovit-klein-7610">Ludovit Klein</a></th>
        <td><a href="/events/correct-3">Correct Event</a></td>
      </tr>
      <tr><th><a href="/fighters/tofiq-musayev-13682">Tofiq Musayev</a></th><td>Aug 1st 2026</td></tr>
    </tbody></table>
    """
    soup = recover_bfo.BeautifulSoup(html, "lxml")
    monkeypatch.setattr(recover_bfo, "candidate_fighter_urls", lambda _name: [("Ludovit Klein", fighter_url)])
    monkeypatch.setattr(recover_bfo, "get_soup", lambda _url: soup)

    match = recover_bfo.find_event_via_fighter("Ludovit Klein", "Tofiq Musayev", "2026-08-01")

    assert match is not None
    assert match["event_url"] == "https://www.bestfightodds.com/events/correct-3"


def test_bfo_fighter_history_rejects_ambiguous_event_matches(monkeypatch):
    fighter_url = "https://www.bestfightodds.com/fighters/ludovit-klein-7610"
    html = """
    <table class="team-stats-table"><tbody>
      <tr class="event-header"><td><a href="/events/first-1">First</a> Aug 1st 2026</td></tr>
      <tr class="main-row"><th><a href="/fighters/ludovit-klein-7610">Ludovit Klein</a></th></tr>
      <tr><th><a href="/fighters/tofiq-musayev-13682">Tofiq Musayev</a></th><td>Aug 1st 2026</td></tr>
      <tr class="event-header"><td><a href="/events/second-2">Second</a> Aug 1st 2026</td></tr>
      <tr class="main-row"><th><a href="/fighters/ludovit-klein-7610">Ludovit Klein</a></th></tr>
      <tr><th><a href="/fighters/tofiq-musayev-13682">Tofiq Musayev</a></th><td>Aug 1st 2026</td></tr>
    </tbody></table>
    """
    soup = recover_bfo.BeautifulSoup(html, "lxml")
    monkeypatch.setattr(recover_bfo, "candidate_fighter_urls", lambda _name: [("Ludovit Klein", fighter_url)])
    monkeypatch.setattr(recover_bfo, "get_soup", lambda _url: soup)

    assert recover_bfo.find_event_via_fighter(
        "Ludovit Klein",
        "Tofiq Musayev",
        "2026-08-01",
    ) is None


WORKFLOW_DIR = Path(__file__).parents[1] / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def test_weekly_schedule_and_dry_run_are_separate_core_wrappers():
    scheduled = _workflow("weekly-model-retrain.yml")
    dry_run = _workflow("weekly-model-refit-dry-run.yml")

    assert "schedule:" in scheduled
    assert "workflow_dispatch:" not in scheduled
    assert "weekly-model-refit-core.yml" in scheduled
    assert "activate_production: true" in scheduled
    assert "force_refit: false" in scheduled

    assert "workflow_dispatch:" in dry_run
    assert "schedule:" not in dry_run
    assert "weekly-model-refit-core.yml" in dry_run
    assert "activate_production: false" in dry_run
    assert "default: false" in dry_run
    assert "secrets:" not in dry_run
    assert "RAILWAY_TOKEN" not in dry_run
    assert "RAILWAY_SSH_PRIVATE_KEY" not in dry_run
    assert "publish_scheduled_refit_bundle.py" not in dry_run


def test_refit_workflows_are_read_only_and_never_commit_generated_artifacts():
    workflows = {
        name: _workflow(name)
        for name in (
            "weekly-model-retrain.yml",
            "weekly-model-refit-core.yml",
            "weekly-model-refit-dry-run.yml",
            "weekly-model-refit-railway-access-check.yml",
        )
    }
    combined = "\n".join(workflows.values())

    for source in workflows.values():
        assert "contents: read" in source
        assert "actions: read" in source
        assert "contents: write" not in source
    for forbidden in (
        "git add",
        "git commit",
        "git push",
        "models/current_production_model.json",
        "candidate_label",
        "PromotionGate",
    ):
        assert forbidden not in combined


def test_core_pins_policy_contract_and_uses_only_isolated_candidate_paths():
    core = _workflow("weekly-model-refit-core.yml")

    assert "workflow_call:" in core
    assert "group: weekly-model-refit" in core
    assert 'data/tmp .codex_stage' in core
    assert core.count("fetch-depth: 0") == 2
    assert core.index("fetch-depth: 0") < core.index("Freeze active production identity")
    assert "POLICY_PATH: config/scheduled_refit_policy_v1.json" in core
    assert "EVALUATION_SPEC: full_live_contract_v6_durability" in core
    assert (
        "FULLFIT_SPEC: "
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    ) in core
    for suffix in ("_pre_recovery", "_audit", "_train"):
        assert suffix in core
    assert 'PRE_RECOVERY_SUBDIR="candidates/${RUN_KEY}_pre_recovery"' in core
    assert 'AUDIT_SUBDIR="candidates/${RUN_KEY}_audit"' in core
    assert 'TRAIN_SUBDIR="candidates/${RUN_KEY}_train"' in core
    assert "--data \"${AUDIT_DIR}/fights_cleaned.csv\"" in core
    assert "--spec \"${FULLFIT_SPEC}\"" in core
    assert "--output-subdir \"${TRAIN_SUBDIR}\"" in core


def test_core_runs_fixed_recovery_coverage_quality_contract_and_parity_gates():
    core = _workflow("weekly-model-refit-core.yml")

    required_fragments = (
        "recover_bfo_moneyline_gaps.py",
        "--fights-path \"${PRE_RECOVERY_DIR}/fights_cleaned.csv\"",
        "check_recent_odds_coverage.py",
        "--skip-non-regression",
        "--minimum-complete-odds-rows 4075",
        "check_production_refit_contract.py preflight",
        "track_c_batch.py",
        "--scheduled-policy \"${POLICY_PATH}\"",
        "check_scheduled_refit_quality.py",
        "parity_replay.py",
        "--mode exact",
        "--mode prefight",
        "--established-only",
        "build_staged_production_bundle.py",
        "check_production_refit_contract.py candidate",
    )
    for fragment in required_fragments:
        assert fragment in core

    assert core.index("check_production_refit_contract.py preflight") < core.index(
        "python -m src.bot train"
    )
    assert core.index("check_scheduled_refit_quality.py") < core.index(
        "python -m src.bot train"
    )
    assert core.index("build_staged_production_bundle.py") < core.index(
        "check_production_refit_contract.py candidate"
    )


def test_core_runs_every_test_file_once_across_four_fail_complete_slices():
    core = _workflow("weekly-model-refit-core.yml")

    assert "fail-fast: false" in core
    assert "shard: [1, 2, 3, 4]" in core
    assert 'glob.glob("tests/test_*.py")' in core
    assert "selected = files[shard::shard_count]" in core
    assert 'python -m pytest -q "${TEST_FILES[@]}"' in core
    assert '"full_test_slices": 4' in core
    assert "UFC_PARITY_PROCESSED_DATA_DIR:" in core
    assert (
        "UFC_PARITY_TRAINING_SPEC: "
        "full_live_contract_v6_durability_corrected_20260805_fullfit"
    ) in core
    assert "needs.candidate.outputs.unverified_artifact" in core


def test_candidate_is_sealed_only_after_tests_and_activation_is_protected():
    core = _workflow("weekly-model-refit-core.yml")

    assert "needs: [candidate, full_test_slices]" in core
    assert "FULL_SUITE_VERIFIED.json" in core
    assert "needs: [candidate, full_test_slices, seal_candidate]" in core
    assert "if: inputs.activate_production" in core
    assert "environment:\n      name: production" in core
    assert "RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}" in core
    assert "RAILWAY_SSH_PRIVATE_KEY: ${{ secrets.RAILWAY_SSH_PRIVATE_KEY }}" in core
    assert "npm install --global @railway/cli@5.30.4" in core
    assert "StrictHostKeyChecking accept-new" in core
    assert "StrictHostKeyChecking=no" not in core
    assert "UserKnownHostsFile=/dev/null" not in core
    assert "--ssh-identity-file \"${RAILWAY_SSH_IDENTITY_FILE}\"" in core
    assert "publish_scheduled_refit_bundle.py" in core
    assert core.count("--activate") == 1
    assert "actions/upload-artifact@v4" in core
    assert "actions/download-artifact@v4" in core


def test_railway_access_check_is_protected_manual_and_read_only():
    access_check = _workflow("weekly-model-refit-railway-access-check.yml")

    assert "workflow_dispatch:" in access_check
    assert "schedule:" not in access_check
    assert "environment:\n      name: production" in access_check
    assert "RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}" in access_check
    assert "RAILWAY_SSH_PRIVATE_KEY: ${{ secrets.RAILWAY_SSH_PRIVATE_KEY }}" in access_check
    assert "refs/heads/main" in access_check
    assert "npm install --global @railway/cli@5.30.4" in access_check
    assert "StrictHostKeyChecking accept-new" in access_check
    assert "StrictHostKeyChecking=no" not in access_check
    assert "UserKnownHostsFile=/dev/null" not in access_check
    assert "check_scheduled_refit_railway_access.py" in access_check
    assert "--ssh-identity-file \"${RAILWAY_SSH_IDENTITY_FILE}\"" in access_check
    for forbidden in (
        "--activate",
        " volume files upload ",
        " promote ",
        " rollback ",
        " railway restart ",
        " volume files delete ",
        " volume files rename ",
    ):
        assert forbidden not in access_check


def test_scheduled_builder_uses_public_parent_identity_not_stale_repo_manifest():
    core = _workflow("weekly-model-refit-core.yml")
    builder_start = core.index("Build and validate immutable v3 bundle")
    builder_end = core.index("Candidate fixed-policy contract gate")
    builder_call = core[builder_start:builder_end]

    assert "--scheduled-refit-policy \"${POLICY_PATH}\"" in builder_call
    assert "--previous-readyz \"${PARENT_READYZ}\"" in builder_call
    assert "--previous-manifest" not in builder_call
    assert "--previous-deployed-git-sha" not in builder_call
    assert "--previous-runtime-lookup-hash" not in builder_call
    assert "--previous-bfo-lineage-manifest" in builder_call
    assert "--expected-fights-sha256 \"${AUDIT_FIGHTS_SHA256}\"" in builder_call
    assert "--expected-features-sha256 \"${AUDIT_FEATURES_SHA256}\"" in builder_call


def test_bfo_lineage_is_restored_before_refresh_and_bound_to_exact_parent():
    core = _workflow("weekly-model-refit-core.yml")

    resolve = core.index("Resolve active BFO lineage identity")
    lookup = core.index("Locate exact parent BFO lineage artifact")
    download = core.index("Download exact parent BFO lineage artifact")
    restore = core.index("Restore authenticated parent BFO lineage")
    refresh_artifact = core.index("Refresh validated parent BFO lineage artifact")
    refresh_raw = core.index("Refresh raw inputs into an isolated pre-recovery snapshot")
    recover = core.index("Recover BFO gaps against the isolated snapshot")
    inventory = core.index("Freeze pretraining input inventory")

    assert resolve < lookup < download < restore < refresh_artifact < refresh_raw
    assert restore < recover < inventory
    assert "scheduled_bfo_lineage_manifest_sha256" in core
    assert 'bundle.get("bundle_id") == root.get("bundle_id")' in core
    assert 'bundle.get("rich_release_id") == root.get("release_id")' in core
    assert "non-root active production is missing its BFO lineage identity" in core
    assert "weekly-refit-bfo-lineage-{source_sha}-{lineage_sha}" in core
    assert '"/repos/${GITHUB_REPOSITORY}/actions/artifacts"' in core
    assert '-f "name=${BFO_LINEAGE_ARTIFACT_NAME}"' in core
    assert "github-token: ${{ github.token }}" in core
    assert "run-id: ${{ steps.bfo_lineage_lookup.outputs.run_id }}" in core
    assert "python scripts/bfo_lineage.py" in core
    assert "overwrite: true" in core[refresh_artifact:refresh_raw]
    assert "retention-days: 90" in core[refresh_artifact:refresh_raw]


def test_candidate_bfo_lineage_is_uploaded_by_seal_before_activation():
    core = _workflow("weekly-model-refit-core.yml")

    seal = core.index("seal_candidate:")
    bind = core.index("Bind candidate BFO lineage artifact name", seal)
    upload_lineage = core.index("Upload candidate BFO lineage before activation", bind)
    upload_verified = core.index("needs.candidate.outputs.verified_artifact", upload_lineage)
    activation = core.index("activate_production:", upload_verified)

    assert seal < bind < upload_lineage < upload_verified < activation
    assert 'expected_path = "provenance/bfo_lineage/manifest.json"' in core
    assert 'lineage.get("manifest_sha256") != lineage_sha' in core
    assert "weekly-refit-bfo-lineage-{source_sha}-{lineage_sha}" in core
    assert "overwrite: true" in core[upload_lineage:upload_verified]
    assert "needs: [candidate, full_test_slices, seal_candidate]" in core


def test_refit_decision_also_detects_existing_row_corrections():
    core = _workflow("weekly-model-refit-core.yml")
    decision_start = core.index("Decide refit versus no-op")
    decision_end = core.index("Run fixed one-arm walk-forward")
    decision = core[decision_start:decision_end]

    assert "age_days >= 14.0" in decision
    assert "audit_max_date > parent_max_date" in decision
    assert "recovered_rows > 0" in decision
    assert "independent_audit_fights_canonical_sha256" in decision
    assert "independent_audit_features_canonical_sha256" in decision
    assert "audit_inputs_match" in decision
    assert 'reasons.append("independent_audit_changed")' in decision


def test_empty_bfo_batch_preserves_attempt_evidence_and_uses_baseline_ledger():
    core = _workflow("weekly-model-refit-core.yml")

    assert 'if [ "${RECOVERED_ROWS}" -eq 0 ]; then' in core
    assert 'cp -- "${BFO_RECOVERED}" "${BFO_PROVENANCE}" "${BFO_UNRESOLVED}"' in core
    assert '"${BFO_EVIDENCE_DIR}/"' in core
    assert 'rm -f -- "${BFO_RECOVERED}" "${BFO_PROVENANCE}"' in core
    assert (
        "data/raw/historical_odds/"
        "bfo_revalidation_20260805_head_source.provenance.jsonl"
    ) in core
    assert "Preserve run evidence even when a gate fails" in core
    assert "if: always()" in core
