from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.data import incremental_refresh as refresh

EVENT = "http://ufcstats.com/event-details/" + "e" * 16
FIGHT = "http://ufcstats.com/fight-details/" + "f" * 16
IDS = ("a" * 16, "b" * 16)
NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


def test_request_intent_directory_links_are_synced_before_atomic_acknowledgment(tmp_path, monkeypatch):
    path = tmp_path / "new-cache" / "original-job" / "request.intent.json"
    syncs = []
    def observe_sync(directory):
        syncs.append((Path(directory), path.exists()))
    monkeypatch.setattr(refresh, "_fsync_directory", observe_sync)
    refresh._atomic(path, b'{"url":"original"}\n')
    assert syncs == [(tmp_path, False), (tmp_path / "new-cache", False), (path.parent, True)]
    assert path.read_bytes() == b'{"url":"original"}\n'


def fight_html():
    def cell(a, b):
        return f"<td><p>{a}</p><p>{b}</p></td>"
    totals = [("Alpha", "Beta"), ("0", "1"), ("20 of 40", "30 of 50"), ("50%", "60%"),
              ("25 of 50", "40 of 60"), ("0 of 0", "1 of 2"), ("0%", "50%"),
              ("0", "0"), ("0", "0"), ("1:00", "2:00")]
    breakdown = [("Alpha", "Beta"), ("20 of 40", "30 of 50"), ("50%", "60%"),
                 ("10 of 20", "10 of 20"), ("5 of 10", "10 of 15"), ("5 of 10", "10 of 15"),
                 ("10 of 20", "10 of 20"), ("5 of 10", "10 of 15"), ("5 of 10", "10 of 15")]
    return (f'<html><h2 class="b-content__title"><a class="b-link" href="{EVENT}">UFC New</a></h2>' + "".join(
        f'<div class="b-fight-details__person"><i class="b-fight-details__person-status">{status}</i>'
        f'<h3 class="b-fight-details__person-name"><a href="http://ufcstats.com/fighter-details/{identity}">{name}</a></h3></div>'
        for name, identity, status in zip(("Alpha", "Beta"), IDS, ("W", "L")))
        + '<i class="b-fight-details__fight-title">Lightweight Bout</i>'
        + '<i class="b-fight-details__text-item">Method: Decision - Unanimous</i>'
        + '<i class="b-fight-details__text-item">Round: 3</i>'
        + '<i class="b-fight-details__text-item">Time: 5:00</i>'
        + '<i class="b-fight-details__text-item">Time format: 3 Rnd (5-5-5)</i>'
        + "<table><tbody><tr>" + "".join(cell(*pair) for pair in totals) + "</tr></tbody></table>"
        + "<table><tbody><tr>" + "".join(cell(*pair) for pair in breakdown) + "</tr></tbody></table></html>")


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "release"
    processed = root / "processed"
    processed.mkdir(parents=True)
    frame = pd.DataFrame([
        {"event_date": "2026-09-12", "fighter_a": "Old Alpha", "fighter_b": "Old Beta",
         "fighter_a_id": "1" * 16, "fighter_b_id": "2" * 16, "value": "original"},
        {"event_date": "2026-09-12", "fighter_a": "Old Alpha", "fighter_b": "Old Beta",
         "fighter_a_id": "1" * 16, "fighter_b_id": "2" * 16, "value": "inherited duplicate"},
    ])
    pins = {}
    for name in refresh.FILES:
        path = processed / name
        frame.to_csv(path, index=False, lineterminator="\r\n")
        pins[name] = {**refresh._pin(path.read_bytes()), "path": "processed/" + name}
    models = root / "models"
    models.mkdir()
    model = models / "model.bin"
    model.write_bytes(b"immutable trusted model")
    manifest = root / "selected-runtime-manifest.json"
    manifest.write_text(json.dumps({"processed": {"files": pins}, "models": {
        "primary": {**refresh._pin(model.read_bytes()), "path": "models/model.bin"}}}))
    return root


@pytest.fixture
def fetcher():
    responses = {
        refresh.INDEX_URL: f'<tr class="b-statistics__table-row"><td><a class="b-link" href="{EVENT}">UFC New</a><span class="b-statistics__date">September 19, 2026</span></td><td>Las Vegas, Nevada, USA</td></tr>',
        EVENT: '<li class="b-list__box-list-item">Date: September 19, 2026</li>' + f'<tr class="b-fight-details__table-row" data-link="{FIGHT}"></tr>',
        FIGHT: fight_html(),
    }
    for name, identity in zip(("Alpha", "Beta"), IDS):
        responses["http://ufcstats.com/fighter-details/" + identity] = (
            f'<h2 class="b-content__title"><span>{name}</span></h2>'
            '<li class="b-list__box-list-item">Height: 5\' 10"</li>'
            '<li class="b-list__box-list-item">Reach: 72"</li>'
            '<li class="b-list__box-list-item">DOB: January 01, 1990</li>')
    calls = []
    def fetch(url):
        calls.append(url)
        return responses[url]
    fetch.calls, fetch.responses = calls, responses
    return fetch


def pair_builder(history, rows, profiles, source_manifest):
    frame = pd.DataFrame([{**row, "value": "genuine new"} for row in rows])
    return frame, frame.copy()


def run(baseline, state, fetcher, **kwargs):
    return refresh.run_incremental_refresh(baseline_dir=baseline, state_dir=state,
        job_id="original-job", decision_at=NOW, fetcher=fetcher, pair_builder=pair_builder, **kwargs)


def test_exact_both_prefixes_inherited_duplicate_models_and_idempotency(baseline, tmp_path, fetcher):
    state = tmp_path / "state"
    original = {name: (baseline / "processed" / name).read_bytes() for name in refresh.FILES}
    result = run(baseline, state, fetcher)
    assert result["status"] == "completed" and result["new_rows"] == 1 and result["total_rows"] == 3
    for name in refresh.FILES:
        assert Path(result["artifacts"][name]["path"]).read_bytes().startswith(original[name])
        assert (baseline / "processed" / name).read_bytes() == original[name]
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["inherited_rows"] == 2 and report["inherited_distinct_identities"] == 1
    calls = list(fetcher.calls)
    assert run(baseline, state, fetcher) == result
    assert fetcher.calls == calls
    serialized = (state / "accepted.json").read_text()
    assert str(tmp_path).replace("\\", "\\\\") not in serialized
    assert "${STATE_DIR}" in serialized and "${BASELINE_DIR}" in serialized


@pytest.mark.parametrize("phase", ["after_parsed_fight", "after_fights_cleaned.csv", "after_features.csv", "after_staged", "after_pointer"])
def test_interrupt_resume_same_job_no_successful_request_replay(baseline, tmp_path, fetcher, phase):
    state = tmp_path / "state"
    def interrupt(actual):
        if actual == phase:
            raise KeyboardInterrupt("simulated interruption")
    with pytest.raises(KeyboardInterrupt):
        run(baseline, state, fetcher, fault=interrupt)
    prior_calls = list(fetcher.calls)
    pointer_before = ((state / "accepted.json").read_bytes(), (state / "accepted.json").stat().st_mtime_ns) if phase == "after_pointer" else None
    original_decision = json.loads((state / "runs" / "original-job" / "journal.json").read_text())["decision_at"]
    result = refresh.recover_incremental_refresh(state_dir=state, baseline_dir=baseline,
        fetcher=fetcher, pair_builder=pair_builder)
    assert result["status"] == "completed"
    assert len(fetcher.calls) == len(set(fetcher.calls))
    if phase != "after_parsed_fight":
        assert fetcher.calls == prior_calls
    assert result["job_id"] == "original-job"
    assert datetime.fromisoformat(result["completed_at"]).utcoffset() is not None
    assert result["completed_at"] == result["finished_at"]
    assert json.loads((state / "runs" / "original-job" / "journal.json").read_text())["decision_at"] == original_decision
    if pointer_before is not None:
        assert ((state / "accepted.json").read_bytes(), (state / "accepted.json").stat().st_mtime_ns) == pointer_before


def test_unanswered_request_holds_original_claim_without_reissuing(baseline, tmp_path, fetcher):
    state = tmp_path / "state"
    def interrupted(url):
        raise KeyboardInterrupt("after request intent")
    with pytest.raises(KeyboardInterrupt):
        run(baseline, state, interrupted)
    assert refresh.inspect_active_refresh(state)["resume_allowed"] is False
    result = refresh.recover_incremental_refresh(state_dir=state, baseline_dir=baseline, fetcher=fetcher)
    assert result["status"] == "held" and fetcher.calls == []
    assert not (state / "accepted.json").exists()
    intent = next((state / "cache" / "original-job").glob("*.intent.json"))
    original_started_at = json.loads(intent.read_text())["started_at"]
    assert run(baseline, state, fetcher, retry_uncertain=True)["status"] == "completed"
    retried = json.loads(intent.read_text())
    assert retried["started_at"] == original_started_at
    assert len(retried["explicit_retry_started_at"]) == 1


def test_corrupt_raw_response_or_model_never_advances(baseline, tmp_path, fetcher):
    state = tmp_path / "state"
    def interrupt(phase):
        if phase == "after_parsed_fight":
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(baseline, state, fetcher, fault=interrupt)
    (baseline / "models" / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model changed"):
        run(baseline, state, fetcher)
    assert not (state / "accepted.json").exists()


def test_result_requires_paired_genuine_stats():
    event = {"url": EVENT, "name": "UFC New", "date": "2026-09-19", "location": "Las Vegas"}
    parsed = refresh._parse_fight(fight_html(), FIGHT, event)
    assert parsed["a_sig_str_landed"] == "20" and parsed["b_sig_str_landed"] == "30"
    with pytest.raises(ValueError, match="paired totals"):
        refresh._parse_fight(fight_html().split("<table>")[0], FIGHT, event)


def test_definitive_saved_quality_failure_is_durable_and_never_published(baseline, tmp_path, fetcher):
    state = tmp_path / "state"
    fetcher.responses[FIGHT] = fight_html().split("<table>")[0]
    result = run(baseline, state, fetcher)
    assert result["status"] == "failed" and result["failure_complete"] is True
    assert result["accepted_pointer_unchanged"] and not result["publication_performed"]
    assert not (state / "accepted.json").exists()
    original_calls = list(fetcher.calls)
    assert run(baseline, state, fetcher) == result
    assert fetcher.calls == original_calls
    assert refresh.inspect_active_refresh(state) is None


def test_portable_seed_recovers_staged_without_network(baseline, tmp_path, fetcher):
    state = tmp_path / "state"
    def interrupt(phase):
        if phase == "after_staged":
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(baseline, state, fetcher, fault=interrupt)
    relocated = tmp_path / "relocated"
    relocated_baseline = tmp_path / "relocated-release"
    shutil.copytree(state, relocated)
    shutil.copytree(baseline, relocated_baseline)
    def no_network(url):
        raise AssertionError("Recovery repeated acquisition")
    result = refresh.recover_incremental_refresh(state_dir=relocated, baseline_dir=relocated_baseline, fetcher=no_network)
    assert result["status"] == "completed"
    assert Path(result["fights_path"]).is_relative_to(relocated)
    assert Path(result["models"]["primary"]["path"]).is_relative_to(relocated_baseline)


def test_real_scientific_builder_uses_observed_stats_without_target_result_leakage(tmp_path):
    event = {"url": EVENT, "name": "UFC New", "date": "2026-09-19", "location": "Las Vegas"}
    row = refresh._parse_fight(fight_html(), FIGHT, event)
    profiles = tmp_path / "profiles.csv"
    pd.DataFrame([{"name": name, "fighter_url": "http://ufcstats.com/fighter-details/" + identity,
                   "height": "5' 10\"", "reach": "72\"", "dob": "1990-01-01", "stance": "Orthodox"}
                  for name, identity in zip(("Alpha", "Beta"), IDS)]).to_csv(profiles, index=False)
    history = pd.DataFrame(columns=["event_date", "fighter_a", "fighter_b", "fighter_a_id", "fighter_b_id"])
    fights, features = refresh.build_new_pair(history, [row], profiles)
    assert len(fights) == len(features) == 1
    assert fights.iloc[0]["a_sig_str_landed"] == 20
    assert fights.iloc[0]["target"] == 1
    # Acquired postfight totals remain observations; debut prematch statistics
    # remain unknown instead of being filled with this fight's own result.
    assert pd.isna(features.iloc[0]["a_roll_slpm"])


def scientific_graph(tmp_path):
    """Own histories plus last-five opponents with independently varied records."""
    from src.data.ufc_refresh import _load_scraped_fighter_lookup, _scraped_fights_to_training_rows
    names = ["Alpha", "Beta", "Gamma"] + [f"Opponent {index}" for index in range(1, 7)] + ["Third One", "Third Two", "Unrelated One", "Unrelated Two"]
    ids = {name: f"{index + 1:016x}" for index, name in enumerate(names)}
    profiles = tmp_path / "profiles.csv"
    pd.DataFrame([{"name": name, "fighter_url": "http://ufcstats.com/fighter-details/" + identity,
                   "height": "5' 10\"", "reach": "72\"", "dob": "1990-01-01", "stance": "Orthodox"}
                  for name, identity in ids.items()]).to_csv(profiles, index=False)
    event = {"url": EVENT, "name": "UFC New", "date": "2026-09-19", "location": "Las Vegas"}
    seed = refresh._parse_fight(fight_html(), FIGHT, event)
    rows = []
    def add(date, a, b, a_wins=True):
        index = len(rows)
        rows.append({**seed, "event_date": date, "fighter_a": a, "fighter_b": b,
                     "fighter_a_id": ids[a], "fighter_b_id": ids[b], "winner": a if a_wins else b,
                     "official_outcome": "W/L" if a_wins else "L/W",
                     "fight_url": "http://ufcstats.com/fight-details/" + f"{index + 1:016x}",
                     "a_sig_str_landed": str(20 + index), "a_sig_str_attempted": str(40 + index),
                     "b_sig_str_landed": str(15 + index), "b_sig_str_attempted": str(35 + index)})
    for index in range(1, 7):
        add(f"2019-0{index}-01", f"Opponent {index}", "Third One", bool(index % 2))
        add(f"2020-0{index}-01", f"Opponent {index}", "Third Two", bool((index + 1) % 3))
        add(f"2021-0{index}-01", "Alpha", f"Opponent {index}", bool(index % 3))
    add("2022-01-01", "Beta", "Opponent 1")
    add("2022-02-01", "Beta", "Opponent 3", False)
    add("2022-03-01", "Gamma", "Opponent 2")
    add("2022-04-01", "Gamma", "Opponent 4", False)
    add("2025-01-01", "Unrelated One", "Unrelated Two")
    first_target = len(rows)
    add("2026-09-19", "Alpha", "Beta")
    add("2026-09-26", "Beta", "Gamma", False)
    add("2026-10-03", "Alpha", "Gamma", False)
    lookup = _load_scraped_fighter_lookup(profiles)
    normalized = _scraped_fights_to_training_rows(pd.DataFrame(rows), fighter_lookup=lookup)
    # No market observation was acquired; structural contract columns carry NaN.
    for column in ("a_odds", "b_odds", "a_ko_odds", "a_sub_odds", "a_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"):
        normalized[column] = float("nan")
    return normalized.iloc[:first_target].copy(), normalized.iloc[first_target:].copy()


def test_incremental_scientific_211_contract_matches_complete_context_and_excludes_future(tmp_path):
    from src.features.build_features import build_features, build_new_features
    from src.model.training_spec import NamedModelTrainingSpec, materialize_and_validate_spec_features
    history, targets = scientific_graph(tmp_path)
    observed = pd.concat([history, targets], ignore_index=True)
    full = build_features(observed)
    incremental = build_new_features(history, targets)
    spec = NamedModelTrainingSpec.load(Path(__file__).resolve().parents[1] / "release_payload" / "models" / "full_live_contract_v6_durability_2022_spec.json")
    assert len(spec.feature_cols) == 211
    expected = materialize_and_validate_spec_features(full.iloc[-len(targets):], spec)[spec.feature_cols].reset_index(drop=True)
    actual = materialize_and_validate_spec_features(incremental, spec)[spec.feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    assert actual["a_opp_strength"].notna().all()
    assert incremental.iloc[2]["a_num_fights"] == incremental.iloc[0]["a_num_fights"] + 1
    future_changed = targets.copy()
    future_changed.loc[future_changed.index[-1], ["a_sig_str_landed", "a_sig_str_attempted", "target"]] = [9999, 10000, 1]
    changed = materialize_and_validate_spec_features(build_new_features(history, future_changed), spec)
    pd.testing.assert_series_equal(actual.iloc[0], changed[spec.feature_cols].iloc[0], check_dtype=False, check_exact=False)


def test_irrelevant_missing_legacy_opponent_is_retained_but_required_association_holds(tmp_path):
    from src.features.build_features import build_new_features, UnresolvedTrainingFighterIdentityError
    history, targets = scientific_graph(tmp_path)
    legacy = history.copy()
    # Opponent 2's old Third One opponent is not a required owner. Its actual
    # observations still supply Opponent 2's own prior win-rate timeline.
    irrelevant = (legacy["fighter_a"] == "Opponent 2") & (legacy["fighter_b"] == "Third One")
    legacy.loc[irrelevant, "fighter_b_id"] = None
    legacy.loc[irrelevant, "fighter_b"] = "Unresolved Legacy Third"
    result = build_new_features(legacy, targets)
    expected = build_new_features(history, targets)
    pd.testing.assert_series_equal(result["a_opp_strength"], expected["a_opp_strength"])
    # A blank inherited ID joins the one genuine observed owner without editing
    # that stored ID. A contradictory name/ID association must still hold.
    needed = (legacy["fighter_a"] == "Alpha") & (legacy["fighter_b"] == "Opponent 2")
    legacy.loc[needed, "fighter_b_id"] = None
    assert len(build_new_features(legacy, targets)) == len(targets)
    assert legacy.loc[needed, "fighter_b_id"].isna().all()
    conflicting = legacy.iloc[-1].copy()
    conflicting["fighter_a"] = "Opponent 2"
    conflicting["fighter_a_id"] = "e" * 16
    legacy = pd.concat([legacy, pd.DataFrame([conflicting])], ignore_index=True)
    with pytest.raises(UnresolvedTrainingFighterIdentityError, match="Required prior owner association|last-five opponent"):
        build_new_features(legacy, targets)


def test_exact_saved_historical_sources_separate_namesakes_and_survive_later_jobs(tmp_path):
    from src.data.ufc_refresh import _load_scraped_fighter_lookup, _scraped_fights_to_training_rows
    state = tmp_path / "state"
    cache = refresh.ResponseCache(state / "cache" / "original", fetcher=lambda url: fight_html())
    cache.get_text(FIGHT)
    old_profile = "http://ufcstats.com/fighter-details/" + IDS[0]
    profile_html = (f'<h2 class="b-content__title"><span>Alpha</span></h2>'
        f'<tr class="b-fight-details__table-row" data-link="{FIGHT}"><td>'
        f'<a href="http://ufcstats.com/fighter-details/{IDS[0]}">Alpha</a>'
        f'<a href="http://ufcstats.com/fighter-details/{IDS[1]}">Beta</a></td><td>'
        f'<p><a href="{EVENT}">UFC New</a></p><p>Sep. 19, 2025</p></td></tr>')
    cache.fetcher = lambda url: profile_html
    cache.get_text(old_profile)
    profiles = state / "runs" / "original" / "profiles.csv"
    profiles.parent.mkdir(parents=True)
    pd.DataFrame([{"name": name, "fighter_url": "http://ufcstats.com/fighter-details/" + identity,
                   "height": "5' 10\"", "reach": "72\"", "dob": "1990-01-01", "stance": "Orthodox"}
                  for name, identity in (("Alpha", IDS[0]), ("Beta", IDS[1]), ("Alpha", "c" * 16), ("Gamma", "d" * 16))]).to_csv(profiles, index=False)
    event = {"url": EVENT, "name": "UFC New", "date": "2025-09-19", "location": "Las Vegas"}
    raw_old = refresh._parse_fight(fight_html(), FIGHT, event)
    history = _scraped_fights_to_training_rows(pd.DataFrame([raw_old]), fighter_lookup=_load_scraped_fighter_lookup(profiles))
    history[["fighter_a_id", "fighter_b_id"]] = None
    untouched = history.copy(deep=True)
    new = {**raw_old, "event_date": "2026-10-10", "fighter_a_id": "c" * 16,
           "fighter_b_id": "d" * 16, "fighter_b": "Gamma", "fight_url": "http://ufcstats.com/fight-details/" + "9" * 16}
    fights, features = refresh.build_new_pair(history, [new], profiles)
    assert fights.iloc[0]["fighter_a_id"] == "c" * 16
    assert pd.isna(features.iloc[0]["a_roll_slpm"])
    retained = features.attrs["authenticated_historical_associations"]
    assert {item["owner_key"] for item in retained} == {"ufcstats:" + identity for identity in IDS}
    pd.testing.assert_frame_equal(history, untouched)
    next_profiles = state / "runs" / "next" / "profiles.csv"
    next_profiles.parent.mkdir(parents=True)
    shutil.copyfile(profiles, next_profiles)
    later_fights, later_features = refresh.build_new_pair(history, [new], next_profiles, retained_associations=retained)
    pd.testing.assert_frame_equal(features, later_features)
    assert later_features.attrs["authenticated_historical_associations"] == retained
    damaged = json.loads(json.dumps(retained))
    damaged[0]["profile_response"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Retained historical association source changed"):
        refresh.build_new_pair(history, [new], next_profiles, retained_associations=damaged)
