"""Measure coverage from the exact retained candidate used for publication."""
from pathlib import Path
import uuid


def _paths(processing):
    from src.data import roster_refresh_integration as bridge

    scheduled = bridge._scheduled()
    raw = Path(processing['staging']['candidate_raw_root'])
    processed = Path(processing['staged_processed_root'])
    return {'active_roster_path': raw / bridge.ROSTER_FILE,
             'processed_fights_path': processed / 'fights_cleaned.csv',
             'scraped_fighters_path': raw / scheduled.BACKFILL_FIGHTERS_PATH.name,
             'supplemental_profiles_path': raw / scheduled.PROFILE_SUPPLEMENT_PATH.name}


def _compute(paths, observed_at):
    from src.data import roster_refresh_integration as bridge

    scheduled = bridge._scheduled()
    with bridge.network_forbidden():
        summary, rows = scheduled.run_audit(**paths)
        unresolved_summary, unresolved = scheduled._build_unresolved_profile_report(
            active_roster_path=paths['active_roster_path'], scraped_fighters_path=paths['scraped_fighters_path'],
            audit_df=rows, supplement_path=paths['supplemental_profiles_path'])
        alerts = scheduled._build_profile_audit_alert_summary(
            active_roster_path=paths['active_roster_path'], audit_df=rows,
            unresolved_df=unresolved, as_of_utc=observed_at)
    return summary, alerts, unresolved_summary


def measure(processing_pin, *, cycle_root, clock):
    from src.data import roster_refresh_integration as bridge

    processing = bridge._load(processing_pin)
    paths = _paths(processing)
    pins = {name: bridge.file_pin(path) for name, path in paths.items() if path.is_file()}
    # Required inputs fail explicitly. Optional supplement absence is itself
    # bound so an unmeasured late file cannot influence publication.
    for name in ('active_roster_path', 'processed_fights_path', 'scraped_fighters_path'):
        bridge._require(name in pins, f'coverage audit input missing: {name}')
    absent = [str(path) for name, path in paths.items() if name not in pins]
    observed_at = bridge._time(clock())
    summary, alerts, unresolved_summary = _compute(paths, observed_at)
    for pin in pins.values():
        bridge.verify_pin(pin, readonly=False)
    bridge._require(all(not Path(path).exists() for path in absent), 'coverage audit input appeared during measurement')
    snapshot = bridge.coverage_snapshot_from_audit(summary, alerts)
    record = {'schema': 1, 'kind': 'roster_refresh_coverage_audit_v1', 'cycle_id': processing['cycle_id'],
              'processing': processing_pin, 'measured_at_utc': observed_at.isoformat(),
              'input_pins': pins, 'absent_optional_inputs': absent,
              'summary': summary, 'alert_summary': alerts, 'unresolved_summary': unresolved_summary,
              'coverage_snapshot': snapshot}
    pin = bridge._save(Path(cycle_root) / f'coverage-audit-{uuid.uuid4().hex}.json', record)
    return pin, snapshot, alerts


def verify(pin, *, processing_pin, decision):
    from src.data import roster_refresh_integration as bridge

    record = bridge._load(pin)
    processing = bridge._load(processing_pin)
    paths = _paths(processing)
    bridge._require(record.get('kind') == 'roster_refresh_coverage_audit_v1' and record.get('schema') == 1
                    and record.get('processing') == processing_pin
                    and record.get('cycle_id') == processing['cycle_id'] == decision['cycle_id'],
                    'measured coverage audit binding changed')
    expected_pins = {name: bridge.file_pin(path) for name, path in paths.items() if path.is_file()}
    absent = [str(path) for name, path in paths.items() if name not in expected_pins]
    bridge._require(record['input_pins'] == expected_pins and record['absent_optional_inputs'] == absent,
                    'coverage audit input set changed')
    bridge._require({'active_roster_path', 'processed_fights_path', 'scraped_fighters_path'} <= set(expected_pins),
                    'required coverage audit inputs missing')
    for source in record['input_pins'].values():
        bridge.verify_pin(source, readonly=False)
    bridge._require(all(not Path(path).exists() for path in record['absent_optional_inputs']),
                    'coverage audit optional input changed')
    summary, alerts, unresolved = _compute(paths, bridge._time(record['measured_at_utc']))
    snapshot = bridge.coverage_snapshot_from_audit(summary, alerts)
    bridge._require(record['summary'] == summary and record['alert_summary'] == alerts
                    and record['unresolved_summary'] == unresolved and record['coverage_snapshot'] == snapshot
                    and decision['coverage_snapshot'] == snapshot and decision['audit_alert_summary'] == alerts,
                    'coverage audit measured values changed')
    for source in record['input_pins'].values():
        bridge.verify_pin(source, readonly=False)
    return record
