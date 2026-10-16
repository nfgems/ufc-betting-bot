"""Build a visual fixture from saved audit reads only; never acquire account data."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.web.refresh_dashboard import _private_legacy_module

source = Path('/audit/account-activity')
positions = json.loads((source/'closed-positions-source.json').read_text(encoding='utf-8-sig'))
original = json.loads((source/'get-3.json').read_text(encoding='utf-8-sig'))
raw_live = json.loads((source/'get-5.json').read_text(encoding='utf-8-sig'))
legacy = _private_legacy_module()

def key(row, position=False):
    return (row.get('title' if position else 'event'), row.get('outcome' if position else 'fighter'), row.get('slug'))

counts = Counter(key(row) for row in original['bets'])
known = {key(row): row for row in original['bets'] if counts[key(row)] == 1}

def actual_time(value):
    if not isinstance(value, str) or 'T' not in value:
        return None
    try:
        instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value if instant.tzinfo is not None and instant.utcoffset() is not None else None
    except ValueError:
        return None

rows = []
for position in positions:
    row = legacy._profile_closed_row(position, matched_bets=[], trades=[], redeem_trades=[])
    saved = known.get(key(position, True), {})
    # This original saved adapter produced aware times only from actual trades.
    # Its calendar-only expiry fallback is deliberately rejected here.
    row['placed_at'] = actual_time(saved.get('placed_at'))
    row['settled_at'] = row['settled_at'] or actual_time(saved.get('settled_at'))
    rows.append(row)
profile_summary = original['summary']
profile = {'total_pnl':profile_summary['profile_total_pnl'], 'positions_value':profile_summary['positions_value'],
           'largest_win':profile_summary['profile_largest_win'], 'predictions':profile_summary['profile_predictions'],
           'username':profile_summary['profile_username']}
summary = legacy._build_profile_summary(rows=rows, raw_live_pnl=raw_live, profile_snapshot=profile,
                                      live_source=profile_summary['_pnl_source'], degraded=True)
summary.update(_profile_source=profile_summary['_profile_source'], accounting_scope='shared_wallet', roi=None,
               original_real_ledger_summary=profile_summary['original_real_ledger_summary'],
               _roi_unavailable_reason='A complete observed wallet cost basis is not available')
print(json.dumps({'summary':summary,'bets':rows}))
