# Tennis Matched-Market Audit Session - 2026-03-23

## Scope

This note records one real dry-run session of the live tennis execution path after the tennis runtime policy cleanup and explicit arming gate work.

The goal was to prove that the current shell can:

- discover active tennis bookmaker markets
- discover active tennis Polymarket winner markets
- fully match live bookmaker and Polymarket rows
- run structured tennis automation controls on real matched rows
- run the Gemini veto layer on those rows
- persist execution-path audit outputs

## Session

- Local session window: 2026-03-23 10:24 PM to 10:27 PM America/New_York
- UTC evidence window from veto log: 2026-03-24T02:25:22Z to 2026-03-24T02:27:00Z
- Command: `python -m src.bot tennis-live --dry-run --model lean_hybrid`
- Local runtime facts:
  - repo `.env` was loaded locally through `src/config.py`
  - tennis LLM veto was enabled for the run
  - this was an observational dry-run only; no orders were submitted

## Observed Market State

- Odds API discovered 2 active tennis tournament sport keys
- bookmaker consensus was built for 11 upcoming tennis matches
- Polymarket discovery returned 8 active tennis winner markets
- live matching produced 4 fully matched execution-path candidates

## Persisted Outputs

- `data/processed/tennis/live_execution_decisions.csv`
- `data/processed/tennis/live_execution_tradeable.csv`
- `data/processed/tennis/live_execution_auto_skipped.csv`
- `data/operator/tennis_veto_log.jsonl`

For the proof session described in this note, the live execution CSVs were first written during the 2026-03-23 10:24 PM to 10:27 PM local run. These CSV filenames are rolling snapshots and may be refreshed by later tennis dry-runs. The proof session should therefore be anchored by the timestamps and counts recorded in this note, plus the matching veto-log entries, rather than assuming the latest contents of the CSVs will stay frozen forever.

## Session Counts

- `execution_status`: `bettable_now=4`
- `automation_status`: `auto_eligible=4`
- `llm_veto_status`: `NO_VETO=4`
- `second_source_status`: `confirmed=4`
- `trade_status`: `tradeable_now=4`

## Example Candidates

| Match | Side | Market ID | Execution Edge | Notes |
| --- | --- | --- | --- | --- |
| Jiri Lehecka vs Taylor Fritz | Taylor Fritz | `1686050` | `+10.0%` | Clean structured pass, `NO_VETO`, second-source confirmed |
| Sebastian Korda vs Martin Landaluce | Sebastian Korda | `1688367` | `+9.6%` | Structured reason `low_history_6_to_10`; LLM cited stale packet context but final status remained `NO_VETO` |
| Valentin Vacherot vs Arthur Fils | Arthur Fils | `1688376` | `+6.8%` | Structured reason `low_history_6_to_10`; strong contradiction reasoning still policy-adjusted to `NO_VETO` |
| Tomas Martin Etcheverry vs Tommy Paul | Tommy Paul | `1688346` | `+2.3%` | Lowest accepted edge in the session; second-source confirmed and trade-ready |

## Interpretation

The current shell met the minimum Commit 3 proof target:

- active tennis Polymarket matchup markets were live
- at least one fully matched candidate existed
- execution-path second-source confirmation ran on real markets
- Gemini veto ran on the matched candidates and persisted evidence
- one short versioned audit note now records the exact observed result

This audit does not close every remaining tennis readiness gap:

- the second source in this session was `polymarket_execution_market`, not an independent external confirmation feed
- several rows carried moderate or strong contradiction reasoning from the LLM but still ended as `NO_VETO` after policy adjustment; that is an audit observation, not a blocker for proving the path executed
- the clean tennis-only commit boundary is still a separate repo hygiene task

## Conclusion

The tennis live shell is now proven at the matched-market dry-run level. The repo can point to this note as the concrete artifact showing that the real execution-path decisioning stack was exercised successfully on live tennis markets on 2026-03-23.
