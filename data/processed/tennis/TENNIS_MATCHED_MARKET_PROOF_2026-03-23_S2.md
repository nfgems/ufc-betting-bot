# Tennis Matched-Market Proof — 2026-03-23 Session 2

## Purpose

Prove the full tennis execution pipeline works end-to-end on real live markets:
discover bookmaker odds, discover Polymarket winner markets, match them to model
predictions, run structured automation controls, run Gemini LLM veto, persist
all artifacts.  This is the second captured session on 2026-03-23.

## Session Facts

| Field | Value |
|---|---|
| Command | `python -m src.bot tennis-live --dry-run --model lean_hybrid` |
| Local window | 2026-03-23 10:38 PM – 10:40 PM America/New_York |
| UTC evidence window | 2026-03-24T02:39:27Z – 2026-03-24T02:40:57Z |
| Mode | Dry-run only — no orders submitted |
| LLM veto | Enabled (Gemini 2.5 Flash) |
| Odds API calls used | 3 (sports list + 2 tournament keys) |
| Odds API remaining | 86,342 |

## Market Discovery

| Stage | Count |
|---|---|
| Active tennis tournament sport keys | 2 |
| Bookmaker consensus matches built | 10 |
| Bookmakers per match | 9 |
| Polymarket winner markets discovered | 8 |
| Fully matched execution candidates | 4 |

All 4 matched candidates were ATP Miami Open Round-of-16 matches scheduled for 2026-03-24.

## Pipeline Status Counts

| Layer | Breakdown |
|---|---|
| `execution_status` | `bettable_now=3`, `llm_auto_block=1` |
| `automation_status` | `auto_eligible=3`, `auto_block=1` |
| `llm_veto_status` | `NO_VETO=3`, `AUTO_BLOCK=1` |
| `second_source_status` | `confirmed=4` |
| `trade_status` | `tradeable_now=3`, `blocked=1` |

## Candidate Detail

### 1. Jiri Lehecka vs Taylor Fritz — TRADEABLE

| Field | Value |
|---|---|
| Decision side | **Taylor Fritz** |
| Market ID | `1686050` |
| Model prob | 75.5% |
| Bookmaker consensus | 63.6% |
| Polymarket price | 65.5% |
| Reference edge | +11.9% |
| Execution edge | **+10.0%** |
| Required edge | 2.0% |
| Hypothetical stake | $20.00 |
| LLM veto | `NO_VETO` (conf 0.90, contradiction: none) |
| Flags | `packet_snapshot_lag_only` |

Clean pass. Fritz is an elite hard-court player; model and all sources agree directionally.

### 2. Sebastian Korda vs Martin Landaluce — TRADEABLE

| Field | Value |
|---|---|
| Decision side | **Sebastian Korda** |
| Market ID | `1688367` |
| Model prob | 87.1% |
| Bookmaker consensus | 76.0% |
| Polymarket price | 77.5% |
| Reference edge | +11.1% |
| Execution edge | **+9.6%** |
| Required edge | 3.0% (low_history_6_to_10 penalty) |
| Hypothetical stake | $20.00 |
| LLM veto | `NO_VETO` (conf 0.85, contradiction: strong) |
| Flags | `packet_rank_lag_significant`, `player_B_massive_breakout_underestimated` |

LLM flagged Landaluce's breakout (rank jumped from packet 322 to live ~123) but policy adjustment forced `NO_VETO` because the model still has Korda favored and both bookmaker consensus and Polymarket agree.

### 3. Tomas Martin Etcheverry vs Tommy Paul — TRADEABLE

| Field | Value |
|---|---|
| Decision side | **Tommy Paul** |
| Market ID | `1688346` |
| Model prob | 72.8% |
| Bookmaker consensus | 69.9% |
| Polymarket price | 70.5% |
| Reference edge | +2.8% |
| Execution edge | **+2.3%** |
| Required edge | 2.0% |
| Hypothetical stake | $9.60 |
| LLM veto | `NO_VETO` (conf 0.80, contradiction: moderate) |
| Flags | `player_b_rank_snapshot_lag`, `player_a_surface_adaptation_and_tournament_conditions`, `player_b_recent_match_fatigue_risk` |

Lowest-edge candidate in the session. LLM noted Paul's fatigue risk and Etcheverry's improving hard-court form but did not escalate to block.

### 4. Valentin Vacherot vs Arthur Fils — BLOCKED

| Field | Value |
|---|---|
| Decision side | **Arthur Fils** |
| Market ID | `1688376` |
| Model prob | 77.3% |
| Bookmaker consensus | 70.1% |
| Polymarket price | 70.5% |
| Reference edge | +7.2% |
| Execution edge | +6.8% |
| Required edge | 3.0% (low_history_6_to_10 penalty) |
| LLM veto | **`AUTO_BLOCK`** (conf 0.90, contradiction: strong) |
| Block reasons | Vacherot rank severely understated (packet 116 vs actual 25), age wrong (25.5 vs 27), Vacherot is 2025 Shanghai Masters champion and seeded 24th at Miami |

Correct block. The model's snapshot data for Vacherot was catastrophically stale — the LLM correctly identified that the entire bet thesis (Fils vs a low-ranked opponent) was invalid because Vacherot is actually a top-25 Masters 1000 champion.

## Persisted Artifacts

| File | Rows | Description |
|---|---|---|
| `data/processed/tennis/live_execution_decisions.csv` | 4 | Full decision frame with all columns |
| `data/processed/tennis/live_execution_tradeable.csv` | 3 | Trade-ready subset |
| `data/processed/tennis/live_execution_auto_skipped.csv` | 1 | Auto-blocked subset (Vacherot/Fils) |
| `data/operator/tennis_veto_log.jsonl` | +4 entries | Gemini veto verdicts with rationale and flags |

Note: The CSV filenames are rolling snapshots overwritten by each tennis dry-run. This proof session is anchored by the UTC timestamps, counts, and matching veto-log entries recorded here, not by assuming the CSV contents stay frozen.

## Key Differences from Session 1

| Aspect | Session 1 (10:24 PM) | Session 2 (10:38 PM) |
|---|---|---|
| Bookmaker consensus matches | 11 | 10 |
| LLM veto outcomes | `NO_VETO=4` | `NO_VETO=3`, `AUTO_BLOCK=1` |
| Vacherot/Fils | Policy-adjusted to `NO_VETO` | **`AUTO_BLOCK`** — LLM caught stale data |
| Trade-ready count | 4 | 3 |

The Vacherot/Fils flip from `NO_VETO` to `AUTO_BLOCK` between sessions demonstrates that the LLM veto layer is non-deterministic and capable of catching real data quality issues when it gets richer source evidence.

## Observations

1. **LLM veto correctly blocked a bad bet.** The Vacherot/Fils `AUTO_BLOCK` is exactly the safety net this layer is designed to provide — the model's snapshot data was months stale and the LLM caught it with rich source evidence.

2. **Snapshot lag is systemic.** Three of the four candidates had LLM-flagged rank discrepancies between the model's training snapshot and current reality. The `snapshot_lag_reasoning_forced_no_veto` policy adjustment is doing heavy lifting.

3. **Second source is still execution-market only.** All 4 rows used `polymarket_execution_market` as the second source, not an independent bookmaker feed. This gap remains open.

4. **Edge distribution is reasonable.** The 3 tradeable candidates span +2.3% to +10.0% execution edge. Stake sizing scaled down proportionally for the thin-edge Etcheverry/Paul match ($9.60 vs $20.00).

## Conclusion

This session confirms the tennis matched-market pipeline is fully operational at the dry-run level. All pipeline stages executed on real Odds API data, real Polymarket markets, and real Gemini veto calls. The LLM veto layer demonstrated genuine protective value by blocking a trade built on stale data. The repo can point to this artifact and the persisted CSVs/JSONL as concrete proof of end-to-end execution on live tennis markets.
