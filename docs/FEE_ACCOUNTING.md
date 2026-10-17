# Observed entry costs

The ledger preserves each original order, trade, price, gross amount and rounded
share quantity. A separate cash-cost record links a confirmed original taker BUY
trade to one observed account activity by transaction, token, quantity and price.
The observed debit determines the full entry cost. Current fee schedules and
reported fee rates never substitute for an actual charge.

Verified exact trade quantities remain separate from the original rounded fields.
Future settlement uses those quantities and the observed debit, retaining exact
decimal P&L alongside the existing cents display. Repeated maintenance verifies
the same evidence without charging a fee again. Ledger locking rejects stale
fill bindings, conflicting evidence and changes to already-settled rows.

The normal trading maintenance cycle enriches original open real S/C/M fills.
Missing or ambiguous evidence leaves accounting incomplete and postpones financial
finalization. Unknown amounts remain null in bot summaries; they are not zero.
The active dashboard keeps exchange wallet totals separate from original bot
accounting, and private linkage evidence stays out of public responses.

This bounded attribution supports uniquely matched taker BUY trades with distinct
transactions. Maker fills, shared transaction activity, missing identities and
incomplete bounded activity reads remain unknown. Existing settled history is
preserved and may omit earlier fees; this change is not a historical repair or
a claim that the entire wallet's lifetime P&L is fee-complete.

Bet selection, sizing rules, risk limits, account submission/cancellation policy,
models and refresh schedule are unchanged. No extra account write is required
to establish a cost. Verification uses synthetic isolated tests and, after
deployment, read-only evidence from normal production cycles.

The original October 16 source manifest and image remain immutable. Replacements
are pinned in `source-change-manifest.json` under `files`; the new cashflow module
is explicitly pinned under `additions`. A successful CI release receipt plus
actual hosted source/behavior establishes deployment, not this document.
