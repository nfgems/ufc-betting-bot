# Soft Newbie Rule — Handoff

## What This Is

Test two alternatives to the current hard `MIN_FIGHTER_FIGHTS=3` skip rule, then backtest all three head-to-head against the current production model to see which performs best.

**Three variants to compare:**
1. **Baseline (current)** — hard skip if either fighter has < 3 UFC fights
2. **Simple threshold drop** — lower the hard skip from 3 to 2 UFC fights, no other changes
3. **Tiered org-aware rule** — replace hard skip with graduated edge penalties based on pre-UFC org tier + half position sizing

---

## Current Behavior

`config.py:190` sets `MIN_FIGHTER_FIGHTS = 3`.

`value.py:199-210` in `_passes_quality_filters()` hard-skips any fight where either fighter has < 3 UFC fights:

```python
if (a_num_fights or 0) < MIN_FIGHTER_FIGHTS:
    return False
if (b_num_fights or 0) < MIN_FIGHTER_FIGHTS:
    return False
```

`find_conviction_bets()` at `value.py:640-645` has the same hard skip.

The backtest in `backtest.py:580-596` passes `a_num_fights`/`b_num_fights` through to `_passes_filters`, so the same hard skip applies during backtesting.

**Problem:** Fighters from major orgs (Bellator, ONE, PFL) with 0-2 UFC fights often have 20+ pro fights and are well-priced by the model — the v6 spec already includes `a_pre_ufc_org_tier_best`, `b_pre_ufc_org_tier_best`, and `diff_pre_ufc_org_tier_best` as features. The model can price them, but the decision layer blocks the bet.

---

## Variant A: Simple Threshold Drop (3 → 2)

The simplest possible change. No new logic, no new params, no new data flowing through.

### Changes

**`src/config.py` (~line 190):**
```python
MIN_FIGHTER_FIGHTS = 2  # was 3
```

That's it. One line. Everything else stays the same — the hard skip still exists, it just lets 2-fight fighters through.

### What this tests

Whether the model is already good enough at pricing 2-fight fighters that the extra bet volume is net positive at current edge thresholds. If this works, it's the lowest-risk change with zero added complexity.

---

## Variant B: Tiered Org-Aware Rule

Replace the hard skip entirely with graduated penalties based on how much UFC + pre-UFC experience the fighter has.

### Org Tier Encoding (already exists)

`build_features.py:624-638` — `_encode_org_tier()`:

| Tier | Value | Orgs |
|------|-------|------|
| 1 (Major) | 1.0 | Bellator, ONE, PFL, KSW, RIZIN, etc. |
| 2 (Feeder) | 2.0 | LFA, DWCS, Cage Warriors, CFFC, Invicta, etc. |
| 3 (Regional) | 3.0 | Everything else / unknown |

Lower number = better org. `pre_ufc_org_tier_best` is the *minimum* (best) tier across all pre-UFC orgs for that fighter. NaN if no pre-UFC data exists.

### Proposed Tiered Rule

| UFC Fights | Org Tier | Action |
|-----------|----------|--------|
| 0 | NaN or 3 (regional/unknown) | **Skip** (keep current behavior) |
| 0 | 2 (feeder) | Allow, require +3% extra edge, half position size |
| 0 | 1 (major) | Allow, require +2% extra edge, half position size |
| 1-2 | NaN or 3 | Allow, require +4% extra edge, half position size |
| 1-2 | 2 (feeder) | Allow, require +3% extra edge, half position size |
| 1-2 | 1 (major) | Allow, require +2% extra edge, half position size |
| 3+ | Any | **Current rules** (no penalty) |

### New config params for `src/config.py`

```python
# Soft newbie rule — tiered penalties instead of hard skip
NEWBIE_FIGHTS_THRESHOLD = 3          # Below this, apply penalties
NEWBIE_MAJOR_EXTRA_EDGE = 0.02       # +2% extra edge for tier-1 org fighters with < 3 UFC fights
NEWBIE_FEEDER_EXTRA_EDGE = 0.03      # +3% extra edge for tier-2 org fighters
NEWBIE_REGIONAL_EXTRA_EDGE = 0.04    # +4% extra edge for tier-3/unknown (1-2 fights only; 0 fights still skipped)
NEWBIE_SIZE_MULTIPLIER = 0.50        # Half position size on all newbie bets
NEWBIE_SKIP_ZERO_FIGHTS_REGIONAL = True  # Hard skip 0-fight fighters with no major/feeder org history
```

### Where to make changes

**1. `src/strategy/value.py` — `_passes_quality_filters()`**

Add two new params: `a_org_tier` and `b_org_tier`. Replace the hard skip block (lines 199-210) with tiered logic.

Add a standalone helper so callers can get the penalty info:

```python
def newbie_penalty(num_fights_a, num_fights_b, org_tier_a, org_tier_b) -> tuple[float, float]:
    """Returns (extra_edge_required, size_multiplier).
    Checks BOTH fighters — penalty is based on the least-experienced fighter.
    """
```

The function currently returns `bool`. To also communicate the extra edge and size multiplier, the simplest approach: have the filter apply the extra edge internally by adding it to the required edge in `_passes_filters`, and return the size multiplier via the helper for callers to apply.

**2. `src/strategy/value.py` — `find_value_bets()` and `find_conviction_bets()`**

- Read `a_pre_ufc_org_tier_best` and `b_pre_ufc_org_tier_best` from each row
- Pass them through to the filter
- Apply `NEWBIE_SIZE_MULTIPLIER` to bet sizing when applicable
- Add `size_multiplier` field to the bet dict so downstream code knows to reduce size
- Conviction bets on newbie fighters should stay blocked or require even more edge (larger sizing = more risk)

**3. `src/strategy/backtest.py` — `_run_single_strategy()` (~line 580)**

Read org_tier from the row, pass to `_passes_filters`. Apply size multiplier to `bet_size` before placing. The features DataFrame already has `a_pre_ufc_org_tier_best` / `b_pre_ufc_org_tier_best` columns.

**4. All other callers of the filter functions**

Search for `a_num_fights=a_fights` to find every call site — `bot.py`, `app.py`, `duo_trader_sweep.py`, `triple_trader_backtest.py` all pass fight counts through and will need to also pass org_tier.

---

## How to Test: Three-Way Backtest

### Step 1: Run baseline backtest (current production rules)

```bash
python -m src.bot backtest
```

Record: ROI, Sharpe, win rate, number of bets, worst drawdown.

### Step 2: Implement Variant A (simple threshold drop)

Change `MIN_FIGHTER_FIGHTS = 2` in config.py. Run backtest again. Record same metrics.

### Step 3: Implement Variant B (tiered rule)

Revert config.py back to `MIN_FIGHTER_FIGHTS = 3` (the tiered logic handles its own thresholds). Implement the org-tier-aware changes in value.py and backtest.py. Run backtest again. Record same metrics.

### Step 4: Compare all three

| Metric | Baseline (3-fight skip) | Variant A (2-fight skip) | Variant B (tiered) |
|--------|------------------------|--------------------------|-------------------|
| Total ROI | | | |
| Sharpe | | | |
| Win rate | | | |
| Number of bets | | | |
| Worst drawdown | | | |
| Newbie bet ROI | N/A | (isolate 2-fight bets) | (isolate penalized bets) |
| Newbie bet count | 0 | | |

### Tagging newbie bets in the bet log

For both variants, add `is_newbie_bet: bool` to the bet log dict in `backtest.py` and `find_value_bets`. This lets you isolate and analyze just the newly-allowed bets after each run.

---

## Decision Framework

- **Variant A wins if:** lowering to 2 is net positive and the extra complexity of tiered logic isn't justified. Simplest path to production.
- **Variant B wins if:** the tiered approach captures more profitable bets than the simple drop, especially on 0-1 fight fighters from major orgs that Variant A still skips.
- **Neither wins if:** newbie bets are clearly negative ROI in both. In that case, keep the current 3-fight skip — the model may not have enough signal on these fighters regardless of their pre-UFC history.
- **Both are roughly equal:** go with Variant A. Simpler is better when the edge is similar.

---

## Risk Notes

- If newbie bets are roughly break-even, it's still worth keeping — more bets at neutral EV means more data for future model improvements.
- If newbie bets are clearly negative ROI, tighten the extra edge requirements (Variant B) or raise back to 3 (Variant A) before giving up entirely.
- The half-sizing in Variant B (`NEWBIE_SIZE_MULTIPLIER=0.50`) is the safety net. Don't remove it even if newbie bets look profitable in backtest — the sample will be small.
- Conviction bets on newbie fighters should stay blocked in both variants since conviction sizing is larger.
- Implement and test Variant A first — it's one line and gives you the baseline for whether the model prices 2-fight fighters well. If it doesn't, Variant B probably won't save it.
