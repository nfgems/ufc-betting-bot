# CLAUDE.md — UFC Betting Bot

## What This Is

ML-powered UFC fight prediction and Polymarket execution bot with experimental tennis support (dry-run only). Covers data collection, feature engineering, model training, backtesting, live prediction, and a Flask dashboard.

## Quick Reference

```bash
# Run tests
pytest -q tests/

# CI smoke checks (run before pushing)
python -m compileall src tests

# Main CLI
python -m src.bot <command>    # scrape, train, predict, live, backtest, etc.
```

## How Things Fit Together

- The CLI orchestrator in `src/bot.py` is the main entry point — all commands route through here.
- All settings, thresholds, and env var loading live in `src/config.py`. Strategy params belong here, not scattered around.
- Model training is driven by named training contracts (feature lists + hyperparams). Every promoted model must have a reproducible spec.
- The production web entrypoint starts Flask + a background monitor + the betting loop.
- Real-money trading requires multiple separate env vars to arm — look at the live control module.

## Conventions

- `PYTHONPATH=.` — all imports are `from src.*`.
- Logging via `logging.getLogger(__name__)`, no print in library code.
- Feature hierarchy: **real observed value > NaN > NEVER median/default**. Only features producible at live inference time belong in training specs.
- UFCStats uses HTTP intentionally (they dropped TLS) — don't "fix" this to HTTPS.
- Odds features get training noise to prevent closing-odds leakage — don't bypass this.
- The CLOB client has thread-safety handling — check before making concurrent Polymarket calls.

## Don't

- **Don't commit `.env`, ledger files, or model `.pkl` artifacts** — these are runtime/sensitive data.
- **Don't add features to training specs without confirming they're available at live inference time** — this is the most important invariant in the project and it's easy to violate.

## Testing

- `pytest` with no special config. Many tests use monkeypatching.
- There are schema contract tests that enforce feature contract integrity — check them when touching model specs or feature builders.
- CI: compile check → artifact integrity → pytest → Docker build.
