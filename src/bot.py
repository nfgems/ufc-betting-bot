"""
Main bot orchestrator — ties together all components.

Usage:
    # Step 1: Train the model (run once, re-run when you have new data)
    python -m src.bot train

    # Step 2: Evaluate model performance
    python -m src.bot evaluate

    # Step 3: Run backtest to validate strategy
    python -m src.bot backtest

    # Step 4: Sensitivity analysis (find best parameters)
    python -m src.bot sensitivity

    # Step 5: Predict upcoming fights
    python -m src.bot predict

    # Step 6: Run live bot (dry run by default)
    python -m src.bot live --dry-run

    # Step 7: Run live bot with real money
    python -m src.bot live

    # Scrape latest data from UFCStats
    python -m src.bot scrape

    # Monitor upcoming events continuously (checks every N hours)
    python -m src.bot monitor

    # Track line movement (snapshot odds periodically)
    python -m src.bot track-lines

    # One-time check of all pre-fight signals for upcoming card
    python -m src.bot signals
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    INITIAL_BANKROLL,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    BLEND_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def cmd_scrape(args):
    """Scrape latest UFC data from UFCStats.com."""
    from src.data.scraper import scrape_all_fights, scrape_all_fighters

    logger.info("Starting UFC data scrape...")
    if args.fighters_only:
        scrape_all_fighters()
    elif args.fights_only:
        scrape_all_fights()
    else:
        scrape_all_fighters()
        scrape_all_fights()
    logger.info("Scraping complete.")


def cmd_train(args):
    """Load data, build features, and train models."""
    from src.data.kaggle_loader import load_kaggle_dataset, save_processed
    from src.features.build_features import build_features, save_features
    from src.model.train import train_all_models

    # Step 1: Load data
    logger.info("Loading data...")
    filepath = Path(args.data) if args.data else None
    fights_df = load_kaggle_dataset(filepath)
    save_processed(fights_df)

    # Step 2: Build features
    logger.info("Building features...")
    features_df = build_features(fights_df)
    save_features(features_df)

    # Step 3: Train models
    logger.info("Training models...")
    results = train_all_models(features_df)

    logger.info(f"Training complete. Models saved to models/")
    logger.info(f"Train size: {len(results['train_df'])}, Test size: {len(results['test_df'])}")


def cmd_evaluate(args):
    """Evaluate trained models on test set."""
    import pandas as pd
    from src.model.evaluate import compare_models, print_feature_importance

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    models = args.models.split(",") if args.models else ["xgboost", "logistic"]
    compare_models(test_df, model_names=models)
    print_feature_importance(model_name=models[0])


def cmd_backtest(args):
    """Run backtest on historical data. Defaults to walk-forward."""
    import pandas as pd
    from src.strategy.backtest import run_backtest, plot_backtest

    if args.static:
        # Static backtest: single train/test split
        logger.info("Running static backtest (single train/test split)...")
        test_path = PROCESSED_DATA_DIR / "test_set.csv"
        if not test_path.exists():
            logger.error("Test set not found. Run 'train' first.")
            return

        test_df = pd.read_csv(test_path, parse_dates=["event_date"])

        result = run_backtest(
            test_df,
            model_name=args.model,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
        )

        plot_backtest(result)
    else:
        # Walk-forward backtest (default): retrain every N months
        from src.features.build_features import build_features
        from src.data.kaggle_loader import load_kaggle_dataset
        from src.strategy.backtest import run_walkforward_backtest

        logger.info("Running walk-forward backtest (retraining every "
                     f"{args.retrain_months} months)...")
        logger.info("Use --static for single train/test split instead.")

        fights_df = load_kaggle_dataset()
        features_df = build_features(fights_df)

        result = run_walkforward_backtest(
            features_df,
            retrain_months=args.retrain_months,
            initial_train_years=args.initial_years,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
        )

        plot_backtest(result)


def cmd_backtest_compare(args):
    """Run comparison backtest: full model vs no-odds baseline."""
    import pandas as pd
    from src.strategy.backtest import run_comparison_backtest, plot_backtest

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    results = run_comparison_backtest(
        test_df,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )

    for name, result in results.items():
        plot_backtest(result)


def cmd_backfill_odds(args):
    """Backfill historical odds from The Odds API for backtesting."""
    import pandas as pd
    from src.data.historical_backfill import run_backfill

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    logger.info(f"Backfilling historical odds for {len(test_df)} fights...")
    logger.info(f"Unique event dates: {test_df['event_date'].nunique()}")
    logger.info(f"Snapshot offsets: {args.offsets} days before event")

    offsets = [int(x) for x in args.offsets.split(",")]
    result = run_backfill(test_df, offsets=offsets, resume=not args.fresh)

    logger.info(f"Backfill complete: {len(result)} total records")


def cmd_sensitivity(args):
    """Run sensitivity analysis across parameter combinations."""
    import pandas as pd
    from src.strategy.backtest import sensitivity_analysis

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])
    sensitivity_analysis(test_df, model_name=args.model)


def cmd_walkforward(args):
    """Run walk-forward backtest with periodic model retraining."""
    import pandas as pd
    from src.features.build_features import build_features
    from src.data.kaggle_loader import load_kaggle_dataset
    from src.strategy.backtest import run_walkforward_backtest, plot_backtest

    logger.info("Loading data and building features for walk-forward backtest...")
    fights_df = load_kaggle_dataset()
    features_df = build_features(fights_df)

    result = run_walkforward_backtest(
        features_df,
        retrain_months=args.retrain_months,
        initial_train_years=args.initial_years,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )

    plot_backtest(result)


def ensure_model_fresh(model_name: str = "xgboost"):
    """Auto-retrain models if they're older than MODEL_RETRAIN_MONTHS."""
    import time
    from src.config import MODELS_DIR, MODEL_RETRAIN_MONTHS

    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        logger.info(f"No model found at {model_path}. Training from scratch...")
        cmd_train(argparse.Namespace(data=None))
        return

    model_age_days = (time.time() - model_path.stat().st_mtime) / 86400
    max_age_days = MODEL_RETRAIN_MONTHS * 30

    if model_age_days > max_age_days:
        logger.info(
            f"Model is {model_age_days:.0f} days old (max: {max_age_days} days). "
            f"Auto-retraining..."
        )
        cmd_train(argparse.Namespace(data=None))
    else:
        logger.info(
            f"Model is {model_age_days:.0f} days old "
            f"(retrain threshold: {max_age_days} days). Using existing model."
        )


def cmd_predict(args):
    """Predict upcoming UFC fights using blended model-market approach."""
    from src.data.odds_client import OddsClient
    from src.model.predict import predict_fight
    from src.model.train import load_model
    from src.strategy.value import blend_probability, _passes_filters
    from src.features.build_features import get_fighter_ufc_fight_count
    from src.data.fighter_lookup import build_fight_features
    from src.data.line_tracker import detect_injury_or_cancellation
    from src.config import MIN_FIGHTER_FIGHTS

    logger.info("Fetching upcoming UFC odds...")
    odds_client = OddsClient()

    try:
        odds = odds_client.get_live_odds()
        odds_df = odds_client.odds_to_dataframe(odds)
        consensus = odds_client.get_consensus_odds(odds_df)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return

    if consensus.empty:
        logger.info("No upcoming UFC fights with odds found.")
        return

    logger.info(f"\nUpcoming UFC fights with predictions:")
    logger.info(f"{'='*80}")

    ensure_model_fresh(args.model)
    model_result = load_model(args.model)
    try:
        no_odds_result = load_model("xgboost_no_odds")
    except FileNotFoundError:
        no_odds_result = None

    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        market_a = fight["a_fair_prob_avg"]
        market_b = fight["b_fair_prob_avg"]

        # Auto-detect fighter experience
        a_fights = get_fighter_ufc_fight_count(fighter_a)
        b_fights = get_fighter_ufc_fight_count(fighter_b)

        exp_warnings = []
        if a_fights < MIN_FIGHTER_FIGHTS:
            exp_warnings.append(f"{fighter_a} ({a_fights} UFC fights)")
        if b_fights < MIN_FIGHTER_FIGHTS:
            exp_warnings.append(f"{fighter_b} ({b_fights} UFC fights)")

        # Check for injury/cancellation signals
        injury_tag = ""
        try:
            injury = detect_injury_or_cancellation(
                fighter_a, fighter_b,
                current_odds={"a_prob": market_a, "b_prob": market_b},
            )
            if injury["suspected"]:
                injury_tag = f"  [INJURY ALERT: {injury['reason']}]"
        except Exception:
            pass

        # Build full feature vector from live fighter stats + odds
        odds_features = {
            "a_implied_prob": market_a,
            "b_implied_prob": market_b,
            "diff_implied_prob": market_a - market_b,
        }
        features = build_fight_features(fighter_a, fighter_b, odds_features=odds_features)
        logger.info(f"  Built {sum(1 for v in features.values() if v is not None)} features for {fighter_a} vs {fighter_b}")

        try:
            pred = predict_fight(features, model_result=model_result)
        except Exception as e:
            logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")
            continue

        # No-odds model prediction for agreement
        no_odds_a = no_odds_b = None
        if no_odds_result:
            try:
                no_odds_pred = predict_fight(features, model_result=no_odds_result)
                no_odds_a = no_odds_pred["prob_a"]
                no_odds_b = no_odds_pred["prob_b"]
            except Exception:
                pass

        # Blend model with market
        blend_a = blend_probability(pred["prob_a"], market_a)
        blend_b = 1.0 - blend_a
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        # Check if value bet passes all filters (including fighter experience)
        value_a = edge_a >= MIN_EDGE_THRESHOLD and _passes_filters(
            blend_a, market_a, edge_a, fighter_a, no_odds_a,
            a_num_fights=a_fights, b_num_fights=b_fights,
        )
        value_b = edge_b >= MIN_EDGE_THRESHOLD and _passes_filters(
            blend_b, market_b, edge_b, fighter_b, no_odds_b,
            a_num_fights=a_fights, b_num_fights=b_fights,
        )
        value_tag = "  *** VALUE ***" if value_a or value_b else ""
        if exp_warnings:
            value_tag += f"  [LOW EXP: {', '.join(exp_warnings)}]"
        if injury_tag:
            value_tag += injury_tag

        no_odds_str = ""
        if no_odds_a is not None:
            no_odds_str = (
                f"\n  No-odds: {fighter_a} {no_odds_a:.1%} | "
                f"{fighter_b} {no_odds_b:.1%}"
            )

        logger.info(
            f"\n{fighter_a} vs {fighter_b}"
            f"\n  Market:  {fighter_a} {market_a:.1%} | "
            f"{fighter_b} {market_b:.1%} "
            f"({fight['num_bookmakers']:.0f} books)"
            f"\n  Model:   {fighter_a} {pred['prob_a']:.1%} | "
            f"{fighter_b} {pred['prob_b']:.1%}"
            f"{no_odds_str}"
            f"\n  Blended: {fighter_a} {blend_a:.1%} | "
            f"{fighter_b} {blend_b:.1%} "
            f"(w={BLEND_WEIGHT:.0%} model)"
            f"\n  Edge:    {fighter_a} {edge_a:+.1%} | {fighter_b} {edge_b:+.1%}"
            f"{value_tag}"
        )


def cmd_monitor(args):
    """Run continuous monitoring of upcoming UFC events."""
    from src.data.live_monitor import run_monitoring_pass
    from src.data.line_tracker import run_line_tracking_pass
    import time as _time

    interval_hours = args.interval
    logger.info(f"Starting continuous monitor (every {interval_hours} hours)")
    logger.info("Press Ctrl+C to stop")

    while True:
        try:
            # Run monitoring pass
            signals = run_monitoring_pass()

            # Also track lines
            line_summary = run_line_tracking_pass()

            logger.info(
                f"\nNext check in {interval_hours} hours. "
                f"Events tracked: {len(signals['events'])}, "
                f"Sharp moves: {line_summary.get('sharp_moves', 0)}"
            )

            _time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user.")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            _time.sleep(300)  # Wait 5 min on error, then retry


def cmd_track_lines(args):
    """Take a snapshot of current odds and analyze line movement."""
    from src.data.line_tracker import run_line_tracking_pass

    summary = run_line_tracking_pass()

    if summary.get("analyses"):
        logger.info(f"\nLine movement for {summary['fights_analyzed']} fights:")
        for fight, analysis in summary["analyses"].items():
            if analysis.get("opening_prob_a") is not None:
                logger.info(
                    f"  {fight}: "
                    f"Opened {analysis['opening_prob_a']:.1%} → "
                    f"Now {analysis['current_prob_a']:.1%} "
                    f"({analysis['movement']:+.1%} {analysis['direction']})"
                    f"{' *** SHARP ***' if analysis['is_sharp_move'] else ''}"
                    f"{' *** STEAM ***' if analysis['steam_move'] else ''}"
                )


def cmd_signals(args):
    """Check all pre-fight signals for upcoming events."""
    from src.data.live_monitor import run_monitoring_pass
    from src.data.prefight_signals import collect_prefight_signals

    signals = run_monitoring_pass()

    for event in signals.get("events", []):
        logger.info(f"\n{'='*60}")
        logger.info(f"Event: {event['title']} ({event['days_to_event']} days away)")
        logger.info(f"{'='*60}")

        for fight in event.get("fights", []):
            fa = fight["fighter_a"]
            fb = fight["fighter_b"]

            # Check if either fighter has signals
            a_short = any(
                r["new_fighter"].lower() == fa.lower()
                for r in signals.get("short_notice_replacements", [])
            )
            b_short = any(
                r["new_fighter"].lower() == fb.lower()
                for r in signals.get("short_notice_replacements", [])
            )
            a_missed = any(
                m["fighter"].lower() == fa.lower()
                for m in signals.get("missed_weights", [])
            )
            b_missed = any(
                m["fighter"].lower() == fb.lower()
                for m in signals.get("missed_weights", [])
            )

            a_over = next(
                (m["over_by"] for m in signals.get("missed_weights", [])
                 if m["fighter"].lower() == fa.lower()), 0.0
            )
            b_over = next(
                (m["over_by"] for m in signals.get("missed_weights", [])
                 if m["fighter"].lower() == fb.lower()), 0.0
            )

            fight_signals = collect_prefight_signals(
                fighter_a=fa,
                fighter_b=fb,
                event_title=event["title"],
                a_is_short_notice=a_short,
                b_is_short_notice=b_short,
                a_missed_weight=a_missed,
                b_missed_weight=b_missed,
                a_weight_over=a_over,
                b_weight_over=b_over,
            )

            logger.info(f"\n  {fa} vs {fb}:")
            if fight_signals["flags"]:
                for flag in fight_signals["flags"]:
                    logger.info(f"    * {flag}")
            else:
                logger.info(f"    No signals detected")


def cmd_positions(args):
    """Show current Polymarket positions and P&L."""
    from src.polymarket.monitor import PositionMonitor

    monitor = PositionMonitor()
    monitor.print_status()


def cmd_dashboard(args):
    """Run live-updating bet & P&L dashboard."""
    from src.polymarket.tracker import run_live_dashboard, auto_settle_from_polymarket, BetLedger
    from src.polymarket.client import ClobClientWrapper

    # Auto-settle any resolved markets first
    ledger = BetLedger()
    settled = auto_settle_from_polymarket(ledger)
    if settled:
        logger.info(f"Auto-settled {settled} bets from resolved markets")

    clob = None
    if not args.offline:
        try:
            clob = ClobClientWrapper()
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            logger.warning(f"Could not connect to CLOB (running offline): {e}")

    run_live_dashboard(
        clob_client=clob,
        refresh_seconds=args.refresh,
        include_dry_runs=not args.real_only,
    )


def cmd_web(args):
    """Launch the web dashboard."""
    from src.web.app import start_server
    from src.polymarket.client import ClobClientWrapper

    clob = None
    if not args.offline:
        try:
            clob = ClobClientWrapper()
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            logger.warning(f"Running offline (no CLOB): {e}")

    start_server(port=args.port, debug=args.debug, clob_client=clob)


def cmd_settle(args):
    """Manually settle a bet or auto-settle from Polymarket."""
    from src.polymarket.tracker import BetLedger, auto_settle_from_polymarket

    ledger = BetLedger()

    if args.auto:
        settled = auto_settle_from_polymarket(ledger)
        logger.info(f"Auto-settled {settled} bets")
        return

    if args.bet_id and args.result:
        won = args.result.lower() in ("win", "won", "w", "yes")
        ledger.settle_bet(args.bet_id, won)
        logger.info(f"Settled bet #{args.bet_id}: {'WON' if won else 'LOST'}")
    else:
        # Show open bets for manual settlement
        open_bets = ledger.open_bets
        if not open_bets:
            logger.info("No open bets to settle.")
            return

        logger.info(f"\nOpen bets ({len(open_bets)}):")
        for bet in open_bets:
            dry = " [DRY RUN]" if bet.get("dry_run") else ""
            logger.info(
                f"  #{bet['id']}: ${bet['amount']:.2f} on {bet['fighter']} "
                f"vs {bet['opponent']} @ {bet['price']:.4f}{dry}"
            )
        logger.info(
            "\nTo settle: python -m src.bot settle --bet-id <id> --result win/loss"
        )
        logger.info(
            "To auto-settle from Polymarket: python -m src.bot settle --auto"
        )


def cmd_live(args):
    """Run the live betting bot."""
    from src.data.odds_client import OddsClient
    from src.model.predict import predict_fight
    from src.model.train import load_model
    from src.polymarket.markets import get_ufc_fight_markets
    from src.polymarket.executor import OrderExecutor
    from src.polymarket.client import ClobClientWrapper
    from src.strategy.bankroll import BankrollManager
    from src.data.line_tracker import get_line_movement_features, detect_injury_or_cancellation
    from src.features.build_features import get_fighter_ufc_fight_count
    from src.data.fighter_lookup import build_fight_features
    from src.config import MIN_FIGHTER_FIGHTS, INJURY_BLOCK_BETS
    import pandas as pd

    dry_run = args.dry_run
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting bot in {mode} mode...")

    # Initialize components
    from src.polymarket.monitor import PositionMonitor

    bankroll = BankrollManager(initial_bankroll=args.bankroll)
    clob = None if dry_run else ClobClientWrapper()
    executor = OrderExecutor(bankroll=bankroll, clob_client=clob, dry_run=dry_run)
    monitor = PositionMonitor(clob_client=clob)
    ensure_model_fresh(args.model)
    model_result = load_model(args.model)

    # 1. Fetch bookmaker consensus odds from The Odds API
    logger.info("Fetching bookmaker odds from The Odds API...")
    odds_client = OddsClient()
    try:
        raw_odds = odds_client.get_live_odds()
        odds_df = odds_client.odds_to_dataframe(raw_odds)
        consensus = odds_client.get_consensus_odds(odds_df)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        logger.info("Set ODDS_API_KEY in .env. Get a free key at https://the-odds-api.com")
        return

    if consensus.empty:
        logger.info("No upcoming UFC fights with bookmaker odds found.")
        return

    logger.info(f"Got bookmaker consensus for {len(consensus)} fights")

    # 2. Get Polymarket markets (execution venue)
    logger.info("Fetching Polymarket UFC markets...")
    markets = get_ufc_fight_markets()
    if markets.empty:
        logger.info("No active UFC markets found on Polymarket.")
        return

    logger.info(f"Found {len(markets)} active Polymarket UFC markets")

    # 3. For each fight, generate ML predictions using bookmaker odds as features,
    #    then compare model output vs Polymarket prices to find value
    logger.info("Generating model predictions and scanning for value...")

    prediction_rows = []
    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]

        # Auto-detect fighter experience
        a_fights = get_fighter_ufc_fight_count(fighter_a)
        b_fights = get_fighter_ufc_fight_count(fighter_b)

        if a_fights < MIN_FIGHTER_FIGHTS or b_fights < MIN_FIGHTER_FIGHTS:
            low_exp = []
            if a_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_a} ({a_fights} fights)")
            if b_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_b} ({b_fights} fights)")
            logger.info(
                f"\n  Skipping {fighter_a} vs {fighter_b}: "
                f"insufficient UFC experience — {', '.join(low_exp)}"
            )
            continue

        # Check for injury/cancellation signals before committing resources
        try:
            injury = detect_injury_or_cancellation(
                fighter_a, fighter_b,
                current_odds={
                    "a_prob": fight["a_fair_prob_avg"],
                    "b_prob": fight["b_fair_prob_avg"],
                },
            )
            if injury["suspected"]:
                if injury["severity"] == "block" and INJURY_BLOCK_BETS:
                    logger.warning(
                        f"\n  SKIPPING {fighter_a} vs {fighter_b}: "
                        f"{injury['reason']}"
                    )
                    continue
                elif injury["severity"] == "warning":
                    logger.info(
                        f"\n  WARNING for {fighter_a} vs {fighter_b}: "
                        f"{injury['reason']}"
                    )
        except Exception:
            pass

        # Build full feature vector from live fighter stats + odds
        odds_features = {
            "a_implied_prob": fight["a_fair_prob_avg"],
            "b_implied_prob": fight["b_fair_prob_avg"],
            "diff_implied_prob": fight["a_fair_prob_avg"] - fight["b_fair_prob_avg"],
        }

        # Add line movement features if we have tracking history
        try:
            line_features = get_line_movement_features(fighter_a, fighter_b)
            odds_features.update(line_features)
        except Exception:
            pass

        features = build_fight_features(fighter_a, fighter_b, odds_features=odds_features)
        logger.info(f"  Built {sum(1 for v in features.values() if v is not None)} features for {fighter_a} vs {fighter_b}")

        # Run ML model prediction
        try:
            pred = predict_fight(features, model_result=model_result)
        except Exception as e:
            logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")
            continue

        logger.info(
            f"\n  {fighter_a} vs {fighter_b}:"
            f"\n    Bookmakers: {fighter_a} {fight['a_fair_prob_avg']:.1%} | "
            f"{fighter_b} {fight['b_fair_prob_avg']:.1%}"
            f"\n    Model:      {fighter_a} {pred['prob_a']:.1%} | "
            f"{fighter_b} {pred['prob_b']:.1%}"
        )

        prediction_rows.append({
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "prob_a": pred["prob_a"],
            "prob_b": pred["prob_b"],
            "confidence": pred["confidence"],
            "event_date": fight.get("commence_time"),
            # Bookmaker consensus used as the market baseline for edge detection
            "a_market_prob": fight["a_fair_prob_avg"],
            "b_market_prob": fight["b_fair_prob_avg"],
        })

    if not prediction_rows:
        logger.info("No predictions generated.")
        return

    predictions = pd.DataFrame(prediction_rows)

    # 4. Execute value bets on Polymarket
    #    The executor matches predictions to Polymarket markets by fighter name
    #    and overrides market_prob with Polymarket's actual prices for execution
    orders = executor.execute_value_bets(
        predictions,
        markets,
        min_edge=args.min_edge,
    )

    # 5. Summary
    order_log = executor.get_order_log()
    if not order_log.empty:
        logger.info(f"\n{'='*60}")
        logger.info(f"EXECUTION SUMMARY ({mode})")
        logger.info(f"{'='*60}")
        logger.info(f"Orders: {len(order_log)}")
        logger.info(f"Total wagered: ${order_log['bet_size_usd'].sum():.2f}")
        logger.info(f"\n{order_log[['fighter', 'bet_size_usd', 'price', 'edge', 'status']].to_string()}")

        # Log orders and snapshot positions
        for _, order in order_log.iterrows():
            monitor.log_order(order.to_dict())
    else:
        logger.info("No value bets found — market is efficient for these fights.")

    # Cancel stale orders and show position status
    if not dry_run:
        monitor.cancel_stale_orders(max_age_hours=24.0)
        monitor.log_positions_snapshot()
        monitor.print_status()

    stats = bankroll.get_stats()
    logger.info(f"\nBankroll: ${stats['bankroll']:.2f} / ${stats['initial_bankroll']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="UFC Betting Bot — Predict fights and bet on Polymarket"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape UFC data")
    scrape_parser.add_argument("--fighters-only", action="store_true")
    scrape_parser.add_argument("--fights-only", action="store_true")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction models")
    train_parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--models", type=str, default="xgboost,logistic")

    # Backtest command (defaults to walk-forward)
    bt_parser = subparsers.add_parser("backtest",
                                       help="Run strategy backtest (walk-forward by default)")
    bt_parser.add_argument("--static", action="store_true",
                           help="Use static single train/test split instead of walk-forward")
    bt_parser.add_argument("--model", type=str, default="xgboost")
    bt_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    bt_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    bt_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)
    bt_parser.add_argument("--retrain-months", type=int, default=6,
                           help="Months between model retraining (default: 6)")
    bt_parser.add_argument("--initial-years", type=int, default=5,
                           help="Years of initial training data (default: 5)")

    # Sensitivity command
    sens_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    sens_parser.add_argument("--model", type=str, default="xgboost")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict upcoming fights")
    pred_parser.add_argument("--model", type=str, default="xgboost")

    # Live command
    live_parser = subparsers.add_parser("live", help="Run live bot")
    live_parser.add_argument("--dry-run", action="store_true", default=True,
                             help="Dry run mode (default: True)")
    live_parser.add_argument("--real", action="store_true",
                             help="Run with real money (disables dry run)")
    live_parser.add_argument("--model", type=str, default="xgboost")
    live_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    live_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)

    # Monitor command
    mon_parser = subparsers.add_parser("monitor", help="Continuous event monitoring")
    mon_parser.add_argument("--interval", type=float, default=6.0,
                            help="Hours between checks (default: 6)")

    # Backtest compare command
    btc_parser = subparsers.add_parser("backtest-compare",
                                        help="Compare full model vs no-odds baseline")
    btc_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    btc_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    btc_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)

    # Backfill odds command
    bf_parser = subparsers.add_parser("backfill-odds",
                                      help="Backfill historical odds from The Odds API")
    bf_parser.add_argument("--offsets", type=str, default="7,3,1",
                           help="Comma-separated day offsets (default: 7,3,1)")
    bf_parser.add_argument("--fresh", action="store_true",
                           help="Start fresh (ignore existing backfill data)")

    # Walk-forward backtest command
    wf_parser = subparsers.add_parser("walkforward",
                                       help="Walk-forward backtest with periodic retraining")
    wf_parser.add_argument("--retrain-months", type=int, default=6,
                            help="Months between model retraining (default: 6)")
    wf_parser.add_argument("--initial-years", type=int, default=5,
                            help="Years of initial training data (default: 5)")
    wf_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    wf_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    wf_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)

    # Positions command
    subparsers.add_parser("positions", help="Show current Polymarket positions and P&L")

    # Web dashboard command
    web_parser = subparsers.add_parser("web",
                                        help="Launch web dashboard (local)")
    web_parser.add_argument("--port", type=int, default=5050,
                             help="Port to run on (default: 5050)")
    web_parser.add_argument("--offline", action="store_true",
                             help="Don't connect to Polymarket for live prices")
    web_parser.add_argument("--debug", action="store_true",
                             help="Run Flask in debug mode")

    # Dashboard command (terminal)
    dash_parser = subparsers.add_parser("dashboard",
                                         help="Terminal-based live dashboard")
    dash_parser.add_argument("--refresh", type=int, default=30,
                              help="Refresh interval in seconds (default: 30)")
    dash_parser.add_argument("--offline", action="store_true",
                              help="Don't fetch live prices from Polymarket")
    dash_parser.add_argument("--real-only", action="store_true",
                              help="Only show real bets (exclude dry runs)")

    # Settle command
    settle_parser = subparsers.add_parser("settle",
                                           help="Settle bets (manual or auto)")
    settle_parser.add_argument("--auto", action="store_true",
                                help="Auto-settle from Polymarket resolved markets")
    settle_parser.add_argument("--bet-id", type=int,
                                help="Bet ID to settle")
    settle_parser.add_argument("--result", type=str,
                                help="Result: win or loss")

    # Track lines command
    subparsers.add_parser("track-lines", help="Snapshot odds and analyze movement")

    # Signals command
    subparsers.add_parser("signals", help="Check pre-fight signals for upcoming events")

    args = parser.parse_args()

    if args.command == "live" and args.real:
        args.dry_run = False

    commands = {
        "scrape": cmd_scrape,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "backtest": cmd_backtest,
        "backtest-compare": cmd_backtest_compare,
        "backfill-odds": cmd_backfill_odds,
        "sensitivity": cmd_sensitivity,
        "walkforward": cmd_walkforward,
        "predict": cmd_predict,
        "live": cmd_live,
        "positions": cmd_positions,
        "web": cmd_web,
        "dashboard": cmd_dashboard,
        "settle": cmd_settle,
        "monitor": cmd_monitor,
        "track-lines": cmd_track_lines,
        "signals": cmd_signals,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
