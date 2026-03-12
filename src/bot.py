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
from datetime import datetime
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
    from src.data.kaggle_loader import load_kaggle_dataset
    from src.features.build_features import build_features
    from src.strategy.backtest import (
        run_comparison_backtest,
        run_walkforward_strategy_comparison,
        plot_backtest,
    )

    if args.walkforward:
        logger.info(
            "Running walk-forward comparison "
            f"(retraining every {args.retrain_months} months)..."
        )
        features_path = PROCESSED_DATA_DIR / "features.csv"
        if features_path.exists():
            logger.info(f"Loading cached features from {features_path}")
            features_df = pd.read_csv(features_path, parse_dates=["event_date"])
        else:
            logger.info("No cached features found. Building features from Kaggle dataset...")
            fights_df = load_kaggle_dataset()
            features_df = build_features(fights_df)

        comparison = run_walkforward_strategy_comparison(
            features_df,
            retrain_months=args.retrain_months,
            initial_train_years=args.initial_years,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
        )

        if not comparison["summary"].empty:
            logger.info("\nWalk-forward strategy summary:")
            logger.info(comparison["summary"].to_string(index=False))

        artifacts = comparison.get("artifacts", {})
        if artifacts:
            logger.info("\nArtifacts:")
            for label, path in artifacts.items():
                logger.info(f"  {label}: {path}")
        return

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


def _blend_tennis_probability(model_prob: float, bookmaker_prob: float, weight: float) -> float:
    """Blend model probability with bookmaker consensus for tennis dry runs."""
    return (weight * float(model_prob)) + ((1.0 - weight) * float(bookmaker_prob))


def _fractional_kelly_stake(probability: float, price: float, bankroll: float, fraction: float) -> float:
    """Return a fractional Kelly stake using Polymarket-style share prices."""
    if price is None or probability is None:
        return 0.0
    if price <= 0.0 or price >= 1.0:
        return 0.0

    b = (1.0 - price) / price
    if b <= 0.0:
        return 0.0

    q = 1.0 - probability
    edge_fraction = ((b * probability) - q) / b
    return max(0.0, edge_fraction) * bankroll * fraction


def _build_tennis_prediction_frame(model_name: str = "surface_elo"):
    """Build live tennis predictions from bookmaker odds and historical ATP/WTA data."""
    from src.data.tennis_data import load_processed_tennis_data
    from src.data.tennis_odds import fetch_live_tennis_consensus
    from src.features.tennis_features import build_live_tennis_features
    from src.model.tennis_model import load_tennis_model, predict_tennis_batch

    try:
        history_df = load_processed_tennis_data()
    except FileNotFoundError:
        logger.error("Processed tennis data not found. Run 'tennis-train' first.")
        return None

    try:
        consensus = fetch_live_tennis_consensus()
    except Exception as exc:
        logger.error(f"Failed to fetch live tennis odds: {exc}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return None

    if consensus.empty:
        logger.info("No live ATP/WTA singles odds found.")
        return None

    try:
        model_result = load_tennis_model(model_name)
    except FileNotFoundError:
        logger.error("Tennis model not found. Run 'tennis-train' first.")
        return None

    live_features = build_live_tennis_features(consensus, history_df)
    predictions = predict_tennis_batch(live_features, model_result=model_result)
    return predictions


def cmd_tennis_discover(args):
    """Discover active tennis sport keys, live bookmaker matches, and Polymarket markets."""
    from src.data.tennis_odds import discover_active_tennis_sports, fetch_live_tennis_consensus
    from src.polymarket.tennis_markets import discover_tennis_markets, match_tennis_markets

    try:
        sports = discover_active_tennis_sports()
        consensus = fetch_live_tennis_consensus(sports=sports)
    except Exception as exc:
        logger.error(f"Failed to discover live tennis sports: {exc}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return

    markets = discover_tennis_markets(active_tennis_sports=sports)
    matched = match_tennis_markets(consensus, markets)

    logger.info("Active tennis sport keys: %s", len(sports))
    for sport in sports[:10]:
        logger.info(
            "  %s | %s | %s",
            sport.get("sport_key"),
            sport.get("tour", "").upper(),
            sport.get("tournament_name"),
        )

    logger.info("Live bookmaker tennis matches: %s", len(consensus))
    logger.info("Discovered active tennis Polymarket markets: %s", len(markets))
    logger.info("Current bookmaker/Polymarket tennis matches: %s", len(matched))

    for _, row in matched.head(10).iterrows():
        logger.info(
            "  MATCHED: %s vs %s | %s | %s",
            row["fighter_a"],
            row["fighter_b"],
            row.get("tour", "").upper(),
            row.get("market_event_title", row.get("event_title", "")),
        )


def cmd_tennis_train(args):
    """Download tennis history, build leak-free features, and train the Stage 1 baseline."""
    import pandas as pd
    from src.data.tennis_data import prepare_tennis_data
    from src.features.tennis_features import build_tennis_features, save_tennis_features
    from src.model.tennis_model import train_tennis_model

    logger.info("Preparing ATP/WTA singles history...")
    matches_df = prepare_tennis_data(
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force_download,
    )
    if matches_df.empty:
        logger.error("No tennis history loaded.")
        return

    logger.info("Building tennis features...")
    features_df = build_tennis_features(matches_df)
    features_path = PROCESSED_DATA_DIR / "tennis" / "features.csv"
    save_tennis_features(features_df, str(features_path))

    logger.info("Training Stage 1 tennis model...")
    model_result = train_tennis_model(features_df, model_name=args.model)

    evaluation_dir = PROCESSED_DATA_DIR / "tennis"
    folds = model_result.get("evaluation_folds")
    if isinstance(folds, pd.DataFrame) and not folds.empty:
        folds.to_csv(evaluation_dir / "walkforward_folds.csv", index=False)

    predictions = model_result.get("evaluation_predictions")
    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        predictions.to_csv(evaluation_dir / "walkforward_predictions.csv", index=False)

    metrics = model_result.get("evaluation_metrics", {})
    calibration = metrics.get("calibration")
    if isinstance(calibration, pd.DataFrame) and not calibration.empty:
        calibration.to_csv(evaluation_dir / "calibration.csv", index=False)

    logger.info("Tennis training complete. Model saved to models/tennis/")
    if metrics:
        logger.info("Walk-forward log loss: %.4f", metrics.get("log_loss", float("nan")))
        logger.info("Walk-forward Brier score: %.4f", metrics.get("brier_score", float("nan")))
        if isinstance(calibration, pd.DataFrame):
            logger.info("Calibration rows: %s", len(calibration))


def cmd_tennis_predict(args):
    """Predict live ATP/WTA singles matches using the Stage 1 baseline."""
    from src.config import TENNIS_BLEND_WEIGHT

    predictions = _build_tennis_prediction_frame(model_name=args.model)
    if predictions is None or predictions.empty:
        return

    logger.info("\nLive tennis predictions:")
    logger.info("%s", "=" * 80)

    for _, row in predictions.sort_values(["commence_time", "tour", "fighter_a"]).iterrows():
        blend_a = _blend_tennis_probability(
            row["prob_a"],
            row["a_fair_prob_avg"],
            TENNIS_BLEND_WEIGHT,
        )
        blend_b = 1.0 - blend_a
        logger.info(
            "\n%s vs %s [%s]",
            row["fighter_a"],
            row["fighter_b"],
            str(row.get("tour", "")).upper(),
        )
        logger.info(
            "  Tournament: %s | Sport key: %s",
            row.get("tournament_name", ""),
            row.get("sport_key", ""),
        )
        logger.info(
            "  Bookmakers: %s %.1f%% | %s %.1f%% (%s books)",
            row["fighter_a"],
            row["a_fair_prob_avg"] * 100,
            row["fighter_b"],
            row["b_fair_prob_avg"] * 100,
            int(row["num_bookmakers"]),
        )
        logger.info(
            "  Model:      %s %.1f%% | %s %.1f%%",
            row["fighter_a"],
            row["prob_a"] * 100,
            row["fighter_b"],
            row["prob_b"] * 100,
        )
        logger.info(
            "  Blended:    %s %.1f%% | %s %.1f%% (w=%.0f%% model)",
            row["fighter_a"],
            blend_a * 100,
            row["fighter_b"],
            blend_b * 100,
            TENNIS_BLEND_WEIGHT * 100,
        )


def cmd_tennis_live(args):
    """Run the Stage 1 tennis dry-run pipeline without placing orders."""
    from src.config import (
        INITIAL_BANKROLL,
        TENNIS_BLEND_WEIGHT,
        TENNIS_KELLY_FRACTION,
        TENNIS_MIN_EDGE_THRESHOLD,
    )
    from src.polymarket.tennis_markets import discover_tennis_markets, match_tennis_markets

    if not args.dry_run:
        logger.error("Real-money tennis trading is not implemented. Use 'tennis-live --dry-run'.")
        return

    predictions = _build_tennis_prediction_frame(model_name=args.model)
    if predictions is None or predictions.empty:
        return

    markets = discover_tennis_markets()
    if markets.empty:
        logger.info("No active tennis Polymarket markets found.")
        return

    matched = match_tennis_markets(predictions, markets)
    if matched.empty:
        logger.info("No live tennis matches could be matched to Polymarket markets.")
        return

    min_edge = args.min_edge if args.min_edge is not None else TENNIS_MIN_EDGE_THRESHOLD
    bankroll_basis = INITIAL_BANKROLL
    opportunities = []

    logger.info("Running tennis dry-run on %s matched markets...", len(matched))
    for _, row in matched.sort_values(["commence_time", "tour", "fighter_a"]).iterrows():
        poly_a = row.get("price_yes")
        poly_b = row.get("price_no")
        if poly_a is None or poly_b is None:
            continue
        if not (0.0 < poly_a < 1.0 and 0.0 < poly_b < 1.0):
            continue

        blend_a = _blend_tennis_probability(
            row["prob_a"],
            row["a_fair_prob_avg"],
            TENNIS_BLEND_WEIGHT,
        )
        blend_b = 1.0 - blend_a
        edge_a = blend_a - poly_a
        edge_b = blend_b - poly_b

        if edge_a >= edge_b:
            bet_on = row["fighter_a"]
            market_price = poly_a
            blended_prob = blend_a
            edge = edge_a
            bet_side = "a"
        else:
            bet_on = row["fighter_b"]
            market_price = poly_b
            blended_prob = blend_b
            edge = edge_b
            bet_side = "b"

        stake = _fractional_kelly_stake(
            probability=blended_prob,
            price=market_price,
            bankroll=bankroll_basis,
            fraction=TENNIS_KELLY_FRACTION,
        )

        if edge < min_edge or stake <= 0.0:
            continue

        decimal_odds = 1.0 / market_price
        opportunities.append(
            {
                "tour": row.get("tour", ""),
                "fighter_a": row["fighter_a"],
                "fighter_b": row["fighter_b"],
                "bet_on": bet_on,
                "bet_side": bet_side,
                "market_price": market_price,
                "decimal_odds": decimal_odds,
                "bookmaker_prob": row["a_fair_prob_avg"] if bet_side == "a" else row["b_fair_prob_avg"],
                "model_prob": row["prob_a"] if bet_side == "a" else row["prob_b"],
                "blended_prob": blended_prob,
                "edge": edge,
                "stake_usd": stake,
                "market_id": row.get("market_id"),
                "token_id_yes": row.get("token_id_yes"),
                "token_id_no": row.get("token_id_no"),
                "sport_key": row.get("sport_key"),
                "tournament_name": row.get("tournament_name"),
            }
        )

    if not opportunities:
        logger.info("No tennis dry-run opportunities cleared the %.1f%% edge threshold.", min_edge * 100)
        return

    for opportunity in sorted(opportunities, key=lambda item: item["edge"], reverse=True):
        logger.info(
            "\nDRY RUN: %s vs %s [%s]",
            opportunity["fighter_a"],
            opportunity["fighter_b"],
            str(opportunity.get("tour", "")).upper(),
        )
        logger.info(
            "  Tournament: %s | Sport key: %s",
            opportunity.get("tournament_name", ""),
            opportunity.get("sport_key", ""),
        )
        logger.info(
            "  Hypothetical bet: %s @ %.3f (%.2f USD, %.2f decimal odds)",
            opportunity["bet_on"],
            opportunity["market_price"],
            opportunity["stake_usd"],
            opportunity["decimal_odds"],
        )
        logger.info(
            "  Model %.1f%% | Blended %.1f%% | Edge %+0.1f%%",
            opportunity["model_prob"] * 100,
            opportunity["blended_prob"] * 100,
            opportunity["edge"] * 100,
        )
        logger.info("  Polymarket market id: %s", opportunity.get("market_id"))

    logger.info(
        "Tennis dry-run complete. Hypothetical bets: %s | Bankroll basis: %.2f USD",
        len(opportunities),
        bankroll_basis,
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
                    f"Opened {analysis['opening_prob_a']:.1%} -> "
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
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    # Auto-settle any resolved markets across all trader ledgers
    settled = 0
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
        if Path(path).exists():
            ledger = BetLedger(path=path)
            settled += auto_settle_from_polymarket(ledger)
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
    from src.polymarket.tracker import (
        BetLedger,
        auto_settle_from_polymarket,
        load_all_trader_ledgers,
        resolve_merged_bet_reference,
    )
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    if args.auto:
        settled = 0
        for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
            if Path(path).exists():
                ledger = BetLedger(path=path)
                settled += auto_settle_from_polymarket(ledger)
        logger.info(f"Auto-settled {settled} bets")
        return

    if args.bet_id and args.result:
        won = args.result.lower() in ("win", "won", "w", "yes")
        target = resolve_merged_bet_reference(args.bet_id, require_open=True)
        if target is None:
            existing = resolve_merged_bet_reference(args.bet_id, require_open=False)
            if existing is not None:
                logger.warning(f"Bet #{args.bet_id} is not open")
            else:
                logger.warning(f"Bet #{args.bet_id} not found in any trader ledger")
        else:
            ledger = BetLedger(path=target["ledger_path"])
            result = ledger.settle_bet(target["original_id"], won)
            if result.ok:
                logger.info(f"Settled bet #{args.bet_id}: {'WON' if won else 'LOST'}")
            elif result.status == "not_open":
                logger.warning(f"Bet #{args.bet_id} is no longer open")
            else:
                logger.warning(f"Bet #{args.bet_id} no longer exists")
    else:
        # Show open bets for manual settlement
        ledger = load_all_trader_ledgers()
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



def cmd_duo_live(args):
    """Run duo traders (S+C) with Single Trader evaluating first, Conviction on remainder."""
    from src.data.odds_client import OddsClient
    from src.model.predict import predict_fight
    from src.model.train import load_model
    from src.polymarket.markets import get_ufc_fight_markets
    from src.polymarket.client import ClobClientWrapper
    from src.strategy.duo_trader import run_duo_traders
    from src.data.line_tracker import get_line_movement_features, detect_injury_or_cancellation
    from src.features.build_features import get_fighter_ufc_fight_count
    from src.data.fighter_lookup import build_fight_features
    from src.config import MIN_FIGHTER_FIGHTS, INJURY_BLOCK_BETS
    import pandas as pd

    dry_run = args.dry_run
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting DUO TRADER bot in {mode} mode...")

    clob = None if dry_run else ClobClientWrapper()

    ensure_model_fresh(args.model)
    model_result = load_model(args.model)
    try:
        no_odds_result = load_model("xgboost_no_odds")
    except FileNotFoundError:
        no_odds_result = None

    # Set up SHAP explainer for prediction explanations
    import numpy as np
    shap_explainer = None
    shap_base_value = None
    try:
        import shap
        raw_model = model_result.get("raw_model")
        if raw_model is not None:
            shap_explainer = shap.TreeExplainer(raw_model)
            ev = shap_explainer.expected_value
            ev = np.atleast_1d(ev)
            shap_base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
    except ImportError:
        logger.info("shap not installed — predictions page will use feature highlights only")
    except Exception as e:
        logger.warning(f"Failed to create SHAP explainer: {e}")

    # Extract feature cols/medians and global importance for cache enrichment
    _feat_cols = model_result["feature_cols"]
    _col_medians = model_result["col_medians"]
    _global_importance = sorted(
        model_result.get("feature_importance", {}).items(),
        key=lambda x: x[1], reverse=True,
    )[:25]

    def _feature_display_name(col: str) -> str:
        """Convert feature column name to human-readable display name."""
        overrides = {
            "diff_elo": "Elo Rating",
            "diff_roll_slpm": "Strikes Landed/Min",
            "diff_roll_sapm": "Strikes Absorbed/Min",
            "diff_roll_str_acc": "Striking Accuracy %",
            "diff_roll_str_def": "Strike Defense %",
            "diff_roll_td_avg": "Takedowns/Fight",
            "diff_roll_td_acc": "Takedown Accuracy %",
            "diff_roll_td_def": "Takedown Defense %",
            "diff_roll_sub_avg": "Submissions/Fight",
            "diff_roll_kd": "Knockdowns/Fight",
            "diff_roll_won": "Recent Win Rate",
            "diff_current_win_streak": "Win Streak",
            "diff_lose_streak": "Losing Streak",
            "diff_num_fights": "UFC Experience",
            "diff_total_rounds": "Rounds Fought",
            "diff_age": "Age",
            "diff_height": "Height (cm)",
            "diff_reach": "Reach (cm)",
            "diff_weight": "Weight",
            "diff_ko_rate": "KO Win Rate",
            "diff_sub_rate": "Submission Win Rate",
            "diff_dec_rate": "Decision Win Rate",
            "diff_win_pct": "Overall Win %",
            "diff_implied_prob": "Betting Odds",
            "diff_wc_rank": "Division Ranking",
            "diff_pfp_rank": "P4P Ranking",
            "diff_strike_diff": "Net Striking",
            "diff_striker_edge": "Striking Advantage",
            "diff_grappler_edge": "Grappling Advantage",
            "diff_days_since_last_fight": "Days Since Last Fight",
            "diff_cage_rust": "Ring Rust",
            "diff_title_bouts": "Title Fight Experience",
            "diff_opp_td_avg": "Opponent Takedowns Faced",
            "diff_opp_sub_avg": "Opponent Subs Faced",
            "diff_opp_str_acc": "Opponent Striking Accuracy",
            "diff_ko_odds_prob": "KO Finish Likelihood",
            "diff_dec_odds_prob": "Decision Likelihood",
        }
        if col in overrides:
            return overrides[col]

        # Handle a_/b_ prefixed features (individual fighter stats)
        for prefix, label in [("a_", "A "), ("b_", "B ")]:
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                diff_key = f"diff_{suffix}"
                if diff_key in overrides:
                    return label + overrides[diff_key]
                break

        # Fallback: clean up the raw name
        name = col.replace("diff_", "").replace("a_", "A ").replace("b_", "B ").replace("roll_", "").replace("opp_", "Opp ").replace("_", " ").title()
        return name

    # 1. Fetch bookmaker consensus odds
    logger.info("Fetching bookmaker odds from The Odds API...")
    odds_client = OddsClient()
    try:
        raw_odds = odds_client.get_live_odds()
        odds_df = odds_client.odds_to_dataframe(raw_odds)
        consensus = odds_client.get_consensus_odds(odds_df)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return

    if consensus.empty:
        logger.info("No upcoming UFC fights with bookmaker odds found.")
        return

    logger.info(f"Got bookmaker consensus for {len(consensus)} fights")

    # 2. Get Polymarket markets
    logger.info("Fetching Polymarket UFC markets...")
    markets = get_ufc_fight_markets()
    if markets.empty:
        logger.info("No active UFC markets found on Polymarket.")
        return

    logger.info(f"Found {len(markets)} active Polymarket UFC markets")

    # 3. Generate predictions (same for both traders — they differ only in blend weight)
    logger.info("Generating model predictions...")
    prediction_rows = []
    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]

        a_fights = get_fighter_ufc_fight_count(fighter_a)
        b_fights = get_fighter_ufc_fight_count(fighter_b)

        low_experience = a_fights < MIN_FIGHTER_FIGHTS or b_fights < MIN_FIGHTER_FIGHTS
        if low_experience:
            low_exp = []
            if a_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_a} ({a_fights} fights)")
            if b_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_b} ({b_fights} fights)")
            logger.info(
                f"\n  Low experience: {', '.join(low_exp)} — "
                f"prediction generated but trading filters may skip"
            )

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
                        f"\n  SKIPPING {fighter_a} vs {fighter_b}: {injury['reason']}"
                    )
                    continue
                elif injury["severity"] == "warning":
                    logger.info(
                        f"\n  WARNING for {fighter_a} vs {fighter_b}: {injury['reason']}"
                    )
        except Exception:
            pass

        odds_features = {
            "a_implied_prob": fight["a_fair_prob_avg"],
            "b_implied_prob": fight["b_fair_prob_avg"],
            "diff_implied_prob": fight["a_fair_prob_avg"] - fight["b_fair_prob_avg"],
        }

        line_features = {}
        try:
            line_features = get_line_movement_features(fighter_a, fighter_b)
            odds_features.update(line_features)
        except Exception:
            pass

        features = build_fight_features(fighter_a, fighter_b, odds_features=odds_features)
        logger.info(f"  Built {sum(1 for v in features.values() if v is not None)} features for {fighter_a} vs {fighter_b}")

        try:
            pred = predict_fight(features, model_result=model_result)
        except Exception as e:
            logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")
            continue

        # No-odds model prediction for Trader C conviction checks
        no_odds_a = no_odds_b = None
        if no_odds_result:
            try:
                no_odds_pred = predict_fight(features, model_result=no_odds_result)
                no_odds_a = no_odds_pred["prob_a"]
                no_odds_b = no_odds_pred["prob_b"]
            except Exception as e:
                logger.warning(
                    f"No-odds prediction failed for {fighter_a} vs {fighter_b}: {e} "
                    f"— model agreement filter will block this fight"
                )

        no_odds_str = ""
        if no_odds_a is not None:
            no_odds_str = (
                f"\n    No-odds:    {fighter_a} {no_odds_a:.1%} | "
                f"{fighter_b} {no_odds_b:.1%}"
            )

        logger.info(
            f"\n  {fighter_a} vs {fighter_b}:"
            f"\n    Bookmakers: {fighter_a} {fight['a_fair_prob_avg']:.1%} | "
            f"{fighter_b} {fight['b_fair_prob_avg']:.1%}"
            f"\n    Model:      {fighter_a} {pred['prob_a']:.1%} | "
            f"{fighter_b} {pred['prob_b']:.1%}"
            f"{no_odds_str}"
        )

        # --- Compute SHAP values for this fight ---
        # Reconstruct the same feature vector that predict_fight() uses,
        # including _missing indicator columns.
        fight_shap_values = []
        if shap_explainer is not None:
            try:
                _base_cols = [c for c in _feat_cols if not c.endswith("_missing")]
                _missing_cols = [c for c in _feat_cols if c.endswith("_missing")]

                base_values = [features.get(col, np.nan) for col in _base_cols]
                X_base = np.array([base_values])

                # Generate missing indicators (1 if NaN, 0 otherwise)
                indicators = [float(np.isnan(X_base[0, i]))
                              for i, col in enumerate(_base_cols)
                              if f"{col}_missing" in _missing_cols]

                # Fill NaNs with training medians
                for i in range(X_base.shape[1]):
                    if np.isnan(X_base[0, i]):
                        X_base[0, i] = _col_medians[i] if i < len(_col_medians) and not np.isnan(_col_medians[i]) else 0.0

                # Combine base + indicators (same as predict_fight)
                if indicators:
                    X = np.column_stack([X_base, np.array([indicators])])
                else:
                    X = X_base

                sv = shap_explainer.shap_values(X)
                # Handle both list (binary) and array output
                if isinstance(sv, list):
                    shap_arr = sv[1][0]  # class 1 = fighter A wins
                else:
                    shap_arr = sv[0]
                # Top 15 by absolute magnitude
                pairs = sorted(
                    zip(_feat_cols, shap_arr.tolist()),
                    key=lambda x: abs(x[1]), reverse=True,
                )[:15]
                fight_shap_values = [
                    {"feature": f, "display_name": _feature_display_name(f), "value": round(v, 4)}
                    for f, v in pairs
                ]
            except Exception as e:
                logger.debug(f"SHAP failed for {fighter_a} vs {fighter_b}: {e}")

        # --- Build feature highlights from top globally-important diff_ features ---
        fight_highlights = []
        for feat_name, importance in _global_importance[:25]:
            if len(fight_highlights) >= 8:
                break
            # Only show differential features (a vs b comparisons)
            if not feat_name.startswith("diff_"):
                continue
            val = features.get(feat_name)
            if val is None:
                continue
            suffix = feat_name[5:]  # strip "diff_"
            a_val = features.get(f"a_{suffix}")
            b_val = features.get(f"b_{suffix}")
            # Skip if both individual values are 0/null (no data)
            if (a_val in (None, 0, 0.0)) and (b_val in (None, 0, 0.0)):
                continue
            favors = None
            if isinstance(val, (int, float)) and val != 0:
                favors = "a" if val > 0 else "b"
            fight_highlights.append({
                "feature": feat_name,
                "display_name": _feature_display_name(feat_name),
                "value": round(float(val), 4) if isinstance(val, (int, float)) else val,
                "a_value": round(float(a_val), 4) if isinstance(a_val, (int, float)) else a_val,
                "b_value": round(float(b_val), 4) if isinstance(b_val, (int, float)) else b_val,
                "favors": favors,
            })

        row_data = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "prob_a": pred["prob_a"],
            "prob_b": pred["prob_b"],
            "confidence": pred["confidence"],
            "event_date": fight.get("commence_time"),
            "a_market_prob": fight["a_fair_prob_avg"],
            "b_market_prob": fight["b_fair_prob_avg"],
            "no_odds_prob_a": no_odds_a,
            "no_odds_prob_b": no_odds_b,
            "a_num_fights": a_fights,
            "b_num_fights": b_fights,
            "shap_values": fight_shap_values,
            "shap_base_value": shap_base_value,
            "feature_highlights": fight_highlights,
            "low_experience": low_experience,
            "method_stats": {
                k: (round(float(v), 4) if isinstance(v, (int, float)) and not np.isnan(v) else None)
                for k in [
                    "a_ko_rate", "b_ko_rate", "a_sub_rate", "b_sub_rate",
                    "a_dec_rate", "b_dec_rate", "a_roll_slpm", "b_roll_slpm",
                    "a_roll_kd", "b_roll_kd", "a_roll_sub_avg", "b_roll_sub_avg",
                    "a_roll_td_avg", "b_roll_td_avg", "a_total_rounds", "b_total_rounds",
                ]
                for v in [features.get(k)]
            },
        }
        # Include line movement metadata for bet filtering
        if line_features:
            row_data["line_movement"] = line_features.get("line_movement")
            row_data["line_is_sharp"] = line_features.get("line_is_sharp")
            row_data["line_steam_move"] = line_features.get("line_steam_move")
        prediction_rows.append(row_data)

    if not prediction_rows:
        logger.info("No predictions generated.")
        return

    # Cache predictions for the web dashboard (model vs market heatmap + predictions page)
    try:
        predictions_cache = LOGS_DIR / "predictions_cache.json"
        import json as _json
        with open(predictions_cache, "w") as f:
            _json.dump({
                "timestamp": datetime.now().isoformat(),
                "predictions": prediction_rows,
                "global_feature_importance": [
                    {"feature": f, "display_name": _feature_display_name(f),
                     "importance": round(v, 4)}
                    for f, v in _global_importance
                ],
            }, f, default=str)
        logger.info(f"Cached {len(prediction_rows)} predictions for dashboard")
    except Exception as e:
        logger.warning(f"Failed to cache predictions: {e}")

    predictions = pd.DataFrame(prediction_rows)

    # 4. Run duo traders (S evaluates first, C on remainder)
    results = run_duo_traders(
        predictions=predictions,
        markets=markets,
        clob=clob,
        dry_run=dry_run,
        min_edge=args.min_edge,
    )

    logger.info(f"\nDuo trader run complete. Total orders: {results['total_orders']}")


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

    # Tennis discovery command
    subparsers.add_parser("tennis-discover", help="Discover live tennis odds and Polymarket markets")

    # Tennis train command
    tennis_train_parser = subparsers.add_parser("tennis-train", help="Train the Stage 1 tennis baseline")
    tennis_train_parser.add_argument("--model", type=str, default="surface_elo")
    tennis_train_parser.add_argument("--start-year", type=int, default=None)
    tennis_train_parser.add_argument("--end-year", type=int, default=None)
    tennis_train_parser.add_argument("--force-download", action="store_true")

    # Tennis predict command
    tennis_predict_parser = subparsers.add_parser("tennis-predict", help="Predict live ATP/WTA singles matches")
    tennis_predict_parser.add_argument("--model", type=str, default="surface_elo")

    # Tennis live command
    tennis_live_parser = subparsers.add_parser("tennis-live", help="Run tennis dry-run discovery, prediction, and edge logging")
    tennis_live_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required dry-run mode; --no-dry-run is rejected because real-money tennis trading is not implemented",
    )
    tennis_live_parser.add_argument("--model", type=str, default="surface_elo")
    tennis_live_parser.add_argument("--min-edge", type=float, default=None)

    # Live command
    live_parser = subparsers.add_parser("live", help="Run duo-trader live bot (S+C)")
    live_parser.add_argument("--dry-run", action="store_true", default=True,
                             help="Dry run mode (default: True)")
    live_parser.add_argument("--real", action="store_true",
                             help="Run with real money (disables dry run)")
    live_parser.add_argument("--model", type=str, default="xgboost")
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
    btc_parser.add_argument("--walkforward", action="store_true",
                            help="Run walk-forward strategy comparison")
    btc_parser.add_argument("--retrain-months", type=int, default=6,
                            help="Months between model retraining (default: 6)")
    btc_parser.add_argument("--initial-years", type=int, default=5,
                            help="Years of initial training data (default: 5)")

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
        "tennis-discover": cmd_tennis_discover,
        "tennis-train": cmd_tennis_train,
        "tennis-predict": cmd_tennis_predict,
        "tennis-live": cmd_tennis_live,
        "live": cmd_duo_live,
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
