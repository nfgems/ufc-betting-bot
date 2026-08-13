"""Enhanced strategy sweep for finalist model candidates.

Runs broad and narrow S+C duo-trader sweeps for each finalist, then compares
results across candidates with volume-adjusted metrics.

Usage:
    python -m src.strategy.finalist_sweep --finalists baseline,betsapi_challenger
    python -m src.strategy.finalist_sweep --finalists baseline --narrow-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import INITIAL_BANKROLL, TRAIN_CUTOFF_DATE
from src.strategy.duo_trader_sweep import (
    SweepConfig,
    build_sweep_configs,
    _evaluate_config,
    _generate_walk_forward_predictions,
    _production_sweep_config,
    _summary_row_from_result,
    _sort_summary_rows,
    _print_sweep_results,
)
from src.strategy.selection_gate import SweepTargetSpec, specs_from_variant_names

logger = logging.getLogger(__name__)

_FINALIST_COMPARISON_COLUMNS = [
    "variant",
    "best_config",
    "profit",
    "total_profit",
    "roi",
    "avg_clv",
    "max_dd",
    "max_drawdown",
    "total_bets",
    "s_bets",
    "s_roi",
    "s_edge_profit_gap",
    "s_edge_profit_flag",
    "s_avg_clv",
    "c_bets",
    "c_roi",
    "c_avg_clv",
    "profit_per_bet",
    "prod_roi",
    "prod_bets",
    "volume_adjusted_flag",
]


# ---------------------------------------------------------------------------
# Narrow sweep helpers
# ---------------------------------------------------------------------------

def _narrow_grid_around(center: float, step: float, n_points: int = 5) -> list[float]:
    """Generate a narrow grid centered on *center* with spacing *step*.

    Returns *n_points* values symmetric around the center, clipped to > 0.
    """
    half = n_points // 2
    values = [center + (i - half) * step for i in range(n_points)]
    return [round(v, 6) for v in values if v > 0]


def narrow_sweep_around(
    top_configs: list[SweepConfig],
    variant_name: str,
    max_configs: int = 50,
) -> list[SweepConfig]:
    """Generate a narrow parameter grid centred on the best broad-sweep configs.

    Takes the top-5 (or fewer) configs from a broad sweep and builds a focused
    grid around their parameter values.

    Parameters
    ----------
    top_configs : list[SweepConfig]
        Best configs from the broad sweep (ideally 3-5).
    variant_name : str
        Model variant name for the narrow configs.
    max_configs : int
        Safety cap on the number of narrow configs generated.

    Returns
    -------
    list[SweepConfig]
        Narrow sweep configs ready for evaluation.
    """
    if not top_configs:
        return []

    # Collect unique parameter centres from the top configs
    s_edges = sorted({c.s_min_edge for c in top_configs})
    s_blends = sorted({c.s_blend_weight for c in top_configs})
    s_agreements = sorted({c.s_model_agreement_min_edge for c in top_configs})
    c_model_probs = sorted({c.c_min_model_prob for c in top_configs})
    c_no_odds_probs = sorted({c.c_min_no_odds_prob for c in top_configs})
    c_shares = sorted({c.c_share for c in top_configs})
    c_min_edges = sorted({c.c_min_edge for c in top_configs})
    c_max_odds = sorted({c.c_max_decimal_odds for c in top_configs})

    # Build narrow grids around each unique centre
    def _expand_set(values: list[float], step: float) -> list[float]:
        expanded = set()
        for v in values:
            for g in _narrow_grid_around(v, step, n_points=5):
                expanded.add(round(g, 6))
        return sorted(expanded)

    narrow_s_edges = _expand_set(s_edges, 0.003)
    narrow_s_blends = _expand_set(s_blends, 0.02)
    narrow_s_agreements = _expand_set(s_agreements, 0.005)
    narrow_c_model_probs = _expand_set(c_model_probs, 0.02)
    narrow_c_no_odds_probs = _expand_set(c_no_odds_probs, 0.02)
    # c_share and c_min_edge and c_max_odds: keep original values (discrete choices)
    narrow_c_shares = sorted(set(c_shares))
    narrow_c_min_edges = sorted(set(c_min_edges))
    narrow_c_max_odds = sorted(set(c_max_odds))

    configs = build_sweep_configs(
        variant_name=variant_name,
        s_edges=narrow_s_edges,
        s_blends=narrow_s_blends,
        s_agreements=narrow_s_agreements,
        c_model_probs=narrow_c_model_probs,
        c_no_odds_probs=narrow_c_no_odds_probs,
        c_shares=narrow_c_shares,
        c_min_edges=narrow_c_min_edges,
        c_max_decimal_odds_values=narrow_c_max_odds,
        include_production_controls=False,
    )

    # Rename to indicate narrow pass
    for i, cfg in enumerate(configs):
        configs[i] = SweepConfig(
            name=f"narrow_{cfg.name}",
            variant_name=cfg.variant_name,
            s_min_edge=cfg.s_min_edge,
            s_blend_weight=cfg.s_blend_weight,
            s_model_agreement_min_edge=cfg.s_model_agreement_min_edge,
            c_min_model_prob=cfg.c_min_model_prob,
            c_min_no_odds_prob=cfg.c_min_no_odds_prob,
            c_share=cfg.c_share,
            c_bet_fraction=cfg.c_bet_fraction,
            c_max_bet_fraction=cfg.c_max_bet_fraction,
            c_min_edge=cfg.c_min_edge,
            c_max_decimal_odds=cfg.c_max_decimal_odds,
            m_enabled=cfg.m_enabled,
        )

    if len(configs) > max_configs:
        logger.warning(
            "Narrow grid produced %d configs, capping at %d",
            len(configs),
            max_configs,
        )
        configs = configs[:max_configs]

    logger.info(
        "Narrow sweep: generated %d configs around %d top broad-sweep centres",
        len(configs),
        len(top_configs),
    )
    return configs


# ---------------------------------------------------------------------------
# Finalist sweep orchestrator
# ---------------------------------------------------------------------------

def run_finalist_sweep(
    finalists: list[str] | list[SweepTargetSpec],
    *,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    run_narrow: bool = True,
    narrow_only: bool = False,
    top_n_broad: int = 5,
) -> dict[str, dict]:
    """Run broad and (optionally) narrow S+C sweeps for each finalist variant.

    For each finalist variant:
      1. Generate walk-forward predictions once.
      2. Evaluate the production-controls config first (anchor row).
      3. Run a broad sweep via ``build_sweep_configs()``.
      4. If *run_narrow*, identify the top-5 broad configs and run a narrow
         second-pass sweep around that region.

    Parameters
    ----------
    finalists : list[str] | list[SweepTargetSpec]
        Variant names to sweep (e.g. ``["baseline", "betsapi_challenger"]``)
        or fully specified ``SweepTargetSpec`` objects.  When strings are
        provided they are converted via ``specs_from_variant_names()`` with
        default dataset/feature/calibration settings.
    initial_bankroll : float
        Starting bankroll for backtests.
    bet_start_date : str
        Date string for the earliest bets.
    run_narrow : bool
        Whether to run the narrow second-pass sweep (default True).
    narrow_only : bool
        Still seed the narrow pass from the broad sweep, but use narrow rows
        only for surfaced winners and comparison.
    top_n_broad : int
        Number of top broad-sweep configs to feed into the narrow sweep.

    Returns
    -------
    dict[str, dict]
        Per-finalist results keyed by candidate_id (or variant name for
        legacy callers). Each value contains:
          - production_row: dict (summary row for production controls)
          - broad_summary: list[dict] (all broad sweep summary rows)
          - broad_top_results: dict (top-5 raw result dicts)
          - narrow_summary: list[dict] | None
          - narrow_top_results: dict | None
          - best_config: dict (single best summary row, narrow-only if requested)
          - dataset_variant: str
          - feature_family: str
          - calibration_method: str
          - retrain_months: int
    """
    # Normalise input: convert list[str] to list[SweepTargetSpec]
    if finalists and isinstance(finalists[0], str):
        specs = specs_from_variant_names(finalists)  # type: ignore[arg-type]
    else:
        specs = list(finalists)  # type: ignore[arg-type]

    all_results: dict[str, dict] = {}
    run_narrow = run_narrow or narrow_only

    for spec in specs:
        variant_name = spec.model_variant
        logger.info(
            "=== Finalist sweep: %s (dataset=%s, features=%s, cal=%s) ===",
            variant_name,
            spec.dataset_variant,
            spec.feature_family,
            spec.calibration_method,
        )

        # Step 1: generate predictions once
        logger.info("Generating walk-forward predictions for %s...", variant_name)
        fold_predictions = _generate_walk_forward_predictions(
            bet_start_date=bet_start_date,
            variant_name=variant_name,
            dataset_variant=spec.dataset_variant,
            feature_family=spec.feature_family,
            calibration_method=spec.calibration_method,
            retrain_months=spec.retrain_months,
        )

        # Step 2: production controls anchor
        prod_config = _production_sweep_config(
            variant_name=variant_name,
            name=f"{variant_name}_production_controls",
            is_baseline=True,
        )
        prod_result = _evaluate_config(
            fold_predictions,
            prod_config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
        )
        prod_row = _summary_row_from_result(prod_config, prod_result)
        prod_row["candidate_id"] = spec.candidate_id
        prod_row["model_variant"] = spec.model_variant
        prod_row["dataset_variant"] = spec.dataset_variant
        prod_row["feature_family"] = spec.feature_family
        prod_row["calibration_method"] = spec.calibration_method
        logger.info(
            "Production controls: %d bets, ROI=%+.1f%%, profit=$%.2f",
            prod_row["total_bets"],
            prod_row["roi"] * 100,
            prod_row["profit"],
        )

        # Step 3: broad sweep
        broad_configs = build_sweep_configs(
            variant_name=variant_name,
            include_production_controls=False,
        )
        logger.info("Broad sweep: evaluating %d configs...", len(broad_configs))

        broad_rows: list[dict] = [prod_row]
        broad_results: dict[str, dict] = {prod_config.name: prod_result}

        for i, config in enumerate(broad_configs, 1):
            result = _evaluate_config(
                fold_predictions,
                config,
                initial_bankroll=initial_bankroll,
                bet_start_date=bet_start_date,
            )
            broad_results[config.name] = result
            row = _summary_row_from_result(config, result)
            row["candidate_id"] = spec.candidate_id
            row["model_variant"] = spec.model_variant
            row["dataset_variant"] = spec.dataset_variant
            row["feature_family"] = spec.feature_family
            row["calibration_method"] = spec.calibration_method
            broad_rows.append(row)
            if i % 50 == 0:
                logger.info("  Broad sweep: %d/%d configs done", i, len(broad_configs))

        _sort_summary_rows(broad_rows)
        if not narrow_only:
            _print_sweep_results(broad_rows[:20])

        # Top-N from broad (excluding the production baseline row)
        non_baseline_rows = [r for r in broad_rows if not r.get("is_baseline")]
        top_broad_rows = non_baseline_rows[:top_n_broad]
        top_broad_configs = [
            _row_to_sweep_config(row, variant_name) for row in top_broad_rows
        ]

        broad_top_results = {
            row["config"]: broad_results[row["config"]]
            for row in top_broad_rows
            if row["config"] in broad_results
        }

        # Step 4: narrow sweep (optional)
        narrow_rows: list[dict] | None = None
        narrow_top_results: dict | None = None

        if run_narrow and top_broad_configs:
            narrow_configs = narrow_sweep_around(
                top_broad_configs, variant_name=variant_name
            )
            if narrow_configs:
                logger.info("Narrow sweep: evaluating %d configs...", len(narrow_configs))
                narrow_rows = []
                narrow_all_results: dict[str, dict] = {}

                for i, config in enumerate(narrow_configs, 1):
                    result = _evaluate_config(
                        fold_predictions,
                        config,
                        initial_bankroll=initial_bankroll,
                        bet_start_date=bet_start_date,
                    )
                    narrow_all_results[config.name] = result
                    row = _summary_row_from_result(config, result)
                    row["candidate_id"] = spec.candidate_id
                    row["model_variant"] = spec.model_variant
                    row["dataset_variant"] = spec.dataset_variant
                    row["feature_family"] = spec.feature_family
                    row["calibration_method"] = spec.calibration_method
                    narrow_rows.append(row)
                    if i % 50 == 0:
                        logger.info("  Narrow sweep: %d/%d configs done", i, len(narrow_configs))

                _sort_summary_rows(narrow_rows)
                _print_sweep_results(narrow_rows[:20])

                narrow_top = narrow_rows[:top_n_broad]
                narrow_top_results = {
                    row["config"]: narrow_all_results[row["config"]]
                    for row in narrow_top
                    if row["config"] in narrow_all_results
                }
            else:
                narrow_all_results = {}
        else:
            narrow_all_results = {}

        # Best config can be selected across both passes or from narrow only.
        if narrow_only:
            if not narrow_rows:
                raise ValueError(
                    f"narrow_only=True requires narrow results for {spec.candidate_id}"
                )
            all_candidate_rows = list(narrow_rows)
        else:
            all_candidate_rows = list(broad_rows)
            if narrow_rows:
                all_candidate_rows.extend(narrow_rows)
        _sort_summary_rows(all_candidate_rows)
        best_row = all_candidate_rows[0] if all_candidate_rows else prod_row
        all_raw_results = dict(broad_results)
        all_raw_results.update(narrow_all_results)
        best_result = all_raw_results.get(best_row["config"], prod_result)

        result_key = spec.candidate_id

        primary_summary = narrow_rows if narrow_rows else broad_rows

        all_results[result_key] = {
            "candidate_id": spec.candidate_id,
            "model_variant": spec.model_variant,
            "production_row": prod_row,
            "production_result": prod_result,
            "broad_summary": broad_rows,
            "broad_top_results": broad_top_results,
            "narrow_summary": narrow_rows,
            "narrow_top_results": narrow_top_results,
            "best_config": best_row,
            "best_result": best_result,
            "primary_summary": primary_summary,
            "dataset_variant": spec.dataset_variant,
            "feature_family": spec.feature_family,
            "calibration_method": spec.calibration_method,
            "retrain_months": spec.retrain_months,
        }

        logger.info(
            "Finalist %s best config: %s (ROI=%+.1f%%, profit=$%.2f, bets=%d)",
            result_key,
            best_row["config"],
            best_row["roi"] * 100,
            best_row["profit"],
            best_row["total_bets"],
        )

    return all_results


# ---------------------------------------------------------------------------
# Cross-finalist comparison
# ---------------------------------------------------------------------------

def compare_finalist_sweeps(
    sweep_results: dict[str, dict],
) -> pd.DataFrame:
    """Compare sweep results across finalist variants.

    Produces a comparison DataFrame with per-variant best-config stats,
    including volume-adjusted ROI to flag candidates that win just by
    placing more bets.

    Parameters
    ----------
    sweep_results : dict[str, dict]
        Output of ``run_finalist_sweep()``.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per finalist.
    """
    rows: list[dict] = []

    for variant_name, data in sweep_results.items():
        # Prefer the explicit best_config selected by the sweep runner. Fall
        # back to primary_summary only for legacy payloads that never stored it.
        primary = data.get("primary_summary")
        best = data.get("best_config") or (primary[0] if primary else {})
        prod = data["production_row"]

        # Volume-adjusted ROI: normalise profit by number of bets
        total_bets = best.get("total_bets", 0)
        profit = best.get("total_profit", best.get("profit", 0.0))
        profit_per_bet = profit / total_bets if total_bets > 0 else 0.0

        prod_bets = prod.get("total_bets", 0)
        prod_profit = prod.get("total_profit", prod.get("profit", 0.0))
        prod_ppb = prod_profit / prod_bets if prod_bets > 0 else 0.0

        # Flag: candidate "wins" mainly by volume (> 50% more bets, lower per-bet profit)
        volume_flag = False
        if prod_bets > 0 and total_bets > prod_bets * 1.5 and profit_per_bet < prod_ppb:
            volume_flag = True

        rows.append(
            {
                "variant": variant_name,
                "best_config": best.get("config", ""),
                "profit": profit,
                "total_profit": profit,
                "roi": best.get("roi", 0.0),
                "avg_clv": best.get("avg_clv", np.nan),
                "max_dd": best.get("max_dd", best.get("max_drawdown", 0.0)),
                "max_drawdown": best.get("max_drawdown", best.get("max_dd", 0.0)),
                "total_bets": total_bets,
                "s_bets": best.get("s_bets", 0),
                "s_roi": best.get("s_roi", 0.0),
                "s_edge_profit_gap": best.get("s_edge_profit_gap", 0.0),
                "s_edge_profit_flag": bool(best.get("s_edge_profit_flag", False)),
                "s_avg_clv": best.get("s_avg_clv", np.nan),
                "c_bets": best.get("c_bets", 0),
                "c_roi": best.get("c_roi", 0.0),
                "c_avg_clv": best.get("c_avg_clv", np.nan),
                "profit_per_bet": profit_per_bet,
                "prod_roi": prod.get("roi", 0.0),
                "prod_bets": prod_bets,
                "volume_adjusted_flag": volume_flag,
            }
        )

    df = pd.DataFrame(rows, columns=_FINALIST_COMPARISON_COLUMNS)
    if not df.empty:
        df = df.sort_values(
            by=["s_edge_profit_flag", "s_edge_profit_gap", "roi"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    # Print comparison
    _print_comparison(df)

    return df


def _print_comparison(df: pd.DataFrame) -> None:
    """Print a formatted finalist comparison table."""
    if df.empty:
        print("No finalist results to compare.")
        return

    print(f"\n{'='*140}")
    print(f"FINALIST SWEEP COMPARISON ({len(df)} variants)")
    print(f"{'='*140}")
    print(
        f"{'Variant':<22} {'Best Config':<45} {'Bets':>5} {'S':>4} {'C':>4} "
        f"{'Profit':>10} {'ROI':>7} {'CLV':>7} {'MaxDD':>6} {'$/bet':>7} {'VolFlag':>7}"
    )
    print(
        f"{'-------':<22} {'-----------':<45} {'----':>5} {'--':>4} {'--':>4} "
        f"{'------':>10} {'---':>7} {'---':>7} {'-----':>6} {'-----':>7} {'-------':>7}"
    )

    for _, row in df.iterrows():
        clv_str = f"{row['avg_clv']:+.1%}" if pd.notna(row["avg_clv"]) else "n/a"
        flag_str = "YES" if row["volume_adjusted_flag"] else ""
        config_short = row["best_config"][:44]
        print(
            f"{row['variant']:<22} {config_short:<45} "
            f"{row['total_bets']:>5} {row['s_bets']:>4} {row['c_bets']:>4} "
            f"${row['profit']:>+9.2f} {row['roi']:>+6.1%} {clv_str:>7} "
            f"{row['max_dd']:>5.1%} ${row['profit_per_bet']:>+6.2f} {flag_str:>7}"
        )

    print(f"{'='*140}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_sweep_config(row: dict, variant_name: str) -> SweepConfig:
    """Reconstruct a SweepConfig from a summary row dict."""
    return SweepConfig(
        name=row.get("config", "unknown"),
        variant_name=variant_name,
        s_min_edge=row.get("s_min_edge", 0.02),
        s_blend_weight=row.get("s_blend_weight", 0.30),
        s_model_agreement_min_edge=row.get("s_model_agreement_min_edge", 0.01),
        c_min_model_prob=row.get("c_min_model_prob", 0.65),
        c_min_no_odds_prob=row.get("c_min_no_odds_prob", 0.50),
        c_share=row.get("c_share", 1.0),
        c_min_edge=row.get("c_min_edge", 0.0),
        c_max_decimal_odds=row.get("c_max_decimal_odds", float("inf")),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for finalist sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run broad + narrow S+C sweeps for finalist model variants"
    )
    parser.add_argument(
        "--finalists",
        type=str,
        required=True,
        help="Comma-separated list of variant names (e.g. baseline,betsapi_challenger)",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=INITIAL_BANKROLL,
        help=f"Initial bankroll (default: {INITIAL_BANKROLL})",
    )
    parser.add_argument(
        "--bet-start-date",
        type=str,
        default=TRAIN_CUTOFF_DATE,
        help=f"Bet start date (default: {TRAIN_CUTOFF_DATE})",
    )
    parser.add_argument(
        "--narrow-only",
        action="store_true",
        default=False,
        help="Still seed from the broad pass, but only surface narrow winners",
    )
    parser.add_argument(
        "--no-narrow",
        action="store_true",
        default=False,
        help="Skip the narrow second-pass sweep",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top broad configs to feed into narrow sweep (default: 5)",
    )

    args = parser.parse_args()
    finalist_names = [name.strip() for name in args.finalists.split(",") if name.strip()]

    if not finalist_names:
        print("Error: --finalists must contain at least one variant name")
        sys.exit(1)
    if args.narrow_only and args.no_narrow:
        parser.error("--narrow-only cannot be combined with --no-narrow")

    sweep_results = run_finalist_sweep(
        finalist_names,
        initial_bankroll=args.bankroll,
        bet_start_date=args.bet_start_date,
        run_narrow=not args.no_narrow,
        narrow_only=args.narrow_only,
        top_n_broad=args.top_n,
    )

    if len(finalist_names) > 1:
        comparison_df = compare_finalist_sweeps(sweep_results)
        logger.info("Comparison table shape: %s", comparison_df.shape)


if __name__ == "__main__":
    main()
