"""One-arm health gate for the fixed scheduled UFC refit contract.

This checker does not compare or select models.  It verifies that one
walk-forward run used the policy-pinned evaluation spec and protocol, binds the
summary to the actual features and fight-level evaluation index, reconciles the
production-gated bet log, and applies fixed absolute health limits derived from
the reviewed seed-42 baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_production_refit_contract as contract_gate
from src.strategy.lab_stats import compute_max_drawdown


class QualityInputError(ValueError):
    """Raised for malformed, missing, mixed-run, or tampered quality evidence."""


_REQUIRED_SUMMARY_COLUMNS = {
    "spec",
    "spec_payload_sha256",
    "policy_sha256",
    "features_sha256",
    "protocol_sha256",
    "evaluation_sample_sha256",
    "model_seed",
    "odds_noise_seed",
    "execution_mode",
    "entry_offset_days",
    "entry_offset_for_features",
    "evaluation_start_date",
    "evaluation_end_date",
    "evaluation_n_fights",
    "evaluation_n_folds",
    "model_brier_score",
    "model_ece",
    "model_n_fights",
    "strategy_execution_mode",
    "strategy_total_bets",
    "strategy_win_rate",
    "strategy_total_wagered",
    "strategy_roi",
    "strategy_total_profit",
    "strategy_avg_clv",
    "strategy_max_drawdown_pct",
}
_EVALUATION_INDEX_COLUMNS = [
    "event_date",
    "fighter_a",
    "fighter_b",
    "target",
    "fold",
    "train_end",
    "test_end",
]
_REQUIRED_BET_COLUMNS = {
    "strategy",
    "event_date",
    "fighter_a",
    "fighter_b",
    "bet_size",
    "profit",
    "clv",
    "edge",
    "won",
    "execution_mode",
    "bankroll_after",
}


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise QualityInputError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise QualityInputError(f"{label} must be finite: {value!r}")
    return parsed


def _integer(value: Any, label: str) -> int:
    parsed = _finite_float(value, label)
    if not parsed.is_integer():
        raise QualityInputError(f"{label} must be an integer")
    return int(parsed)


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise QualityInputError(f"{label} must be a strict boolean")


def canonical_evaluation_index(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _EVALUATION_INDEX_COLUMNS if column not in frame.columns]
    if missing or frame.empty:
        raise QualityInputError(
            "evaluation index is empty or missing columns: " + ", ".join(missing)
        )
    if list(frame.columns) != _EVALUATION_INDEX_COLUMNS:
        raise QualityInputError(
            "evaluation index must contain exactly the fixed columns in fixed order"
        )
    result = frame.copy()
    for column in ("event_date", "train_end", "test_end"):
        parsed = pd.to_datetime(result[column], errors="coerce")
        if parsed.isna().any():
            raise QualityInputError(f"evaluation index contains invalid {column}")
        result[column] = parsed.dt.strftime("%Y-%m-%d")
    for column in ("fighter_a", "fighter_b"):
        if result[column].isna().any() or (result[column].astype(str).str.strip() == "").any():
            raise QualityInputError(f"evaluation index contains a missing {column}")
        result[column] = result[column].astype(str)
    result["target"] = pd.to_numeric(result["target"], errors="coerce")
    if result["target"].isna().any() or not set(result["target"].astype(int)).issubset({0, 1}):
        raise QualityInputError("evaluation index target must contain only 0/1")
    result["target"] = result["target"].astype(int)
    result["fold"] = pd.to_numeric(result["fold"], errors="coerce")
    if result["fold"].isna().any() or (result["fold"] <= 0).any():
        raise QualityInputError("evaluation index fold must contain positive integers")
    if not (result["fold"] % 1 == 0).all():
        raise QualityInputError("evaluation index fold must contain integers")
    result["fold"] = result["fold"].astype(int)
    if result.duplicated(["event_date", "fighter_a", "fighter_b"]).any():
        raise QualityInputError("evaluation index contains duplicate fight identities")
    return result.sort_values(_EVALUATION_INDEX_COLUMNS, kind="stable").reset_index(drop=True)


def evaluation_index_sha256(frame: pd.DataFrame) -> str:
    canonical = canonical_evaluation_index(frame)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_summary(path: Path, *, policy: dict[str, Any], policy_sha256: str, features_sha256: str) -> pd.Series:
    if not path.is_file():
        raise QualityInputError(f"Track C summary is missing: {path}")
    try:
        summary = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise QualityInputError(f"cannot read Track C summary {path}: {exc}") from exc
    missing = sorted(_REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
    if missing:
        raise QualityInputError(f"Track C summary is missing columns: {missing}")
    if len(summary) != 1:
        raise QualityInputError("scheduled health summary must contain exactly one arm")
    row = summary.iloc[0]
    fixed = policy["contract"]
    expected = {
        "spec": fixed["evaluation_spec_name"],
        "spec_payload_sha256": fixed["evaluation_spec_payload_sha256"],
        "policy_sha256": policy_sha256,
        "features_sha256": features_sha256,
        "protocol_sha256": contract_gate.scheduled_protocol_sha256(policy),
    }
    for field, expected_value in expected.items():
        if str(row[field]).strip() != str(expected_value):
            raise QualityInputError(
                f"summary {field} does not match fixed evidence: "
                f"expected {expected_value!r}, got {row[field]!r}"
            )
    evaluation = policy["evaluation"]
    if _integer(row["model_seed"], "summary.model_seed") != evaluation["model_seed"]:
        raise QualityInputError("summary model seed does not match policy")
    if _integer(row["odds_noise_seed"], "summary.odds_noise_seed") != evaluation["odds_noise_seed"]:
        raise QualityInputError("summary odds-noise seed does not match policy")
    if str(row["execution_mode"]).strip() != evaluation["execution_mode"]:
        raise QualityInputError("summary execution mode does not match policy")
    if str(row["strategy_execution_mode"]).strip() != evaluation["execution_mode"]:
        raise QualityInputError("strategy execution mode does not match policy")
    if not math.isclose(
        _finite_float(row["entry_offset_days"], "summary.entry_offset_days"),
        float(evaluation["entry_offset_days"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise QualityInputError("summary entry offset does not match policy")
    if _strict_bool(row["entry_offset_for_features"], "summary.entry_offset_for_features") is not True:
        raise QualityInputError("summary did not use T-1 odds in model features")
    return row


def _read_evaluation_index(path: Path, *, expected_sha256: str) -> pd.DataFrame:
    if not path.is_file():
        raise QualityInputError(f"evaluation-index artifact is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise QualityInputError(f"cannot read evaluation index {path}: {exc}") from exc
    canonical = canonical_evaluation_index(frame)
    actual = evaluation_index_sha256(canonical)
    if actual != expected_sha256:
        raise QualityInputError(
            f"evaluation-index SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return canonical


def _parse_won(series: pd.Series) -> pd.Series:
    values: list[bool] = []
    for index, value in series.items():
        values.append(_strict_bool(value, f"bet_log.won[{index}]"))
    return pd.Series(values, index=series.index, dtype=bool)


def _fight_key(date: Any, fighter_a: Any, fighter_b: Any) -> tuple[str, str, str]:
    parsed = pd.to_datetime(date, errors="coerce")
    if pd.isna(parsed):
        raise QualityInputError(f"invalid fight date: {date!r}")
    names = sorted((str(fighter_a).strip(), str(fighter_b).strip()))
    if not names[0] or not names[1]:
        raise QualityInputError("fight identity has a missing fighter")
    return parsed.strftime("%Y-%m-%d"), names[0], names[1]


def _read_and_reconcile_bets(
    path: Path,
    *,
    row: pd.Series,
    evaluation_index: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not path.is_file():
        raise QualityInputError(f"production-gated bet log is missing: {path}")
    try:
        bets = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise QualityInputError(f"cannot read bet log {path}: {exc}") from exc
    missing = sorted(_REQUIRED_BET_COLUMNS - set(bets.columns))
    if missing:
        raise QualityInputError(f"bet log is missing columns: {missing}")
    if bets.empty:
        raise QualityInputError("production-gated bet log is empty")
    expected_strategy = policy["evaluation"]["strategy_name"]
    if set(bets["strategy"].dropna().astype(str)) != {expected_strategy}:
        raise QualityInputError("bet log contains a non-policy strategy")
    expected_mode = policy["evaluation"]["execution_mode"]
    if set(bets["execution_mode"].dropna().astype(str)) != {expected_mode}:
        raise QualityInputError("bet log contains a non-policy execution mode")

    bets = bets.copy()
    bets["won"] = _parse_won(bets["won"])
    for column in ("bet_size", "profit", "clv", "edge", "bankroll_after"):
        bets[column] = pd.to_numeric(bets[column], errors="coerce")
    if not bets[["bet_size", "profit", "clv", "edge", "bankroll_after"]].apply(
        lambda values: values.map(math.isfinite)
    ).all().all():
        raise QualityInputError("bet log contains non-finite numeric evidence")
    if (bets["bet_size"] <= 0).any():
        raise QualityInputError("bet log contains a non-positive stake")

    keys = [
        _fight_key(row_value.event_date, row_value.fighter_a, row_value.fighter_b)
        for row_value in bets.itertuples(index=False)
    ]
    if len(keys) != len(set(keys)):
        raise QualityInputError("bet log contains duplicate fight bets")
    evaluated_keys = {
        _fight_key(row_value.event_date, row_value.fighter_a, row_value.fighter_b)
        for row_value in evaluation_index.itertuples(index=False)
    }
    outside = [key for key in keys if key not in evaluated_keys]
    if outside:
        raise QualityInputError(f"bet log contains fights outside the evaluation sample: {outside[:3]}")

    total_bets = len(bets)
    total_wagered = float(bets["bet_size"].sum())
    total_profit = float(bets["profit"].sum())
    roi = total_profit / total_wagered
    avg_clv = float(bets["clv"].mean())
    win_rate = float(bets["won"].mean())
    initial_bankroll = float(policy["evaluation"]["initial_bankroll"])
    drawdown = float(
        compute_max_drawdown([initial_bankroll, *bets["bankroll_after"].astype(float).tolist()])[
            "max_drawdown_pct"
        ]
    )
    actual = {
        "strategy_total_bets": float(total_bets),
        "strategy_total_wagered": total_wagered,
        "strategy_total_profit": total_profit,
        "strategy_roi": roi,
        "strategy_avg_clv": avg_clv,
        "strategy_win_rate": win_rate,
        "strategy_max_drawdown_pct": drawdown,
    }
    tolerances = {
        "strategy_total_bets": (0.0, 0.0),
        "strategy_total_wagered": (1e-8, 0.02),
        "strategy_total_profit": (1e-8, 0.02),
        "strategy_roi": (1e-8, 1e-8),
        "strategy_avg_clv": (1e-8, 1e-8),
        "strategy_win_rate": (1e-8, 1e-8),
        "strategy_max_drawdown_pct": (1e-8, 1e-8),
    }
    for field, actual_value in actual.items():
        expected = _finite_float(row[field], f"summary.{field}")
        rel_tol, abs_tol = tolerances[field]
        if not math.isclose(expected, actual_value, rel_tol=rel_tol, abs_tol=abs_tol):
            raise QualityInputError(
                f"{field} mismatch: summary={expected:.12g}, log={actual_value:.12g}"
            )
    return bets, actual


def evaluate_quality(
    *,
    policy_path: Path,
    features_path: Path,
    artifacts_dir: Path,
    now: Any = None,
) -> dict[str, Any]:
    policy = contract_gate.load_policy(policy_path)
    policy_identity = contract_gate.policy_file_identity(policy_path, policy)
    registry_errors, _evaluation_spec, fullfit_spec = contract_gate.validate_policy_registry(policy)
    if fullfit_spec is None:
        return {"errors": registry_errors, "policy": policy_identity}
    features, feature_errors = contract_gate.inspect_features(
        features_path,
        policy=policy,
        fullfit_spec=fullfit_spec,
        now=now,
    )
    spec_name = policy["contract"]["evaluation_spec_name"]
    summary_path = artifacts_dir / "track_c_summary.csv"
    index_path = artifacts_dir / f"{spec_name}_evaluation_index.csv"
    bets_path = artifacts_dir / f"{spec_name}_production_gated_bets.csv"
    row = _read_summary(
        summary_path,
        policy=policy,
        policy_sha256=policy_identity["sha256"],
        features_sha256=features["sha256"],
    )
    summary_sample_sha = str(row["evaluation_sample_sha256"]).strip()
    evaluation_index = _read_evaluation_index(index_path, expected_sha256=summary_sample_sha)

    evaluation_start = pd.to_datetime(row["evaluation_start_date"], errors="coerce")
    evaluation_end = pd.to_datetime(row["evaluation_end_date"], errors="coerce")
    if pd.isna(evaluation_start) or pd.isna(evaluation_end) or evaluation_start > evaluation_end:
        raise QualityInputError("summary has an invalid evaluation window")
    index_start = pd.to_datetime(evaluation_index["event_date"]).min()
    index_end = pd.to_datetime(evaluation_index["event_date"]).max()
    if evaluation_start.normalize() != index_start.normalize() or evaluation_end.normalize() != index_end.normalize():
        raise QualityInputError("summary evaluation window does not match evaluation index")
    if evaluation_end.strftime("%Y-%m-%d") != features["snapshot_max_event_date"]:
        raise QualityInputError("evaluation end date does not match refreshed feature snapshot")
    n_fights = _integer(row["evaluation_n_fights"], "summary.evaluation_n_fights")
    model_n = _integer(row["model_n_fights"], "summary.model_n_fights")
    n_folds = _integer(row["evaluation_n_folds"], "summary.evaluation_n_folds")
    if n_fights != len(evaluation_index) or model_n != n_fights:
        raise QualityInputError("summary/model fight counts do not match evaluation index")
    if n_folds != int(evaluation_index["fold"].nunique()):
        raise QualityInputError("summary fold count does not match evaluation index")

    _bets, reconciled = _read_and_reconcile_bets(
        bets_path,
        row=row,
        evaluation_index=evaluation_index,
        policy=policy,
    )

    health_errors = [*registry_errors, *feature_errors]
    baseline = policy["baseline"]
    limits = policy["health_limits"]
    baseline_start = pd.Timestamp(baseline["evaluation_start_date"])
    baseline_end = pd.Timestamp(baseline["evaluation_end_date"])
    if evaluation_start.normalize() != baseline_start.normalize():
        health_errors.append(
            f"evaluation start changed from fixed baseline {baseline_start.date()}"
        )
    if evaluation_end.normalize() < baseline_end.normalize():
        health_errors.append(
            f"evaluation end {evaluation_end.date()} predates baseline {baseline_end.date()}"
        )
    if n_fights < limits["minimum_evaluation_fights"]:
        health_errors.append(
            f"evaluation fights {n_fights} below minimum {limits['minimum_evaluation_fights']}"
        )
    if n_folds < limits["minimum_evaluation_folds"]:
        health_errors.append(
            f"evaluation folds {n_folds} below minimum {limits['minimum_evaluation_folds']}"
        )

    brier = _finite_float(row["model_brier_score"], "summary.model_brier_score")
    ece = _finite_float(row["model_ece"], "summary.model_ece")
    if not 0.0 <= brier <= 1.0 or not 0.0 <= ece <= 1.0:
        raise QualityInputError("Brier and ECE must be inside [0, 1]")
    max_brier = baseline["model_brier_score"] * (
        1.0 + limits["maximum_brier_relative_regression"]
    )
    max_ece = baseline["model_ece"] + limits["maximum_ece_absolute_regression"]
    if brier > max_brier:
        health_errors.append(f"model Brier {brier:.9f} exceeds fixed maximum {max_brier:.9f}")
    if ece > max_ece:
        health_errors.append(f"model ECE {ece:.9f} exceeds fixed maximum {max_ece:.9f}")
    if reconciled["strategy_total_bets"] < limits["minimum_strategy_bets"]:
        health_errors.append(
            f"strategy bets {int(reconciled['strategy_total_bets'])} below minimum "
            f"{limits['minimum_strategy_bets']}"
        )
    if reconciled["strategy_roi"] < limits["minimum_strategy_roi"]:
        health_errors.append("strategy ROI is below the fixed minimum")
    if reconciled["strategy_total_profit"] < limits["minimum_strategy_total_profit"]:
        health_errors.append("strategy total profit is below the fixed minimum")
    max_drawdown = baseline["strategy_max_drawdown_pct"] * limits["maximum_drawdown_multiplier"]
    if reconciled["strategy_max_drawdown_pct"] > max_drawdown:
        health_errors.append(
            f"strategy drawdown {reconciled['strategy_max_drawdown_pct']:.9f} "
            f"exceeds fixed maximum {max_drawdown:.9f}"
        )
    min_clv = baseline["strategy_avg_clv"] - limits["maximum_clv_absolute_regression"]
    if reconciled["strategy_avg_clv"] < min_clv:
        health_errors.append(
            f"strategy average CLV {reconciled['strategy_avg_clv']:.9f} "
            f"is below fixed minimum {min_clv:.9f}"
        )

    return {
        "errors": health_errors,
        "policy": policy_identity,
        "spec": spec_name,
        "features": features,
        "artifacts": {
            "summary": str(summary_path.resolve(strict=False)),
            "evaluation_index": str(index_path.resolve(strict=False)),
            "bets": str(bets_path.resolve(strict=False)),
            "evaluation_sample_sha256": summary_sample_sha,
            "protocol_sha256": str(row["protocol_sha256"]),
        },
        "metrics": {
            "evaluation_n_fights": n_fights,
            "evaluation_n_folds": n_folds,
            "model_brier_score": brier,
            "model_ece": ece,
            **reconciled,
        },
        "derived_limits": {
            "maximum_model_brier_score": max_brier,
            "maximum_model_ece": max_ece,
            "maximum_strategy_drawdown_pct": max_drawdown,
            "minimum_strategy_avg_clv": min_clv,
        },
        "baseline_reference": {
            "comparison_role": baseline["comparison_role"],
            "evidence_path": baseline["evidence_path"],
            "evidence_sha256": baseline["evidence_sha256"],
            "evidence_root_source_manifest_sha256": baseline[
                "evidence_root_source_manifest_sha256"
            ],
            "evidence_protocol_sha256": baseline["evidence_protocol_sha256"],
            "scheduled_protocol_sha256": baseline["scheduled_protocol_sha256"],
            "shared_settings_verified": True,
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate_quality(
            policy_path=args.policy,
            features_path=args.features,
            artifacts_dir=args.artifacts_dir,
        )
        errors = list(result.pop("errors", []))
        status = "HEALTHY" if not errors else "UNHEALTHY"
        payload = {"status": status, "healthy": not errors, "errors": errors, **result}
        _write_report(args.report, payload)
        if errors:
            print("Scheduled refit health is UNHEALTHY:", file=sys.stderr)
            for error in errors:
                print(f" - {error}", file=sys.stderr)
            return 1
        print("Scheduled refit health is HEALTHY.")
        return 0
    except Exception as exc:
        payload = {"status": "ERROR", "healthy": False, "errors": [str(exc)]}
        try:
            _write_report(args.report, payload)
        except Exception as report_exc:
            print(f"Could not write required quality report: {report_exc}", file=sys.stderr)
        print(f"Scheduled refit health ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
