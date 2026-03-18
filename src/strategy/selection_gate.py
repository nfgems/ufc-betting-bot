"""Candidate selection gate for filtering model variants before strategy testing.

Evaluates model candidates against a frozen control arm and selects finalists
that show genuine calibration improvement without material degradation elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Gate thresholds
_MAX_DEGRADATION = 0.05  # 5% max degradation on any core metric
_FRESH_DATA_CUTOFF = "2024-12-14"  # Post-cutoff window for credibility check
_FRESH_DATA_MIN_SAMPLES = 20  # Minimum sample size in fresh window
_LOW_COVERAGE_FLOOR = 0.10  # Slice must cover at least 10% of data
_YEARLY_DEGRADATION = 0.10  # Match promotion gate tolerance for yearly regressions


@dataclass
class CandidateResult:
    """Model candidate evaluation results from the model lab."""

    name: str
    dataset_variant: str
    feature_family: str
    calibration: str
    retrain_months: int | None = None
    metrics: dict = field(default_factory=dict)
    # Expected keys in metrics:
    #   accuracy, brier, log_loss, ece
    #   bootstrap_ci: dict of {metric: (lower, upper)}
    sliced_metrics: dict = field(default_factory=dict)
    # Expected keys in sliced_metrics:
    #   by_year: {year: {metric: value, ...}}
    #   by_coverage: {coverage_bucket: {metric: value, n_samples: int, ...}}
    #   fresh_window: {metric: value, n_samples: int, ...}


@dataclass
class SweepTargetSpec:
    """Concrete specification for a sweep target, produced by selection gate.

    Bridges the gap between ``CandidateResult`` (model evaluation output) and
    the downstream sweep / promotion pipeline.  Every field needed to reproduce
    a sweep run is captured here so that finalist_sweep and duo_trader_sweep
    do not have to guess defaults.
    """

    candidate_id: str  # unique ID for tracking
    model_variant: str  # variant name from ALL_VARIANTS
    dataset_variant: str  # e.g. "append_only_2026", "hybrid_legacy_first"
    feature_family: str  # e.g. "production", "production_betsapi_expanded"
    calibration_method: str  # e.g. "isotonic", "sigmoid"
    retrain_months: int = 4

    @classmethod
    def from_candidate(cls, candidate: CandidateResult) -> "SweepTargetSpec":
        """Create a sweep target spec from a CandidateResult."""
        return cls(
            candidate_id=f"{candidate.name}_{candidate.dataset_variant}_{candidate.feature_family}",
            model_variant=candidate.name,
            dataset_variant=candidate.dataset_variant,
            feature_family=candidate.feature_family,
            calibration_method=candidate.calibration,
            retrain_months=candidate.retrain_months or 4,
        )


def specs_from_variant_names(
    variant_names: list[str],
    dataset_variant: str = "append_only_2026",
    feature_family: str = "production",
    calibration_method: str | None = None,
    retrain_months: int = 4,
) -> list[SweepTargetSpec]:
    """Adapter: convert legacy variant name list to SweepTargetSpecs.

    When callers only have a flat ``list[str]`` of variant names (the old
    interface), this helper synthesises ``SweepTargetSpec`` objects with
    sensible defaults so they can be consumed by the new spec-aware sweep
    pipeline.

    Parameters
    ----------
    variant_names : list[str]
        Model variant names (e.g. ``["baseline", "betsapi_challenger"]``).
    dataset_variant : str
        Default dataset variant to assign.
    feature_family : str
        Default feature family to assign.
    calibration_method : str | None
        Default calibration method to assign. When ``None``, each variant keeps
        its factory default calibration method.
    retrain_months : int
        Walk-forward retrain cadence to carry into downstream sweeps.

    Returns
    -------
    list[SweepTargetSpec]
    """
    from src.strategy.model_variants import ALL_VARIANTS

    return [
        SweepTargetSpec(
            candidate_id=f"{name}_{dataset_variant}_{feature_family}",
            model_variant=name,
            dataset_variant=dataset_variant,
            feature_family=feature_family,
            calibration_method=(
                calibration_method
                if calibration_method is not None
                else ALL_VARIANTS[name]().calibration_method
            ),
            retrain_months=retrain_months,
        )
        for name in variant_names
    ]


class SelectionGate:
    """Evaluate model candidates against a frozen control arm.

    The gate checks calibration improvement, guards against regressions,
    validates performance on fresh data, and rejects slice-driven flukes.
    """

    def __init__(self, control_arm_metrics: dict) -> None:
        """Initialise with frozen control arm metrics.

        Parameters
        ----------
        control_arm_metrics : dict
            Must contain at least: accuracy, brier, log_loss, ece.
            Optionally: bootstrap_ci, sliced_metrics (same schema as CandidateResult).
        """
        required = {"accuracy", "brier", "log_loss", "ece"}
        missing = required - set(control_arm_metrics.keys())
        if missing:
            raise ValueError(f"Control arm metrics missing keys: {missing}")
        self.control = control_arm_metrics

    # ------------------------------------------------------------------
    # Single-candidate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, candidate: CandidateResult) -> dict:
        """Evaluate a single candidate against the control arm.

        Returns a dict with:
          passed: bool -- True if candidate passes all gates
          gates: dict of {gate_name: {passed: bool, detail: str}}
          composite_score: float -- ranking score (higher is better)
        """
        gates: dict[str, dict] = {}

        # Gate 1: Calibration improvement -- must improve Brier OR ECE
        brier_delta = self._relative_delta("brier", candidate.metrics.get("brier"))
        ece_delta = self._relative_delta("ece", candidate.metrics.get("ece"))
        # For Brier and ECE, lower is better, so negative delta = improvement
        brier_improves = brier_delta < 0
        ece_improves = ece_delta < 0
        gates["calibration_improvement"] = {
            "passed": brier_improves or ece_improves,
            "detail": (
                f"Brier delta={brier_delta:+.4f} ({'better' if brier_improves else 'worse'}), "
                f"ECE delta={ece_delta:+.4f} ({'better' if ece_improves else 'worse'})"
            ),
        }

        # Gate 2: No material degradation (>5%) on any core metric
        degradation_details = []
        degradation_passed = True
        for metric in ("accuracy", "brier", "log_loss", "ece"):
            cand_val = candidate.metrics.get(metric)
            if cand_val is None:
                continue
            ctrl_val = self.control[metric]
            if ctrl_val == 0:
                continue
            # For accuracy, higher is better; for brier/log_loss/ece, lower is better
            if metric == "accuracy":
                rel_change = (ctrl_val - cand_val) / ctrl_val  # positive = degradation
            else:
                rel_change = (cand_val - ctrl_val) / ctrl_val  # positive = degradation
            if rel_change > _MAX_DEGRADATION:
                degradation_passed = False
                degradation_details.append(
                    f"{metric}: {rel_change:+.1%} degradation (limit {_MAX_DEGRADATION:.0%})"
                )
        gates["no_material_degradation"] = {
            "passed": degradation_passed,
            "detail": "; ".join(degradation_details) if degradation_details else "All metrics within tolerance",
        }

        # Gate 3: Credible on fresh data window (post-2024-12-14)
        fresh = candidate.sliced_metrics.get("fresh_window", {})
        fresh_n = fresh.get("n_samples", 0)
        fresh_passed = fresh_n >= _FRESH_DATA_MIN_SAMPLES
        fresh_detail = (
            f"n_samples={fresh_n} (min={_FRESH_DATA_MIN_SAMPLES})"
            + ("" if fresh_passed else " -- INSUFFICIENT fresh-window data")
        )
        control_fresh_brier = self._fresh_window_brier(self.control)
        fresh_brier = fresh.get("brier")
        if (
            fresh_passed
            and fresh_brier is not None
            and control_fresh_brier is not None
        ):
            if fresh_brier > control_fresh_brier:
                fresh_passed = False
                fresh_detail += (
                    f" -- fresh Brier {fresh_brier:.4f} worse than control {control_fresh_brier:.4f}"
                )
            else:
                fresh_detail += (
                    f"; fresh Brier {fresh_brier:.4f} beats/ties control {control_fresh_brier:.4f}"
                )
        gates["fresh_data_credibility"] = {
            "passed": fresh_passed,
            "detail": fresh_detail,
        }

        # Gate 4: Year-by-year calibration must stay stable
        yearly_passed = True
        yearly_details = []
        control_years = self._normalized_year_metrics(self.control)
        candidate_years = self._normalized_year_metrics(
            {"sliced_metrics": candidate.sliced_metrics}
        )
        for year, cand_year in sorted(candidate_years.items()):
            ctrl_year = control_years.get(year)
            if ctrl_year is None:
                continue

            cand_brier = cand_year.get("brier")
            ctrl_brier = ctrl_year.get("brier")
            if ctrl_brier not in (None, 0) and cand_brier is not None:
                brier_degradation = (cand_brier - ctrl_brier) / ctrl_brier
                if brier_degradation > _YEARLY_DEGRADATION:
                    yearly_passed = False
                    yearly_details.append(
                        f"{year} brier {cand_brier:.4f} vs control {ctrl_brier:.4f}"
                    )

            cand_ece = cand_year.get("ece")
            ctrl_ece = ctrl_year.get("ece")
            if ctrl_ece not in (None, 0) and cand_ece is not None:
                ece_degradation = (cand_ece - ctrl_ece) / ctrl_ece
                if ece_degradation > _YEARLY_DEGRADATION:
                    yearly_passed = False
                    yearly_details.append(
                        f"{year} ece {cand_ece:.4f} vs control {ctrl_ece:.4f}"
                    )
        gates["year_by_year_stability"] = {
            "passed": yearly_passed,
            "detail": "; ".join(yearly_details) if yearly_details else "OK",
        }

        # Gate 5: Not driven by a tiny or low-coverage slice
        slice_passed = True
        slice_detail = "OK"
        by_coverage = candidate.sliced_metrics.get("by_coverage", {})
        if by_coverage:
            # Check if improvement is concentrated in a low-coverage bucket
            total_samples = sum(
                bucket.get("n_samples", 0) for bucket in by_coverage.values()
            )
            if total_samples > 0:
                improving_buckets = []
                for bucket_name, bucket_metrics in by_coverage.items():
                    bucket_n = bucket_metrics.get("n_samples", 0)
                    bucket_share = bucket_n / total_samples if total_samples > 0 else 0
                    bucket_brier = bucket_metrics.get("brier")
                    if bucket_brier is not None and bucket_brier < self.control.get("brier", 1.0):
                        improving_buckets.append((bucket_name, bucket_share, bucket_n))
                # Fail if ALL improvement comes from slices covering < 10% of data
                if improving_buckets and all(share < _LOW_COVERAGE_FLOOR for _, share, _ in improving_buckets):
                    slice_passed = False
                    names = [name for name, _, _ in improving_buckets]
                    slice_detail = f"Improvement driven entirely by low-coverage slices: {names}"
        gates["not_slice_driven"] = {
            "passed": slice_passed,
            "detail": slice_detail,
        }

        # Gate 6: Accuracy-only wins are rejected
        accuracy_only = False
        cand_acc = candidate.metrics.get("accuracy")
        ctrl_acc = self.control.get("accuracy")
        if cand_acc is not None and ctrl_acc is not None and cand_acc > ctrl_acc:
            # Wins on accuracy -- check if it also wins on calibration
            if not (brier_improves or ece_improves):
                accuracy_only = True
        gates["not_accuracy_only"] = {
            "passed": not accuracy_only,
            "detail": "Accuracy-only improvement without calibration gain" if accuracy_only else "OK",
        }

        # Composite score: Brier improvement + ECE improvement - log_loss degradation
        # All deltas are relative (negative = improvement for brier/ece, positive = degradation for log_loss)
        log_loss_delta = self._relative_delta("log_loss", candidate.metrics.get("log_loss"))
        composite = (-brier_delta) + (-ece_delta) - max(0.0, log_loss_delta)

        all_passed = all(g["passed"] for g in gates.values())

        return {
            "passed": all_passed,
            "gates": gates,
            "composite_score": composite,
            "brier_delta": brier_delta,
            "ece_delta": ece_delta,
            "log_loss_delta": log_loss_delta,
        }

    # ------------------------------------------------------------------
    # Multi-candidate selection
    # ------------------------------------------------------------------

    def select_finalists_with_specs(
        self,
        candidates: list[CandidateResult],
        max_finalists: int = 3,
    ) -> list[tuple[CandidateResult, SweepTargetSpec]]:
        """Select top-N candidates that pass all gates, with sweep specs attached.

        Parameters
        ----------
        candidates : list[CandidateResult]
            All model candidates to evaluate.
        max_finalists : int
            Maximum number of finalists to return (default 3).

        Returns
        -------
        list[tuple[CandidateResult, SweepTargetSpec]]
            Finalists sorted by composite score (best first), each paired
            with a ``SweepTargetSpec`` ready for the sweep pipeline.
        """
        evaluated: list[tuple[CandidateResult, dict]] = []
        for candidate in candidates:
            result = self.evaluate(candidate)
            if result["passed"]:
                evaluated.append((candidate, result))
            else:
                failed_gates = [
                    name for name, g in result["gates"].items() if not g["passed"]
                ]
                logger.info(
                    "Candidate '%s' rejected: failed gates %s",
                    candidate.name,
                    failed_gates,
                )

        # Sort by composite score descending
        evaluated.sort(key=lambda x: x[1]["composite_score"], reverse=True)

        pairs: list[tuple[CandidateResult, SweepTargetSpec]] = [
            (cand, SweepTargetSpec.from_candidate(cand))
            for cand, _ in evaluated[:max_finalists]
        ]
        logger.info(
            "Selection gate: %d/%d candidates passed, returning top %d finalists",
            len(evaluated),
            len(candidates),
            len(pairs),
        )
        return pairs

    def select_finalists(
        self,
        candidates: list[CandidateResult],
        max_finalists: int = 3,
    ) -> list[CandidateResult]:
        """Select top-N candidates that pass all gates, ranked by composite score.

        Convenience wrapper around :meth:`select_finalists_with_specs` that
        returns only the ``CandidateResult`` objects for backward
        compatibility.

        Parameters
        ----------
        candidates : list[CandidateResult]
            All model candidates to evaluate.
        max_finalists : int
            Maximum number of finalists to return (default 3).

        Returns
        -------
        list[CandidateResult]
            Finalists sorted by composite score (best first).
        """
        return [
            cand for cand, _spec in
            self.select_finalists_with_specs(candidates, max_finalists)
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_identity(candidate: CandidateResult) -> tuple[str, str, str, str]:
        """Return the full candidate identity used for finalist reporting."""
        return (
            candidate.name,
            candidate.dataset_variant,
            candidate.feature_family,
            candidate.calibration,
        )

    def generate_report(
        self,
        candidates: list[CandidateResult],
        finalists: list[CandidateResult],
    ) -> str:
        """Generate a markdown report of the selection gate results.

        Parameters
        ----------
        candidates : list[CandidateResult]
            All candidates that were evaluated.
        finalists : list[CandidateResult]
            The selected finalists.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        lines: list[str] = []
        lines.append("# Selection Gate Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\nControl arm metrics:")
        for metric in ("accuracy", "brier", "log_loss", "ece"):
            lines.append(f"  - {metric}: {self.control[metric]:.6f}")

        lines.append(f"\n## Candidates Evaluated: {len(candidates)}")
        lines.append("")

        finalist_ids = {self._candidate_identity(f) for f in finalists}

        # Table header
        lines.append(
            "| Candidate | Dataset | Feature Family | Calibration | Brier | ECE | "
            "LogLoss | Composite | Fresh N | Passed | Finalist |"
        )
        lines.append(
            "|-----------|---------|----------------|-------------|-------|-----|"
            "---------|-----------|---------|--------|----------|"
        )

        for candidate in candidates:
            result = self.evaluate(candidate)
            brier = candidate.metrics.get("brier", float("nan"))
            ece = candidate.metrics.get("ece", float("nan"))
            log_loss = candidate.metrics.get("log_loss", float("nan"))
            fresh_n = candidate.sliced_metrics.get("fresh_window", {}).get("n_samples", 0)
            is_finalist = self._candidate_identity(candidate) in finalist_ids
            lines.append(
                f"| {candidate.name} | {candidate.dataset_variant} | "
                f"{candidate.feature_family} | {candidate.calibration} | "
                f"{brier:.4f} | {ece:.4f} | "
                f"{log_loss:.4f} | {result['composite_score']:+.4f} | "
                f"{fresh_n} | {'PASS' if result['passed'] else 'FAIL'} | "
                f"{'YES' if is_finalist else ''} |"
            )

        # Gate details for failed candidates
        failed = [c for c in candidates if self._candidate_identity(c) not in finalist_ids]
        if failed:
            lines.append("\n## Failed Candidates - Gate Details")
            for candidate in failed:
                result = self.evaluate(candidate)
                if not result["passed"]:
                    lines.append(f"\n### {candidate.name}")
                    for gate_name, gate_info in result["gates"].items():
                        status = "PASS" if gate_info["passed"] else "FAIL"
                        lines.append(f"  - [{status}] {gate_name}: {gate_info['detail']}")

        # Finalist summary
        lines.append(f"\n## Finalists: {len(finalists)}")
        for i, finalist in enumerate(finalists, 1):
            result = self.evaluate(finalist)
            lines.append(
                f"\n### #{i}: {finalist.name}"
                f"\n  - Composite score: {result['composite_score']:+.4f}"
                f"\n  - Brier delta: {result['brier_delta']:+.4f}"
                f"\n  - ECE delta: {result['ece_delta']:+.4f}"
                f"\n  - Dataset: {finalist.dataset_variant}"
                f"\n  - Feature family: {finalist.feature_family}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _relative_delta(self, metric: str, candidate_value: float | None) -> float:
        """Compute relative delta: (candidate - control) / control.

        For Brier/ECE/log_loss, negative delta means improvement.
        """
        if candidate_value is None:
            return 0.0
        ctrl = self.control.get(metric, 0)
        if ctrl == 0:
            return 0.0
        return (candidate_value - ctrl) / abs(ctrl)

    @staticmethod
    def _fresh_window_brier(payload: dict) -> float | None:
        """Return fresh-window Brier from either top-level or sliced metrics."""
        top_level = payload.get("fresh_data_brier")
        if top_level is not None:
            return float(top_level)
        nested = (
            payload.get("sliced_metrics", {})
            .get("fresh_window", {})
            .get("brier")
        )
        return float(nested) if nested is not None else None

    @staticmethod
    def _normalized_year_metrics(payload: dict) -> dict[int, dict]:
        """Return year-slice metrics keyed by integer year."""
        raw_years = payload.get("year_by_year")
        if not raw_years:
            raw_years = payload.get("sliced_metrics", {}).get("by_year", {})

        if isinstance(raw_years, list):
            year_rows = raw_years
        elif isinstance(raw_years, dict):
            year_rows = [
                {"year": year, **metrics}
                for year, metrics in raw_years.items()
                if isinstance(metrics, dict)
            ]
        else:
            return {}

        normalized: dict[int, dict] = {}
        for row in year_rows:
            try:
                year = int(row["year"])
            except (KeyError, TypeError, ValueError):
                continue
            normalized[year] = {
                "accuracy": row.get("accuracy"),
                "brier": row.get("brier_score", row.get("brier")),
                "ece": row.get("ece"),
                "log_loss": row.get("log_loss"),
                "n_samples": row.get("n_samples", row.get("n")),
            }
        return normalized
