"""Read a final refresh publication and score its bouts with observed live odds.

This adapter is read-only. It does not collect sources, retrain models, write
refresh state, or submit orders. The immutable final report supplies its saved
statistical feature calculation; current bookmaker observations supply only
the moneyline inputs for this invocation. Missing live inputs stay native NaN.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from src.data.name_utils import normalize_person_name, normalize_ufcstats_id
from src.data.refresh_predictions import (
    MODEL_CONTRACT, _decision_time, _json_value, _load_models, _verify_file,
)


class RefreshTradingError(ValueError):
    """A current trading input cannot be bound to the final served generation."""


def _probability(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise RefreshTradingError(f"Invalid current probability: {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefreshTradingError(f"Invalid current probability: {name}") from exc
    if not math.isfinite(number) or not 0 < number < 1:
        raise RefreshTradingError(f"Invalid current probability: {name}")
    return number


def _event_day(fight):
    """Keep provider event IDs separate from UFCStats IDs and official dates."""
    explicit = fight.get("official_event_date") or fight.get("card_date")
    if explicit:
        try:
            day = date.fromisoformat(str(explicit)).isoformat()
        except ValueError as exc:
            raise RefreshTradingError("Invalid official card date") from exc
        current = fight.get("commence_time") or fight.get("event_date")
        if current and _event_day({"commence_time": current}) != day:
            raise RefreshTradingError("Current commence time conflicts with the official card date")
        return day
    value = fight.get("commence_time") or fight.get("event_date")
    if not value:
        raise RefreshTradingError("Current matchup has no observed event date/time")
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    if len(text) == 10:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise RefreshTradingError("Invalid current matchup date") from exc
    try:
        instant = _decision_time(value)
    except (TypeError, ValueError) as exc:
        raise RefreshTradingError("Current event time must be timezone aware") from exc
    return instant.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _live_feature(column):
    """Discard weekly dynamic inputs without claiming a current observation."""
    return (
        "implied_prob" in column or "odds_prob" in column
        or column.startswith(("line_", "opening_", "closing_"))
        or column in {"a_rank", "b_rank", "diff_rank", "a_is_ranked", "b_is_ranked", "diff_is_ranked"}
        or "_wc_rank" in column or "_pfp_rank" in column
        or "ranking" in column
    )


def _swap_provenance(provenance):
    result = deepcopy(provenance)
    for key in provenance:
        if key.startswith("fighter_a_"):
            counterpart = "fighter_b_" + key[len("fighter_a_"):]
            result[key] = deepcopy(provenance.get(counterpart))
            result[counterpart] = deepcopy(provenance[key])
    return result


class PublishedTradingSnapshot:
    """A pinned, internally consistent final pair for one trading cycle.

    Call ``read`` once per cycle. A publication made during market collection
    belongs to the next cycle; this instance continues to use its original pair.
    ``read`` deliberately never reads the producer's mutable accepted.json.
    """

    def __init__(self, service):
        self.service = service
        self.publication = None
        self.snapshot = None
        self.report = None
        self.models = None
        self.rows = []

    def read(self):
        publication, snapshot, report = self.service.published_pair()
        manifest_path = Path(self.service.runtime_manifest).resolve()
        manifest = json.loads(manifest_path.read_bytes())
        if report.get("model_contract") != MODEL_CONTRACT:
            raise RefreshTradingError("Final report has an unsupported model contract")
        _verify_file(manifest_path, report["selected_runtime_manifest"])
        models, model_pins = _load_models(manifest_path, manifest)
        if any(any(model_pins[role][key] != snapshot["models"][role].get(key)
                   for key in ("bytes", "sha256")) for role in model_pins):
            raise RefreshTradingError("Trading models differ from the final served generation")
        selection = (manifest_path.parent / manifest["selection"]["path"]).resolve()
        if not selection.is_relative_to(manifest_path.parent):
            raise RefreshTradingError("Selected source manifest escapes the release root")
        _verify_file(selection, manifest["selection"])
        columns = models["primary"]["feature_cols"]
        rows = deepcopy(report["predictions"])
        identities = set()
        for row in rows:
            if not isinstance(row.get("model_features"), dict) or set(row["model_features"]) != set(columns):
                raise RefreshTradingError("Final bout features differ from the exact accepted contract")
            for value in row["model_features"].values():
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise RefreshTradingError("Final bout contains an invalid feature value")
            fighter_ids = [normalize_ufcstats_id(row.get(f"fighter_{side}_id")) for side in ("a", "b")]
            names = [normalize_person_name(row.get(f"fighter_{side}")) for side in ("a", "b")]
            if not all(fighter_ids) or fighter_ids[0] == fighter_ids[1] or not all(names) or names[0] == names[1]:
                raise RefreshTradingError("Final bout lacks distinct exact fighter identities")
            try:
                day = date.fromisoformat(row["event_date"]).isoformat()
            except (KeyError, TypeError, ValueError) as exc:
                raise RefreshTradingError("Final bout has no valid official event date") from exc
            identity = (day, tuple(sorted(names)))
            if identity in identities:
                raise RefreshTradingError("Final card contains an ambiguous name/date matchup")
            identities.add(identity)
        self.publication = deepcopy(publication)
        self.snapshot = deepcopy(snapshot)
        self.report = deepcopy(report)
        self.models = models
        self.rows = rows
        self.binding = {
            "release_id": manifest["release_id"],
            "manifest_sha256": report["selected_runtime_manifest"]["sha256"],
            "selection_sha256": manifest["selection"]["sha256"],
            "processed_sha256": {name: record["sha256"] for name, record in snapshot["artifacts"].items()},
            "model_sha256": {role: record["sha256"] for role, record in snapshot["models"].items()},
            "accepted_generation": report["accepted_generation"],
            "final_report_sha256": publication["prediction_report"]["sha256"],
        }
        return self

    def match_fight(self, fight):
        """Return the saved bout and orientation after exact name/date matching."""
        if self.report is None:
            raise RefreshTradingError("Read the final publication before scoring")
        names = [normalize_person_name(fight.get(f"fighter_{side}")) for side in ("a", "b")]
        day = _event_day(fight)
        matches = []
        for row in self.rows:
            saved_names = [normalize_person_name(row[f"fighter_{side}"]) for side in ("a", "b")]
            if row["event_date"] == day and names in (saved_names, list(reversed(saved_names))):
                matches.append((row, names != saved_names))
        if len(matches) != 1:
            raise RefreshTradingError("Current matchup is absent or ambiguous in the final published card")
        row, reversed_sides = matches[0]
        for side, saved_side in zip(("a", "b"), ("b", "a") if reversed_sides else ("a", "b")):
            supplied = fight.get(f"fighter_{side}_id")
            if supplied is not None and normalize_ufcstats_id(supplied) != row[f"fighter_{saved_side}_id"]:
                raise RefreshTradingError("Current fighter ID conflicts with the final official card")
        official_id = fight.get("official_event_id")
        if official_id is not None and official_id != row.get("event_id"):
            raise RefreshTradingError("Current official event ID conflicts with the final card")
        return deepcopy(row), reversed_sides

    def score(self, fights, *, observed_at, decision_at=None, max_observation_age_seconds=None):
        """Score current consensus observations; retain the no-odds companion.

        ``observed_at`` is the timestamp of the caller's real current collection,
        never the weekly prediction timestamp. Provider event IDs are preserved
        for downstream matching; exact official IDs remain separate metadata.
        Missing bookmaker odds produce diagnostic native-NaN predictions, while
        the existing trader's missing-market/no-odds gates remain in control.
        """
        from src.model.orientation import swap_directional_frame
        from src.model.predict import predict_fight
        import pandas as pd

        if self.models is None:
            raise RefreshTradingError("Read the final publication before scoring")
        instant = _decision_time(decision_at)
        observed = _decision_time(observed_at) if observed_at is not None else None
        if observed is None or observed > instant:
            raise RefreshTradingError("Current odds need a real nonfuture observation timestamp")
        if max_observation_age_seconds is not None:
            if (not math.isfinite(float(max_observation_age_seconds))
                    or float(max_observation_age_seconds) <= 0):
                raise RefreshTradingError("Observation age limit must be positive")
            if (instant - observed).total_seconds() > float(max_observation_age_seconds):
                raise RefreshTradingError("Current bookmaker observations are stale")
        columns = self.models["primary"]["feature_cols"]
        results = []
        for fight in fights:
            saved, reversed_sides = self.match_fight(fight)
            features = {column: saved["model_features"][column] for column in columns}
            provenance = deepcopy(saved.get("feature_provenance") or {})
            if reversed_sides:
                features = swap_directional_frame(pd.DataFrame([features]), columns=columns).iloc[0].to_dict()
                provenance = _swap_provenance(provenance)
            cleared = [column for column in columns if _live_feature(column)]
            for column in cleared:
                features[column] = None
            a_prob = _probability(fight.get("a_fair_prob_avg"), "a_fair_prob_avg")
            b_prob = _probability(fight.get("b_fair_prob_avg"), "b_fair_prob_avg")
            if a_prob is not None and b_prob is not None and not math.isclose(a_prob + b_prob, 1, abs_tol=1e-8):
                raise RefreshTradingError("Current consensus probabilities are not a normalized pair")
            features.update(a_implied_prob=a_prob, b_implied_prob=b_prob,
                            diff_implied_prob=a_prob - b_prob if a_prob is not None and b_prob is not None else None)
            # The saved contract supplies feature order, native missing-value
            # preprocessing, and the same orientation symmetrization as the bot.
            prediction = predict_fight(features, model_result=self.models["primary"])
            independent = predict_fight(features, model_result=self.models["no_odds"])
            for result in (prediction, independent):
                a, b = result.get("prob_a"), result.get("prob_b")
                if (not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                        or not math.isfinite(a) or not math.isfinite(b)
                        or not 0 <= a <= 1 or not 0 <= b <= 1
                        or not math.isclose(a + b, 1, abs_tol=1e-12)):
                    raise RefreshTradingError("Accepted model produced invalid probabilities")
            counts = {}
            for side in ("a", "b"):
                value = features.get(f"{side}_num_fights")
                if value is None or not math.isfinite(float(value)) or value < 0 or float(value) != int(value):
                    raise RefreshTradingError("Final fighter history has no observed UFC fight count")
                counts[f"{side}_num_fights"] = int(value)
            provenance.update(
                selected_runtime_binding=deepcopy(self.binding),
                served_generation=self.report["accepted_generation"],
                saved_feature_decision_at=self.report["decision_at"],
                live_odds_observed_at=observed.isoformat(),
                live_inference_decision_at=instant.isoformat(),
                dynamic_features_cleared=cleared,
                current_odds_source="bookmaker_consensus",
                final_report_pin=deepcopy(self.publication["prediction_report"]),
            )
            result = {
                **deepcopy(fight), **prediction, **counts,
                "event_date": fight.get("commence_time") or fight.get("event_date"),
                "fighter_a_id": saved["fighter_b_id" if reversed_sides else "fighter_a_id"],
                "fighter_b_id": saved["fighter_a_id" if reversed_sides else "fighter_b_id"],
                "official_event_id": saved["event_id"], "official_event_date": saved["event_date"],
                "card_date": saved["event_date"], "weight_class": saved.get("weight_class"),
                "is_title_bout": saved.get("is_title_bout"), "num_rounds": saved.get("num_rounds"),
                "is_empty_arena": saved.get("is_empty_arena"),
                "a_market_prob": a_prob, "b_market_prob": b_prob,
                "no_odds_prob_a": independent["prob_a"], "no_odds_prob_b": independent["prob_b"],
                "accepted_generation": self.report["accepted_generation"],
                "model_contract": MODEL_CONTRACT, "prediction_timestamp": instant.isoformat(),
                "market_inputs_available": a_prob is not None and b_prob is not None,
                "model_features": features, "feature_provenance": provenance,
            }
            results.append(_json_value(result))
        return results
