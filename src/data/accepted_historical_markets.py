"""Read explicitly selected, accepted historical market files.

Their recorded time may be unknown. This module preserves accepted model inputs;
it does not certify a historical bet entry or change ordinary observation readers.
The selection manifest supplies acceptance and exact file hashes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path


logger = logging.getLogger(__name__)

METHOD_FEATURES = tuple(
    f"{side}_{method}_odds_prob"
    for side in ("a", "b") for method in ("ko", "sub", "dec")
)
MONEYLINE_FEATURES = ("a_implied_prob", "b_implied_prob", "diff_implied_prob")
TIMING_POLICY = "historical_market_timing_unknown"


class AcceptedMarketError(ValueError):
    """A selected file or concrete market binding/value is invalid."""


def _name(value: object) -> str:
    # Formatting only: no punctuation removal, phonetic or similar-name matching.
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def _date(value: object) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError) as exc:
        raise AcceptedMarketError(f"Invalid historical market date: {value!r}") from exc


def _probability(value: object, *, field: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AcceptedMarketError(f"Invalid accepted market value in {field}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AcceptedMarketError(f"Invalid accepted probability in {field}: {value!r}")
    return number


def _read_csv(path: Path, sha256: str, required: set[str]) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", str(sha256)):
        raise AcceptedMarketError("An explicit accepted SHA-256 is required")
    # Read once: validate and parse the identical bytes. I/O failures propagate.
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != sha256:
        raise AcceptedMarketError(f"Accepted historical market file changed: {path}")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")), strict=True)
        if not required.issubset(reader.fieldnames or ()):
            raise AcceptedMarketError(f"Accepted historical market schema is incomplete: {path}")
        rows = []
        for ordinal, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise AcceptedMarketError(f"Malformed accepted market row {ordinal}: {path}")
            rows.append({**row, "source_row_number": ordinal})
        return rows
    except (UnicodeDecodeError, csv.Error) as exc:
        raise AcceptedMarketError(f"Cannot read accepted historical market file: {path}") from exc


@dataclass(frozen=True)
class AcceptedMarketSelection:
    features: dict[str, float | None]
    records: tuple[dict, ...]
    timing_policy: str = TIMING_POLICY
    verified_bet_entry: bool = False

    def raw_moneyline_odds(self) -> dict[str, float]:
        """American-odds carrier for the existing historical feature builder.

        These reproduce the accepted fair probabilities. They are not claimed
        to be original offered prices or executable historical bet entries.
        """
        result = {}
        for side in ("a", "b"):
            probability = self.features.get(f"{side}_implied_prob")
            if probability is not None:
                if not 0.0 < probability < 1.0:
                    raise AcceptedMarketError("Moneyline probability cannot be zero or one")
                result[f"{side}_odds"] = (
                    -100.0 * probability / (1.0 - probability)
                    if probability > 0.5 else 100.0 * (1.0 - probability) / probability
                )
        return result


class AcceptedHistoricalMarkets:
    """An exact date/opponent/side lookup over two explicitly accepted files."""

    def __init__(self, *, moneyline_path: Path, moneyline_sha256: str,
                 method_odds_path: Path, method_odds_sha256: str,
                 reviewed_name_aliases: Mapping[str, str] | None = None):
        self._aliases = {_name(key): _name(value) for key, value in (reviewed_name_aliases or {}).items()}
        self._moneyline = {}
        self._methods = {}
        identity = {"event_date", "fighter_a", "fighter_b"}
        sources = (
            ("moneyline", Path(moneyline_path), moneyline_sha256,
             identity | {"a_fair_prob", "b_fair_prob"}, self._moneyline),
            ("method", Path(method_odds_path), method_odds_sha256,
             identity | set(METHOD_FEATURES), self._methods),
        )
        for family, path, sha256, columns, target in sources:
            for row in _read_csv(path, sha256, columns):
                key = self._key(row["event_date"], row["fighter_a"], row["fighter_b"])
                values = {name: _probability(row[name], field=name) for name in (
                    ("a_fair_prob", "b_fair_prob") if family == "moneyline" else METHOD_FEATURES
                )}
                if family == "moneyline":
                    left, right = values["a_fair_prob"], values["b_fair_prob"]
                    if (left is None) != (right is None):
                        raise AcceptedMarketError("Accepted moneyline pair has only one fighter side")
                    if left is not None and (
                        not 0.0 < left < 1.0 or not 0.0 < right < 1.0
                        or not math.isclose(left + right, 1.0, abs_tol=1e-10, rel_tol=0.0)
                    ):
                        raise AcceptedMarketError("Accepted fair moneyline probabilities disagree")
                    if key in target:
                        raise AcceptedMarketError("Duplicate fight in selected moneyline bridge")
                # The accepted method table contains reviewed replacement rows
                # appended after older rows. Keep the accepted whole-row order;
                # do not refill a replacement's real blank from the older row.
                # A repeated key is therefore allowed here but never silent:
                # the selection builder is expected to have removed accidental
                # duplicates (9 older partial rows were found on 2026-09-17).
                elif key in target:
                    logger.warning(
                        "Duplicate fight in selected method table; keeping the later row %s (earlier row %s)",
                        row["source_row_number"], target[key]["raw"]["source_row_number"],
                    )
                target[key] = {"family": family, "path": str(path), "sha256": sha256,
                               "raw": row, "values": values}

    def _canonical_name(self, value: object) -> str:
        name = _name(value)
        return self._aliases.get(name, name)

    def _key(self, event_date: object, fighter_a: object, fighter_b: object) -> tuple:
        left, right = self._canonical_name(fighter_a), self._canonical_name(fighter_b)
        if not left or not right or left == right:
            raise AcceptedMarketError("Historical market requires two distinct named fighters")
        return _date(event_date), *sorted((left, right))

    def for_fight(self, *, event_date: object, fighter_a: str, fighter_b: str) -> AcceptedMarketSelection:
        """Select only this dated matchup; preserve the requested fighter sides.

        The files predate stable-ID columns. Callers must use the names from
        their independently established dated fight binding, never names from
        an unrelated current roster search. Returned records retain the exact
        original row and its side orientation for that binding's evidence.
        """
        key = self._key(event_date, fighter_a, fighter_b)
        features = {}
        records = []
        for target in (self._moneyline, self._methods):
            selected = target.get(key)
            if selected is None:
                continue
            original = selected["raw"]
            reverse = self._canonical_name(fighter_a) == self._canonical_name(original["fighter_b"])
            values = selected["values"]
            if selected["family"] == "moneyline":
                left, right = values["a_fair_prob"], values["b_fair_prob"]
                if reverse:
                    left, right = right, left
                features.update(a_implied_prob=left, b_implied_prob=right,
                                diff_implied_prob=None if left is None else left - right)
            else:
                for name in METHOD_FEATURES:
                    source_name = ("b" if name[0] == "a" else "a") + name[1:] if reverse else name
                    features[name] = values[source_name]
            records.append({"family": selected["family"], "path": selected["path"],
                            "sha256": selected["sha256"], "raw": dict(original),
                            "reversed": reverse, "event_date": key[0],
                            "fighter_a": fighter_a, "fighter_b": fighter_b,
                            "timing_policy": TIMING_POLICY, "verified_bet_entry": False})
        return AcceptedMarketSelection(features=features, records=tuple(records))


def merge_accepted_historical_features(current: Mapping[str, object],
                                       accepted: AcceptedMarketSelection) -> dict:
    """Preserve accepted numbers through strict-reader unknowns; report conflicts.

    Pass actual newly selected observations here, not superseded legacy rows.
    Existing accepted within-file replacement decisions were applied at load.
    """
    result = dict(current)
    for field, value in accepted.features.items():
        if value is None:
            continue
        previous = result.get(field)
        if previous is not None:
            try:
                previous = float(previous)
            except (TypeError, ValueError) as exc:
                raise AcceptedMarketError(f"Invalid current market value for {field}") from exc
            if not math.isnan(previous) and not math.isclose(previous, value, rel_tol=0.0, abs_tol=1e-10):
                raise AcceptedMarketError(f"Conflicting accepted and current historical market value: {field}")
        result[field] = value
    return result
