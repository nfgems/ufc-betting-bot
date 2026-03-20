"""Repair stale embedded training_spec metadata in no-odds model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.features.build_features import exclude_market_derived_features


def _default_paths_from_manifest(manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []

    promoted_alias_targets = payload.get("promoted_alias_targets", {})
    alias_path = promoted_alias_targets.get("no_odds_model")
    if alias_path:
        paths.append(Path(alias_path))

    promoted_from = payload.get("promoted_from", {})
    candidate_path = promoted_from.get("no_odds_model")
    if candidate_path:
        paths.append(Path(candidate_path))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _repair_artifact(path: Path) -> bool:
    result = joblib.load(path)
    artifact_cols = result.get("feature_cols")
    spec = result.get("training_spec")
    if not isinstance(artifact_cols, list):
        raise TypeError(f"{path} is missing feature_cols")
    if not isinstance(spec, dict):
        raise TypeError(f"{path} is missing training_spec")

    spec_cols = spec.get("feature_cols")
    if not isinstance(spec_cols, list):
        raise TypeError(f"{path} is missing training_spec.feature_cols")

    expected_no_odds_cols = exclude_market_derived_features(spec_cols)
    if artifact_cols != spec_cols and artifact_cols != expected_no_odds_cols:
        raise ValueError(
            f"{path} does not match the supported legacy no-odds metadata repair shape"
        )

    repaired_spec = dict(spec)
    repaired_spec["feature_cols"] = list(artifact_cols)

    name = str(repaired_spec.get("name", "") or "").strip()
    repaired_spec["name"] = (
        f"{name}_no_odds" if name and not name.endswith("_no_odds") else (name or "xgboost_no_odds")
    )

    description = str(repaired_spec.get("description", "") or "").strip()
    if description and "no-odds" not in description.lower():
        repaired_spec["description"] = f"{description} (no-odds variant)"
    elif not description:
        repaired_spec["description"] = "No-odds variant"

    if repaired_spec == spec:
        return False

    result["training_spec"] = repaired_spec
    joblib.dump(result, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit no-odds artifact paths to repair. Defaults to the active production paths in models/current_production_model.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/current_production_model.json"),
        help="Manifest used to locate the active no-odds artifacts when no explicit paths are provided.",
    )
    args = parser.parse_args()

    paths = args.paths or _default_paths_from_manifest(args.manifest)
    if not paths:
        raise SystemExit("No no-odds artifact paths were resolved.")

    for path in paths:
        changed = _repair_artifact(path)
        status = "repaired" if changed else "already up to date"
        print(f"{status}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
