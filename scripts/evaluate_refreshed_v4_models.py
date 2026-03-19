"""Evaluate saved model artifacts against one explicit refreshed V4 test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.refreshed_v4_eval import (  # noqa: E402
    evaluate_models_against_test_set,
    write_evaluation_payload,
)


def _parse_labeled_model_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value '{value}'. Expected label=path_to_model.pkl"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value '{value}'. Expected label=path_to_model.pkl"
        )
    return label, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-set",
        required=True,
        type=Path,
        help="Path to the fixed refreshed V4 test_set.csv to score against.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=_parse_labeled_model_arg,
        help="Labeled model artifact in the form label=path_to_model.pkl. Repeat for each model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, the payload is printed only.",
    )
    args = parser.parse_args()

    payload = evaluate_models_against_test_set(
        test_set_path=args.test_set,
        labeled_models=args.model,
    )

    if args.output is not None:
        write_evaluation_payload(payload, args.output)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
