"""Create or verify a complete model-source and raw-input hash inventory.

The inventory deliberately hashes file bytes rather than relying on ``git diff``
so untracked source and raw inputs are part of the training identity too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import write_json_atomically


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def inventory_paths() -> list[Path]:
    """Return the intentionally broad model-affecting source/input scope."""
    candidates: set[Path] = set()
    for root, pattern in (
        (REPO_ROOT / "src", "*.py"),
        (REPO_ROOT / "scripts", "*.py"),
        (REPO_ROOT / "data" / "raw", "*"),
    ):
        if not root.exists():
            continue
        candidates.update(path for path in root.rglob(pattern) if path.is_file())

    for pattern in ("requirements*.txt",):
        candidates.update(path for path in REPO_ROOT.glob(pattern) if path.is_file())
    for filename in (
        "pyproject.toml",
        "setup.cfg",
        "Dockerfile",
        ".dockerignore",
        ".python-version",
    ):
        path = REPO_ROOT / filename
        if path.is_file():
            candidates.add(path)

    return sorted(
        candidates,
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )


def build_inventory(*, run_id: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    source_count = 0
    raw_count = 0

    for path in inventory_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        size = int(path.stat().st_size)
        sha256 = _sha256_file(path)
        category = "raw_input" if relative.startswith("data/raw/") else "source"
        raw_count += int(category == "raw_input")
        source_count += int(category == "source")
        rows.append(
            {
                "path": relative,
                "category": category,
                "bytes": size,
                "sha256": sha256,
            }
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(sha256.encode("ascii"))
        aggregate.update(b"\0")

    git_head = _git_output("rev-parse", "HEAD").decode("ascii").strip()
    git_status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    git_diff = _git_output("diff", "--binary", "HEAD")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "git_dirty": bool(git_status.strip()),
        "git_status_sha256": _sha256_bytes(git_status),
        "git_diff_sha256": _sha256_bytes(git_diff),
        "inventory_sha256": aggregate.hexdigest(),
        "scope": {
            "source": ["src/**/*.py", "scripts/**/*.py", "requirements*.txt", "pyproject.toml", "setup.cfg", "Dockerfile", ".dockerignore", ".python-version"],
            "raw_inputs": ["data/raw/**/*"],
            "excluded_sensitive": [".env", "betting/account ledger files"],
            "notes": [
                "The BFO provenance ledger is under data/raw and is included.",
                "Runtime package versions and exact training CLI arguments are pinned in the staged bundle manifest.",
            ],
        },
        "file_count": len(rows),
        "source_file_count": source_count,
        "raw_input_file_count": raw_count,
        "files": rows,
        "git_status": git_status.decode("utf-8", errors="replace").splitlines(),
    }


def compare_inventories(expected: dict[str, object], actual: dict[str, object]) -> dict[str, object]:
    expected_rows = {
        str(row["path"]): row
        for row in expected.get("files", [])
        if isinstance(row, dict) and row.get("path")
    }
    actual_rows = {
        str(row["path"]): row
        for row in actual.get("files", [])
        if isinstance(row, dict) and row.get("path")
    }
    added = sorted(set(actual_rows) - set(expected_rows))
    removed = sorted(set(expected_rows) - set(actual_rows))
    changed = sorted(
        path
        for path in set(expected_rows) & set(actual_rows)
        if (
            expected_rows[path].get("sha256") != actual_rows[path].get("sha256")
            or expected_rows[path].get("bytes") != actual_rows[path].get("bytes")
        )
    )
    git_head_matches = expected.get("git_head") == actual.get("git_head")
    git_diff_matches = (
        expected.get("git_diff_sha256") == actual.get("git_diff_sha256")
    )
    return {
        "ok": (
            not added
            and not removed
            and not changed
            and git_head_matches
            and git_diff_matches
        ),
        "expected_inventory_sha256": expected.get("inventory_sha256"),
        "actual_inventory_sha256": actual.get("inventory_sha256"),
        "git_head_matches": git_head_matches,
        "git_diff_matches": git_diff_matches,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()

    current = build_inventory(run_id=args.run_id)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing inventory: {output}")
        write_json_atomically(current, output)
        print(json.dumps({key: current[key] for key in ("run_id", "git_head", "inventory_sha256", "file_count", "source_file_count", "raw_input_file_count")}, indent=2))
        return 0

    expected_path = args.verify if args.verify.is_absolute() else REPO_ROOT / args.verify
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    comparison = compare_inventories(expected, current)
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
