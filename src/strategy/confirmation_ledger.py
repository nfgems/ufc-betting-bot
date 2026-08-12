"""Durability checks for one-shot confirmation evidence.

Confirmation evidence is only globally meaningful once the exact files are in
the current commit and that commit is the tip of origin's authoritative default
branch.  Keeping this small check outside the evaluation orchestrator lets the
lower-level confirmation predictor enforce the same boundary.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable

# The one repository identity whose default-branch tip may anchor confirmation
# evidence (DIR-NRA-P3-013). A locally substituted `origin` URL must never
# satisfy the remote anchor.
CANONICAL_ORIGIN_IDENTITY = "github.com/nfgems/ufc-betting-bot"


def _normalized_origin_identity(url: str) -> str:
    """Reduce an origin URL to a scheme-independent host/owner/repo identity."""
    normalized = str(url or "").strip().lower()
    if normalized.startswith("git@") and ":" in normalized:
        host, _, path = normalized[len("git@"):].partition(":")
        normalized = f"{host}/{path}"
    else:
        for scheme in ("ssh://git@", "ssh://", "https://", "http://", "git://"):
            if normalized.startswith(scheme):
                normalized = normalized[len(scheme):]
                break
    if normalized.endswith(".git"):
        normalized = normalized[: -len(".git")]
    return normalized.rstrip("/")


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "confirmation durability requires an available, healthy git checkout"
        ) from exc
    return completed.stdout.strip()


def _git_succeeds(repo_root: Path, *args: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "confirmation durability requires an available git executable"
        ) from exc
    if completed.returncode not in {0, 1}:
        raise ValueError("confirmation durability git verification failed")
    return completed.returncode == 0


def require_remotely_anchored_git_artifacts(
    repo_root: Path,
    paths: Iterable[Path],
    *,
    label: str,
) -> str | None:
    """Require exact artifact bytes at origin's authoritative default-branch tip.

    Returns the anchored default-branch tip SHA so callers can record it in
    durable evidence payloads (DIR-NRA-P3-013), or ``None`` when there was
    nothing to anchor.
    """
    repo_root = repo_root.resolve()
    resolved_set = {Path(path).resolve() for path in paths}
    claim_root = (repo_root / "evidence" / "confirmation_claims").resolve()
    if any(path.is_relative_to(claim_root) for path in resolved_set):
        resolved_set.add((repo_root / ".gitattributes").resolve())
    resolved_paths = sorted(resolved_set)
    if not resolved_paths:
        return None

    remotes = _git_output(repo_root, "remote").splitlines()
    if "origin" not in remotes:
        raise ValueError(f"{label} has no canonical origin ledger remote")
    origin_url = _git_output(repo_root, "remote", "get-url", "origin")
    if _normalized_origin_identity(origin_url) != CANONICAL_ORIGIN_IDENTITY:
        raise ValueError(
            f"{label} origin remote is not the canonical ledger remote "
            f"({CANONICAL_ORIGIN_IDENTITY})"
        )
    remote_rows = _git_output(
        repo_root,
        "ls-remote",
        "--symref",
        "--exit-code",
        "origin",
        "HEAD",
    ).splitlines()
    default_refs = {
        fields[1]
        for row in remote_rows
        if len(fields := row.split()) == 3
        and fields[0] == "ref:"
        and fields[1].startswith("refs/heads/")
        and fields[2] == "HEAD"
    }
    default_tip_shas = {
        fields[0]
        for row in remote_rows
        if len(fields := row.split()) == 2
        and fields[1] == "HEAD"
        and len(fields[0]) == 40
    }
    if len(default_refs) != 1 or len(default_tip_shas) != 1:
        raise ValueError(
            f"{label} cannot resolve origin's authoritative default branch"
        )
    head_sha = _git_output(repo_root, "rev-parse", "HEAD")
    if default_tip_shas != {head_sha}:
        raise ValueError(
            f"{label} requires HEAD to be the exact pushed origin default-branch tip"
        )

    failures: list[str] = []
    for path in resolved_paths:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{label} artifact is outside the repository") from exc
        if not path.is_file():
            failures.append(f"{relative} (missing)")
            continue
        try:
            _git_output(repo_root, "ls-files", "--error-unmatch", "--", relative)
        except ValueError:
            failures.append(f"{relative} (untracked)")
            continue
        if not _git_succeeds(repo_root, "diff", "--quiet", "HEAD", "--", relative):
            failures.append(f"{relative} (bytes differ from HEAD)")
            continue
        if relative != ".gitattributes":
            committed_blob = _git_output(repo_root, "rev-parse", f"HEAD:{relative}")
            current_blob = _git_output(
                repo_root,
                "hash-object",
                "--no-filters",
                relative,
            )
            if current_blob != committed_blob:
                failures.append(f"{relative} (raw bytes differ from committed blob)")
                continue
        if relative.startswith("evidence/confirmation_claims/"):
            text_attribute = _git_output(
                repo_root,
                "check-attr",
                "--cached",
                "text",
                "--",
                relative,
            )
            if not text_attribute.endswith(": text: unset"):
                failures.append(
                    f"{relative} (must be protected from checkout byte conversion)"
                )
    if failures:
        raise ValueError(
            f"{label} must be committed and pushed with exact current bytes: "
            + ", ".join(failures)
        )
    return head_sha


def require_recorded_anchor_in_history(
    repo_root: Path,
    anchored_tip_sha: object,
    *,
    label: str,
) -> str:
    """Require a recorded anchored origin tip that is in the local history."""
    sha = str(anchored_tip_sha or "").strip().lower()
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError(f"{label} records no valid anchored origin tip")
    if not _git_succeeds(
        Path(repo_root).resolve(),
        "merge-base",
        "--is-ancestor",
        sha,
        "HEAD",
    ):
        raise ValueError(
            f"{label} anchored origin tip is not an ancestor of the current checkout"
        )
    return sha


__all__ = [
    "CANONICAL_ORIGIN_IDENTITY",
    "require_recorded_anchor_in_history",
    "require_remotely_anchored_git_artifacts",
]
