from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = REPO_ROOT / "entrypoint.sh"


def _shell_function_source(name: str) -> str:
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def _run_bash_assertions(assertions: str) -> subprocess.CompletedProcess[str]:
    bash = None
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"
        if git_bash.exists():
            bash = str(git_bash)
    bash = bash or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required to verify the container entrypoint")
    script = (
        "set -euo pipefail\n"
        f"{_shell_function_source('atomic_copy_verified')}\n"
        f"{_shell_function_source('sync_image_bfo_recovery_files')}\n"
        'test_dir="$(mktemp -d)"\n'
        "trap 'rm -rf \"$test_dir\"' EXIT\n"
        f"{assertions}"
    )
    return subprocess.run(
        [bash],
        cwd=REPO_ROOT,
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )


def test_atomic_copy_success_and_existing_destination_preservation():
    completed = _run_bash_assertions(
        """
printf 'complete-source\n' > "$test_dir/source"
atomic_copy_verified "$test_dir/source" "$test_dir/destination" no-replace
cmp -s "$test_dir/source" "$test_dir/destination"
printf 'preserve-existing\n' > "$test_dir/destination"
if atomic_copy_verified "$test_dir/source" "$test_dir/destination" no-replace; then
    copy_status=0
else
    copy_status=$?
fi
[ "$copy_status" -eq 2 ]
[ "$(cat "$test_dir/destination")" = 'preserve-existing' ]
[ -z "$(find "$test_dir" -name '.*.tmp.*' -print -quit)" ]
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_bfo_overlay_replaces_only_same_name_image_recovery_files():
    completed = _run_bash_assertions(
        """
mkdir -p "$test_dir/image" "$test_dir/runtime"
printf 'approved-new\n' > "$test_dir/image/historical_odds_bfo_recovered_batch.csv"
printf 'same\n' > "$test_dir/image/historical_odds_bfo_recovered_same.csv"
printf 'not-a-recovery\n' > "$test_dir/image/other.csv"
printf 'stale-old\n' > "$test_dir/runtime/historical_odds_bfo_recovered_batch.csv"
printf 'same\n' > "$test_dir/runtime/historical_odds_bfo_recovered_same.csv"
printf 'runtime-only\n' > "$test_dir/runtime/historical_odds_bfo_recovered_runtime_only.csv"
printf 'unrelated-runtime\n' > "$test_dir/runtime/other.csv"

sync_image_bfo_recovery_files "$test_dir/image" "$test_dir/runtime"

[ "$(cat "$test_dir/runtime/historical_odds_bfo_recovered_batch.csv")" = 'approved-new' ]
[ "$(cat "$test_dir/runtime/historical_odds_bfo_recovered_same.csv")" = 'same' ]
[ "$(cat "$test_dir/runtime/historical_odds_bfo_recovered_runtime_only.csv")" = 'runtime-only' ]
[ "$(cat "$test_dir/runtime/other.csv")" = 'unrelated-runtime' ]
[ -z "$(find "$test_dir/runtime" -name '.*.tmp.*' -print -quit)" ]
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_atomic_copy_rejects_directory_race_and_failed_copy_without_damage():
    completed = _run_bash_assertions(
        """
printf 'replacement\n' > "$test_dir/source"
mkdir "$test_dir/directory-target"
if atomic_copy_verified "$test_dir/source" "$test_dir/directory-target" replace; then
    exit 21
fi
[ -d "$test_dir/directory-target" ]

mv() {
    target="${!#}"
    printf 'concurrent-writer\n' > "$target"
    command mv "$@"
}
if atomic_copy_verified "$test_dir/source" "$test_dir/race-destination" no-replace; then
    race_status=0
else
    race_status=$?
fi
unset -f mv
[ "$race_status" -eq 2 ]
[ "$(cat "$test_dir/race-destination")" = 'concurrent-writer' ]

cp() {
    command cp "$@"
    last_arg="${!#}"
    printf 'torn' > "$last_arg"
    return 1
}
printf 'keep-on-copy-failure\n' > "$test_dir/failure-destination"
if atomic_copy_verified "$test_dir/source" "$test_dir/failure-destination" replace; then
    exit 23
fi
unset -f cp
[ "$(cat "$test_dir/failure-destination")" = 'keep-on-copy-failure' ]
[ -z "$(find "$test_dir" -name '.*.tmp.*' -print -quit)" ]
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_entrypoint_selects_only_a_verified_complete_bundle_store_release():
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert (
        'BUNDLE_STORE_ROOT="$PERSISTENT_DATA_DIR/production_bundle/store"'
        in source
    )
    assert 'if [ -e "$BUNDLE_STORE_ROOT" ]; then' in source
    assert '"$BUNDLE_STORE_ROOT/.production_bundle_store.json"' in source
    assert '"$BUNDLE_STORE_ROOT/active_bundle.json"' in source
    assert "scripts/install_staged_production_bundle.py" in source
    assert 'ACTIVE_BUNDLE_RESOLUTION="$(' in source
    assert 'resolve_bundle_field active_manifest_path' in source
    assert 'resolve_bundle_field active_models_dir' in source
    assert 'resolve_bundle_field active_processed_dir' in source
    assert 'resolve_bundle_field active_lookup_dir' in source
    assert "--field active_manifest_path" not in source
    assert 'PRODUCTION_BUNDLE_MANIFEST="$ACTIVE_LOOKUP_DIR/runtime_manifest.json"' in source
    assert 'export UFC_PROCESSED_DIR="$ACTIVE_LOOKUP_DIR"' in source
    assert 'ACTIVATE_SOURCE_GENERATION_ARG="--activate-source-generation"' in source
    assert 'LIVE_TRADING_MODE_NORMALIZED="$(' in source
    assert 'real|live) REAL_MONEY_MODE=1' in source
    assert 'if [ "$REAL_MONEY_MODE" -eq 1 ]' in source
    assert "real-money mode requires the verified persistent production bundle store" in source
    assert r'--source-manifest \"$SOURCE_PRODUCTION_BUNDLE_MANIFEST\"' in source
    assert r'--source-processed-dir \"$SOURCE_PROCESSED_DIR\"' in source
    assert r'--target-processed-dir \"$ACTIVE_LOOKUP_DIR\"' in source
    assert "scripts/startup_parity_gate.py" in source
    assert '--receipt "$STARTUP_PARITY_RECEIPT"' in source


def test_entrypoint_does_not_repair_immutable_release_models():
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert (
        'if [ "$BUNDLE_STORE_ACTIVE" -eq 0 ] '
        '&& [ -f "$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl" ]; then'
        in source
    )


def test_entrypoint_keeps_release_read_only_and_only_chowns_mutable_lookup():
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert 'chown root:app "$PERSISTENT_DATA_DIR"' in source
    assert 'chmod 1775 "$PERSISTENT_DATA_DIR"' in source
    assert '! -name production_bundle -exec chown -R app:app -- {} +' in source
    assert 'chown root:root "$PERSISTENT_DATA_DIR/production_bundle"' in source
    assert 'chown -R root:root "$BUNDLE_STORE_ROOT"' in source
    assert 'chmod -R a=rX "$ACTIVE_RELEASE_DIR"' in source
    assert 'chown -R app:app "$ACTIVE_LOOKUP_DIR"' in source
    assert 'chown app:app "$PERSISTENT_DATA_DIR"' not in source
    assert (
        'chown -R app:app "$PERSISTENT_DATA_DIR/production_bundle"'
        not in source
    )
    assert (
        'chown -R app:app "$PERSISTENT_DATA_DIR/production_bundle/current" '
        '"$ACTIVE_MODEL_DIR"'
        in source
    )
