from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = REPO_ROOT / "entrypoint.sh"


def _atomic_copy_function_source() -> str:
    source = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"^atomic_copy_verified\(\) \{\n.*?^\}\n",
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
        f"{_atomic_copy_function_source()}\n"
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
