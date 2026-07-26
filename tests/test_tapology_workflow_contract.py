from pathlib import Path
import subprocess

import yaml


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "tapology-profile-refresh.yml"
)
GITATTRIBUTES_PATH = Path(__file__).resolve().parents[1] / ".gitattributes"


def _workflow() -> dict:
    # BaseLoader preserves GitHub's `on` key and expression values as strings.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_scheduled_and_default_path_has_a_verified_linux_browser_runtime() -> None:
    workflow = _workflow()
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    runner_input = dispatch_inputs["runner"]
    primary = workflow["jobs"]["primary"]

    assert set(dispatch_inputs) == {
        "limit",
        "probe_only",
        "sync_active_roster",
        "runner",
        "probe_name",
        "probe_url",
    }
    assert runner_input["default"] == "ubuntu-latest"
    assert runner_input["options"] == [
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
    ]
    assert primary["runs-on"] == "${{ inputs.runner || 'ubuntu-latest' }}"
    assert primary["continue-on-error"] == "true"

    install = _named_step(primary, "Install Linux browser runtime")
    verify = _named_step(primary, "Verify Linux browser fallback")
    assert install["if"] == "runner.os == 'Linux'"
    assert verify["if"] == "runner.os == 'Linux'"
    assert "sudo apt-get install -y xvfb xauth" in install["run"]
    assert "TAPOLOGY_BROWSER_BINARY=" in install["run"]
    assert "TAPOLOGY_XVFB_BINARY=" in install["run"]
    assert "_tapology_virtual_display" in verify["run"]
    assert "webdriver.Chrome" in verify["run"]


def test_classified_egress_failure_uses_a_fresh_alternate_runner_pool() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    primary = jobs["primary"]
    fallback = jobs["fallback"]
    pool = _named_step(primary, "Select alternate runner pool")["run"]
    attempt = _named_step(primary, "Run Tapology refresh attempt")["run"]

    assert primary["outputs"]["outcome"] == "${{ steps.verdict.outputs.outcome }}"
    assert primary["outputs"]["artifact_name"] == (
        "${{ steps.verdict.outputs.artifact_name }}"
    )
    assert primary["outputs"]["fallback_runner"] == (
        "${{ steps.pool.outputs.fallback_runner }}"
    )
    assert "fallback_runner=macos-latest" in pool
    assert "fallback_runner=ubuntu-latest" in pool
    assert fallback["needs"] == "primary"
    assert fallback["if"] == (
        "${{ needs.primary.outputs.outcome == 'retryable_hosted_transport' }}"
    )
    assert fallback["runs-on"] == "${{ needs.primary.outputs.fallback_runner }}"
    assert fallback["continue-on-error"] == "true"
    assert fallback["outputs"]["outcome"] == "${{ steps.verdict.outputs.outcome }}"
    assert fallback["outputs"]["artifact_name"] == (
        "${{ steps.verdict.outputs.artifact_name }}"
    )
    assert primary["env"]["USE_CONFIGURED_PROXY"] == "false"
    assert fallback["env"]["USE_CONFIGURED_PROXY"] == "true"
    assert primary["steps"] == fallback["steps"]

    assert 'payload.get("failure_kind")' in attempt
    assert 'hosted_egress_blocked|network_unavailable)' in attempt
    assert '[ "$ACTION" = "access_probe_failed" ]' in attempt
    assert '[ "$PROGRESS_STATE" = "source_error" ]' in attempt
    assert 'echo "outcome=retryable_hosted_transport"' in attempt
    assert "--probe-name \"$PROBE_NAME\"" in attempt
    assert "--probe-url \"$PROBE_URL\"" in attempt


def test_optional_secrets_and_per_attempt_diagnostics_are_wired() -> None:
    workflow = _workflow()
    primary = workflow["jobs"]["primary"]
    upload = _named_step(primary, "Upload attempt result and diagnostics")
    attempt_step = _named_step(primary, "Run Tapology refresh attempt")
    attempt = attempt_step["run"]
    attempt_env = attempt_step["env"]

    assert "TAPOLOGY_PROXY_URL" not in workflow["env"]
    assert "TAPOLOGY_READER_API_KEY" not in workflow["env"]
    assert "BRAVE_SEARCH_API_KEY" not in workflow["env"]
    assert attempt_env["TAPOLOGY_PROXY_URL"] == "${{ secrets.TAPOLOGY_PROXY_URL }}"
    assert attempt_env["TAPOLOGY_READER_API_KEY"] == (
        "${{ secrets.TAPOLOGY_READER_API_KEY || secrets.JINA_API_KEY }}"
    )
    assert attempt_env["BRAVE_SEARCH_API_KEY"] == (
        "${{ secrets.BRAVE_SEARCH_API_KEY }}"
    )
    assert 'TAPOLOGY_PROXY_URL="$RUN_PROXY"' in attempt
    assert "tapology-diagnostics/attempt.json" in attempt
    assert "tapology-diagnostics/attempt.log" in attempt
    assert upload["if"] == "always()"
    assert "tapology-diagnostics/" in upload["with"]["path"]
    assert "tapology-result/" in upload["with"]["path"]
    assert upload["with"]["if-no-files-found"] == "error"


def test_attempt_success_is_published_only_after_its_artifact_exists() -> None:
    workflow = _workflow()
    primary = workflow["jobs"]["primary"]
    upload = _named_step(primary, "Upload attempt result and diagnostics")
    verdict = _named_step(primary, "Publish attempt verdict")
    package = _named_step(primary, "Package successful supplement")

    assert upload["id"] == "artifact"
    assert package["id"] == "package"
    assert verdict["id"] == "verdict"
    assert verdict["if"] == "${{ !cancelled() }}"
    assert verdict["env"]["RAW_OUTCOME"] == "${{ steps.attempt.outputs.outcome }}"
    assert verdict["env"]["PACKAGE_OUTCOME"] == "${{ steps.package.outcome }}"
    assert verdict["env"]["ARTIFACT_OUTCOME"] == "${{ steps.artifact.outcome }}"
    assert verdict["env"]["ARTIFACT_NAME"] == (
        "tapology-${{ env.ATTEMPT_NAME }}-"
        "${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert '[ "$RAW_OUTCOME" = "success" ] &&' in verdict["run"]
    assert '[ "$PACKAGE_OUTCOME" = "success" ] &&' in verdict["run"]
    assert '[ "$ARTIFACT_OUTCOME" = "success" ]; then' in verdict["run"]
    assert '[ "$RAW_OUTCOME" = "retryable_hosted_transport" ]' in verdict["run"]
    assert 'echo "artifact_name=$ARTIFACT_NAME"' in verdict["run"]
    assert primary["steps"].index(upload) < primary["steps"].index(verdict)


def test_unhealthy_or_exhausted_attempts_cannot_be_a_green_noop() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    attempt = _named_step(jobs["primary"], "Run Tapology refresh attempt")["run"]
    finalize = jobs["finalize"]
    select = _named_step(
        finalize, "Validate and install successful result"
    )["run"]

    assert "--require-source-health" in attempt
    assert 'if [ "$STATUS" -ne 0 ]' in attempt
    assert 'if [ "$ACTION" != "refreshed" ]' in attempt
    assert '[ "$SOURCE_HEALTH" != "true" ]' in attempt
    assert '[ "$PROBE_OK" != "true" ]' in attempt
    assert "recovered|no_fields_available|no_recoverable_candidates" in attempt
    assert "Unhealthy or unknown progress state" in attempt

    assert finalize["if"] == "${{ always() && !cancelled() }}"
    assert 'if [ "$PRIMARY_OUTCOME" = "success" ]' in select
    assert 'elif [ "$FALLBACK_OUTCOME" = "success" ]' in select
    assert "No attempt produced a validated result" in select
    assert "exit 70" in select
    assert '[ "$PROBE_ONLY" = "true" ]' in select
    assert "Tapology probe mutated data" in select


def test_attempts_are_read_only_and_finalizer_is_the_only_committer() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    all_scripts = "\n".join(
        step.get("run", "")
        for job in jobs.values()
        for step in job["steps"]
    )
    commit_step = _named_step(
        jobs["finalize"], "Commit updated Tapology supplement if changed"
    )
    commit = commit_step["run"]
    package = _named_step(jobs["primary"], "Package successful supplement")["run"]
    install = _named_step(
        jobs["finalize"], "Validate and install successful result"
    )["run"]

    assert workflow["permissions"]["contents"] == "read"
    assert jobs["finalize"]["permissions"]["contents"] == "write"
    assert workflow["concurrency"]["group"] == "tapology-profile-refresh"
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert commit_step["if"] == "${{ inputs.probe_only != true }}"
    assert all_scripts.count("git commit ") == 1
    assert all_scripts.count("git push ") == 1
    assert "base-supplement-blob.txt" in package
    assert "EXPECTED_BASE_BLOB" in install
    assert "CURRENT_BASE_BLOB" in install
    assert "refusing to overwrite it" in install
    assert "GITHUB_REF_TYPE" in commit
    assert "git fetch origin" in commit
    assert 'git rebase "origin/$GITHUB_REF_NAME"' in commit
    assert 'git push origin "HEAD:$GITHUB_REF_NAME"' in commit


def test_finalizer_downloads_the_exact_artifact_from_the_producing_attempt() -> None:
    workflow = _workflow()
    finalize = workflow["jobs"]["finalize"]
    primary_download = _named_step(finalize, "Download primary result")
    fallback_download = _named_step(finalize, "Download fallback result")

    assert primary_download["with"]["name"] == (
        "${{ needs.primary.outputs.artifact_name }}"
    )
    assert fallback_download["with"]["name"] == (
        "${{ needs.fallback.outputs.artifact_name }}"
    )


def test_cross_runner_supplement_artifact_has_canonical_line_endings() -> None:
    attributes = GITATTRIBUTES_PATH.read_text(encoding="utf-8").splitlines()
    assert (
        "data/raw/ufc_fighters_profile_supplement.csv text eol=lf"
        in attributes
    )

    workflow = _workflow()
    package = _named_step(
        workflow["jobs"]["primary"],
        "Package successful supplement",
    )["run"]
    install = _named_step(
        workflow["jobs"]["finalize"],
        "Validate and install successful result",
    )["run"]
    assert '--path="$SUPPLEMENT"' in package
    assert "result-supplement-blob.txt" in package
    assert 'git cat-file blob "$RESULT_BLOB"' in package
    assert 'git hash-object --no-filters "$RESULT"' in install
    assert "EXPECTED_RESULT_BLOB" in install


def test_git_clean_filter_canonicalizes_windows_csv_line_endings() -> None:
    path = "data/raw/ufc_fighters_profile_supplement.csv"
    lf_bytes = b"name,source\nExample Fighter,tapology\n"
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")

    def hash_bytes(*arguments: str, data: bytes) -> str:
        completed = subprocess.run(
            ["git", "hash-object", *arguments],
            input=data,
            check=True,
            capture_output=True,
        )
        return completed.stdout.decode("ascii").strip()

    canonical_lf = hash_bytes("--no-filters", "--stdin", data=lf_bytes)
    filtered_crlf = hash_bytes(f"--path={path}", "--stdin", data=crlf_bytes)
    raw_crlf = hash_bytes("--no-filters", "--stdin", data=crlf_bytes)

    assert filtered_crlf == canonical_lf
    assert raw_crlf != canonical_lf


def test_attempt_and_finalizer_check_out_the_current_serialized_branch_head() -> None:
    workflow = _workflow()
    for job_name in ("primary", "fallback", "finalize"):
        checkout = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("uses") == "actions/checkout@v6"
        )
        assert checkout["with"] == {
            "ref": "${{ github.ref_name }}",
            "fetch-depth": "0",
        }
