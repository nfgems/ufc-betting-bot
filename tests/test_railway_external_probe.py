import json

import requests

from scripts import probe_railway_tapology_runtime as railway_probe


def test_railway_external_probe_refuses_local_result(monkeypatch, capsys):
    for name in (
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_DEPLOYMENT_ID",
        "RAILWAY_REPLICA_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    exit_code = railway_probe.main([])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["railway_environment"] is False
    assert payload["ok"] is False
    assert "not accepted as production evidence" in payload["error"]
    assert payload["implementation"]["fallback_scrapers_sha256"]
    assert payload["implementation"]["fightdx_request_max_attempts"] >= 1


def test_railway_external_probe_rejects_nan_as_published_field():
    assert railway_probe._published_value(float("nan")) is False
    assert railway_probe._published_value(None) is False
    assert railway_probe._published_value("NaN") is False
    assert railway_probe._published_value("N/A") is False
    assert railway_probe._published_value("??") is False
    assert railway_probe._published_value(182.88) is True


def test_railway_runtime_requires_deployed_replica_markers(monkeypatch):
    required = {
        "RAILWAY_PROJECT_ID": "project",
        "RAILWAY_SERVICE_ID": "service",
        "RAILWAY_ENVIRONMENT_ID": "environment",
        "RAILWAY_DEPLOYMENT_ID": "deployment",
        "RAILWAY_REPLICA_ID": "replica",
    }
    for name, value in required.items():
        monkeypatch.setenv(name, value)

    assert railway_probe._running_on_railway() is True
    monkeypatch.delenv("RAILWAY_REPLICA_ID")
    assert railway_probe._running_on_railway() is False


def test_runtime_binding_requires_and_matches_both_deployed_hashes():
    identity = {
        "RAILWAY_PROJECT_ID": "project",
        "RAILWAY_SERVICE_ID": "service",
        "RAILWAY_ENVIRONMENT_ID": "environment",
        "RAILWAY_DEPLOYMENT_ID": "deployment",
        "RAILWAY_GIT_COMMIT_SHA": "commit",
    }
    implementation = {
        "fallback_scrapers_sha256": "scraper-hash",
        "probe_sha256": "probe-hash",
    }

    missing_probe_hash = railway_probe._runtime_binding_report(
        identity=identity,
        implementation=implementation,
        expected_project_id="project",
        expected_service_id="service",
        expected_environment_id="environment",
        expected_source_sha256="scraper-hash",
    )
    assert missing_probe_hash["ok"] is False
    assert "probe_sha256" in missing_probe_hash["missing_expectations"]

    exact = railway_probe._runtime_binding_report(
        identity=identity,
        implementation=implementation,
        expected_project_id="project",
        expected_service_id="service",
        expected_environment_id="environment",
        expected_source_sha256="SCRAPER-HASH",
        expected_probe_sha256="PROBE-HASH",
        expected_deployment_id="deployment",
        expected_git_commit_sha="commit",
    )
    assert exact["ok"] is True

    wrong_service = railway_probe._runtime_binding_report(
        identity=identity,
        implementation=implementation,
        expected_project_id="project",
        expected_service_id="wrong-service",
        expected_environment_id="environment",
        expected_source_sha256="scraper-hash",
        expected_probe_sha256="probe-hash",
    )
    assert wrong_service["ok"] is False
    assert wrong_service["mismatches"] == [
        {
            "field": "RAILWAY_SERVICE_ID",
            "expected": "wrong-service",
            "actual": "service",
        }
    ]


def test_tapology_probe_proves_reader_wide_circuit_suppression_and_real_recovery(monkeypatch):
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "project")
    monkeypatch.setattr(railway_probe.fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "TAPOLOGY_READER_BASE_URL",
        "https://r.jina.ai",
    )
    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "TAPOLOGY_READER_REQUEST_DELAY_SECONDS",
        0,
    )
    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "_tapology_reader_request_variants",
        lambda **_kwargs: [{}],
    )

    def response(url: str, text: str = "") -> requests.Response:
        result = requests.Response()
        result.status_code = 200
        result.url = url
        result._content = text.encode("utf-8")
        return result

    def fake_real_get(url, *_args, **_kwargs):
        prepared = str(url)
        if "/search" in prepared:
            return response(
                prepared,
                "Title: Search Fighters, Bouts & Events | Tapology\nSearch Results (1)",
            )
        return response(prepared, "profile fetched")

    monkeypatch.setattr(requests, "get", fake_real_get)

    def fake_search(_name, limit=5, *, diagnostics=None):
        search_url = railway_probe.fallback_scrapers._tapology_fetch_url(
            railway_probe.fallback_scrapers.TAPOLOGY_SEARCH_URL,
            {"term": railway_probe.TAPOLOGY_PROBE_NAME},
        )
        railway_probe.fallback_scrapers._get_tapology_search_markdown_with_reader(search_url)
        diagnostics.update({"healthy": True, "candidate_count": 1})
        return [
            "https://www.tapology.com"
            f"{railway_probe.TAPOLOGY_PROFILE_PATH}"
        ][:limit]

    def fake_profile(url):
        railway_probe.fallback_scrapers.requests.get(
            f"https://r.jina.ai/https://www.tapology.com{railway_probe.TAPOLOGY_PROFILE_PATH}"
        )
        return {
            "name": "Ian Garry",
            "record": "16-1-0",
            "height": 190.5,
            "reach": 188.0,
            "weight": 170.0,
            "dob": "1997-11-17",
        }

    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "search_tapology_candidates",
        fake_search,
    )
    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "scrape_tapology_profile",
        fake_profile,
    )
    monkeypatch.setattr(
        railway_probe.fallback_scrapers,
        "scrape_tapology_fights",
        lambda *_args, **_kwargs: [{"opponent": "Test Opponent"}],
    )

    summary = railway_probe._tapology_probe()

    assert summary["ok"] is True
    assert summary["injected_status_count"] == 1
    assert summary["synthetic_reader_wide_status_code"] == 401
    assert "status 401" in summary["circuit_open_error"]
    assert summary["circuit_opened_after_synthetic_reader_wide_status"] is True
    assert summary["circuit_suppressed_request_while_open"] is True
    assert summary["circuit_recovered_through_real_request"] is True
    assert summary["reader_profile_request_count"] == 1


def test_fightdx_probe_proves_retry_and_single_profile_fetch(monkeypatch):
    profile_html = """
    <html><body>
      <h1>Felix Klinkhammer</h1>
      <span class="info-stat-label">Height</span>
      <span class="info-stat-value">6' 2&quot;</span>
      <span class="info-stat-label">Record</span>
      <span class="info-stat-value">11-1-0</span>
    </body></html>
    """

    def fake_real_get(url, *_args, **_kwargs):
        result = requests.Response()
        result.status_code = 200
        result.url = str(url)
        result._content = profile_html.encode("utf-8")
        return result

    monkeypatch.setattr(requests, "get", fake_real_get)
    monkeypatch.setattr(railway_probe.fallback_scrapers.time, "sleep", lambda _seconds: None)

    summary = railway_probe._fightdx_probe(exercise_retry=True)

    assert summary["ok"] is True
    assert summary["injected_timeout_count"] == 1
    assert summary["profile_attempt_count"] == 2
    assert summary["profile_request_count"] == 1
    assert summary["parsed_name"] == "Felix Klinkhammer"
    assert summary["parsed_height_cm"] > 0
