import argparse

import pytest

import scripts.wait_for_production_deployment as deployment_gate


EXPECTED_BUNDLE_ID = "ufc-production-refit-20260806"
EXPECTED_RELEASE_ID = "r-" + "a" * 20
EXPECTED_INSTALLED_MANIFEST_SHA = "0" * 64
EXPECTED_SOURCE_MANIFEST_SHA = "f" * 64
EXPECTED_DEPLOYED_SHA = "a" * 40
EXPECTED_TRAINING_SHA = "b" * 40
EXPECTED_SPEC = "full_live_contract_v6_durability_corrected_20260805_fullfit"
EXPECTED_MODEL_SHA = "1" * 64
EXPECTED_NO_ODDS_SHA = "2" * 64
EXPECTED_LOGISTIC_SHA = "3" * 64
EXPECTED_TRAINING_FIGHTS_SHA = "4" * 64
EXPECTED_TRAINING_FEATURES_SHA = "5" * 64
EXPECTED_RUNTIME_FIGHTS_SHA = "6" * 64
EXPECTED_RUNTIME_FEATURES_SHA = "7" * 64
EXPECTED_SNAPSHOT_DATE = "2026-08-01"


def _ready_payload(**overrides):
    payload = {
        "ready": True,
        "errors": [],
        "requested_live_mode": "real",
        "effective_live_mode": "real",
        "armed_for_real": True,
        "trading_enabled": True,
        "trading_live": True,
        "components": {
            "betting_loop": {
                "state": "running",
                "thread_alive": True,
            }
        },
        "production_bundle": {
            "bundle_id": EXPECTED_BUNDLE_ID,
            "rich_release_id": EXPECTED_RELEASE_ID,
            "installed_manifest_sha256": EXPECTED_INSTALLED_MANIFEST_SHA,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
            "deployed_git_sha": EXPECTED_DEPLOYED_SHA,
            "training_source_git_sha": EXPECTED_TRAINING_SHA,
            "model_spec_name": EXPECTED_SPEC,
            "model_sha256": EXPECTED_MODEL_SHA,
            "no_odds_model_sha256": EXPECTED_NO_ODDS_SHA,
            "logistic_model_sha256": EXPECTED_LOGISTIC_SHA,
            "immutable_training_fights_sha256": EXPECTED_TRAINING_FIGHTS_SHA,
            "immutable_training_features_sha256": EXPECTED_TRAINING_FEATURES_SHA,
            "immutable_training_snapshot_max_event_date": EXPECTED_SNAPSHOT_DATE,
            "processed_fights_sha256": EXPECTED_RUNTIME_FIGHTS_SHA,
            "processed_features_sha256": EXPECTED_RUNTIME_FEATURES_SHA,
        },
    }
    bundle_overrides = overrides.pop("production_bundle", None)
    component_overrides = overrides.pop("components", None)
    payload.update(overrides)
    if bundle_overrides:
        payload["production_bundle"].update(bundle_overrides)
    if component_overrides:
        payload["components"].update(component_overrides)
    return payload


def _full_expectations(*, include_runtime: bool = True) -> dict[str, str]:
    expectations = {
        "expected_bundle_id": EXPECTED_BUNDLE_ID,
        "expected_release_id": EXPECTED_RELEASE_ID,
        "expected_installed_manifest_sha256": EXPECTED_INSTALLED_MANIFEST_SHA,
        "expected_source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "expected_deployed_git_sha": EXPECTED_DEPLOYED_SHA,
        "expected_training_source_git_sha": EXPECTED_TRAINING_SHA,
        "expected_spec": EXPECTED_SPEC,
        "expected_model_sha256": EXPECTED_MODEL_SHA,
        "expected_no_odds_model_sha256": EXPECTED_NO_ODDS_SHA,
        "expected_logistic_model_sha256": EXPECTED_LOGISTIC_SHA,
        "expected_training_fights_sha256": EXPECTED_TRAINING_FIGHTS_SHA,
        "expected_training_features_sha256": EXPECTED_TRAINING_FEATURES_SHA,
        "expected_training_snapshot_max_event_date": EXPECTED_SNAPSHOT_DATE,
    }
    if include_runtime:
        expectations.update(
            {
                "expected_runtime_lookup_fights_sha256": EXPECTED_RUNTIME_FIGHTS_SHA,
                "expected_runtime_lookup_features_sha256": EXPECTED_RUNTIME_FEATURES_SHA,
            }
        )
    return expectations


def test_evaluate_readyz_requires_complete_release_and_live_trading_identity():
    passed, reason = deployment_gate.evaluate_readyz(
        200,
        _ready_payload(),
        **_full_expectations(),
    )

    assert passed is True
    assert EXPECTED_BUNDLE_ID in reason
    assert EXPECTED_RELEASE_ID in reason
    assert "live betting loop" in reason


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("bundle_id", "other-bundle", "bundle_id"),
        ("rich_release_id", "r-other", "rich_release_id"),
        ("installed_manifest_sha256", "8" * 64, "installed_manifest_sha256"),
        ("source_manifest_sha256", "8" * 64, "source_manifest_sha256"),
        ("deployed_git_sha", "c" * 40, "deployed_git_sha"),
        ("training_source_git_sha", "d" * 40, "training_source_git_sha"),
        ("model_spec_name", "wrong_spec", "model_spec_name"),
        ("model_sha256", "8" * 64, "model_sha256"),
        ("no_odds_model_sha256", "8" * 64, "no_odds_model_sha256"),
        ("logistic_model_sha256", "8" * 64, "logistic_model_sha256"),
        (
            "immutable_training_fights_sha256",
            "8" * 64,
            "immutable_training_fights_sha256",
        ),
        (
            "immutable_training_features_sha256",
            "8" * 64,
            "immutable_training_features_sha256",
        ),
        (
            "immutable_training_snapshot_max_event_date",
            "2026-07-31",
            "immutable_training_snapshot_max_event_date",
        ),
        ("processed_fights_sha256", "8" * 64, "processed_fights_sha256"),
        ("processed_features_sha256", "8" * 64, "processed_features_sha256"),
    ],
)
def test_evaluate_readyz_rejects_any_identity_mismatch(field, value, reason_fragment):
    passed, reason = deployment_gate.evaluate_readyz(
        200,
        _ready_payload(production_bundle={field: value}),
        **_full_expectations(),
    )

    assert passed is False
    assert reason_fragment in reason


def test_runtime_lookup_is_optional_but_never_substitutes_for_immutable_training():
    newer_lookup = _ready_payload(
        production_bundle={
            "processed_fights_sha256": "8" * 64,
            "processed_features_sha256": "9" * 64,
        }
    )
    passed, _ = deployment_gate.evaluate_readyz(
        200,
        newer_lookup,
        **_full_expectations(include_runtime=False),
    )
    assert passed is True

    del newer_lookup["production_bundle"]["immutable_training_fights_sha256"]
    passed, reason = deployment_gate.evaluate_readyz(
        200,
        newer_lookup,
        **_full_expectations(include_runtime=False),
    )
    assert passed is False
    assert "immutable_training_fights_sha256" in reason


@pytest.mark.parametrize(
    ("payload_override", "reason_fragment"),
    [
        ({"requested_live_mode": "dry_run"}, "requested_live_mode"),
        ({"effective_live_mode": "off"}, "effective_live_mode"),
        ({"armed_for_real": False}, "armed_for_real"),
        ({"trading_enabled": False}, "trading_enabled"),
        ({"trading_live": False}, "trading_live"),
        (
            {"components": {"betting_loop": {"state": "starting", "thread_alive": True}}},
            "state",
        ),
        (
            {"components": {"betting_loop": {"state": "running", "thread_alive": False}}},
            "thread_alive",
        ),
    ],
)
def test_evaluate_readyz_fails_closed_on_nonlive_runtime(payload_override, reason_fragment):
    passed, reason = deployment_gate.evaluate_readyz(
        200,
        _ready_payload(**payload_override),
        **_full_expectations(),
    )

    assert passed is False
    assert reason_fragment in reason


@pytest.mark.parametrize(
    ("status", "payload", "reason_fragment"),
    [
        (503, {"ready": False, "errors": ["startup still running"]}, "startup still running"),
        (200, {"ready": True}, "requested_live_mode"),
        (200, ["not", "an", "object"], "not an object"),
    ],
)
def test_evaluate_readyz_explains_failed_gate(status, payload, reason_fragment):
    passed, reason = deployment_gate.evaluate_readyz(
        status,
        payload,
        **_full_expectations(),
    )

    assert passed is False
    assert reason_fragment in reason


def test_parser_requires_complete_expected_release_identity():
    args = deployment_gate.build_parser().parse_args(
        [
            "--expected-bundle-id",
            EXPECTED_BUNDLE_ID,
            "--expected-release-id",
            EXPECTED_RELEASE_ID,
            "--expected-installed-manifest-sha256",
            EXPECTED_INSTALLED_MANIFEST_SHA,
            "--expected-source-manifest-sha256",
            EXPECTED_SOURCE_MANIFEST_SHA,
            "--expected-deployed-git-sha",
            EXPECTED_DEPLOYED_SHA,
            "--expected-training-source-git-sha",
            EXPECTED_TRAINING_SHA,
            "--expected-spec",
            EXPECTED_SPEC,
            "--expected-model-sha256",
            EXPECTED_MODEL_SHA,
            "--expected-no-odds-model-sha256",
            EXPECTED_NO_ODDS_SHA,
            "--expected-logistic-model-sha256",
            EXPECTED_LOGISTIC_SHA,
            "--expected-training-fights-sha256",
            EXPECTED_TRAINING_FIGHTS_SHA,
            "--expected-training-features-sha256",
            EXPECTED_TRAINING_FEATURES_SHA,
        ]
    )

    assert args.expected_bundle_id == EXPECTED_BUNDLE_ID
    assert args.successes_required == 2


def test_wait_requires_two_consecutive_matches_and_resets_after_flapping():
    stale = _ready_payload(production_bundle={"rich_release_id": "r-stale"})
    responses = iter(
        [
            OSError("temporary DNS failure"),
            (200, _ready_payload()),
            (200, stale),
            (200, _ready_payload()),
            (200, _ready_payload()),
        ]
    )
    clock = [0.0]
    messages: list[str] = []

    def fetch(_url, _timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    result = deployment_gate.wait_for_production_deployment(
        url="https://example.test/readyz",
        timeout_seconds=30,
        poll_seconds=2,
        request_timeout_seconds=1,
        successes_required=2,
        fetch=fetch,
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        emit=messages.append,
        **_full_expectations(),
    )

    assert result == _ready_payload()
    assert clock[0] == 8
    assert sum("confirmation 1/2" in message for message in messages) == 2
    assert "temporary DNS failure" in messages[0]


def test_wait_times_out_with_last_human_readable_gate_reason():
    clock = [0.0]

    with pytest.raises(TimeoutError, match="rich_release_id"):
        deployment_gate.wait_for_production_deployment(
            url="https://example.test/readyz",
            timeout_seconds=5,
            poll_seconds=2,
            request_timeout_seconds=1,
            successes_required=2,
            fetch=lambda _url, _timeout: (
                200,
                _ready_payload(production_bundle={"rich_release_id": "r-stale"}),
            ),
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            emit=lambda _message: None,
            **_full_expectations(),
        )

    assert clock[0] == 5


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_positive_seconds_rejects_nonfinite_and_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        deployment_gate._positive_seconds(value)


def test_invalid_expected_sha_fails_before_polling():
    expectations = _full_expectations()
    expectations["expected_training_fights_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="expected_training_fights_sha256"):
        deployment_gate.wait_for_production_deployment(
            url="https://example.test/readyz",
            timeout_seconds=5,
            poll_seconds=1,
            request_timeout_seconds=1,
            fetch=lambda *_args: pytest.fail("fetch must not be called"),
            **expectations,
        )


def test_wait_refuses_to_weaken_two_sample_requirement():
    with pytest.raises(ValueError, match="at least 2"):
        deployment_gate.wait_for_production_deployment(
            url="https://example.test/readyz",
            timeout_seconds=5,
            poll_seconds=1,
            request_timeout_seconds=1,
            successes_required=1,
            fetch=lambda *_args: pytest.fail("fetch must not be called"),
            **_full_expectations(),
        )


def test_fetch_readyz_uses_json_request_headers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return 200

        def read(self):
            return b'{"ready": true}'

    def fake_urlopen(request, timeout):
        captured["accept"] = request.get_header("Accept")
        captured["cache_control"] = request.get_header("Cache-control")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(deployment_gate, "urlopen", fake_urlopen)

    status, payload = deployment_gate.fetch_readyz("https://example.test/readyz", 7)

    assert status == 200
    assert payload == {"ready": True}
    assert captured == {
        "accept": "application/json",
        "cache_control": "no-cache",
        "timeout": 7,
    }
