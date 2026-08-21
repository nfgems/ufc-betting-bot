from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from portable_regression_support import (
    PortableContractError,
    verify_sha256_bindings,
)


FROZEN_PREFIT_SHA256 = (
    "288f544beb64955ec0560ed97f581b7f22a9342422b1b15dc0d14e0da00be75c"
)
HISTORICAL_HANDOFF_SHA256 = (
    "b426d11c3adfd4bf768aabbfaf033f5bd0f248cfd74a9238ad1dbbb2a524c49e"
)
HISTORICAL_TEST_SHA256 = {
    "staged_bundle_test": (
        "649b4ee7b32534035146938880a23603605f9a76c6f3b6806a9e39c40984cc1d"
    ),
    "offline_candidate_test": (
        "142149fe4bde6b4ac162444a3076c3ac9ab9577d8443742ab958896d5ad0dde7"
    ),
    "confirmation_fold_test": (
        "dd0deedb302a4d685cfb5f3b9536f9747eaecccd918be7c0036549de5c25f4cc"
    ),
}


def _expected_bindings() -> dict[str, str]:
    return {
        "immutable_pre_fit_contract": FROZEN_PREFIT_SHA256,
        "receipt_pre_fit_contract": FROZEN_PREFIT_SHA256,
        "manifest_pre_fit_contract": FROZEN_PREFIT_SHA256,
        "pre_fit_historical_handoff": HISTORICAL_HANDOFF_SHA256,
        "receipt_historical_handoff": HISTORICAL_HANDOFF_SHA256,
        **{
            f"historical_{label}": digest
            for label, digest in HISTORICAL_TEST_SHA256.items()
        },
    }


def test_synthetic_chain_preserves_immutable_prefit_and_historical_bindings():
    expected = _expected_bindings()
    synthetic_contract_chain = dict(expected)

    assert verify_sha256_bindings(synthetic_contract_chain, expected) == expected


def test_historical_handoff_binding_fails_closed_against_later_drift():
    expected = _expected_bindings()
    later_handoff_sha256 = hashlib.sha256(
        b"later append-only living-handoff state"
    ).hexdigest()
    assert later_handoff_sha256 != HISTORICAL_HANDOFF_SHA256
    drifted = {**expected, "receipt_historical_handoff": later_handoff_sha256}

    with pytest.raises(PortableContractError) as exc_info:
        verify_sha256_bindings(drifted, expected)

    message = str(exc_info.value)
    assert "receipt_historical_handoff" in message
    assert HISTORICAL_HANDOFF_SHA256 in message
    assert later_handoff_sha256 in message


def test_portable_fixture_uses_short_repository_local_pytest_basetemp(
    tmp_path_factory,
):
    leaf = tmp_path_factory.mktemp("p")
    pytest_runtime = (Path.cwd() / ".pytest-runtime").resolve()

    assert leaf.resolve().is_relative_to(pytest_runtime)
    assert leaf.parent.name == "basetemp"
