import copy
import json

import pytest
from eth_abi import decode
from eth_account import Account
from eth_utils import keccak

from py_builder_relayer_client.builder.derive import derive
from py_builder_relayer_client.config import get_contract_config as get_relayer_contract_config

from src.polymarket.client import ClobClientWrapper
from src.polymarket import tracker as polymarket_tracker


TEST_PRIVATE_KEY = "0x" + ("11" * 32)
TEST_OWNER = Account.from_key(TEST_PRIVATE_KEY).address
TEST_SAFE = derive(TEST_OWNER, get_relayer_contract_config(137).safe_factory)


def _position(condition_hex: str, *, outcome_index: int, redeemable: bool = True, title: str = "Test Market") -> dict:
    return {
        "conditionId": condition_hex,
        "outcomeIndex": outcome_index,
        "redeemable": redeemable,
        "title": title,
        "size": 5,
    }


def test_group_redeemable_positions_groups_by_condition_and_outcome():
    groups = ClobClientWrapper._group_redeemable_positions(
        [
            _position("0x" + ("22" * 32), outcome_index=1, title="B Market"),
            _position("0x" + ("11" * 32), outcome_index=0, title="A Market"),
            _position("0x" + ("11" * 32), outcome_index=1, title="A Market"),
            _position("0x" + ("33" * 32), outcome_index=0, redeemable=False, title="Ignored"),
        ]
    )

    assert [group["condition_id"] for group in groups] == [
        "0x" + ("11" * 32),
        "0x" + ("22" * 32),
    ]
    assert groups[0]["index_sets"] == [1, 2]
    assert groups[0]["position_count"] == 2
    assert groups[1]["index_sets"] == [2]


def test_redeem_positions_builds_conditional_tokens_call(monkeypatch):
    wrapper = ClobClientWrapper(private_key=TEST_PRIVATE_KEY, funder_address=TEST_SAFE)
    calls = []

    def fake_relayer_request(method, path, *, params=None, body=None, auth_required=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "auth_required": auth_required,
            }
        )
        if path == "/deployed":
            return {"deployed": True}
        if path == "/nonce":
            return {"nonce": "7"}
        if path == "/submit":
            return {"transactionID": "tx-1", "transactionHash": "0xabc"}
        raise AssertionError(f"Unexpected relayer call: {path}")

    monkeypatch.setattr(wrapper, "_relayer_request", fake_relayer_request)
    monkeypatch.setattr(wrapper, "can_redeem_positions", lambda: True)

    result = wrapper.redeem_positions(
        "0x" + ("44" * 32),
        [1, 2],
        wait=False,
    )

    submit_body = next(call["body"] for call in calls if call["path"] == "/submit")
    selector = bytes.fromhex(submit_body["data"][2:10])
    decoded = decode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        bytes.fromhex(submit_body["data"][10:]),
    )

    assert result["transaction_id"] == "tx-1"
    assert submit_body["proxyWallet"].lower() == TEST_SAFE.lower()
    assert submit_body["nonce"] == "7"
    assert selector == keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
    assert decoded[1] == bytes(32)
    assert decoded[2] == bytes.fromhex("44" * 32)
    assert list(decoded[3]) == [1, 2]


def test_redeem_redeemable_positions_reports_missing_config():
    wrapper = ClobClientWrapper(private_key=TEST_PRIVATE_KEY, funder_address=TEST_SAFE)

    summary = wrapper.redeem_redeemable_positions(
        positions=[
            _position("0x" + ("55" * 32), outcome_index=0, redeemable=True),
        ],
        wait=False,
    )

    assert summary["reason"] == "redeemer_not_configured"
    assert summary["redeemable_positions"] == 1
    assert summary["redeemable_conditions"] == 1
    assert summary["redeemed_positions"] == 0


def test_redeem_redeemable_positions_tracks_submitted_counts_without_wait(monkeypatch):
    wrapper = ClobClientWrapper(private_key=TEST_PRIVATE_KEY, funder_address=TEST_SAFE)

    monkeypatch.setattr(wrapper, "can_redeem_positions", lambda: True)
    monkeypatch.setattr(
        wrapper,
        "redeem_positions",
        lambda condition_id, index_sets, **kwargs: {
            "condition_id": condition_id,
            "index_sets": index_sets,
            "transaction_id": "tx-1",
            "transaction_hash": "0xabc",
        },
    )

    summary = wrapper.redeem_redeemable_positions(
        positions=[
            _position("0x" + ("66" * 32), outcome_index=0, redeemable=True, title="A"),
            _position("0x" + ("66" * 32), outcome_index=1, redeemable=True, title="A"),
        ],
        wait=False,
    )

    assert summary["reason"] == ""
    assert summary["submitted_conditions"] == 1
    assert summary["submitted_positions"] == 2
    assert summary["redeemed_conditions"] == 0
    assert summary["redeemed_positions"] == 0
    assert summary["waited_for_confirmation"] is False


class _FakeRedeemClient:
    def __init__(self, summary: dict, *, transactions: dict[str, dict | None] | None = None):
        self._summary = copy.deepcopy(summary)
        self._transactions = transactions or {}
        self.redeem_calls = 0
        self.transaction_checks: list[str] = []

    def redeem_redeemable_positions(self, *, wait: bool = False):
        self.redeem_calls += 1
        summary = copy.deepcopy(self._summary)
        summary["waited_for_confirmation"] = bool(wait)
        return summary

    def get_relayer_transaction(self, transaction_id: str):
        self.transaction_checks.append(transaction_id)
        payload = self._transactions.get(transaction_id)
        return copy.deepcopy(payload) if isinstance(payload, dict) else payload


def test_auto_redeem_helper_applies_cooldown_for_background_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(polymarket_tracker, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS", "6")

    empty_summary = {
        "redeemable_positions": 0,
        "redeemable_conditions": 0,
        "submitted_conditions": 0,
        "submitted_positions": 0,
        "redeemed_conditions": 0,
        "redeemed_positions": 0,
        "results": [],
        "errors": [],
        "reason": "no_redeemable_positions",
    }

    first_client = _FakeRedeemClient(empty_summary)
    first = polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=first_client,
        wait=False,
        source="auto",
    )
    second_client = _FakeRedeemClient(empty_summary)
    second = polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=second_client,
        wait=False,
        source="auto",
    )

    assert first["reason"] == "no_redeemable_positions"
    assert first_client.redeem_calls == 1
    assert second["reason"] == "auto_redeem_cooldown"
    assert second_client.redeem_calls == 0
    assert second["cooldown_remaining_seconds"] > 0


def test_auto_redeem_helper_skips_when_previous_submission_is_still_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(polymarket_tracker, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS", "0")

    submitted_summary = {
        "redeemable_positions": 2,
        "redeemable_conditions": 1,
        "submitted_conditions": 1,
        "submitted_positions": 2,
        "redeemed_conditions": 0,
        "redeemed_positions": 0,
        "results": [
            {
                "condition_id": "0x" + ("77" * 32),
                "transaction_id": "tx-1",
                "transaction_hash": "0xabc",
                "title": "Pending Market",
                "position_count": 2,
            }
        ],
        "errors": [],
        "reason": "",
    }

    first = polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=_FakeRedeemClient(submitted_summary),
        wait=False,
        source="auto",
    )
    second_client = _FakeRedeemClient(
        {
            "redeemable_positions": 0,
            "redeemable_conditions": 0,
            "submitted_conditions": 0,
            "submitted_positions": 0,
            "redeemed_conditions": 0,
            "redeemed_positions": 0,
            "results": [],
            "errors": [],
            "reason": "no_redeemable_positions",
        },
        transactions={"tx-1": {"state": "STATE_PENDING", "transactionHash": "0xabc"}},
    )
    second = polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=second_client,
        wait=False,
        source="auto",
    )

    assert first["submitted_conditions"] == 1
    assert second["reason"] == "redeem_submission_pending"
    assert second_client.redeem_calls == 0
    assert second_client.transaction_checks == ["tx-1"]
    assert second["pending_transactions"][0]["transaction_id"] == "tx-1"


def test_auto_redeem_helper_clears_stale_missing_pending_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(polymarket_tracker, "LOGS_DIR", tmp_path)
    monkeypatch.setenv("POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS", "0")
    monkeypatch.setenv("POLYMARKET_AUTO_REDEEM_PENDING_TTL_HOURS", "1")

    submitted_summary = {
        "redeemable_positions": 1,
        "redeemable_conditions": 1,
        "submitted_conditions": 1,
        "submitted_positions": 1,
        "redeemed_conditions": 0,
        "redeemed_positions": 0,
        "results": [
            {
                "condition_id": "0x" + ("88" * 32),
                "transaction_id": "tx-stale",
                "transaction_hash": "0xdef",
                "title": "Stale Market",
                "position_count": 1,
            }
        ],
        "errors": [],
        "reason": "",
    }

    polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=_FakeRedeemClient(submitted_summary),
        wait=False,
        source="auto",
    )

    state_path = tmp_path / polymarket_tracker.REDEEM_STATE_FILENAME
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state["pending_transactions"][0]["submitted_at"] = "2020-01-01T00:00:00+00:00"
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)

    second_client = _FakeRedeemClient(
        {
            "redeemable_positions": 0,
            "redeemable_conditions": 0,
            "submitted_conditions": 0,
            "submitted_positions": 0,
            "redeemed_conditions": 0,
            "redeemed_positions": 0,
            "results": [],
            "errors": [],
            "reason": "no_redeemable_positions",
        },
        transactions={"tx-stale": None},
    )
    second = polymarket_tracker.auto_redeem_positions_from_polymarket(
        clob_client=second_client,
        wait=False,
        source="auto",
    )

    assert second["reason"] == "no_redeemable_positions"
    assert second_client.transaction_checks == ["tx-stale"]
    assert second_client.redeem_calls == 1
