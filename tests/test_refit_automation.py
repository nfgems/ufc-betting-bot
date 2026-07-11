"""Tests for the unattended-refit plumbing: atomic-write retry, built_at
stamping, and BFO recovered-batch overwrite protection."""

import json
import sys

import pandas as pd
import pytest

import scripts.recover_bfo_moneyline_gaps as recover_bfo
import scripts.reconcile_production_bundle_manifest as reconcile_script
from src.data import io_utils


def test_write_csv_atomically_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    target.write_text("old\n", encoding="utf-8")

    real_replace = io_utils.os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(5, "Access is denied", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", flaky_replace)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)

    io_utils.write_csv_atomically(pd.DataFrame({"a": [1]}), target)

    assert attempts["count"] == 3
    assert "a" in target.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_write_csv_atomically_reraises_after_exhausted_retries(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"

    def always_denied(src, dst):
        raise PermissionError(5, "Access is denied", str(dst))

    monkeypatch.setattr(io_utils.os, "replace", always_denied)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        io_utils.write_csv_atomically(pd.DataFrame({"a": [1]}), target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_stamp_manifest_built_at_now_updates_only_timestamps(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "model_spec_name": "prod_spec",
                "built_at": "2026-06-11T06:05:30Z",
                "manifest_updated_at": "2026-06-11T06:05:30Z",
                "selection_basis": "evidence",
            }
        ),
        encoding="utf-8",
    )

    stamp = reconcile_script.stamp_manifest_built_at_now(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["built_at"] == stamp
    assert payload["manifest_updated_at"] == stamp
    assert payload["built_at"] != "2026-06-11T06:05:30Z"
    assert payload["bundle_id"] == "bundle-1"
    assert payload["selection_basis"] == "evidence"
    assert not manifest_path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_recover_bfo_refuses_to_overwrite_existing_batch(tmp_path, monkeypatch):
    recovered = tmp_path / "historical_odds_bfo_recovered_existing.csv"
    recovered.write_text("event_date,fighter_a,fighter_b\n2026-06-27,A,B\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_bfo_moneyline_gaps.py",
            "--recovered-output",
            str(recovered),
            "--unresolved-output",
            str(tmp_path / "unresolved.csv"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        recover_bfo.main()

    assert excinfo.value.code == 2
    assert "2026-06-27,A,B" in recovered.read_text(encoding="utf-8")
