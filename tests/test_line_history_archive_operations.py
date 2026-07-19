import gzip
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone

import pytest

from src.data import line_history_archive as archive


PREFIX = "ufc/line-history/v1"
ODDS_KEY = f"{PREFIX}/odds/2026/01/odds_20260101_120000.csv.gz"
POLYMARKET_KEY = (
    f"{PREFIX}/polymarket/2026/01/polymarket_20260101_120000.csv.gz"
)


def _configure_archive(monkeypatch):
    monkeypatch.setattr(archive, "LINE_HISTORY_ARCHIVE_BUCKET", "archive-bucket")
    monkeypatch.setattr(archive, "LINE_HISTORY_ARCHIVE_PREFIX", PREFIX)


class _ListClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def list_objects_v2(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _GetClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _archive_response(
    source: bytes,
    *,
    mtime_ns: int = 1_767_268_800_000_000_000,
    source_size: int | None = None,
    source_sha256: str | None = None,
    compressed: bytes | None = None,
):
    metadata = {
        "source-size-bytes": str(len(source) if source_size is None else source_size),
        "source-mtime-ns": str(mtime_ns),
        "source-sha256": source_sha256 or hashlib.sha256(source).hexdigest(),
    }
    return {
        "Body": io.BytesIO(
            gzip.compress(source, compresslevel=6, mtime=0)
            if compressed is None
            else compressed
        ),
        "ContentEncoding": "gzip",
        "Metadata": metadata,
    }


def test_archive_list_scopes_prefix_and_paginates_to_limit(monkeypatch):
    _configure_archive(monkeypatch)
    modified = datetime(2026, 1, 2, tzinfo=timezone.utc)
    second_key = f"{PREFIX}/odds/2026/01/odds_20260102_120000.csv.gz"
    third_key = f"{PREFIX}/odds/2026/01/odds_20260103_120000.csv.gz"
    client = _ListClient(
        [
            {
                "Contents": [
                    {"Key": ODDS_KEY, "Size": 100, "LastModified": modified, "ETag": '"a"'},
                    {"Key": second_key, "Size": 200, "LastModified": modified, "ETag": '"b"'},
                ],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            },
            {
                "Contents": [
                    {"Key": third_key, "Size": 300, "LastModified": modified, "ETag": '"c"'},
                ],
                "IsTruncated": True,
                "NextContinuationToken": "page-3",
            },
        ]
    )

    page = archive.list_line_history_archive(
        category="odds",
        year=2026,
        month=1,
        limit=3,
        client=client,
    )

    assert [item.key for item in page.objects] == [ODDS_KEY, second_key, third_key]
    assert [item.etag for item in page.objects] == ["a", "b", "c"]
    assert page.next_cursor == "page-3"
    assert client.calls == [
        {
            "Bucket": "archive-bucket",
            "Prefix": f"{PREFIX}/odds/2026/01/",
            "MaxKeys": 3,
        },
        {
            "Bucket": "archive-bucket",
            "Prefix": f"{PREFIX}/odds/2026/01/",
            "MaxKeys": 1,
            "ContinuationToken": "page-2",
        },
    ]


def test_archive_list_forwards_input_cursor_and_ignores_unrelated_keys(monkeypatch):
    _configure_archive(monkeypatch)
    client = _ListClient(
        [
            {
                "Contents": [
                    {
                        "Key": "other/prefix/odds_20260101_120000.csv.gz",
                        "Size": 1,
                        "LastModified": datetime.now(timezone.utc),
                    }
                ],
                "IsTruncated": False,
            }
        ]
    )

    page = archive.list_line_history_archive(limit=10, cursor="previous", client=client)

    assert page.objects == ()
    assert page.next_cursor is None
    assert client.calls[0]["ContinuationToken"] == "previous"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"year": 2026}, "--year requires --category"),
        ({"category": "odds", "month": 1}, "--month requires --year"),
        ({"category": "invalid"}, "category must be"),
        ({"limit": 0}, "limit must be"),
        ({"limit": 10_001}, "limit must be"),
        ({"cursor": ""}, "cursor must be"),
    ],
)
def test_archive_list_rejects_invalid_filters(monkeypatch, kwargs, message):
    _configure_archive(monkeypatch)
    with pytest.raises(ValueError, match=message):
        archive.list_line_history_archive(client=_ListClient([]), **kwargs)


@pytest.mark.parametrize(
    "key",
    [
        "../ufc/line-history/v1/odds/2026/01/odds_20260101_120000.csv.gz",
        f"{PREFIX}/odds/2026/../odds_20260101_120000.csv.gz",
        f"{PREFIX}/odds/2026/13/odds_20260101_120000.csv.gz",
        f"{PREFIX}/odds/2026/01/odds_20260230_120000.csv.gz",
        f"{PREFIX}/odds/2026/01/polymarket_20260101_120000.csv.gz",
        f"{PREFIX}/odds/2026/01/odds_20260101_120000.csv",
        f"{PREFIX}\\odds\\2026\\01\\odds_20260101_120000.csv.gz",
        "other/odds/2026/01/odds_20260101_120000.csv.gz",
    ],
)
def test_archive_key_validation_rejects_unsafe_or_foreign_keys(monkeypatch, key):
    _configure_archive(monkeypatch)
    with pytest.raises(ValueError):
        archive.validate_line_history_archive_key(key)


def test_restore_streams_validates_and_preserves_source_mtime(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    source = b"fighter_a,fighter_b\nAlpha,Beta\n"
    mtime_ns = 1_767_268_800_000_000_000
    response = _archive_response(source, mtime_ns=mtime_ns)
    body = response["Body"]
    client = _GetClient(response)

    restored = archive.restore_line_history_snapshot(
        ODDS_KEY,
        output_dir=tmp_path / "restored",
        client=client,
    )

    assert restored == (tmp_path / "restored" / "odds_20260101_120000.csv").resolve()
    assert restored.read_bytes() == source
    assert restored.stat().st_mtime_ns == mtime_ns
    assert client.calls == [{"Bucket": "archive-bucket", "Key": ODDS_KEY}]
    assert body.closed
    assert not list(restored.parent.glob(".*.tmp"))


def test_restore_accepts_legacy_object_without_sha256_metadata(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    source = b"fighter_a,fighter_b\nLegacy,Snapshot\n"
    response = _archive_response(source)
    response["Metadata"].pop("source-sha256")

    restored = archive.restore_line_history_snapshot(
        ODDS_KEY,
        output_dir=tmp_path,
        client=_GetClient(response),
    )

    assert restored.read_bytes() == source


def test_restore_refuses_existing_destination_before_download(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    output_dir = tmp_path / "restored"
    output_dir.mkdir()
    destination = output_dir / "odds_20260101_120000.csv"
    destination.write_text("keep me")
    client = _GetClient(_archive_response(b"replacement"))

    with pytest.raises(FileExistsError, match="--force"):
        archive.restore_line_history_snapshot(ODDS_KEY, output_dir=output_dir, client=client)

    assert destination.read_text() == "keep me"
    assert client.calls == []


def test_force_restore_keeps_existing_file_when_gzip_is_corrupt(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    output_dir = tmp_path / "restored"
    output_dir.mkdir()
    destination = output_dir / "odds_20260101_120000.csv"
    destination.write_text("original")
    response = _archive_response(b"replacement", compressed=b"not gzip")

    with pytest.raises((gzip.BadGzipFile, EOFError)):
        archive.restore_line_history_snapshot(
            ODDS_KEY,
            output_dir=output_dir,
            overwrite=True,
            client=_GetClient(response),
        )

    assert destination.read_text() == "original"
    assert not list(output_dir.glob(".*.tmp"))


@pytest.mark.parametrize("failure", ["size", "checksum", "limit"])
def test_restore_rejects_integrity_and_size_failures(monkeypatch, tmp_path, failure):
    _configure_archive(monkeypatch)
    source = b"1234567890"
    kwargs = {}
    if failure == "size":
        response = _archive_response(source, source_size=len(source) + 1)
    elif failure == "checksum":
        response = _archive_response(source, source_sha256="0" * 64)
    else:
        response = _archive_response(source)
        kwargs["max_uncompressed_bytes"] = len(source) - 1

    with pytest.raises(ValueError):
        archive.restore_line_history_snapshot(
            ODDS_KEY,
            output_dir=tmp_path,
            client=_GetClient(response),
            **kwargs,
        )

    assert not (tmp_path / "odds_20260101_120000.csv").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_restore_closes_body_when_metadata_is_malformed(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    response = _archive_response(b"data")
    response["Metadata"]["source-size-bytes"] = "not-an-integer"
    body = response["Body"]

    with pytest.raises(ValueError, match="source-size-bytes"):
        archive.restore_line_history_snapshot(
            ODDS_KEY,
            output_dir=tmp_path,
            client=_GetClient(response),
        )

    assert body.closed
    assert not list(tmp_path.glob(".*.tmp"))


def test_force_restore_atomically_replaces_valid_existing_file(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    destination = tmp_path / "polymarket_20260101_120000.csv"
    destination.write_text("old")
    source = b"market,price\nexample,0.5\n"

    restored = archive.restore_line_history_snapshot(
        POLYMARKET_KEY,
        output_dir=tmp_path,
        overwrite=True,
        client=_GetClient(_archive_response(source)),
    )

    assert restored == destination.resolve()
    assert destination.read_bytes() == source


def test_upload_records_source_sha256_metadata(monkeypatch, tmp_path):
    _configure_archive(monkeypatch)
    source = tmp_path / "odds_20260101T120000.csv"
    source.write_bytes(b"fighter_a,fighter_b\nAlpha,Beta\n")
    os.utime(source, (1_767_268_800, 1_767_268_800))
    uploads = []

    class Client:
        def put_object(self, **kwargs):
            uploads.append(kwargs)

    archive.archive_line_history_snapshot(source, client=Client())

    assert uploads[0]["Metadata"]["source-sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_upload_rejects_key_that_restore_tooling_cannot_recover(monkeypatch, tmp_path):
    source = tmp_path / "odds_manual_copy.csv"
    source.write_text("fighter_a,fighter_b\nAlpha,Beta\n", encoding="utf-8")
    uploads = []

    class FakeClient:
        def put_object(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(archive, "LINE_HISTORY_ARCHIVE_BUCKET", "archive")

    with pytest.raises(ValueError, match="not an odds or Polymarket CSV snapshot"):
        archive.archive_line_history_snapshot(source, client=FakeClient())

    assert uploads == []
    assert source.exists()


def test_line_history_archive_cli_list_json_and_restore_default(monkeypatch, capsys):
    import src.bot as bot

    modified = datetime(2026, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(
        archive,
        "list_line_history_archive",
        lambda **_kwargs: archive.ArchivePage(
            objects=(archive.ArchivedSnapshot(ODDS_KEY, 123, modified, "etag"),),
            next_cursor="next-page",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bot.py", "line-history-archive", "list", "--category", "odds", "--json"],
    )

    bot.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["objects"][0]["key"] == ODDS_KEY
    assert payload["next_cursor"] == "next-page"

    calls = []

    def fake_restore(key, **kwargs):
        calls.append((key, kwargs))
        return kwargs["output_dir"] / "odds_20260101_120000.csv"

    monkeypatch.setattr(archive, "restore_line_history_snapshot", fake_restore)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bot.py", "line-history-archive", "restore", ODDS_KEY],
    )

    bot.main()
    assert calls == [
        (
            ODDS_KEY,
            {
                "output_dir": bot.DATA_DIR / "restored_line_history",
                "overwrite": False,
            },
        )
    ]
    assert "Restored archived snapshot" in capsys.readouterr().out
