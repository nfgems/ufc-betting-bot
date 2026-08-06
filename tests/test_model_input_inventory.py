from scripts.build_model_input_inventory import compare_inventories


def _inventory(*rows):
    return {
        "git_head": "a" * 40,
        "git_diff_sha256": "b" * 64,
        "inventory_sha256": "aggregate",
        "files": [
            {"path": path, "bytes": size, "sha256": sha256}
            for path, size, sha256 in rows
        ],
    }


def test_compare_inventories_accepts_identical_untracked_file_bytes():
    expected = _inventory(("src/untracked_model_code.py", 3, "abc"))
    actual = _inventory(("src/untracked_model_code.py", 3, "abc"))

    result = compare_inventories(expected, actual)

    assert result["ok"] is True
    assert result["added"] == []
    assert result["removed"] == []
    assert result["changed"] == []


def test_compare_inventories_reports_added_removed_and_changed_paths():
    expected = _inventory(
        ("data/raw/removed.csv", 1, "old"),
        ("src/changed.py", 2, "before"),
    )
    actual = _inventory(
        ("data/raw/added.csv", 1, "new"),
        ("src/changed.py", 3, "after"),
    )

    result = compare_inventories(expected, actual)

    assert result["ok"] is False
    assert result["added"] == ["data/raw/added.csv"]
    assert result["removed"] == ["data/raw/removed.csv"]
    assert result["changed"] == ["src/changed.py"]


def test_compare_inventories_rejects_base_or_tracked_diff_drift():
    expected = _inventory(("src/model.py", 3, "abc"))
    actual = _inventory(("src/model.py", 3, "abc"))
    actual["git_diff_sha256"] = "c" * 64

    result = compare_inventories(expected, actual)

    assert result["ok"] is False
    assert result["git_head_matches"] is True
    assert result["git_diff_matches"] is False
