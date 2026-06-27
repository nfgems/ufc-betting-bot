from src import config


def test_resolve_default_models_dir_prefers_image_hosted_models(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"
    legacy_models_dir = project_root / "models"
    legacy_models_dir.mkdir(parents=True)
    (legacy_models_dir / "xgboost_model.pkl").write_bytes(b"image")

    resolved = config._resolve_default_models_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
    )

    assert resolved == legacy_models_dir


def test_resolve_default_models_dir_returns_image_hosted_path_even_without_artifacts(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"
    legacy_models_dir = project_root / "models"

    resolved = config._resolve_default_models_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
    )

    assert resolved == legacy_models_dir


def test_resolve_default_logs_dir_prefers_hosted_volume_mount(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"
    volume_mount = project_root / "logs"

    resolved = config._resolve_default_logs_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
        hosted_volume_mount=volume_mount,
    )

    assert resolved == volume_mount


def test_logs_dir_from_env_collapses_legacy_hosted_logs_child(tmp_path, monkeypatch):
    project_root = tmp_path / "app"
    volume_mount = project_root / "logs"
    monkeypatch.setenv("TEST_UFC_LOGS_DIR", str(volume_mount / "logs"))

    resolved = config._logs_dir_from_env(
        "TEST_UFC_LOGS_DIR",
        volume_mount,
        volume_mount,
        hosted_volume_mount=volume_mount,
    )

    assert resolved == volume_mount


def test_logs_dir_from_env_preserves_explicit_non_duplicate_path(tmp_path, monkeypatch):
    project_root = tmp_path / "app"
    volume_mount = project_root / "logs"
    explicit_logs = project_root / "separate-logs"
    monkeypatch.setenv("TEST_UFC_LOGS_DIR", str(explicit_logs))

    resolved = config._logs_dir_from_env(
        "TEST_UFC_LOGS_DIR",
        volume_mount,
        volume_mount,
        hosted_volume_mount=volume_mount,
    )

    assert resolved == explicit_logs


def test_resolve_default_logs_dir_falls_back_to_data_logs_without_hosted_volume(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"

    resolved = config._resolve_default_logs_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
        hosted_volume_mount=None,
    )

    assert resolved == data_dir / "logs"


def test_line_and_injury_market_alerts_are_advisory_by_default():
    assert config.LINE_MOVEMENT_FILTER is False
    assert config.LINE_SHARP_BLOCK is False
    assert config.INJURY_BLOCK_BETS is False
