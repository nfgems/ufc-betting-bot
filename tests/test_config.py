from src import config


def test_resolve_default_models_dir_prefers_persistent_hosted_models(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"
    legacy_models_dir = project_root / "models"
    persistent_models_dir = data_dir / "models"

    legacy_models_dir.mkdir(parents=True)
    persistent_models_dir.mkdir(parents=True)
    (legacy_models_dir / "xgboost_model.pkl").write_bytes(b"legacy")
    (persistent_models_dir / "xgboost_model.pkl").write_bytes(b"persistent")

    resolved = config._resolve_default_models_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
    )

    assert resolved == persistent_models_dir


def test_resolve_default_models_dir_falls_back_to_legacy_hosted_models(tmp_path):
    project_root = tmp_path / "app"
    data_dir = project_root / "data"
    legacy_models_dir = project_root / "models"

    legacy_models_dir.mkdir(parents=True)
    (legacy_models_dir / "xgboost_model.pkl").write_bytes(b"legacy")

    resolved = config._resolve_default_models_dir(
        project_root,
        data_dir,
        hosted_project_root=project_root,
    )

    assert resolved == legacy_models_dir
