from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_railway_deployment_waits_for_runtime_readiness() -> None:
    config = tomllib.loads((PROJECT_ROOT / "railway.toml").read_text(encoding="utf-8"))
    deploy = config["deploy"]

    assert deploy["healthcheckPath"] == "/readyz"
    assert deploy["healthcheckTimeout"] == 300
    assert "healthcheck" not in deploy
