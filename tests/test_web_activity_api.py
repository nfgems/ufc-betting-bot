from src.web import app as web_app


def test_api_bot_activity_returns_entries_with_snapshot_headers(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-11 03:08:53,176 [INFO] src.polymarket.executor: "
        "Skipping Charles Johnson: already have open bet on market 1510646\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["X-Bot-Activity-Server-Time"]
    assert response.headers["X-Bot-Activity-Log-MTime"]
    assert response.headers["X-Bot-Activity-Last-Entry"] == "2026-03-11 03:08:53"

    payload = response.get_json()
    assert isinstance(payload, list)
    assert payload[0]["level"] == "INFO"
    assert payload[0]["source"] == "src.polymarket.executor"
    assert "Skipping Charles Johnson" in payload[0]["message"]
