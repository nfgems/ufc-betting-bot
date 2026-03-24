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


def test_api_bot_activity_snapshot_returns_metadata_and_entries_together(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-11 04:28:42,975 [INFO] src.polymarket.executor: "
        "Limit bid placed for Charles Johnson\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity-snapshot")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["server_time"]
    assert payload["last_entry"] == "2026-03-11 04:28:42"
    assert payload["log_mtime"]
    assert payload["entry_count"] == 1
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["message"] == "Limit bid placed for Charles Johnson"


def test_api_bot_activity_downgrades_handled_geoblock_warning(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-12 04:11:45,000 [WARNING] src.polymarket.executor: "
        "Failed to place limit bid for Movsar Evloev: "
        "PolyApiException[status_code=403, error_message={'error': "
        "'Trading restricted in your region'}]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["level"] == "INFO"
    assert payload[0]["raw_level"] == "WARNING"
    assert payload[0]["activity_kind"] == "handled_order_rejection"
    assert "Failed to place limit bid for Movsar Evloev" in payload[0]["message"]


def test_api_bot_activity_snapshot_downgrades_handled_geoblock_warning(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-12 04:11:45,000 [WARNING] src.polymarket.executor: "
        "Failed to place limit bid for Movsar Evloev: "
        "PolyApiException[status_code=403, error_message={'error': "
        "'Trading restricted in your region, please refer to available regions - "
        "https://docs.polymarket.com/developers/CLOB/geoblock'}]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity-snapshot")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["entry_count"] == 1
    assert payload["entries"][0]["level"] == "INFO"
    assert payload["entries"][0]["raw_level"] == "WARNING"
    assert payload["entries"][0]["activity_kind"] == "handled_order_rejection"


def test_api_bot_activity_can_filter_tennis_entries_server_side(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-24 03:19:30,000 [INFO] src.bot: Built 202 features for Abdulrakhman Yakhyaev vs Brendson Ribeiro\n"
        "2026-03-24 03:20:00,000 [INFO] src.bot: Checking tennis trading authorization: source=portfolio\n"
        "2026-03-24 03:20:01,000 [INFO] src.strategy.tennis_llm_operator: Tennis LLM veto disabled\n"
        "2026-03-24 03:20:02,000 [INFO] werkzeug: 100.64.0.2 - - [24/Mar/2026 03:20:02] \"GET /tennis HTTP/1.1\" 200 -\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity?sport=tennis")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    # 2 tennis entries + 1 general (werkzeug) entry — sport filter includes both
    assert len(payload) == 3
    sports = {entry["sport"] for entry in payload}
    assert sports <= {"tennis", "general"}
    assert all("Abdulrakhman Yakhyaev" not in entry["message"] for entry in payload)


def test_api_bot_activity_snapshot_can_filter_ufc_entries_server_side(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-24 03:19:30,000 [INFO] src.bot: Built 202 features for Abdulrakhman Yakhyaev vs Brendson Ribeiro\n"
        "2026-03-24 03:20:00,000 [INFO] src.bot: Checking tennis trading authorization: source=portfolio\n"
        '2026-03-24 03:20:05,000 [INFO] werkzeug: 100.64.0.2 - - [24/Mar/2026 03:20:05] "GET /api/summary HTTP/1.1" 200 -\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity-snapshot?sport=ufc")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    # 1 ufc entry + 1 general (werkzeug) entry — tennis entry excluded
    assert payload["entry_count"] == 2
    sports = {e["sport"] for e in payload["entries"]}
    assert sports == {"ufc", "general"}
    assert any("Built 202 features" in e["message"] for e in payload["entries"])
    assert all("tennis trading authorization" not in e["message"] for e in payload["entries"])


def test_api_bot_activity_keeps_non_geoblock_403_as_warning(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-12 04:11:45,000 [WARNING] src.polymarket.executor: "
        "Failed to place limit bid for Movsar Evloev: "
        "PolyApiException[status_code=403, error_message={'error': 'Forbidden'}]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/bot-activity")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload[0]["level"] == "WARNING"
    assert payload[0]["raw_level"] == "WARNING"
    assert "activity_kind" not in payload[0]


def test_api_significant_actions_can_filter_tennis_entries_server_side(tmp_path, monkeypatch):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "2026-03-24 03:19:30,000 [INFO] src.strategy.tennis_llm_operator: Market order placed for Iga Swiatek\n"
        "2026-03-24 03:19:31,000 [INFO] src.polymarket.executor: Market order placed for Charles Oliveira\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/significant-actions?sport=tennis")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload) == 1
    assert payload[0]["sport"] == "tennis"
    assert "Iga Swiatek" in payload[0]["message"]
    assert all("Charles Oliveira" not in entry["message"] for entry in payload)


def test_api_geoblock_status_returns_live_transport_diagnostics(monkeypatch):
    class _FakeClob:
        def get_geoblock_status(self):
            return {
                "status_code": 200,
                "blocked": False,
                "ip": "163.176.191.39",
                "country": "BR",
                "region": "Sao Paulo",
                "error": "",
            }

    import src.polymarket.client as client_mod

    monkeypatch.setenv("CLOB_PROXY_URL", "http://user:pass@163.176.191.39:3128")
    monkeypatch.setattr(client_mod, "_proxy_patched", True)
    monkeypatch.setattr(web_app, "_clob_client", _FakeClob())
    client = web_app.app.test_client()

    response = client.get("/api/geoblock-status")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["available"] is True
    assert payload["blocked"] is False
    assert payload["ip"] == "163.176.191.39"
    assert payload["country"] == "BR"
    assert payload["region"] == "Sao Paulo"
    assert payload["proxy_configured"] is True
    assert payload["proxy_enabled"] is True
    assert payload["proxy_target"] == "163.176.191.39:3128"


def test_activity_page_disables_browser_caching():
    client = web_app.app.test_client()

    response = client.get("/activity")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["Expires"] == "0"
