import logging

from src.polymarket import market_lookup, markets
from src.web import app as web_app


def test_market_lookup_does_not_duplicate_gamma_terminal_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        market_lookup,
        "get_ufc_fight_markets",
        lambda: (_ for _ in ()).throw(
            markets.GammaEventsUnavailableError("Gamma unavailable")
        ),
    )

    with caplog.at_level(logging.INFO):
        assert market_lookup.load_supported_market_token_lookup() == {}

    matching = [
        record for record in caplog.records if "Gamma unavailable" in record.getMessage()
    ]
    assert len(matching) == 1
    assert matching[0].levelno == logging.INFO


def test_dashboard_token_map_does_not_duplicate_gamma_terminal_warning(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        markets,
        "get_ufc_fight_markets",
        lambda: (_ for _ in ()).throw(
            markets.GammaEventsUnavailableError("Gamma unavailable")
        ),
    )

    with caplog.at_level(logging.INFO):
        assert web_app._build_token_to_fighter_map() == {}

    matching = [
        record for record in caplog.records if "Gamma unavailable" in record.getMessage()
    ]
    assert len(matching) == 1
    assert matching[0].levelno == logging.INFO
