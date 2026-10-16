from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pandas as pd
import pytest

from src import refresh_live


def test_orders_disabled_before_any_read(monkeypatch, tmp_path):
    monkeypatch.setattr(refresh_live, "_run_cycle", lambda **kw: pytest.fail("read before gate"))
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    with pytest.raises(RuntimeError, match="requires LIVE_TRADING_MODE=real"):
        refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path, dry_run=False)
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    with pytest.raises(RuntimeError, match="Orders-disabled"):
        refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)


def test_busy_and_interrupt_release_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    monkeypatch.setattr(refresh_live, "LOGS_DIR", tmp_path)
    calls = []
    def interrupt(**kw):
        calls.append(kw)
        raise KeyboardInterrupt("local cut")
    monkeypatch.setattr(refresh_live, "_run_cycle", interrupt)
    with refresh_live.trading_consumer_lock(tmp_path / "refresh-trading.lock"):
        assert refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)["status"] == "busy"
    with pytest.raises(KeyboardInterrupt):
        refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)
    monkeypatch.setattr(refresh_live, "_run_cycle", lambda **kw: {"status": "restarted"})
    assert refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)["status"] == "restarted"
    assert len(calls) == 1


def test_process_owner_lock(tmp_path):
    path = tmp_path / "owner.lock"
    program = ("from src.refresh_live import trading_consumer_lock, TradingConsumerBusy\n"
               "import sys\ntry:\n with trading_consumer_lock(sys.argv[1]): sys.exit(2)\n"
               "except TradingConsumerBusy: sys.exit(0)\n")
    with refresh_live.trading_consumer_lock(path):
        assert subprocess.run([sys.executable, "-c", program, str(path)], timeout=15).returncode == 0
    assert subprocess.run([sys.executable, "-c", program, str(path)], timeout=15).returncode == 2


def _market(**overrides):
    return {"fighter_a": "Beta Fighter", "fighter_b": "Alpha Fighter",
            "event_date": "2099-10-17T21:00:00+00:00", "token_id_yes": "beta-token",
            "token_id_no": "alpha-token", "price_yes": .3, "price_no": .7,
            "market_id": "fixture-market", "active": True, "closed": False,
            "accepting_orders": True, **overrides}


def _prediction():
    return {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter",
            "official_event_date": "2099-10-17", "event_date": "2099-10-17T21:00:00+00:00",
            "commence_time": "2099-10-17T21:00:00+00:00", "prob_a": .7, "prob_b": .3,
            "no_odds_prob_a": .65, "no_odds_prob_b": .35}


def test_outcome_token_orientation_through_actual_matcher():
    from src.polymarket.executor import OrderExecutor
    predictions = [_prediction()]
    markets = refresh_live._bound_markets(pd.DataFrame([_market()]), predictions)
    matched = OrderExecutor.__new__(OrderExecutor)._match_predictions_to_markets(pd.DataFrame(predictions), markets)
    assert matched.iloc[0]["token_id_yes"] == "alpha-token"
    assert matched.iloc[0]["token_id_no"] == "beta-token"
    assert matched.iloc[0]["a_market_prob"] == .7


@pytest.mark.parametrize("markets", [
    [_market(token_id_no="")], [_market(token_id_no="beta-token")],
    [_market(token_id_no=" beta-token ")],
    [_market(), _market(market_id="ambiguous-second")],
])
def test_missing_or_ambiguous_token_mapping_refuses(markets):
    with pytest.raises(ValueError, match="mapping"):
        refresh_live._bound_markets(pd.DataFrame(markets), [_prediction()])


def test_wrong_card_date_cannot_use_same_names():
    assert refresh_live._bound_markets(pd.DataFrame([_market(event_date="2099-10-18T21:00:00Z")]),
                                       [_prediction()]).empty


def test_cli_dispatch_never_legacy_load_or_model_refresh(monkeypatch, tmp_path):
    from src import bot
    monkeypatch.setattr(bot, "ensure_model_fresh", lambda *a: pytest.fail("legacy model refresh"))
    seen = []
    monkeypatch.setattr(refresh_live, "run_refresh_live_cycle", lambda **kw: seen.append(kw) or {"status": "idle"})
    result = bot.cmd_duo_live(SimpleNamespace(refresh_root=tmp_path, refresh_baseline=tmp_path,
                                             dry_run=True, min_edge=.02))
    assert result == {"status": "idle"}
    assert seen[0]["dry_run"] is True


@pytest.mark.parametrize("bookmaker_inputs_available", [True, False])
def test_real_portfolio_runner_with_current_read_boundaries_orders_disabled(monkeypatch, tmp_path, bookmaker_inputs_available):
    from src.data import refresh_service, refresh_trading, odds_client, line_tracker
    from src.polymarket import markets
    from src.strategy import duo_trader
    from src import bot
    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    monkeypatch.setattr(refresh_live, "LOGS_DIR", tmp_path)
    consensus = {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter",
                 "commence_time": "2099-10-17T21:00:00+00:00", "event_id": "observed-fixture",
                 "a_fair_prob_avg": .7, "b_fair_prob_avg": .3}
    seen = []
    class Consumer:
        def __init__(self, service):
            self.rows = [{**consensus, "event_date": "2099-10-17"}]
            self.publication = {"generation_id": "final-good", "job_id": "original-job"}
            self.report = {"event_date": "2099-10-17"}
            self.binding = {"accepted_generation": "final-good"}
        def read(self): return self
        def score(self, fights, **kw):
            assert fights == [consensus]
            seen.append(kw)
            return [{**_prediction(), **consensus, "model_features": {}, "feature_provenance": {},
                     "a_num_fights": 10, "b_num_fights": 10,
                     "market_inputs_available": bookmaker_inputs_available}]
    class Odds:
        def get_live_odds(self): return [{"observed": True}]
        def odds_to_dataframe(self, rows): return rows
        def get_consensus_odds(self, rows): return pd.DataFrame([consensus])
    monkeypatch.setattr(refresh_service, "RefreshService", lambda **kw: kw)
    monkeypatch.setattr(refresh_trading, "PublishedTradingSnapshot", Consumer)
    monkeypatch.setattr(odds_client, "OddsClient", Odds)
    def market_read(**kw):
        assert kw == {"bout_contexts": [
            {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter",
             "event_date": "2099-10-17", "official_event_date": "2099-10-17"},
            consensus], "require_fresh": True}
        return pd.DataFrame([_market()])
    monkeypatch.setattr(markets, "get_ufc_fight_markets", market_read)
    monkeypatch.setattr(bot, "_live_prediction_quality_assessment", lambda *a, **kw: {"blocked": False, "reasons": []})
    monkeypatch.setattr(bot, "_withhold_unconfirmed_identity", lambda row: [])
    monkeypatch.setattr(line_tracker, "detect_injury_or_cancellation", lambda *a, **kw: {"severity": "none"})
    monkeypatch.setattr(duo_trader, "_fetch_polymarket_account_state", lambda **kw: {})
    actual_runner = duo_trader.run_duo_traders
    def observed_runner(*a, **kw):
        assert kw["dry_run"] is True and kw["clob"] is None
        return actual_runner(*a, **kw)
    monkeypatch.setattr(duo_trader, "run_duo_traders", observed_runner)
    result = refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)
    assert result["accepted_generation"] == "final-good"
    assert result["orders_disabled"] is True and result["trading_enabled"] is False
    assert result["prediction_count"] == 1
    assert result["executable_prediction_count"] == int(bookmaker_inputs_available)
    if not bookmaker_inputs_available:
        assert result["predictions"][0]["trade_block_reason"] == "missing_current_bookmaker_inputs"
    assert result.get("total_orders", 0) == 0
    assert seen[0]["observed_at"].tzinfo is not None


@pytest.mark.parametrize("bookmaker_count, current_market_exists", [(0, True), (1, True), (1, False)])
def test_full_card_markets_survive_partial_or_empty_bookmaker_feed(monkeypatch, tmp_path, bookmaker_count,
                                                                 current_market_exists):
    from src.data import refresh_service, refresh_trading, odds_client
    from src.polymarket import markets
    from src.strategy import duo_trader
    from src import bot

    monkeypatch.setenv("LIVE_TRADING_MODE", "off")
    monkeypatch.setattr(refresh_live, "LOGS_DIR", tmp_path)
    consensus = {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter",
                 "commence_time": "2099-10-17T22:00:00+00:00",
                 "a_fair_prob_avg": .7, "b_fair_prob_avg": .3}
    card = [{"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter", "event_date": "2099-10-17"},
            {"fighter_a": "Alphabeta Fighter", "fighter_b": "Betamax Fighter", "event_date": "2099-10-17"}]
    scored, executed = [], []

    class Consumer:
        def __init__(self, service):
            # Weekly data must not override Gamma with an old bout clock.
            self.rows = [{**row, "commence_time": "2099-10-17T12:00:00Z"} for row in card]
            self.publication = {"generation_id": "final-good", "job_id": "original-job"}
            self.report = {"event_date": "2099-10-17"}
            self.binding = {"accepted_generation": "final-good"}
        def read(self): return self
        def score(self, fights, **kw):
            scored.extend(fights)
            assert fights == [consensus][:bookmaker_count]
            return [{**_prediction(), **consensus, "model_features": {}, "feature_provenance": {},
                     "a_num_fights": 10, "b_num_fights": 10, "market_inputs_available": True}][:bookmaker_count]

    class Odds:
        def get_live_odds(self): return []
        def odds_to_dataframe(self, rows): return rows
        def get_consensus_odds(self, rows): return pd.DataFrame([consensus][:bookmaker_count])

    def discover(**kw):
        assert kw == {"bout_contexts": [
            {**row, "official_event_date": row["event_date"]} for row in card
        ] + [consensus][:bookmaker_count], "require_fresh": True}
        return pd.DataFrame(([_market()] if current_market_exists else []) + [
            # These names also match the current pair via the legacy fuzzy
            # matcher. Displaying their quote must never expose their tokens
            # to execution of the distinct bookmaker-covered pair.
            _market(fighter_a="Betamax Fighter", fighter_b="Alphabeta Fighter", token_id_yes="delta-token",
                    token_id_no="gamma-token", price_yes=.42, price_no=.59, market_id="second-market"),
            _market(fighter_a="Offcard Fighter", fighter_b="Other Fighter"),
            _market(fighter_a="Alphabeta Fighter", fighter_b="Betamax Fighter", event_date="2099-10-18T21:00:00Z"),
        ])

    def execute(predictions, mapped, **kw):
        assert kw["dry_run"] is True
        executed.append(predictions.to_dict("records"))
        assert len(mapped) == 1 and mapped.iloc[0]["market_id"] == "fixture-market"
        return {"status": "evaluated", "total_orders": 0}

    monkeypatch.setattr(refresh_service, "RefreshService", lambda **kw: kw)
    monkeypatch.setattr(refresh_trading, "PublishedTradingSnapshot", Consumer)
    monkeypatch.setattr(odds_client, "OddsClient", Odds)
    monkeypatch.setattr(markets, "get_ufc_fight_markets", discover)
    monkeypatch.setattr(bot, "_live_prediction_quality_assessment", lambda *a, **kw: {"blocked": False, "reasons": []})
    monkeypatch.setattr(bot, "_withhold_unconfirmed_identity", lambda row: [])
    monkeypatch.setattr(refresh_live, "INJURY_BLOCK_BETS", False)
    monkeypatch.setattr(duo_trader, "run_duo_traders", execute)
    result = refresh_live.run_refresh_live_cycle(root=tmp_path, baseline=tmp_path)

    assert result["matched_market_count"] == 1 + int(current_market_exists)
    assert result["prediction_count"] == result["executable_prediction_count"] == bookmaker_count
    assert result["current_card"]["missing_bookmaker_bout_count"] == 2 - bookmaker_count
    assert result["mapped_markets"][-1]["price_no"] == .59  # Preserve the observed quote, without normalization.
    assert scored == [consensus][:bookmaker_count]
    assert len(executed) == bookmaker_count * int(current_market_exists)
    if executed:
        assert executed[0][0]["fighter_a"] == "Alpha Fighter"
    assert result["total_orders"] == result["real_submissions"] == 0


def test_shared_token_on_bookmaker_missing_bout_still_refuses():
    card = [_prediction(), {"fighter_a": "Gamma Fighter", "fighter_b": "Delta Fighter",
                           "official_event_date": "2099-10-17"}]
    with pytest.raises(ValueError, match="mapping"):
        refresh_live._bound_markets(pd.DataFrame([
            _market(), _market(fighter_a="Gamma Fighter", fighter_b="Delta Fighter", token_id_no="delta-token")
        ]), card)
