import logging
import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.polymarket import markets as polymarket_markets
from src.polymarket.markets import get_ufc_fight_markets, parse_fight_market


class _GammaEventsResponse:
    def __init__(self, events):
        self._events = events

    def raise_for_status(self):
        return None

    def json(self):
        return self._events


@pytest.fixture(autouse=True)
def _reset_gamma_events_fetch_state():
    polymarket_markets._reset_ufc_events_fetch_state()
    yield
    polymarket_markets._reset_ufc_events_fetch_state()


def test_find_ufc_events_logs_recovered_retry_at_info(monkeypatch, caplog):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise polymarket_markets.requests.Timeout("timed out")
        return _GammaEventsResponse([])

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)
    monkeypatch.setattr(polymarket_markets.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO, logger="src.polymarket.markets"):
        events = polymarket_markets.find_ufc_events()

    retry_records = [
        record
        for record in caplog.records
        if "Failed to fetch UFC events" in record.getMessage()
    ]
    assert events == []
    assert len(calls) == 2
    assert len(retry_records) == 1
    assert retry_records[0].levelno == logging.INFO
    assert "attempt 1/3" in retry_records[0].getMessage()
    assert "retrying in 1s" in retry_records[0].getMessage()


def test_find_ufc_events_raises_typed_error_once_only_after_retry_exhaustion(
    monkeypatch,
    caplog,
):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise polymarket_markets.requests.Timeout("timed out")

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)
    monkeypatch.setattr(polymarket_markets.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.INFO, logger="src.polymarket.markets"):
        with pytest.raises(
            polymarket_markets.GammaEventsUnavailableError,
            match="discovery unavailable after 3 attempts",
        ):
            polymarket_markets.find_ufc_events()

    failure_records = [
        record
        for record in caplog.records
        if "Failed to fetch UFC events" in record.getMessage()
    ]
    assert len(calls) == 3
    assert [record.levelno for record in failure_records] == [
        logging.INFO,
        logging.INFO,
        logging.WARNING,
    ]
    assert "after 3 attempts" in failure_records[-1].getMessage()
    assert (
        failure_records[-1].alert_incident_key
        == "polymarket_gamma_ufc_events_unavailable"
    )


def test_find_ufc_events_failure_cooldown_suppresses_second_retry_sequence(
    monkeypatch,
):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise polymarket_markets.requests.Timeout("timed out")

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)
    monkeypatch.setattr(polymarket_markets.time, "sleep", lambda _seconds: None)

    with pytest.raises(polymarket_markets.GammaEventsUnavailableError):
        polymarket_markets.find_ufc_events()
    with pytest.raises(
        polymarket_markets.GammaEventsUnavailableError,
        match="retry suppressed",
    ):
        polymarket_markets.find_ufc_events()

    assert len(calls) == 3


def test_find_ufc_events_short_success_cache_returns_defensive_copy(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return _GammaEventsResponse([{"id": "event-1", "title": "UFC card"}])

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)

    first = polymarket_markets.find_ufc_events()
    first[0]["title"] = "mutated by caller"
    second = polymarket_markets.find_ufc_events()

    assert len(calls) == 1
    assert second == [{"id": "event-1", "title": "UFC card"}]


def test_find_ufc_events_fresh_read_bypasses_short_success_cache(monkeypatch):
    responses = [
        [{"id": "event-1", "price": "0.40"}],
        [{"id": "event-1", "price": "0.45"}],
    ]
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return _GammaEventsResponse(responses[len(calls) - 1])

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)

    monitor_snapshot = polymarket_markets.find_ufc_events()
    trading_snapshot = polymarket_markets.find_ufc_events(require_fresh=True)

    assert len(calls) == 2
    assert monitor_snapshot[0]["price"] == "0.40"
    assert trading_snapshot[0]["price"] == "0.45"


def test_fresh_read_retries_after_older_refresh_completes(monkeypatch):
    fetch_started = threading.Event()
    release_fetch = threading.Event()
    trading_request_started = threading.Event()
    calls = []
    results = {}
    real_next_sequence = polymarket_markets._next_gamma_events_request_sequence
    responses = [
        [{"id": "event-1", "snapshot": "monitor"}],
        [{"id": "event-1", "snapshot": "trading"}],
    ]

    def controlled_sequence():
        sequence = real_next_sequence()
        if (
            threading.current_thread().name == "gamma-trading-reader"
            and not trading_request_started.is_set()
        ):
            trading_request_started.set()
        return sequence

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        call_index = len(calls) - 1
        if call_index == 0:
            fetch_started.set()
            assert release_fetch.wait(timeout=2)
        return _GammaEventsResponse(responses[call_index])

    def read_for_monitor():
        results["monitor"] = polymarket_markets.find_ufc_events()

    def read_for_trading():
        results["trading"] = polymarket_markets.find_ufc_events(require_fresh=True)

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)
    monkeypatch.setattr(
        polymarket_markets,
        "_next_gamma_events_request_sequence",
        controlled_sequence,
    )

    monitor_thread = threading.Thread(target=read_for_monitor)
    monitor_thread.start()
    assert fetch_started.wait(timeout=2)
    trading_thread = threading.Thread(
        target=read_for_trading,
        name="gamma-trading-reader",
    )
    trading_thread.start()
    assert trading_request_started.wait(timeout=2)
    release_fetch.set()
    monitor_thread.join(timeout=2)
    trading_thread.join(timeout=2)

    assert not monitor_thread.is_alive()
    assert not trading_thread.is_alive()
    assert len(calls) == 2
    assert results["monitor"] == [{"id": "event-1", "snapshot": "monitor"}]
    assert results["trading"] == [{"id": "event-1", "snapshot": "trading"}]


def test_fresh_read_coalesces_with_fetch_started_after_request(monkeypatch):
    trading_request_started = threading.Event()
    continue_trading_request = threading.Event()
    calls = []
    results = {}
    real_next_sequence = polymarket_markets._next_gamma_events_request_sequence

    def controlled_sequence():
        sequence = real_next_sequence()
        if (
            threading.current_thread().name == "gamma-waiting-trading-reader"
            and not trading_request_started.is_set()
        ):
            trading_request_started.set()
            assert continue_trading_request.wait(timeout=2)
        return sequence

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return _GammaEventsResponse([{"id": "event-1", "snapshot": "shared"}])

    def read_for_trading():
        results["trading"] = polymarket_markets.find_ufc_events(require_fresh=True)

    monkeypatch.setattr(polymarket_markets.requests, "get", fake_get)
    monkeypatch.setattr(
        polymarket_markets,
        "_next_gamma_events_request_sequence",
        controlled_sequence,
    )

    trading_thread = threading.Thread(
        target=read_for_trading,
        name="gamma-waiting-trading-reader",
    )
    trading_thread.start()
    assert trading_request_started.wait(timeout=2)

    monitor_snapshot = polymarket_markets.find_ufc_events()
    continue_trading_request.set()
    trading_thread.join(timeout=2)

    assert not trading_thread.is_alive()
    assert len(calls) == 1
    assert monitor_snapshot == results["trading"] == [
        {"id": "event-1", "snapshot": "shared"}
    ]


def test_get_ufc_fight_markets_forwards_fresh_requirement(monkeypatch):
    calls = []

    def fake_find(*, require_fresh=False):
        calls.append(require_fresh)
        return []

    monkeypatch.setattr(polymarket_markets, "find_ufc_events", fake_find)

    markets = polymarket_markets.get_ufc_fight_markets(require_fresh=True)

    assert markets.empty
    assert calls == [True]


def test_parse_fight_market_prefers_actual_game_start_to_listing_timestamp():
    parsed = parse_fight_market(
        {
            'id': 'market-1',
            'question': 'Rousey vs. Carano',
            'conditionId': 'condition-1',
            'outcomes': ['Rousey', 'Carano'],
            'clobTokenIds': ['token-a', 'token-b'],
            'outcomePrices': ['0.57', '0.43'],
            'startDate': '2026-02-19T16:04:56.534Z',
            'gameStartTime': '2026-05-17 01:00:00+00',
            'endDate': '2026-05-17T00:00:00Z',
        },
        event={
            'id': 'event-1',
            'startDate': '2026-02-19T16:05:54.555811Z',
            'startTime': '2026-05-17T01:00:00Z',
            'eventDate': '2026-05-16',
        },
    )

    assert parsed is not None
    assert parsed['event_date'] == '2026-05-17 01:00:00+00'


def test_parse_fight_market_rejects_orderbook_midpoint_when_prices_missing():
    parsed = parse_fight_market(
        {
            'id': 'market-1',
            'question': 'Matt Schnell vs. Alessandro Costa',
            'conditionId': 'condition-1',
            'outcomes': ['Matt Schnell', 'Alessandro Costa'],
            'clobTokenIds': ['token-a', 'token-b'],
            'bestBid': 0.14,
            'bestAsk': 0.17,
            'gameStartTime': '2026-06-06 17:00:00+00',
        },
        event={'id': 'event-1'},
    )

    assert parsed is None


def test_parse_fight_market_rejects_one_sided_book():
    parsed = parse_fight_market(
        {
            'id': 'market-1',
            'question': 'Matt Schnell vs. Alessandro Costa',
            'conditionId': 'condition-1',
            'outcomes': ['Matt Schnell', 'Alessandro Costa'],
            'clobTokenIds': ['token-a', 'token-b'],
            'bestBid': 0.14,
            'gameStartTime': '2026-06-06 17:00:00+00',
        },
        event={'id': 'event-1'},
    )

    assert parsed is None


def test_parse_fight_market_rejects_single_valid_outcome_price():
    parsed = parse_fight_market(
        {
            'id': 'market-1',
            'question': 'Alpha vs. Beta',
            'conditionId': 'condition-1',
            'outcomes': ['Alpha', 'Beta'],
            'clobTokenIds': ['token-a', 'token-b'],
            'outcomePrices': ['0', None],
            'bestBid': 0.24,
            'bestAsk': 0.26,
            'gameStartTime': '2026-06-06 17:00:00+00',
        },
        event={'id': 'event-1'},
    )

    assert parsed is None


def test_get_ufc_fight_markets_keeps_future_fights_with_past_listing_timestamp(monkeypatch):
    future_start = datetime.now(timezone.utc) + timedelta(days=2)
    listed_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace('+00:00', 'Z')

    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [
            {
                'id': 'event-1',
                'title': 'UFC Test Card: Alpha vs. Beta',
                'startDate': listed_at,
                'startTime': future_start.isoformat().replace('+00:00', 'Z'),
                'eventDate': future_start.date().isoformat(),
                'markets': [
                    {
                        'id': 'market-1',
                        'question': 'Alpha vs. Beta',
                        'conditionId': 'condition-1',
                        'outcomes': ['Alpha', 'Beta'],
                        'clobTokenIds': ['token-a', 'token-b'],
                        'outcomePrices': ['0.55', '0.45'],
                        'bestBid': 0.54,
                        'bestAsk': 0.56,
                        'liquidityNum': 1000,
                        'volume': 250,
                        'startDate': listed_at,
                        'gameStartTime': future_start.isoformat(sep=' '),
                        'endDate': (future_start + timedelta(hours=6)).isoformat().replace('+00:00', 'Z'),
                        'active': True,
                        'closed': False,
                    }
                ],
            }
        ],
    )

    markets = get_ufc_fight_markets()

    assert len(markets) == 1
    assert markets.iloc[0]['fighter_a'] == 'Alpha'
    assert markets.iloc[0]['event_date'] == future_start.isoformat(sep=' ')


def _fight_event(
    *,
    card_start,
    accepting_orders=True,
    fighter_a='Alpha',
    fighter_b='Beta',
):
    return {
        'id': 'event-1',
        'title': 'UFC Test Card: Alpha vs. Beta',
        'startTime': card_start.isoformat().replace('+00:00', 'Z'),
        'eventDate': card_start.date().isoformat(),
        'markets': [
            {
                'id': 'market-1',
                'question': f'{fighter_a} vs. {fighter_b}',
                'conditionId': 'condition-1',
                'outcomes': [fighter_a, fighter_b],
                'clobTokenIds': ['token-a', 'token-b'],
                'outcomePrices': ['0.55', '0.45'],
                'gameStartTime': card_start.isoformat(sep=' '),
                'active': True,
                'closed': False,
                'acceptingOrders': accepting_orders,
            }
        ],
    }


def test_get_ufc_fight_markets_prefers_confirmed_bout_time_over_generic_card_start(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    bout_start = datetime.now(timezone.utc) + timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start)],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'commence_time': bout_start.isoformat(),
            }
        ]
    )

    assert len(markets) == 1
    assert markets.iloc[0]['event_date'] == bout_start.isoformat()
    assert markets.iloc[0]['polymarket_event_date'] == card_start.isoformat(sep=' ')
    assert markets.iloc[0]['event_time_source'] == 'bout_context'


def test_get_ufc_fight_markets_matches_cross_source_alias_for_bout_time(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    bout_start = datetime.now(timezone.utc) + timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [
            _fight_event(
                card_start=card_start,
                fighter_a='Ludovit Klein',
                fighter_b='Tofiq Musayev',
            )
        ],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': "L'udovit Klein",
                'fighter_b': 'Tofiq Musayev',
                'commence_time': bout_start.isoformat(),
            }
        ]
    )

    assert len(markets) == 1
    assert markets.iloc[0]['event_date'] == bout_start.isoformat()
    assert markets.iloc[0]['event_time_source'] == 'bout_context'


def test_get_ufc_fight_markets_uses_bout_time_to_close_market_safely(monkeypatch):
    card_start = datetime.now(timezone.utc) + timedelta(hours=4)
    bout_start = datetime.now(timezone.utc) + timedelta(minutes=30)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start)],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'commence_time': bout_start.isoformat(),
            }
        ]
    )

    assert markets.empty


def test_get_ufc_fight_markets_does_not_override_nonaccepting_orderbook(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    bout_start = datetime.now(timezone.utc) + timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [
            _fight_event(card_start=card_start, accepting_orders=False)
        ],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'commence_time': bout_start.isoformat(),
            }
        ]
    )

    assert markets.empty


def test_get_ufc_fight_markets_rejects_unrelated_bout_date_override(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    unrelated_bout_start = datetime.now(timezone.utc) + timedelta(days=7)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start, accepting_orders=True)],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'commence_time': unrelated_bout_start.isoformat(),
            }
        ]
    )

    assert markets.empty


def test_bout_context_index_keeps_official_and_odds_alias_pairs():
    context = {
        'fighter_a': 'Official Alpha',
        'fighter_b': 'Official Beta',
        'odds_fighter_a': 'Odds Alias Alpha',
        'odds_fighter_b': 'Odds Alias Beta',
    }

    indexed = polymarket_markets._index_bout_contexts([context])

    assert indexed[polymarket_markets._fight_pair_key('Official Alpha', 'Official Beta')] is context
    assert indexed[polymarket_markets._fight_pair_key('Odds Alias Alpha', 'Odds Alias Beta')] is context


def test_get_ufc_fight_markets_allows_bounded_confirmed_card_fallback(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start, accepting_orders=True)],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[{'fighter_a': 'Alpha', 'fighter_b': 'Beta'}]
    )

    assert len(markets) == 1
    assert markets.iloc[0]['event_time_source'] == 'current_card_fallback'


def test_get_ufc_fight_markets_does_not_revive_unconfirmed_card_market(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start, accepting_orders=True)],
    )

    markets = get_ufc_fight_markets()

    assert markets.empty


def test_get_ufc_fight_markets_does_not_treat_card_date_as_bout_midnight(monkeypatch):
    card_start = datetime.now(timezone.utc) - timedelta(hours=2)
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [_fight_event(card_start=card_start, accepting_orders=True)],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[
            {
                'fighter_a': 'Alpha',
                'fighter_b': 'Beta',
                'event_date': card_start.date().isoformat(),
            }
        ]
    )

    assert len(markets) == 1
    assert markets.iloc[0]['event_time_source'] == 'current_card_fallback'


@pytest.mark.parametrize(
    ('card_age', 'accepting_orders'),
    [
        (timedelta(hours=2), False),
        (timedelta(hours=13), True),
    ],
)
def test_get_ufc_fight_markets_rejects_unsafe_card_fallbacks(
    monkeypatch,
    card_age,
    accepting_orders,
):
    card_start = datetime.now(timezone.utc) - card_age
    monkeypatch.setattr(
        polymarket_markets,
        'find_ufc_events',
        lambda limit=200: [
            _fight_event(
                card_start=card_start,
                accepting_orders=accepting_orders,
            )
        ],
    )

    markets = get_ufc_fight_markets(
        bout_contexts=[{'fighter_a': 'Alpha', 'fighter_b': 'Beta'}]
    )

    assert markets.empty
