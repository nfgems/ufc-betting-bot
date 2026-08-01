from datetime import datetime, timedelta, timezone

import pytest

from src.polymarket import markets as polymarket_markets
from src.polymarket.markets import get_ufc_fight_markets, parse_fight_market


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


def test_parse_fight_market_falls_back_to_orderbook_midpoint_when_prices_missing():
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

    assert parsed is not None
    assert parsed['price_yes'] == pytest.approx(0.155)
    assert parsed['price_no'] == pytest.approx(0.845)


def test_parse_fight_market_keeps_prices_missing_for_one_sided_book():
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

    assert parsed is not None
    assert parsed['price_yes'] is None
    assert parsed['price_no'] is None


def test_parse_fight_market_completes_single_valid_outcome_price():
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

    assert parsed is not None
    assert parsed['price_yes'] == pytest.approx(0.0)
    assert parsed['price_no'] == pytest.approx(1.0)


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
