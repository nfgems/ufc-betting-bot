from datetime import datetime, timedelta, timezone

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
