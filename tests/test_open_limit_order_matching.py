from src.web.app import (
    _display_edge_for_limit_order,
    _is_recovered_limit_placeholder,
    _pick_best_limit_match,
)


def test_recovered_limit_placeholder_is_detected():
    recovered = {
        "id": 4,
        "model_prob": 0.0,
        "edge": 0.0,
        "market_id": "",
        "order_type": "limit_bid",
        "placed_at": "2026-03-11T01:38:59.000000+00:00",
    }

    assert _is_recovered_limit_placeholder(recovered) is True
    assert _display_edge_for_limit_order(recovered) is None


def test_pick_best_limit_match_prefers_real_bet_over_recovered_placeholder():
    recovered = {
        "id": 4,
        "fighter": "Movsar Evloev",
        "model_prob": 0.0,
        "edge": 0.0,
        "market_id": "",
        "order_type": "limit_bid",
        "order_id": "recovered-order",
        "placed_at": "2026-03-11T01:38:59.000000+00:00",
    }
    real = {
        "id": 9,
        "fighter": "Movsar Evloev",
        "model_prob": 0.7599,
        "edge": 0.028,
        "market_id": "1510731",
        "order_type": "limit_bid",
        "order_id": None,
        "placed_at": "2026-03-11T19:03:00.000000+00:00",
    }

    picked = _pick_best_limit_match([recovered, real])

    assert picked == real
    assert _display_edge_for_limit_order(picked) == 0.028


def test_pick_best_limit_match_prefers_latest_real_entry():
    older = {
        "id": 2,
        "fighter": "Christian Leroy Duncan",
        "model_prob": 0.799,
        "edge": 0.021,
        "market_id": "1510769",
        "order_type": "limit_bid",
        "order_id": None,
        "placed_at": "2026-03-11T18:00:00.000000+00:00",
    }
    newer = {
        "id": 3,
        "fighter": "Christian Leroy Duncan",
        "model_prob": 0.799,
        "edge": 0.033,
        "market_id": "1510769",
        "order_type": "limit_bid",
        "order_id": None,
        "placed_at": "2026-03-11T19:03:00.000000+00:00",
    }

    assert _pick_best_limit_match([older, newer]) == newer


def test_pick_best_limit_match_prefers_limit_entry_over_market_entry():
    market = {
        "id": 10,
        "fighter": "Charles Johnson",
        "model_prob": 0.676,
        "edge": 0.061,
        "market_id": "1510646",
        "order_type": "market",
        "order_id": None,
        "placed_at": "2026-03-11T19:10:00.000000+00:00",
    }
    limit_bid = {
        "id": 11,
        "fighter": "Charles Johnson",
        "model_prob": 0.676,
        "edge": 0.026,
        "market_id": "1510646",
        "order_type": "limit_bid",
        "order_id": None,
        "placed_at": "2026-03-11T19:03:00.000000+00:00",
    }

    assert _pick_best_limit_match([market, limit_bid]) == limit_bid
