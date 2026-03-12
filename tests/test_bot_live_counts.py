from src import bot


def test_resolve_live_fight_counts_prefers_live_feature_counts():
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": 17,
            "b_num_fights": "9",
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda _: 0,
    )

    assert counts == (17, 9)


def test_resolve_live_fight_counts_falls_back_when_live_counts_missing():
    fallback_counts = {
        "Israel Adesanya": 18,
        "Joseph Pyfer": 0,
    }
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": None,
            "b_num_fights": float("nan"),
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda name: fallback_counts[name],
    )

    assert counts == (18, 0)
