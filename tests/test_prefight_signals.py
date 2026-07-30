from src import bot
from src.data import live_monitor, prefight_signals


def _disable_line_lookup(monkeypatch):
    monkeypatch.setattr(
        prefight_signals,
        "get_line_movement_features",
        lambda *_args, **_kwargs: {},
    )


def test_short_notice_flag_never_invents_notice_period(monkeypatch):
    _disable_line_lookup(monkeypatch)

    signals = prefight_signals.collect_prefight_signals(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        a_is_short_notice=True,
    )

    assert signals["flags"] == [
        "Alpha Fighter is a short-notice replacement",
    ]
    assert "60-day notice" not in " ".join(signals["flags"])


def test_short_notice_flag_labels_detection_countdown(monkeypatch):
    _disable_line_lookup(monkeypatch)

    signals = prefight_signals.collect_prefight_signals(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        b_is_short_notice=True,
        b_days_until_event_at_detection=3,
    )

    assert signals["flags"] == [
        "Beta Fighter is a short-notice replacement "
        "(late-detected with 3 days until event)",
    ]


def test_cmd_signals_passes_persisted_detection_countdown(monkeypatch):
    captured_calls = []
    event = {
        "title": "UFC Test",
        "date": "August 1, 2026",
        "url": "https://www.ufc.com/event/ufc-test",
        "days_to_event": 2,
        "fights": [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
            },
        ],
    }
    event["event_key"] = live_monitor.event_identity_key(event)
    monkeypatch.setattr(
        live_monitor,
        "run_monitoring_pass",
        lambda: {
            "events": [event],
            "short_notice_replacements": [
                {
                    "new_fighter": "Beta Fighter",
                    "replaced_fighter": "Gamma Fighter",
                    "days_until_event_at_detection": 3,
                    "is_short_notice": True,
                    "event_key": event["event_key"],
                },
            ],
            "missed_weights": [],
        },
    )

    def capture_prefight_signals(**kwargs):
        captured_calls.append(kwargs)
        return {"flags": []}

    monkeypatch.setattr(
        prefight_signals,
        "collect_prefight_signals",
        capture_prefight_signals,
    )

    bot.cmd_signals(None)

    assert len(captured_calls) == 1
    assert captured_calls[0]["a_is_short_notice"] is False
    assert captured_calls[0]["a_days_until_event_at_detection"] is None
    assert captured_calls[0]["b_is_short_notice"] is True
    assert captured_calls[0]["b_days_until_event_at_detection"] == 3


def test_cmd_signals_scopes_shared_fighter_to_event_identity(monkeypatch):
    captured_calls = []
    events = [
        {
            "title": "UFC Event One",
            "date": "August 1, 2026",
            "url": "https://www.ufc.com/event/ufc-event-one",
            "days_to_event": 2,
            "fights": [
                {
                    "fighter_a": "Shared Fighter",
                    "fighter_b": "First Opponent",
                },
            ],
        },
        {
            "title": "UFC Event Two",
            "date": "August 8, 2026",
            "url": "https://www.ufc.com/event/ufc-event-two",
            "days_to_event": 9,
            "fights": [
                {
                    "fighter_a": "Shared Fighter",
                    "fighter_b": "Second Opponent",
                },
            ],
        },
    ]
    for event in events:
        event["event_key"] = live_monitor.event_identity_key(event)

    monkeypatch.setattr(
        live_monitor,
        "run_monitoring_pass",
        lambda: {
            "events": events,
            "short_notice_replacements": [
                {
                    "new_fighter": "Shared Fíghter",
                    "replaced_fighter": "Original Fighter",
                    "days_until_event_at_detection": 2,
                    "is_short_notice": True,
                    "event_key": events[0]["event_key"],
                },
            ],
            "missed_weights": [],
        },
    )

    def capture_prefight_signals(**kwargs):
        captured_calls.append(kwargs)
        return {"flags": []}

    monkeypatch.setattr(
        prefight_signals,
        "collect_prefight_signals",
        capture_prefight_signals,
    )

    bot.cmd_signals(None)

    assert [call["a_is_short_notice"] for call in captured_calls] == [True, False]
    assert [
        call["a_days_until_event_at_detection"] for call in captured_calls
    ] == [2, None]


def test_short_notice_event_day_countdown_is_never_negative(monkeypatch):
    _disable_line_lookup(monkeypatch)

    signals = prefight_signals.collect_prefight_signals(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        a_is_short_notice=True,
        a_days_until_event_at_detection=-1,
    )

    assert signals["flags"] == [
        "Alpha Fighter is a short-notice replacement "
        "(late-detected with 0 days until event)",
    ]
