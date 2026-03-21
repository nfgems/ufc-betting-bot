import argparse
import sys

import pytest

import src.bot as bot


def test_tennis_live_parser_defaults_to_dry_run(monkeypatch):
    captured = {}

    def _capture(args):
        captured["dry_run"] = args.dry_run

    monkeypatch.setattr(bot, "cmd_tennis_live", _capture)
    monkeypatch.setattr(sys, "argv", ["bot", "tennis-live"])

    bot.main()

    assert captured["dry_run"] is True


def test_tennis_live_parser_allows_guard_to_receive_no_dry_run(monkeypatch):
    captured = {}

    def _capture(args):
        captured["dry_run"] = args.dry_run

    monkeypatch.setattr(bot, "cmd_tennis_live", _capture)
    monkeypatch.setattr(sys, "argv", ["bot", "tennis-live", "--no-dry-run"])

    bot.main()

    assert captured["dry_run"] is False


def test_cmd_tennis_live_rejects_non_dry_run_before_predictions(monkeypatch):
    errors = []

    def _record_error(message, *args):
        errors.append(message % args if args else message)

    monkeypatch.setattr(bot.logger, "error", _record_error)
    monkeypatch.setattr(
        bot,
        "_build_tennis_prediction_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build predictions")),
    )

    with pytest.raises(RuntimeError, match="Real-money tennis trading is not implemented"):
        bot.cmd_tennis_live(argparse.Namespace(dry_run=False, model="surface_elo", min_edge=None))
