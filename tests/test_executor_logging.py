import logging

from src.polymarket.executor import _log_order_failure, _order_failure_is_warning


def test_order_failure_is_warning_for_geoblock():
    exc = RuntimeError(
        "PolyApiException[status_code=403, error_message={'error': "
        "'Trading restricted in your region'}]"
    )

    assert _order_failure_is_warning(exc) is True


def test_log_order_failure_uses_warning_for_known_api_rejection(caplog):
    exc = RuntimeError(
        "PolyApiException[status_code=403, error_message={'error': "
        "'Trading restricted in your region'}]"
    )

    with caplog.at_level(logging.INFO, logger="src.polymarket.executor"):
        _log_order_failure("Failed to place limit bid", "Charles Johnson", exc)

    assert any(
        record.levelname == "WARNING"
        and "Failed to place limit bid for Charles Johnson" in record.message
        for record in caplog.records
    )


def test_log_order_failure_keeps_unexpected_exceptions_as_errors(caplog):
    exc = RuntimeError("socket exploded")

    with caplog.at_level(logging.INFO, logger="src.polymarket.executor"):
        _log_order_failure("Failed to place limit bid", "Charles Johnson", exc)

    assert any(
        record.levelname == "ERROR"
        and "Failed to place limit bid for Charles Johnson" in record.message
        for record in caplog.records
    )
