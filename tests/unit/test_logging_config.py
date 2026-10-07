import logging

from logging_config import configure_logging, StructuredFormatter


def test_configure_logging():
    configure_logging()

    logger = logging.getLogger("test_logger")

    assert logger.isEnabledFor(logging.INFO)


def test_structured_formatter():
    formatter = StructuredFormatter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )

    formatted = formatter.format(record)

    assert "level=INFO" in formatted
    assert "logger=test_logger" in formatted
    assert "message=Test message" in formatted
    assert "timestamp=" in formatted