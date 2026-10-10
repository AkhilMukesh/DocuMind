
from logging_config import StructuredFormatter
from request_context import reset_request_id, set_request_id
import logging


def test_structured_formatter_includes_request_id():
    _, token = set_request_id("trace-123")

    try:
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Tracing test",
            args=(),
            exc_info=None,
        )

        formatted = StructuredFormatter().format(record)

        assert "request_id=trace-123" in formatted
        assert "message=Tracing test" in formatted

    finally:
        reset_request_id(token)
