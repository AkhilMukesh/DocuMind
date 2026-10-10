
import logging

import pytest

from error_logging import log_operation_errors


def test_success_does_not_log_error(caplog):
    with caplog.at_level(logging.ERROR, logger="error_logging"):
        with log_operation_errors(
            "document_retrieval",
            document_id="doc-123"
        ):
            result = "success"

    assert result == "success"
    assert "Operation failed" not in caplog.text


def test_failure_logs_operation_and_traceback(caplog):
    with caplog.at_level(logging.ERROR, logger="error_logging"):
        with pytest.raises(ValueError, match="Invalid input"):
            with log_operation_errors(
                "document_ingestion",
                filename="policy.txt"
            ):
                raise ValueError("Invalid input")

    assert "Operation failed" in caplog.text
    assert "document_ingestion" in caplog.text
    assert "policy.txt" in caplog.text
    assert "ValueError: Invalid input" in caplog.text
    assert "Traceback" in caplog.text


def test_original_exception_is_preserved():
    original_error = ValueError("Original failure")

    with pytest.raises(ValueError) as error:
        with log_operation_errors("test_operation"):
            raise original_error

    assert error.value is original_error
