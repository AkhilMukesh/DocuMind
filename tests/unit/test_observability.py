
import logging
from types import SimpleNamespace

import pytest

import rag_pipeline
from logging_config import StructuredFormatter
from metrics import metrics
from request_context import get_request_id


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


def configure_fake_pipeline(monkeypatch, fake_llm):
    document = SimpleNamespace(
        page_content="Employees can work remotely.",
        metadata={"filename": "company_policy.txt"}
    )

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_documents",
        lambda question, document_id=None: [document]
    )

    monkeypatch.setattr(
        rag_pipeline,
        "format_documents",
        lambda documents: (
            "[1] company_policy.txt\n"
            "Employees can work remotely."
        )
    )

    fake_prompt = SimpleNamespace(
        invoke=lambda values: values
    )

    monkeypatch.setattr(
        rag_pipeline,
        "rag_prompt",
        fake_prompt
    )

    monkeypatch.setattr(
        rag_pipeline,
        "llm",
        fake_llm
    )


def test_success_records_metrics_and_restores_request_context(
    monkeypatch,
    caplog
):
    class FakeLLM:
        def invoke(self, messages):
            # The request ID must be active during LLM execution.
            assert get_request_id() is not None
            return "Employees can work remotely. [1]"

    configure_fake_pipeline(monkeypatch, FakeLLM())

    with caplog.at_level(
        logging.INFO,
        logger="rag_pipeline"
    ):
        answer, sources = rag_pipeline.ask_question(
            "Can employees work remotely?"
        )

    counters = metrics.snapshot()["counters"]

    assert answer == "Employees can work remotely. [1]"
    assert len(sources) == 1

    assert counters["rag.request.started"] == 1
    assert counters["rag.request.success"] == 1
    assert counters["llm.request.success"] == 1

    assert "RAG question processing started" in caplog.text
    assert "LLM request completed" in caplog.text

    # The request context must be cleared after completion.
    assert get_request_id() is None


def test_llm_failure_records_metrics_and_restores_context(
    monkeypatch,
    caplog
):
    class FakeLLM:
        def invoke(self, messages):
            assert get_request_id() is not None
            raise RuntimeError("Simulated LLM failure")

    configure_fake_pipeline(monkeypatch, FakeLLM())

    with caplog.at_level(
        logging.ERROR,
        logger="rag_pipeline"
    ):
        with pytest.raises(
            RuntimeError,
            match="Simulated LLM failure"
        ):
            rag_pipeline.ask_question("Test question")

    counters = metrics.snapshot()["counters"]

    assert counters["rag.request.started"] == 1
    assert counters["rag.request.failure"] == 1
    assert counters["llm.request.failure"] == 1

    assert "RAG question processing failed" in caplog.text
    assert "LLM request failed" in caplog.text
    assert get_request_id() is None


def test_structured_formatter_includes_request_id(monkeypatch):
    from request_context import set_request_id, reset_request_id

    _, token = set_request_id("observability-test-123")

    try:
        record = logging.LogRecord(
            name="rag_pipeline",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Observability test",
            args=(),
            exc_info=None
        )

        formatted = StructuredFormatter().format(record)

        assert "request_id=observability-test-123" in formatted
        assert "level=INFO" in formatted
        assert "message=Observability test" in formatted
    finally:
        reset_request_id(token)
