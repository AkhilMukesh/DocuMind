
from types import SimpleNamespace

import pytest
import rag_pipeline
from metrics import metrics


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


def test_successful_question_records_metrics(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            return "Employees can work remotely. [1]"

    configure_fake_pipeline(monkeypatch, FakeLLM())

    answer, sources = rag_pipeline.ask_question(
        "Can employees work remotely?"
    )

    snapshot = metrics.snapshot()
    counters = snapshot["counters"]
    durations = snapshot["durations"]

    assert answer == "Employees can work remotely. [1]"
    assert len(sources) == 1

    assert counters["rag.request.started"] == 1
    assert counters["rag.request.success"] == 1
    assert counters["llm.request.started"] == 1
    assert counters["llm.request.success"] == 1
    assert counters["retrieval.success"] == 1
    assert counters["retrieval.documents_returned"] == 1

    assert durations["llm.request.duration_seconds"]["count"] == 1


def test_llm_failure_records_failure_metrics(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            raise RuntimeError("Simulated LLM failure")

    configure_fake_pipeline(monkeypatch, FakeLLM())

    with pytest.raises(RuntimeError, match="Simulated LLM failure"):
        rag_pipeline.ask_question("Test question")

    counters = metrics.snapshot()["counters"]
    durations = metrics.snapshot()["durations"]

    assert counters["rag.request.started"] == 1
    assert counters["rag.request.failure"] == 1
    assert counters["llm.request.started"] == 1
    assert counters["llm.request.failure"] == 1
    assert "rag.request.success" not in counters

    assert durations["llm.request.duration_seconds"]["count"] == 1
