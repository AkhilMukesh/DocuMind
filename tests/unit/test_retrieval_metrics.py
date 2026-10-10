
from types import SimpleNamespace

import pytest
import rag_pipeline
from metrics import metrics


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


def test_retrieval_success_records_duration(monkeypatch):
    documents = [
        SimpleNamespace(page_content="Test document", metadata={})
    ]

    class FakeRetriever:
        def invoke(self, question):
            return documents

    monkeypatch.setattr(
        rag_pipeline,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    result = rag_pipeline.retrieve_documents("Test question")

    snapshot = metrics.snapshot()

    assert result == documents
    assert snapshot["counters"]["retrieval.started"] == 1
    assert snapshot["counters"]["retrieval.success"] == 1
    assert snapshot["counters"]["retrieval.documents_returned"] == 1

    duration = snapshot["durations"]["retrieval.duration_seconds"]
    assert duration["count"] == 1
    assert duration["total_seconds"] >= 0


def test_retrieval_failure_records_metrics(monkeypatch):
    class FakeRetriever:
        def invoke(self, question):
            raise RuntimeError("Simulated retrieval failure")

    monkeypatch.setattr(
        rag_pipeline,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    with pytest.raises(RuntimeError, match="Simulated retrieval failure"):
        rag_pipeline.retrieve_documents("Test question")

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["retrieval.started"] == 1
    assert snapshot["counters"]["retrieval.failure"] == 1
    assert "retrieval.success" not in snapshot["counters"]

    duration = snapshot["durations"]["retrieval.duration_seconds"]
    assert duration["count"] == 1
