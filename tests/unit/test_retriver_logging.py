
import logging

import pytest
import retriever


def test_retrieval_logs_success(monkeypatch, caplog):
    expected_documents = ["document-1", "document-2"]

    class FakeRetriever:
        def invoke(self, query):
            return expected_documents

    monkeypatch.setattr(
        retriever,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    with caplog.at_level(logging.INFO, logger="retriever"):
        result = retriever.retrieve_documents("test question")

    assert result == expected_documents
    assert "Retrieval started" in caplog.text
    assert "Retrieval completed" in caplog.text
    assert "result_count=2" in caplog.text
    assert "duration_seconds=" in caplog.text
    assert "document_filter=disabled" in caplog.text


def test_retrieval_logs_failure_and_reraises(
    monkeypatch,
    caplog
):
    class FakeRetriever:
        def invoke(self, query):
            raise RuntimeError("Simulated retrieval failure")

    monkeypatch.setattr(
        retriever,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    with caplog.at_level(logging.ERROR, logger="retriever"):
        with pytest.raises(
            RuntimeError,
            match="Simulated retrieval failure"
        ):
            retriever.retrieve_documents("test question")

    assert "Retrieval failed" in caplog.text
    assert "Simulated retrieval failure" in caplog.text
    assert "duration_seconds=" in caplog.text
