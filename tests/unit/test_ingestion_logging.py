
import logging
from types import SimpleNamespace

import pytest
import ingestion


def test_ingestion_logs_success(tmp_path, monkeypatch, caplog):
    file_path = tmp_path / "company_policy.txt"
    file_path.write_text("Example policy", encoding="utf-8")

    fake_chunks = [
        SimpleNamespace(metadata={}),
        SimpleNamespace(metadata={}),
    ]

    monkeypatch.setattr(
        ingestion,
        "calculate_file_hash",
        lambda path: "test-hash"
    )
    monkeypatch.setattr(
        ingestion,
        "document_exists",
        lambda content_hash: False
    )
    monkeypatch.setattr(
        ingestion,
        "load_document",
        lambda path: ["fake-document"]
    )
    monkeypatch.setattr(
        ingestion,
        "split_documents",
        lambda documents: fake_chunks
    )

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, chunks):
            assert len(chunks) == 2

    monkeypatch.setattr(ingestion, "Chroma", FakeVectorStore)

    registered = {}

    def fake_add_document(**kwargs):
        registered.update(kwargs)

    monkeypatch.setattr(
        ingestion,
        "add_document",
        fake_add_document
    )

    with caplog.at_level(logging.INFO, logger="ingestion"):
        result = ingestion.ingest_document(
            file_path,
            "company_policy.txt"
        )

    assert result["status"] == "success"
    assert result["chunk_count"] == 2
    assert registered["content_hash"] == "test-hash"

    assert "Document ingestion started" in caplog.text
    assert "Document loading completed" in caplog.text
    assert "Document chunking completed" in caplog.text
    assert "Vector storage completed" in caplog.text
    assert "Document ingestion completed" in caplog.text


def test_ingestion_logs_duplicate(tmp_path, monkeypatch, caplog):
    file_path = tmp_path / "company_policy.txt"
    file_path.write_text("Example policy", encoding="utf-8")

    monkeypatch.setattr(
        ingestion,
        "calculate_file_hash",
        lambda path: "existing-hash"
    )
    monkeypatch.setattr(
        ingestion,
        "document_exists",
        lambda content_hash: True
    )

    with caplog.at_level(logging.INFO, logger="ingestion"):
        result = ingestion.ingest_document(
            file_path,
            "company_policy.txt"
        )

    assert result["status"] == "duplicate"
    assert "reason=duplicate" in caplog.text
    assert "Document ingestion completed" not in caplog.text


def test_ingestion_logs_failure_and_reraises(
    tmp_path,
    monkeypatch,
    caplog
):
    file_path = tmp_path / "company_policy.txt"
    file_path.write_text("Example policy", encoding="utf-8")

    monkeypatch.setattr(
        ingestion,
        "calculate_file_hash",
        lambda path: "test-hash"
    )
    monkeypatch.setattr(
        ingestion,
        "document_exists",
        lambda content_hash: False
    )

    def fail_loading(path):
        raise ValueError("Simulated loader failure")

    monkeypatch.setattr(ingestion, "load_document", fail_loading)

    with caplog.at_level(logging.ERROR, logger="ingestion"):
        with pytest.raises(ValueError, match="Simulated loader failure"):
            ingestion.ingest_document(
                file_path,
                "company_policy.txt"
            )

    assert "Document ingestion failed" in caplog.text
    assert "ValueError: Simulated loader failure" in caplog.text
