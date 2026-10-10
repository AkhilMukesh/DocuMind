
import logging
from types import SimpleNamespace

import pytest
import rag_pipeline


def configure_fake_pipeline(monkeypatch, fake_llm):
    document = SimpleNamespace(
        page_content="Employees can work remotely.",
        metadata={"filename": "company_policy.txt"}
    )

    # Mock retrieval without accessing Chroma.
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_documents",
        lambda question, document_id=None: [document]
    )

    # Mock context formatting.
    monkeypatch.setattr(
        rag_pipeline,
        "format_documents",
        lambda documents: (
            "[1] company_policy.txt\n"
            "Employees can work remotely."
        )
    )

    # IMPORTANT:
    # Do not patch rag_pipeline.rag_prompt.invoke directly.
    # Replace the whole prompt object with a simple fake object.
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


def test_ask_question_logs_llm_success(monkeypatch, caplog):
    class FakeLLM:
        def invoke(self, messages):
            return "Employees can work remotely. [1]"

    configure_fake_pipeline(monkeypatch, FakeLLM())

    with caplog.at_level(
        logging.INFO,
        logger="rag_pipeline"
    ):
        answer, sources = rag_pipeline.ask_question(
            "What does the policy say?"
        )

    assert answer == "Employees can work remotely. [1]"
    assert len(sources) == 1

    assert "LLM request started" in caplog.text
    assert "LLM request completed" in caplog.text
    assert "duration_seconds=" in caplog.text
    assert "RAG question processing completed" in caplog.text


def test_ask_question_logs_llm_failure(monkeypatch, caplog):
    class FakeLLM:
        def invoke(self, messages):
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

    assert "LLM request failed" in caplog.text
    assert "RAG question processing failed" in caplog.text
    assert "Simulated LLM failure" in caplog.text
