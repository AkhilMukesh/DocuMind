from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import rag_pipeline


def test_format_documents():
    documents = [
        Document(
            page_content="Employees can work 3 days from home.",
            metadata={
                "filename": "company_policy.txt"
            }
        ),
        Document(
            page_content="Employees must follow working hours.",
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    context = rag_pipeline.format_documents(documents)

    assert "[1] company_policy.txt" in context
    assert "[2] company_policy.txt" in context
    assert "Employees can work 3 days from home." in context
    assert "Employees must follow working hours." in context


def test_ask_question_returns_answer_and_sources(monkeypatch):
    documents = [
        Document(
            page_content="Employees can work 3 days from home.",
            metadata={
                "filename": "company_policy.txt",
                "document_id": "doc-001"
            }
        )
    ]

    class FakeRetriever:
        def invoke(self, question):
            return documents

    monkeypatch.setattr(
        rag_pipeline,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    class FakeLLMResponse:
        content = "Employees can work 3 days from home. [1]"

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
            content="Employees can work 3 days from home. [1]"
        )

    monkeypatch.setattr(
        rag_pipeline,
        "llm",
        FakeLLM()
    )

    answer, sources = rag_pipeline.ask_question(
        "How many days can employees work from home?"
    )

    assert answer == "Employees can work 3 days from home. [1]"

    assert len(sources) == 1
    assert sources[0]["citation"] == "[1]"
    assert sources[0]["location"] == "company_policy.txt"
    assert sources[0]["content"] == (
        "Employees can work 3 days from home."
    )


def test_ask_question_passes_document_id(monkeypatch):
    captured = {}

    documents = [
        Document(
            page_content="Security policy.",
            metadata={
                "filename": "security_policy.txt",
                "document_id": "doc-002"
            }
        )
    ]

    class FakeRetriever:
        def invoke(self, question):
            return documents

    def fake_create_retriever(document_id=None):
        captured["document_id"] = document_id
        return FakeRetriever()

    monkeypatch.setattr(
        rag_pipeline,
        "create_retriever",
        fake_create_retriever
    )

    class FakeLLMResponse:
        content = "Security policy information. [1]"

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
            content="Security policy information. [1]"
        )

    monkeypatch.setattr(
        rag_pipeline,
        "llm",
        FakeLLM()
    )

    answer, sources = rag_pipeline.ask_question(
        "What are the security requirements?",
        document_id="doc-002"
    )

    assert captured["document_id"] == "doc-002"
    assert answer == "Security policy information. [1]"
    assert sources[0]["citation"] == "[1]"

def test_ask_question_sends_retrieved_context_to_llm(monkeypatch):
    documents = [
        Document(
            page_content="Employees can work 3 days from home.",
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    class FakeRetriever:
        def invoke(self, question):
            return documents

    monkeypatch.setattr(
        rag_pipeline,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["messages"] = messages

            return AIMessage(
                content="Employees can work 3 days from home. [1]"
            )

    monkeypatch.setattr(
        rag_pipeline,
        "llm",
        FakeLLM()
    )

    answer, sources = rag_pipeline.ask_question(
        "How many days can employees work from home?"
    )

    assert "Employees can work 3 days from home." in str(
        captured["messages"]
    )

    assert answer == (
        "Employees can work 3 days from home. [1]"
    )