from langchain_core.documents import Document

import retriever


def test_create_retriever_uses_mmr(monkeypatch):

    captured = {}

    class FakeVectorStore:

        def as_retriever(
            self,
            search_type,
            search_kwargs
        ):

            captured["search_type"] = search_type
            captured["search_kwargs"] = search_kwargs

            return "fake-retriever"

    monkeypatch.setattr(
        retriever,
        "vector_store",
        FakeVectorStore()
    )

    result = retriever.create_retriever()

    assert result == "fake-retriever"

    assert captured["search_type"] == "mmr"

    assert captured["search_kwargs"]["k"] == 2


def test_create_retriever_with_document_filter(
    monkeypatch
):

    captured = {}

    class FakeVectorStore:

        def as_retriever(
            self,
            search_type,
            search_kwargs
        ):

            captured["search_type"] = search_type
            captured["search_kwargs"] = search_kwargs

            return "fake-retriever"

    monkeypatch.setattr(
        retriever,
        "vector_store",
        FakeVectorStore()
    )

    retriever.create_retriever(
        document_id="doc-123"
    )

    assert captured["search_kwargs"]["k"] == 2

    assert captured["search_kwargs"]["filter"] == {
        "document_id": "doc-123"
    }


def test_retrieve_documents(monkeypatch):

    expected_documents = [
        Document(
            page_content=(
                "Employees can work 3 days from home."
            ),
            metadata={
                "document_id": "doc-001",
                "filename": "company_policy.txt"
            }
        ),
        Document(
            page_content=(
                "Employees must follow working hours."
            ),
            metadata={
                "document_id": "doc-001",
                "filename": "company_policy.txt"
            }
        )
    ]

    class FakeRetriever:

        def invoke(self, query):

            return expected_documents

    monkeypatch.setattr(
        retriever,
        "create_retriever",
        lambda document_id=None: FakeRetriever()
    )

    results = retriever.retrieve_documents(
        "How many days can I work from home?"
    )

    assert len(results) == 2

    assert (
        results[0].metadata["document_id"]
        == "doc-001"
    )

    assert (
        results[0].metadata["filename"]
        == "company_policy.txt"
    )


def test_retrieve_documents_with_document_id(
    monkeypatch
):

    captured_document_id = {}

    expected_documents = [
        Document(
            page_content="Security policy.",
            metadata={
                "document_id": "doc-002",
                "filename": "security_policy.txt"
            }
        )
    ]

    class FakeRetriever:

        def invoke(self, query):

            return expected_documents

    def fake_create_retriever(document_id=None):

        captured_document_id["value"] = document_id

        return FakeRetriever()

    monkeypatch.setattr(
        retriever,
        "create_retriever",
        fake_create_retriever
    )

    results = retriever.retrieve_documents(
        "What are the security requirements?",
        document_id="doc-002"
    )

    assert captured_document_id["value"] == "doc-002"

    assert len(results) == 1

    assert (
        results[0].metadata["document_id"]
        == "doc-002"
    )