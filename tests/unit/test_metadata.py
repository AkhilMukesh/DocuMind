from langchain_core.documents import Document

from chunking import split_documents


def test_metadata_is_preserved_after_chunking():

    document = Document(
        page_content=(
            "Employees can work 3 days from home "
            "and 2 days from office after manager approval."
        ),
        metadata={
            "document_id": "doc-123",
            "filename": "company_policy.txt",
            "source": "company_policy.txt"
        }
    )

    chunks = split_documents([document])

    assert chunks

    for chunk in chunks:

        assert chunk.metadata["document_id"] == "doc-123"
        assert chunk.metadata["filename"] == "company_policy.txt"
        assert chunk.metadata["source"] == "company_policy.txt"


def test_metadata_is_preserved_for_multiple_documents():

    documents = [
        Document(
            page_content="Company policy information.",
            metadata={
                "document_id": "doc-001",
                "filename": "company_policy.txt"
            }
        ),
        Document(
            page_content="Security policy information.",
            metadata={
                "document_id": "doc-002",
                "filename": "security_policy.txt"
            }
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) == 2

    assert chunks[0].metadata["document_id"] == "doc-001"
    assert chunks[0].metadata["filename"] == "company_policy.txt"

    assert chunks[1].metadata["document_id"] == "doc-002"
    assert chunks[1].metadata["filename"] == "security_policy.txt"