from langchain_core.documents import Document

from chunking import split_documents


# --------------------------------------------------
# Test basic chunking
# --------------------------------------------------

def test_split_documents_creates_chunks():

    content = "This is a test sentence. " * 100

    documents = [
        Document(
            page_content=content,
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    chunks = split_documents(documents)

    assert chunks
    assert len(chunks) > 1


# --------------------------------------------------
# Test content is preserved
# --------------------------------------------------

def test_split_documents_preserves_content():

    content = (
        "Employees can work 3 days from home "
        "and 2 days from office after manager approval."
    )

    documents = [
        Document(
            page_content=content,
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    chunks = split_documents(documents)

    combined_content = " ".join(
        chunk.page_content
        for chunk in chunks
    )

    assert "3 days from home" in combined_content
    assert "2 days from office" in combined_content


# --------------------------------------------------
# Test metadata preservation
# --------------------------------------------------

def test_split_documents_preserves_metadata():

    documents = [
        Document(
            page_content="Company policy information.",
            metadata={
                "filename": "company_policy.txt",
                "document_id": "doc-123"
            }
        )
    ]

    chunks = split_documents(documents)

    assert chunks

    for chunk in chunks:

        assert chunk.metadata["filename"] == (
            "company_policy.txt"
        )

        assert chunk.metadata["document_id"] == (
            "doc-123"
        )


# --------------------------------------------------
# Test chunk size
# --------------------------------------------------

def test_chunk_size_is_within_expected_limit():

    content = "A" * 2000

    documents = [
        Document(
            page_content=content,
            metadata={}
        )
    ]

    chunks = split_documents(documents)

    assert chunks

    for chunk in chunks:

        assert len(chunk.page_content) <= 500