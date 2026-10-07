from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from chunking import split_documents


def test_document_chunk_embedding_retrieval_integration(tmp_path):
    documents = [
        Document(
            page_content=(
                "Employees can work 3 days from home "
                "and 2 days from office after manager approval."
            ),
            metadata={
                "document_id": "integration-doc-001",
                "filename": "company_policy.txt",
            },
        ),
        Document(
            page_content=(
                "Employees must use strong passwords "
                "and keep account credentials private."
            ),
            metadata={
                "document_id": "integration-doc-001",
                "filename": "company_policy.txt",
            },
        ),
    ]

    chunks = split_documents(documents)

    assert len(chunks) == 2

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="integration_test_collection",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma"),
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    results = retriever.invoke(
        "How many days can employees work from home?"
    )

    assert len(results) == 1

    result = results[0]

    assert "3 days from home" in result.page_content
    assert result.metadata["document_id"] == "integration-doc-001"
    assert result.metadata["filename"] == "company_policy.txt"