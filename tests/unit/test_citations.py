from langchain_core.documents import Document

from rag_pipeline import build_sources


def test_build_sources_creates_citation_numbers():
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

    sources = build_sources(documents)

    assert len(sources) == 2

    assert sources[0]["citation"] == "[1]"
    assert sources[1]["citation"] == "[2]"


def test_build_sources_preserves_filename():
    documents = [
        Document(
            page_content="Company policy information.",
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    sources = build_sources(documents)

    assert sources[0]["location"] == "company_policy.txt"


def test_build_sources_preserves_content():
    documents = [
        Document(
            page_content="Employees can work 3 days from home.",
            metadata={
                "filename": "company_policy.txt"
            }
        )
    ]

    sources = build_sources(documents)

    assert (
        sources[0]["content"]
        == "Employees can work 3 days from home."
    )


def test_build_sources_handles_pdf_page_number():
    documents = [
        Document(
            page_content="Security information.",
            metadata={
                "filename": "security_policy.pdf",
                "page": 2
            }
        )
    ]

    sources = build_sources(documents)

    assert sources[0]["citation"] == "[1]"
    assert sources[0]["location"] == "security_policy.pdf — Page 3"


def test_build_sources_falls_back_to_source():
    documents = [
        Document(
            page_content="Company policy information.",
            metadata={
                "source": "company_policy.txt"
            }
        )
    ]

    sources = build_sources(documents)

    assert sources[0]["location"] == "company_policy.txt"


def test_build_sources_handles_missing_filename_and_source():
    documents = [
        Document(
            page_content="Some document content.",
            metadata={}
        )
    ]

    sources = build_sources(documents)

    assert sources[0]["location"] == "unknown"