import pytest


@pytest.fixture
def sample_text():

    return """
    Company Remote Work Policy

    Employees can work remotely up to three days per week.

    Employees must attend all required meetings during
    their scheduled working hours.

    Employees must maintain a reliable internet connection
    while working remotely.

    Confidential company information must only be accessed
    using company-approved devices.
    """


@pytest.fixture
def sample_document_data():

    return {
        "filename": "company_policy.txt",
        "file_type": "txt",
        "content_hash": "test-hash-123",
        "chunk_count": 4
    }


@pytest.fixture
def temporary_text_file(
    tmp_path,
    sample_text
):

    file_path = tmp_path / "company_policy.txt"

    file_path.write_text(
        sample_text,
        encoding="utf-8"
    )

    return file_path