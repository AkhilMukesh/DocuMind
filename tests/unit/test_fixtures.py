def test_sample_text(sample_text):

    assert "remote" in sample_text.lower()

    assert "three days" in sample_text.lower()


def test_sample_document_data(sample_document_data):

    assert sample_document_data["filename"] == (
        "company_policy.txt"
    )

    assert sample_document_data["file_type"] == "txt"


def test_temporary_text_file(temporary_text_file):

    assert temporary_text_file.exists()

    assert temporary_text_file.name == (
        "company_policy.txt"
    )

    content = temporary_text_file.read_text(
        encoding="utf-8"
    )

    assert "three days" in content.lower()