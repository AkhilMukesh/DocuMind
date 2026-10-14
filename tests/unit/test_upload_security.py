
from io import BytesIO
from zipfile import ZipFile

import pytest

from upload_security import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    validate_question,
    validate_upload,
)


def make_docx():
    buffer = BytesIO()

    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types></Types>"
        )
        archive.writestr(
            "word/document.xml",
            "<document></document>"
        )

    return buffer.getvalue()


def test_accepts_valid_utf8_text():
    assert validate_upload(
        "policy.txt",
        "Remote work policy".encode("utf-8")
    )


def test_rejects_unsupported_extension():
    with pytest.raises(UploadValidationError):
        validate_upload("malware.exe", b"content")


def test_rejects_pdf_with_invalid_signature():
    with pytest.raises(UploadValidationError, match="valid PDF"):
        validate_upload("policy.pdf", b"not a PDF")


def test_rejects_empty_file():
    with pytest.raises(UploadValidationError, match="empty"):
        validate_upload("policy.txt", b"")


def test_rejects_oversized_file():
    with pytest.raises(UploadValidationError, match="10 MiB"):
        validate_upload(
            "policy.txt",
            b"x" * (MAX_UPLOAD_BYTES + 1)
        )


def test_accepts_valid_docx_structure():
    assert validate_upload("policy.docx", make_docx())


def test_rejects_invalid_docx():
    with pytest.raises(UploadValidationError, match="valid DOCX"):
        validate_upload("policy.docx", b"not a zip file")


def test_question_is_trimmed():
    assert validate_question("  Can I work remotely?  ") == (
        "Can I work remotely?"
    )


def test_rejects_empty_question():
    with pytest.raises(ValueError, match="enter a question"):
        validate_question("   ")


def test_rejects_question_over_limit():
    with pytest.raises(ValueError, match="2000-character"):
        validate_question("x" * 2001)
