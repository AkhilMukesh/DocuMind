from docx import Document
from reportlab.pdfgen import canvas
import pytest

from document_loader import load_document


# --------------------------------------------------
# TXT
# --------------------------------------------------

def test_load_txt_document(temporary_text_file):

    documents = load_document(
        str(temporary_text_file)
    )

    assert documents
    assert len(documents) >= 1

    content = "\n".join(
        document.page_content
        for document in documents
    )

    assert "three days" in content.lower()


# --------------------------------------------------
# Missing file
# --------------------------------------------------

def test_load_missing_document(tmp_path):

    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(str(missing_file))


# --------------------------------------------------
# Unsupported file
# --------------------------------------------------

def test_load_unsupported_document(tmp_path):

    unsupported_file = tmp_path / "company_policy.csv"

    unsupported_file.write_text(
        "some,data",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_document(str(unsupported_file))


# --------------------------------------------------
# PDF
# --------------------------------------------------

def test_load_pdf_document(tmp_path):

    pdf_file = tmp_path / "company_policy.pdf"

    pdf = canvas.Canvas(str(pdf_file))

    pdf.drawString(
        100,
        750,
        "Employees can work remotely three days per week."
    )

    pdf.save()

    documents = load_document(
        str(pdf_file)
    )

    assert documents

    content = "\n".join(
        document.page_content
        for document in documents
    )

    assert "three days" in content.lower()


# --------------------------------------------------
# DOCX
# --------------------------------------------------

def test_load_docx_document(tmp_path):

    docx_file = tmp_path / "company_policy.docx"

    document = Document()

    document.add_paragraph(
        "Employees can work remotely three days per week."
    )

    document.save(str(docx_file))

    documents = load_document(
        str(docx_file)
    )

    assert documents

    content = "\n".join(
        document.page_content
        for document in documents
    )

    assert "three days" in content.lower()