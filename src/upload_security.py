
from pathlib import Path
from io import BytesIO
from zipfile import BadZipFile, ZipFile


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


class UploadValidationError(ValueError):
    """Raised when an uploaded document fails security validation."""


def validate_upload(filename, content):
    if not isinstance(filename, str) or not filename.strip():
        raise UploadValidationError("A filename is required.")

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file type. Upload a PDF, TXT, or DOCX file."
        )

    if not isinstance(content, bytes):
        raise UploadValidationError("File content must be bytes.")

    if not content:
        raise UploadValidationError("The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            "File exceeds the 10 MiB upload limit."
        )

    if extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError(
                "The uploaded file is not a valid PDF."
            )

    elif extension == ".txt":
        if b"\x00" in content:
            raise UploadValidationError(
                "The text file contains invalid binary content."
            )

        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadValidationError(
                "The text file must use UTF-8 encoding."
            ) from exc

    elif extension == ".docx":
        try:
            with ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())

                if (
                    "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                ):
                    raise UploadValidationError(
                        "The uploaded file is not a valid DOCX document."
                    )

                if archive.testzip() is not None:
                    raise UploadValidationError(
                        "The DOCX archive is corrupted."
                    )

        except BadZipFile as exc:
            raise UploadValidationError(
                "The uploaded file is not a valid DOCX document."
            ) from exc

    return True



MAX_QUESTION_LENGTH = 2000


def validate_question(question):
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            "Question exceeds the 2000-character limit."
        )

    return question
