from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)


# --------------------------------------------------
# Load document
# --------------------------------------------------

def load_document(file_path):
    """
    Load a TXT, PDF, or DOCX document.

    Returns:
        List of LangChain Document objects.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".txt":
        loader = TextLoader(str(file_path))

    elif extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    else:
        raise ValueError(
            f"Unsupported document type: {extension}"
        )

    return loader.load()


# --------------------------------------------------
# Local development test
# --------------------------------------------------

if __name__ == "__main__":

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "company_policy.txt"
    )

    documents = load_document(policy_path)

    for document in documents:

        print("Content:")
        print(document.page_content)

        print("Metadata:")
        print(document.metadata)