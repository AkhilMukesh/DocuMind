from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load document
# --------------------------------------------------

def load_document(file_path):
    """
    Load a document based on its file extension.
    """

    extension = file_path.suffix.lower()

    if extension == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    elif extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader.load()


# --------------------------------------------------
# Chunk document
# --------------------------------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return text_splitter.split_documents(documents)


# --------------------------------------------------
# Store documents
# --------------------------------------------------

def ingest_document(file_path):

    documents = load_document(file_path)

    chunks = split_documents(documents)

    vector_store = Chroma(
        collection_name="company_policy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    vector_store.add_documents(chunks)

    return len(documents), len(chunks)