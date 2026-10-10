import hashlib
import uuid
from pathlib import Path

import logging
import time
from metrics import metrics

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from document_registry import (
    initialize_database,
    add_document,
    document_exists,
    delete_document,
    get_document,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# Calculate file hash
# --------------------------------------------------

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# --------------------------------------------------
# Load document
# --------------------------------------------------

def load_document(file_path):

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

    return text_splitter.split_documents(
        documents
    )


# --------------------------------------------------
# Ingest document
# --------------------------------------------------


def ingest_document(file_path, original_filename):
    start_time = time.perf_counter()
    metrics.increment("ingestion.started")
    file_path = Path(file_path)

    logger.info(
        "Document ingestion started | filename=%s",
        original_filename
    )

    try:
        # ----------------------------------------------
        # Calculate file hash
        # ----------------------------------------------

        content_hash = calculate_file_hash(file_path)

        logger.info(
            "Document hash calculated | filename=%s",
            original_filename
        )

        # ----------------------------------------------
        # Duplicate check
        # ----------------------------------------------

        if document_exists(content_hash):
            elapsed_seconds = time.perf_counter() - start_time

            logger.info(
                "Document ingestion skipped | reason=duplicate "
                "| filename=%s | duration_seconds=%.3f",
                original_filename,
                elapsed_seconds
            )
            metrics.increment("ingestion.duplicate")
            metrics.observe_duration(
                "ingestion.duration_seconds",
                    elapsed_seconds
            )
            return {
                "status": "duplicate",
                "filename": original_filename
            }

        # ----------------------------------------------
        # Load document
        # ----------------------------------------------

        documents = load_document(file_path)

        logger.info(
            "Document loading completed | filename=%s "
            "| page_or_document_count=%s",
            original_filename,
            len(documents)
        )

        # ----------------------------------------------
        # Chunk document
        # ----------------------------------------------

        chunks = split_documents(documents)

        logger.info(
            "Document chunking completed | filename=%s "
            "| chunk_count=%s",
            original_filename,
            len(chunks)
        )

        # ----------------------------------------------
        # Create document ID
        # ----------------------------------------------

        document_id = str(uuid.uuid4())

        # ----------------------------------------------
        # Add document ID and filename to metadata
        # ----------------------------------------------

        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = original_filename

        # ----------------------------------------------
        # Store chunks in Chroma
        # ----------------------------------------------

        vector_store = Chroma(
            collection_name="company_policy",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PATH)
        )

        vector_store.add_documents(chunks)

        logger.info(
            "Vector storage completed | filename=%s "
            "| document_id=%s | chunk_count=%s",
            original_filename,
            document_id,
            len(chunks)
        )

        # ----------------------------------------------
        # Register document in SQLite
        # ----------------------------------------------

        add_document(
            document_id=document_id,
            filename=original_filename,
            file_type=file_path.suffix.lower(),
            content_hash=content_hash,
            chunk_count=len(chunks)
        )

        # Log success only after registry registration
        elapsed_seconds = time.perf_counter() - start_time


        logger.info(
            "Document ingestion completed | filename=%s "
            "| document_id=%s | chunk_count=%s "
            "| duration_seconds=%.3f",
            original_filename,
            document_id,
            len(chunks),
            elapsed_seconds
        )
        metrics.increment("ingestion.success")
        metrics.observe_duration(
             "ingestion.duration_seconds",
             elapsed_seconds
        )
        return {
            "status": "success",
            "filename": original_filename,
            "document_id": document_id,
            "chunk_count": len(chunks)
        }

    except Exception:
        elapsed_seconds = time.perf_counter() - start_time
        metrics.increment("ingestion.failure")
        metrics.observe_duration(
        "ingestion.duration_seconds",
         elapsed_seconds
        )

        logger.exception(
            "Document ingestion failed | filename=%s "
            "| duration_seconds=%.3f",
            original_filename,
            elapsed_seconds
        )

        # Preserve existing error propagation.
        raise


def delete_document_completely(document_id):

    document = get_document(document_id)

    if document is None:

        return {
            "status": "not_found"
        }

    vector_store = Chroma(
        collection_name="company_policy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    # Remove all chunks belonging to this document
    vector_store.delete(
        where={
            "document_id": document_id
        }
    )

    # Remove document registry record
    deleted = delete_document(
        document_id
    )

    if not deleted:

        return {
            "status": "registry_delete_failed"
        }

    return {
        "status": "success",
        "filename": document[1],
        "document_id": document_id
    }