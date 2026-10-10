from pathlib import Path
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time


logger = logging.getLogger(__name__)

# --------------------------------------------------
# Project paths this
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Vector store
# --------------------------------------------------

vector_store = Chroma(
    collection_name="company_policy",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)


# --------------------------------------------------
# Create retriever
# --------------------------------------------------

def create_retriever(document_id=None):

    search_kwargs = {
        "k": 2
    }

    if document_id:

        search_kwargs["filter"] = {
            "document_id": document_id
        }

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )


# --------------------------------------------------
# Retrieve documents
# --------------------------------------------------


def retrieve_documents(query, document_id=None):
    start_time = time.perf_counter()

    filter_status = "enabled" if document_id else "disabled"

    logger.info(
        "Retrieval started | document_filter=%s",
        filter_status
    )

    try:
        retriever = create_retriever(
            document_id=document_id
        )

        documents = retriever.invoke(query)

        elapsed_seconds = time.perf_counter() - start_time

        logger.info(
            "Retrieval completed | result_count=%s "
            "| document_filter=%s | duration_seconds=%.3f",
            len(documents),
            filter_status,
            elapsed_seconds
        )

        return documents

    except Exception:
        elapsed_seconds = time.perf_counter() - start_time

        logger.exception(
            "Retrieval failed | document_filter=%s "
            "| duration_seconds=%.3f",
            filter_status,
            elapsed_seconds
        )
        raise


# --------------------------------------------------
# Local development test
# --------------------------------------------------

if __name__ == "__main__":

    query = "How many days can I work from home?"

    results = retrieve_documents(query)

    print("QUESTION:")
    print(query)

    print("\nRETRIEVED DOCUMENTS:")

    for index, document in enumerate(results):

        print(f"\n--- Result {index + 1} ---")

        print("Content:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)