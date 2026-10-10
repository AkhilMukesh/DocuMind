import os
from pathlib import Path
import logging
import time
from metrics import metrics
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from request_context import (
    set_request_id,
    reset_request_id,
    get_request_id,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to the .env file."
    )


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR DATABASE
# ============================================================

vector_store = Chroma(
    collection_name="company_policy",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(
    """
You are DocuMind AI, a document question-answering assistant.

Your task is to answer the user's question using ONLY the
information provided in the context.

Rules:

1. Use only the provided context.

2. Do not use outside knowledge.

3. Do not invent or assume facts.

4. If the answer is not available in the context, say:

   "The information is not available in the provided documents."

5. When making a factual claim, include the citation number
   of the context item that supports the claim.

6. Use citations in this format:

   [1]
   [2]

7. Do not create citation numbers that do not exist in the context.

8. Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# RETRIEVER
# ============================================================

def create_retriever(document_id=None):
    """
    Create a retriever.

    If document_id is provided, retrieval is restricted
    to that document.

    Otherwise, all indexed documents can be searched.
    """

    search_kwargs = {
        "k": 2
    }

    if document_id:
        search_kwargs["filter"] = {
            "document_id": document_id
        }

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================


def retrieve_documents(question, document_id=None):
    start_time = time.perf_counter()
    metrics.increment("retrieval.started")

    try:
        active_retriever = create_retriever(
            document_id=document_id
        )

        documents = active_retriever.invoke(question)

        elapsed_seconds = time.perf_counter() - start_time

        metrics.increment("retrieval.success")
        metrics.increment(
            "retrieval.documents_returned",
            len(documents)
        )
        metrics.observe_duration(
            "retrieval.duration_seconds",
            elapsed_seconds
        )

        logger.info(
            "Retrieval completed | request_id=%s "
            "| result_count=%s | duration_seconds=%.3f",
            get_request_id() or "-",
            len(documents),
            elapsed_seconds
        )

        return documents

    except Exception:
        elapsed_seconds = time.perf_counter() - start_time

        metrics.increment("retrieval.failure")
        metrics.observe_duration(
            "retrieval.duration_seconds",
            elapsed_seconds
        )

        logger.exception(
            "Retrieval failed | request_id=%s "
            "| duration_seconds=%.3f",
            get_request_id() or "-",
            elapsed_seconds
        )
        raise


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_documents(documents):
    """
    Convert retrieved documents into citation-aware context.
    """

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        filename = document.metadata.get(
            "filename",
            document.metadata.get(
                "source",
                "unknown"
            )
        )

        page = document.metadata.get("page")

        if page is not None:

            page_number = page + 1

            location = (
                f"{filename} — Page "
                f"{page_number}"
            )

        else:

            location = filename

        formatted_documents.append(
            f"[{index}] {location}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        formatted_documents
    )


# ============================================================
# BUILD SOURCES
# ============================================================

def build_sources(documents):
    """
    Convert retrieved documents into structured source objects.

    These sources are displayed in the Streamlit UI.
    """

    sources = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        filename = document.metadata.get(
            "filename",
            document.metadata.get(
                "source",
                "unknown"
            )
        )

        page = document.metadata.get("page")

        if page is not None:

            location = (
                f"{filename} — Page "
                f"{page + 1}"
            )

        else:

            location = filename

        sources.append(
            {
                "citation": f"[{index}]",
                "location": location,
                "content": document.page_content
            }
        )

    return sources


# ============================================================
# MAIN RAG PIPELINE
# ============================================================


def ask_question(question, document_id=None):
    request_id, token = set_request_id()

    metrics.increment("rag.request.started")

    logger.info(
        "RAG question processing started | request_id=%s",
        request_id
    )

    try:
        documents = retrieve_documents(
            question,
            document_id=document_id
        )

        metrics.increment("retrieval.success")
        metrics.increment(
            "retrieval.documents_returned",
            len(documents)
        )

        logger.info(
            "Document retrieval completed | request_id=%s "
            "| result_count=%s",
            request_id,
            len(documents)
        )

        context = format_documents(documents)

        messages = rag_prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        start_time = time.perf_counter()
        metrics.increment("llm.request.started")

        logger.info(
            "LLM request started | request_id=%s",
            request_id
        )

        try:
            response = llm.invoke(messages)

        except Exception:
            elapsed_seconds = time.perf_counter() - start_time

            metrics.increment("llm.request.failure")
            metrics.observe_duration(
                "llm.request.duration_seconds",
                elapsed_seconds
            )

            logger.exception(
                "LLM request failed | request_id=%s "
                "| duration_seconds=%.3f",
                request_id,
                elapsed_seconds
            )
            raise

        elapsed_seconds = time.perf_counter() - start_time

        metrics.increment("llm.request.success")
        metrics.observe_duration(
            "llm.request.duration_seconds",
            elapsed_seconds
        )

        logger.info(
            "LLM request completed | request_id=%s "
            "| duration_seconds=%.3f",
            request_id,
            elapsed_seconds
        )

        answer = StrOutputParser().invoke(response)
        sources = build_sources(documents)

        metrics.increment("rag.request.success")

        logger.info(
            "RAG question processing completed | request_id=%s "
            "| source_count=%s",
            request_id,
            len(sources)
        )

        return answer, sources

    except Exception:
        metrics.increment("rag.request.failure")

        logger.exception(
            "RAG question processing failed | request_id=%s",
            request_id
        )
        raise

    finally:
        reset_request_id(token)




def answer_question_with_documents(question, document_id=None):
    """
    Retrieve documents once and generate an answer using
    exactly those retrieved documents.

    Returns:
        answer,
        documents,
        sources
    """

    documents = retrieve_documents(
        question,
        document_id=document_id
    )

    context = format_documents(documents)

    messages = rag_prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(messages)

    answer = StrOutputParser().invoke(response)

    sources = build_sources(documents)

    return answer, documents, sources