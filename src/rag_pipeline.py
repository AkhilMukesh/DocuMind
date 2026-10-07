import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


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
    """
    Retrieve relevant documents without calling the LLM.

    Used by:
    - Retrieval evaluation
    - Debugging
    - Testing
    - RAG pipeline
    """

    active_retriever = create_retriever(
        document_id=document_id
    )

    documents = active_retriever.invoke(question)

    return documents


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
    logger.info("RAG question processing started")

    documents = retrieve_documents(
        question,
        document_id=document_id
    )

    logger.info(
        "Document retrieval completed | result_count=%s",
        len(documents)
    )

    context = format_documents(documents)

    messages = rag_prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    logger.info("LLM request started")

    response = llm.invoke(messages)

    logger.info("LLM request completed")

    answer = StrOutputParser().invoke(response)

    sources = build_sources(documents)

    logger.info(
        "RAG question processing completed | source_count=%s",
        len(sources)
    )

    return answer, sources

    # --------------------------------------------------------
    # 1. Retrieve documents
    # --------------------------------------------------------

    documents = retrieve_documents(
        question,
        document_id=document_id
    )

    # --------------------------------------------------------
    # 2. Build context
    # --------------------------------------------------------

    context = format_documents(
        documents
    )

    # --------------------------------------------------------
    # 3. Build prompt
    # --------------------------------------------------------

    messages = rag_prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # --------------------------------------------------------
    # 4. Call LLM
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )

    # --------------------------------------------------------
    # 5. Parse answer
    # --------------------------------------------------------

    answer = StrOutputParser().invoke(
        response
    )

    # --------------------------------------------------------
    # 6. Build sources
    # --------------------------------------------------------

    sources = build_sources(
        documents
    )

    # --------------------------------------------------------
    # 7. Return
    # --------------------------------------------------------

    return answer, sources


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