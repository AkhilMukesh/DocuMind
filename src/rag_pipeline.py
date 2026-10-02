import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")


if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to the .env file."
    )


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Vector database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="company_policy",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# RAG prompt
# --------------------------------------------------

rag_prompt = ChatPromptTemplate.from_template(
    """
You are DocuMind AI, a document question-answering assistant.

Your task is to answer the user's question using ONLY the information
provided in the context.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- Do not invent or assume facts.
- If the answer is not available in the context, say:
  "The information is not available in the provided documents."
- Keep the answer concise and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# Helper: format retrieved documents
# --------------------------------------------------

def format_documents(documents):
    formatted_documents = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown"
        )

        formatted_documents.append(
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# --------------------------------------------------
# Main RAG function
# --------------------------------------------------

def ask_question(question):
    """
    Ask a question against the indexed documents.

    Returns:
        answer: Generated answer
        sources: List of source documents
    """

    documents = retriever.invoke(question)

    context = format_documents(documents)

    messages = rag_prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(messages)

    answer = StrOutputParser().invoke(response)

    sources = []

    for document in documents:
        source = document.metadata.get("source")

        if source and source not in sources:
            sources.append(source)

    return answer, sources