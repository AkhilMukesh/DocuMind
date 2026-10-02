import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from evaluation_dataset import evaluation_dataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")


if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing."
    )


# --------------------------------------------------
# Embeddings
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


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the provided context.

If the answer is not available in the context, say:
"The information is not available in the provided documents."

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
# Document formatting
# --------------------------------------------------

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


parser = StrOutputParser()


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

correct = 0


for item in evaluation_dataset:

    question = item["question"]
    expected_answer = item["expected_answer"]

    documents = retriever.invoke(question)

    context = format_documents(documents)

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = llm.invoke(messages)

    answer = parser.invoke(response)

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    print("\nEXPECTED:")
    print(expected_answer)

    print("\nGENERATED:")
    print(answer)

    # Simple baseline comparison.
    # This is intentionally strict and will not capture
    # semantically equivalent wording.
    if answer.strip().lower() == expected_answer.strip().lower():
        correct += 1


accuracy = correct / len(evaluation_dataset)

print("\n" + "=" * 60)
print("GENERATION EVALUATION")
print("=" * 60)

print(f"Exact-match accuracy: {accuracy:.2%}")