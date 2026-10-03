from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


citation_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating citation correctness in a RAG system.

The generated answer contains citations such as [1], [2].

Your task is to determine whether the cited context actually
supports the factual claim associated with the citation.

Context:

{context}

Generated Answer:

{answer}

Rules:

1. Examine every citation in the answer.
2. Find the corresponding context item.
3. Determine whether that context supports the claim.
4. Do not use outside knowledge.
5. A citation is CORRECT if the cited context directly supports
   the claim.
6. A citation is INCORRECT if the cited context does not support
   the claim.

Return ONLY:

CORRECT

if all citations are supported.

Otherwise return:

INCORRECT
"""
)


def evaluate_citation_correctness(context, answer):

    messages = citation_prompt.invoke(
        {
            "context": context,
            "answer": answer
        }
    )

    response = llm.invoke(messages)

    result = response.content.strip().upper()

    if "INCORRECT" in result:
        return "INCORRECT"

    if "CORRECT" in result:
        return "CORRECT"

    return "UNKNOWN"