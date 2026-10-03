from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


completeness_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating citation completeness in a RAG system.

The generated answer was produced using the provided context.

Your task is to determine whether all important factual claims
in the generated answer have citations.

Context:

{context}

Generated Answer:

{answer}

Rules:

1. Identify the important factual claims in the answer.
2. Determine whether each important factual claim has a citation.
3. A citation looks like [1], [2], [3], etc.
4. Claims that contain factual information should normally have
   a citation.
5. Do not require citations for conversational phrases.
6. Do not use outside knowledge.
7. If every important factual claim has a citation, return:

COMPLETE

8. If one or more important factual claims do not have a citation,
   return:

INCOMPLETE

Return ONLY:

COMPLETE

or

INCOMPLETE
"""
)


def evaluate_citation_completeness(context, answer):

    messages = completeness_prompt.invoke(
        {
            "context": context,
            "answer": answer
        }
    )

    response = llm.invoke(messages)

    result = response.content.strip().upper()

    if "INCOMPLETE" in result:
        return "INCOMPLETE"

    if "COMPLETE" in result:
        return "COMPLETE"

    return "UNKNOWN"