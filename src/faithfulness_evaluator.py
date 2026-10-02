import os 

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing."
    )


evaluator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


faithfulness_prompt = """
You are evaluating whether an AI-generated answer
is supported by the provided context.

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Determine whether the generated answer contains
claims that are supported by the context.

Rules:

1. Every factual claim must be supported by the context.
2. Do not use outside knowledge.
3. If the answer contains an unsupported factual claim,
   return FAIL.
4. If all factual claims are supported by the context,
   return PASS.

Return ONLY:

PASS
or

FAIL
"""

def evaluate_faithfulness(
    context,
    question,
    answer
):
    prompt = faithfulness_prompt.format(
        context=context,
        question=question,
        answer=answer,
    )

    response = evaluator_llm.invoke(prompt)

    result = response.content.strip().upper()

    return result


if __name__ == "__main__":

    context = """
    Employees can work remotely up to three days per week.
    """

    question = "How many days can employees work remotely?"

    answer = """
    Employees can work remotely up to three days per week and receive an additional remote-work allowance.
    """

    result = evaluate_faithfulness(
        context,
        question,
        answer
    )

    print("Faithfulness:", result)
