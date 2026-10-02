import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------
# Check API key
# --------------------------------------------------

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing."
    )


# --------------------------------------------------
# Evaluation LLM
# --------------------------------------------------

evaluator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

evaluation_prompt = """
You are evaluating an AI-generated answer.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Determine whether the generated answer is semantically
equivalent to the expected answer.

The wording does not need to be identical.

Return ONLY one of:

PASS
FAIL
"""

def evaluate_answer(question,expected_answer,generated_answer):
    prompt = evaluation_prompt.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )
    response = evaluator_llm.invoke(prompt)
    result = response.content.strip().upper()

    return result


if __name__ == "__main__":

    result = evaluate_answer(
        question="How many days can employees work remotely?",
        expected_answer=(
            "Employees can work remotely up to three days per week."
        ),
        generated_answer=(
            "Employees are allowed to work remotely for "
            "a maximum of three days each week."
        )
    )

    print("Evaluation:", result)