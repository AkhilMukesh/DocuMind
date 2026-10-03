from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import ask_question
from evaluation_dataset import evaluation_dataset


def normalize(text):
    return " ".join(text.lower().split())


def exact_match(actual, expected):
    return normalize(actual) == normalize(expected)


def keyword_score(actual, expected_keywords):
    actual_text = normalize(actual)

    if not expected_keywords:
        return 0.0

    matched = 0

    for keyword in expected_keywords:
        if normalize(keyword) in actual_text:
            matched += 1

    return matched / len(expected_keywords)


def evaluate():
    total_exact_match = 0.0
    total_keyword_score = 0.0

    print()
    print("=" * 60)
    print("RAG GENERATION EVALUATION v2.0")
    print("=" * 60)

    for index, item in enumerate(evaluation_dataset, start=1):

        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_keywords = item["expected_keywords"]

        answer, sources = ask_question(question)

        exact = exact_match(
            answer,
            expected_answer
        )

        keyword = keyword_score(
            answer,
            expected_keywords
        )

        total_exact_match += int(exact)
        total_keyword_score += keyword

        print()
        print(f"Question {index}")
        print("-" * 60)

        print(f"Question:")
        print(question)

        print()
        print("Expected Answer:")
        print(expected_answer)

        print()
        print("Actual Answer:")
        print(answer)

        print()
        print(f"Exact Match:     {int(exact)}")
        print(f"Keyword Score:   {keyword:.2f}")

    count = len(evaluation_dataset)

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Exact Match:     "
        f"{total_exact_match / count:.2%}"
    )

    print(
        f"Keyword Score:   "
        f"{total_keyword_score / count:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()