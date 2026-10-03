from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import answer_question_with_documents
from evaluation_dataset import evaluation_dataset
from semantic_evaluator import evaluate_answer


def evaluate():
    correct = 0

    print()
    print("=" * 60)
    print("RAG SEMANTIC GENERATION EVALUATION")
    print("=" * 60)

    for index, item in enumerate(evaluation_dataset, start=1):

        question = item["question"]
        expected_answer = item["expected_answer"]

        generated_answer, documents, sources = answer_question_with_documents(question)

        result = evaluate_answer(
            question,
            expected_answer,
            generated_answer
        )

        if result == "CORRECT":
            correct += 1

        print()
        print(f"Question {index}")
        print("-" * 60)

        print(f"Question: {question}")

        print()
        print(f"Expected:")
        print(expected_answer)

        print()
        print(f"Generated:")
        print(generated_answer)

        print()
        print(f"Semantic Result: {result}")

    total = len(evaluation_dataset)

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Semantic Correctness: "
        f"{correct / total:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()