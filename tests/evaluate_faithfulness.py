from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import (
    answer_question_with_documents,
    format_documents
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_dataset import evaluation_dataset

from faithfulness_evaluator import (
    evaluate_faithfulness as check_faithfulness
)


def evaluate():

    faithful_count = 0

    print()
    print("=" * 60)
    print("RAG FAITHFULNESS EVALUATION")
    print("=" * 60)

    for index, item in enumerate(evaluation_dataset, start=1):

        question = item["question"]

        # Retrieve documents and generate answer
        # using the same retrieved documents.
        answer, documents, sources = answer_question_with_documents(
            question
        )

        # Build the exact context used by the RAG pipeline.
        context = format_documents(documents)

        # Evaluate faithfulness.
        result = check_faithfulness(
            question,
            context,
            answer
        )

        if result == "FAITHFUL":
            faithful_count += 1

        print()
        print(f"Question {index}")
        print("-" * 60)

        print("Question:")
        print(question)

        print()
        print("Generated Answer:")
        print(answer)

        print()
        print("Faithfulness:")
        print(result)

    total = len(evaluation_dataset)

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Faithfulness: "
        f"{faithful_count / total:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()