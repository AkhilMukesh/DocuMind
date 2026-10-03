from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from rag_pipeline import (
    answer_question_with_documents,
    format_documents
)

from evaluation_dataset import evaluation_dataset

from citation_completeness_evaluator import (
    evaluate_citation_completeness
)


def evaluate():

    complete_count = 0

    print()
    print("=" * 60)
    print("RAG CITATION COMPLETENESS EVALUATION")
    print("=" * 60)

    for index, item in enumerate(evaluation_dataset, start=1):

        question = item["question"]

        # Retrieve documents and generate the answer
        # using the same retrieved documents.
        answer, documents, sources = (
            answer_question_with_documents(question)
        )

        # Build the exact context used by the RAG pipeline.
        context = format_documents(documents)

        # Evaluate citation completeness.
        result = evaluate_citation_completeness(
            context,
            answer
        )

        if result == "COMPLETE":
            complete_count += 1

        print()
        print(f"Question {index}")
        print("-" * 60)

        print("Question:")
        print(question)

        print()
        print("Generated Answer:")
        print(answer)

        print()
        print("Citation Completeness:")
        print(result)

    total = len(evaluation_dataset)

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Citation Completeness: "
        f"{complete_count / total:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()