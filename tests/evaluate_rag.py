from pathlib import Path
from regression_baseline import save_baseline
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

from semantic_evaluator import (
    evaluate_answer
)

from faithfulness_evaluator import (
    evaluate_faithfulness
)

from citation_evaluator import (
    evaluate_citation_correctness
)

from citation_completeness_evaluator import (
    evaluate_citation_completeness
)


def evaluate():

    semantic_correct = 0
    faithful = 0
    citation_correct = 0
    citation_complete = 0

    total = len(evaluation_dataset)

    print()
    print("=" * 70)
    print("DOCUMIND AI — UNIFIED RAG EVALUATION")
    print("=" * 70)

    for index, item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = item["question"]
        expected_answer = item["expected_answer"]

        # --------------------------------------------------
        # 1. Retrieve + Generate
        # --------------------------------------------------

        answer, documents, sources = (
            answer_question_with_documents(question)
        )

        # Use the exact documents retrieved
        # for generating the answer.
        context = format_documents(documents)

        # --------------------------------------------------
        # 2. Semantic Correctness
        # --------------------------------------------------

        semantic_result = evaluate_answer(
            question,
            expected_answer,
            answer
        )

        if semantic_result == "CORRECT":
            semantic_correct += 1

        # --------------------------------------------------
        # 3. Faithfulness
        # --------------------------------------------------

        faithfulness_result = evaluate_faithfulness(
            question,
            context,
            answer
        )

        if faithfulness_result == "FAITHFUL":
            faithful += 1

        # --------------------------------------------------
        # 4. Citation Correctness
        # --------------------------------------------------

        citation_result = evaluate_citation_correctness(
            context,
            answer
        )

        if citation_result == "CORRECT":
            citation_correct += 1

        # --------------------------------------------------
        # 5. Citation Completeness
        # --------------------------------------------------

        completeness_result = (
            evaluate_citation_completeness(
                context,
                answer
            )
        )

        if completeness_result == "COMPLETE":
            citation_complete += 1

        # --------------------------------------------------
        # Print Question Results
        # --------------------------------------------------

        print()
        print(f"Question {index}")
        print("-" * 70)

        print("Question:")
        print(question)

        print()
        print("Generated Answer:")
        print(answer)

        print()
        print("Evaluation:")
        print(
            f"Semantic Correctness : {semantic_result}"
        )
        print(
            f"Faithfulness         : {faithfulness_result}"
        )
        print(
            f"Citation Correctness : {citation_result}"
        )
        print(
            f"Citation Completeness: {completeness_result}"
        )

        # ------------------------------------------------------
    # Calculate overall metrics
    # ------------------------------------------------------

    results = {
        "semantic_correctness": (
            semantic_correct / total
        ),
        "faithfulness": (
            faithful / total
        ),
        "citation_correctness": (
            citation_correct / total
        ),
        "citation_completeness": (
            citation_complete / total
        )
    }

    # ------------------------------------------------------
    # Print Overall Results
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL RAG EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Semantic Correctness : "
        f"{results['semantic_correctness']:.2%}"
    )

    print(
        f"Faithfulness         : "
        f"{results['faithfulness']:.2%}"
    )

    print(
        f"Citation Correctness : "
        f"{results['citation_correctness']:.2%}"
    )

    print(
        f"Citation Completeness: "
        f"{results['citation_completeness']:.2%}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Save regression baseline
    # ------------------------------------------------------

    save_baseline(results)


if __name__ == "__main__":
    evaluate()