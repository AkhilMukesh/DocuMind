from pathlib import Path
import sys

# --------------------------------------------------
# Allow imports from src/
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import retrieve_documents
from evaluation_dataset import evaluation_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

K = 2


# --------------------------------------------------
# Check whether a document is relevant
# --------------------------------------------------

def is_relevant(document, expected_sources):

    source = document.metadata.get(
        "filename",
        document.metadata.get(
            "source",
            ""
        )
    )

    actual_filename = Path(source).name

    return actual_filename in expected_sources

# --------------------------------------------------
# Hit@K
# --------------------------------------------------

def calculate_hit_at_k(documents, expected_sources):
    for document in documents:
        if is_relevant(document, expected_sources):
            return 1

    return 0


# --------------------------------------------------
# Precision@K
# --------------------------------------------------

def calculate_precision_at_k(documents, expected_sources):
    if not documents:
        return 0.0

    relevant_count = sum(
        1
        for document in documents
        if is_relevant(document, expected_sources)
    )

    return relevant_count / len(documents)


# --------------------------------------------------
# Recall@K
# --------------------------------------------------

def calculate_recall_at_k(documents, expected_sources):

    if not expected_sources:
        return 0.0

    retrieved_sources = set()

    for document in documents:

        source = document.metadata.get(
            "filename",
            document.metadata.get(
                "source",
                ""
            )
        )

        actual_filename = Path(source).name

        if actual_filename in expected_sources:
            retrieved_sources.add(actual_filename)

    return len(retrieved_sources) / len(expected_sources)


# --------------------------------------------------
# Reciprocal Rank
# --------------------------------------------------

def calculate_reciprocal_rank(documents, expected_sources):
    for rank, document in enumerate(documents, start=1):

        if is_relevant(document, expected_sources):
            return 1 / rank

    return 0.0


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

def evaluate():

    total_hit = 0
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    print()
    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION v2.0")
    print("=" * 60)

    for index, item in enumerate(evaluation_dataset, start=1):

        question = item["question"]
        expected_sources = item["expected_sources"]

        documents = retrieve_documents(question)

        documents = documents[:K]

        print("\nRETRIEVED DOCUMENTS")

        for rank, document in enumerate(documents, start=1):
            print(f"\nRank {rank}")
            print("Metadata:", document.metadata)
            print("Content:", document.page_content[:200])

        hit = calculate_hit_at_k(
            documents,
            expected_sources
        )

        precision = calculate_precision_at_k(
            documents,
            expected_sources
        )

        recall = calculate_recall_at_k(
            documents,
            expected_sources
        )

        reciprocal_rank = calculate_reciprocal_rank(
            documents,
            expected_sources
        )

        total_hit += hit
        total_precision += precision
        total_recall += recall
        total_mrr += reciprocal_rank

        print()
        print(f"Question {index}")
        print("-" * 60)
        print(question)

        print(f"Hit@{K}:       {hit}")
        print(f"Precision@{K}: {precision:.2f}")
        print(f"Recall@{K}:    {recall:.2f}")
        print(f"MRR:           {reciprocal_rank:.2f}")

    count = len(evaluation_dataset)

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"Hit@{K}:       {total_hit / count:.2%}")
    print(f"Precision@{K}: {total_precision / count:.2%}")
    print(f"Recall@{K}:    {total_recall / count:.2%}")
    print(f"MRR:           {total_mrr / count:.2f}")

    print("=" * 60)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    evaluate()