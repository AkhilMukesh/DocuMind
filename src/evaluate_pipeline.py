from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from evaluation_dataset import evaluation_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"    
    )

vector_store = Chroma(
    collection_name="company_policy",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)

k=2

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": k}
)

total_hits = 0


for item in evaluation_dataset:
    question = item["question"]
    expected_keywords = item["expected_keywords"]
    documents =     retriever.invoke(question)

    retrieved_text = " ".join(
            document.page_content.lower()
            for document in documents
        )
    
    hit = any(
            keyword.lower() in retrieved_text
            for keyword in expected_keywords
    )

    if hit:
        total_hits += 1

    print("\nQuestion:")
    print(question)

    print("Hit@2:", hit)

    score = total_hits / len(evaluation_dataset)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Hit@{k}: {score:.2%}")

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    print("\nRETRIEVED DOCUMENTS:")
    for index, document in enumerate(documents, start=1):
        print(f"\n--- Document {index} ---")
        print(document.page_content)


    