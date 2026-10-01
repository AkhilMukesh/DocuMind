# """
# Strategy	Main idea
# Similarity	Find the most similar chunks
# MMR	Find relevant but less redundant chunks
# Score threshold	Only return sufficiently relevant chunks

# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "score_threshold": 0.5,
#         "k": 3
#     }
# )
# """


from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_strore = Chroma(
    collection_name = "company_policy", 
    embedding_function = embeddings,
    persist_directory = str(CHROMA_PATH)
)

reteriver = vector_strore.as_retriever(
    search_type="mmr",  #Relevant results that are also diverse. That's where MMR — Maximal Marginal Relevance can help.
    search_kwargs={"k":2}
    )

query = "How many days can i work from home?"
results = reteriver.invoke(query)

print("QUESTION:")
print(query)

print("\nRETRIEVED DOCUMENTS:")

for index, document in enumerate(results):

    print(f"\n--- Result {index + 1} ---")

    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

