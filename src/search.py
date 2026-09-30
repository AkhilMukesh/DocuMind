from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

policy_path = Path(__file__).resolve().parents[1] / "data" / "company_policy.txt"

CHROMA_PATH = Path(__file__).resolve().parents[1] / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#loading chroma DB
vector_store = Chroma(
    collection_name = "company_policy",
    embedding_function = embeddings,
    persist_directory = str(CHROMA_PATH)
)

query = "How many days can i work from home?"

results = vector_store.similarity_search(
    query,
    k=2 #top k matching results
)

print("Question")
print(query)

print(f"Retrived Documents")

for index,result in enumerate(results):
    print(f"---result--{index+1}---")
    print("Contents")
    print(result.page_content)

    print("Metadata")
    print(result.metadata)