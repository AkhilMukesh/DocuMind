from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


policy_path = Path(__file__).resolve().parents[1] / "data" / "company_policy.txt"

CHROMA_PATH = Path(__file__).resolve().parents[1] / "chroma_db"


#1. load documents
loader = TextLoader(
    str(policy_path),
    encoding="utf-8"
)

documents = loader.load()

print(f"Doucment loaded: {len(documents)}")

#2. create chuknks for the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 40
)

chunks = text_splitter.split_documents(documents)

print(f"Chuck Created: {len(chunks)}")

#3. create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#4. create persisent vecotr chroma db
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name = "company_policy",
    persist_directory = str(CHROMA_PATH)

)
print(f"Vecotr Database create sucessfully")
print("database location")
print(CHROMA_PATH)