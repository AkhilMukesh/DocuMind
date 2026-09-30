from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

policy_path = Path(__file__).resolve().parents[1] / "data" / "company_policy.txt"

loader = TextLoader(str(policy_path))

#create a document object using langchain
documents = loader.load()

#used for chuncking 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 40
)

chunks = text_splitter.split_documents(documents)

print(f"Number of Chucks: {chunks}")

for i,chunk in enumerate(chunks):
    print(f"-----chunk------{i+1}-----------")
    print(chunk.page_content)
    print(f"metadata")
    print(chunk.metadata)

