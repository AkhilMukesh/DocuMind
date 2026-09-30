from pathlib import Path

from langchain_community.document_loaders import TextLoader

policy_path = Path(__file__).resolve().parents[1] / "data" / "company_policy.txt"
loader = TextLoader(str(policy_path))

documents = loader.load()

for document in documents:
    print("Content:")
    print(document.page_content)

    print("Metadata:")
    print(document.metadata)