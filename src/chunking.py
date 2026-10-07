from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Text splitter configuration
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40
)


# --------------------------------------------------
# Split documents
# --------------------------------------------------

def split_documents(documents):
    """
    Split LangChain documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked LangChain Document objects.
    """

    return text_splitter.split_documents(documents)


# --------------------------------------------------
# Local development test
# --------------------------------------------------

if __name__ == "__main__":

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "company_policy.txt"
    )

    loader = TextLoader(str(policy_path))

    documents = loader.load()

    chunks = split_documents(documents)

    print(f"Number of Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):

        print(f"----- Chunk {i + 1} -----")
        print(chunk.page_content)
        print("Metadata:")
        print(chunk.metadata)