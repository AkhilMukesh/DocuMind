import streamlit as st
from pathlib import Path
import tempfile

from rag_pipeline import ask_question
from ingestion import ingest_document


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📄 DocuMind AI")

st.write(
    "Upload documents and ask questions using "
    "Retrieval-Augmented Generation."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Document Management")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"]
    )

    if uploaded_file is not None:

        if st.button("Process Document", type="primary"):

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    suffix = Path(
                        uploaded_file.name
                    ).suffix

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temporary_file:

                        temporary_file.write(
                            uploaded_file.getbuffer()
                        )

                        temporary_path = Path(
                            temporary_file.name
                        )

                    document_count, chunk_count = (
                        ingest_document(
                            temporary_path
                        )
                    )

                st.success(
                    "Document processed successfully!"
                )

                st.info(
                    f"Loaded documents: {document_count}\n\n"
                    f"Created chunks: {chunk_count}"
                )

            except Exception as error:

                st.error(
                    "Failed to process the document."
                )

                st.exception(error)


# --------------------------------------------------
# Question
# --------------------------------------------------

st.subheader("Ask DocuMind")


question = st.text_input(
    "Your question",
    placeholder=(
        "Example: How many days can employees "
        "work remotely?"
    )
)


# --------------------------------------------------
# Ask
# --------------------------------------------------

if st.button("Ask DocuMind"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching documents..."
            ):

                answer, sources = ask_question(
                    question
                )

            st.subheader("Answer")

            st.write(answer)

            st.subheader("Sources")

            if sources:

                for source in sources:

                    st.write(
                        f"📄 {source}"
                    )

            else:

                st.write(
                    "No sources found."
                )

        except Exception as error:

            st.error(
                "Something went wrong."
            )

            st.exception(error)