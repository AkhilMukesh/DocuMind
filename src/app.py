import streamlit as st
from pathlib import Path
import tempfile

from rag_pipeline import ask_question
from ingestion import (
    ingest_document,
    delete_document_completely
)
from document_registry import get_documents

from document_registry import (
    initialize_database,
    get_documents
)

initialize_database()

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

                    result = ingest_document(
                        temporary_path,
                        uploaded_file.name
                    )

                if result["status"] == "duplicate":

                    st.warning(
                        "This document has already been uploaded."
                    )

                else:
                    st.success(
                        "Document processed successfully!"
                    )

                    st.write(
                        f"Chunks created: "
                        f"{result['chunk_count']}"
                    )


            except Exception as error:

                st.error(
                    "Failed to process the document."
                )

                st.exception(error)

    st.divider()
    st.subheader("Indexed Documents")

documents = get_documents()

document_options = {
    document[1]: document[0]
    for document in documents
}
selected_document = st.selectbox(
    "Search within",
    options=[
        "All documents"
    ] + list(document_options.keys())
)

selected_document_id = None

if selected_document != "All documents":

    selected_document_id = document_options[
        selected_document
    ]


if not documents:

    st.info(
        "No documents have been indexed yet."
    )

else:

    for document in documents:

        document_id = document[0]
        filename = document[1]
        file_type = document[2]
        chunk_count = document[3]
        uploaded_at = document[4]

        with st.expander(filename):

            st.write(
                f"Type: {file_type}"
            )

            st.write(
                f"Chunks: {chunk_count}"
            )

            st.write(
                f"Uploaded: {uploaded_at}"
            )

            st.caption(
                f"Document ID: {document_id}"
            )

            if st.button(
                "Delete Document",
                key=f"delete_{document_id}"
            ):

                try:

                    with st.spinner(
                        "Deleting document..."
                    ):

                        result = (
                            delete_document_completely(
                                document_id
                            )
                        )

                    if result["status"] == "success":

                        st.success(
                            f"{filename} was deleted."
                        )

                        st.rerun()

                    elif result["status"] == "not_found":

                        st.warning(
                            "Document was not found."
                        )

                    else:

                        st.error(
                            "Failed to delete document."
                        )

                except Exception as error:

                    st.error(
                        "An error occurred while deleting "
                        "the document."
                    )

                    st.exception(error)

# --------------------------------------------------
# Question
# --------------------------------------------------

st.subheader("Ask DocuMind")


question = st.text_input(
    "Your question",
    placeholder=(
        "Example: Ask....."
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
                    question,
                    document_id=selected_document_id
                )

            st.subheader("Answer")

            st.write(answer)

            st.subheader("Sources")

            if sources:

                for source in sources:

                    st.markdown(
                        f"**{source['citation']} "
                        f"{source['location']}**"
        )

                    with st.expander(
                     "View retrieved evidence"
        ):

                        st.write(
                        source["content"]
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