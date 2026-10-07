import sqlite3
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "documind.db"


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# --------------------------------------------------
# Create table
# --------------------------------------------------

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            chunk_count INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


# --------------------------------------------------
# Add document
# --------------------------------------------------

def add_document(
    document_id,
    filename,
    file_type,
    content_hash,
    chunk_count
):
    logger.info(
        "Registering document | document_id=%s | filename=%s | chunks=%s",
        document_id,
        filename,
        chunk_count
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO documents (
            id, filename, file_type, content_hash, chunk_count, uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            filename,
            file_type,
            content_hash,
            chunk_count,
            datetime.utcnow().isoformat()
        )
    )

    connection.commit()
    connection.close()

    logger.info(
        "Document registered successfully | document_id=%s",
        document_id
    )


# --------------------------------------------------
# Check duplicate
# --------------------------------------------------

def document_exists(content_hash):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM documents
        WHERE content_hash = ?
        """,
        (content_hash,)
    )

    result = cursor.fetchone()

    connection.close()

    return result is not None


# --------------------------------------------------
# Get all documents
# --------------------------------------------------

def get_documents():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            file_type,
            chunk_count,
            uploaded_at
        FROM documents
        ORDER BY uploaded_at DESC
        """
    )

    documents = cursor.fetchall()

    connection.close()

    return documents

def delete_document(document_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM documents
        WHERE id = ?
        """,
        (document_id,)
    )

    deleted = cursor.rowcount > 0

    connection.commit()
    connection.close()

    return deleted


def get_document(document_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            file_type,
            content_hash,
            chunk_count,
            uploaded_at
        FROM documents
        WHERE id = ?
        """,
        (document_id,)
    )

    document = cursor.fetchone()

    connection.close()

    return document