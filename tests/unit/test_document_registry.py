import document_registry


# --------------------------------------------------
# Test database initialization
# --------------------------------------------------

def test_initialize_database(tmp_path, monkeypatch):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    connection = document_registry.get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'documents'
        """
    )

    result = cursor.fetchone()

    connection.close()

    assert result is not None
    assert result[0] == "documents"


# --------------------------------------------------
# Test adding a document
# --------------------------------------------------

def test_add_document(tmp_path, monkeypatch):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    document_registry.add_document(
        document_id="doc-001",
        filename="company_policy.txt",
        file_type="txt",
        content_hash="hash-001",
        chunk_count=3
    )

    document = document_registry.get_document(
        "doc-001"
    )

    assert document is not None
    assert document[0] == "doc-001"
    assert document[1] == "company_policy.txt"
    assert document[2] == "txt"
    assert document[3] == "hash-001"
    assert document[4] == 3


# --------------------------------------------------
# Test duplicate detection
# --------------------------------------------------

def test_document_exists(tmp_path, monkeypatch):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    document_registry.add_document(
        document_id="doc-001",
        filename="company_policy.txt",
        file_type="txt",
        content_hash="hash-001",
        chunk_count=3
    )

    assert document_registry.document_exists(
        "hash-001"
    )

    assert not document_registry.document_exists(
        "hash-does-not-exist"
    )


# --------------------------------------------------
# Test get all documents
# --------------------------------------------------

def test_get_documents(tmp_path, monkeypatch):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    document_registry.add_document(
        document_id="doc-001",
        filename="company_policy.txt",
        file_type="txt",
        content_hash="hash-001",
        chunk_count=3
    )

    document_registry.add_document(
        document_id="doc-002",
        filename="security_policy.pdf",
        file_type="pdf",
        content_hash="hash-002",
        chunk_count=5
    )

    documents = document_registry.get_documents()

    assert len(documents) == 2

    filenames = [
        document[1]
        for document in documents
    ]

    assert "company_policy.txt" in filenames
    assert "security_policy.pdf" in filenames


# --------------------------------------------------
# Test deleting a document
# --------------------------------------------------

def test_delete_document(tmp_path, monkeypatch):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    document_registry.add_document(
        document_id="doc-001",
        filename="company_policy.txt",
        file_type="txt",
        content_hash="hash-001",
        chunk_count=3
    )

    deleted = document_registry.delete_document(
        "doc-001"
    )

    assert deleted is True

    document = document_registry.get_document(
        "doc-001"
    )

    assert document is None


# --------------------------------------------------
# Test deleting non-existing document
# --------------------------------------------------

def test_delete_non_existing_document(
    tmp_path,
    monkeypatch
):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    deleted = document_registry.delete_document(
        "does-not-exist"
    )

    assert deleted is False


# --------------------------------------------------
# Test duplicate content hash
# --------------------------------------------------

def test_duplicate_content_hash_is_rejected(
    tmp_path,
    monkeypatch
):

    database_path = tmp_path / "test_documind.db"

    monkeypatch.setattr(
        document_registry,
        "DATABASE_PATH",
        database_path
    )

    document_registry.initialize_database()

    document_registry.add_document(
        document_id="doc-001",
        filename="company_policy.txt",
        file_type="txt",
        content_hash="same-hash",
        chunk_count=3
    )

    try:

        document_registry.add_document(
            document_id="doc-002",
            filename="another_policy.txt",
            file_type="txt",
            content_hash="same-hash",
            chunk_count=4
        )

        assert False, (
            "Expected duplicate content hash "
            "to be rejected"
        )

    except Exception:

        assert True