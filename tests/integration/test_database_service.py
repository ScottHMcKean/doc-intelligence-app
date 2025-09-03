"""Integration tests for the DatabaseService."""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import psycopg2
import psycopg2.extras


def create_mock_cursor():
    """Create a properly configured mock cursor with context manager support."""
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=None)
    return mock_cursor


import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.database.service import DatabaseService


class TestDatabaseService:
    """Integration tests for the database service."""

    def test_database_service_initialization(self, test_config, mock_databricks_client):
        """Test database service initialization."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        assert db_service.client is not None
        assert db_service.config is not None

    def test_connection_context_manager(self, test_config, mock_databricks_client):
        """Test database connection context manager."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Test connection context manager
        with db_service.get_connection() as conn:
            # Connection should be available (mocked)
            assert conn is not None

    def test_connection_without_client(self, test_config):
        """Test connection handling when client is not available."""
        db_service = DatabaseService(client=None, config=test_config)

        with db_service.get_connection() as conn:
            # Should return None when no client
            assert conn is None

    def test_connection_test(self, test_config, mock_databricks_client):
        """Test database connection testing."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            success, message = db_service.test_connection()

            assert success is True
            assert "successful" in message.lower()

    def test_connection_test_failure(self, test_config, mock_databricks_client):
        """Test database connection testing failure."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = None

            success, message = db_service.test_connection()

            assert success is False
            assert "not available" in message.lower()

    def test_table_creation(self, test_config, mock_databricks_client):
        """Test database table creation."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            result = db_service.create_tables()

            # Should succeed with mocked connection
            assert result is True

    def test_user_operations(self, test_config, mock_databricks_client):
        """Test user creation and management operations."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None  # User doesn't exist
        mock_cursor.fetchone.side_effect = [
            None,
            {"id": 12345, "username": "test@databricks.com"},
        ]  # First call returns None, second returns user
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test user creation
            user = db_service.create_user("test@databricks.com", 12345)

            assert user is not None
            assert user["id"] == 12345
            assert user["username"] == "test@databricks.com"

    def test_user_exists_check(self, test_config, mock_databricks_client):
        """Test user existence checking."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)  # User exists
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test user existence
            exists = db_service.user_exists("test@databricks.com")

            assert exists is True

    def test_user_authentication_verification(
        self, test_config, mock_databricks_client
    ):
        """Test user authentication verification."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)  # User authenticated
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test authentication verification
            is_authenticated = db_service.verify_user_authentication(
                "test@databricks.com", 12345
            )

            assert is_authenticated is True

    def test_document_operations(self, test_config, mock_databricks_client):
        """Test document creation and management operations."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_document = {
            "id": str(uuid.uuid4()),
            "doc_hash": "test_hash",
            "filename": "test.txt",
            "status": "uploaded",
            "user_id": 12345,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        mock_cursor.fetchone.return_value = mock_document
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test document creation
            document = db_service.create_document(
                user_id=12345, doc_hash="test_hash", filename="test.txt"
            )

            assert document is not None
            assert document["doc_hash"] == "test_hash"
            assert document["filename"] == "test.txt"

    def test_document_status_update(self, test_config, mock_databricks_client):
        """Test document status updates."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 1  # One row updated
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test status update
            success = db_service.update_document_status(
                document_id=str(uuid.uuid4()), status="processed"
            )

            assert success is True

    def test_document_retrieval_by_hash(self, test_config, mock_databricks_client):
        """Test document retrieval by hash."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_document = {
            "id": str(uuid.uuid4()),
            "doc_hash": "test_hash",
            "filename": "test.txt",
            "status": "processed",
        }
        mock_cursor.fetchone.return_value = mock_document
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test document retrieval
            document = db_service.get_document_by_hash("test_hash")

            assert document is not None
            assert document["doc_hash"] == "test_hash"

    def test_user_documents_retrieval(self, test_config, mock_databricks_client):
        """Test retrieving documents for a user."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_documents = [
            {"id": str(uuid.uuid4()), "filename": "doc1.txt", "status": "processed"},
            {"id": str(uuid.uuid4()), "filename": "doc2.txt", "status": "uploaded"},
        ]
        mock_cursor.fetchall.return_value = mock_documents
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test user documents retrieval
            documents = db_service.get_user_documents(12345)

            assert len(documents) == 2
            assert documents[0]["filename"] == "doc1.txt"

    def test_conversation_operations(self, test_config, mock_databricks_client):
        """Test conversation creation and management."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conversation = {
            "id": str(uuid.uuid4()),
            "user_id": 12345,
            "title": "Test Conversation",
            "thread_id": str(uuid.uuid4()),
            "status": "active",
        }
        mock_cursor.fetchone.return_value = mock_conversation
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test conversation creation
            conversation = db_service.create_conversation(
                user_id=12345, title="Test Conversation", thread_id=str(uuid.uuid4())
            )

            assert conversation is not None
            assert conversation["title"] == "Test Conversation"

    def test_message_operations(self, test_config, mock_databricks_client):
        """Test message creation and retrieval."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_message = {
            "id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "role": "user",
            "content": "Hello, world!",
            "created_at": datetime.now(timezone.utc),
        }
        mock_cursor.fetchone.return_value = mock_message
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test message creation
            message = db_service.add_message(
                conversation_id=str(uuid.uuid4()), role="user", content="Hello, world!"
            )

            assert message is not None
            assert message["role"] == "user"
            assert message["content"] == "Hello, world!"

    def test_document_chunks_operations(
        self, test_config, mock_databricks_client, test_document_chunks
    ):
        """Test document chunk storage and retrieval."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_chunk = {
            "id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "chunk_index": 0,
            "content": "Test chunk content",
        }
        mock_cursor.fetchone.return_value = mock_chunk
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test chunk storage
            chunks = db_service.store_document_chunks(
                document_id=str(uuid.uuid4()), chunks=test_document_chunks
            )

            assert len(chunks) == len(test_document_chunks)
            assert chunks[0]["content"] == "Test chunk content"

    def test_vector_search(
        self, test_config, mock_databricks_client, test_vector_embedding
    ):
        """Test vector similarity search."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_result = {
            "id": str(uuid.uuid4()),
            "content": "Relevant content",
            "distance": 0.1,
            "filename": "test.txt",
        }
        mock_cursor.fetchall.return_value = [mock_result]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test vector search
            results = db_service.vector_search(
                query_embedding=test_vector_embedding, limit=5
            )

            assert len(results) == 1
            assert results[0]["content"] == "Relevant content"

    def test_conversation_document_association(
        self, test_config, mock_databricks_client
    ):
        """Test adding documents to conversations."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (["doc1", "doc2"],)  # Current document IDs
        mock_cursor.rowcount = 1  # One row updated
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test adding documents to conversation
            success = db_service.add_documents_to_conversation(
                conversation_id=str(uuid.uuid4()), document_hashes=["doc3", "doc4"]
            )

            assert success is True

    def test_ai_parsing_completion_check(self, test_config, mock_databricks_client):
        """Test AI parsing completion checking."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_chunks = [
            {"id": str(uuid.uuid4()), "content": "Chunk 1", "chunk_index": 0},
            {"id": str(uuid.uuid4()), "content": "Chunk 2", "chunk_index": 1},
        ]
        mock_document = {"id": str(uuid.uuid4()), "filename": "test.txt"}
        mock_cursor.fetchall.return_value = mock_chunks
        mock_cursor.fetchone.side_effect = [mock_document]  # For document query

        with patch.object(db_service, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn

            # Test AI parsing completion check
            result = db_service.check_ai_parsing_completion(
                file_path="/path/to/test.txt", user_id=12345
            )

            assert result is not None
            assert result["completed"] is True
            assert result["total_chunks"] == 2

    def test_error_handling_without_client(self, test_config):
        """Test error handling when client is not available."""
        db_service = DatabaseService(client=None, config=test_config)

        # All operations should return None or empty results
        assert db_service.create_user("test", 123) is None
        assert db_service.user_exists("test") is False
        assert db_service.create_document(123, "hash", "file.txt") is None
        assert db_service.get_user_documents(123) == []
        assert db_service.create_conversation(123, "title", "thread") is None
        assert db_service.add_message("conv", "user", "content") is None

    def test_database_connection_string_generation(
        self, test_config, mock_databricks_client
    ):
        """Test database connection string generation."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # The connection string generation is handled internally
        # We can test that the service can be initialized with the config
        assert db_service.config.get("database.instance_name") == "test-instance"
        assert db_service.config.get("database.database") == "test_db"
        assert db_service.config.get("database.user") == "test_user"
