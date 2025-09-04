"""Integration tests for the DatabaseService."""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import psycopg2
import psycopg2.extras

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.database.service import DatabaseService
from doc_intel.database.models import (
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
)


def create_mock_cursor():
    """Create a properly configured mock cursor with context manager support."""
    mock_cursor = Mock()
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=None)
    return mock_cursor


class TestDatabaseService:
    """Integration tests for the database service."""

    def test_database_service_initialization(self, test_config, mock_databricks_client):
        """Test database service initialization."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        assert db_service.client is not None
        assert db_service.config is not None

    def test_connection_live_property(self, test_config, mock_databricks_client):
        """Test database connection live property."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            is_live = db_service.connection_live

            assert is_live is True

    def test_connection_live_failure(self, test_config, mock_databricks_client):
        """Test database connection live property failure."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock connection to raise exception
        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            is_live = db_service.connection_live

            assert is_live is False

    def test_connect_to_pg(self, test_config, mock_databricks_client):
        """Test direct PostgreSQL connection."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock psycopg2.connect
        mock_conn = Mock()
        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value = mock_conn

            result = db_service.connect_to_pg()

            assert result == mock_conn
            mock_connect.assert_called_once()

    def test_run_pg_query(self, test_config, mock_databricks_client):
        """Test running PostgreSQL queries."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [(1,)]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            result = db_service.run_pg_query("SELECT 1")

            assert result == [(1,)]
            mock_cursor.execute.assert_called_once_with("SELECT 1")
            mock_conn.commit.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_user_operations(self, test_config, mock_databricks_client):
        """Test user creation and management operations."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,  # First call (check by ID) returns None
            None,  # Second call (check by username) returns None
            {
                "id": "12345",
                "username": "test@databricks.com",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },  # Third call (INSERT RETURNING)
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            # Test user creation (no parameters needed - uses cached user)
            user = db_service.create_user()

            assert user is not None
            assert isinstance(user, User)
            assert user.id == "12345"
            assert user.username == "test@databricks.com"

    def test_user_exists_check(self, test_config, mock_databricks_client):
        """Test user existence checking."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = (1,)  # User exists
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            exists = db_service.user_exists()

            assert exists is True

    def test_document_operations(self, test_config, mock_databricks_client):
        """Test document creation and management operations."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": str(uuid.uuid4()),
            "user_id": "12345",
            "raw_path": "/test/path",
            "processed_path": "/processed/path",
            "metadata": {"test": "data"},
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            # Test document creation
            document = db_service.create_document(
                raw_path="/test/path",
                processed_path="/processed/path",
                metadata={"test": "data"},
            )

            assert document is not None
            assert isinstance(document, Document)
            assert document.id is not None

    def test_document_retrieval_by_id(self, test_config, mock_databricks_client):
        """Test document retrieval by ID."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": str(uuid.uuid4()),
            "user_id": "12345",
            "raw_path": "/test/path",
            "processed_path": "/processed/path",
            "metadata": {"test": "data"},
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            document = db_service.get_document_by_id("test_id")

            assert document is not None
            assert document["id"] is not None

    def test_user_documents_retrieval(self, test_config, mock_databricks_client):
        """Test retrieving documents for a user."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": str(uuid.uuid4()),
                "user_id": "12345",
                "raw_path": "/test/path1",
                "processed_path": "/processed/path1",
                "metadata": {"test": "data1"},
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": str(uuid.uuid4()),
                "user_id": "12345",
                "raw_path": "/test/path2",
                "processed_path": "/processed/path2",
                "metadata": {"test": "data2"},
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            documents = db_service.get_user_documents()

            assert len(documents) == 2
            assert all(isinstance(doc, Document) for doc in documents)

    def test_conversation_operations(self, test_config, mock_databricks_client):
        """Test conversation creation and management."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": "test-session-123",
            "user_id": "12345",
            "doc_ids": ["doc1", "doc2"],
            "metadata": {"title": "Test Conversation", "test": "data"},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            # Test conversation creation
            conversation = db_service.create_conversation(
                conversation_id="test-session-123",
                doc_ids=["doc1", "doc2"],
                metadata={"title": "Test Conversation", "test": "data"},
            )

            assert conversation is not None
            assert isinstance(conversation, Conversation)
            assert conversation.id == "test-session-123"

    def test_message_operations(self, test_config, mock_databricks_client):
        """Test message creation and retrieval."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": str(uuid.uuid4()),
            "conv_id": str(uuid.uuid4()),
            "role": "user",
            "content": {"type": "text", "content": "Hello"},
            "metadata": {"test": "data"},
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            # Test message creation
            message = db_service.add_message(
                conv_id=str(uuid.uuid4()),
                role="user",
                content={"type": "text", "content": "Hello"},
                metadata={"test": "data"},
            )

            assert message is not None
            assert isinstance(message, Message)
            assert message.role == "user"

    def test_chunks_operations(self, test_config, mock_databricks_client, test_chunks):
        """Test chunk storage and retrieval."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": str(uuid.uuid4()),
                "doc_id": str(uuid.uuid4()),
                "content": "Test chunk content",
                "embedding": [0.1] * 768,
                "metadata": {"chunk_index": 0},
                "created_at": datetime.now(timezone.utc),
            }
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            # Test chunk storage
            result = db_service.store_document_chunks(
                document_id=str(uuid.uuid4()),
                chunks=test_chunks,
            )

            assert result is True

            # Test chunk retrieval
            chunks = db_service.get_document_chunks(str(uuid.uuid4()))

            assert len(chunks) == 1
            assert chunks[0]["content"] == "Test chunk content"

    def test_error_handling_without_client(self, test_config):
        """Test error handling when client is not available."""
        with pytest.raises(Exception):
            DatabaseService(client=None, config=test_config)

    def test_database_connection_string_generation(
        self, test_config, mock_databricks_client
    ):
        """Test database connection string generation."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Test that connection parameters are properly set
        assert db_service._connection_params is not None
        assert "host" in db_service._connection_params
        assert "dbname" in db_service._connection_params
        assert "user" in db_service._connection_params
        assert "password" in db_service._connection_params
        assert "sslmode" in db_service._connection_params

    def test_setup_database_instance(self, test_config, mock_databricks_client):
        """Test database instance setup."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor for extension creation
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            result = db_service.setup_database_instance()

            assert result is True

    def test_create_tables(self, test_config, mock_databricks_client):
        """Test table creation."""
        db_service = DatabaseService(client=mock_databricks_client, config=test_config)

        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = create_mock_cursor()
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(db_service, "connect_to_pg") as mock_connect:
            mock_connect.return_value = mock_conn

            result = db_service.create_tables()

            assert result is True
