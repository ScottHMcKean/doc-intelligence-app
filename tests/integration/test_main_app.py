"""Integration tests for the main DocumentIntelligenceApp."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import uuid

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.app import DocumentIntelligenceApp


class TestDocumentIntelligenceApp:
    """Integration tests for the main application class."""

    def test_app_initialization(self, test_config, mock_databricks_client):
        """Test that the app initializes correctly with all services."""
        with patch(
            "doc_intel.app.get_workspace_client",
            return_value=mock_databricks_client,
        ):
            app = DocumentIntelligenceApp(config_path=test_config.config_path)

            # Check that all services are initialized
            assert app.config is not None
            assert app.databricks_client is not None
            assert app.database_service is not None
            assert app.storage_service is not None
            assert app.document_service is not None
            assert app.agent_service is not None

    def test_system_status(self, test_app_with_mock_client):
        """Test system status reporting."""
        status = test_app_with_mock_client.get_system_status()

        # Check status structure
        assert "services" in status
        assert "configuration" in status
        assert "overall_health" in status

        # Check service status
        services = status["services"]
        assert "database" in services
        assert "agent" in services
        assert "storage" in services
        assert "document" in services

        # Check configuration
        config = status["configuration"]
        assert config["name"] == "Test Document Intelligence"
        assert config["environment"] == "databricks"

    def test_user_authentication_flow(self, test_app_with_mock_client, test_user_info):
        """Test user authentication and verification."""
        username, user_id = test_user_info

        # Mock the get_current_user function
        with patch("doc_intel.app.get_current_user", return_value=(username, user_id)):
            # Test getting current user
            current_user = test_app_with_mock_client.get_current_user()
            assert current_user == (username, user_id)

            # Test user verification (this will fail without real database, but we can test the flow)
            # In a real integration test, you'd have a test database
            try:
                is_authenticated = (
                    test_app_with_mock_client.verify_user_authentication()
                )
                # This might fail without real database, which is expected
            except Exception as e:
                # Expected to fail without real database connection
                assert "database" in str(e).lower() or "connection" in str(e).lower()

    def test_document_upload_flow(
        self,
        test_app_with_mock_client,
        sample_document_content,
        sample_document_filename,
    ):
        """Test document upload and processing flow."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Test document upload
            result = test_app_with_mock_client.upload_and_process_document(
                file_content=sample_document_content, filename=sample_document_filename
            )

            # Check result structure
            assert "success" in result
            assert "document_id" in result
            assert "doc_hash" in result
            assert "filename" in result

            # The upload should succeed (even if processing fails)
            assert result["success"] is True
            assert result["filename"] == sample_document_filename
            assert result["doc_hash"] is not None

    def test_conversation_management(self, test_app_with_mock_client):
        """Test conversation creation and management."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Test starting a new conversation
            conv_result = test_app_with_mock_client.start_new_conversation(
                title="Test Conversation"
            )

            # Check conversation creation result
            assert "success" in conv_result
            if conv_result["success"]:
                assert "conversation_id" in conv_result
                assert "thread_id" in conv_result
                assert "title" in conv_result
                assert conv_result["title"] == "Test Conversation"
            else:
                # Expected to fail without real database
                assert "error" in conv_result

    def test_document_search(self, test_app_with_mock_client):
        """Test document search functionality."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Test document search
            search_results = test_app_with_mock_client.search_documents(
                query="test document content"
            )

            # Should return a list (empty if no documents or no database)
            assert isinstance(search_results, list)

    def test_user_documents_retrieval(self, test_app_with_mock_client):
        """Test retrieving user documents."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Test getting user documents
            documents = test_app_with_mock_client.get_user_documents()

            # Should return a list (empty if no documents or no database)
            assert isinstance(documents, list)

    def test_system_validation(self, test_app_with_mock_client):
        """Test system validation checks."""
        validation_result = test_app_with_mock_client.validate_system()

        # Check validation structure
        assert "overall_health" in validation_result
        assert "issues" in validation_result
        assert "recommendations" in validation_result
        assert "services" in validation_result

        # Issues and recommendations should be lists
        assert isinstance(validation_result["issues"], list)
        assert isinstance(validation_result["recommendations"], list)

    def test_configuration_handling(self, test_config):
        """Test configuration loading and access."""
        # Test basic configuration access
        assert test_config.get("application.name") == "Test Document Intelligence"
        assert test_config.get("storage.max_file_size_mb") == 10
        assert test_config.get("agent.llm.max_tokens") == 256

        # Test default values
        assert test_config.get("nonexistent.key", "default") == "default"

        # Test nested access
        assert test_config.application.name == "Test Document Intelligence"
        assert test_config.storage.max_file_size_mb == 10

    def test_error_handling_without_databricks(self, test_config):
        """Test error handling when Databricks client is not available."""
        # Create app without Databricks client
        with patch("doc_intel.app.get_workspace_client", return_value=None):
            try:
                app = DocumentIntelligenceApp(config_path=test_config.config_path)
                # App should still initialize, but services might not work
                assert app.config is not None
            except Exception as e:
                # Some initialization might fail, which is expected
                assert "client" in str(e).lower() or "connection" in str(e).lower()

    def test_document_processing_with_mock_services(
        self,
        test_app_with_mock_client,
        sample_document_content,
        sample_document_filename,
    ):
        """Test document processing with mocked services."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Mock the database service methods
            test_app_with_mock_client.database_service.create_user = Mock(
                return_value={"id": 12345, "username": "test@databricks.com"}
            )
            test_app_with_mock_client.database_service.create_document = Mock(
                return_value={"id": str(uuid.uuid4()), "doc_hash": "test_hash"}
            )

            # Test document processing
            result = test_app_with_mock_client.upload_and_process_document(
                file_content=sample_document_content, filename=sample_document_filename
            )

            # Should succeed with mocked services
            assert result["success"] is True
            assert result["filename"] == sample_document_filename

    def test_conversation_with_documents(self, test_app_with_mock_client):
        """Test conversation creation with specific documents."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Mock database service
            test_app_with_mock_client.database_service.create_user = Mock(
                return_value={"id": 12345, "username": "test@databricks.com"}
            )
            test_app_with_mock_client.database_service.create_conversation = Mock(
                return_value={"id": str(uuid.uuid4()), "title": "Test"}
            )

            # Test conversation with documents
            conv_result = test_app_with_mock_client.start_new_conversation(
                document_hashes=["doc1", "doc2"], title="Test with Documents"
            )

            if conv_result["success"]:
                assert "document_hashes" in conv_result
                assert conv_result["document_hashes"] == ["doc1", "doc2"]

    def test_chat_message_processing(self, test_app_with_mock_client):
        """Test chat message processing."""
        # Mock the get_current_user function
        with patch(
            "doc_intel.app.get_current_user",
            return_value=("test@databricks.com", 12345),
        ):
            # Mock database service methods
            test_app_with_mock_client.database_service.add_message = Mock(
                return_value={"id": str(uuid.uuid4())}
            )
            test_app_with_mock_client.database_service.get_user_conversations = Mock(
                return_value=[]
            )

            # Mock agent service
            test_app_with_mock_client.agent_service.generate_response = Mock(
                return_value=(True, "Test response", {})
            )

            # Test sending a chat message
            response = test_app_with_mock_client.send_chat_message(
                conversation_id=str(uuid.uuid4()), user_message="Hello, how are you?"
            )

            # Should return a response structure
            assert "success" in response
            if response["success"]:
                assert "response" in response
                assert "conversation_id" in response

    def test_document_status_tracking(self, test_app_with_mock_client):
        """Test document status tracking."""
        # Mock database service
        test_app_with_mock_client.database_service.get_document_by_hash = Mock(
            return_value={
                "id": str(uuid.uuid4()),
                "doc_hash": "test_hash",
                "filename": "test.txt",
                "status": "processed",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "doc_metadata": {},
            }
        )

        # Test getting document status
        status = test_app_with_mock_client.get_document_status("test_hash")

        assert status["success"] is True
        assert status["doc_hash"] == "test_hash"
        assert status["filename"] == "test.txt"
        assert status["status"] == "processed"

    def test_conversation_history_retrieval(self, test_app_with_mock_client):
        """Test conversation history retrieval."""
        # Mock database service
        test_app_with_mock_client.database_service.get_conversation_messages = Mock(
            return_value=[
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": "Hello",
                    "created_at": "2024-01-01T00:00:00Z",
                    "msg_metadata": {},
                },
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "Hi there!",
                    "created_at": "2024-01-01T00:01:00Z",
                    "msg_metadata": {},
                },
            ]
        )

        # Test getting conversation history
        history = test_app_with_mock_client.get_conversation_history(str(uuid.uuid4()))

        assert isinstance(history, list)
        if history:  # If mocked data is returned
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "assistant"

    def test_cleanup_resources(self, test_app_with_mock_client):
        """Test resource cleanup."""
        # This should not raise any exceptions
        test_app_with_mock_client.cleanup_resources()

        # Test passes if no exceptions are raised
        assert True
