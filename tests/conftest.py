"""Pytest configuration and fixtures for Document Intelligence tests."""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock
import yaml

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.config import DocConfig
from doc_intel.app import DocumentIntelligenceApp


@pytest.fixture
def test_config_data() -> Dict[str, Any]:
    """Test configuration data."""
    return {
        "application": {
            "name": "Test Document Intelligence",
            "debug_mode": True,
            "log_level": "DEBUG",
            "databricks_host": "https://test.databricks.net",
        },
        "storage": {
            "volume_path": "/Volumes/test/default/documents",
            "max_file_size_mb": 10,
            "allowed_extensions": [".pdf", ".txt", ".md"],
            "catalog": "test",
            "schema": "default",
        },
        "agent": {
            "llm": {
                "max_tokens": 256,
                "temperature": 0.1,
                "endpoint": "test-llm-endpoint",
            },
            "retrieval": {
                "embedding_endpoint": "test-embedding-endpoint",
                "similarity_threshold": 0.7,
                "max_results": 5,
            },
        },
        "database": {
            "instance_name": "test-instance",
            "database": "test_db",
            "user": "test_user",
        },
        "document": {"timeout_minutes": 5, "max_retries": 2, "auto_process": True},
    }


@pytest.fixture
def test_config_file(test_config_data: Dict[str, Any]) -> str:
    """Create a temporary test configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config_data, f)
        return f.name


@pytest.fixture
def test_config(test_config_file: str) -> DocConfig:
    """Create a test configuration object."""
    return DocConfig(test_config_file)


@pytest.fixture
def mock_databricks_client():
    """Mock Databricks workspace client."""
    from databricks.sdk import WorkspaceClient

    client = Mock(spec=WorkspaceClient)

    # Mock user info
    mock_email = Mock()
    mock_email.value = "test@databricks.com"
    client.current_user.me.return_value = Mock(
        user_name="test@databricks.com", id=12345, emails=[mock_email]
    )

    # Mock database operations
    client.database.get_database_instance.return_value = Mock(
        read_write_dns="test-db.databricks.net"
    )
    client.database.generate_database_credential.return_value = Mock(token="test-token")

    # Mock file operations
    client.files.upload.return_value = None
    client.files.download.return_value = Mock(
        contents=Mock(read=lambda: b"test content")
    )
    client.files.get_metadata.return_value = Mock()
    client.files.list_directory_contents.return_value = []

    # Mock volume operations
    client.volumes.get.return_value = Mock()
    client.volumes.create.return_value = Mock()
    client.volumes.list.return_value = []

    # Mock job operations
    client.jobs.create.return_value = Mock(job_id=123)
    client.jobs.run_now.return_value = Mock(run_id=456)
    client.jobs.get_run.return_value = Mock(
        state=Mock(
            life_cycle_state=Mock(value="TERMINATED"),
            result_state=Mock(value="SUCCESS"),
        ),
        start_time=None,
        end_time=None,
        run_page_url="https://test.databricks.net/jobs/456",
    )

    return client


@pytest.fixture
def sample_document_content() -> bytes:
    """Sample document content for testing."""
    return b"""This is a test document for the Document Intelligence application.
    
It contains multiple paragraphs to test document processing capabilities.
The document includes various types of content that would typically be found
in business documents.

Key features to test:
- Text extraction
- Chunking
- Embedding generation
- Vector search
- RAG capabilities

This document should be processed successfully by the system."""


@pytest.fixture
def sample_document_filename() -> str:
    """Sample document filename."""
    return "test_document.txt"


@pytest.fixture
def test_user_info() -> tuple[str, int]:
    """Test user information."""
    return ("test@databricks.com", 12345)


@pytest.fixture
def cleanup_test_files():
    """Cleanup function for test files."""
    files_to_cleanup = []

    def add_file(filepath: str):
        files_to_cleanup.append(filepath)

    yield add_file

    # Cleanup
    for filepath in files_to_cleanup:
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except Exception:
            pass


@pytest.fixture
def test_app_with_mock_client(
    test_config: DocConfig, mock_databricks_client
) -> DocumentIntelligenceApp:
    """Create a test app instance with mocked Databricks client."""
    # Temporarily patch the get_workspace_client function
    import doc_intel.app as app_module

    original_get_client = app_module.get_workspace_client
    app_module.get_workspace_client = lambda host, token: mock_databricks_client

    try:
        app = DocumentIntelligenceApp(config_path=test_config.config_path)
        return app
    finally:
        # Restore original function
        app_module.get_workspace_client = original_get_client


@pytest.fixture
def test_database_connection_string() -> str:
    """Test database connection string for integration tests."""
    # This would be a real connection string for integration tests
    # In practice, you'd use environment variables or test-specific config
    return "postgresql://test_user:test_pass@localhost:5432/test_db"


@pytest.fixture
def test_vector_embedding() -> list[float]:
    """Sample vector embedding for testing."""
    # 768-dimensional embedding (typical for many models)
    return [0.1] * 768


@pytest.fixture
def test_chunks() -> list[dict[str, Any]]:
    """Sample document chunks for testing."""
    return [
        {
            "content": "This is the first chunk of the test document.",
            "metadata": {"chunk_index": 0, "type": "paragraph"},
            "token_count": 10,
            "embedding": [0.1] * 768,
        },
        {
            "content": "This is the second chunk with different content.",
            "metadata": {"chunk_index": 1, "type": "paragraph"},
            "token_count": 9,
            "embedding": [0.2] * 768,
        },
        {
            "content": "Final chunk containing summary information.",
            "metadata": {"chunk_index": 2, "type": "summary"},
            "token_count": 7,
            "embedding": [0.3] * 768,
        },
    ]


@pytest.fixture
def test_conversation_messages() -> list[dict[str, str]]:
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "What is this document about?"},
        {
            "role": "assistant",
            "content": "This document appears to be about testing the Document Intelligence application.",
        },
        {"role": "user", "content": "Can you summarize the key points?"},
        {
            "role": "assistant",
            "content": "The key points include text extraction, chunking, embedding generation, vector search, and RAG capabilities.",
        },
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_databricks: mark test as requiring Databricks connection"
    )
    config.addinivalue_line(
        "markers", "requires_database: mark test as requiring database connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests that might be slow
        if any(
            keyword in item.name
            for keyword in ["database", "vector", "embedding", "llm"]
        ):
            item.add_marker(pytest.mark.slow)

        # Add requires_databricks marker to tests that need Databricks
        if any(
            keyword in item.name
            for keyword in ["databricks", "workspace", "job", "volume"]
        ):
            item.add_marker(pytest.mark.requires_databricks)

        # Add requires_database marker to tests that need database
        if any(
            keyword in item.name
            for keyword in ["database", "postgres", "vector_search"]
        ):
            item.add_marker(pytest.mark.requires_database)
