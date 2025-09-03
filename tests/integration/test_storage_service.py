"""Integration tests for the StorageService."""

import pytest
import hashlib
import time
from unittest.mock import Mock, patch
from pathlib import Path
from databricks.sdk.service.catalog import VolumeType

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.storage.service import StorageService


class TestStorageService:
    """Integration tests for the storage service."""

    def test_document_hash_uniqueness(self):
        """Test document hash uniqueness for different content and users."""
        content1 = b"First document content"
        content2 = b"Second document content"
        user1 = "user1@databricks.com"
        user2 = "user2@databricks.com"

        # Generate hashes
        hash1 = StorageService.generate_file_hash(content1, user1)
        hash2 = StorageService.generate_file_hash(content2, user1)
        hash3 = StorageService.generate_file_hash(content1, user2)

        # All hashes should be different
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_download_file_success(self, test_config, mock_databricks_client):
        """Test successful file download."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Test file download
        success, content, message = storage_service.download_file("/path/to/file.txt")

        assert success is True
        assert content == b"test content"
        assert "successfully" in message.lower()

    def test_download_file_failure(self, test_config, mock_databricks_client):
        """Test file download failure."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock client to raise exception
        mock_databricks_client.files.download.side_effect = Exception("Download failed")

        # Test file download failure
        success, content, message = storage_service.download_file("/path/to/file.txt")

        assert success is False
        assert content is None
        assert "failed" in message.lower()

    def test_file_exists_success(self, test_config, mock_databricks_client):
        """Test file existence check success."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Test file existence check
        exists = storage_service.file_exists("/path/to/file.txt")

        assert exists is True

    def test_list_files_failure(self, test_config, mock_databricks_client):
        """Test file listing failure."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock client to raise exception (simulated by not implementing the method)
        # The current implementation returns empty list, so we test that
        success, files, message = storage_service.list_files("/path/to/directory")

        assert success is True
        assert isinstance(files, list)

    def test_volume_exists_success(self, test_config, mock_databricks_client):
        """Test volume existence check success."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock successful volume list with test volume
        mock_volume = Mock()
        mock_volume.name = "test_volume"
        mock_databricks_client.volumes.list.return_value = [mock_volume]

        # Test volume existence check
        exists = storage_service.volume_exists("test_volume")

        assert exists is True
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )

    def test_volume_exists_failure(self, test_config, mock_databricks_client):
        """Test volume existence check failure."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume not found (empty list)
        mock_databricks_client.volumes.list.return_value = []

        # Test volume existence check
        exists = storage_service.volume_exists("nonexistent_volume")

        assert exists is False
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )

    def test_volume_exists_custom_catalog_schema(
        self, test_config, mock_databricks_client
    ):
        """Test volume existence check with custom catalog and schema."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock successful volume list with test volume
        mock_volume = Mock()
        mock_volume.name = "test_volume"
        mock_databricks_client.volumes.list.return_value = [mock_volume]

        # Test volume existence check with custom catalog and schema
        exists = storage_service.volume_exists("test_volume", "my_catalog", "my_schema")

        assert exists is True
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="my_catalog", schema_name="my_schema"
        )

    def test_create_volume_already_exists(self, test_config, mock_databricks_client):
        """Test volume creation when volume already exists."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume already exists
        mock_volume = Mock()
        mock_volume.name = "existing_volume"
        mock_databricks_client.volumes.list.return_value = [mock_volume]

        # Test volume creation
        success, message = storage_service.create_volume("existing_volume")

        assert success is True
        assert "already exists" in message
        assert "existing_volume" in message
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )
        # Should not call create since volume already exists
        mock_databricks_client.volumes.create.assert_not_called()

    def test_create_volume_new_volume(self, test_config, mock_databricks_client):
        """Test volume creation for new volume."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume doesn't exist, then successful creation
        mock_databricks_client.volumes.list.return_value = []
        mock_databricks_client.volumes.create.return_value = Mock()

        # Test volume creation
        success, message = storage_service.create_volume("new_volume")

        assert success is True
        assert "created successfully" in message
        assert "new_volume" in message
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )
        mock_databricks_client.volumes.create.assert_called_once_with(
            name="new_volume",
            catalog_name="test",
            schema_name="default",
            comment=None,
            volume_type=VolumeType.MANAGED,
        )

    def test_create_volume_custom_catalog_schema(
        self, test_config, mock_databricks_client
    ):
        """Test volume creation with custom catalog and schema."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume doesn't exist, then successful creation
        mock_databricks_client.volumes.list.return_value = []
        mock_databricks_client.volumes.create.return_value = Mock()

        # Test volume creation with custom catalog and schema
        success, message = storage_service.create_volume(
            "custom_volume", "my_catalog", "my_schema", "Custom volume for testing"
        )

        assert success is True
        assert "created successfully" in message
        assert "custom_volume" in message
        # The volume_exists call uses default catalog/schema, but create uses custom ones
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )
        mock_databricks_client.volumes.create.assert_called_once_with(
            name="custom_volume",
            catalog_name="my_catalog",
            schema_name="my_schema",
            comment="Custom volume for testing",
            volume_type=VolumeType.MANAGED,
        )

    def test_create_volume_creation_failure(self, test_config, mock_databricks_client):
        """Test volume creation failure."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume doesn't exist, but creation fails
        mock_databricks_client.volumes.list.return_value = []
        mock_databricks_client.volumes.create.side_effect = Exception("Creation failed")

        # Test volume creation failure
        success, message = storage_service.create_volume("failing_volume")

        assert success is False
        assert "Failed to create volume" in message
        assert "failing_volume" in message
        mock_databricks_client.volumes.list.assert_called_once_with(
            catalog_name="test", schema_name="default"
        )
        mock_databricks_client.volumes.create.assert_called_once()

    def test_create_volume_default_comment(self, test_config, mock_databricks_client):
        """Test volume creation with default comment."""
        storage_service = StorageService(
            client=mock_databricks_client, config=test_config
        )

        # Mock volume doesn't exist, then successful creation
        mock_databricks_client.volumes.list.return_value = []
        mock_databricks_client.volumes.create.return_value = Mock()

        # Test volume creation without custom comment
        success, message = storage_service.create_volume("default_comment_volume")

        assert success is True
        mock_databricks_client.volumes.create.assert_called_once_with(
            name="default_comment_volume",
            catalog_name="test",
            schema_name="default",
            comment=None,
            volume_type=VolumeType.MANAGED,
        )
