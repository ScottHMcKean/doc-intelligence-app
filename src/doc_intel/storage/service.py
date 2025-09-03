"""Storage service for Databricks volumes."""

import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from doc_intel.config import DocConfig
from doc_intel.utils import check_workspace_client
from doc_intel.config import check_config

logger = logging.getLogger(__name__)


class StorageService:
    """File storage with Databricks volumes"""

    def __init__(self, client: WorkspaceClient, config: DocConfig):
        self.client = client
        check_workspace_client(client)
        self.user_id = self.client.current_user.me().id

        self.config = config
        check_config(config)

        self.max_file_size_mb = config.storage.max_file_size_mb
        self.allowed_extensions = config.storage.allowed_extensions

    @staticmethod
    def generate_file_hash(file_content: bytes, user_id: str) -> str:
        """Generate a unique hash for the file based on content and user."""
        content_hash = hashlib.sha256(file_content).hexdigest()[:16]
        timestamp = str(int(time.time()))
        return f"{user_id}_{timestamp}_{content_hash}"

    def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        volume_name: str,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Upload file to Databricks volume.

        Args:
            file_content: The content of the file to upload.
            file_name: The name of the file to upload (with extension)
            volume_name: The volume name.
            catalog_name: The catalog name (defaults to config value if None).
            schema: The schema name (defaults to config value if None).

        Returns:
            Tuple of (success, upload_path, message)
        """
        if catalog_name is None:
            catalog_name = self.config.storage.catalog
        if schema_name is None:
            schema_name = self.config.storage.schema

        dest_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{file_name}"

        try:
            # Upload to volume using the Files API
            self.client.files.upload(dest_path, file_content, overwrite=True)
            logger.info(f"Successfully uploaded file to {dest_path}")
            return (
                True,
                dest_path,
                f"File uploaded successfully: {dest_path}",
            )

        except Exception as e:
            logger.error(f"Failed to upload file {dest_path}: {str(e)}")
            return False, dest_path, f"Upload failed: {str(e)}"

    def download_file(self, file_path: str) -> Tuple[bool, Optional[bytes], str]:
        """
        Download file from Databricks volume.

        Returns:
            Tuple of (success, content, message)
        """
        try:
            result = self.client.files.download(file_path)
            logger.info(f"Successfully downloaded file from: {file_path}")
            # Extract bytes from the streaming response
            content_bytes = result.contents.read()
            return True, content_bytes, "File downloaded successfully"

        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {str(e)}")
            return False, None, f"Download failed: {str(e)}"

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in Databricks volume."""
        try:
            # Use get_metadata instead of download for faster file existence check
            self.client.files.get_metadata(file_path)
            return True
        except Exception:
            return False

    def list_files(self, directory_path: str) -> Tuple[bool, list[str], str]:
        """
        List files in a directory.

        Returns:
            Tuple of (success, file_list, message)
        """
        try:
            files = [
                x.name
                for x in self.client.files.list_directory_contents(directory_path)
                if not x.is_directory
            ]

            logger.info(f"Successfully listed {len(files)} files in {directory_path}")
            return True, files, f"Found {len(files)} files"
        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {str(e)}")
            return False, [], f"Failed to list files: {str(e)}"

    def get_file_info(self, file_path: str) -> Tuple[bool, Optional[dict], str]:
        """
        Get detailed information about a file.

        Returns:
            Tuple of (success, file_info, message)
        """
        try:
            metadata = self.client.files.get_metadata(file_path).as_shallow_dict()
            logger.info(f"Successfully retrieved file metadata for {file_path}")
            return True, metadata, "File info retrieved successfully"
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {str(e)}")
            return False, None, f"Failed to get file info: {str(e)}"

    def volume_exists(
        self, volume_name: str, catalog_name: str = None, schema_name: str = None
    ) -> bool:
        """
        Check if a volume exists in the specified catalog and schema.

        Args:
            volume_name: Name of the volume to check
            catalog_name: Name of the catalog (default: "main")
            schema_name: Name of the schema (default: "default")

        Returns:
            True if volume exists, False otherwise
        """

        if catalog_name is None:
            catalog_name = self.config.storage.catalog
        if schema_name is None:
            schema_name = self.config.storage.schema

        return any(
            vol.name == volume_name
            for vol in self.client.volumes.list(
                catalog_name=catalog_name, schema_name=schema_name
            )
        )

    def create_volume(
        self,
        volume_name: str,
        catalog_name: str = None,
        schema_name: str = None,
        comment: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Create a volume if it doesn't already exist.

        Args:
            volume_name: Name of the volume to create
            catalog_name: Name of the catalog (default: "main")
            schema_name: Name of the schema (default: "default")
            comment: Optional comment for the volume

        Returns:
            Tuple of (success, message)
        """
        if catalog_name is None:
            catalog_name = self.config.storage.catalog
        if schema_name is None:
            schema_name = self.config.storage.schema

        try:
            # Check if volume already exists
            if self.volume_exists(volume_name):
                logger.info(
                    f"Volume {volume_name} already exists in {catalog_name}.{schema_name}"
                )
                return (
                    True,
                    f"Volume {volume_name} already exists in {catalog_name}.{schema_name}",
                )

            self.client.volumes.create(
                name=volume_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                comment=comment,
                volume_type=VolumeType.MANAGED,
            )

            logger.info(
                f"Successfully created volume {volume_name} in {catalog_name}.{schema_name}"
            )
            return (
                True,
                f"Volume {volume_name} created successfully in {catalog_name}.{schema_name}",
            )

        except Exception as e:
            logger.error(f"Failed to create volume {volume_name}: {str(e)}")
            return False, f"Failed to create volume {volume_name}: {str(e)}"
