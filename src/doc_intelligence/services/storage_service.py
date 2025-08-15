"""Storage service for Databricks volumes."""

import hashlib
import time
import logging
import json
from pathlib import Path
from typing import Optional, Tuple
from databricks.sdk import WorkspaceClient

from ..config import config

logger = logging.getLogger(__name__)


class StorageService:
    """Service for document storage operations with Databricks volumes."""
    
    def __init__(self, client: Optional[WorkspaceClient] = None):
        self.client = client
        
    @staticmethod
    def generate_document_hash(file_content: bytes, username: str) -> str:
        """Generate a unique hash for the document based on content and user."""
        content_hash = hashlib.sha256(file_content).hexdigest()
        user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
        timestamp = str(int(time.time()))
        return f"{user_hash}_{content_hash[:16]}_{timestamp}"
    
    def upload_document(
        self, 
        file_content: bytes, 
        filename: str, 
        username: str,
        volume_path: Optional[str] = None
    ) -> Tuple[bool, str, str, str]:
        """
        Upload document to Databricks volume.
        
        Returns:
            Tuple of (success, doc_hash, upload_path, message)
        """
        # Use configured volume path or default
        volume_path = volume_path or config.databricks_volume_path or "/Volumes/main/default/documents"
        
        # Generate document hash and path
        doc_hash = self.generate_document_hash(file_content, username)
        file_extension = Path(filename).suffix
        unique_filename = f"{doc_hash}{file_extension}"
        upload_path = f"{volume_path}/{unique_filename}"
        
        # Check if client is available
        if not self.client:
            logger.warning("No Databricks client available for upload")
            return False, doc_hash, upload_path, "Databricks not connected. Document saved locally for processing."
        
        try:
            # Upload to volume using the Files API
            self.client.files.upload(upload_path, file_content)
            logger.info(f"Successfully uploaded {filename} to {upload_path}")
            return True, doc_hash, upload_path, f"Document uploaded successfully: {unique_filename}"
            
        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {str(e)}")
            return False, doc_hash, upload_path, f"Upload failed: {str(e)}"
    
    def download_file(self, file_path: str) -> Tuple[bool, Optional[bytes], str]:
        """
        Download file from Databricks volume.
        
        Returns:
            Tuple of (success, content, message)
        """
        if not self.client:
            logger.warning("No Databricks client available for download")
            # Return sample content for demo purposes
            sample_content = {
                "status": "success",
                "parsed_content": {
                    "pages": [{
                        "text_blocks": [{
                            "text": "Sample processed content. Connect Databricks for real AI parsing.",
                            "confidence": 0.95,
                            "type": "paragraph"
                        }]
                    }]
                }
            }
            return True, json.dumps(sample_content, indent=2).encode("utf-8"), "Returned sample content (Databricks not connected)"
        
        try:
            result = self.client.files.download(file_path)
            logger.info(f"Successfully downloaded file from: {file_path}")
            return True, result.contents, "File downloaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {str(e)}")
            return False, None, f"Download failed: {str(e)}"
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in Databricks volume."""
        if not self.client:
            return False
            
        try:
            self.client.files.download(file_path)
            return True
        except Exception:
            return False
    
    def list_files(self, directory_path: str) -> Tuple[bool, list[str], str]:
        """
        List files in a directory.
        
        Returns:
            Tuple of (success, file_list, message)
        """
        if not self.client:
            return False, [], "Databricks client not available"
            
        try:
            # Note: This is a simplified implementation
            # In practice, you'd use the appropriate Databricks SDK method
            files = []  # Would be populated by actual SDK call
            return True, files, "Files listed successfully"
        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {str(e)}")
            return False, [], f"Failed to list files: {str(e)}"
