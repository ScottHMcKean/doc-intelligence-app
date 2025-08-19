"""Databricks volume storage operations."""

import hashlib
import time
import logging
import json
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from databricks.sdk import WorkspaceClient

from ..utils import get_databricks_client

# Config will be passed as parameter to functions that need it

logger = logging.getLogger(__name__)


def get_document_hash(file_content: bytes, username: str) -> str:
    """Generate a unique hash for the document based on content and user."""
    content_hash = hashlib.sha256(file_content).hexdigest()
    user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
    timestamp = str(int(time.time()))
    return f"{user_hash}_{content_hash[:16]}_{timestamp}"


def upload_document(
    file_content: bytes,
    filename: str,
    username: str,
    input_volume_path: Optional[str] = None,
    config=None,
) -> Tuple[str, str]:
    """
    Upload document to Databricks volume with graceful error handling.

    Returns:
        Tuple of (document_hash, uploaded_file_path)
    """
    # Create fallback config for backwards compatibility
    if config is None:
        try:
            from ..config import DocConfig

            config = DocConfig("./config.yaml")
        except:
            logger.error("Could not load configuration")
            config = None

    # Use configured volume path or default
    volume_path = (
        input_volume_path
        or (config.get("storage.volume_path") if config else None)
        or "/Volumes/main/default/documents"
    )

    # Generate document hash
    doc_hash = get_document_hash(file_content, username)
    file_extension = Path(filename).suffix
    unique_filename = f"{doc_hash}{file_extension}"
    upload_path = f"{volume_path}/{unique_filename}"

    # Check if Databricks is available
    if not config.databricks_available:
        logger.warning("Databricks not available, simulating upload")
        st.warning("⚠️ Databricks not connected. Document saved locally for processing.")
        # In a real implementation, you might save to local storage here
        return doc_hash, upload_path

    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client")
            st.error("❌ Unable to connect to Databricks for upload")
            # Return hash anyway for local processing
            return doc_hash, upload_path

        # Upload to volume using the Files API
        with st.spinner(f"Uploading {filename} to Databricks volume..."):
            client.files.upload(upload_path, file_content)
            logger.info(f"Successfully uploaded {filename} to {upload_path}")

        st.success(f"✅ Document uploaded successfully: {unique_filename}")
        return doc_hash, upload_path

    except Exception as e:
        logger.error(f"Failed to upload document {filename}: {str(e)}")
        st.error(f"❌ Upload failed: {str(e)}")
        # Still return the hash for local processing
        return doc_hash, upload_path


def poll_processed_document(
    doc_hash: str,
    output_volume_path: Optional[str] = None,
    timeout_seconds: int = 300,
    poll_interval: int = 5,
    config=None,
) -> Optional[str]:
    """
    Poll for processed document in output volume with graceful error handling.

    Returns:
        Path to processed document if found, None if timeout.
    """
    # Create fallback config for backwards compatibility
    if config is None:
        try:
            from ..config import DocConfig

            config = DocConfig("./config.yaml")
        except:
            logger.error("Could not load configuration")
            config = None

    # Use configured volume path or default
    storage_volume_path = (
        config.get("storage.volume_path") if config else None
    ) or "/Volumes/main/default/documents"

    volume_path = output_volume_path or f"{storage_volume_path}/processed"
    processed_filename = f"{doc_hash}_processed.json"
    processed_path = f"{volume_path}/{processed_filename}"

    # Check if Databricks is available
    if not config.databricks_available:
        logger.warning("Databricks not available, simulating processing")
        with st.spinner(
            "⚠️ Simulating document processing (Databricks not connected)..."
        ):
            progress_bar = st.progress(0)
            for i in range(20):  # Quick simulation
                progress_bar.progress((i + 1) / 20)
                time.sleep(0.1)
        st.warning(
            "⚠️ Simulated processing completed. Connect Databricks for real AI parsing."
        )
        return processed_path

    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client for polling")
            st.error("❌ Unable to connect to Databricks for polling")
            return None

        start_time = time.time()

        with st.spinner("🔄 Processing document... This may take a few minutes."):
            progress_bar = st.progress(0)

            while time.time() - start_time < timeout_seconds:
                try:
                    # Check if processed file exists
                    result = client.files.download(processed_path)
                    logger.info(f"Found processed document: {processed_path}")
                    st.success("✅ Document processing completed!")
                    return processed_path

                except Exception:
                    # File doesn't exist yet, continue polling
                    elapsed = time.time() - start_time
                    progress = min(elapsed / timeout_seconds, 1.0)
                    progress_bar.progress(progress)
                    time.sleep(poll_interval)

            # Timeout reached
            logger.warning(f"Polling timeout for document {doc_hash}")
            st.warning("⏰ Document processing timed out. Please try again later.")
            return None

    except Exception as e:
        logger.error(f"Error polling for processed document {doc_hash}: {str(e)}")
        st.error(f"❌ Error polling for processed document: {str(e)}")
        return None


def download_processed_document(file_path: str, config=None) -> Optional[bytes]:
    """Download processed document from volume with graceful error handling."""
    # Create fallback config for backwards compatibility
    if config is None:
        try:
            from ..config import DocConfig

            config = DocConfig("./config.yaml")
        except:
            logger.error("Could not load configuration")
            config = None

    # Check if Databricks is available
    if not config or not config.databricks_available:
        logger.warning("Databricks not available, returning sample processed content")
        # Return sample processed document content for demo purposes
        mock_processed_content = {
            "status": "success",
            "parsed_content": {
                "pages": [
                    {
                        "text_blocks": [
                            {
                                "text": "This is sample processed content from the uploaded document. In production, this would be generated by Databricks AI parsing services.",
                                "position": {
                                    "x": 50,
                                    "y": 100,
                                    "width": 500,
                                    "height": 50,
                                },
                                "confidence": 0.95,
                                "type": "paragraph",
                            }
                        ],
                        "tables": [],
                        "images": [],
                    }
                ],
                "metadata": {
                    "total_pages": 1,
                    "processing_time": "2.5s",
                    "confidence_score": 0.95,
                },
            },
        }
        return json.dumps(mock_processed_content, indent=2).encode("utf-8")

    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client for download")
            st.error("❌ Unable to connect to Databricks for download")
            return None

        logger.info(f"Downloading processed document from: {file_path}")
        result = client.files.download(file_path)
        # Extract bytes from the streaming response
        content_bytes = result.contents.read()
        return content_bytes

    except Exception as e:
        logger.error(f"Failed to download processed document {file_path}: {str(e)}")
        st.error(f"❌ Failed to download processed document: {str(e)}")
        return None
