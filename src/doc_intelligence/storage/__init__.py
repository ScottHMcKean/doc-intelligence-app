"""Storage module for document upload and retrieval."""

from .volume_storage import (
    upload_document,
    get_document_hash,
    poll_processed_document,
    download_processed_document,
)

__all__ = [
    "upload_document",
    "get_document_hash",
    "poll_processed_document",
    "download_processed_document",
]
