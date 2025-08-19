"""Document processing workflow orchestrator."""

import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

from databricks.sdk import WorkspaceClient
from ..services import (
    StorageService,
    DocumentService,
    DatabaseService,
)
from ..utils import get_current_user

logger = logging.getLogger(__name__)


class DocumentWorkflow:
    """Orchestrates end-to-end document processing workflow."""

    def __init__(
        self,
        client: Optional[WorkspaceClient],
        storage_service: StorageService,
        document_service: DocumentService,
        database_service: DatabaseService,
    ):
        self.client = client
        self.storage_service = storage_service
        self.document_service = document_service
        self.database_service = database_service

    def process_document(
        self, file_content: bytes, filename: str, username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document end-to-end.

        Args:
            file_content: Document content as bytes
            filename: Original filename
            username: Username (will get from auth service if not provided)

        Returns:
            Dict with processing results and status
        """
        logger.info(f"Starting document processing workflow for: {filename}")

        # Get username if not provided
        if not username:
            username = get_current_user(self.client)

        # Create user record
        user = self.database_service.create_user(username)
        if not user:
            return {
                "success": False,
                "error": "Failed to create user record",
                "stage": "user_creation",
            }

        try:
            # Stage 1: Upload document
            upload_success, doc_hash, upload_path, upload_message = (
                self.storage_service.upload_document(file_content, filename, username)
            )

            if not upload_success:
                logger.warning(f"Upload failed but continuing: {upload_message}")

            # Stage 2: Create document record
            document = self.database_service.create_document(
                user_id=str(user.id),
                doc_hash=doc_hash,
                filename=filename,
                status="uploaded",
            )

            if not document:
                return {
                    "success": False,
                    "error": "Failed to create document record",
                    "stage": "document_creation",
                    "doc_hash": doc_hash,
                }

            # Stage 3: Queue processing job (if upload successful)
            run_id = None
            if upload_success:
                output_path = f"{Path(upload_path).parent}/processed"
                job_success, run_id, job_message = (
                    self.document_service.queue_document_processing(
                        input_path=upload_path,
                        output_path=str(output_path),
                        doc_hash=doc_hash,
                    )
                )

                if job_success:
                    # Update document status
                    self.database_service.update_document_status(
                        str(document.id), "processing"
                    )
                    logger.info(f"Job queued successfully: {run_id}")
                else:
                    logger.warning(f"Job queue failed: {job_message}")

            # Stage 4: Process content directly for immediate chunking
            # This allows the chat to work even if Databricks processing fails
            content_str = self._extract_text_content(file_content, filename)
            if content_str:
                self._process_content_immediately(str(document.id), content_str)

            return {
                "success": True,
                "document_id": str(document.id),
                "doc_hash": doc_hash,
                "filename": filename,
                "upload_success": upload_success,
                "upload_message": upload_message,
                "run_id": run_id,
                "stage": "completed",
            }

        except Exception as e:
            logger.error(f"Document processing workflow failed: {str(e)}")
            return {"success": False, "error": str(e), "stage": "workflow_error"}

    def poll_and_finalize_processing(
        self, doc_hash: str, run_id: Optional[int] = None, timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Poll for AI parsing completion by checking the database for processed results.
        The AI parsing job will append rows to the database with user_id and file_path.

        Returns:
            Dict with polling results
        """
        logger.info(f"Polling for AI parsing completion: {doc_hash}")

        # Get document
        document = self.database_service.get_document_by_hash(doc_hash)
        if not document:
            return {
                "success": False,
                "error": "Document not found",
                "doc_hash": doc_hash,
            }

        # Extract file path and user ID for polling
        file_path = document.get("original_path") or document.get("filename")
        user_id = document.get("user_id")

        if not file_path or not user_id:
            return {
                "success": False,
                "error": "Document missing file path or user ID",
                "doc_hash": doc_hash,
            }

        logger.info(f"Polling database for file: {file_path}, user: {user_id}")

        # Poll the database every 2 seconds for AI parsing completion
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if AI parsing has completed by looking for processed chunks
            completion_result = self.database_service.check_ai_parsing_completion(
                file_path, user_id
            )

            if completion_result and completion_result.get("completed"):
                logger.info(f"AI parsing completed for document {doc_hash}")

                # Update document status to processed
                self.database_service.update_document_status(
                    str(document["id"]), "processed"
                )

                return {
                    "success": True,
                    "doc_hash": doc_hash,
                    "message": "AI parsing completed successfully",
                    "results": completion_result,
                    "total_chunks": completion_result.get("total_chunks", 0),
                }

            # Wait 2 seconds before next poll
            time.sleep(2)

            # Log progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                logger.info(f"Still polling... {int(elapsed)}s elapsed")

        # Timeout reached
        logger.warning(f"AI parsing polling timeout for document {doc_hash}")
        self.database_service.update_document_status(
            str(document["id"]), "failed", "AI parsing timeout"
        )

        return {
            "success": False,
            "error": "AI parsing timeout - no results found in database",
            "doc_hash": doc_hash,
        }

    def _extract_text_content(
        self, file_content: bytes, filename: str
    ) -> Optional[str]:
        """Extract text content from file for immediate processing."""
        try:
            # Simple text extraction - in production this would be more sophisticated
            file_extension = Path(filename).suffix.lower()

            if file_extension in [".txt", ".md"]:
                return file_content.decode("utf-8", errors="ignore")
            elif file_extension == ".pdf":
                # Placeholder - would use PyPDF2 or similar
                return f"[PDF Content] Filename: {filename}\nThis is placeholder text content extracted from the PDF file. In production, this would use proper PDF parsing."
            elif file_extension in [".doc", ".docx"]:
                # Placeholder - would use python-docx or similar
                return f"[Document Content] Filename: {filename}\nThis is placeholder text content extracted from the document file. In production, this would use proper document parsing."
            else:
                return f"[File Content] Filename: {filename}\nUnsupported file type for text extraction: {file_extension}"

        except Exception as e:
            logger.error(f"Failed to extract text content: {e}")
            return f"[Error] Failed to extract content from {filename}: {str(e)}"

    def _process_content_immediately(self, document_id: str, content: str):
        """Process content immediately for basic chunking and storage."""
        try:
            # Simple chunking - split by paragraphs or sentences
            chunks = self._simple_chunk_text(content)

            # Prepare chunk data for database storage
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append(
                    {
                        "content": chunk,
                        "embedding": None,  # Embeddings are handled separately
                        "metadata": {
                            "chunk_index": i,
                            "processing_method": "immediate",
                            "chunk_type": "text",
                        },
                        "token_count": len(chunk.split()),  # Simple token count
                    }
                )

            # Store chunks in database
            self.database_service.store_document_chunks(document_id, chunk_data)
            logger.info(f"Stored {len(chunk_data)} chunks for immediate processing")

        except Exception as e:
            logger.error(f"Failed to process content immediately: {e}")

    def _simple_chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Simple text chunking by paragraphs and sentences."""
        # Split by double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we have no chunks or chunks are too large, split by sentences
        if not chunks or any(len(chunk) > chunk_size * 2 for chunk in chunks):
            sentences = text.replace("\n", " ").split(". ")
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip() + ".")
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip() + ".")

        return chunks

    def _process_ai_parsed_content(self, document_id: str, processed_content: bytes):
        """Process AI-parsed content from Databricks job."""
        try:
            import json

            # Parse the processed content
            content_str = processed_content.decode("utf-8")
            parsed_data = json.loads(content_str)

            if parsed_data.get("status") != "success":
                logger.error(
                    f"AI processing failed: {parsed_data.get('error', 'Unknown error')}"
                )
                return

            # Extract and process the parsed content
            parsed_content_data = parsed_data.get("parsed_content", {})

            # For now, just extract text from pages
            # In production, this would be more sophisticated
            all_text = []
            for page in parsed_content_data.get("pages", []):
                for block in page.get("text_blocks", []):
                    all_text.append(block.get("text", ""))

            if all_text:
                content = "\n\n".join(all_text)
                # Re-process with AI-parsed content (overwrites immediate processing)
                self._process_content_immediately(document_id, content)
                logger.info("Successfully processed AI-parsed content")

        except Exception as e:
            logger.error(f"Failed to process AI-parsed content: {e}")

    def get_document_status(self, doc_hash: str) -> Dict[str, Any]:
        """Get current document processing status."""
        document = self.database_service.get_document_by_hash(doc_hash)
        if not document:
            return {"success": False, "error": "Document not found"}

        return {
            "success": True,
            "doc_hash": doc_hash,
            "filename": document.filename,
            "status": document.status,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "metadata": document.doc_metadata,
        }
