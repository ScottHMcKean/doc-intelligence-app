"""Document processing workflow orchestrator."""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from ..services import (
    AuthService, StorageService, ProcessingService, 
    DatabaseService, EmbeddingService
)

logger = logging.getLogger(__name__)


class DocumentWorkflow:
    """Orchestrates end-to-end document processing workflow."""
    
    def __init__(
        self,
        auth_service: AuthService,
        storage_service: StorageService,
        processing_service: ProcessingService,
        database_service: DatabaseService,
        embedding_service: EmbeddingService
    ):
        self.auth_service = auth_service
        self.storage_service = storage_service
        self.processing_service = processing_service
        self.database_service = database_service
        self.embedding_service = embedding_service
    
    def process_document(
        self,
        file_content: bytes,
        filename: str,
        username: Optional[str] = None
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
            username = self.auth_service.get_current_user()
        
        # Create user record
        user = self.database_service.create_user(username)
        if not user:
            return {
                "success": False,
                "error": "Failed to create user record",
                "stage": "user_creation"
            }
        
        try:
            # Stage 1: Upload document
            upload_success, doc_hash, upload_path, upload_message = self.storage_service.upload_document(
                file_content, filename, username
            )
            
            if not upload_success:
                logger.warning(f"Upload failed but continuing: {upload_message}")
            
            # Stage 2: Create document record
            document = self.database_service.create_document(
                user_id=str(user.id),
                doc_hash=doc_hash,
                filename=filename,
                status="uploaded"
            )
            
            if not document:
                return {
                    "success": False,
                    "error": "Failed to create document record",
                    "stage": "document_creation",
                    "doc_hash": doc_hash
                }
            
            # Stage 3: Queue processing job (if upload successful)
            run_id = None
            if upload_success:
                output_path = f"{Path(upload_path).parent}/processed"
                job_success, run_id, job_message = self.processing_service.queue_document_processing(
                    input_path=upload_path,
                    output_path=str(output_path),
                    doc_hash=doc_hash
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
                "stage": "completed"
            }
            
        except Exception as e:
            logger.error(f"Document processing workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stage": "workflow_error"
            }
    
    def poll_and_finalize_processing(
        self, 
        doc_hash: str, 
        run_id: Optional[int] = None,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Poll for processing completion and finalize.
        
        Returns:
            Dict with polling results
        """
        logger.info(f"Polling for document processing completion: {doc_hash}")
        
        # Get document
        document = self.database_service.get_document_by_hash(doc_hash)
        if not document:
            return {
                "success": False,
                "error": "Document not found",
                "doc_hash": doc_hash
            }
        
        # If we have a run_id, poll for job completion
        if run_id:
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                job_success, status_info, status_message = self.processing_service.check_job_status(run_id)
                
                if job_success and status_info:
                    state = status_info.get("state")
                    result_state = status_info.get("result_state")
                    
                    if state == "TERMINATED":
                        if result_state == "SUCCESS":
                            logger.info(f"Job {run_id} completed successfully")
                            break
                        else:
                            logger.error(f"Job {run_id} failed with result: {result_state}")
                            self.database_service.update_document_status(
                                str(document.id), "failed", f"Job failed: {result_state}"
                            )
                            return {
                                "success": False,
                                "error": f"Processing job failed: {result_state}",
                                "doc_hash": doc_hash
                            }
                
                time.sleep(5)  # Poll every 5 seconds
            else:
                # Timeout reached
                logger.warning(f"Polling timeout for document {doc_hash}")
                self.database_service.update_document_status(
                    str(document.id), "failed", "Processing timeout"
                )
                return {
                    "success": False,
                    "error": "Processing timeout",
                    "doc_hash": doc_hash
                }
        
        # Try to download processed content
        processed_path = f"{self.storage_service.client}/processed/{doc_hash}_processed.json"
        download_success, processed_content, download_message = self.storage_service.download_file(processed_path)
        
        if download_success and processed_content:
            # Process the AI-parsed content
            self._process_ai_parsed_content(str(document.id), processed_content)
            
            # Update document status
            self.database_service.update_document_status(str(document.id), "processed")
            
            return {
                "success": True,
                "doc_hash": doc_hash,
                "message": "Document processing completed successfully"
            }
        else:
            logger.warning(f"Failed to download processed content: {download_message}")
            # Keep the immediate processing results
            self.database_service.update_document_status(str(document.id), "processed")
            return {
                "success": True,
                "doc_hash": doc_hash,
                "message": "Document processed with basic chunking (AI processing unavailable)"
            }
    
    def _extract_text_content(self, file_content: bytes, filename: str) -> Optional[str]:
        """Extract text content from file for immediate processing."""
        try:
            # Simple text extraction - in production this would be more sophisticated
            file_extension = Path(filename).suffix.lower()
            
            if file_extension in ['.txt', '.md']:
                return file_content.decode('utf-8', errors='ignore')
            elif file_extension == '.pdf':
                # Placeholder - would use PyPDF2 or similar
                return f"[PDF Content] Filename: {filename}\nThis is placeholder text content extracted from the PDF file. In production, this would use proper PDF parsing."
            elif file_extension in ['.doc', '.docx']:
                # Placeholder - would use python-docx or similar
                return f"[Document Content] Filename: {filename}\nThis is placeholder text content extracted from the document file. In production, this would use proper document parsing."
            else:
                return f"[File Content] Filename: {filename}\nUnsupported file type for text extraction: {file_extension}"
                
        except Exception as e:
            logger.error(f"Failed to extract text content: {e}")
            return f"[Error] Failed to extract content from {filename}: {str(e)}"
    
    def _process_content_immediately(self, document_id: str, content: str):
        """Process content immediately for basic chunking and embedding."""
        try:
            # Chunk the content
            chunk_success, chunks, chunk_message = self.embedding_service.chunk_text(content)
            if not chunk_success:
                logger.warning(f"Chunking failed: {chunk_message}")
                return
            
            # Generate embeddings
            embed_success, embeddings, embed_message = self.embedding_service.generate_embeddings(chunks)
            if not embed_success:
                logger.warning(f"Embedding generation failed: {embed_message}")
                embeddings = [None] * len(chunks)  # Use None for embeddings
            
            # Prepare chunk data
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings or [None] * len(chunks))):
                chunk_data.append({
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_index": i,
                        "processing_method": "immediate",
                        "chunk_type": "text"
                    },
                    "token_count": len(chunk.split())  # Simple token count
                })
            
            # Store chunks
            self.database_service.store_document_chunks(document_id, chunk_data)
            logger.info(f"Stored {len(chunk_data)} chunks for immediate processing")
            
        except Exception as e:
            logger.error(f"Failed to process content immediately: {e}")
    
    def _process_ai_parsed_content(self, document_id: str, processed_content: bytes):
        """Process AI-parsed content from Databricks job."""
        try:
            import json
            
            # Parse the processed content
            content_str = processed_content.decode('utf-8')
            parsed_data = json.loads(content_str)
            
            if parsed_data.get("status") != "success":
                logger.error(f"AI processing failed: {parsed_data.get('error', 'Unknown error')}")
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
            return {
                "success": False,
                "error": "Document not found"
            }
        
        return {
            "success": True,
            "doc_hash": doc_hash,
            "filename": document.filename,
            "status": document.status,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "metadata": document.doc_metadata
        }
